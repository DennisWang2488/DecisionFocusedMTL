"""Advanced torch trainer implementing ffo, nce, and lancer methods."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..advanced.lancer import LancerConfig, LancerTrainer
from ..advanced.nce import NCESolutionPool
from ..advanced.predictors import PredictorHandle, build_predictor, flatten_param_grads
from ..metrics import cosine, l2_norm
from ..schedules import alpha_value, lr_value
from ..tasks.medical_resource_allocation import MedicalResourceAllocationTask
from .torch_utils import merge_guided_dec_pred_gradient, resolve_device_or_warn, to_torch


ADVANCED_METHODS = {"ffo", "lancer", "nce"}


@dataclass
class _FFOState:
    layer_cache: Dict[int, Any]
    cfg: Dict[str, Any]
    budget: float
    device: torch.device
    dtype: torch.dtype


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _torch(x: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return to_torch(x, device=device, dtype=dtype)


def _build_ffo_layer(batch_size: int, ffo_cfg: Dict[str, Any]):
    import cvxpy as cp
    from ..advanced.ffolayer_local import FFOLayer

    pred_param = cp.Parameter(batch_size)
    cost_param = cp.Parameter(batch_size, nonneg=True)
    d = cp.Variable(batch_size)
    objective = cp.Minimize(0.5 * cp.sum_squares(d) - pred_param @ d)
    constraints = [d >= 0, cp.sum(cp.multiply(cost_param, d)) <= float(ffo_cfg.get("budget", 2500.0))]
    problem = cp.Problem(objective, constraints)
    return FFOLayer(
        problem,
        parameters=[pred_param, cost_param],
        variables=[d],
        alpha=float(ffo_cfg.get("alpha", 100.0)),
        dual_cutoff=float(ffo_cfg.get("dual_cutoff", 1e-3)),
        slack_tol=float(ffo_cfg.get("slack_tol", 1e-8)),
        eps=float(ffo_cfg.get("eps", 1e-7)),
        backward_eps=float(ffo_cfg.get("backward_eps", 1e-3)),
        max_workers=int(ffo_cfg.get("max_workers", 4)),
        verbose=bool(ffo_cfg.get("verbose", False)),
    )


def _ffo_grad_direction(
    state: _FFOState,
    pred: np.ndarray,
    true: np.ndarray,
    cost: np.ndarray,
) -> Tuple[np.ndarray, float]:
    bsz = int(pred.shape[0])
    if bsz not in state.layer_cache:
        cfg_local = dict(state.cfg)
        cfg_local["budget"] = state.budget
        state.layer_cache[bsz] = _build_ffo_layer(batch_size=bsz, ffo_cfg=cfg_local)
    layer = state.layer_cache[bsz]

    p_t = _torch(pred, device=state.device, dtype=state.dtype).unsqueeze(0).requires_grad_(True)
    c_t = _torch(np.clip(cost, 1e-8, None), device=state.device, dtype=state.dtype).unsqueeze(0)
    y_t = _torch(true, device=state.device, dtype=state.dtype).reshape(-1)

    solver_args = dict(state.cfg.get("solver_args", {}))
    if "solver" in solver_args and isinstance(solver_args["solver"], str):
        import cvxpy as cp

        solver_name = str(solver_args["solver"]).upper()
        solver_args["solver"] = getattr(cp, solver_name, solver_args["solver"])
    t0 = perf_counter()
    d_t, = layer(p_t, c_t, solver_args=solver_args)
    upper = -(d_t[0].reshape(-1) * y_t).mean()
    upper.backward()
    backward_ms = (perf_counter() - t0) * 1000.0
    if p_t.grad is None:
        return np.zeros_like(pred), backward_ms
    grad = p_t.grad.detach().cpu().numpy().reshape(-1)
    return grad, backward_ms


def _param_grad_from_pred_grad(
    predictor: PredictorHandle,
    out: torch.Tensor,
    grad_out: np.ndarray,
    retain_graph: bool,
) -> np.ndarray:
    predictor.module.zero_grad(set_to_none=True)
    out.backward(_torch(grad_out.reshape(-1), device=predictor.device, dtype=out.dtype), retain_graph=retain_graph)
    return flatten_param_grads(predictor.module)


def _eval_split(
    task: MedicalResourceAllocationTask,
    predictor: PredictorHandle,
    split_name: str,
    fairness_smoothing: float,
) -> Dict[str, float]:
    split = task._splits[split_name]
    pred_raw = predictor.predict_numpy(split.x).reshape(-1)
    pred = np.log1p(np.exp(-np.abs(pred_raw))) + np.maximum(pred_raw, 0.0) + 1e-6
    return task.evaluate_split(split_name, pred=pred, fairness_smoothing=fairness_smoothing)


def _run_single(
    task: MedicalResourceAllocationTask,
    train_cfg: Dict[str, Any],
    method: str,
    predictor_family: str,
    seed: int,
    lambdas: List[float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if method not in ADVANCED_METHODS:
        raise ValueError(f"Unknown advanced method: {method}")

    device = resolve_device_or_warn(str(train_cfg.get("device", "cuda")))
    steps = int(train_cfg["steps_per_lambda"])
    batch_size = int(train_cfg["batch_size"])
    lr0 = float(train_cfg["lr"])
    lr_decay = float(train_cfg.get("lr_decay", 0.0))
    fairness_smoothing = float(train_cfg.get("fairness_smoothing", 1e-6))
    log_every = int(train_cfg.get("log_every", 1))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))
    explode_threshold = float(train_cfg.get("explode_threshold", 1e9))

    predictor = build_predictor(
        family=predictor_family,
        input_dim=task._splits["train"].x.shape[1],
        output_dim=1,
        seed=13_579 + int(seed) * 101 + (0 if predictor_family == "linear" else 1),
        device=device,
        mlp_hidden_dim=int(train_cfg.get("mlp_hidden_dim", 64)),
        mlp_layers=int(train_cfg.get("mlp_layers", 2)),
    )
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr0)

    lancer_cfg = LancerConfig(**dict(train_cfg.get("lancer", {})))
    lancer: LancerTrainer | None = None
    if method == "lancer":
        lancer = LancerTrainer(z_dim=1, device=device, cfg=lancer_cfg)
        train_split = task._splits["train"]
        xb_all = _torch(train_split.x, device=device)
        yb_all = _torch(train_split.y.reshape(-1, 1), device=device)
        lancer.warm_start_predictor(predictor.module, xb_all, yb_all)

    nce_cfg = dict(train_cfg.get("nce", {}))
    nce_solve_ratio = float(nce_cfg.get("solve_ratio", 1.0))
    nce_refresh_interval = max(1, int(nce_cfg.get("refresh_interval", 1)))
    nce_pool: NCESolutionPool | None = None
    if method == "nce":
        nce_pool = NCESolutionPool(pool_size=int(nce_cfg.get("pool_size", 32)), sense="maximize")

    ffo_cfg = dict(train_cfg.get("ffo", {}))
    ffo_state: _FFOState | None = None
    if method == "ffo":
        ffo_state = _FFOState(layer_cache={}, cfg=ffo_cfg, budget=float(task.budget), device=device, dtype=torch.float32)

    rng = np.random.default_rng(seed * 10_003 + 17)
    stage_rows: List[Dict[str, Any]] = []
    iter_rows: List[Dict[str, Any]] = []
    cumulative_wallclock = 0.0

    for stage_idx, lam in enumerate(lambdas):
        stage_start = perf_counter()
        nan_or_inf_steps = 0
        exploding_steps = 0
        solver_calls_total = 0
        decision_ms_total = 0.0
        norm_combined_vals: List[float] = []
        cos_dec_pred_vals: List[float] = []
        cos_dec_fair_vals: List[float] = []
        cos_pred_fair_vals: List[float] = []
        lancer_surrogate_vals: List[float] = []
        ffo_backward_vals: List[float] = []

        for t in range(steps):
            b = task.sample_batch("train", batch_size=batch_size, rng=rng)
            xb = _torch(b.x, device=device)
            raw_pred = predictor.module(xb).reshape(-1)
            pred_pos = torch.nn.functional.softplus(raw_pred) + 1e-6
            pred_np = pred_pos.detach().cpu().numpy()

            out = task.compute_batch(
                raw_pred=pred_np,
                true=b.y,
                cost=b.cost,
                race=b.race,
                need_grads=True,
                fairness_smoothing=fairness_smoothing,
            )

            g_dec_pred = np.asarray(out["grad_dec"], dtype=float).reshape(-1)
            g_pred_pred = np.asarray(out["grad_pred"], dtype=float).reshape(-1)
            g_fair_pred = np.asarray(out["grad_fair"], dtype=float).reshape(-1)
            g_method_pred = g_dec_pred.copy()
            ffo_backward_ms = 0.0
            lancer_surrogate_loss = 0.0

            if method == "ffo":
                assert ffo_state is not None
                g_ffo, ffo_backward_ms = _ffo_grad_direction(ffo_state, pred=pred_np, true=b.y, cost=b.cost)
                g_method_pred = g_ffo
            elif method == "nce":
                assert nce_pool is not None
                if (t % nce_refresh_interval == 0) and (rng.uniform() <= nce_solve_ratio):
                    nce_pool.update(np.asarray(out["decision_pred"], dtype=float).reshape(-1))
                nce_out = nce_pool.loss_and_grad(pred_score=pred_np, true_decision=np.asarray(out["decision_true"], dtype=float).reshape(-1))
                g_method_pred = np.asarray(nce_out.grad_pred, dtype=float)
            elif method == "lancer":
                assert lancer is not None
                z_pred_np = pred_np.reshape(-1, 1)
                z_true_np = np.asarray(b.y, dtype=float).reshape(-1, 1)
                f_hat = np.full(z_pred_np.shape[0], float(out["loss_dec"]), dtype=float)
                lancer_surrogate_loss = lancer.update_surrogate(z_pred_np, z_true_np, f_hat)
                g_lancer, lancer_pred_loss = lancer.predictor_grad(
                    pred_pos.reshape(-1, 1),
                    _torch(z_true_np, device=device),
                )
                lancer_surrogate_loss = float(lancer_surrogate_loss + lancer_pred_loss)
                g_method_pred = g_lancer.reshape(-1)

            alpha_t = alpha_value(t=t, schedule_cfg=train_cfg["alpha_schedule"])
            beta_t = float(lam)
            g_dec_pred_guided = merge_guided_dec_pred_gradient(
                g_dec=g_method_pred,
                g_pred=g_pred_pred,
                alpha_t=alpha_t,
            )
            g_comb_pred = g_dec_pred_guided + beta_t * g_fair_pred

            g_dec_param = _param_grad_from_pred_grad(predictor, pred_pos, g_dec_pred, retain_graph=True)
            g_pred_param = _param_grad_from_pred_grad(predictor, pred_pos, g_pred_pred, retain_graph=True)
            g_fair_param = _param_grad_from_pred_grad(predictor, pred_pos, g_fair_pred, retain_graph=True)

            predictor.module.zero_grad(set_to_none=True)
            pred_pos.backward(_torch(g_comb_pred, device=device, dtype=pred_pos.dtype), retain_graph=False)
            g_comb_param = flatten_param_grads(predictor.module)

            grad_norm = l2_norm(g_comb_param)
            lr_t = lr_value(t=t, lr=lr0, lr_decay=lr_decay)
            for group in optimizer.param_groups:
                group["lr"] = lr_t

            nan_or_inf_flag = bool(
                np.isnan(g_comb_param).any()
                or np.isinf(g_comb_param).any()
                or any(np.isnan(float(out[k])) or np.isinf(float(out[k])) for k in ["loss_dec", "loss_pred", "loss_fair"])
            )
            if nan_or_inf_flag:
                nan_or_inf_steps += 1
                predictor.module.zero_grad(set_to_none=True)
            else:
                if grad_norm > explode_threshold:
                    exploding_steps += 1
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(predictor.module.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

            solver_calls_total += int(out["solver_calls"])
            decision_ms_total += float(out["decision_ms"])

            cos_dec_pred = cosine(g_dec_param, g_pred_param)
            cos_dec_fair = cosine(g_dec_param, g_fair_param)
            cos_pred_fair = cosine(g_pred_param, g_fair_param)
            cos_dec_pred_vals.append(cos_dec_pred)
            cos_dec_fair_vals.append(cos_dec_fair)
            cos_pred_fair_vals.append(cos_pred_fair)
            norm_combined_vals.append(grad_norm)
            lancer_surrogate_vals.append(float(lancer_surrogate_loss))
            ffo_backward_vals.append(float(ffo_backward_ms))

            if t % max(log_every, 1) == 0:
                iter_rows.append(
                    {
                        "task": task.name,
                        "method": method,
                        "predictor_family": predictor_family,
                        "seed": seed,
                        "stage_idx": stage_idx,
                        "lambda": lam,
                        "iter": t,
                        "alpha_t": alpha_t,
                        "beta_t": beta_t,
                        "lr_t": lr_t,
                        "loss_dec": float(out["loss_dec"]),
                        "loss_dec_normalized": float(out["loss_dec_normalized"]),
                        "loss_dec_normalized_pred_obj": float(out["loss_dec_normalized_pred_obj"]),
                        "loss_pred": float(out["loss_pred"]),
                        "loss_fair": float(out["loss_fair"]),
                        "grad_norm_dec": l2_norm(g_dec_param),
                        "grad_norm_pred": l2_norm(g_pred_param),
                        "grad_norm_fair": l2_norm(g_fair_param),
                        "grad_norm_combined": grad_norm,
                        "cos_dec_pred": cos_dec_pred,
                        "cos_dec_fair": cos_dec_fair,
                        "cos_pred_fair": cos_pred_fair,
                        "solver_calls": int(out["solver_calls"]),
                        "decision_ms": float(out["decision_ms"]),
                        "nan_or_inf_flag": int(nan_or_inf_flag),
                        "nan_or_inf_steps_so_far": nan_or_inf_steps,
                        "exploding_steps_so_far": exploding_steps,
                        "nce_pool_size": int(len(nce_pool.pool)) if nce_pool is not None else 0,
                        "lancer_surrogate_loss": float(lancer_surrogate_loss),
                        "ffo_backward_ms": float(ffo_backward_ms),
                        "device": str(device),
                    }
                )

        stage_wallclock = float(perf_counter() - stage_start)
        cumulative_wallclock += stage_wallclock
        val_metrics = _eval_split(task, predictor, "val", fairness_smoothing=fairness_smoothing)
        test_metrics = _eval_split(task, predictor, "test", fairness_smoothing=fairness_smoothing)

        stage_rows.append(
            {
                "task": task.name,
                "method": method,
                "predictor_family": predictor_family,
                "seed": seed,
                "stage_idx": stage_idx,
                "lambda": lam,
                "val_regret": val_metrics["regret"],
                "val_regret_normalized": val_metrics["regret_normalized"],
                "val_regret_normalized_pred_obj": val_metrics["regret_normalized_pred_obj"],
                "val_fairness": val_metrics["fairness"],
                "val_pred_mse": val_metrics["pred_mse"],
                "test_regret": test_metrics["regret"],
                "test_regret_normalized": test_metrics["regret_normalized"],
                "test_regret_normalized_pred_obj": test_metrics["regret_normalized_pred_obj"],
                "test_fairness": test_metrics["fairness"],
                "test_pred_mse": test_metrics["pred_mse"],
                "stage_wallclock_sec": stage_wallclock,
                "cumulative_wallclock_sec": cumulative_wallclock,
                "solver_calls_train": solver_calls_total,
                "solver_calls_eval": val_metrics["solver_calls_eval"] + test_metrics["solver_calls_eval"],
                "decision_ms_train": decision_ms_total,
                "decision_ms_eval": val_metrics["decision_ms_eval"] + test_metrics["decision_ms_eval"],
                "nan_or_inf_steps": nan_or_inf_steps,
                "nan_steps": nan_or_inf_steps,
                "exploding_steps": exploding_steps,
                "avg_grad_norm_combined": _safe_mean(norm_combined_vals),
                "grad_norm_min": float(np.min(norm_combined_vals)) if norm_combined_vals else 0.0,
                "grad_norm_median": float(np.median(norm_combined_vals)) if norm_combined_vals else 0.0,
                "grad_norm_max": float(np.max(norm_combined_vals)) if norm_combined_vals else 0.0,
                "avg_cos_dec_pred": _safe_mean(cos_dec_pred_vals),
                "avg_cos_dec_fair": _safe_mean(cos_dec_fair_vals),
                "avg_cos_pred_fair": _safe_mean(cos_pred_fair_vals),
                "weight_norm": float(np.sqrt(sum(float((p.detach().cpu().numpy() ** 2).sum()) for p in predictor.module.parameters()))),
                "nce_pool_size": int(len(nce_pool.pool)) if nce_pool is not None else 0,
                "lancer_surrogate_loss": _safe_mean(lancer_surrogate_vals),
                "ffo_backward_ms": _safe_mean(ffo_backward_vals),
                "device": str(device),
            }
        )
    return stage_rows, iter_rows


def run_advanced_methods(
    task: MedicalResourceAllocationTask,
    train_cfg: Dict[str, Any],
    methods: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not isinstance(task, MedicalResourceAllocationTask):
        raise ValueError("Advanced methods currently support medical_resource_allocation only.")

    predictor_families = [str(x).strip() for x in train_cfg.get("predictor_families", ["linear"]) if str(x).strip()]
    if not predictor_families:
        predictor_families = ["linear"]

    seeds = [int(s) for s in train_cfg["seeds"]]
    lambdas = [float(v) for v in train_cfg["lambdas"]]

    stage_rows: List[Dict[str, Any]] = []
    iter_rows: List[Dict[str, Any]] = []
    for method in methods:
        for family in predictor_families:
            for seed in seeds:
                stg, itr = _run_single(
                    task=task,
                    train_cfg=train_cfg,
                    method=method,
                    predictor_family=family,
                    seed=seed,
                    lambdas=lambdas,
                )
                stage_rows.extend(stg)
                iter_rows.extend(itr)
    return stage_rows, iter_rows
