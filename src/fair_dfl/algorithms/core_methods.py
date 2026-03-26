"""Core torch trainer implementing the 5 DFL methods: fpto, dfl, fdfl, plg, fplg."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from ..metrics import cosine, l2_norm, project_orthogonal
from ..models import build_predictor, PredictorHandle, PostProcessor
from ..models.registry import _resolve_model_config
from ..schedules import alpha_value, lr_value
from ..tasks.base import BaseTask, SplitData, TaskData
from .mo_handler import (
    MultiObjectiveGradientHandler,
    WeightedSumHandler,
    PCGradHandler,
    MGDAHandler,
    CAGradHandler,
    PLGHandler3Obj,
    FAMOHandler,
    NestedPLGFairPrimaryHandler,
    NestedPLGPredPrimaryHandler,
)
from ..tasks.medical_resource_allocation import MedicalResourceAllocationTask
from ..tasks.portfolio_qp_simplex import PortfolioQPSimplexTask
from ..tasks.resource_allocation import ResourceAllocationTask
from .torch_utils import (
    backward_param_grad_from_output_grad,
    flatten_param_grads,
    merge_guided_dec_pred_gradient,
    parameter_l2_norm,
    resolve_device_or_warn,
    to_torch,
)


@dataclass
class MethodSpec:
    use_dec: bool
    use_pred: bool
    use_fair: bool
    pred_weight_mode: str
    continuation: bool
    allow_orthogonalization: bool


BASE_METHOD_SPECS: Dict[str, MethodSpec] = {
    "fplg": MethodSpec(
        use_dec=True,
        use_pred=True,
        use_fair=True,
        pred_weight_mode="schedule",
        continuation=True,
        allow_orthogonalization=True,
    ),
    # Internal-only baseline for Pareto sweep plotting.
    "grid_restart": MethodSpec(
        use_dec=True,
        use_pred=True,
        use_fair=True,
        pred_weight_mode="schedule",
        continuation=False,
        allow_orthogonalization=True,
    ),
    "fdfl": MethodSpec(
        use_dec=True,
        use_pred=False,
        use_fair=True,
        pred_weight_mode="zero",
        continuation=False,
        allow_orthogonalization=False,
    ),
    "plg": MethodSpec(
        use_dec=True,
        use_pred=True,
        use_fair=False,
        pred_weight_mode="schedule",
        continuation=False,
        allow_orthogonalization=False,
    ),
    "fpto": MethodSpec(
        use_dec=False,
        use_pred=True,
        use_fair=True,
        pred_weight_mode="fixed1",
        continuation=False,
        allow_orthogonalization=False,
    ),
    "dfl": MethodSpec(
        use_dec=True,
        use_pred=False,
        use_fair=False,
        pred_weight_mode="zero",
        continuation=False,
        allow_orthogonalization=False,
    ),
    "saa": MethodSpec(
        use_dec=False,
        use_pred=True,
        use_fair=False,
        pred_weight_mode="fixed1",
        continuation=False,
        allow_orthogonalization=False,
    ),
    "var_dro": MethodSpec(
        use_dec=False,
        use_pred=True,
        use_fair=False,
        pred_weight_mode="fixed1",
        continuation=False,
        allow_orthogonalization=False,
    ),
    "wass_dro": MethodSpec(
        use_dec=False,
        use_pred=True,
        use_fair=False,
        pred_weight_mode="fixed1",
        continuation=False,
        allow_orthogonalization=False,
    ),
}

PUBLIC_METHODS = ("fpto", "dfl", "fdfl", "plg", "fplg", "saa", "var_dro", "wass_dro")
METHOD_SPECS: Dict[str, MethodSpec] = {name: BASE_METHOD_SPECS[name] for name in PUBLIC_METHODS}

# Human-readable aliases → canonical abbreviation
METHOD_ALIASES: Dict[str, str] = {
    "prediction_only": "fpto",
    "decision_focused": "dfl",
    "fair_decision_focused": "fdfl",
    "pred_loss_guided": "plg",
    "fair_pred_loss_guided": "fplg",
    "sample_average_approximation": "saa",
    "variance_dro": "var_dro",
    "wasserstein_dro": "wass_dro",
}
REVERSE_ALIASES: Dict[str, str] = {v: k for k, v in METHOD_ALIASES.items()}


class _MLP2x64Softplus(nn.Module):
    """Legacy architecture kept for backward compatibility only.
    New code should use models.build_predictor({"arch": "mlp", ...}) instead.
    """
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _resolve_base_spec(method_name: str) -> MethodSpec:
    if method_name not in BASE_METHOD_SPECS:
        raise ValueError(f"Unknown method: {method_name}")
    return BASE_METHOD_SPECS[method_name]


def _sample_batch(split: SplitData, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    n = split.x.shape[0]
    replace = n < batch_size
    idx = rng.choice(n, size=batch_size, replace=replace)
    return split.x[idx], split.y[idx]


def _pred_weight(mode: str, t: int, alpha_schedule_cfg: Dict[str, Any]) -> float:
    if mode == "zero":
        return 0.0
    if mode == "fixed1":
        return 1.0
    if mode == "schedule":
        return alpha_value(t=t, schedule_cfg=alpha_schedule_cfg)
    raise ValueError(f"Unsupported pred weight mode: {mode}")


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _build_predictor(
    family: str,
    input_dim: int,
    output_dim: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    train_cfg: Dict[str, Any] | None = None,
) -> Tuple[nn.Module, str, PostProcessor]:
    """Build predictor using the unified models package.

    Returns (module, family_name, post_processor).
    Supports both legacy 'predictor_family' strings and new 'model' config dicts.
    """
    if train_cfg is not None and "model" in train_cfg:
        # New-style model config
        model_cfg = _resolve_model_config(train_cfg)
    else:
        # Legacy: convert family string
        model_cfg = _resolve_model_config({"predictor_family": family})

    # Determine init mode and seed
    init_mode = model_cfg.get("init_mode", "default")
    fam = str(family).strip().lower()

    # For exact backward compat with legacy code, override seed/init
    if fam == "linear" and init_mode == "default":
        model_cfg["init_mode"] = "legacy_core"
        build_seed = seed
    elif fam == "mlp_2x64_softplus" and init_mode == "default":
        build_seed = 13_579 + seed * 101 + 1
    else:
        build_seed = 13_579 + seed * 101 + 1

    # Determine post-transform: medical tasks need softplus for non-MLP-softplus
    post_transform = "none"
    if fam == "mlp_2x64_softplus":
        # Legacy MLP has softplus built-in, so no external post-processor needed
        post_transform = "none"
    elif fam == "linear":
        # Linear models need external softplus for medical tasks
        # (applied conditionally at call site based on task type)
        post_transform = "none"  # caller decides

    handle = build_predictor(
        config=model_cfg,
        input_dim=input_dim,
        output_dim=output_dim,
        seed=build_seed,
        device=device,
        dtype=dtype,
        post_transform=post_transform,
    )
    return handle.module, handle.arch, handle.post_processor


def _medical_pred_from_model_output(
    model_out: torch.Tensor,
    family: str,
    post_processor: PostProcessor | None = None,
) -> torch.Tensor:
    """Apply positivity transform to model output for medical tasks.

    For models that already output positive values (e.g. legacy mlp_2x64_softplus),
    only adds eps. For others, applies softplus + eps.
    """
    if post_processor is not None and post_processor.transform != "none":
        return post_processor(model_out)
    # Legacy behavior
    if family == "linear":
        return torch.nn.functional.softplus(model_out) + 1e-6
    if family in ("mlp_2x64_softplus",):
        return model_out + 1e-6
    # New architectures: always apply softplus for medical tasks
    return torch.nn.functional.softplus(model_out) + 1e-6


def _eval_split(
    task: BaseTask,
    model: nn.Module,
    split: SplitData,
    fairness_smoothing: float,
    device: torch.device,
    dtype: torch.dtype,
    override_pred: np.ndarray | None = None,
) -> Dict[str, float]:
    if override_pred is not None:
        raw_pred = override_pred
    else:
        model.eval()
        with torch.no_grad():
            raw_pred = model(to_torch(split.x, device=device, dtype=dtype)).detach().cpu().numpy()
        model.train()
    out = task.compute(
        raw_pred=raw_pred,
        true=split.y,
        need_grads=False,
        fairness_smoothing=fairness_smoothing,
    )
    metrics = {
        "regret": float(out["loss_dec"]),
        "pred_mse": float(out["loss_pred"]),
        "fairness": float(out["loss_fair"]),
        "solver_calls_eval": float(out["solver_calls"]),
        "decision_ms_eval": float(out["decision_ms"]),
    }
    if "loss_dec_normalized" in out:
        metrics["regret_normalized"] = float(out["loss_dec_normalized"])
    if "loss_dec_normalized_true" in out:
        metrics["regret_normalized_true"] = float(out["loss_dec_normalized_true"])
    if "loss_dec_normalized_pred_obj" in out:
        metrics["regret_normalized_pred_obj"] = float(out["loss_dec_normalized_pred_obj"])
    return metrics


def _eval_split_medical(
    task: MedicalResourceAllocationTask,
    model: nn.Module,
    split_name: str,
    fairness_smoothing: float,
    device: torch.device,
    dtype: torch.dtype,
    family: str,
    override_pred: np.ndarray | None = None,
    post_processor: PostProcessor | None = None,
) -> Dict[str, float]:
    if override_pred is not None:
        return task.evaluate_split(split=split_name, pred=override_pred, fairness_smoothing=fairness_smoothing)
    split = task._splits[split_name]
    model.eval()
    with torch.no_grad():
        raw_out = model(to_torch(split.x, device=device, dtype=dtype)).reshape(-1)
        pred = _medical_pred_from_model_output(raw_out, family=family, post_processor=post_processor).detach().cpu().numpy().reshape(-1)
    model.train()
    return task.evaluate_split(split=split_name, pred=pred, fairness_smoothing=fairness_smoothing)


def _active_spec(base_spec: MethodSpec, iter_idx: int, warmstart_steps: int) -> MethodSpec:
    if warmstart_steps <= 0:
        return base_spec
    if iter_idx < warmstart_steps:
        return BASE_METHOD_SPECS["fpto"]
    return base_spec


def _method_uses_fpto_warmstart(method_name: str, train_cfg: Dict[str, Any]) -> bool:
    if method_name == "fpto":
        return False
    if "fpto_warmstart_methods" in train_cfg:
        methods = {str(x).strip() for x in train_cfg.get("fpto_warmstart_methods", [])}
        return method_name in methods
    if "fpto_warmstart_enabled" in train_cfg:
        return bool(train_cfg.get("fpto_warmstart_enabled"))
    return False


def _combine_prediction_gradients(
    *,
    method_name: str,
    iter_spec: MethodSpec,
    g_dec_pred: np.ndarray,
    g_pred_pred: np.ndarray,
    g_fair_pred: np.ndarray,
    alpha_t: float,
    beta_t: float,
    guided_scale_mode: str,
    guided_norm_floor: float,
) -> tuple[np.ndarray, Dict[str, float]]:
    diag = {
        "guided_norm_dec": float("nan"),
        "guided_norm_pred": float("nan"),
        "guided_scale": float("nan"),
        "guided_ratio_pred_over_dec": float("nan"),
        "guided_dir_norm": float("nan"),
    }
    if iter_spec.use_dec and iter_spec.use_pred:
        fairness_into_pred = iter_spec.use_fair and method_name in {"fplg", "grid_restart"}
        pred_branch = g_pred_pred + beta_t * g_fair_pred if fairness_into_pred else g_pred_pred
        g_guided, guided_diag = merge_guided_dec_pred_gradient(
            g_dec=g_dec_pred,
            g_pred=pred_branch,
            alpha_t=alpha_t,
            scale_mode=guided_scale_mode,
            norm_floor=guided_norm_floor,
            return_diag=True,
        )
        diag = {
            "guided_norm_dec": float(guided_diag["norm_dec"]),
            "guided_norm_pred": float(guided_diag["norm_pred"]),
            "guided_scale": float(guided_diag["guided_scale"]),
            "guided_ratio_pred_over_dec": float(guided_diag["ratio_pred_over_dec"]),
            "guided_dir_norm": float(guided_diag["dir_norm"]),
        }
        if fairness_into_pred:
            return g_guided, diag
        return g_guided + beta_t * g_fair_pred, diag

    if iter_spec.use_dec:
        return g_dec_pred + beta_t * g_fair_pred, diag
    if iter_spec.use_pred:
        return alpha_t * g_pred_pred + beta_t * g_fair_pred, diag
    return beta_t * g_fair_pred, diag


def _softplus_np(x: np.ndarray) -> np.ndarray:
    positive = np.maximum(x, 0.0)
    exp_term = np.exp(-np.abs(x))
    return positive + np.log1p(exp_term)


def _finite_diff_decision_grad(
    task: BaseTask,
    raw_pred: np.ndarray,
    true: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, int, float]:
    """Finite-difference gradient of mean decision regret w.r.t. raw_pred.

    Uses per-instance solves (no full-batch re-solves) for separable tasks.
    """
    t0 = perf_counter()
    bsz = int(raw_pred.shape[0])
    dim = int(raw_pred.shape[1])
    grad = np.zeros_like(raw_pred, dtype=float)

    if isinstance(task, ResourceAllocationTask):
        costs = np.asarray(task._current_costs, dtype=float)  # bound via runner
        raw = np.asarray(raw_pred, dtype=float)
        pred_pos = _softplus_np(raw) + 1e-5
        y_true = np.asarray(true, dtype=float)

        obj_true = np.zeros(bsz, dtype=float)
        for b in range(bsz):
            d_true = task._solve_allocation_batch(y_true[b : b + 1], costs)
            obj_true[b] = float(task._objective(d_true, y_true[b : b + 1])[0])

        solver_calls = 0
        for b in range(bsz):
            base_pos = pred_pos[b].copy()
            base_raw = raw[b]
            yb = y_true[b : b + 1]
            for j in range(dim):
                plus_pos = base_pos.copy()
                minus_pos = base_pos.copy()
                plus_pos[j] = float(_softplus_np(np.array([base_raw[j] + eps]))[0] + 1e-5)
                minus_pos[j] = float(_softplus_np(np.array([base_raw[j] - eps]))[0] + 1e-5)

                d_plus = task._solve_allocation_batch(plus_pos[None, :], costs)
                d_minus = task._solve_allocation_batch(minus_pos[None, :], costs)
                obj_pred_plus = float(task._objective(d_plus, yb)[0])
                obj_pred_minus = float(task._objective(d_minus, yb)[0])
                regret_plus = max(float(obj_true[b] - obj_pred_plus), 0.0)
                regret_minus = max(float(obj_true[b] - obj_pred_minus), 0.0)
                grad[b, j] = (regret_plus - regret_minus) / (2.0 * float(eps) * float(bsz))
                solver_calls += 2

        decision_ms = (perf_counter() - t0) * 1000.0
        return grad, int(solver_calls + bsz), float(decision_ms)

    if isinstance(task, PortfolioQPSimplexTask):
        if task._cvx_problem is None:
            raise RuntimeError("PortfolioQPSimplexTask missing CVXPY context; bind_context must run before training.")
        sigma = np.asarray(task._cvx_problem["sigma"], dtype=float)
        mu_pred = np.asarray(raw_pred, dtype=float)
        mu_true = np.asarray(true, dtype=float)

        obj_true = np.zeros(bsz, dtype=float)
        for b in range(bsz):
            w_true = task._solve_single(mu_true[b])
            obj_true[b] = float(task._objective(w_true, mu_true[b], sigma, task.risk_aversion))

        solver_calls = 0
        for b in range(bsz):
            mu_base = mu_pred[b].copy()
            mu_true_b = mu_true[b]
            for j in range(dim):
                mu_plus = mu_base.copy()
                mu_minus = mu_base.copy()
                mu_plus[j] += float(eps)
                mu_minus[j] -= float(eps)
                w_plus = task._solve_single(mu_plus)
                w_minus = task._solve_single(mu_minus)
                obj_pred_plus = float(task._objective(w_plus, mu_true_b, sigma, task.risk_aversion))
                obj_pred_minus = float(task._objective(w_minus, mu_true_b, sigma, task.risk_aversion))
                regret_plus = max(float(obj_true[b] - obj_pred_plus), 0.0)
                regret_minus = max(float(obj_true[b] - obj_pred_minus), 0.0)
                grad[b, j] = (regret_plus - regret_minus) / (2.0 * float(eps) * float(bsz))
                solver_calls += 2

        decision_ms = (perf_counter() - t0) * 1000.0
        return grad, int(solver_calls + bsz), float(decision_ms)

    raise ValueError(f"Finite-difference decision gradient is not implemented for task: {type(task).__name__}")


def _train_single_stage(
    task: BaseTask,
    data: TaskData,
    model: nn.Module,
    family: str,
    base_spec: MethodSpec,
    train_cfg: Dict[str, Any],
    lambda_value: float,
    seed: int,
    method_name: str,
    stage_idx: int,
    device: torch.device,
    dtype: torch.dtype,
    post_processor: PostProcessor | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rng = np.random.default_rng(seed * 10_000 + stage_idx * 113 + 7)
    steps = int(train_cfg["steps_per_lambda"])
    batch_size = int(train_cfg["batch_size"])
    lr0 = float(train_cfg["lr"])
    lr_decay = float(train_cfg.get("lr_decay", 0.0))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))
    explode_threshold = float(train_cfg.get("explode_threshold", 1e12))
    fairness_smoothing = float(train_cfg.get("fairness_smoothing", 1e-6))
    log_every = int(train_cfg.get("log_every", 1))
    beta_mode = str(train_cfg.get("beta_mode", "penalty"))
    fair_tau = float(train_cfg.get("fair_tau", 0.0))
    dual_lr = float(train_cfg.get("dual_lr", 0.01))
    warmstart_fraction = float(train_cfg.get("warmstart_fraction", 0.25))

    optimizer = torch.optim.SGD(model.parameters(), lr=lr0)

    orth_cfg = train_cfg.get("orthogonalization", {})
    orth_enabled = bool(orth_cfg.get("enabled", False)) and base_spec.allow_orthogonalization
    orth_ref = str(orth_cfg.get("reference", "pred"))
    orth_conflict_threshold = float(orth_cfg.get("conflict_threshold", -0.1))

    warmstart_steps = 0
    if _method_uses_fpto_warmstart(method_name=method_name, train_cfg=train_cfg):
        warmstart_steps = int(np.ceil(max(0.0, warmstart_fraction) * float(steps)))
        warmstart_steps = min(max(warmstart_steps, 0), steps)

    dual_lambda = float(lambda_value if beta_mode == "penalty" else 0.0)

    nan_or_inf_steps = 0
    exploding_steps = 0
    solver_calls_total = 0
    decision_ms_total = 0.0
    solver_calls_fd_total = 0
    decision_ms_fd_total = 0.0

    cos_dec_pred_list: List[float] = []
    cos_dec_fair_list: List[float] = []
    cos_pred_fair_list: List[float] = []
    norm_combined_list: List[float] = []
    iter_logs: List[Dict[str, Any]] = []

    decision_backend = str(train_cfg.get("decision_grad_backend", "analytic")).strip().lower()
    if decision_backend not in {"analytic", "finite_diff"}:
        raise ValueError("training.decision_grad_backend must be one of: 'analytic', 'finite_diff'.")
    fd_methods = {str(x).strip() for x in train_cfg.get("decision_grad_fd_methods", []) if str(x).strip()}
    fd_eps = float(train_cfg.get("decision_grad_fd_eps", 1e-3))
    guided_scale_mode = str(train_cfg.get("guided_merge_scale_mode", "geom")).strip().lower()
    guided_norm_floor = float(train_cfg.get("guided_merge_norm_floor", 1e-3))

    mo_method = train_cfg.get("mo_method", None)
    mo_handler: MultiObjectiveGradientHandler | None = None
    if mo_method == "weighted_sum":
        mo_handler = WeightedSumHandler(weights=train_cfg.get("mo_weights", {}))
    elif mo_method == "pcgrad":
        mo_handler = PCGradHandler()
    elif mo_method == "mgda":
        mo_handler = MGDAHandler()
    elif mo_method == "cagrad":
        mo_handler = CAGradHandler(c=float(train_cfg.get("mo_cagrad_c", 0.5)))
    elif mo_method == "plg3":
        mo_handler = PLGHandler3Obj(
            kappa_0=float(train_cfg.get("mo_plg_kappa_0", 1.0)),
            kappa_decay=float(train_cfg.get("mo_plg_kappa_decay", 0.01)),
        )
    elif mo_method == "famo":
        mo_handler = FAMOHandler(
            n_tasks=3,
            gamma=float(train_cfg.get("mo_famo_gamma", 1e-3)),
            w_lr=float(train_cfg.get("mo_famo_w_lr", 0.025)),
            min_loss=float(train_cfg.get("mo_famo_min_loss", 1e-8)),
        )
    elif mo_method == "plg_fp":
        mo_handler = NestedPLGFairPrimaryHandler(
            kappa1_0=float(train_cfg.get("mo_plg_kappa1_0", 1.0)),
            kappa2_0=float(train_cfg.get("mo_plg_kappa2_0", 1.0)),
            kappa_decay=float(train_cfg.get("mo_plg_kappa_decay", 0.01)),
        )
    elif mo_method == "plg_pp":
        mo_handler = NestedPLGPredPrimaryHandler(
            kappa1_0=float(train_cfg.get("mo_plg_kappa1_0", 1.0)),
            kappa2_0=float(train_cfg.get("mo_plg_kappa2_0", 1.0)),
            kappa_decay=float(train_cfg.get("mo_plg_kappa_decay", 0.01)),
        )
    elif mo_method is not None:
        raise ValueError(f"Unknown mo_method: {mo_method}")

    # --- SAA: model-free constant predictor (skip training) ---
    saa_mean = 0.0
    if method_name == "saa":
        if isinstance(task, MedicalResourceAllocationTask):
            train_split = task._splits["train"]
            saa_mean = float(np.mean(train_split.y))
        else:
            saa_mean = float(np.mean(data.train.y))
        steps = 0  # skip training loop entirely

    stage_start = perf_counter()
    for t in range(steps):
        do_log = bool(t % max(log_every, 1) == 0)
        iter_spec = _active_spec(base_spec=base_spec, iter_idx=t, warmstart_steps=warmstart_steps)
        use_fd = bool(decision_backend == "finite_diff" and iter_spec.use_dec and method_name in fd_methods)
        need_dec_grads = bool(iter_spec.use_dec and (not use_fd))

        if isinstance(task, MedicalResourceAllocationTask):
            if use_fd:
                raise ValueError("Finite-difference decision gradients are not supported for medical_resource_allocation.")
            batch = task.sample_batch("train", batch_size=batch_size, rng=rng)
            xb_t = to_torch(batch.x, device=device, dtype=dtype)
            raw_out_t = model(xb_t).reshape(-1)
            pred_t = _medical_pred_from_model_output(raw_out_t, family=family, post_processor=post_processor)
            pred_np = pred_t.detach().cpu().numpy().reshape(-1)
            out = task.compute_batch(
                raw_pred=pred_np,
                true=batch.y,
                cost=batch.cost,
                race=batch.race,
                need_grads=need_dec_grads,
                fairness_smoothing=fairness_smoothing,
            )
        else:
            xb, yb = _sample_batch(data.train, batch_size=batch_size, rng=rng)
            xb_t = to_torch(xb, device=device, dtype=dtype)
            pred_t = model(xb_t)
            pred_np = pred_t.detach().cpu().numpy()
            try:
                out = task.compute(
                    raw_pred=pred_np,
                    true=yb,
                    need_grads=need_dec_grads,
                    fairness_smoothing=fairness_smoothing,
                )
            except ValueError as exc:
                msg = str(exc)
                analytic_missing = "analytic decision gradients" in msg.lower()
                if iter_spec.use_dec and need_dec_grads and analytic_missing:
                    # Auto-fallback to FD when analytic decision gradients are unavailable.
                    out = task.compute(
                        raw_pred=pred_np,
                        true=yb,
                        need_grads=False,
                        fairness_smoothing=fairness_smoothing,
                    )
                    use_fd = True
                else:
                    raise ValueError(
                        f"Decision-gradient path failed for method='{method_name}' on task='{task.name}': {msg}"
                    ) from exc

        solver_calls_fd = 0
        decision_ms_fd = 0.0

        if use_fd:
            g_dec_pred, solver_calls_fd, decision_ms_fd = _finite_diff_decision_grad(
                task=task,
                raw_pred=pred_np,
                true=yb,
                eps=fd_eps,
            )
        else:
            g_dec_pred = np.asarray(out["grad_dec"], dtype=float).reshape(pred_np.shape) if iter_spec.use_dec else np.zeros_like(pred_np)
        g_pred_pred = np.asarray(out["grad_pred"], dtype=float).reshape(pred_np.shape) if iter_spec.use_pred else np.zeros_like(pred_np)
        g_fair_pred = np.asarray(out["grad_fair"], dtype=float).reshape(pred_np.shape) if iter_spec.use_fair else np.zeros_like(pred_np)

        # --- VarDRO: f-divergence variance-regularized prediction gradient ---
        if method_name == "var_dro":
            dro_eps = float(train_cfg.get("dro_epsilon", 0.1))
            if isinstance(task, MedicalResourceAllocationTask):
                true_np = batch.y
            else:
                true_np = yb
            # Per-sample MSE: average over output dims, keep batch dim
            per_sample_loss = ((pred_np - true_np) ** 2).reshape(pred_np.shape[0], -1).mean(axis=-1)
            mean_loss = per_sample_loss.mean()
            std_loss = per_sample_loss.std()
            if std_loss > 1e-12:
                dro_weights = 1.0 + dro_eps * (per_sample_loss - mean_loss) / std_loss
                dro_weights = np.maximum(dro_weights, 0.0)
            else:
                dro_weights = np.ones_like(per_sample_loss)
            # Expand weights to broadcast with (batch, n_outputs) gradient shape
            dro_weights = dro_weights.reshape(-1, *([1] * (g_pred_pred.ndim - 1)))
            g_pred_pred = g_pred_pred * dro_weights
            out["loss_pred"] = float(mean_loss + dro_eps * std_loss)

        # --- WassDRO: Wasserstein DRO via input gradient penalty ---
        wass_dro_penalty_val = 0.0
        if method_name == "wass_dro":
            wdro_eps = float(train_cfg.get("wdro_epsilon", 0.1))
            xb_wdro = xb_t.detach().clone().requires_grad_(True)
            if isinstance(task, MedicalResourceAllocationTask):
                raw_wdro = model(xb_wdro).reshape(-1)
                pred_wdro = post_processor(raw_wdro)
                yb_wdro = to_torch(batch.y, device=device, dtype=dtype)
                per_sample_mse = (pred_wdro - yb_wdro) ** 2
            else:
                pred_wdro = model(xb_wdro)
                if post_processor.transform != "none":
                    pred_wdro = post_processor(pred_wdro)
                yb_wdro = to_torch(yb, device=device, dtype=dtype)
                per_sample_mse = ((pred_wdro - yb_wdro) ** 2).reshape(
                    pred_wdro.shape[0], -1
                ).mean(dim=-1)
            grad_x = torch.autograd.grad(
                per_sample_mse.sum(), xb_wdro, create_graph=True,
            )[0]
            grad_norms = (grad_x ** 2).sum(dim=-1).sqrt()
            penalty = wdro_eps * grad_norms.mean()
            wass_dro_penalty_val = float(penalty.item())
            model.zero_grad(set_to_none=True)
            penalty.backward()
            wass_dro_penalty_param_grad = flatten_param_grads(model)
            out["wdro_grad_penalty"] = wass_dro_penalty_val
            out["loss_pred"] = float(per_sample_mse.mean().item()) + wass_dro_penalty_val

        alpha_t = _pred_weight(iter_spec.pred_weight_mode, t=t, alpha_schedule_cfg=train_cfg["alpha_schedule"])
        if not iter_spec.use_fair:
            beta_t = 0.0
        elif beta_mode == "penalty":
            beta_t = float(lambda_value)
        elif beta_mode == "constraint":
            beta_t = float(dual_lambda)
        else:
            raise ValueError(f"Unknown beta_mode: {beta_mode}")

        # Orthogonalization: skip when mo_handler is active so that it receives
        # raw per-objective gradients and handles all combination itself.
        if orth_enabled and iter_spec.use_fair and mo_handler is None:
            ref_grad = g_pred_pred if orth_ref == "pred" else g_dec_pred
            fair_ref_cos = cosine(g_fair_pred, ref_grad)
            if fair_ref_cos < orth_conflict_threshold:
                g_fair_pred = project_orthogonal(g_fair_pred, ref_grad)

        g_comb_pred, guided_diag = _combine_prediction_gradients(
            method_name=method_name,
            iter_spec=iter_spec,
            g_dec_pred=g_dec_pred,
            g_pred_pred=g_pred_pred,
            g_fair_pred=g_fair_pred,
            alpha_t=alpha_t,
            beta_t=beta_t,
            guided_scale_mode=guided_scale_mode,
            guided_norm_floor=guided_norm_floor,
        )

        g_dec_param = backward_param_grad_from_output_grad(
            module=model,
            output=pred_t,
            grad_out=g_dec_pred,
            retain_graph=True,
            device=device,
        )
        g_pred_param = backward_param_grad_from_output_grad(
            module=model,
            output=pred_t,
            grad_out=g_pred_pred,
            retain_graph=True,
            device=device,
        )
        g_fair_param = backward_param_grad_from_output_grad(
            module=model,
            output=pred_t,
            grad_out=g_fair_pred,
            retain_graph=True,
            device=device,
        )

        # WassDRO: add gradient penalty contribution to pred param grad
        if method_name == "wass_dro":
            g_pred_param = g_pred_param + wass_dro_penalty_param_grad

        if mo_handler is not None:
            mo_grads = {
                "pred_loss": g_pred_param,
                "decision_regret": g_dec_param,
                "pred_fairness": g_fair_param,
            }
            mo_losses_dict = {
                "pred_loss": float(out["loss_pred"]),
                "decision_regret": float(out["loss_dec"]),
                "pred_fairness": float(out["loss_fair"]),
            }
            g_comb_param = mo_handler.compute_direction(mo_grads, mo_losses_dict, step=t, epsilon=1e-4)

            # Set model gradients directly from the MO handler output.
            model.zero_grad(set_to_none=True)
            offset = 0
            with torch.no_grad():
                for p in model.parameters():
                    numel = p.numel()
                    p.grad = torch.as_tensor(
                        g_comb_param[offset:offset + numel],
                        dtype=p.dtype, device=p.device,
                    ).reshape(p.shape)
                    offset += numel
        else:
            model.zero_grad(set_to_none=True)
            pred_t.backward(to_torch(g_comb_pred, device=device, dtype=pred_t.dtype), retain_graph=False)
            g_comb_param = flatten_param_grads(model)

        cos_dec_pred = cosine(g_dec_param, g_pred_param)
        cos_dec_fair = cosine(g_dec_param, g_fair_param)
        cos_pred_fair = cosine(g_pred_param, g_fair_param)
        cos_dec_pred_list.append(cos_dec_pred)
        cos_dec_fair_list.append(cos_dec_fair)
        cos_pred_fair_list.append(cos_pred_fair)

        grad_norm = l2_norm(g_comb_param)
        norm_combined_list.append(grad_norm)

        nan_or_inf_flag = bool(
            np.isnan(g_comb_param).any()
            or np.isinf(g_comb_param).any()
            or any(
                np.isnan(float(out[k])) or np.isinf(float(out[k]))
                for k in ["loss_dec", "loss_pred", "loss_fair"]
            )
        )

        lr_t = lr_value(t=t, lr=lr0, lr_decay=lr_decay)
        for group in optimizer.param_groups:
            group["lr"] = lr_t

        if nan_or_inf_flag:
            nan_or_inf_steps += 1
            model.zero_grad(set_to_none=True)
            delta_theta_l2 = 0.0
        else:
            params_before = None
            if do_log:
                with torch.no_grad():
                    params_before = [p.detach().clone() for p in model.parameters()]

            if grad_norm > explode_threshold:
                exploding_steps += 1
            if grad_clip_norm > 0.0 and grad_norm > grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            # FAMO requires a weight update after each parameter step:
            # re-evaluate losses on the same batch with updated parameters,
            # then let FAMO adjust its softmax logits accordingly.
            if isinstance(mo_handler, FAMOHandler):
                model.eval()
                with torch.no_grad():
                    if isinstance(task, MedicalResourceAllocationTask):
                        new_raw_out = model(xb_t).reshape(-1)
                        new_pred = _medical_pred_from_model_output(new_raw_out, family=family, post_processor=post_processor)
                        new_pred_np = new_pred.detach().cpu().numpy().reshape(-1)
                        new_out = task.compute_batch(
                            raw_pred=new_pred_np,
                            true=batch.y,
                            cost=batch.cost,
                            race=batch.race,
                            need_grads=False,
                            fairness_smoothing=fairness_smoothing,
                        )
                    else:
                        new_pred = model(xb_t)
                        new_pred_np = new_pred.detach().cpu().numpy()
                        new_out = task.compute(
                            raw_pred=new_pred_np,
                            true=yb,
                            need_grads=False,
                            fairness_smoothing=fairness_smoothing,
                        )
                model.train()
                mo_handler.update_weights({
                    "pred_loss": float(new_out["loss_pred"]),
                    "decision_regret": float(new_out["loss_dec"]),
                    "pred_fairness": float(new_out["loss_fair"]),
                })

            if beta_mode == "constraint" and iter_spec.use_fair:
                dual_lambda = max(0.0, dual_lambda + dual_lr * (float(out["loss_fair"]) - fair_tau))

            solver_calls_total += int(out["solver_calls"]) + int(solver_calls_fd)
            decision_ms_total += float(out["decision_ms"]) + float(decision_ms_fd)
            solver_calls_fd_total += int(solver_calls_fd)
            decision_ms_fd_total += float(decision_ms_fd)

            delta_theta_l2 = 0.0
            if params_before is not None:
                with torch.no_grad():
                    total = 0.0
                    for p, pb in zip(model.parameters(), params_before):
                        d = (p.detach() - pb).reshape(-1)
                        total += float((d * d).sum().item())
                    delta_theta_l2 = float(np.sqrt(total))

        if do_log:
            iter_row: Dict[str, Any] = {
                "task": task.name,
                "method": method_name,
                "seed": seed,
                "stage_idx": stage_idx,
                "lambda": lambda_value,
                "iter": t,
                "alpha_t": alpha_t,
                "beta_t": beta_t,
                "lr_t": lr_t,
                "loss_dec": float(out["loss_dec"]),
                "loss_pred": float(out["loss_pred"]),
                "loss_fair": float(out["loss_fair"]),
                "grad_norm_dec": l2_norm(g_dec_param),
                "grad_norm_pred": l2_norm(g_pred_param),
                "grad_norm_fair": l2_norm(g_fair_param),
                "grad_norm_combined": grad_norm,
                "delta_theta_l2": float(delta_theta_l2),
                # guided_diag is NaN-filled when mo_handler is active (unused path).
                "guided_norm_dec": float(guided_diag["guided_norm_dec"]) if mo_handler is None else float("nan"),
                "guided_norm_pred": float(guided_diag["guided_norm_pred"]) if mo_handler is None else float("nan"),
                "guided_scale": float(guided_diag["guided_scale"]) if mo_handler is None else float("nan"),
                "guided_ratio_pred_over_dec": float(guided_diag["guided_ratio_pred_over_dec"]) if mo_handler is None else float("nan"),
                "guided_dir_norm": float(guided_diag["guided_dir_norm"]) if mo_handler is None else float("nan"),
                "cos_dec_pred": cos_dec_pred,
                "cos_dec_fair": cos_dec_fair,
                "cos_pred_fair": cos_pred_fair,
                "solver_calls": int(out["solver_calls"]),
                "decision_ms": float(out["decision_ms"]),
                "solver_calls_fd": int(solver_calls_fd),
                "decision_ms_fd": float(decision_ms_fd),
                "nan_or_inf_flag": int(nan_or_inf_flag),
                "nan_or_inf_steps_so_far": nan_or_inf_steps,
                "exploding_steps_so_far": exploding_steps,
                "device": str(device),
            }
            if "loss_dec_normalized" in out:
                iter_row["loss_dec_normalized"] = float(out["loss_dec_normalized"])
            if "loss_dec_normalized_true" in out:
                iter_row["loss_dec_normalized_true"] = float(out["loss_dec_normalized_true"])
            if "loss_dec_normalized_pred_obj" in out:
                iter_row["loss_dec_normalized_pred_obj"] = float(out["loss_dec_normalized_pred_obj"])
            if mo_handler is not None:
                iter_row.update(mo_handler.extra_logs())
                iter_row["mo_method"] = str(mo_method)
                iter_row["grad_backend"] = "finite_diff" if use_fd else "closed_form"
            iter_logs.append(iter_row)

    stage_wallclock = float(perf_counter() - stage_start)

    # SAA: build constant-prediction arrays for evaluation
    saa_override_val = None
    saa_override_test = None
    if method_name == "saa":
        if isinstance(task, MedicalResourceAllocationTask):
            saa_override_val = np.full(task._splits["val"].y.shape[0], saa_mean)
            saa_override_test = np.full(task._splits["test"].y.shape[0], saa_mean)
        else:
            saa_override_val = np.full(data.val.y.shape, saa_mean)
            saa_override_test = np.full(data.test.y.shape, saa_mean)

    if isinstance(task, MedicalResourceAllocationTask):
        val_metrics = _eval_split_medical(
            task=task,
            model=model,
            split_name="val",
            fairness_smoothing=fairness_smoothing,
            device=device,
            dtype=dtype,
            family=family,
            override_pred=saa_override_val,
            post_processor=post_processor,
        )
        test_metrics = _eval_split_medical(
            task=task,
            model=model,
            split_name="test",
            fairness_smoothing=fairness_smoothing,
            device=device,
            dtype=dtype,
            family=family,
            override_pred=saa_override_test,
            post_processor=post_processor,
        )
    else:
        val_metrics = _eval_split(
            task=task,
            model=model,
            split=data.val,
            fairness_smoothing=fairness_smoothing,
            device=device,
            dtype=dtype,
            override_pred=saa_override_val,
        )
        test_metrics = _eval_split(
            task=task,
            model=model,
            split=data.test,
            fairness_smoothing=fairness_smoothing,
            device=device,
            dtype=dtype,
            override_pred=saa_override_test,
        )

    grad_min = float(np.min(norm_combined_list)) if norm_combined_list else 0.0
    grad_median = float(np.median(norm_combined_list)) if norm_combined_list else 0.0
    grad_max = float(np.max(norm_combined_list)) if norm_combined_list else 0.0

    stage_row: Dict[str, Any] = {
        "task": task.name,
        "method": method_name,
        "seed": seed,
        "stage_idx": stage_idx,
        "lambda": lambda_value,
        "val_regret": val_metrics["regret"],
        "val_fairness": val_metrics["fairness"],
        "val_pred_mse": val_metrics["pred_mse"],
        "test_regret": test_metrics["regret"],
        "test_fairness": test_metrics["fairness"],
        "test_pred_mse": test_metrics["pred_mse"],
        "stage_wallclock_sec": stage_wallclock,
        "solver_calls_train": solver_calls_total,
        "solver_calls_eval": val_metrics["solver_calls_eval"] + test_metrics["solver_calls_eval"],
        "decision_ms_train": decision_ms_total,
        "decision_ms_eval": val_metrics["decision_ms_eval"] + test_metrics["decision_ms_eval"],
        "solver_calls_fd_train": solver_calls_fd_total,
        "decision_ms_fd_train": decision_ms_fd_total,
        "nan_or_inf_steps": nan_or_inf_steps,
        "nan_steps": nan_or_inf_steps,
        "exploding_steps": exploding_steps,
        "avg_grad_norm_combined": _safe_mean(norm_combined_list),
        "grad_norm_min": grad_min,
        "grad_norm_median": grad_median,
        "grad_norm_max": grad_max,
        "avg_cos_dec_pred": _safe_mean(cos_dec_pred_list),
        "avg_cos_dec_fair": _safe_mean(cos_dec_fair_list),
        "avg_cos_pred_fair": _safe_mean(cos_pred_fair_list),
        "weight_norm": parameter_l2_norm(model),
        "device": str(device),
    }
    if "regret_normalized" in val_metrics:
        stage_row["val_regret_normalized"] = float(val_metrics["regret_normalized"])
        stage_row["test_regret_normalized"] = float(test_metrics["regret_normalized"])
    if "regret_normalized_true" in val_metrics:
        stage_row["val_regret_normalized_true"] = float(val_metrics["regret_normalized_true"])
        stage_row["test_regret_normalized_true"] = float(test_metrics["regret_normalized_true"])
    if "regret_normalized_pred_obj" in val_metrics:
        stage_row["val_regret_normalized_pred_obj"] = float(val_metrics["regret_normalized_pred_obj"])
        stage_row["test_regret_normalized_pred_obj"] = float(test_metrics["regret_normalized_pred_obj"])

    return stage_row, iter_logs


def _run_method_seed(
    task: BaseTask,
    data: TaskData,
    train_cfg: Dict[str, Any],
    seed: int,
    method_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_spec = _resolve_base_spec(method_name)
    lambdas = [float(x) for x in train_cfg["lambdas"]]
    force_lambda_path_all_methods = bool(train_cfg.get("force_lambda_path_all_methods", False))
    if (not base_spec.use_fair) and (not force_lambda_path_all_methods):
        lambdas = [0.0]

    family = str(train_cfg.get("predictor_family", "linear"))
    device = resolve_device_or_warn(str(train_cfg.get("device", "cuda")))
    dtype = torch.float64
    model, family, post_processor = _build_predictor(
        family=family,
        input_dim=data.train.x.shape[1],
        output_dim=data.train.y.shape[1],
        seed=seed,
        device=device,
        dtype=dtype,
        train_cfg=train_cfg,
    )
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    stage_rows: List[Dict[str, Any]] = []
    iter_rows: List[Dict[str, Any]] = []
    cumulative_wallclock = 0.0

    for stage_idx, lam in enumerate(lambdas):
        restart_per_lambda = bool(train_cfg.get("restart_per_lambda", False))
        if stage_idx > 0 and (restart_per_lambda or (not base_spec.continuation)):
            model.load_state_dict(initial_state)

        stage_row, iter_logs = _train_single_stage(
            task=task,
            data=data,
            model=model,
            family=family,
            base_spec=base_spec,
            train_cfg=train_cfg,
            lambda_value=lam,
            seed=seed,
            method_name=method_name,
            stage_idx=stage_idx,
            device=device,
            dtype=dtype,
            post_processor=post_processor,
        )
        cumulative_wallclock += float(stage_row["stage_wallclock_sec"])
        stage_row["cumulative_wallclock_sec"] = cumulative_wallclock
        stage_rows.append(stage_row)
        iter_rows.extend(iter_logs)

    return stage_rows, iter_rows


def run_core_methods(
    task: BaseTask,
    data: TaskData,
    train_cfg: Dict[str, Any],
    methods: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    seeds = [int(s) for s in train_cfg["seeds"]]
    stage_rows: List[Dict[str, Any]] = []
    iter_rows: List[Dict[str, Any]] = []

    for method in methods:
        if method not in METHOD_SPECS:
            raise ValueError(f"Unknown method: {method}")
        for seed in seeds:
            rows, logs = _run_method_seed(
                task=task,
                data=data,
                train_cfg=train_cfg,
                seed=seed,
                method_name=method,
            )
            stage_rows.extend(rows)
            iter_rows.extend(logs)
    return stage_rows, iter_rows
