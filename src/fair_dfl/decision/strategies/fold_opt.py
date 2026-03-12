"""Fold-Opt (FFO) decision gradient strategy.

Uses a differentiable CVXPY layer to compute decision gradients
via implicit differentiation of the KKT conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict

import numpy as np
import torch

from ...tasks.base import BaseTask
from ...tasks.medical_resource_allocation import MedicalResourceAllocationTask
from ..interface import DecisionGradientStrategy, DecisionResult


def _torch(x: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=device)


def _build_ffo_layer(batch_size: int, ffo_cfg: Dict[str, Any]):
    import cvxpy as cp
    from ...advanced.ffolayer_local import FFOLayer

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


@dataclass
class _FFOState:
    layer_cache: Dict[int, Any] = field(default_factory=dict)
    cfg: Dict[str, Any] = field(default_factory=dict)
    budget: float = 2500.0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float32


class FoldOptStrategy(DecisionGradientStrategy):
    """FFO: Fold-Optimization layer for differentiable decision solving.

    Wraps a CVXPY problem in a differentiable layer that computes
    gradients via implicit differentiation of KKT conditions.
    """

    def __init__(
        self,
        ffo_cfg: Dict[str, Any] | None = None,
        budget: float = 2500.0,
        device: torch.device | None = None,
    ) -> None:
        self._state = _FFOState(
            cfg=dict(ffo_cfg or {}),
            budget=budget,
            device=device or torch.device("cpu"),
        )

    def compute(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        task: BaseTask,
        need_grads: bool = True,
        fairness_smoothing: float = 1e-6,
        **ctx: Any,
    ) -> DecisionResult:
        task_output = ctx.get("task_output")
        if task_output is not None:
            out = task_output
            base_solver_calls = 0
            base_decision_ms = 0.0
        elif isinstance(task, MedicalResourceAllocationTask):
            out = task.compute_batch(
                raw_pred=pred,
                true=true,
                cost=ctx["cost"],
                race=ctx["race"],
                need_grads=True,
                fairness_smoothing=fairness_smoothing,
            )
            base_solver_calls = int(out.get("solver_calls", 0))
            base_decision_ms = float(out.get("decision_ms", 0.0))
        else:
            out = task.compute(
                raw_pred=pred, true=true,
                need_grads=False, fairness_smoothing=fairness_smoothing,
            )
            base_solver_calls = int(out.get("solver_calls", 0))
            base_decision_ms = float(out.get("decision_ms", 0.0))

        if not need_grads:
            return DecisionResult(
                loss_dec=float(out["loss_dec"]),
                grad_dec=np.zeros_like(pred),
                solver_calls=base_solver_calls,
                decision_ms=base_decision_ms,
                task_output=out,
            )

        # Compute FFO gradient
        cost = ctx.get("cost", np.ones_like(pred))
        grad, backward_ms = self._ffo_grad(pred, true, cost)

        return DecisionResult(
            loss_dec=float(out["loss_dec"]),
            grad_dec=grad,
            solver_calls=base_solver_calls,
            decision_ms=base_decision_ms + backward_ms,
            task_output=out,
            extra={"ffo_backward_ms": backward_ms},
        )

    def _ffo_grad(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        cost: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        bsz = int(pred.shape[0])
        state = self._state
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

    def reset(self) -> None:
        self._state.layer_cache.clear()

    def supports_task(self, task: BaseTask) -> bool:
        return isinstance(task, MedicalResourceAllocationTask)
