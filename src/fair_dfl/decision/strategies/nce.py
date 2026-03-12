"""NCE (Neural Contrastive Exploration) decision gradient strategy.

Uses a contrastive loss against a solution pool to compute
decision gradients without explicit solver differentiation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...advanced.nce import NCESolutionPool
from ...tasks.base import BaseTask
from ...tasks.medical_resource_allocation import MedicalResourceAllocationTask
from ..interface import DecisionGradientStrategy, DecisionResult


class NCEStrategy(DecisionGradientStrategy):
    """NCE: contrastive surrogate loss from a solution pool.

    Maintains a pool of past solver solutions and computes a contrastive
    gradient that encourages predictions to yield decisions closer to
    the oracle solution.
    """

    def __init__(
        self,
        pool_size: int = 32,
        solve_ratio: float = 1.0,
        refresh_interval: int = 1,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._pool = NCESolutionPool(pool_size=pool_size, sense="maximize")
        self._solve_ratio = solve_ratio
        self._refresh_interval = max(1, refresh_interval)
        self._rng = rng or np.random.default_rng(0)
        self._step = 0

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
                raw_pred=pred, true=true,
                cost=ctx["cost"], race=ctx["race"],
                need_grads=True, fairness_smoothing=fairness_smoothing,
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
            self._step += 1
            return DecisionResult(
                loss_dec=float(out["loss_dec"]),
                grad_dec=np.zeros_like(pred),
                solver_calls=base_solver_calls,
                decision_ms=base_decision_ms,
                task_output=out,
            )

        # Refresh pool
        if (self._step % self._refresh_interval == 0) and (self._rng.uniform() <= self._solve_ratio):
            self._pool.update(np.asarray(out["decision_pred"], dtype=float).reshape(-1))

        # NCE gradient
        nce_out = self._pool.loss_and_grad(
            pred_score=pred.reshape(-1),
            true_decision=np.asarray(out["decision_true"], dtype=float).reshape(-1),
        )
        grad_dec = np.asarray(nce_out.grad_pred, dtype=float).reshape(pred.shape)

        self._step += 1
        return DecisionResult(
            loss_dec=float(out["loss_dec"]),
            grad_dec=grad_dec,
            solver_calls=base_solver_calls,
            decision_ms=base_decision_ms,
            task_output=out,
            extra={"nce_pool_size": len(self._pool.pool)},
        )

    def reset(self) -> None:
        self._step = 0
        # Keep pool across stages

    def supports_task(self, task: BaseTask) -> bool:
        return isinstance(task, MedicalResourceAllocationTask)
