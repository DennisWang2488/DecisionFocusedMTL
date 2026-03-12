"""Finite-difference decision gradient strategy.

Computes decision gradients by perturbing predictions and re-solving
the optimization problem. Generalized to work with any task that
implements solve_decision() and evaluate_objective().
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from ...tasks.base import BaseTask
from ...tasks.medical_resource_allocation import MedicalResourceAllocationTask
from ...tasks.resource_allocation import ResourceAllocationTask
from ..interface import DecisionGradientStrategy, DecisionResult


def _softplus_np(x: np.ndarray) -> np.ndarray:
    positive = np.maximum(x, 0.0)
    exp_term = np.exp(-np.abs(x))
    return positive + np.log1p(exp_term)


class FiniteDiffStrategy(DecisionGradientStrategy):
    """Compute decision gradients via central finite differences.

    For each prediction element, perturbs by +/- eps, re-solves the
    optimization problem, and estimates the gradient from regret differences.

    Works with any task that implements solve_decision() and evaluate_objective(),
    or falls back to task-specific logic for known task types.
    """

    def __init__(self, eps: float = 1e-3) -> None:
        self.eps = eps

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
                need_grads=False,
                fairness_smoothing=fairness_smoothing,
            )
            base_solver_calls = int(out.get("solver_calls", 0))
            base_decision_ms = float(out.get("decision_ms", 0.0))
        else:
            out = task.compute(
                raw_pred=pred,
                true=true,
                need_grads=False,
                fairness_smoothing=fairness_smoothing,
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

        # Compute finite-diff gradient
        t0 = perf_counter()

        # Try generic interface first
        if hasattr(task, "solve_decision") and hasattr(task, "evaluate_objective"):
            try:
                grad, solver_calls = self._generic_finite_diff(pred, true, task, **ctx)
                decision_ms = (perf_counter() - t0) * 1000.0
                return DecisionResult(
                    loss_dec=float(out["loss_dec"]),
                    grad_dec=grad,
                    solver_calls=base_solver_calls + solver_calls,
                    decision_ms=base_decision_ms + decision_ms,
                    task_output=out,
                )
            except NotImplementedError:
                pass

        # Fall back to task-specific implementations
        if isinstance(task, ResourceAllocationTask):
            grad, solver_calls = self._fd_resource_allocation(pred, true, task)
        else:
            raise ValueError(
                f"Finite-difference not implemented for {type(task).__name__}. "
                f"Implement solve_decision() and evaluate_objective() on the task."
            )

        decision_ms = (perf_counter() - t0) * 1000.0
        return DecisionResult(
            loss_dec=float(out["loss_dec"]),
            grad_dec=grad,
            solver_calls=base_solver_calls + solver_calls,
            decision_ms=base_decision_ms + decision_ms,
            task_output=out,
        )

    def _generic_finite_diff(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        task: BaseTask,
        **ctx: Any,
    ) -> tuple[np.ndarray, int]:
        """Generic finite-diff using task.solve_decision() and task.evaluate_objective()."""
        bsz = int(pred.shape[0])
        dim = int(pred.reshape(bsz, -1).shape[1])
        pred_flat = pred.reshape(bsz, dim)
        grad = np.zeros_like(pred_flat, dtype=float)
        solver_calls = 0

        # Compute true objective for each sample
        obj_true = np.zeros(bsz, dtype=float)
        for b in range(bsz):
            d_true = task.solve_decision(true[b:b+1], **ctx)
            obj_true[b] = task.evaluate_objective(d_true, true[b:b+1], **ctx)
            solver_calls += 1

        for b in range(bsz):
            for j in range(dim):
                pred_plus = pred_flat[b].copy()
                pred_minus = pred_flat[b].copy()
                pred_plus[j] += self.eps
                pred_minus[j] -= self.eps

                d_plus = task.solve_decision(pred_plus[None, :], **ctx)
                d_minus = task.solve_decision(pred_minus[None, :], **ctx)

                obj_plus = task.evaluate_objective(d_plus, true[b:b+1], **ctx)
                obj_minus = task.evaluate_objective(d_minus, true[b:b+1], **ctx)

                regret_plus = max(float(obj_true[b] - obj_plus), 0.0)
                regret_minus = max(float(obj_true[b] - obj_minus), 0.0)

                grad[b, j] = (regret_plus - regret_minus) / (2.0 * self.eps * bsz)
                solver_calls += 2

        return grad.reshape(pred.shape), solver_calls

    def _fd_resource_allocation(
        self,
        raw_pred: np.ndarray,
        true: np.ndarray,
        task: ResourceAllocationTask,
    ) -> tuple[np.ndarray, int]:
        """Legacy finite-diff for ResourceAllocationTask with softplus transform."""
        costs = np.asarray(task._current_costs, dtype=float)
        raw = np.asarray(raw_pred, dtype=float)
        pred_pos = _softplus_np(raw) + 1e-5
        y_true = np.asarray(true, dtype=float)
        bsz = int(raw.shape[0])
        dim = int(raw.shape[1])
        grad = np.zeros_like(raw, dtype=float)

        obj_true = np.zeros(bsz, dtype=float)
        for b in range(bsz):
            d_true = task._solve_allocation_batch(y_true[b:b+1], costs)
            obj_true[b] = float(task._objective(d_true, y_true[b:b+1])[0])

        solver_calls = 0
        for b in range(bsz):
            base_pos = pred_pos[b].copy()
            base_raw = raw[b]
            yb = y_true[b:b+1]
            for j in range(dim):
                plus_pos = base_pos.copy()
                minus_pos = base_pos.copy()
                plus_pos[j] = float(_softplus_np(np.array([base_raw[j] + self.eps]))[0] + 1e-5)
                minus_pos[j] = float(_softplus_np(np.array([base_raw[j] - self.eps]))[0] + 1e-5)

                d_plus = task._solve_allocation_batch(plus_pos[None, :], costs)
                d_minus = task._solve_allocation_batch(minus_pos[None, :], costs)
                obj_pred_plus = float(task._objective(d_plus, yb)[0])
                obj_pred_minus = float(task._objective(d_minus, yb)[0])
                regret_plus = max(float(obj_true[b] - obj_pred_plus), 0.0)
                regret_minus = max(float(obj_true[b] - obj_pred_minus), 0.0)
                grad[b, j] = (regret_plus - regret_minus) / (2.0 * self.eps * bsz)
                solver_calls += 2

        return grad, int(solver_calls + bsz)

    def supports_task(self, task: BaseTask) -> bool:
        if hasattr(task, "solve_decision") and hasattr(task, "evaluate_objective"):
            return True
        return isinstance(task, ResourceAllocationTask)
