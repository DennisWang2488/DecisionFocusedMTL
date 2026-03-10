"""Task layer for data-driven optimization experiments used by the 8 active methods."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict

import numpy as np

from ..losses import group_fairness_loss_and_grad, mse_loss_and_grad, softplus_with_grad
from .base import BaseTask, SplitData, TaskData, add_bias_column


@dataclass
class ResourceAllocationTask(BaseTask):
    n_samples_train: int
    n_samples_val: int
    n_samples_test: int
    n_features: int
    n_items: int
    alpha_fair: float
    budget: float
    group_bias: float
    noise_std: float
    fairness_type: str = "mad"
    fairness_ge_alpha: float = 2.0

    def __post_init__(self) -> None:
        self.name = "resource_allocation"
        self.n_outputs = self.n_items
        if self.alpha_fair <= 0.0:
            raise ValueError("alpha_fair must be positive.")

    def generate_data(self, seed: int) -> TaskData:
        rng = np.random.default_rng(seed)
        groups = np.zeros(self.n_items, dtype=int)
        groups[self.n_items // 2 :] = 1
        costs = rng.uniform(0.7, 1.6, size=self.n_items)
        true_w = rng.normal(loc=0.0, scale=0.8, size=(self.n_features + 1, self.n_items))

        group_shift = np.where(groups == 0, -self.group_bias, self.group_bias)

        def sample(n_rows: int) -> SplitData:
            x = rng.normal(size=(n_rows, self.n_features))
            raw = add_bias_column(x) @ true_w
            noise = rng.normal(scale=self.noise_std, size=raw.shape)
            y_raw = raw + group_shift[None, :] + noise
            y_pos, _ = softplus_with_grad(y_raw)
            y = y_pos + 0.05
            return SplitData(x=x, y=y)

        train = sample(n_rows=self.n_samples_train)
        val = sample(n_rows=self.n_samples_val)
        test = sample(n_rows=self.n_samples_test)

        return TaskData(
            train=train,
            val=val,
            test=test,
            groups=groups,
            meta={"costs": costs},
        )

    def _solve_allocation_batch(self, benefit: np.ndarray, costs: np.ndarray) -> np.ndarray:
        eps = 1e-10
        alpha = self.alpha_fair
        if abs(alpha - 1.0) < 1e-12:
            d = self.budget / (benefit.shape[1] * costs)
            return np.repeat(d[None, :], benefit.shape[0], axis=0)

        clipped = np.clip(benefit, eps, None)
        exponent = (1.0 - alpha) / alpha
        numer = (costs[None, :] ** (-1.0 / alpha)) * (clipped**exponent)
        denom = np.sum(costs[None, :] * numer, axis=1, keepdims=True) + eps
        return self.budget * numer / denom

    def _objective(self, decision: np.ndarray, true_benefit: np.ndarray) -> np.ndarray:
        eps = 1e-10
        u = np.clip(decision * true_benefit, eps, None)
        alpha = self.alpha_fair
        if abs(alpha - 1.0) < 1e-12:
            return np.sum(np.log(u), axis=1)
        return np.sum(u ** (1.0 - alpha) / (1.0 - alpha), axis=1)

    def _grad_obj_wrt_decision(self, decision: np.ndarray, true_benefit: np.ndarray) -> np.ndarray:
        eps = 1e-10
        u = np.clip(decision * true_benefit, eps, None)
        alpha = self.alpha_fair
        if abs(alpha - 1.0) < 1e-12:
            return true_benefit / u
        return true_benefit * (u ** (-alpha))

    def _allocation_jacobian(self, pred_benefit: np.ndarray, decision: np.ndarray, costs: np.ndarray) -> np.ndarray:
        n = pred_benefit.size
        alpha = self.alpha_fair
        if abs(alpha - 1.0) < 1e-12:
            return np.zeros((n, n), dtype=float)
        eps = 1e-10
        pred = np.clip(pred_benefit, eps, None)
        term = (1.0 / alpha - 1.0) / pred
        jac = -np.outer(decision, decision * term * costs) / self.budget
        diag = decision * term * (1.0 - costs * decision / self.budget)
        np.fill_diagonal(jac, diag)
        return jac

    def _decision_regret_and_grad(
        self,
        pred_pos: np.ndarray,
        true: np.ndarray,
        costs: np.ndarray,
        need_grads: bool,
    ) -> tuple[float, float, float, np.ndarray, int]:
        d_true = self._solve_allocation_batch(true, costs)
        d_pred = self._solve_allocation_batch(pred_pos, costs)
        obj_true = self._objective(d_true, true)
        obj_pred = self._objective(d_pred, true)
        # Numerical solvers should yield obj_true >= obj_pred, but clip in case of
        # floating point drift so regret-based metrics remain non-negative.
        regret = np.maximum(obj_true - obj_pred, 0.0)
        loss_dec = float(np.mean(regret))
        denom_bounded = np.abs(obj_true) + np.abs(obj_pred) + 1e-8
        loss_dec_normalized = float(np.mean(regret / denom_bounded))
        loss_dec_normalized_true = float(np.mean(regret / (np.abs(obj_true) + 1e-8)))

        grad_pred = np.zeros_like(pred_pos)
        if need_grads:
            batch = pred_pos.shape[0]
            grad_obj_d = self._grad_obj_wrt_decision(d_pred, true)
            for b in range(batch):
                grad_d = -grad_obj_d[b] / float(batch)
                jac = self._allocation_jacobian(pred_pos[b], d_pred[b], costs)
                grad_pred[b] = jac.T @ grad_d
        solver_calls = int(2 * pred_pos.shape[0])
        return loss_dec, loss_dec_normalized, loss_dec_normalized_true, grad_pred, solver_calls

    def compute(
        self,
        raw_pred: np.ndarray,
        true: np.ndarray,
        need_grads: bool,
        fairness_smoothing: float = 1e-6,
    ) -> Dict[str, np.ndarray | float]:
        costs = self._current_costs
        pred_pos, pred_pos_grad = softplus_with_grad(raw_pred)
        pred_pos = pred_pos + 1e-5

        loss_pred, grad_pred_pos = mse_loss_and_grad(pred_pos, true)
        loss_fair, grad_fair_pos = group_fairness_loss_and_grad(
            pred_pos,
            true,
            self._current_groups,
            fairness_type=self.fairness_type,
            smoothing=fairness_smoothing,
            ge_alpha=self.fairness_ge_alpha,
        )

        t0 = perf_counter()
        loss_dec, loss_dec_normalized, loss_dec_normalized_true, grad_dec_pos, solver_calls = self._decision_regret_and_grad(
            pred_pos=pred_pos,
            true=true,
            costs=costs,
            need_grads=need_grads,
        )
        decision_ms = (perf_counter() - t0) * 1000.0

        # Always provide prediction/fairness gradients (FPTO needs these even when
        # analytic decision gradients are not requested). Only gate decision
        # gradients on need_grads.
        grad_pred = grad_pred_pos * pred_pos_grad
        grad_fair = grad_fair_pos * pred_pos_grad
        grad_dec = grad_dec_pos * pred_pos_grad if need_grads else np.zeros_like(raw_pred)

        return {
            "loss_dec": loss_dec,
            "loss_dec_normalized": loss_dec_normalized,
            "loss_dec_normalized_true": loss_dec_normalized_true,
            "loss_pred": loss_pred,
            "loss_fair": loss_fair,
            "grad_dec": grad_dec,
            "grad_pred": grad_pred,
            "grad_fair": grad_fair,
            "solver_calls": solver_calls,
            "decision_ms": decision_ms,
        }

    def bind_context(self, groups: np.ndarray, costs: np.ndarray) -> None:
        self._current_groups = groups
        self._current_costs = costs
