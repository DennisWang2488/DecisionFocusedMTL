"""Synthetic multi-dimensional knapsack task."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List

import cvxpy as cp
import numpy as np

from ..losses import group_fairness_loss_and_grad, mse_loss_and_grad, softplus_with_grad
from .base import BaseTask, SplitData, TaskData


@dataclass
class MultiDimKnapsackTask(BaseTask):
    """Multi-dimensional knapsack with LP and alpha-fair objectives."""

    n_samples_train: int
    n_samples_val: int
    n_samples_test: int
    n_features: int
    n_items: int
    n_constraints: int
    scenario: str
    alpha_fair: float = 2.0
    group_bias: float = 0.3
    noise_std_lo: float = 0.1
    noise_std_hi: float = 0.5
    poly_degree: int = 2
    budget_tightness: float = 0.5
    fairness_type: str = "mad"
    fairness_ge_alpha: float = 2.0
    _current_groups: np.ndarray = field(default=None, repr=False, init=False)
    _current_A: np.ndarray = field(default=None, repr=False, init=False)
    _current_b: np.ndarray = field(default=None, repr=False, init=False)
    _cvx_problem: cp.Problem = field(default=None, repr=False, init=False)
    _cvx_r_param: cp.Parameter = field(default=None, repr=False, init=False)
    _cvx_d_var: cp.Variable = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        if self.scenario not in {"lp", "alpha_fair"}:
            raise ValueError(f"scenario must be 'lp' or 'alpha_fair', got {self.scenario!r}")
        if self.alpha_fair <= 0.0:
            raise ValueError("alpha_fair must be positive.")
        self.name = "md_knapsack"
        self.n_outputs = self.n_items

    def generate_data(self, seed: int) -> TaskData:
        rng = np.random.default_rng(seed)

        groups = np.zeros(self.n_items, dtype=int)
        groups[self.n_items // 2:] = 1

        A = rng.uniform(0.5, 1.5, size=(self.n_constraints, self.n_items))
        b = self.budget_tightness * A.sum(axis=1)

        rng_w = np.random.default_rng(seed + 1000)
        W_list = [
            rng_w.normal(scale=1.0 / degree, size=(self.n_features, self.n_items))
            for degree in range(1, self.poly_degree + 1)
        ]

        group_shift = np.where(groups == 0, self.group_bias, -self.group_bias)
        noise_scale = np.where(groups == 0, self.noise_std_lo, self.noise_std_hi)

        def sample(n_rows: int) -> SplitData:
            x = rng.normal(size=(n_rows, self.n_features))
            raw = np.zeros((n_rows, self.n_items), dtype=float)
            for degree, weights in enumerate(W_list, start=1):
                raw += np.power(x, degree) @ weights
            noise = rng.normal(size=(n_rows, self.n_items)) * noise_scale[None, :]
            y_raw = raw + group_shift[None, :] + noise
            y_pos, _ = softplus_with_grad(y_raw)
            y = y_pos + 0.05
            return SplitData(x=x, y=y)

        return TaskData(
            train=sample(self.n_samples_train),
            val=sample(self.n_samples_val),
            test=sample(self.n_samples_test),
            groups=groups,
            meta={"A": A, "b": b},
        )

    def bind_context(self, groups: np.ndarray, A: np.ndarray, b: np.ndarray) -> None:
        self._current_groups = np.asarray(groups, dtype=int)
        self._current_A = np.asarray(A, dtype=float)
        self._current_b = np.asarray(b, dtype=float)
        self._build_cvxpy(A=self._current_A, b=self._current_b)

    def _build_cvxpy(self, A: np.ndarray, b: np.ndarray) -> None:
        d = cp.Variable(self.n_items, nonneg=True)
        r = cp.Parameter(self.n_items, nonneg=True)
        constraints = [A @ d <= b]

        if self.scenario == "lp":
            objective = cp.Maximize(r @ d)
        else:
            alpha = self.alpha_fair
            utility = cp.multiply(r, d)
            constraints.append(d >= 1e-6)
            if abs(alpha - 1.0) < 1e-12:
                objective = cp.Maximize(cp.sum(cp.log(utility)))
            elif alpha < 1.0:
                objective = cp.Maximize(cp.sum(cp.power(utility, 1.0 - alpha)) / (1.0 - alpha))
            else:
                objective = cp.Maximize(-cp.sum(cp.power(utility, 1.0 - alpha)) / (alpha - 1.0))

        self._cvx_problem = cp.Problem(objective, constraints)
        self._cvx_r_param = r
        self._cvx_d_var = d

    def build_cvxpy_problem(self):
        if self._cvx_problem is None:
            raise RuntimeError("Call bind_context() before build_cvxpy_problem().")
        return self._cvx_problem, [self._cvx_r_param], [self._cvx_d_var]

    def _solve_single(self, benefit: np.ndarray) -> np.ndarray:
        self._cvx_r_param.value = np.clip(np.asarray(benefit, dtype=float), 1e-8, None)
        try:
            self._cvx_problem.solve(solver=cp.ECOS, warm_start=True)
        except cp.SolverError:
            self._cvx_problem.solve(solver=cp.SCS, warm_start=False)
        if self._cvx_d_var.value is None:
            return np.zeros(self.n_items, dtype=float)
        return np.clip(np.asarray(self._cvx_d_var.value, dtype=float), 0.0, None)

    def _solve_batch(self, benefit: np.ndarray) -> np.ndarray:
        benefit_2d = np.atleast_2d(np.asarray(benefit, dtype=float))
        out = np.zeros_like(benefit_2d)
        for idx in range(benefit_2d.shape[0]):
            out[idx] = self._solve_single(benefit_2d[idx])
        return out

    def _objective_batch(self, decision: np.ndarray, true_benefit: np.ndarray) -> np.ndarray:
        decision_2d = np.atleast_2d(np.asarray(decision, dtype=float))
        true_2d = np.atleast_2d(np.asarray(true_benefit, dtype=float))
        if self.scenario == "lp":
            return np.sum(true_2d * decision_2d, axis=1)

        eps = 1e-10
        alpha = self.alpha_fair
        utility = np.clip(true_2d * decision_2d, eps, None)
        if abs(alpha - 1.0) < 1e-12:
            return np.sum(np.log(utility), axis=1)
        if alpha < 1.0:
            return np.sum(np.power(utility, 1.0 - alpha) / (1.0 - alpha), axis=1)
        return np.sum(-np.power(utility, 1.0 - alpha) / (alpha - 1.0), axis=1)

    def _decision_regret(
        self,
        pred_pos: np.ndarray,
        true: np.ndarray,
    ) -> tuple[float, float, float, int, float]:
        t0 = perf_counter()
        d_true = self._solve_batch(true)
        d_pred = self._solve_batch(pred_pos)
        solver_calls = 2 * pred_pos.shape[0]
        decision_ms = (perf_counter() - t0) * 1000.0

        obj_true = self._objective_batch(d_true, true)
        obj_pred = self._objective_batch(d_pred, true)
        regret = np.maximum(obj_true - obj_pred, 0.0)
        loss_dec = float(np.mean(regret))
        denom = np.abs(obj_true) + np.abs(obj_pred) + 1e-8
        loss_dec_norm = float(np.mean(regret / denom))
        loss_dec_norm_true = float(np.mean(regret / (np.abs(obj_true) + 1e-8)))
        return loss_dec, loss_dec_norm, loss_dec_norm_true, solver_calls, decision_ms

    def compute(
        self,
        raw_pred: np.ndarray,
        true: np.ndarray,
        need_grads: bool,
        fairness_smoothing: float = 1e-6,
    ) -> Dict[str, Any]:
        if need_grads:
            raise ValueError(
                "MultiDimKnapsackTask does not provide analytic decision gradients. "
                "Use training.decision_grad_backend='finite_diff'."
            )

        pred_pos, pred_pos_grad = softplus_with_grad(np.asarray(raw_pred, dtype=float))
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
        loss_dec, loss_dec_norm, loss_dec_norm_true, solver_calls, decision_ms = self._decision_regret(
            pred_pos,
            np.asarray(true, dtype=float),
        )

        return {
            "loss_dec": loss_dec,
            "loss_dec_normalized": loss_dec_norm,
            "loss_dec_normalized_true": loss_dec_norm_true,
            "loss_pred": loss_pred,
            "loss_fair": loss_fair,
            "grad_dec": np.zeros_like(raw_pred),
            "grad_pred": grad_pred_pos * pred_pos_grad,
            "grad_fair": grad_fair_pos * pred_pos_grad,
            "solver_calls": solver_calls,
            "decision_ms": decision_ms,
        }

    def solve_decision(self, pred: np.ndarray, **ctx: Any) -> np.ndarray:
        pred_2d = np.atleast_2d(np.asarray(pred, dtype=float))
        pred_pos, _ = softplus_with_grad(pred_2d)
        pred_pos = pred_pos + 1e-5
        return self._solve_batch(pred_pos)

    def solve_oracle_decision(self, true: np.ndarray, **ctx: Any) -> np.ndarray:
        true_2d = np.atleast_2d(np.asarray(true, dtype=float))
        return self._solve_batch(np.clip(true_2d, 1e-8, None))

    def evaluate_objective(self, decision: np.ndarray, true: np.ndarray, **ctx: Any) -> float:
        return float(np.mean(self._objective_batch(decision, true)))

    def supported_gradient_strategies(self) -> List[str]:
        return ["finite_diff"]
