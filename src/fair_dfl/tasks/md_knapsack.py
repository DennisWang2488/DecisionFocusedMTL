"""Synthetic multi-dimensional knapsack task."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List

import warnings

import cvxpy as cp
import numpy as np

from ..losses import group_fairness_loss_and_grad, mse_loss_and_grad, softplus_with_grad
from .base import BaseTask, SplitData, TaskData

# Solver preference: MOSEK (most accurate, requires license) → CLARABEL
# (bundled with CVXPY 1.4+) → SCS (last resort, may be inaccurate).
_SOLVER_CHAIN = [
    (cp.MOSEK,    {}),
    (cp.CLARABEL, {}),
    (cp.SCS,      {"eps": 1e-6, "max_iters": 10000}),
]
_SCS_WARNED = False


@dataclass
class MultiDimKnapsackTask(BaseTask):
    """Multi-dimensional knapsack with LP and alpha-fair objectives."""

    n_samples_train: int
    n_samples_val: int
    n_samples_test: int
    n_features: int
    n_items: int
    n_budget_dims: int
    scenario: str
    alpha_fair: float = 2.0
    group_bias: float = 0.3
    noise_std_lo: float = 0.1
    noise_std_hi: float = 0.5
    poly_degree: int = 2
    budget_tightness: float = 0.5
    fairness_type: str = "mad"
    fairness_ge_alpha: float = 2.0
    group_ratio: float = 0.5
    decision_mode: str = "group"  # "group" = two-level group alpha-fair (like healthcare), "item" = item-level
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
        if not (0.0 < self.group_ratio <= 1.0):
            raise ValueError(f"group_ratio must be in (0, 1], got {self.group_ratio}")
        if self.decision_mode not in {"group", "item"}:
            raise ValueError(f"decision_mode must be 'group' or 'item', got {self.decision_mode!r}")
        self.name = "md_knapsack"
        self.n_outputs = self.n_items

    def generate_data(self, seed: int) -> TaskData:
        rng = np.random.default_rng(seed)

        # group_ratio controls fraction of items in group 0 (majority group)
        n_group0 = max(1, min(self.n_items - 1, int(round(self.group_ratio * self.n_items))))
        groups = np.zeros(self.n_items, dtype=int)
        groups[n_group0:] = 1

        A = rng.uniform(0.5, 1.5, size=(self.n_budget_dims, self.n_items))
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
        # LP allows negative benefits (needed by SPO+); alpha-fair needs nonneg
        # for power functions.
        r = cp.Parameter(self.n_items, nonneg=(self.scenario != "lp"))
        constraints = [A @ d <= b]

        if self.scenario == "lp":
            constraints.append(d <= 1)  # bounded [0,1] knapsack
            objective = cp.Maximize(r @ d)
        else:
            alpha = self.alpha_fair
            utility = cp.multiply(r, d)
            constraints.append(d >= 1e-6)

            use_group = (
                self.decision_mode == "group"
                and self._current_groups is not None
                and len(np.unique(self._current_groups)) > 1
            )

            if use_group and alpha < 1.0:
                # Two-level group alpha-fairness, 0 < alpha < 1.
                # Paper Eq: g_k = sum_{i in k} u_i^{1-a}/(1-a), then
                #           Phi = sum_k g_k^{1-a}/(1-a).
                # DCP: power(affine, 0<p<1) = concave;
                #      G_k = sum(concave)/pos = concave;
                #      power(concave, 0<p<1) = concave.
                groups = self._current_groups
                group_utilities = []
                for g in np.unique(groups):
                    mask = groups == g
                    G_k = cp.sum(cp.power(utility[mask], 1.0 - alpha)) / (1.0 - alpha)
                    group_utilities.append(G_k)
                terms = [cp.power(G_k, 1.0 - alpha) for G_k in group_utilities]
                objective = cp.Maximize(cp.sum(terms) / (1.0 - alpha))

            elif use_group and alpha >= 2.0 - 1e-12:
                # Two-level group alpha-fairness, alpha >= 2.
                # Direct g_k = (a-1)/S_k is not DCP, but algebraic expansion
                # yields a DCP form:
                #   Phi = c * sum_k S_k^{a-1}
                # where S_k = sum_{i in k} u_i^{1-a}, c = (a-1)^{1-a}/(1-a) < 0.
                #
                # DCP: power(affine, p<0) = convex;
                #      S_k = sum(convex) = convex, nonneg;
                #      power(convex_nonneg, p>=1) = convex  (since a-1 >= 1);
                #      c * sum(convex) with c<0 = concave.
                #
                # NOTE: At alpha=2, this is equivalent to item-level
                # (the inner reciprocal and outer power cancel exactly).
                groups = self._current_groups
                c = (alpha - 1.0) ** (1.0 - alpha) / (1.0 - alpha)
                S_k_list = []
                for g in np.unique(groups):
                    mask = groups == g
                    S_k = cp.sum(cp.power(utility[mask], 1.0 - alpha))
                    S_k_list.append(S_k)
                terms = cp.hstack([cp.power(S_k, alpha - 1.0) for S_k in S_k_list])
                objective = cp.Maximize(c * cp.sum(terms))

            else:
                # Item-level alpha-fairness.
                # Used for: decision_mode="item", single group, alpha=1,
                # or 1 < alpha < 2 (two-level not DCP for that range).
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
        """Solve with benefit clipped to positive (safe for alpha-fair power objectives)."""
        self._cvx_r_param.value = np.clip(np.asarray(benefit, dtype=float), 1e-8, None)
        return self._run_solver()

    def _solve_raw(self, benefit: np.ndarray) -> np.ndarray:
        """Solve LP with raw benefit vector (may contain negatives). For SPO+ only."""
        self._cvx_r_param.value = np.asarray(benefit, dtype=float)
        return self._run_solver()

    def _run_solver(self) -> np.ndarray:
        global _SCS_WARNED
        for solver, kwargs in _SOLVER_CHAIN:
            try:
                self._cvx_problem.solve(solver=solver, **kwargs)
                if self._cvx_d_var.value is not None:
                    if solver == cp.SCS and not _SCS_WARNED:
                        warnings.warn(
                            "Knapsack solver fell back to SCS (MOSEK and CLARABEL unavailable "
                            "or failed). Results may be inaccurate. Install MOSEK for best results.",
                            stacklevel=2,
                        )
                        _SCS_WARNED = True
                    return np.clip(np.asarray(self._cvx_d_var.value, dtype=float), 0.0, None)
            except cp.SolverError:
                continue
        return np.zeros(self.n_items, dtype=float)

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

        use_group = (
            self.decision_mode == "group"
            and self._current_groups is not None
            and len(np.unique(self._current_groups)) > 1
        )

        if not use_group:
            # Item-level alpha-fair evaluation
            if abs(alpha - 1.0) < 1e-12:
                return np.sum(np.log(utility), axis=1)
            if alpha < 1.0:
                return np.sum(np.power(utility, 1.0 - alpha) / (1.0 - alpha), axis=1)
            return np.sum(-np.power(utility, 1.0 - alpha) / (alpha - 1.0), axis=1)

        # Two-level group alpha-fairness (matches healthcare _group_objective).
        # Inner: G_k = per-group alpha-fair aggregate of item utilities.
        # Outer: Phi = alpha-fair aggregate across groups.
        groups = self._current_groups
        unique_groups = np.unique(groups)
        batch = utility.shape[0]

        gk_arr = np.zeros((batch, len(unique_groups)), dtype=float)
        for idx, g in enumerate(unique_groups):
            mask = groups == g
            yk = utility[:, mask]  # (batch, n_items_in_group)
            if abs(alpha - 1.0) < 1e-12:
                gk_arr[:, idx] = np.sum(np.log(yk), axis=1)
            elif 0.0 < alpha < 1.0:
                gk_arr[:, idx] = np.sum(np.power(yk, 1.0 - alpha), axis=1) / (1.0 - alpha)
            elif alpha > 1.0:
                gk_arr[:, idx] = (alpha - 1.0) / np.clip(
                    np.sum(np.power(yk, 1.0 - alpha), axis=1), eps, None
                )
            else:  # alpha == 0
                gk_arr[:, idx] = np.sum(yk, axis=1)

        gk_arr = np.clip(gk_arr, eps, None)

        if abs(alpha - 1.0) < 1e-12:
            return np.sum(np.log(gk_arr), axis=1)
        if abs(alpha) < 1e-12:
            return np.sum(gk_arr, axis=1)
        return np.sum(np.power(gk_arr, 1.0 - alpha) / (1.0 - alpha), axis=1)

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
        skip_regret: bool = False,
    ) -> Dict[str, Any]:
        if need_grads:
            raise ValueError(
                "MultiDimKnapsackTask does not provide analytic decision gradients. "
                "Use training.decision_grad_backend='finite_diff'."
            )

        # LP: raw predictions are valid benefits (can be any real number).
        # Alpha-fair: softplus ensures positive benefits for power objectives.
        if self.scenario == "lp":
            pred_pos = np.asarray(raw_pred, dtype=float)
            pred_pos_grad = np.ones_like(pred_pos)
        else:
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

        if skip_regret:
            loss_dec = 0.0
            loss_dec_norm = 0.0
            loss_dec_norm_true = 0.0
            solver_calls = 0
            decision_ms = 0.0
            dec_fair = {}
        else:
            loss_dec, loss_dec_norm, loss_dec_norm_true, solver_calls, decision_ms = self._decision_regret(
                pred_pos,
                np.asarray(true, dtype=float),
            )
            # Decision-level fairness metrics (evaluation only)
            d_pred = self._solve_batch(pred_pos)
            dec_fair = self._decision_fairness_metrics(d_pred, np.asarray(true, dtype=float))

        result = {
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
        result.update(dec_fair)
        return result

    def solve_decision(self, pred: np.ndarray, **ctx: Any) -> np.ndarray:
        pred_2d = np.atleast_2d(np.asarray(pred, dtype=float))
        if self.scenario == "lp":
            return self._solve_batch(pred_2d)
        pred_pos, _ = softplus_with_grad(pred_2d)
        pred_pos = pred_pos + 1e-5
        return self._solve_batch(pred_pos)

    def solve_oracle_decision(self, true: np.ndarray, **ctx: Any) -> np.ndarray:
        true_2d = np.atleast_2d(np.asarray(true, dtype=float))
        return self._solve_batch(np.clip(true_2d, 1e-8, None))

    def evaluate_objective(self, decision: np.ndarray, true: np.ndarray, **ctx: Any) -> float:
        return float(np.mean(self._objective_batch(decision, true)))

    def _decision_fairness_metrics(
        self, decision: np.ndarray, true_benefit: np.ndarray,
    ) -> Dict[str, float]:
        """Decision-level fairness metrics (evaluation only, not for training)."""
        groups = self._current_groups
        g0 = groups == 0
        g1 = groups == 1
        if g0.sum() == 0 or g1.sum() == 0:
            return {"decision_alloc_gap": 0.0, "decision_selection_gap": 0.0,
                    "decision_welfare_gap": 0.0}

        d2 = np.atleast_2d(decision)
        t2 = np.atleast_2d(true_benefit)

        # 1. Allocation gap: |mean(d[group0]) - mean(d[group1])|
        alloc_gap = abs(float(d2[:, g0].mean()) - float(d2[:, g1].mean()))

        # 2. Selection rate gap (d_i > 0.5 counts as "selected")
        sel = d2 > 0.5
        sel_gap = abs(float(sel[:, g0].mean()) - float(sel[:, g1].mean()))

        # 3. Welfare gap: |mean(r*d per group0) - mean(r*d per group1)|
        welfare = t2 * d2
        welfare_gap = abs(float(welfare[:, g0].mean()) - float(welfare[:, g1].mean()))

        return {
            "decision_alloc_gap": alloc_gap,
            "decision_selection_gap": sel_gap,
            "decision_welfare_gap": welfare_gap,
        }

    def supported_gradient_strategies(self) -> List[str]:
        if self.scenario == "lp":
            return ["finite_diff", "spsa", "spo_plus"]
        return ["finite_diff", "spsa"]
