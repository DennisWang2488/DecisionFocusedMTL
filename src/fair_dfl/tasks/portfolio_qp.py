"""Task layer for data-driven optimization experiments used by the 8 active methods."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List

import numpy as np

from ..losses import group_fairness_loss_and_grad, mse_loss_and_grad
from .base import BaseTask, SplitData, TaskData, add_bias_column


@dataclass
class PortfolioQPTask(BaseTask):
    n_samples_train: int
    n_samples_val: int
    n_samples_test: int
    n_features: int
    n_assets: int
    risk_aversion: float
    group_bias: float
    noise_std: float
    fairness_type: str = "mad"
    fairness_ge_alpha: float = 2.0

    def __post_init__(self) -> None:
        self.name = "portfolio_qp"
        self.n_outputs = self.n_assets

    def generate_data(self, seed: int) -> TaskData:
        rng = np.random.default_rng(seed)
        groups = np.zeros(self.n_assets, dtype=int)
        groups[self.n_assets // 2 :] = 1

        a = rng.normal(size=(self.n_assets, self.n_assets))
        sigma = (a.T @ a) / float(self.n_assets) + 0.2 * np.eye(self.n_assets)
        true_w = rng.normal(loc=0.0, scale=0.5, size=(self.n_features + 1, self.n_assets))
        group_shift = np.where(groups == 0, -self.group_bias, self.group_bias)

        def sample(n_rows: int) -> SplitData:
            x = rng.normal(size=(n_rows, self.n_features))
            raw = add_bias_column(x) @ true_w
            y = raw + group_shift[None, :] + rng.normal(scale=self.noise_std, size=raw.shape)
            return SplitData(x=x, y=y)

        train = sample(n_rows=self.n_samples_train)
        val = sample(n_rows=self.n_samples_val)
        test = sample(n_rows=self.n_samples_test)
        return TaskData(
            train=train,
            val=val,
            test=test,
            groups=groups,
            meta={"sigma": sigma},
        )

    def _prepare_kkt(self, sigma: np.ndarray) -> None:
        ridge = 1e-4
        q = self.risk_aversion * sigma + ridge * np.eye(sigma.shape[0])
        q_inv = np.linalg.inv(q)
        ones = np.ones(q.shape[0])
        v = q_inv @ ones
        b = float(ones @ v)
        jac = q_inv - np.outer(v, v) / b
        self._kkt_cache = {"q": q, "q_inv": q_inv, "ones": ones, "v": v, "b": b, "jac": jac}

    def _solve_weights(self, mu: np.ndarray) -> np.ndarray:
        q_inv = self._kkt_cache["q_inv"]
        ones = self._kkt_cache["ones"]
        v = self._kkt_cache["v"]
        b = self._kkt_cache["b"]
        nu = (mu @ v - 1.0) / b
        return (mu - nu[:, None] * ones[None, :]) @ q_inv.T

    def _objective(self, w: np.ndarray, mu_true: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        ret = np.sum(mu_true * w, axis=1)
        quad = np.einsum("bi,ij,bj->b", w, sigma, w)
        return ret - 0.5 * self.risk_aversion * quad

    def _decision_regret_and_grad(
        self,
        mu_pred: np.ndarray,
        mu_true: np.ndarray,
        sigma: np.ndarray,
        need_grads: bool,
    ) -> tuple[float, float, float, np.ndarray, int]:
        w_true = self._solve_weights(mu_true)
        w_pred = self._solve_weights(mu_pred)
        obj_true = self._objective(w_true, mu_true, sigma)
        obj_pred = self._objective(w_pred, mu_true, sigma)
        # obj_true should dominate, but clip for numerical robustness.
        regret = np.maximum(obj_true - obj_pred, 0.0)
        loss_dec = float(np.mean(regret))
        denom_bounded = np.abs(obj_true) + np.abs(obj_pred) + 1e-8
        loss_dec_normalized = float(np.mean(regret / denom_bounded))
        loss_dec_normalized_true = float(np.mean(regret / (np.abs(obj_true) + 1e-8)))

        grad_mu = np.zeros_like(mu_pred)
        if need_grads:
            jac = self._kkt_cache["jac"]
            grad_w = -(mu_true - self.risk_aversion * (w_pred @ sigma))
            grad_mu = (grad_w @ jac.T) / float(mu_pred.shape[0])
        solver_calls = int(2 * mu_pred.shape[0])
        return loss_dec, loss_dec_normalized, loss_dec_normalized_true, grad_mu, solver_calls

    def compute(
        self,
        raw_pred: np.ndarray,
        true: np.ndarray,
        need_grads: bool,
        fairness_smoothing: float = 1e-6,
    ) -> Dict[str, np.ndarray | float]:
        sigma = self._current_sigma
        loss_pred, grad_pred = mse_loss_and_grad(raw_pred, true)
        loss_fair, grad_fair = group_fairness_loss_and_grad(
            raw_pred,
            true,
            self._current_groups,
            fairness_type=self.fairness_type,
            smoothing=fairness_smoothing,
            ge_alpha=self.fairness_ge_alpha,
        )

        t0 = perf_counter()
        loss_dec, loss_dec_normalized, loss_dec_normalized_true, grad_dec, solver_calls = self._decision_regret_and_grad(
            mu_pred=raw_pred,
            mu_true=true,
            sigma=sigma,
            need_grads=need_grads,
        )
        decision_ms = (perf_counter() - t0) * 1000.0

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

    # ------------------------------------------------------------------
    # Generic decision gradient interface
    # ------------------------------------------------------------------

    def solve_decision(self, pred: np.ndarray, **ctx: Any) -> np.ndarray:
        pred_2d = np.atleast_2d(pred)
        return self._solve_weights(pred_2d)

    def evaluate_objective(self, decision: np.ndarray, true: np.ndarray, **ctx: Any) -> float:
        if not hasattr(self, "_current_sigma"):
            raise RuntimeError("Call bind_context() first.")
        decision_2d = np.atleast_2d(decision)
        true_2d = np.atleast_2d(true)
        return float(np.mean(self._objective(decision_2d, true_2d, self._current_sigma)))

    def supported_gradient_strategies(self) -> List[str]:
        return ["analytic", "finite_diff"]

    def bind_context(self, groups: np.ndarray, sigma: np.ndarray) -> None:
        self._current_groups = groups
        self._current_sigma = sigma
        self._prepare_kkt(sigma=sigma)
