"""Tests for multi-objective gradient handlers."""

import numpy as np
import pytest

from fair_dfl.algorithms.mo_handler import (
    AlignMOHandler,
    CAGradHandler,
    MGDAHandler,
    PCGradHandler,
    WeightedSumHandler,
)
from fair_dfl.metrics import cosine


class TestWeightedSumHandler:
    def test_equal_weights(self):
        handler = WeightedSumHandler(weights={})
        g1 = np.array([1.0, 0.0])
        g2 = np.array([0.0, 1.0])
        direction = handler.compute_direction(
            grads={"a": g1, "b": g2}, losses={"a": 1.0, "b": 1.0}, step=0,
        )
        np.testing.assert_allclose(direction, [0.5, 0.5])

    def test_custom_weights(self):
        handler = WeightedSumHandler(weights={"a": 3.0, "b": 1.0})
        g1 = np.array([1.0, 0.0])
        g2 = np.array([0.0, 1.0])
        direction = handler.compute_direction(
            grads={"a": g1, "b": g2}, losses={"a": 1.0, "b": 1.0}, step=0,
        )
        np.testing.assert_allclose(direction, [0.75, 0.25])

    def test_extra_logs_populated(self):
        handler = WeightedSumHandler(weights={})
        g1 = np.array([1.0, 0.0])
        g2 = np.array([0.0, 1.0])
        handler.compute_direction(
            grads={"a": g1, "b": g2}, losses={"a": 1.0, "b": 1.0}, step=0,
        )
        logs = handler.extra_logs()
        assert "mo_grad_norm_a" in logs
        assert "mo_cos_a_b" in logs
        assert "stationarity_proxy" in logs


class TestPCGradHandler:
    def test_no_conflict_unchanged(self):
        handler = PCGradHandler()
        g1 = np.array([1.0, 0.0])
        g2 = np.array([0.0, 1.0])  # orthogonal, no conflict
        direction = handler.compute_direction(
            grads={"a": g1, "b": g2}, losses={"a": 1.0, "b": 1.0}, step=0,
        )
        np.testing.assert_allclose(direction, [1.0, 1.0])

    def test_conflicting_projection(self):
        handler = PCGradHandler()
        g1 = np.array([1.0, 0.0])
        g2 = np.array([-1.0, 0.0])  # directly conflicting
        direction = handler.compute_direction(
            grads={"a": g1, "b": g2}, losses={"a": 1.0, "b": 1.0}, step=0,
        )
        # After projection, both should lose their conflicting component
        assert abs(direction[0]) < 0.01

    def test_output_finite(self):
        handler = PCGradHandler()
        rng = np.random.default_rng(42)
        grads = {f"obj_{i}": rng.standard_normal(20) for i in range(3)}
        losses = {f"obj_{i}": rng.random() for i in range(3)}
        direction = handler.compute_direction(grads=grads, losses=losses, step=0)
        assert not np.any(np.isnan(direction))
        assert not np.any(np.isinf(direction))


class TestMGDAHandler:
    def test_aligned_grads(self):
        handler = MGDAHandler()
        g1 = np.array([1.0, 0.0])
        g2 = np.array([1.0, 1.0])
        direction = handler.compute_direction(
            grads={"a": g1, "b": g2}, losses={"a": 1.0, "b": 1.0}, step=0,
        )
        # Direction should roughly align with both
        assert cosine(direction, g1) > -0.1
        assert cosine(direction, g2) > -0.1

    def test_output_finite(self):
        handler = MGDAHandler()
        rng = np.random.default_rng(99)
        grads = {f"obj_{i}": rng.standard_normal(20) for i in range(3)}
        losses = {f"obj_{i}": rng.random() for i in range(3)}
        direction = handler.compute_direction(grads=grads, losses=losses, step=0)
        assert not np.any(np.isnan(direction))


class TestCAGradHandler:
    def test_output_finite(self):
        handler = CAGradHandler(c=0.5)
        rng = np.random.default_rng(77)
        grads = {f"obj_{i}": rng.standard_normal(20) for i in range(3)}
        losses = {f"obj_{i}": rng.random() for i in range(3)}
        direction = handler.compute_direction(grads=grads, losses=losses, step=0)
        assert not np.any(np.isnan(direction))
        assert not np.any(np.isinf(direction))

    def test_extra_logs(self):
        handler = CAGradHandler(c=0.5)
        rng = np.random.default_rng(88)
        grads = {f"obj_{i}": rng.standard_normal(10) for i in range(2)}
        losses = {f"obj_{i}": rng.random() for i in range(2)}
        handler.compute_direction(grads=grads, losses=losses, step=0)
        logs = handler.extra_logs()
        assert "stationarity_proxy" in logs


class TestAlignMOHandler:
    """4-mode / 2-binary-decision framework, cold start, numerics, state."""

    _NAMES = ("decision_regret", "pred_loss", "pred_fairness")

    def _grads_aligned(self, dim: int = 4):
        g = np.ones(dim)
        return {n: g.copy() for n in self._NAMES}

    def _grads_conflicting(self, dim: int = 4):
        g = np.ones(dim)
        return {
            "decision_regret": g.copy(),
            "pred_loss": -g.copy(),
            "pred_fairness": g.copy(),
        }

    def _grads_scale_imbalanced(self, dim: int = 4):
        base = np.ones(dim)
        return {
            "decision_regret": 1000.0 * base,
            "pred_loss": 1.0 * base,
            "pred_fairness": 1.0 * base,
        }

    def _grads_scale_imbalanced_conflict(self, dim: int = 4):
        """Imbalanced AND pairwise-conflicting."""
        base = np.ones(dim)
        return {
            "decision_regret": 1000.0 * base,
            "pred_loss": -1.0 * base,
            "pred_fairness": base.copy(),
        }

    def _losses(self):
        return {n: 1.0 for n in self._NAMES}

    # --- original 6 cases (refreshed) -------------------------------------

    def test_warmup_falls_back_to_scalarized(self):
        h = AlignMOHandler(T_warmup=10)
        h.set_step_context(mu=1.0, lam=1.0)
        h.compute_direction(self._grads_conflicting(), self._losses(), step=0)
        assert h.extra_logs()["mode_this_step"] == "scalarized"

    def test_aligned_gradients_pick_scalarized(self):
        h = AlignMOHandler(T_warmup=0, beta_ema=0.0)
        h.set_step_context(mu=1.0, lam=1.0)
        h.compute_direction(self._grads_aligned(), self._losses(), step=20)
        assert h.extra_logs()["mode_this_step"] == "scalarized"

    def test_conflicting_balanced_pick_projected(self):
        h = AlignMOHandler(T_warmup=0, beta_ema=0.0, tau_conflict=-0.1)
        h.set_step_context(mu=1.0, lam=1.0)
        h.compute_direction(self._grads_conflicting(), self._losses(), step=20)
        assert h.extra_logs()["mode_this_step"] == "projected"

    def test_imbalanced_compatible_picks_anchored(self):
        h = AlignMOHandler(T_warmup=0, beta_ema=0.0, tau_scale=2.0)
        h.set_step_context(mu=0.0, lam=1.0)  # forces mu_floor
        h.compute_direction(self._grads_scale_imbalanced(), self._losses(), step=20)
        logs = h.extra_logs()
        assert logs["mode_this_step"] == "anchored"
        assert logs["mu_eff_used"] >= 0.1 - 1e-9

    def test_nan_gradient_produces_finite_direction(self):
        h = AlignMOHandler(T_warmup=0)
        g = np.array([np.nan, np.inf, 1.0, -np.inf])
        grads = {n: g.copy() for n in self._NAMES}
        h.set_step_context(mu=1.0, lam=1.0)
        d = h.compute_direction(grads, self._losses(), step=20)
        assert np.all(np.isfinite(d))

    def test_ema_persistence_and_mode_switch_count(self):
        h = AlignMOHandler(T_warmup=0, beta_ema=0.5)
        h.set_step_context(mu=1.0, lam=1.0)
        h.compute_direction(self._grads_aligned(), self._losses(), step=20)
        assert h._ema_c_dp is not None
        prev_switches = h.extra_logs()["n_mode_switches_so_far"]
        h.compute_direction(self._grads_conflicting(), self._losses(), step=21)
        h.compute_direction(self._grads_conflicting(), self._losses(), step=22)
        h.compute_direction(self._grads_aligned(), self._losses(), step=23)
        assert h.extra_logs()["n_mode_switches_so_far"] > prev_switches

    def test_extra_logs_contain_contract_keys(self):
        h = AlignMOHandler()
        h.set_step_context(mu=1.0, lam=1.0)
        h.compute_direction(self._grads_aligned(), self._losses(), step=0)
        logs = h.extra_logs()
        for k in ["mode_this_step", "regime_scale", "regime_direction",
                  "regime_scale_this_step",
                  "c_dp", "c_df", "c_pf", "r_dp", "r_df",
                  "mu_eff_used", "post_scale_used",
                  "n_projections", "n_mode_switches_so_far"]:
            assert k in logs, f"missing key: {k}"

    # --- 2.3c: six new cases covering the 4-mode framework ---------------

    def test_balanced_compatible_no_projection_no_normalization(self):
        """Case (i): scalarized. mu=lam=1, aligned, balanced scales."""
        h = AlignMOHandler(T_warmup=0, beta_ema=0.0)
        h.set_step_context(mu=1.0, lam=1.0)
        grads = self._grads_aligned()
        d = h.compute_direction(grads, self._losses(), step=20)
        logs = h.extra_logs()
        assert logs["mode_this_step"] == "scalarized"
        assert logs["n_projections"] == 0.0
        assert logs["post_scale_used"] == 1.0
        expected = grads["decision_regret"] + grads["pred_loss"] + grads["pred_fairness"]
        np.testing.assert_allclose(d, expected)

    def test_balanced_conflict_direction_equals_pcgrad_on_raw(self):
        """Case (ii): projected. direction matches PCGrad(weighted raw grads)."""
        h = AlignMOHandler(T_warmup=0, beta_ema=0.0, tau_conflict=-0.1)
        mu, lam = 0.7, 1.3
        h.set_step_context(mu=mu, lam=lam)
        grads = self._grads_conflicting()
        d = h.compute_direction(grads, self._losses(), step=20)
        assert h.extra_logs()["mode_this_step"] == "projected"
        # Reference: PCGrad on the same weighted raw grads.
        weighted = {
            "decision_regret": grads["decision_regret"].copy(),
            "pred_loss": mu * grads["pred_loss"],
            "pred_fairness": lam * grads["pred_fairness"],
        }
        ref = PCGradHandler().compute_direction(weighted, self._losses(), step=20)
        np.testing.assert_allclose(d, ref, atol=1e-10)

    def test_imbalanced_compatible_unit_scaled_by_mean_norm(self):
        """Case (iii): anchored. direction = mean_norm * (u_dec + mu_eff*u_pred + lam*u_fair)."""
        h = AlignMOHandler(T_warmup=0, beta_ema=0.0, tau_scale=2.0)
        mu, lam = 0.0, 1.0  # mu=0 triggers mu_floor
        h.set_step_context(mu=mu, lam=lam)
        grads = self._grads_scale_imbalanced()
        d = h.compute_direction(grads, self._losses(), step=20)
        logs = h.extra_logs()
        assert logs["mode_this_step"] == "anchored"
        norms = [float(np.linalg.norm(grads[n])) for n in self._NAMES]
        mean_norm = float(np.mean(norms))
        mu_eff = max(mu, 0.1)
        u = {n: grads[n] / np.linalg.norm(grads[n]) for n in self._NAMES}
        expected = mean_norm * (u["decision_regret"] + mu_eff * u["pred_loss"] + lam * u["pred_fairness"])
        np.testing.assert_allclose(d, expected, atol=1e-10)
        assert abs(logs["post_scale_used"] - mean_norm) < 1e-10

    def test_imbalanced_conflict_pcgrad_on_unit_gradients_scaled(self):
        """Case (iv): anchored_projected — the case the 3-mode handler mis-routed.

        direction must equal mean_norm * PCGrad(unit-norm weighted grads).
        """
        h = AlignMOHandler(T_warmup=0, beta_ema=0.0,
                           tau_scale=2.0, tau_conflict=-0.1)
        mu, lam = 0.0, 1.0
        h.set_step_context(mu=mu, lam=lam)
        grads = self._grads_scale_imbalanced_conflict()
        d = h.compute_direction(grads, self._losses(), step=20)
        logs = h.extra_logs()
        assert logs["mode_this_step"] == "anchored_projected"
        assert logs["regime_scale"] == "imbalanced"
        assert logs["regime_direction"] == "conflict"
        # Reference: normalize, weight, then PCGrad, then scale.
        norms = [float(np.linalg.norm(grads[n])) for n in self._NAMES]
        mean_norm = float(np.mean(norms))
        mu_eff = max(mu, 0.1)
        u = {n: grads[n] / np.linalg.norm(grads[n]) for n in self._NAMES}
        weighted = {
            "decision_regret": u["decision_regret"],
            "pred_loss": mu_eff * u["pred_loss"],
            "pred_fairness": lam * u["pred_fairness"],
        }
        ref_core = PCGradHandler().compute_direction(weighted, self._losses(), step=20)
        np.testing.assert_allclose(d, mean_norm * ref_core, atol=1e-9)

    def test_mode_switch_logging_across_regimes(self):
        """Case (v): traverse cells and confirm all 4 diagnostic fields."""
        h = AlignMOHandler(T_warmup=0, beta_ema=0.0,
                           tau_scale=2.0, tau_conflict=-0.1)
        h.set_step_context(mu=1.0, lam=1.0)
        # (balanced, conflict) -> projected
        h.compute_direction(self._grads_conflicting(), self._losses(), step=20)
        l1 = dict(h.extra_logs())
        assert l1["mode_this_step"] == "projected"
        assert l1["regime_scale"] == "balanced"
        assert l1["regime_direction"] == "conflict"
        # (imbalanced, conflict) -> anchored_projected
        h.compute_direction(self._grads_scale_imbalanced_conflict(), self._losses(), step=21)
        l2 = dict(h.extra_logs())
        assert l2["mode_this_step"] == "anchored_projected"
        # (balanced, compatible) -> scalarized
        h.compute_direction(self._grads_aligned(), self._losses(), step=22)
        l3 = dict(h.extra_logs())
        assert l3["mode_this_step"] == "scalarized"
        assert l3["regime_scale"] == "balanced"
        assert l3["regime_direction"] == "compatible"
        # At least two mode switches over three steps.
        assert l3["n_mode_switches_so_far"] >= 2

    def test_tau_align_kwarg_accepted_for_backward_compat(self):
        """Case (vi): passing the deprecated tau_align kwarg must not raise."""
        h = AlignMOHandler(tau_align=0.1, tau_conflict=-0.1, tau_scale=2.0)
        h.set_step_context(mu=1.0, lam=1.0)
        d = h.compute_direction(self._grads_aligned(), self._losses(), step=0)
        assert np.all(np.isfinite(d))
        # tau_align no longer appears in diagnostics.
        assert "mo_alignmo_tau_align" not in h.extra_logs()
