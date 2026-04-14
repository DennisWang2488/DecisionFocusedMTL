"""Gradient and solver correctness tests for MedicalResourceAllocationTask."""

import numpy as np
import pytest

from fair_dfl.tasks.medical_resource_allocation import MedicalResourceAllocationTask


def _make_task(**overrides):
    """Create a minimal MedicalResourceAllocationTask for testing."""
    defaults = dict(
        data_csv="data/data_processed.csv",
        n_sample=200,
        data_seed=42,
        split_seed=2,
        test_fraction=0.3,
        val_fraction=0.2,
        alpha_fair=2.0,
        budget=500.0,
        decision_mode="group",
        fairness_type="mad",
    )
    defaults.update(overrides)
    return MedicalResourceAllocationTask(**defaults)


@pytest.fixture(scope="module")
def task_and_data():
    task = _make_task()
    data = task.generate_data(seed=42)
    return task, data


class TestSolverFeasibility:
    def test_group_solver_nonneg(self, task_and_data):
        task, _ = task_and_data
        s = task._splits["train"]
        pred = np.clip(s.y, 1e-6, None)
        d = task._solve_group(pred, s.cost, s.race, budget=task.budget, alpha=task.alpha_fair)
        assert np.all(d >= -1e-10), "Allocations should be non-negative"

    def test_group_solver_budget(self, task_and_data):
        task, _ = task_and_data
        s = task._splits["train"]
        pred = np.clip(s.y, 1e-6, None)
        d = task._solve_group(pred, s.cost, s.race, budget=task.budget, alpha=task.alpha_fair)
        total_cost = float(np.sum(d * s.cost))
        assert total_cost == pytest.approx(task.budget, rel=0.05), (
            f"Total cost {total_cost:.2f} should be close to budget {task.budget:.2f}"
        )

    def test_individual_solver_nonneg(self, task_and_data):
        task, _ = task_and_data
        s = task._splits["train"]
        pred = np.clip(s.y, 1e-6, None)
        d = task._solve_alpha_fair(pred, s.cost, alpha=task.alpha_fair, budget=task.budget)
        assert np.all(d >= -1e-10)

    def test_individual_solver_budget(self, task_and_data):
        task, _ = task_and_data
        s = task._splits["train"]
        pred = np.clip(s.y, 1e-6, None)
        d = task._solve_alpha_fair(pred, s.cost, alpha=task.alpha_fair, budget=task.budget)
        total_cost = float(np.sum(d * s.cost))
        assert total_cost == pytest.approx(task.budget, rel=0.05)


class TestDecisionRegretGrad:
    def test_regret_nonneg(self, task_and_data):
        task, _ = task_and_data
        s = task._splits["train"]
        pred = np.clip(s.y + np.random.randn(s.y.shape[0]) * 0.5, 1e-6, None)
        out = task.compute_batch(
            raw_pred=pred, true=s.y, cost=s.cost, race=s.race,
            need_grads=True, fairness_smoothing=1e-6,
        )
        assert out["loss_dec"] >= 0.0

    def test_grad_finite(self, task_and_data):
        task, _ = task_and_data
        s = task._splits["train"]
        pred = np.clip(s.y + np.random.randn(s.y.shape[0]) * 0.5, 1e-6, None)
        out = task.compute_batch(
            raw_pred=pred, true=s.y, cost=s.cost, race=s.race,
            need_grads=True, fairness_smoothing=1e-6,
        )
        assert not np.any(np.isnan(out["grad_dec"]))
        assert not np.any(np.isinf(out["grad_dec"]))

    def test_perfect_pred_zero_regret(self, task_and_data):
        task, _ = task_and_data
        s = task._splits["train"]
        out = task.compute_batch(
            raw_pred=s.y.copy(), true=s.y, cost=s.cost, race=s.race,
            need_grads=False, fairness_smoothing=1e-6,
        )
        assert out["loss_dec"] == pytest.approx(0.0, abs=1e-6)


class TestVJPConsistency:
    """Check that the VJP gives the same result as v @ full_Jacobian."""

    @pytest.mark.slow
    def test_vjp_matches_jacobian(self, task_and_data):
        task, _ = task_and_data
        s = task._splits["train"]
        # Use a small subset to make the n x n Jacobian tractable
        n = min(50, s.y.shape[0])
        pred = np.clip(s.y[:n], 1e-6, None)
        cost = s.cost[:n]
        race = s.race[:n]
        v = np.random.randn(n)

        vjp_result = task._solve_group_vjp(v, pred, cost, race, budget=task.budget, alpha=task.alpha_fair)
        jac = task._solve_group_grad_jacobian(pred, cost, race, budget=task.budget, alpha=task.alpha_fair)
        expected = v @ jac

        np.testing.assert_allclose(vjp_result, expected, atol=1e-6, rtol=1e-4)


class TestFairnessMetrics:
    def test_mad_symmetric(self, task_and_data):
        task, _ = task_and_data
        s = task._splits["train"]
        pred = s.y.copy()
        loss, grad = task._fair_loss_and_grad_mad(pred, s.y, s.race, smoothing=1e-6)
        # Perfect prediction => per-group MSE = 0 => MAD ≈ sqrt(smoothing)
        assert loss < 0.01

    def test_gap_symmetric(self, task_and_data):
        task, _ = task_and_data
        s = task._splits["train"]
        pred = s.y.copy()
        loss, grad = task._fair_loss_and_grad_gap(pred, s.y, s.race, smoothing=1e-6)
        assert loss < 0.01

    def test_atkinson_perfect_pred(self, task_and_data):
        task, _ = task_and_data
        s = task._splits["train"]
        pred = s.y.copy()
        loss, grad = task._fair_loss_and_grad_atkinson(pred, s.y, s.race, smoothing=1e-6)
        # Perfect prediction => all group MSEs equal (= smoothing) => Atkinson = 0
        assert loss == pytest.approx(0.0, abs=1e-6)

    def test_dp_fd_check(self, task_and_data):
        """Finite-difference check on demographic-parity gradient."""
        task, _ = task_and_data
        s = task._splits["train"]
        # Use a small, reproducible perturbation of true labels for stable FD
        rng = np.random.default_rng(2024)
        n = min(60, s.y.shape[0])
        pred = (s.y[:n] + 0.5 * rng.standard_normal(n)).astype(np.float64)
        race = s.race[:n]

        loss, grad = task._fair_loss_and_grad_dp(pred, s.y[:n], race, smoothing=1e-6)
        eps = 1e-5
        fd = np.zeros_like(pred)
        for i in range(n):
            p_plus = pred.copy(); p_plus[i] += eps
            p_minus = pred.copy(); p_minus[i] -= eps
            l_plus, _ = task._fair_loss_and_grad_dp(p_plus, s.y[:n], race, smoothing=1e-6)
            l_minus, _ = task._fair_loss_and_grad_dp(p_minus, s.y[:n], race, smoothing=1e-6)
            fd[i] = (l_plus - l_minus) / (2 * eps)
        np.testing.assert_allclose(grad, fd, atol=1e-5)

    def test_dp_independent_of_true(self, task_and_data):
        """DP fairness only depends on predictions, not labels."""
        task, _ = task_and_data
        s = task._splits["train"]
        pred = (s.y + 0.3 * np.random.default_rng(11).standard_normal(s.y.shape[0])).astype(float)
        true_a = s.y
        true_b = s.y + 100.0  # arbitrary shift
        l_a, g_a = task._fair_loss_and_grad_dp(pred, true_a, s.race, smoothing=1e-6)
        l_b, g_b = task._fair_loss_and_grad_dp(pred, true_b, s.race, smoothing=1e-6)
        assert l_a == pytest.approx(l_b)
        np.testing.assert_allclose(g_a, g_b)

    def test_bias_parity_fd_check(self, task_and_data):
        """Finite-difference check on bias-parity gradient."""
        task, _ = task_and_data
        s = task._splits["train"]
        rng = np.random.default_rng(31415)
        n = min(60, s.y.shape[0])
        pred = (s.y[:n] + 0.5 * rng.standard_normal(n)).astype(np.float64)
        race = s.race[:n]

        loss, grad = task._fair_loss_and_grad_bias_parity(pred, s.y[:n], race, smoothing=1e-6)
        eps = 1e-5
        fd = np.zeros_like(pred)
        for i in range(n):
            p_plus = pred.copy(); p_plus[i] += eps
            p_minus = pred.copy(); p_minus[i] -= eps
            l_plus, _ = task._fair_loss_and_grad_bias_parity(p_plus, s.y[:n], race, smoothing=1e-6)
            l_minus, _ = task._fair_loss_and_grad_bias_parity(p_minus, s.y[:n], race, smoothing=1e-6)
            fd[i] = (l_plus - l_minus) / (2 * eps)
        np.testing.assert_allclose(grad, fd, atol=1e-5)

    def test_bias_parity_zero_when_uniform_bias(self):
        """If both groups have the same per-group mean residual, BP loss ~ 0."""
        task = _make_task()
        task.generate_data(seed=42)
        s = task._splits["train"]
        # Uniform additive bias: pred = y + 0.7 for everyone
        pred = (s.y + 0.7).astype(float)
        loss, _ = task._fair_loss_and_grad_bias_parity(pred, s.y, s.race, smoothing=1e-6)
        assert loss < 1e-2

    def test_bias_parity_distinct_from_dp(self, task_and_data):
        """At perfect prediction, DP can be large (group means differ in truth)
        but bias parity is ~0 (residuals are zero)."""
        task, _ = task_and_data
        s = task._splits["train"]
        pred = s.y.copy()  # perfect predictions
        l_dp, _ = task._fair_loss_and_grad_dp(pred, s.y, s.race, smoothing=1e-6)
        l_bp, _ = task._fair_loss_and_grad_bias_parity(pred, s.y, s.race, smoothing=1e-6)
        # Healthcare benefits do differ across groups in the data, so the per-group
        # mean prediction differs even with perfect prediction => DP > 0.
        # Bias parity depends only on residuals which are all zero => BP ~ 0.
        assert l_bp < 1e-2
        # We don't assert l_dp > 0 strictly (depends on data) but they should diverge.
        if l_dp > 1e-3:
            assert l_dp > l_bp

    def test_fairness_dispatch(self, task_and_data):
        task, _ = task_and_data
        s = task._splits["train"]
        pred = s.y + np.random.randn(s.y.shape[0]) * 0.1
        for ft in ["mad", "gap", "atkinson", "dp", "demographic_parity", "bp", "bias_parity"]:
            task_ft = _make_task(fairness_type=ft)
            task_ft.generate_data(seed=42)
            loss, grad = task_ft._compute_fairness(pred[:len(task_ft._splits["train"].y)],
                                                     task_ft._splits["train"].y,
                                                     task_ft._splits["train"].race, smoothing=1e-6)
            assert np.isfinite(loss)
            assert not np.any(np.isnan(grad))
