"""Finite-difference gradient checks for all loss functions in losses.py."""

import numpy as np
import pytest

from fair_dfl.losses import (
    group_fairness_loss_and_grad,
    group_mse_atkinson_loss_and_grad,
    group_mse_gap_loss_and_grad,
    group_mse_generalized_entropy_loss_and_grad,
    group_mse_mad_loss_and_grad,
    mse_loss_and_grad,
    softplus_with_grad,
)


def _fd_grad(loss_fn, pred, eps=1e-5):
    """Compute finite-difference gradient of a scalar loss w.r.t. pred."""
    grad = np.zeros_like(pred)
    for i in range(pred.size):
        p_plus = pred.copy().ravel()
        p_plus[i] += eps
        p_minus = pred.copy().ravel()
        p_minus[i] -= eps
        loss_plus = loss_fn(p_plus.reshape(pred.shape))
        loss_minus = loss_fn(p_minus.reshape(pred.shape))
        grad.ravel()[i] = (loss_plus - loss_minus) / (2 * eps)
    return grad


class TestSoftplusWithGrad:
    def test_basic_values(self):
        z = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        value, sigmoid = softplus_with_grad(z)
        expected_value = np.log(1.0 + np.exp(z))
        np.testing.assert_allclose(value, expected_value, atol=1e-6)

    def test_sigmoid_range(self):
        z = np.array([-1000.0, -100.0, -1.0, 0.0, 1.0, 100.0, 1000.0])
        _, sigmoid = softplus_with_grad(z)
        assert np.all(sigmoid >= 0.0)
        assert np.all(sigmoid <= 1.0)
        assert not np.any(np.isnan(sigmoid))
        assert not np.any(np.isinf(sigmoid))

    def test_numerical_stability_extreme_values(self):
        z = np.array([-500.0, 500.0, -1000.0, 1000.0])
        value, sigmoid = softplus_with_grad(z)
        assert not np.any(np.isnan(value))
        assert not np.any(np.isinf(value))
        assert not np.any(np.isnan(sigmoid))
        assert not np.any(np.isinf(sigmoid))

    def test_gradient_is_sigmoid(self):
        z = np.linspace(-5, 5, 50)
        value, sigmoid = softplus_with_grad(z)
        fd = np.zeros_like(z)
        eps = 1e-5
        for i in range(len(z)):
            zp = z.copy(); zp[i] += eps
            zm = z.copy(); zm[i] -= eps
            fd[i] = (softplus_with_grad(zp)[0][i] - softplus_with_grad(zm)[0][i]) / (2 * eps)
        np.testing.assert_allclose(sigmoid, fd, atol=1e-4)


class TestMSELossAndGrad:
    def test_zero_loss_when_equal(self):
        x = np.random.randn(10, 3)
        loss, grad = mse_loss_and_grad(x, x)
        assert loss == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(grad, 0.0, atol=1e-12)

    def test_fd_check(self):
        rng = np.random.default_rng(42)
        pred = rng.standard_normal((8, 4))
        true = rng.standard_normal((8, 4))
        loss, grad = mse_loss_and_grad(pred, true)
        fd = _fd_grad(lambda p: mse_loss_and_grad(p, true)[0], pred)
        np.testing.assert_allclose(grad, fd, atol=1e-4)


class TestGroupMSEMADLossAndGrad:
    def test_single_group_returns_zero(self):
        pred = np.random.randn(5, 10)
        true = np.random.randn(5, 10)
        groups = np.zeros(10, dtype=int)
        loss, grad = group_mse_mad_loss_and_grad(pred, true, groups)
        # With one group, MAD should be ~sqrt(smoothing) since gap=0
        assert loss < 0.01

    def test_fd_check(self):
        rng = np.random.default_rng(123)
        pred = rng.standard_normal((4, 6))
        true = rng.standard_normal((4, 6))
        groups = np.array([0, 0, 0, 1, 1, 1])
        loss, grad = group_mse_mad_loss_and_grad(pred, true, groups)
        fd = _fd_grad(lambda p: group_mse_mad_loss_and_grad(p, true, groups)[0], pred)
        np.testing.assert_allclose(grad, fd, atol=1e-4)

    def test_fd_check_three_groups(self):
        rng = np.random.default_rng(456)
        pred = rng.standard_normal((3, 9))
        true = rng.standard_normal((3, 9))
        groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        loss, grad = group_mse_mad_loss_and_grad(pred, true, groups)
        fd = _fd_grad(lambda p: group_mse_mad_loss_and_grad(p, true, groups)[0], pred)
        np.testing.assert_allclose(grad, fd, atol=1e-4)


class TestGroupMSEGapLossAndGrad:
    def test_single_group_returns_zero(self):
        pred = np.random.randn(5, 10)
        true = np.random.randn(5, 10)
        groups = np.zeros(10, dtype=int)
        loss, grad = group_mse_gap_loss_and_grad(pred, true, groups)
        assert loss == 0.0
        np.testing.assert_allclose(grad, 0.0)

    def test_fd_check_two_groups(self):
        rng = np.random.default_rng(789)
        pred = rng.standard_normal((4, 8))
        true = rng.standard_normal((4, 8))
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        loss, grad = group_mse_gap_loss_and_grad(pred, true, groups)
        fd = _fd_grad(lambda p: group_mse_gap_loss_and_grad(p, true, groups)[0], pred)
        np.testing.assert_allclose(grad, fd, atol=1e-4)

    def test_fd_check_three_groups(self):
        rng = np.random.default_rng(101)
        pred = rng.standard_normal((3, 9))
        true = rng.standard_normal((3, 9))
        groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        loss, grad = group_mse_gap_loss_and_grad(pred, true, groups)
        fd = _fd_grad(lambda p: group_mse_gap_loss_and_grad(p, true, groups)[0], pred)
        np.testing.assert_allclose(grad, fd, atol=1e-4)


class TestGroupMSEGELossAndGrad:
    def test_fd_check_alpha_2(self):
        rng = np.random.default_rng(202)
        pred = rng.standard_normal((4, 6)) + 3.0  # offset to avoid near-zero MSE
        true = rng.standard_normal((4, 6))
        groups = np.array([0, 0, 0, 1, 1, 1])
        loss, grad = group_mse_generalized_entropy_loss_and_grad(pred, true, groups, alpha=2.0)
        fd = _fd_grad(
            lambda p: group_mse_generalized_entropy_loss_and_grad(p, true, groups, alpha=2.0)[0],
            pred,
        )
        np.testing.assert_allclose(grad, fd, atol=1e-4)

    def test_fd_check_alpha_1(self):
        rng = np.random.default_rng(303)
        pred = rng.standard_normal((4, 6)) + 3.0
        true = rng.standard_normal((4, 6))
        groups = np.array([0, 0, 0, 1, 1, 1])
        loss, grad = group_mse_generalized_entropy_loss_and_grad(pred, true, groups, alpha=1.0)
        fd = _fd_grad(
            lambda p: group_mse_generalized_entropy_loss_and_grad(p, true, groups, alpha=1.0)[0],
            pred,
        )
        np.testing.assert_allclose(grad, fd, atol=1e-4)


class TestGroupMSEAtkinsonLossAndGrad:
    def test_single_group_returns_zero(self):
        pred = np.random.randn(5, 10) + 5.0
        true = np.random.randn(5, 10)
        groups = np.zeros(10, dtype=int)
        loss, grad = group_mse_atkinson_loss_and_grad(pred, true, groups)
        assert loss == 0.0

    def test_fd_check_epsilon_half(self):
        rng = np.random.default_rng(404)
        pred = rng.standard_normal((4, 6)) + 3.0
        true = rng.standard_normal((4, 6))
        groups = np.array([0, 0, 0, 1, 1, 1])
        loss, grad = group_mse_atkinson_loss_and_grad(pred, true, groups, epsilon=0.5)
        fd = _fd_grad(
            lambda p: group_mse_atkinson_loss_and_grad(p, true, groups, epsilon=0.5)[0],
            pred,
        )
        np.testing.assert_allclose(grad, fd, atol=1e-3)

    def test_fd_check_epsilon_one(self):
        rng = np.random.default_rng(505)
        pred = rng.standard_normal((4, 6)) + 3.0
        true = rng.standard_normal((4, 6))
        groups = np.array([0, 0, 0, 1, 1, 1])
        loss, grad = group_mse_atkinson_loss_and_grad(pred, true, groups, epsilon=1.0)
        fd = _fd_grad(
            lambda p: group_mse_atkinson_loss_and_grad(p, true, groups, epsilon=1.0)[0],
            pred,
        )
        np.testing.assert_allclose(grad, fd, atol=1e-3)


class TestGroupFairnessDispatcher:
    def test_mad_dispatches(self):
        rng = np.random.default_rng(1)
        pred = rng.standard_normal((3, 6))
        true = rng.standard_normal((3, 6))
        groups = np.array([0, 0, 0, 1, 1, 1])
        l1, g1 = group_fairness_loss_and_grad(pred, true, groups, fairness_type="mad")
        l2, g2 = group_mse_mad_loss_and_grad(pred, true, groups)
        assert l1 == pytest.approx(l2)
        np.testing.assert_allclose(g1, g2)

    def test_gap_dispatches(self):
        rng = np.random.default_rng(2)
        pred = rng.standard_normal((3, 6))
        true = rng.standard_normal((3, 6))
        groups = np.array([0, 0, 0, 1, 1, 1])
        l1, g1 = group_fairness_loss_and_grad(pred, true, groups, fairness_type="gap")
        l2, g2 = group_mse_gap_loss_and_grad(pred, true, groups)
        assert l1 == pytest.approx(l2)
        np.testing.assert_allclose(g1, g2)

    def test_atkinson_dispatches(self):
        rng = np.random.default_rng(3)
        pred = rng.standard_normal((3, 6)) + 3.0
        true = rng.standard_normal((3, 6))
        groups = np.array([0, 0, 0, 1, 1, 1])
        l1, g1 = group_fairness_loss_and_grad(pred, true, groups, fairness_type="atkinson")
        l2, g2 = group_mse_atkinson_loss_and_grad(pred, true, groups)
        assert l1 == pytest.approx(l2)
        np.testing.assert_allclose(g1, g2)

    def test_unknown_raises(self):
        pred = np.zeros((3, 4))
        true = np.zeros((3, 4))
        groups = np.array([0, 0, 1, 1])
        with pytest.raises(ValueError, match="Unknown fairness_type"):
            group_fairness_loss_and_grad(pred, true, groups, fairness_type="bogus")
