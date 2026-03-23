"""Tests for multi-objective gradient handlers."""

import numpy as np
import pytest

from fair_dfl.algorithms.mo_handler import (
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
