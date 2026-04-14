"""Schema and smoke tests for the redesigned MultiDimKnapsackTask."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from experiments.configs import ALL_METHOD_CONFIGS, DEFAULT_TRAIN_CFG
from fair_dfl.runner import _build_task, run_experiment_unified
from fair_dfl.tasks import MultiDimKnapsackTask


def _task_cfg(**overrides) -> dict:
    cfg = {
        "name": "md_knapsack",
        "n_samples_train": 12,
        "n_samples_val": 6,
        "n_samples_test": 8,
        "n_features": 4,
        "n_resources": 2,
        "scenario": "alpha_fair",
        "alpha_fair": 2.0,
        "snr": 5.0,
        "poly_degree": 2,
        "benefit_group_bias": 0.4,
        "cost_group_bias": 0.0,
        "benefit_noise_ratio": 1.0,
        "cost_noise_ratio": 1.0,
        "budget_tightness": 0.5,
        "fairness_type": "mad",
        "data_seed": 7,
        "group_ratio": 0.5,
    }
    cfg.update(overrides)
    return cfg


def _train_cfg(**overrides) -> dict:
    train = copy.deepcopy(DEFAULT_TRAIN_CFG)
    train.update(
        {
            "device": "cpu",
            "decision_grad_backend": "finite_diff",
            "seeds": [11],
            "lambdas": [0.0],
            "steps_per_lambda": 1,
            "log_every": 1,
            "batch_size": -1,
            "eval_train": True,
            "model": {
                "arch": "linear",
                "hidden_dim": 8,
                "n_layers": 1,
                "activation": "relu",
                "dropout": 0.0,
                "batch_norm": False,
                "init_mode": "default",
            },
        }
    )
    train.update(overrides)
    return train


# ----------------------------------------------------------------------
# Schema and data-generation
# ----------------------------------------------------------------------

def test_build_task_md_knapsack_alpha_fair():
    task, data = _build_task(_task_cfg())

    assert isinstance(task, MultiDimKnapsackTask)
    assert task.name == "md_knapsack"
    assert task.n_resources == 2
    # Per-individual layout: x is (n, n_features), y is (n, n_resources).
    assert data.train.x.shape == (12, 4)
    assert data.train.y.shape == (12, 2)
    assert data.val.x.shape == (6, 4)
    assert data.val.y.shape == (6, 2)
    assert data.test.x.shape == (8, 4)
    assert data.test.y.shape == (8, 2)
    # Groups are now per-individual (per-row), not per-item.
    assert data.groups.shape == (12,)


def test_build_task_md_knapsack_lp():
    task, data = _build_task(_task_cfg(scenario="lp", alpha_fair=1.0))
    assert isinstance(task, MultiDimKnapsackTask)
    assert task.scenario == "lp"
    assert data.train.y.shape == (12, 2)


def test_per_split_storage_includes_cost_groups_budgets():
    task, _ = _build_task(_task_cfg())
    for split in ("train", "val", "test"):
        s = task._splits[split]
        assert s.cost.shape == s.y.shape, "cost must be per-individual, per-resource"
        assert s.groups.shape == (s.x.shape[0],)
        assert s.budgets.shape == (task.n_resources,)
        assert np.all(s.cost > 0.0), "costs must be positive"
        assert np.all(s.budgets > 0.0)


def test_cost_is_independent_of_features():
    """Cost is generated independently of features — the prediction task
    only sees benefit. Two task instances with identical seeds but different
    benefit knobs should still produce the same cost vector."""
    task_a, _ = _build_task(_task_cfg(benefit_group_bias=0.0))
    task_b, _ = _build_task(_task_cfg(benefit_group_bias=0.8))
    np.testing.assert_allclose(
        task_a._splits["train"].cost,
        task_b._splits["train"].cost,
    )


def test_snr_decouples_from_benefit_bias():
    """Varying benefit_group_bias should not change the (group-0) noise std
    of the benefit signal — only the per-group mean shift."""
    rng = np.random.default_rng(0)
    diffs = []
    for bias in (0.0, 0.5, 1.0):
        task, _ = _build_task(_task_cfg(benefit_group_bias=bias, n_samples_train=2000))
        s = task._splits["train"]
        # Group-0 std should be roughly the same regardless of bias because
        # the per-group bias is an additive shift, not a noise rescale.
        g0 = s.groups == 0
        diffs.append(s.y[g0, 0].std())
    # All three stds should be within 25% of each other.
    diffs = np.asarray(diffs)
    assert (diffs.max() / diffs.min()) < 1.25


# ----------------------------------------------------------------------
# Group-imbalance knobs are wired through
# ----------------------------------------------------------------------

def test_benefit_group_bias_creates_per_group_mean_gap():
    task, _ = _build_task(_task_cfg(benefit_group_bias=1.5, n_samples_train=400))
    s = task._splits["train"]
    g0 = s.groups == 0
    g1 = s.groups == 1
    mean_g0 = s.y[g0, 0].mean()
    mean_g1 = s.y[g1, 0].mean()
    assert mean_g0 > mean_g1, (
        "+benefit bias on g0 should raise group-0 mean above group-1 mean"
    )


def test_cost_group_bias_creates_per_group_cost_gap_only():
    """Cost imbalance must affect the cost stats but NOT the benefit stats."""
    task_neutral, _ = _build_task(_task_cfg(cost_group_bias=0.0, n_samples_train=400))
    task_biased, _ = _build_task(_task_cfg(cost_group_bias=0.4, n_samples_train=400))
    s_n = task_neutral._splits["train"]
    s_b = task_biased._splits["train"]
    # Benefits unchanged.
    np.testing.assert_allclose(s_n.y, s_b.y)
    # Costs differ in per-group means.
    g0 = s_b.groups == 0
    g1 = s_b.groups == 1
    assert s_b.cost[g0, 0].mean() - s_b.cost[g1, 0].mean() > 0.5


# ----------------------------------------------------------------------
# Smoke tests — runner end-to-end
# ----------------------------------------------------------------------

def test_run_experiment_unified_md_knapsack_alpha_fair_smoke():
    stage_df, iter_df = run_experiment_unified(
        {"task": _task_cfg(), "training": _train_cfg()},
        method_configs={"FDFL": ALL_METHOD_CONFIGS["FDFL"]},
    )
    assert not stage_df.empty
    assert not iter_df.empty
    assert set(stage_df["method"]) == {"fdfl"}
    # Train/val/test metrics all populated.
    for col in (
        "train_regret", "train_pred_mse", "train_fairness",
        "val_regret",   "val_pred_mse",   "val_fairness",
        "test_regret",  "test_pred_mse",  "test_fairness",
    ):
        assert col in stage_df.columns, f"missing column: {col}"
        assert not stage_df[col].isna().any(), f"NaN in {col}"


def test_run_experiment_unified_md_knapsack_lp_smoke():
    stage_df, _ = run_experiment_unified(
        {"task": _task_cfg(scenario="lp", alpha_fair=1.0), "training": _train_cfg()},
        method_configs={"FPTO": ALL_METHOD_CONFIGS["FPTO"]},
    )
    assert not stage_df.empty


def test_per_group_diagnostics_recorded_in_stage_csv():
    stage_df, _ = run_experiment_unified(
        {"task": _task_cfg(n_samples_train=40, benefit_group_bias=0.6),
         "training": _train_cfg()},
        method_configs={"FPTO": ALL_METHOD_CONFIGS["FPTO"]},
    )
    # Per-group benefit / cost / decision summaries should land in the row.
    diag_cols = [c for c in stage_df.columns if c.startswith("train_group_")]
    expected_keys = {
        "train_group_0_benefit_mean_r0", "train_group_1_benefit_mean_r0",
        "train_group_0_cost_mean_r0",    "train_group_1_cost_mean_r0",
        "train_group_0_decision_mean_r0",
    }
    assert expected_keys.issubset(set(diag_cols)), (
        f"expected diagnostic columns missing; got {sorted(diag_cols)[:5]}..."
    )
    # The imbalance should be visible: g0 benefit mean > g1 benefit mean.
    g0_mean = float(stage_df["train_group_0_benefit_mean_r0"].iloc[0])
    g1_mean = float(stage_df["train_group_1_benefit_mean_r0"].iloc[0])
    assert g0_mean > g1_mean


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def test_invalid_scenario_rejected():
    with pytest.raises(ValueError, match="scenario"):
        _build_task(_task_cfg(scenario="bogus"))


def test_invalid_n_resources_rejected():
    with pytest.raises(ValueError, match="n_resources"):
        _build_task(_task_cfg(n_resources=0))


def test_invalid_snr_rejected():
    with pytest.raises(ValueError, match="snr"):
        _build_task(_task_cfg(snr=-1.0))
