from __future__ import annotations

import copy

from experiments.configs import ALL_METHOD_CONFIGS, DEFAULT_TRAIN_CFG
from fair_dfl.runner import _build_task, run_experiment_unified
from fair_dfl.tasks import MultiDimKnapsackTask


def _task_cfg() -> dict:
    return {
        "name": "md_knapsack",
        "n_samples_train": 8,
        "n_samples_val": 4,
        "n_samples_test": 4,
        "n_features": 3,
        "n_items": 4,
        "n_constraints": 2,
        "scenario": "alpha_fair",
        "alpha_fair": 2.0,
        "poly_degree": 2,
        "group_bias": 0.2,
        "noise_std_lo": 0.05,
        "noise_std_hi": 0.1,
        "budget_tightness": 0.5,
        "data_seed": 7,
        "fairness_type": "mad",
    }


def test_build_task_md_knapsack():
    task, data = _build_task(_task_cfg())

    assert isinstance(task, MultiDimKnapsackTask)
    assert task.name == "md_knapsack"
    assert data.train.x.shape == (8, 3)
    assert data.train.y.shape == (8, 4)
    assert data.groups.shape == (4,)
    assert data.meta["A"].shape == (2, 4)
    assert data.meta["b"].shape == (2,)


def test_run_experiment_unified_md_knapsack_smoke():
    train_cfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
    train_cfg.update(
        {
            "device": "cpu",
            "decision_grad_backend": "finite_diff",
            "seeds": [11],
            "lambdas": [0.0],
            "steps_per_lambda": 1,
            "log_every": 1,
            "batch_size": -1,
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

    stage_df, iter_df = run_experiment_unified(
        {"task": _task_cfg(), "training": train_cfg},
        method_configs={"FDFL": ALL_METHOD_CONFIGS["FDFL"]},
    )

    assert not stage_df.empty
    assert not iter_df.empty
    assert set(stage_df["method"]) == {"fdfl"}
