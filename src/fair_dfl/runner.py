"""Top-level experiment runner and public method registry dispatch."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .algorithms.core_methods import METHOD_SPECS as CORE_METHOD_SPECS, METHOD_ALIASES, run_core_methods
from .algorithms.advanced_methods import ADVANCED_METHODS, run_advanced_methods
from .training import run_methods as _run_methods_unified, resolve_method_spec
from .tasks import (
    MedicalResourceAllocationTask,
    PortfolioQPMultiConstraintTask,
    PortfolioQPTask,
    PortfolioQPSimplexTask,
    PyEPONonlinearKnapsackTask,
    PyEPOPortfolioQPTask,
    PyEPOPortfolioQPSimplexTask,
    ResourceAllocationTask,
)
from .tasks.base import BaseTask, SplitData, TaskData


DEFAULT_METHODS = [
    "fplg",
    "fdfl",
    "plg",
    "fpto",
    "dfl",
]
PUBLIC_METHODS = [
    "fpto",
    "dfl",
    "fdfl",
    "plg",
    "fplg",
    "ffo",
    "nce",
    "lancer",
    "saa",
    "var_dro",
    "wass_dro",
]


def _apply_subset_fraction(
    task: BaseTask,
    data: TaskData,
    train_subset_fraction: float,
    subset_seed: int,
) -> TaskData:
    frac = float(train_subset_fraction)
    if frac >= 1.0:
        return data
    if not (0.0 < frac <= 1.0):
        raise ValueError("training.train_subset_fraction must be in (0, 1].")

    rng = np.random.default_rng(int(subset_seed) * 1009 + 17)

    def choose_idx(n_rows: int) -> np.ndarray:
        n_keep = max(1, int(round(float(n_rows) * frac)))
        n_keep = min(n_keep, n_rows)
        if n_keep == n_rows:
            return np.arange(n_rows, dtype=int)
        return np.sort(rng.choice(n_rows, size=n_keep, replace=False))

    train_idx = choose_idx(data.train.x.shape[0])
    val_idx = choose_idx(data.val.x.shape[0])
    test_idx = np.arange(data.test.x.shape[0], dtype=int)

    train = SplitData(x=data.train.x[train_idx], y=data.train.y[train_idx])
    val = SplitData(x=data.val.x[val_idx], y=data.val.y[val_idx])
    test = SplitData(x=data.test.x[test_idx], y=data.test.y[test_idx])

    if isinstance(task, MedicalResourceAllocationTask):
        for split_name, idx in [("train", train_idx), ("val", val_idx)]:
            med_split = task._splits[split_name]
            task._splits[split_name] = type(med_split)(
                x=med_split.x[idx],
                y=med_split.y[idx],
                cost=med_split.cost[idx],
                race=med_split.race[idx],
            )

    meta = dict(data.meta)
    if "n_train" in meta:
        meta["n_train"] = np.asarray([train.x.shape[0]], dtype=np.int64)
    if "n_val" in meta:
        meta["n_val"] = np.asarray([val.x.shape[0]], dtype=np.int64)
    if "n_test" in meta:
        meta["n_test"] = np.asarray([test.x.shape[0]], dtype=np.int64)
    if "n_total" in meta:
        meta["n_total"] = np.asarray([train.x.shape[0] + val.x.shape[0] + test.x.shape[0]], dtype=np.int64)

    return TaskData(
        train=train,
        val=val,
        test=test,
        groups=data.groups,
        meta=meta,
    )


def _build_task(task_cfg: Dict[str, Any]) -> Tuple[BaseTask, TaskData]:
    name = str(task_cfg["name"])
    fairness_type = str(task_cfg.get("fairness_type", "mad"))
    fairness_ge_alpha = float(task_cfg.get("fairness_ge_alpha", 2.0))
    if name == "resource_allocation":
        task = ResourceAllocationTask(
            n_samples_train=int(task_cfg["n_samples_train"]),
            n_samples_val=int(task_cfg["n_samples_val"]),
            n_samples_test=int(task_cfg["n_samples_test"]),
            n_features=int(task_cfg["n_features"]),
            n_items=int(task_cfg["n_items"]),
            alpha_fair=float(task_cfg["alpha_fair"]),
            budget=float(task_cfg["budget"]),
            group_bias=float(task_cfg.get("group_bias", 0.0)),
            noise_std=float(task_cfg.get("noise_std", 0.2)),
            fairness_type=fairness_type,
            fairness_ge_alpha=fairness_ge_alpha,
        )
        data_seed = int(task_cfg.get("data_seed", 42))
        data = task.generate_data(seed=data_seed)
        task.bind_context(groups=data.groups, costs=data.meta["costs"])
        return task, data

    if name == "portfolio_qp":
        task = PortfolioQPTask(
            n_samples_train=int(task_cfg["n_samples_train"]),
            n_samples_val=int(task_cfg["n_samples_val"]),
            n_samples_test=int(task_cfg["n_samples_test"]),
            n_features=int(task_cfg["n_features"]),
            n_assets=int(task_cfg["n_assets"]),
            risk_aversion=float(task_cfg["risk_aversion"]),
            group_bias=float(task_cfg.get("group_bias", 0.0)),
            noise_std=float(task_cfg.get("noise_std", 0.3)),
            fairness_type=fairness_type,
            fairness_ge_alpha=fairness_ge_alpha,
        )
        data_seed = int(task_cfg.get("data_seed", 42))
        data = task.generate_data(seed=data_seed)
        task.bind_context(groups=data.groups, sigma=data.meta["sigma"])
        return task, data

    if name == "portfolio_qp_simplex":
        task = PortfolioQPSimplexTask(
            n_samples_train=int(task_cfg["n_samples_train"]),
            n_samples_val=int(task_cfg["n_samples_val"]),
            n_samples_test=int(task_cfg["n_samples_test"]),
            n_features=int(task_cfg["n_features"]),
            n_assets=int(task_cfg["n_assets"]),
            risk_aversion=float(task_cfg["risk_aversion"]),
            group_bias=float(task_cfg.get("group_bias", 0.0)),
            noise_std=float(task_cfg.get("noise_std", 0.3)),
            fairness_type=fairness_type,
            fairness_ge_alpha=fairness_ge_alpha,
            cvxpy_solvers=[str(s) for s in task_cfg.get("cvxpy_solvers", [])] or None,
        )
        data_seed = int(task_cfg.get("data_seed", 42))
        data = task.generate_data(seed=data_seed)
        task.bind_context(groups=data.groups, sigma=data.meta["sigma"])
        return task, data

    if name == "pyepo_nonlinear_knapsack":
        task = PyEPONonlinearKnapsackTask(
            n_samples_train=int(task_cfg["n_samples_train"]),
            n_samples_val=int(task_cfg["n_samples_val"]),
            n_samples_test=int(task_cfg["n_samples_test"]),
            n_features=int(task_cfg["n_features"]),
            n_items=int(task_cfg["n_items"]),
            alpha_fair=float(task_cfg["alpha_fair"]),
            budget=float(task_cfg["budget"]),
            group_bias=float(task_cfg.get("group_bias", 0.0)),
            noise_std=float(task_cfg.get("noise_std", 0.2)),
            fairness_type=fairness_type,
            fairness_ge_alpha=fairness_ge_alpha,
            pyepo_dim=int(task_cfg.get("pyepo_dim", 1)),
            pyepo_deg=int(task_cfg.get("pyepo_deg", 2)),
            pyepo_noise_width=float(task_cfg.get("pyepo_noise_width", 0.1)),
        )
        data_seed = int(task_cfg.get("data_seed", 42))
        data = task.generate_data(seed=data_seed)
        task.bind_context(groups=data.groups, costs=data.meta["costs"])
        return task, data

    if name == "pyepo_portfolio_qp":
        task = PyEPOPortfolioQPTask(
            n_samples_train=int(task_cfg["n_samples_train"]),
            n_samples_val=int(task_cfg["n_samples_val"]),
            n_samples_test=int(task_cfg["n_samples_test"]),
            n_features=int(task_cfg["n_features"]),
            n_assets=int(task_cfg["n_assets"]),
            risk_aversion=float(task_cfg["risk_aversion"]),
            group_bias=float(task_cfg.get("group_bias", 0.0)),
            noise_std=float(task_cfg.get("noise_std", 0.3)),
            fairness_type=fairness_type,
            fairness_ge_alpha=fairness_ge_alpha,
            pyepo_deg=int(task_cfg.get("pyepo_deg", 2)),
            pyepo_noise_level=float(task_cfg.get("pyepo_noise_level", 1.0)),
        )
        data_seed = int(task_cfg.get("data_seed", 42))
        data = task.generate_data(seed=data_seed)
        task.bind_context(groups=data.groups, sigma=data.meta["sigma"])
        return task, data

    if name == "pyepo_portfolio_qp_simplex":
        task = PyEPOPortfolioQPSimplexTask(
            n_samples_train=int(task_cfg["n_samples_train"]),
            n_samples_val=int(task_cfg["n_samples_val"]),
            n_samples_test=int(task_cfg["n_samples_test"]),
            n_features=int(task_cfg["n_features"]),
            n_assets=int(task_cfg["n_assets"]),
            risk_aversion=float(task_cfg["risk_aversion"]),
            group_bias=float(task_cfg.get("group_bias", 0.0)),
            noise_std=float(task_cfg.get("noise_std", 0.3)),
            fairness_type=fairness_type,
            fairness_ge_alpha=fairness_ge_alpha,
            pyepo_deg=int(task_cfg.get("pyepo_deg", 2)),
            pyepo_noise_level=float(task_cfg.get("pyepo_noise_level", 1.0)),
            cvxpy_solvers=[str(s) for s in task_cfg.get("cvxpy_solvers", [])] or None,
        )
        data_seed = int(task_cfg.get("data_seed", 42))
        data = task.generate_data(seed=data_seed)
        task.bind_context(groups=data.groups, sigma=data.meta["sigma"])
        return task, data

    if name == "portfolio_qp_multi_constraint":
        task = PortfolioQPMultiConstraintTask(
            n_samples_train=int(task_cfg["n_samples_train"]),
            n_samples_val=int(task_cfg["n_samples_val"]),
            n_samples_test=int(task_cfg["n_samples_test"]),
            n_features=int(task_cfg["n_features"]),
            n_assets=int(task_cfg["n_assets"]),
            n_factors=int(task_cfg["n_factors"]),
            risk_aversion=float(task_cfg["risk_aversion"]),
            group_bias=float(task_cfg.get("group_bias", 0.0)),
            noise_std=float(task_cfg.get("noise_std", 0.3)),
        )
        data_seed = int(task_cfg.get("data_seed", 42))
        data = task.generate_data(seed=data_seed)
        task.bind_context(
            groups=data.groups,
            sigma=data.meta["sigma"],
            constraints=data.meta["constraints"],
            targets=data.meta["targets"],
        )
        return task, data

    if name == "medical_resource_allocation":
        task = MedicalResourceAllocationTask(
            data_csv=str(task_cfg["data_csv"]),
            n_sample=int(task_cfg.get("n_sample", 5000)),
            data_seed=int(task_cfg.get("data_seed", 42)),
            split_seed=int(task_cfg.get("split_seed", 2)),
            test_fraction=float(task_cfg.get("test_fraction", 0.5)),
            val_fraction=float(task_cfg.get("val_fraction", 0.2)),
            alpha_fair=float(task_cfg.get("alpha_fair", 2.0)),
            budget=float(task_cfg.get("budget", 2500.0)),
            decision_mode=str(task_cfg.get("decision_mode", "group")),
            fairness_type=str(task_cfg.get("fairness_type", "mad")),
            budget_rho=float(task_cfg.get("budget_rho", 0.35)),
        )
        data = task.generate_data(seed=int(task_cfg.get("data_seed", 42)))
        return task, data

    raise ValueError(f"Unknown task name: {name}")


def run_experiment(
    cfg: Dict[str, Any],
    methods: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selected_methods = methods or DEFAULT_METHODS
    selected_methods = [METHOD_ALIASES.get(m, m) for m in selected_methods]
    for m in selected_methods:
        if m not in PUBLIC_METHODS:
            raise ValueError(f"Unknown method: {m}")

    task, data = _build_task(cfg["task"])
    train_cfg_raw = cfg.get("training", {})
    train_cfg = dict(train_cfg_raw)
    pareto_sweep_mode = bool(train_cfg.get("pareto_sweep_mode", train_cfg.get("frontier_mode", False)))
    if pareto_sweep_mode:
        lambdas = [float(v) for v in train_cfg.get("lambdas", [0.0])]
        train_cfg["lambdas"] = lambdas if lambdas else [0.0]
    else:
        # Non-sweep mode runs a single training lambda.
        train_cfg["lambdas"] = [float(train_cfg.get("lambda_train", 0.0))]

    subset_fraction = float(train_cfg.get("train_subset_fraction", 1.0))
    subset_seed = int(cfg.get("task", {}).get("data_seed", 42))
    data = _apply_subset_fraction(task=task, data=data, train_subset_fraction=subset_fraction, subset_seed=subset_seed)

    core_methods = [m for m in selected_methods if m in CORE_METHOD_SPECS]
    advanced_methods = [m for m in selected_methods if m in ADVANCED_METHODS]

    stage_rows: List[Dict[str, Any]] = []
    iter_rows: List[Dict[str, Any]] = []

    if core_methods:
        stg, itr = run_core_methods(
            task=task,
            data=data,
            train_cfg=train_cfg,
            methods=core_methods,
        )
        stage_rows.extend(stg)
        iter_rows.extend(itr)

    if advanced_methods:
        if not bool(train_cfg.get("advanced_backend", True)):
            raise ValueError("Advanced methods requested but training.advanced_backend is false.")
        stg, itr = run_advanced_methods(
            task=task,  # type: ignore[arg-type]
            train_cfg=train_cfg,
            methods=advanced_methods,
        )
        stage_rows.extend(stg)
        iter_rows.extend(itr)

    return pd.DataFrame(stage_rows), pd.DataFrame(iter_rows)


def run_experiment_unified(
    cfg: Dict[str, Any],
    method_configs: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run experiments using the unified training loop.

    Unlike run_experiment(), this accepts the full method_configs dict
    from configs.py (with inline use_dec/use_pred/use_fair flags) and
    uses the unified training loop for ALL methods including FFO/NCE/LANCER.

    Args:
        cfg: Experiment config with "task" and "training" keys.
        method_configs: Dict of {method_name: method_config} from ALL_METHOD_CONFIGS.

    Returns:
        (stage_results_df, iter_logs_df)
    """
    task, data = _build_task(cfg["task"])
    train_cfg_raw = cfg.get("training", {})
    train_cfg = dict(train_cfg_raw)
    pareto_sweep_mode = bool(train_cfg.get("pareto_sweep_mode", train_cfg.get("frontier_mode", False)))
    if pareto_sweep_mode:
        lambdas = [float(v) for v in train_cfg.get("lambdas", [0.0])]
        train_cfg["lambdas"] = lambdas if lambdas else [0.0]
    else:
        train_cfg["lambdas"] = [float(train_cfg.get("lambda_train", 0.0))]

    subset_fraction = float(train_cfg.get("train_subset_fraction", 1.0))
    subset_seed = int(cfg.get("task", {}).get("data_seed", 42))
    data = _apply_subset_fraction(
        task=task, data=data,
        train_subset_fraction=subset_fraction,
        subset_seed=subset_seed,
    )

    stage_rows, iter_rows = _run_methods_unified(
        task=task,
        data=data,
        train_cfg=train_cfg,
        method_configs=method_configs,
    )
    return pd.DataFrame(stage_rows), pd.DataFrame(iter_rows)
