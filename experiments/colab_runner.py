"""Shared experiment runner for parallel Colab notebooks.

All worker notebooks import from this module. Handles:
  - Method grid definition
  - Healthcare and knapsack run functions
  - Progress display
  - Result aggregation

Each notebook handles its own Drive mounting and path setup in its first cell.
This module is imported AFTER paths are configured.

Usage (in a Colab notebook):
    # (notebook setup cell mounts Drive, sets sys.path, cd's to DRIVE_ROOT)
    from experiments.colab_runner import *
    run_healthcare_slice(alphas=[0.5], results_dir=HC_RESULTS)
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================
# Constants
# ======================================================================

SEEDS = [11, 22, 33, 44, 55]
ALPHAS = [0.5, 2.0]
HIDDEN_DIMS = [64]
LAMBDAS_SWEEP = [0.0, 0.5, 1.0, 5.0]

# Methods that use decision gradients — need mini-batch for solver calls
DECISION_GRAD_METHODS = {'FDFL-Scal', 'FDFL-PCGrad', 'FDFL-MGDA', 'FDFL-CAGrad'}

# Default results paths — relative to DRIVE_ROOT (cwd set by notebook setup cell)
HC_RESULTS_DEFAULT = "results/final/healthcare"
KN_RESULTS_DEFAULT = "results/final/knapsack"

# Unfairness levels for knapsack
UNFAIRNESS_LEVELS = {
    "mild":   {"group_bias": 0.1, "noise_std_lo": 0.1, "noise_std_hi": 0.2, "group_ratio": 0.5},
    "medium": {"group_bias": 0.3, "noise_std_lo": 0.1, "noise_std_hi": 0.5, "group_ratio": 0.5},
    "high":   {"group_bias": 0.3, "noise_std_lo": 0.1, "noise_std_hi": 0.5, "group_ratio": 0.67},
}

METHOD_GRID = {
    "FPTO":        {"config": "FPTO",    "lambdas": LAMBDAS_SWEEP},
    "SAA":         {"config": "SAA",     "lambdas": [0.0]},
    "WDRO":        {"config": "WDRO",    "lambdas": [0.0]},
    "FDFL-Scal":   {"config": "FPLG",    "lambdas": LAMBDAS_SWEEP},
    "FDFL-PCGrad": {"config": "PCGrad",  "lambdas": [0.0]},
    "FDFL-MGDA":   {"config": "MGDA",    "lambdas": [0.0]},
    "FDFL-CAGrad": {"config": "CAGrad",  "lambdas": [0.0]},
}


# ======================================================================
# Checkpoint helpers
# ======================================================================

def _done(result_dir: Path) -> bool:
    return (result_dir / "stage_results.csv").exists()


def _save_run(result_dir: Path, stage_df, iter_df, meta: dict):
    result_dir.mkdir(parents=True, exist_ok=True)
    if not stage_df.empty:
        stage_df.to_csv(result_dir / "stage_results.csv", index=False)
    if not iter_df.empty:
        iter_df.to_csv(result_dir / "iter_logs.csv", index=False)
    with open(result_dir / "run_config.json", "w") as f:
        json.dump(meta, f, indent=2)


# Keys that belong inside train_cfg["model"] rather than train_cfg
_MODEL_KEYS = {"hidden_dim", "n_layers", "activation", "dropout", "batch_norm",
               "arch", "init_mode"}


def _apply_train_overrides(train_cfg: dict, overrides: dict) -> None:
    """Apply notebook-level training overrides to a train_cfg dict (in-place).

    Model-level keys (hidden_dim, n_layers, etc.) are routed into
    train_cfg["model"]; everything else goes directly into train_cfg.
    """
    for k, v in overrides.items():
        if k in _MODEL_KEYS:
            train_cfg["model"][k] = v
        else:
            train_cfg[k] = v


# ======================================================================
# Healthcare runner
# ======================================================================

def run_healthcare_slice(
    alphas: list[float] | None = None,
    hidden_dims: list[int] | None = None,
    seeds: list[int] | None = None,
    methods: list[str] | None = None,
    results_dir: str = HC_RESULTS_DEFAULT,
    data_csv: str = "data/data_processed.csv",
    device: str | None = None,
    steps: int = 70,
    task_overrides: dict | None = None,
    train_overrides: dict | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Run a slice of the healthcare grid. Skips completed runs.

    Call with different alphas/methods from different notebooks for parallelism.

    Parameters
    ----------
    task_overrides : dict, optional
        Override task-level parameters (n_sample, budget_rho, test_fraction,
        val_fraction, decision_mode, fairness_type, data_seed, split_seed).
    train_overrides : dict, optional
        Override training parameters. Model-level keys (hidden_dim, n_layers,
        activation, dropout, batch_norm) are routed into the model sub-config.
        Other keys (lr, lr_decay, grad_clip_norm, batch_size, log_every, etc.)
        go directly into the training config.
    overwrite : bool
        If False (default), skip runs that already have stage_results.csv.
        Set True to re-run and overwrite existing results.
    """
    import torch
    from fair_dfl.runner import run_experiment_unified
    from experiments.configs import ALL_METHOD_CONFIGS, DEFAULT_TRAIN_CFG

    alphas = alphas or ALPHAS
    hidden_dims = hidden_dims or HIDDEN_DIMS
    seeds = seeds or SEEDS
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t_ovr = task_overrides or {}
    tr_ovr = train_overrides or {}

    selected = {k: v for k, v in METHOD_GRID.items()
                if methods is None or k in methods}

    # Compute batch size from data (unless overridden)
    df_data = pd.read_csv(data_csv)
    n_sample = t_ovr.get("n_sample", 0)
    n_total = n_sample if (n_sample > 0 and n_sample < len(df_data)) else len(df_data)
    test_frac = t_ovr.get("test_fraction", 0.5)
    n_test = int(round(test_frac * n_total))
    auto_batch_size = n_total - n_test

    total = len(selected) * len(alphas) * len(hidden_dims) * len(seeds)
    print(f"\n{'='*60}")
    print(f"Healthcare slice: {list(selected.keys())}")
    print(f"Alphas={alphas}, HiddenDims={hidden_dims}, Seeds={seeds}")
    print(f"Device={device}, Total runs={total}")
    if t_ovr:
        print(f"Task overrides: {t_ovr}")
    if tr_ovr:
        print(f"Train overrides: {tr_ovr}")
    print(f"{'='*60}")

    all_stage = []
    errors = []
    times = []
    skipped = 0
    run_idx = 0

    for method_label, grid_spec in selected.items():
        config_name = grid_spec["config"]
        lambdas = grid_spec["lambdas"]
        method_spec = copy.deepcopy(ALL_METHOD_CONFIGS[config_name])

        for alpha in alphas:
            for hd in hidden_dims:
                for seed in seeds:
                    run_idx += 1
                    rdir = Path(results_dir) / method_label / f"alpha_{alpha}_hd_{hd}" / f"seed_{seed}"

                    if _done(rdir) and not overwrite:
                        skipped += 1
                        try:
                            all_stage.append(pd.read_csv(rdir / "stage_results.csv"))
                        except Exception:
                            pass
                        continue

                    print(f"  [{run_idx}/{total}] {method_label} a={alpha} hd={hd} s={seed}", end=" ", flush=True)

                    try:
                        task_cfg = {
                            "name": "medical_resource_allocation",
                            "data_csv": data_csv,
                            "n_sample": t_ovr.get("n_sample", 0),
                            "data_seed": t_ovr.get("data_seed", 42),
                            "split_seed": t_ovr.get("split_seed", 2),
                            "test_fraction": t_ovr.get("test_fraction", 0.5),
                            "val_fraction": t_ovr.get("val_fraction", 0.0),
                            "alpha_fair": alpha,
                            "budget": t_ovr.get("budget", -1),
                            "budget_rho": t_ovr.get("budget_rho", 0.35),
                            "decision_mode": t_ovr.get("decision_mode", "group"),
                            "fairness_type": t_ovr.get("fairness_type", "mad"),
                        }

                        train_cfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
                        train_cfg["seeds"] = [seed]
                        train_cfg["lambdas"] = lambdas
                        train_cfg["steps_per_lambda"] = steps
                        train_cfg["batch_size"] = tr_ovr.get("batch_size", auto_batch_size)
                        train_cfg["model"]["hidden_dim"] = hd
                        train_cfg["device"] = device
                        _apply_train_overrides(train_cfg, tr_ovr)

                        for k, v in method_spec.items():
                            if k not in {"method", "use_dec", "use_pred", "use_fair",
                                         "pred_weight_mode", "continuation",
                                         "allow_orthogonalization"}:
                                train_cfg[k] = v

                        cfg = {"task": task_cfg, "training": train_cfg}
                        t0 = time.time()
                        stage_df, iter_df = run_experiment_unified(
                            cfg, method_configs={config_name: method_spec}
                        )
                        elapsed = time.time() - t0
                        times.append(elapsed)

                        for df in (stage_df, iter_df):
                            if not df.empty:
                                df["method_label"] = method_label
                                df["alpha_fair"] = alpha
                                df["hidden_dim"] = hd
                                df["config_name"] = config_name

                        _save_run(rdir, stage_df, iter_df, {
                            "method_label": method_label, "config_name": config_name,
                            "alpha": alpha, "hidden_dim": hd, "seed": seed,
                            "lambdas": lambdas, "elapsed_sec": elapsed,
                        })
                        if not stage_df.empty:
                            all_stage.append(stage_df)
                        print(f"({elapsed:.1f}s)")

                        # Estimate remaining after first run
                        if len(times) == 1:
                            remaining = total - run_idx
                            print(f"    Est. remaining: {remaining * elapsed / 60:.0f}min")

                    except Exception as e:
                        errors.append({"method": method_label, "alpha": alpha,
                                       "hd": hd, "seed": seed, "error": str(e)})
                        print(f"ERROR: {e}")
                        traceback.print_exc()

    print(f"\nDone: {run_idx-skipped-len(errors)} new, {skipped} skipped, {len(errors)} errors")
    if times:
        print(f"Avg: {np.mean(times):.1f}s/run, Total: {sum(times)/60:.1f}min")
    if errors:
        print("Errors:", errors)

    if all_stage:
        result = pd.concat(all_stage, ignore_index=True)
        agg_path = Path(results_dir) / "stage_results_all.csv"
        result.to_csv(agg_path, index=False)
        print(f"Saved: {agg_path} ({len(result)} rows)")
        return result
    return pd.DataFrame()


# ======================================================================
# Knapsack runner
# ======================================================================

def run_knapsack_slice(
    alphas: list[float] | None = None,
    unfairness_levels: list[str] | None = None,
    seeds: list[int] | None = None,
    methods: list[str] | None = None,
    results_dir: str = KN_RESULTS_DEFAULT,
    device: str = "cpu",
    steps: int = 20,
    unfairness_configs: dict | None = None,
    task_overrides: dict | None = None,
    train_overrides: dict | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Run a slice of the knapsack grid. Skips completed runs.

    All methods use 5 seeds (SPSA makes decision-gradient methods fast enough).
    Override by passing seeds= explicitly.

    Parameters
    ----------
    unfairness_configs : dict, optional
        Override the default UNFAIRNESS_LEVELS dict. Keys are level names,
        values are dicts with group_bias, noise_std_lo, noise_std_hi, group_ratio.
    task_overrides : dict, optional
        Override task-level parameters (n_items, budget_tightness, poly_degree,
        n_samples_train, n_samples_test, n_features, n_constraints, etc.).
    train_overrides : dict, optional
        Override training parameters. Model-level keys (hidden_dim, n_layers,
        activation, dropout, batch_norm) are routed into the model sub-config.
        Other keys (lr, lr_decay, grad_clip_norm, log_every,
        decision_grad_backend, fdfl_batch_size, etc.) go directly into the
        training config. Use fdfl_batch_size to override the mini-batch size
        for decision-gradient methods (default 32).
    """
    from fair_dfl.runner import run_experiment_unified
    from experiments.configs import ALL_METHOD_CONFIGS, DEFAULT_TRAIN_CFG

    alphas = alphas or ALPHAS
    uf_configs = unfairness_configs if unfairness_configs is not None else UNFAIRNESS_LEVELS
    unfairness_levels = unfairness_levels or list(uf_configs.keys())
    default_seeds = seeds or SEEDS
    t_ovr = task_overrides or {}
    tr_ovr = train_overrides or {}

    selected = {k: v for k, v in METHOD_GRID.items()
                if methods is None or k in methods}

    total = len(selected) * len(alphas) * len(unfairness_levels) * len(default_seeds)

    print(f"\n{'='*60}")
    print(f"Knapsack slice: {list(selected.keys())}")
    print(f"Alphas={alphas}, Unfairness={unfairness_levels}")
    print(f"Seeds: {default_seeds}")
    if t_ovr:
        print(f"Task overrides: {t_ovr}")
    if tr_ovr:
        print(f"Train overrides: {tr_ovr}")
    if unfairness_configs is not None:
        for k, v in uf_configs.items():
            print(f"  {k}: {v}")
    print(f"Total runs={total}")
    print(f"{'='*60}")

    all_stage = []
    errors = []
    times = []
    skipped = 0
    run_idx = 0

    for method_label, grid_spec in selected.items():
        config_name = grid_spec["config"]
        lambdas = grid_spec["lambdas"]
        method_spec = copy.deepcopy(ALL_METHOD_CONFIGS[config_name])

        for alpha in alphas:
            for uf_name in unfairness_levels:
                uf = uf_configs[uf_name]
                for seed in default_seeds:
                    run_idx += 1
                    rdir = Path(results_dir) / method_label / f"alpha_{alpha}_uf_{uf_name}" / f"seed_{seed}"

                    if _done(rdir) and not overwrite:
                        skipped += 1
                        try:
                            all_stage.append(pd.read_csv(rdir / "stage_results.csv"))
                        except Exception:
                            pass
                        continue

                    print(f"  [{run_idx}/{total}] {method_label} a={alpha} uf={uf_name} s={seed}", end=" ", flush=True)

                    try:
                        task_cfg = {
                            "name": "md_knapsack",
                            "n_samples_train": t_ovr.get("n_samples_train", 200),
                            "n_samples_val": 0,
                            "n_samples_test": t_ovr.get("n_samples_test", 80),
                            "n_features": t_ovr.get("n_features", 5),
                            "n_items": t_ovr.get("n_items", 7),
                            "n_constraints": t_ovr.get("n_constraints", 3),
                            "scenario": "alpha_fair",
                            "alpha_fair": alpha,
                            "poly_degree": t_ovr.get("poly_degree", 2),
                            "group_bias": uf["group_bias"],
                            "noise_std_lo": uf["noise_std_lo"],
                            "noise_std_hi": uf["noise_std_hi"],
                            "group_ratio": uf["group_ratio"],
                            "budget_tightness": t_ovr.get("budget_tightness", 0.5),
                            "decision_mode": t_ovr.get("decision_mode", "group"),
                            "data_seed": 42, "fairness_type": "mad",
                        }

                        fdfl_bsz = tr_ovr.get("fdfl_batch_size", 32)
                        dec_backend = tr_ovr.get("decision_grad_backend", "spsa")

                        train_cfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
                        train_cfg["seeds"] = [seed]
                        train_cfg["lambdas"] = lambdas
                        train_cfg["steps_per_lambda"] = steps
                        train_cfg["batch_size"] = (
                            fdfl_bsz if method_label in DECISION_GRAD_METHODS else -1
                        )
                        train_cfg["decision_grad_backend"] = dec_backend
                        train_cfg["device"] = device
                        train_cfg["model"]["hidden_dim"] = 64
                        train_cfg["log_every"] = 2
                        _apply_train_overrides(train_cfg, {
                            k: v for k, v in tr_ovr.items()
                            if k not in ("fdfl_batch_size", "decision_grad_backend")
                        })

                        for k, v in method_spec.items():
                            if k not in {"method", "use_dec", "use_pred", "use_fair",
                                         "pred_weight_mode", "continuation",
                                         "allow_orthogonalization"}:
                                train_cfg[k] = v

                        cfg = {"task": task_cfg, "training": train_cfg}
                        t0 = time.time()
                        stage_df, iter_df = run_experiment_unified(
                            cfg, method_configs={config_name: method_spec}
                        )
                        elapsed = time.time() - t0
                        times.append(elapsed)

                        for df in (stage_df, iter_df):
                            if not df.empty:
                                df["method_label"] = method_label
                                df["alpha_fair"] = alpha
                                df["unfairness_level"] = uf_name
                                df["group_bias"] = uf["group_bias"]
                                df["group_ratio"] = uf["group_ratio"]
                                df["config_name"] = config_name

                        _save_run(rdir, stage_df, iter_df, {
                            "method_label": method_label, "config_name": config_name,
                            "alpha": alpha, "unfairness": uf_name, "seed": seed,
                            "lambdas": lambdas, "elapsed_sec": elapsed,
                        })
                        if not stage_df.empty:
                            all_stage.append(stage_df)
                        print(f"({elapsed:.1f}s)")

                        if len(times) == 1:
                            remaining = total - run_idx
                            print(f"    Est. remaining: {remaining * elapsed / 60:.0f}min")

                    except Exception as e:
                        errors.append({"method": method_label, "alpha": alpha,
                                       "uf": uf_name, "seed": seed, "error": str(e)})
                        print(f"ERROR: {e}")
                        traceback.print_exc()

    print(f"\nDone: {run_idx-skipped-len(errors)} new, {skipped} skipped, {len(errors)} errors")
    if times:
        print(f"Avg: {np.mean(times):.1f}s/run, Total: {sum(times)/60:.1f}min")
    if errors:
        print("Errors:", errors)

    if all_stage:
        result = pd.concat(all_stage, ignore_index=True)
        agg_path = Path(results_dir) / "stage_results_all.csv"
        result.to_csv(agg_path, index=False)
        print(f"Saved: {agg_path} ({len(result)} rows)")
        return result
    return pd.DataFrame()


# ======================================================================
# LP Knapsack runner
# ======================================================================

LP_RESULTS_DEFAULT = "results/final/lp_knapsack"

LP_UNFAIRNESS_LEVELS = {
    "mild":   {"group_bias": 0.2, "noise_std_lo": 0.05, "noise_std_hi": 0.5,  "group_ratio": 0.5},
    "medium": {"group_bias": 0.4, "noise_std_lo": 0.05, "noise_std_hi": 1.0,  "group_ratio": 0.65},
    "high":   {"group_bias": 0.6, "noise_std_lo": 0.05, "noise_std_hi": 1.5,  "group_ratio": 0.75},
}


def run_lp_knapsack_slice(
    unfairness_levels: list[str] | None = None,
    seeds: list[int] | None = None,
    methods: list[str] | None = None,
    results_dir: str = LP_RESULTS_DEFAULT,
    device: str = "cpu",
    steps: int = 200,
    unfairness_configs: dict | None = None,
    task_overrides: dict | None = None,
    train_overrides: dict | None = None,
) -> pd.DataFrame:
    """Run LP knapsack experiment with SPO+ decision gradients.

    No alpha loop — LP objective has no alpha parameter.
    FDFL methods use decision_grad_backend='spo_plus'.
    """
    from fair_dfl.runner import run_experiment_unified
    from experiments.configs import ALL_METHOD_CONFIGS, DEFAULT_TRAIN_CFG

    uf_configs = unfairness_configs if unfairness_configs is not None else LP_UNFAIRNESS_LEVELS
    unfairness_levels = unfairness_levels or list(uf_configs.keys())
    default_seeds = seeds or SEEDS
    t_ovr = task_overrides or {}
    tr_ovr = train_overrides or {}

    selected = {k: v for k, v in METHOD_GRID.items()
                if methods is None or k in methods}

    total = len(selected) * len(unfairness_levels) * len(default_seeds)

    print(f"\n{'='*60}")
    print(f"LP Knapsack (SPO+): {list(selected.keys())}")
    print(f"Unfairness={unfairness_levels}, Seeds={default_seeds}")
    if t_ovr:
        print(f"Task overrides: {t_ovr}")
    if tr_ovr:
        print(f"Train overrides: {tr_ovr}")
    if unfairness_configs is not None:
        for k, v in uf_configs.items():
            print(f"  {k}: {v}")
    print(f"Total runs={total}")
    print(f"{'='*60}")

    all_stage = []
    errors = []
    times = []
    skipped = 0
    run_idx = 0

    for method_label, grid_spec in selected.items():
        config_name = grid_spec["config"]
        lambdas = grid_spec["lambdas"]
        method_spec = copy.deepcopy(ALL_METHOD_CONFIGS[config_name])
        is_fdfl = method_label in DECISION_GRAD_METHODS

        for uf_name in unfairness_levels:
            uf = uf_configs[uf_name]
            for seed in default_seeds:
                run_idx += 1
                rdir = Path(results_dir) / method_label / f"uf_{uf_name}" / f"seed_{seed}"

                if _done(rdir):
                    skipped += 1
                    try:
                        all_stage.append(pd.read_csv(rdir / "stage_results.csv"))
                    except Exception:
                        pass
                    continue

                print(f"  [{run_idx}/{total}] {method_label} uf={uf_name} s={seed}", end=" ", flush=True)

                try:
                    task_cfg = {
                        "name": "md_knapsack",
                        "n_samples_train": t_ovr.get("n_samples_train", 80),
                        "n_samples_val": 0,
                        "n_samples_test": t_ovr.get("n_samples_test", 80),
                        "n_features": t_ovr.get("n_features", 5),
                        "n_items": t_ovr.get("n_items", 10),
                        "n_constraints": t_ovr.get("n_constraints", 3),
                        "scenario": "lp",
                        "alpha_fair": 1.0,
                        "poly_degree": t_ovr.get("poly_degree", 2),
                        "group_bias": uf["group_bias"],
                        "noise_std_lo": uf["noise_std_lo"],
                        "noise_std_hi": uf["noise_std_hi"],
                        "group_ratio": uf["group_ratio"],
                        "budget_tightness": t_ovr.get("budget_tightness", 0.3),
                        "data_seed": 42, "fairness_type": "mad",
                    }

                    fdfl_bsz = tr_ovr.get("fdfl_batch_size", 32)
                    dec_backend = tr_ovr.get("decision_grad_backend", "spo_plus")

                    train_cfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
                    train_cfg["seeds"] = [seed]
                    train_cfg["lambdas"] = lambdas
                    train_cfg["steps_per_lambda"] = steps
                    train_cfg["lr"] = tr_ovr.get("lr", 0.002)
                    train_cfg["batch_size"] = fdfl_bsz if is_fdfl else -1
                    train_cfg["decision_grad_backend"] = dec_backend if is_fdfl else "analytic"
                    train_cfg["device"] = device
                    train_cfg["model"]["hidden_dim"] = 64
                    train_cfg["log_every"] = 2
                    _apply_train_overrides(train_cfg, {
                        k: v for k, v in tr_ovr.items()
                        if k not in ("fdfl_batch_size", "decision_grad_backend", "lr")
                    })

                    for k, v in method_spec.items():
                        if k not in {"method", "use_dec", "use_pred", "use_fair",
                                     "pred_weight_mode", "continuation",
                                     "allow_orthogonalization"}:
                            train_cfg[k] = v

                    cfg = {"task": task_cfg, "training": train_cfg}
                    t0 = time.time()
                    stage_df, iter_df = run_experiment_unified(
                        cfg, method_configs={config_name: method_spec}
                    )
                    elapsed = time.time() - t0
                    times.append(elapsed)

                    for df in (stage_df, iter_df):
                        if not df.empty:
                            df["method_label"] = method_label
                            df["unfairness_level"] = uf_name
                            df["group_bias"] = uf["group_bias"]
                            df["group_ratio"] = uf["group_ratio"]
                            df["config_name"] = config_name

                    _save_run(rdir, stage_df, iter_df, {
                        "method_label": method_label, "config_name": config_name,
                        "unfairness": uf_name, "seed": seed,
                        "lambdas": lambdas, "elapsed_sec": elapsed,
                    })
                    if not stage_df.empty:
                        all_stage.append(stage_df)
                    print(f"({elapsed:.1f}s)")

                    if len(times) == 1:
                        remaining = total - run_idx
                        print(f"    Est. remaining: {remaining * elapsed / 60:.0f}min")

                except Exception as e:
                    errors.append({"method": method_label,
                                   "uf": uf_name, "seed": seed, "error": str(e)})
                    print(f"ERROR: {e}")
                    traceback.print_exc()

    print(f"\nDone: {run_idx-skipped-len(errors)} new, {skipped} skipped, {len(errors)} errors")
    if times:
        print(f"Avg: {np.mean(times):.1f}s/run, Total: {sum(times)/60:.1f}min")
    if errors:
        print("Errors:", errors)

    if all_stage:
        result = pd.concat(all_stage, ignore_index=True)
        agg_path = Path(results_dir) / "stage_results_all.csv"
        result.to_csv(agg_path, index=False)
        print(f"Saved: {agg_path} ({len(result)} rows)")
        return result
    return pd.DataFrame()


# ======================================================================
# Progress display
# ======================================================================

def show_progress(results_dir: str, name: str = "") -> pd.DataFrame:
    """Display progress and metrics summary — separate table per alpha."""
    p = Path(results_dir)
    csvs = sorted(p.rglob("stage_results.csv"))
    if not csvs:
        print(f"{name}: No results yet")
        return pd.DataFrame()

    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)

    print(f"\n{'='*60}")
    print(f"{name} — {len(df)} rows from {len(csvs)} runs")
    print(f"{'='*60}")

    if "method_label" in df.columns:
        print(f"Methods: {sorted(df['method_label'].unique())}")
    if "seed" in df.columns:
        print(f"Seeds:   {sorted(df['seed'].unique())}")
    if "alpha_fair" in df.columns:
        print(f"Alphas:  {sorted(df['alpha_fair'].unique())}")

    metric_cols = [c for c in ["test_regret_normalized", "test_fairness", "test_pred_mse"]
                   if c in df.columns]
    group_cols = [c for c in ["method_label", "lambda"] if c in df.columns]

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 200)

    if metric_cols and group_cols and "alpha_fair" in df.columns:
        for alpha in sorted(df["alpha_fair"].unique()):
            sub = df[df["alpha_fair"] == alpha]
            print(f"\n--- alpha = {alpha} (mean +/- std over seeds) ---")
            summary = sub.groupby(group_cols)[metric_cols].agg(["mean", "std"])
            summary.columns = [f"{m}_{s}" for m, s in summary.columns]
            print(summary.to_string())
    elif metric_cols and group_cols:
        print(f"\nMetrics (mean over seeds):")
        summary = df.groupby(group_cols)[metric_cols].agg(["mean", "std"])
        summary.columns = [f"{m}_{s}" for m, s in summary.columns]
        print(summary.to_string())

    return df


def aggregate_results(
    hc_dir: str = HC_RESULTS_DEFAULT,
    kn_dir: str = KN_RESULTS_DEFAULT,
    output_dir: str = "results/final",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate all per-run CSVs into single files for each experiment."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    hc_df = pd.DataFrame()
    kn_df = pd.DataFrame()

    hc_csvs = sorted(Path(hc_dir).rglob("stage_results.csv"))
    if hc_csvs:
        hc_df = pd.concat([pd.read_csv(c) for c in hc_csvs], ignore_index=True)
        hc_path = Path(hc_dir) / "stage_results_all.csv"
        hc_df.to_csv(hc_path, index=False)
        print(f"Healthcare: {len(hc_df)} rows -> {hc_path}")

    kn_csvs = sorted(Path(kn_dir).rglob("stage_results.csv"))
    if kn_csvs:
        kn_df = pd.concat([pd.read_csv(c) for c in kn_csvs], ignore_index=True)
        kn_path = Path(kn_dir) / "stage_results_all.csv"
        kn_df.to_csv(kn_path, index=False)
        print(f"Knapsack:   {len(kn_df)} rows -> {kn_path}")

    return hc_df, kn_df
