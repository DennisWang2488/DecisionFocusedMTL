#!/usr/bin/env python3
"""
Ablation experiment runner — standalone script for local execution.

Runs three ablation experiments and saves results to results/ablation/:
  1. No continuation — disable continuation=True for all methods
  2. Single lambda   — run only lambda=0.5 to remove cross-stage effects
  3. Fairness compare — run MAD vs Gap vs Atkinson fairness measures

Usage:
  # Run all three experiments with n=1000
  python run_ablation.py --n-sample 1000

  # Run only experiment 1
  python run_ablation.py --experiments 1 --n-sample 1000

  # Quick test
  python run_ablation.py --experiments 2 --n-sample 500

  # Dry run
  python run_ablation.py --dry-run

  # Custom alpha
  python run_ablation.py --alpha 0.5 --n-sample 1000
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from fair_dfl.runner import run_experiment_unified  # noqa: E402
from fair_dfl.training.method_spec import resolve_method_spec  # noqa: E402
from configs import (  # noqa: E402
    ALL_METHOD_CONFIGS,
    DEFAULT_TRAIN_CFG,
    make_task_cfg,
    compute_full_batch_size,
    describe_method,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATA_CSV = str(SCRIPT_DIR / "data" / "data_processed.csv")
DEFAULT_RESULTS_DIR = str(SCRIPT_DIR / "results" / "ablation")
ABLATION_VAL_FRACTION = 0.0

ABLATION_METHODS = ["FPTO", "FDFL", "FFO", "WS-equal", "WS-dec", "MGDA", "PCGrad"]


def _run_experiment(
    method_configs: dict,
    train_cfg: dict,
    task_cfg: dict,
    batch_size: int,
    experiment_name: str,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one experiment (all methods) and return (stage_df, iter_df)."""
    _method_to_config = {name.lower(): name for name in method_configs}

    all_stage = []
    all_iter = []
    total = len(method_configs)

    for idx, (name, spec) in enumerate(method_configs.items(), 1):
        desc = describe_method(name, spec)
        if verbose:
            print(f"  [{idx}/{total}] {name} ({desc})")

        cfg_copy = copy.deepcopy(train_cfg)
        cfg_copy["batch_size"] = batch_size

        cfg = {"task": task_cfg, "training": cfg_copy}
        single_method = {name: spec}

        t0 = time.time()
        stage_df, iter_df = run_experiment_unified(cfg, method_configs=single_method)
        elapsed = time.time() - t0

        stage_df["config_name"] = name
        iter_df["config_name"] = name

        if verbose and not stage_df.empty and "test_regret" in stage_df.columns:
            print(
                f"    Done in {elapsed:.1f}s | "
                f"regret={stage_df['test_regret'].mean():.4f}, "
                f"fairness={stage_df['test_fairness'].mean():.6f}, "
                f"mse={stage_df['test_pred_mse'].mean():.4f}"
            )

        all_stage.append(stage_df)
        all_iter.append(iter_df)

    stage_combined = pd.concat(all_stage, ignore_index=True) if all_stage else pd.DataFrame()
    iter_combined = pd.concat(all_iter, ignore_index=True) if all_iter else pd.DataFrame()
    return stage_combined, iter_combined


def run_exp1_no_continuation(
    alpha: float, data_csv: str, n_sample: int, batch_size: int,
    results_dir: str, verbose: bool = True,
) -> None:
    """Experiment 1: Disable continuation for all methods."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: No Continuation (all methods reset per lambda stage)")
    print("=" * 70)

    methods = {}
    for name in ABLATION_METHODS:
        cfg = copy.deepcopy(ALL_METHOD_CONFIGS[name])
        cfg["continuation"] = False
        methods[name] = cfg

    train_cfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
    train_cfg["log_every"] = 1
    task_cfg = make_task_cfg(data_csv, n_sample, alpha, val_fraction=ABLATION_VAL_FRACTION)

    stage_df, iter_df = _run_experiment(
        method_configs=methods,
        train_cfg=train_cfg,
        task_cfg=task_cfg,
        batch_size=batch_size,
        experiment_name="no_continuation",
        verbose=verbose,
    )

    stage_df["alpha_fair"] = alpha
    iter_df["alpha_fair"] = alpha

    stage_path = os.path.join(results_dir, "stage_no_continuation.csv")
    iter_path = os.path.join(results_dir, "iter_no_continuation.csv")
    stage_df.to_csv(stage_path, index=False)
    iter_df.to_csv(iter_path, index=False)
    print(f"\n  Saved: {stage_path} ({len(stage_df)} rows)")
    print(f"  Saved: {iter_path} ({len(iter_df)} rows)")


def run_exp2_single_lambda(
    alpha: float, data_csv: str, n_sample: int, batch_size: int,
    results_dir: str, verbose: bool = True,
) -> None:
    """Experiment 2: Single lambda=0.5, no continuation."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Single Lambda = 0.5 (no cross-stage effects)")
    print("=" * 70)

    methods = {}
    for name in ABLATION_METHODS:
        cfg = copy.deepcopy(ALL_METHOD_CONFIGS[name])
        cfg["continuation"] = False
        methods[name] = cfg

    train_cfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
    train_cfg["lambdas"] = [0.5]
    train_cfg["log_every"] = 1
    task_cfg = make_task_cfg(data_csv, n_sample, alpha, val_fraction=ABLATION_VAL_FRACTION)

    stage_df, iter_df = _run_experiment(
        method_configs=methods,
        train_cfg=train_cfg,
        task_cfg=task_cfg,
        batch_size=batch_size,
        experiment_name="single_lambda",
        verbose=verbose,
    )

    stage_df["alpha_fair"] = alpha
    iter_df["alpha_fair"] = alpha

    stage_path = os.path.join(results_dir, "stage_single_lambda.csv")
    iter_path = os.path.join(results_dir, "iter_single_lambda.csv")
    stage_df.to_csv(stage_path, index=False)
    iter_df.to_csv(iter_path, index=False)
    print(f"\n  Saved: {stage_path} ({len(stage_df)} rows)")
    print(f"  Saved: {iter_path} ({len(iter_df)} rows)")


def run_exp3_fairness_compare(
    alpha: float, data_csv: str, n_sample: int, batch_size: int,
    results_dir: str, verbose: bool = True,
) -> None:
    """Experiment 3: Compare MAD vs Gap vs Atkinson fairness measures."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Fairness Measure Comparison (MAD / Gap / Atkinson)")
    print("=" * 70)

    fairness_types = ["mad", "gap", "atkinson"]

    for ft in fairness_types:
        print(f"\n  --- Fairness type: {ft.upper()} ---")

        methods = {}
        for name in ABLATION_METHODS:
            cfg = copy.deepcopy(ALL_METHOD_CONFIGS[name])
            cfg["continuation"] = False
            methods[name] = cfg

        train_cfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
        train_cfg["lambdas"] = [0.5]
        train_cfg["log_every"] = 1
        task_cfg = make_task_cfg(
            data_csv, n_sample, alpha, fairness_type=ft, val_fraction=ABLATION_VAL_FRACTION
        )

        stage_df, iter_df = _run_experiment(
            method_configs=methods,
            train_cfg=train_cfg,
            task_cfg=task_cfg,
            batch_size=batch_size,
            experiment_name=f"fairness_{ft}",
            verbose=verbose,
        )

        stage_df["alpha_fair"] = alpha
        iter_df["alpha_fair"] = alpha
        stage_df["fairness_type"] = ft
        iter_df["fairness_type"] = ft

        stage_path = os.path.join(results_dir, f"stage_fairness_{ft}.csv")
        iter_path = os.path.join(results_dir, f"iter_fairness_{ft}.csv")
        stage_df.to_csv(stage_path, index=False)
        iter_df.to_csv(iter_path, index=False)
        print(f"    Saved: {stage_path} ({len(stage_df)} rows)")
        print(f"    Saved: {iter_path} ({len(iter_df)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation experiments for fair DFL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiments", nargs="+", type=int, default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Which experiments to run (default: all three).",
    )
    parser.add_argument(
        "--alpha", type=float, default=2.0,
        help="Alpha-fairness value (default: 2.0).",
    )
    parser.add_argument(
        "--n-sample", type=int, default=1000,
        help="Number of patients (0=all, default: 1000).",
    )
    parser.add_argument(
        "--data-csv", type=str, default=DEFAULT_DATA_CSV,
        help="Path to data_processed.csv.",
    )
    parser.add_argument(
        "--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
        help="Output directory for CSVs.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show configuration without running.",
    )

    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # Compute batch size
    batch_size = compute_full_batch_size(
        args.data_csv, args.n_sample, val_fraction=ABLATION_VAL_FRACTION
    )

    print("=" * 70)
    print("ABLATION EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Experiments:    {args.experiments}")
    print(f"Methods:        {ABLATION_METHODS}")
    print(f"Alpha:          {args.alpha}")
    print(f"N-sample:       {args.n_sample}")
    print(f"Batch size:     {batch_size}")
    print(f"Seeds:          {DEFAULT_TRAIN_CFG['seeds']}")
    print(f"Lambdas:        {DEFAULT_TRAIN_CFG['lambdas']}")
    print(f"Steps/lambda:   {DEFAULT_TRAIN_CFG['steps_per_lambda']}")
    print(f"Results dir:    {args.results_dir}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Would run:")
        n_lam = len(DEFAULT_TRAIN_CFG["lambdas"])
        n_seeds = len(DEFAULT_TRAIN_CFG["seeds"])
        n_methods = len(ABLATION_METHODS)
        if 1 in args.experiments:
            print(f"  Exp 1: No continuation — {n_methods} methods x {n_lam} lambdas x {n_seeds} seeds")
        if 2 in args.experiments:
            print(f"  Exp 2: Single lambda=0.5 — {n_methods} methods x 1 lambda x {n_seeds} seeds")
        if 3 in args.experiments:
            print(f"  Exp 3: Fairness compare — {n_methods} methods x 3 fairness types x {n_seeds} seeds")
        return

    t_total = time.time()

    if 1 in args.experiments:
        try:
            run_exp1_no_continuation(
                alpha=args.alpha, data_csv=args.data_csv,
                n_sample=args.n_sample, batch_size=batch_size,
                results_dir=args.results_dir,
            )
        except Exception as e:
            print(f"\n  ERROR in Experiment 1: {e}")
            traceback.print_exc()

    if 2 in args.experiments:
        try:
            run_exp2_single_lambda(
                alpha=args.alpha, data_csv=args.data_csv,
                n_sample=args.n_sample, batch_size=batch_size,
                results_dir=args.results_dir,
            )
        except Exception as e:
            print(f"\n  ERROR in Experiment 2: {e}")
            traceback.print_exc()

    if 3 in args.experiments:
        try:
            run_exp3_fairness_compare(
                alpha=args.alpha, data_csv=args.data_csv,
                n_sample=args.n_sample, batch_size=batch_size,
                results_dir=args.results_dir,
            )
        except Exception as e:
            print(f"\n  ERROR in Experiment 3: {e}")
            traceback.print_exc()

    total_elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"ALL DONE in {total_elapsed:.1f}s")
    print(f"Results in: {args.results_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
