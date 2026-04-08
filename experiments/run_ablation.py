#!/usr/bin/env python3
"""
Ablation experiment runner — standalone script for local execution.

Runs up to six ablation experiments and saves results to results/ablation/:
  1. No continuation — disable continuation=True for all methods
  2. Single lambda   — run only lambda=0.5 to remove cross-stage effects
  3. Fairness compare — run MAD vs Gap vs Atkinson fairness measures
  4. Predictor architecture — linear, MLP, wider MLP, ResNet, FT-Transformer
  5. Learning rate × steps grid
  6. Optimizer — SGD vs Adam

Usage:
  # Run all experiments with n=1000
  python run_ablation.py --n-sample 1000 --experiments 1 2 3 4 5 6

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
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fair_dfl.runner import run_experiment_unified  # noqa: E402
from fair_dfl.training.method_spec import resolve_method_spec  # noqa: E402
from .configs import (  # noqa: E402
    ALL_METHOD_CONFIGS,
    DEFAULT_TRAIN_CFG,
    make_task_cfg,
    compute_full_batch_size,
    describe_method,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATA_CSV = str(REPO_ROOT / "data" / "data_processed.csv")
DEFAULT_RESULTS_DIR = str(REPO_ROOT / "results" / "ablation")
ABLATION_VAL_FRACTION = 0.0

ABLATION_METHODS = ["FPTO", "FDFL", "WS-equal", "WS-dec", "MGDA", "PCGrad"]

# ---------------------------------------------------------------------------
# Hyperparameter grids for Exp 4-6
# ---------------------------------------------------------------------------
ARCHITECTURE_CONFIGS = {
    "linear": {"arch": "linear"},
    "mlp_64x2": {"arch": "mlp", "hidden_dim": 64, "n_layers": 2},
    "mlp_128x3": {"arch": "mlp", "hidden_dim": 128, "n_layers": 3},
    "resnet": {"arch": "resnet_tabular", "hidden_dim": 128, "n_blocks": 3, "dropout": 0.1},
    "ft_transformer": {"arch": "ft_transformer", "d_token": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.1},
}

LR_STEPS_GRID = [
    {"lr": lr, "steps_per_lambda": steps}
    for lr in [0.0001, 0.0005, 0.001, 0.005]
    for steps in [50, 100, 200]
]

OPTIMIZER_CONFIGS = ["sgd", "adam"]


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
            summary = (
                f"    Done in {elapsed:.1f}s | "
                f"regret={stage_df['test_regret'].mean():.4f}, "
                f"fairness={stage_df['test_fairness'].mean():.6f}, "
                f"mse={stage_df['test_pred_mse'].mean():.4f}"
            )
            if "test_regret_normalized" in stage_df.columns:
                summary += f", norm_regret={stage_df['test_regret_normalized'].mean():.4f}"
            print(summary)

        all_stage.append(stage_df)
        all_iter.append(iter_df)

    stage_combined = pd.concat(all_stage, ignore_index=True) if all_stage else pd.DataFrame()
    iter_combined = pd.concat(all_iter, ignore_index=True) if all_iter else pd.DataFrame()
    return stage_combined, iter_combined


# ---------------------------------------------------------------------------
# Shared helper: build no-continuation method configs
# ---------------------------------------------------------------------------
def _make_nocont_methods() -> dict:
    methods = {}
    for name in ABLATION_METHODS:
        cfg = copy.deepcopy(ALL_METHOD_CONFIGS[name])
        cfg["continuation"] = False
        methods[name] = cfg
    return methods


# =========================================================================
# Experiments 1-3 (original)
# =========================================================================

def run_exp1_no_continuation(
    alpha: float, data_csv: str, n_sample: int, batch_size: int,
    results_dir: str, verbose: bool = True,
) -> None:
    """Experiment 1: Disable continuation for all methods."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: No Continuation (all methods reset per lambda stage)")
    print("=" * 70)

    methods = _make_nocont_methods()

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
    stage_df["fairness_type"] = "mad"
    iter_df["fairness_type"] = "mad"

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

    methods = _make_nocont_methods()

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
    stage_df["fairness_type"] = "mad"
    iter_df["fairness_type"] = "mad"

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

        methods = _make_nocont_methods()

        train_cfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
        train_cfg["lambdas"] = [0.0, 0.5]
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


# =========================================================================
# Experiments 4-6 (hyperparameter ablations)
# =========================================================================

def run_exp4_architecture(
    alpha: float, data_csv: str, n_sample: int, batch_size: int,
    results_dir: str, verbose: bool = True,
) -> None:
    """Experiment 4: Predictor architecture comparison."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Predictor Architecture")
    print("=" * 70)

    all_stage = []
    all_iter = []

    for arch_name, model_cfg in ARCHITECTURE_CONFIGS.items():
        print(f"\n  --- Architecture: {arch_name} ---")

        methods = _make_nocont_methods()
        train_cfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
        train_cfg["lambdas"] = [0.5]
        train_cfg["log_every"] = 1
        train_cfg["model"] = model_cfg
        task_cfg = make_task_cfg(data_csv, n_sample, alpha, val_fraction=ABLATION_VAL_FRACTION)

        stage_df, iter_df = _run_experiment(
            method_configs=methods,
            train_cfg=train_cfg,
            task_cfg=task_cfg,
            batch_size=batch_size,
            experiment_name=f"arch_{arch_name}",
            verbose=verbose,
        )

        stage_df["alpha_fair"] = alpha
        iter_df["alpha_fair"] = alpha
        stage_df["hp_config"] = arch_name
        iter_df["hp_config"] = arch_name
        all_stage.append(stage_df)
        all_iter.append(iter_df)

    combined_stage = pd.concat(all_stage, ignore_index=True) if all_stage else pd.DataFrame()
    combined_iter = pd.concat(all_iter, ignore_index=True) if all_iter else pd.DataFrame()

    stage_path = os.path.join(results_dir, "stage_hp_architecture.csv")
    iter_path = os.path.join(results_dir, "iter_hp_architecture.csv")
    combined_stage.to_csv(stage_path, index=False)
    combined_iter.to_csv(iter_path, index=False)
    print(f"\n  Saved: {stage_path} ({len(combined_stage)} rows)")
    print(f"  Saved: {iter_path} ({len(combined_iter)} rows)")


def run_exp5_lr_steps(
    alpha: float, data_csv: str, n_sample: int, batch_size: int,
    results_dir: str, verbose: bool = True,
) -> None:
    """Experiment 5: Learning rate × steps grid."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Learning Rate × Steps Grid")
    print("=" * 70)

    all_stage = []
    all_iter = []

    for hp in LR_STEPS_GRID:
        hp_label = f"lr={hp['lr']}_steps={hp['steps_per_lambda']}"
        print(f"\n  --- {hp_label} ---")

        methods = _make_nocont_methods()
        train_cfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
        train_cfg["lambdas"] = [0.5]
        train_cfg["log_every"] = 1
        train_cfg["lr"] = hp["lr"]
        train_cfg["steps_per_lambda"] = hp["steps_per_lambda"]
        task_cfg = make_task_cfg(data_csv, n_sample, alpha, val_fraction=ABLATION_VAL_FRACTION)

        stage_df, iter_df = _run_experiment(
            method_configs=methods,
            train_cfg=train_cfg,
            task_cfg=task_cfg,
            batch_size=batch_size,
            experiment_name=f"lr_steps_{hp_label}",
            verbose=verbose,
        )

        stage_df["alpha_fair"] = alpha
        iter_df["alpha_fair"] = alpha
        stage_df["hp_config"] = hp_label
        iter_df["hp_config"] = hp_label
        all_stage.append(stage_df)
        all_iter.append(iter_df)

    combined_stage = pd.concat(all_stage, ignore_index=True) if all_stage else pd.DataFrame()
    combined_iter = pd.concat(all_iter, ignore_index=True) if all_iter else pd.DataFrame()

    stage_path = os.path.join(results_dir, "stage_hp_lr_steps.csv")
    iter_path = os.path.join(results_dir, "iter_hp_lr_steps.csv")
    combined_stage.to_csv(stage_path, index=False)
    combined_iter.to_csv(iter_path, index=False)
    print(f"\n  Saved: {stage_path} ({len(combined_stage)} rows)")
    print(f"  Saved: {iter_path} ({len(combined_iter)} rows)")


def run_exp6_optimizer(
    alpha: float, data_csv: str, n_sample: int, batch_size: int,
    results_dir: str, verbose: bool = True,
) -> None:
    """Experiment 6: Optimizer comparison (SGD vs Adam)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Optimizer (SGD vs Adam)")
    print("=" * 70)

    all_stage = []
    all_iter = []

    for opt in OPTIMIZER_CONFIGS:
        print(f"\n  --- Optimizer: {opt.upper()} ---")

        methods = _make_nocont_methods()
        train_cfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
        train_cfg["lambdas"] = [0.5]
        train_cfg["log_every"] = 1
        train_cfg["optimizer"] = opt
        task_cfg = make_task_cfg(data_csv, n_sample, alpha, val_fraction=ABLATION_VAL_FRACTION)

        stage_df, iter_df = _run_experiment(
            method_configs=methods,
            train_cfg=train_cfg,
            task_cfg=task_cfg,
            batch_size=batch_size,
            experiment_name=f"opt_{opt}",
            verbose=verbose,
        )

        stage_df["alpha_fair"] = alpha
        iter_df["alpha_fair"] = alpha
        stage_df["hp_config"] = opt
        iter_df["hp_config"] = opt
        all_stage.append(stage_df)
        all_iter.append(iter_df)

    combined_stage = pd.concat(all_stage, ignore_index=True) if all_stage else pd.DataFrame()
    combined_iter = pd.concat(all_iter, ignore_index=True) if all_iter else pd.DataFrame()

    stage_path = os.path.join(results_dir, "stage_hp_optimizer.csv")
    iter_path = os.path.join(results_dir, "iter_hp_optimizer.csv")
    combined_stage.to_csv(stage_path, index=False)
    combined_iter.to_csv(iter_path, index=False)
    print(f"\n  Saved: {stage_path} ({len(combined_stage)} rows)")
    print(f"  Saved: {iter_path} ({len(combined_iter)} rows)")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run ablation experiments for fair DFL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiments", nargs="+", type=int, default=[1, 2, 3],
        choices=[1, 2, 3, 4, 5, 6],
        help="Which experiments to run (default: 1 2 3).",
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
            print(f"  Exp 3: Fairness compare — {n_methods} methods x 2 lambdas x 3 fairness types x {n_seeds} seeds")
        if 4 in args.experiments:
            n_arch = len(ARCHITECTURE_CONFIGS)
            print(f"  Exp 4: Architecture — {n_methods} methods x {n_arch} architectures x {n_seeds} seeds")
        if 5 in args.experiments:
            n_hp = len(LR_STEPS_GRID)
            print(f"  Exp 5: LR x Steps — {n_methods} methods x {n_hp} HP configs x {n_seeds} seeds")
        if 6 in args.experiments:
            n_opt = len(OPTIMIZER_CONFIGS)
            print(f"  Exp 6: Optimizer — {n_methods} methods x {n_opt} optimizers x {n_seeds} seeds")
        return

    t_total = time.time()

    _EXP_FUNCS = {
        1: ("Experiment 1", run_exp1_no_continuation),
        2: ("Experiment 2", run_exp2_single_lambda),
        3: ("Experiment 3", run_exp3_fairness_compare),
        4: ("Experiment 4", run_exp4_architecture),
        5: ("Experiment 5", run_exp5_lr_steps),
        6: ("Experiment 6", run_exp6_optimizer),
    }

    for exp_id in sorted(args.experiments):
        label, func = _EXP_FUNCS[exp_id]
        try:
            func(
                alpha=args.alpha, data_csv=args.data_csv,
                n_sample=args.n_sample, batch_size=batch_size,
                results_dir=args.results_dir,
            )
        except Exception as e:
            print(f"\n  ERROR in {label}: {e}")
            traceback.print_exc()

    total_elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"ALL DONE in {total_elapsed:.1f}s")
    print(f"Results in: {args.results_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
