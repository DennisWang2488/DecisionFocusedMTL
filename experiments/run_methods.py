#!/usr/bin/env python3
"""
Standalone experiment runner — run any subset of methods and append to existing CSVs.

Usage examples:
  # Run PLG and FPLG, append to existing results
  python run_methods.py --methods PLG FPLG

  # Run only MGDA at alpha=2.0
  python run_methods.py --methods MGDA --alphas 2.0

  # Run everything from scratch, overwrite existing results
  python run_methods.py --all --overwrite

  # Quick test with 500 samples
  python run_methods.py --methods PLG --n-sample 500

  # Dry run — show what would be run without running it
  python run_methods.py --methods PLG FPLG --dry-run

  # Custom output directory
  python run_methods.py --methods PLG --results-dir ./results_v2
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — make sure src/ is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fair_dfl.runner import run_experiment_unified  # noqa: E402
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
DEFAULT_RESULTS_DIR = str(REPO_ROOT / "results")

# ---------------------------------------------------------------------------
# Run metadata — attached to every result row for provenance tracking
# ---------------------------------------------------------------------------
_RUN_ID = str(uuid.uuid4())[:12]
_TIMESTAMP_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _run_metadata() -> dict:
    """Return metadata dict to attach to every result row."""
    return {
        "run_id": _RUN_ID,
        "timestamp_utc": _TIMESTAMP_UTC,
        "git_commit": _git_commit_hash(),
        "python_version": platform.python_version(),
        "has_validation": True,  # overridden per-task if val_fraction=0
    }


def run_single(alpha: float, method_name: str, method_spec: dict,
               base_train_cfg: dict, data_csv: str, n_sample: int,
               batch_size: int, fairness_type: str = "mad", verbose: bool = True):
    """Run a single (alpha, fairness_type, method) combination via the unified training loop."""
    task_cfg = make_task_cfg(data_csv, n_sample, alpha, fairness_type=fairness_type)
    train_cfg = copy.deepcopy(base_train_cfg)
    train_cfg["batch_size"] = batch_size

    cfg = {"task": task_cfg, "training": train_cfg}
    method_configs = {method_name: method_spec}

    t0 = time.time()
    stage_df, iter_df = run_experiment_unified(cfg, method_configs=method_configs)
    elapsed = time.time() - t0

    # Tag results with experiment context and provenance metadata
    meta = _run_metadata()
    meta["has_validation"] = task_cfg.get("val_fraction", 0.2) > 0
    for df in (stage_df, iter_df):
        df["config_name"] = method_name
        df["alpha_fair"] = alpha
        df["fairness_type"] = fairness_type
        for k, v in meta.items():
            df[k] = v

    if verbose:
        n_rows = len(stage_df)
        if n_rows > 0 and "test_regret" in stage_df.columns:
            summary = (
                f"    Done in {elapsed:.1f}s | {n_rows} stage rows | "
                f"mean regret={stage_df['test_regret'].mean():.4f}, "
                f"fairness={stage_df['test_fairness'].mean():.6f}, "
                f"pred_mse={stage_df['test_pred_mse'].mean():.4f}"
            )
            if "test_regret_normalized" in stage_df.columns:
                summary += f", norm_regret={stage_df['test_regret_normalized'].mean():.4f}"
            print(summary)
        else:
            print(f"    Done in {elapsed:.1f}s | {n_rows} stage rows")

    return stage_df, iter_df, elapsed


def load_existing_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def append_and_save(existing_df: pd.DataFrame, new_df: pd.DataFrame,
                    path: str, dedup_cols: list[str] | None = None) -> pd.DataFrame:
    """Append new results, optionally deduplicate, and save."""
    if existing_df.empty:
        combined = new_df
    elif new_df.empty:
        combined = existing_df
    else:
        combined = pd.concat([existing_df, new_df], ignore_index=True)

    shared_dedup_cols = []
    if dedup_cols:
        shared_dedup_cols = [c for c in dedup_cols if c in new_df.columns and c in existing_df.columns]

    if shared_dedup_cols and not combined.empty:
        # Remove old rows for (alpha, method, seed, lambda) combos present in new data
        existing_keys = set(existing_df[shared_dedup_cols].apply(tuple, axis=1))
        new_keys = set(new_df[shared_dedup_cols].apply(tuple, axis=1))
        overlap = existing_keys & new_keys
        if overlap and not existing_df.empty:
            # Keep only the NEW version of overlapping rows
            mask = ~existing_df[shared_dedup_cols].apply(tuple, axis=1).isin(new_keys)
            combined = pd.concat([existing_df[mask], new_df], ignore_index=True)

    combined.to_csv(path, index=False)
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Run fair DFL methods and append results to existing CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--methods", nargs="+", type=str,
        help="Method names to run (e.g., PLG FPLG MGDA). Case-insensitive.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all methods in the registry.",
    )
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=[0.5, 2.0],
        help="Alpha-fairness values (default: 0.5 2.0).",
    )
    parser.add_argument(
        "--n-sample", type=int, default=0,
        help="Number of patients (0 = all, default: 0).",
    )
    parser.add_argument(
        "--data-csv", type=str, default=DEFAULT_DATA_CSV,
        help="Path to data_processed.csv.",
    )
    parser.add_argument(
        "--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
        help="Directory for stage_results_full.csv and iter_logs_full.csv.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing CSVs instead of appending.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be run without executing.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Override seeds (default: uses base_train_cfg seeds [11,22,33]).",
    )
    parser.add_argument(
        "--lambdas", nargs="+", type=float, default=None,
        help="Override lambda values (default: [0.0, 0.05, 0.2, 0.5]).",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Override steps_per_lambda.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device (cuda/cpu).",
    )
    parser.add_argument(
        "--fairness-type", type=str, default="mad",
        choices=["mad", "gap", "atkinson"],
        help="Prediction-side fairness metric (default: mad).",
    )
    parser.add_argument(
        "--list-methods", action="store_true",
        help="List all available methods and exit.",
    )

    args = parser.parse_args()

    # --list-methods
    if args.list_methods:
        print(f"{len(ALL_METHOD_CONFIGS)} methods available\n")
        print("Available methods:")
        for name, spec in ALL_METHOD_CONFIGS.items():
            desc = describe_method(name, spec)
            print(f"  {name:15s} -> {desc}")
        return

    # Resolve methods
    if args.all:
        selected = list(ALL_METHOD_CONFIGS.keys())
    elif args.methods:
        # Case-insensitive lookup
        name_map = {k.lower(): k for k in ALL_METHOD_CONFIGS}
        selected = []
        for m in args.methods:
            key = m.lower()
            if key not in name_map:
                print(f"ERROR: Unknown method '{m}'. Use --list-methods to see options.")
                sys.exit(1)
            selected.append(name_map[key])
    else:
        parser.error("Specify --methods or --all. Use --list-methods to see options.")
        return

    # Build training config
    train_cfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
    if args.seeds:
        train_cfg["seeds"] = args.seeds
    if args.lambdas:
        train_cfg["lambdas"] = args.lambdas
    if args.steps:
        train_cfg["steps_per_lambda"] = args.steps
    if args.device:
        train_cfg["device"] = args.device

    # Paths
    os.makedirs(args.results_dir, exist_ok=True)
    stage_csv = os.path.join(args.results_dir, "stage_results_full.csv")
    iter_csv = os.path.join(args.results_dir, "iter_logs_full.csv")

    # Dry run
    total_runs = len(args.alphas) * len(selected)
    print("=" * 60)
    print(f"Methods:        {selected}")
    print(f"Alphas:         {args.alphas}")
    print(f"Fairness type:  {args.fairness_type}")
    print(f"Lambdas:        {train_cfg['lambdas']}")
    print(f"Seeds:          {train_cfg['seeds']}")
    print(f"Steps:          {train_cfg['steps_per_lambda']}")
    print(f"N-sample:       {args.n_sample} (0=all)")
    print(f"Device:         {train_cfg['device']}")
    print(f"Results:        {args.results_dir}")
    print(f"Overwrite:      {args.overwrite}")
    print(f"Total runs:     {total_runs}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would execute:")
        for alpha in args.alphas:
            for name in selected:
                spec = ALL_METHOD_CONFIGS[name]
                desc = describe_method(name, spec)
                print(f"  alpha={alpha}, fairness={args.fairness_type}, method={name} ({desc})")
        return

    # Compute batch size
    batch_size = compute_full_batch_size(args.data_csv, args.n_sample)
    print(f"Batch size (full training set): {batch_size}")

    # Load existing results
    if args.overwrite:
        existing_stage = pd.DataFrame()
        existing_iter = pd.DataFrame()
    else:
        existing_stage = load_existing_csv(stage_csv)
        existing_iter = load_existing_csv(iter_csv)
        if not existing_stage.empty:
            existing_methods = existing_stage["config_name"].unique().tolist()
            print(f"Existing results: {len(existing_stage)} stage rows, "
                  f"methods: {existing_methods}")

    # Run experiments
    new_stage_dfs = []
    new_iter_dfs = []
    errors = []
    run_idx = 0

    for alpha in args.alphas:
        for name in selected:
            run_idx += 1
            spec = ALL_METHOD_CONFIGS[name]
            desc = describe_method(name, spec)
            print(f"\n[{run_idx}/{total_runs}] alpha={alpha}, method={name} ({desc})")

            try:
                stage_df, iter_df, elapsed = run_single(
                    alpha=alpha,
                    method_name=name,
                    method_spec=spec,
                    base_train_cfg=train_cfg,
                    data_csv=args.data_csv,
                    n_sample=args.n_sample,
                    batch_size=batch_size,
                    fairness_type=args.fairness_type,
                )
                new_stage_dfs.append(stage_df)
                new_iter_dfs.append(iter_df)
            except Exception as e:
                errors.append({"alpha": alpha, "method": name, "error": str(e)})
                print(f"    ERROR: {e}")
                traceback.print_exc()

    # Combine new results
    new_stages = pd.concat(new_stage_dfs, ignore_index=True) if new_stage_dfs else pd.DataFrame()
    new_iters = pd.concat(new_iter_dfs, ignore_index=True) if new_iter_dfs else pd.DataFrame()

    # Append and save
    if not new_stages.empty:
        stage_dedup = ["config_name", "alpha_fair", "fairness_type", "seed", "lambda"]
        iter_dedup = ["config_name", "alpha_fair", "fairness_type", "seed", "stage_idx", "iter"]

        # For stage results, check which dedup cols actually exist
        stage_dedup_valid = [c for c in stage_dedup if c in new_stages.columns]
        combined_stage = append_and_save(existing_stage, new_stages, stage_csv,
                                         dedup_cols=stage_dedup_valid if stage_dedup_valid else None)
        print(f"\nStage results saved: {stage_csv} ({len(combined_stage)} total rows)")

    if not new_iters.empty:
        iter_dedup_valid = ["config_name", "alpha_fair", "fairness_type", "seed", "stage_idx", "iter"]
        iter_dedup_valid = [c for c in iter_dedup_valid if c in new_iters.columns]
        combined_iter = append_and_save(existing_iter, new_iters, iter_csv,
                                        dedup_cols=iter_dedup_valid if iter_dedup_valid else None)
        print(f"Iter logs saved:     {iter_csv} ({len(combined_iter)} total rows)")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Done: {len(new_stages)} new stage rows, {len(new_iters)} new iter rows, "
          f"{len(errors)} errors")
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  alpha={e['alpha']}, method={e['method']}: {e['error']}")


if __name__ == "__main__":
    main()
