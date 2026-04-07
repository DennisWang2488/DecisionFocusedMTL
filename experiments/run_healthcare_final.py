#!/usr/bin/env python3
"""Final healthcare experiment runner — full grid for INFORMS JoC submission.

Experimental grid:
  alpha        : {0.5, 2.0}
  hidden_dim   : {64, 128}
  seeds        : [11, 22, 33, 44, 55]
  methods      : FPTO (lambda=0/0.5/1/5), SAA, WDRO,
                 FDFL-Scal (lambda=0/0.5/1/5), FDFL-PCGrad, FDFL-MGDA, FDFL-CAGrad

Note: FPTO(lambda=0) = PTO, FDFL-Scal(lambda=0) = DFL.
No validation split — 50/50 train/test.

Usage:
  python experiments/run_healthcare_final.py
  python experiments/run_healthcare_final.py --dry-run
  python experiments/run_healthcare_final.py --methods FPTO SAA --alphas 0.5
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from fair_dfl.runner import run_experiment_unified  # noqa: E402
from experiments.configs import (  # noqa: E402
    ALL_METHOD_CONFIGS,
    DEFAULT_TRAIN_CFG,
    compute_full_batch_size,
    describe_method,
)

# ======================================================================
# Experiment constants
# ======================================================================
SEEDS = [11, 22, 33, 44, 55]
ALPHAS = [0.5, 2.0]
HIDDEN_DIMS = [64]
STEPS_PER_LAMBDA = 70
DATA_CSV = str(REPO_ROOT / "data" / "data_processed.csv")
RESULTS_BASE = str(REPO_ROOT / "results" / "final" / "healthcare")
LAMBDAS_SWEEP = [0.0, 0.5, 1.0, 5.0]

# -----------------------------------------------------------------------
# Method grid
#
# FPTO with lambda=0 is PTO (predict-then-optimize, no fairness).
# FDFL-Scal with lambda=0 is DFL (decision-focused, no pred fairness).
# Lambda-sweep methods run all 4 lambdas in one shot (continuation mode).
# MOO methods ignore lambda (set to 0).
# -----------------------------------------------------------------------
METHOD_GRID = {
    # --- Two-stage (predict-then-optimize family) ---
    "FPTO":         {"config": "FPTO",     "lambdas": LAMBDAS_SWEEP},
    # --- Data-driven optimization baselines ---
    "SAA":          {"config": "SAA",      "lambdas": [0.0]},
    "WDRO":         {"config": "WDRO",     "lambdas": [0.0]},
    # --- Integrated: scalarized (DFL + fairness via lambda) ---
    "FDFL-Scal":    {"config": "FPLG",     "lambdas": LAMBDAS_SWEEP},
    # --- Integrated + MOO (no lambda needed) ---
    "FDFL-PCGrad":  {"config": "PCGrad",   "lambdas": [0.0]},
    "FDFL-MGDA":    {"config": "MGDA",     "lambdas": [0.0]},
    "FDFL-CAGrad":  {"config": "CAGrad",   "lambdas": [0.0]},
}


def make_task_cfg_no_val(
    data_csv: str, n_sample: int, alpha_fair: float,
    fairness_type: str = "mad",
) -> dict:
    """Task config with NO validation split — 50/50 train/test."""
    return {
        "name": "medical_resource_allocation",
        "data_csv": data_csv,
        "n_sample": n_sample,
        "data_seed": 42,
        "split_seed": 2,
        "test_fraction": 0.5,
        "val_fraction": 0.0,
        "alpha_fair": alpha_fair,
        "budget": -1,
        "budget_rho": 0.35,
        "decision_mode": "group",
        "fairness_type": fairness_type,
    }


def compute_train_size(data_csv: str, n_sample: int) -> int:
    """Compute training set size for full-batch training (no val split)."""
    df = pd.read_csv(data_csv)
    n_total = n_sample if (0 < n_sample < len(df)) else len(df)
    n_test = int(round(0.5 * n_total))
    return n_total - n_test


def _result_path(base_dir: str, method_label: str, alpha: float,
                 hidden_dim: int, seed: int) -> Path:
    tag = f"alpha_{alpha}_hd_{hidden_dim}"
    d = Path(base_dir) / method_label / tag / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _done_marker(result_dir: Path) -> Path:
    return result_dir / "stage_results.csv"


def run_single(
    method_label: str,
    config_name: str,
    method_spec: dict,
    alpha: float,
    hidden_dim: int,
    seed: int,
    lambdas: list[float],
    data_csv: str,
    device: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Run a single (method, alpha, hidden_dim, seed) configuration."""
    task_cfg = make_task_cfg_no_val(data_csv, n_sample=0, alpha_fair=alpha)
    batch_size = compute_train_size(data_csv, n_sample=0)

    train_cfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
    train_cfg["seeds"] = [seed]
    train_cfg["lambdas"] = lambdas
    train_cfg["steps_per_lambda"] = STEPS_PER_LAMBDA
    train_cfg["batch_size"] = batch_size
    train_cfg["model"]["hidden_dim"] = hidden_dim
    if device:
        train_cfg["device"] = device

    # Apply method-specific config overrides
    for k, v in method_spec.items():
        if k not in {"method", "use_dec", "use_pred", "use_fair",
                     "pred_weight_mode", "continuation", "allow_orthogonalization"}:
            train_cfg[k] = v

    cfg = {"task": task_cfg, "training": train_cfg}

    t0 = time.time()
    stage_df, iter_df = run_experiment_unified(cfg, method_configs={config_name: method_spec})
    elapsed = time.time() - t0

    # Tag results
    for df in (stage_df, iter_df):
        if not df.empty:
            df["method_label"] = method_label
            df["alpha_fair"] = alpha
            df["hidden_dim"] = hidden_dim
            df["config_name"] = config_name

    return stage_df, iter_df, elapsed


def main():
    parser = argparse.ArgumentParser(description="Final healthcare experiment grid")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Subset of method labels (default: all)")
    parser.add_argument("--alphas", nargs="+", type=float, default=None)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--results-dir", default=RESULTS_BASE)
    parser.add_argument("--data-csv", default=DATA_CSV)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    alphas = args.alphas or ALPHAS
    hidden_dims = args.hidden_dims or HIDDEN_DIMS
    seeds = args.seeds or SEEDS

    # Resolve methods
    if args.methods:
        methods = {}
        name_map = {k.lower(): k for k in METHOD_GRID}
        for m in args.methods:
            key = m if m in METHOD_GRID else name_map.get(m.lower())
            if key and key in METHOD_GRID:
                methods[key] = METHOD_GRID[key]
            else:
                print(f"WARNING: Unknown method '{m}'. Available: {list(METHOD_GRID.keys())}")
        if not methods:
            return
    else:
        methods = METHOD_GRID

    total_runs = len(methods) * len(alphas) * len(hidden_dims) * len(seeds)

    print("=" * 70)
    print("HEALTHCARE FINAL EXPERIMENT")
    print("=" * 70)
    print(f"Methods:     {list(methods.keys())}")
    print(f"Alphas:      {alphas}")
    print(f"Hidden dims: {hidden_dims}")
    print(f"Seeds:       {seeds}")
    print(f"Val split:   NONE (50/50 train/test)")
    print(f"Total runs:  {total_runs}")
    print(f"Results:     {args.results_dir}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Would execute:")
        for method_label, grid_spec in methods.items():
            for alpha in alphas:
                for hd in hidden_dims:
                    for seed in seeds:
                        rdir = _result_path(args.results_dir, method_label, alpha, hd, seed)
                        skip = "SKIP" if _done_marker(rdir).exists() and not args.overwrite else ""
                        print(f"  {method_label:15s} alpha={alpha} hd={hd:3d} seed={seed} "
                              f"lambdas={grid_spec['lambdas']} {skip}")
        return

    batch_size = compute_train_size(args.data_csv, n_sample=0)
    print(f"Batch size (full training set): {batch_size}")

    elapsed_times: list[float] = []
    errors: list[dict] = []
    skipped = 0
    run_idx = 0
    all_stage: list[pd.DataFrame] = []
    all_iter: list[pd.DataFrame] = []

    for method_label, grid_spec in methods.items():
        config_name = grid_spec["config"]
        lambdas = grid_spec["lambdas"]
        method_spec = copy.deepcopy(ALL_METHOD_CONFIGS[config_name])

        for alpha in alphas:
            for hd in hidden_dims:
                for seed in seeds:
                    run_idx += 1
                    rdir = _result_path(args.results_dir, method_label, alpha, hd, seed)

                    if _done_marker(rdir).exists() and not args.overwrite:
                        skipped += 1
                        print(f"  [{run_idx}/{total_runs}] SKIP {method_label} "
                              f"alpha={alpha} hd={hd} seed={seed}")
                        try:
                            all_stage.append(pd.read_csv(_done_marker(rdir)))
                        except Exception:
                            pass
                        continue

                    print(f"\n[{run_idx}/{total_runs}] {method_label} "
                          f"alpha={alpha} hd={hd} seed={seed}")

                    try:
                        stage_df, iter_df, elapsed = run_single(
                            method_label=method_label,
                            config_name=config_name,
                            method_spec=method_spec,
                            alpha=alpha, hidden_dim=hd, seed=seed,
                            lambdas=lambdas, data_csv=args.data_csv,
                            device=args.device,
                        )
                        elapsed_times.append(elapsed)

                        if not stage_df.empty:
                            stage_df.to_csv(rdir / "stage_results.csv", index=False)
                            all_stage.append(stage_df)
                        if not iter_df.empty:
                            iter_df.to_csv(rdir / "iter_logs.csv", index=False)
                            all_iter.append(iter_df)

                        run_meta = {
                            "method_label": method_label, "config_name": config_name,
                            "alpha": alpha, "hidden_dim": hd, "seed": seed,
                            "lambdas": lambdas, "elapsed_sec": elapsed,
                        }
                        with open(rdir / "run_config.json", "w") as f:
                            json.dump(run_meta, f, indent=2)

                        print(f"    Done in {elapsed:.1f}s")
                        if len(elapsed_times) == 1:
                            remaining = total_runs - run_idx
                            print(f"    Est. remaining: {remaining * elapsed / 3600:.1f}h")

                    except Exception as e:
                        errors.append({
                            "method": method_label, "alpha": alpha,
                            "hidden_dim": hd, "seed": seed, "error": str(e),
                        })
                        print(f"    ERROR: {e}")
                        traceback.print_exc()

    # Save aggregates
    agg_dir = Path(args.results_dir)
    agg_dir.mkdir(parents=True, exist_ok=True)
    if all_stage:
        combined = pd.concat(all_stage, ignore_index=True)
        combined.to_csv(agg_dir / "stage_results_all.csv", index=False)
        print(f"\nAggregate: {agg_dir / 'stage_results_all.csv'} ({len(combined)} rows)")
    if all_iter:
        combined = pd.concat(all_iter, ignore_index=True)
        combined.to_csv(agg_dir / "iter_logs_all.csv", index=False)
        print(f"Aggregate: {agg_dir / 'iter_logs_all.csv'} ({len(combined)} rows)")

    print(f"\n{'=' * 70}")
    print(f"DONE: {run_idx - skipped - len(errors)} completed, "
          f"{skipped} skipped, {len(errors)} errors")
    if elapsed_times:
        print(f"Total: {sum(elapsed_times)/3600:.2f}h "
              f"(avg {sum(elapsed_times)/len(elapsed_times):.1f}s/run)")
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  {e['method']} alpha={e['alpha']} hd={e['hidden_dim']} "
                  f"seed={e['seed']}: {e['error']}")


if __name__ == "__main__":
    main()
