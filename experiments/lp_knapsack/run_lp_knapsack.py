#!/usr/bin/env python3
"""LP knapsack experiment — decision-focused fair learning with SPO+.

Linear-program knapsack:  max r^T d,  A d <= b,  0 <= d <= 1.
LP solutions are at vertices (most d_i = 0 or 1), so prediction quality
directly determines item selection and decision regret.

Decision gradient: SPO+ (Elmachtoub & Grigas 2022) for FDFL methods.
Fairness: prediction-level MAD during training; decision-level metrics
(allocation gap, selection rate gap, welfare gap) for evaluation.

Grid:
  unfairness  : {mild, medium, high}
  seeds       : [11, 22, 33, 44, 55]
  methods     : FPTO (lambda sweep), SAA, WDRO,
                FDFL-Scal (lambda sweep), FDFL-PCGrad, FDFL-MGDA, FDFL-CAGrad

Usage:
  python experiments/lp_knapsack/run_lp_knapsack.py
  python experiments/lp_knapsack/run_lp_knapsack.py --dry-run
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from fair_dfl.runner import run_experiment_unified  # noqa: E402
from experiments.configs import ALL_METHOD_CONFIGS, DEFAULT_TRAIN_CFG  # noqa: E402

# ======================================================================
# Experiment constants
# ======================================================================
SEEDS = [11, 22, 33, 44, 55]
STEPS_BASELINE = 200
STEPS_FDFL = 80
HIDDEN_DIM = 64
N_TRAIN, N_TEST = 80, 80
N_ITEMS = 10
N_CONSTRAINTS = 3
N_FEATURES = 5
POLY_DEGREE = 2
BUDGET_TIGHTNESS = 0.3
LR = 0.002
RESULTS_BASE = str(REPO_ROOT / "results" / "final" / "lp_knapsack")
LAMBDAS_SWEEP = [0.0, 0.5, 1.0, 5.0]

DECISION_GRAD_METHODS = {"FDFL-Scal", "FDFL-PCGrad", "FDFL-MGDA", "FDFL-CAGrad"}

UNFAIRNESS_LEVELS = {
    "mild": {
        "group_bias": 0.2,
        "noise_std_lo": 0.05,
        "noise_std_hi": 0.5,
        "group_ratio": 0.5,
    },
    "medium": {
        "group_bias": 0.4,
        "noise_std_lo": 0.05,
        "noise_std_hi": 1.0,
        "group_ratio": 0.65,
    },
    "high": {
        "group_bias": 0.6,
        "noise_std_lo": 0.05,
        "noise_std_hi": 1.5,
        "group_ratio": 0.75,
    },
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


def _result_path(base_dir: str, method_label: str,
                 unfairness: str, seed: int) -> Path:
    d = Path(base_dir) / method_label / f"uf_{unfairness}" / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_task_cfg(unfairness: str) -> dict:
    uf = UNFAIRNESS_LEVELS[unfairness]
    return {
        "name": "md_knapsack",
        "n_samples_train": N_TRAIN,
        "n_samples_val": 0,
        "n_samples_test": N_TEST,
        "n_features": N_FEATURES,
        "n_items": N_ITEMS,
        "n_constraints": N_CONSTRAINTS,
        "scenario": "lp",
        "alpha_fair": 1.0,  # unused for LP, required by dataclass
        "poly_degree": POLY_DEGREE,
        "group_bias": uf["group_bias"],
        "noise_std_lo": uf["noise_std_lo"],
        "noise_std_hi": uf["noise_std_hi"],
        "group_ratio": uf["group_ratio"],
        "budget_tightness": BUDGET_TIGHTNESS,
        "data_seed": 42,
        "fairness_type": "mad",
    }


def main():
    parser = argparse.ArgumentParser(description="LP knapsack experiment")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--unfairness", nargs="+", default=None,
                        choices=["mild", "medium", "high"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--results-dir", default=RESULTS_BASE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    unfairness_levels = args.unfairness or list(UNFAIRNESS_LEVELS.keys())
    seeds = args.seeds or SEEDS

    if args.methods:
        methods = {k: METHOD_GRID[k] for k in args.methods if k in METHOD_GRID}
    else:
        methods = METHOD_GRID

    total_runs = len(methods) * len(unfairness_levels) * len(seeds)

    print("=" * 70)
    print("LP KNAPSACK EXPERIMENT (SPO+ decision gradients)")
    print("=" * 70)
    print(f"Methods:     {list(methods.keys())}")
    print(f"Unfairness:  {unfairness_levels}")
    print(f"Seeds:       {seeds}")
    print(f"Items:       {N_ITEMS}, Budget: {BUDGET_TIGHTNESS}")
    print(f"Train/Test:  {N_TRAIN}/{N_TEST}, Poly: {POLY_DEGREE}")
    print(f"LR:          {LR}")
    print(f"Total runs:  {total_runs}")
    print(f"Results:     {args.results_dir}")
    print("=" * 70)

    if args.dry_run:
        for ml, gs in methods.items():
            for uf in unfairness_levels:
                for s in seeds:
                    rdir = _result_path(args.results_dir, ml, uf, s)
                    skip = "SKIP" if (rdir / "stage_results.csv").exists() else ""
                    print(f"  {ml:15s} uf={uf:6s} seed={s} {skip}")
        return

    elapsed_times, errors, all_stage = [], [], []
    skipped = 0
    run_idx = 0

    for method_label, grid_spec in methods.items():
        config_name = grid_spec["config"]
        lambdas = grid_spec["lambdas"]
        method_spec = copy.deepcopy(ALL_METHOD_CONFIGS[config_name])
        is_fdfl = method_label in DECISION_GRAD_METHODS

        for uf in unfairness_levels:
            for seed in seeds:
                run_idx += 1
                rdir = _result_path(args.results_dir, method_label, uf, seed)

                if (rdir / "stage_results.csv").exists() and not args.overwrite:
                    skipped += 1
                    try:
                        all_stage.append(pd.read_csv(rdir / "stage_results.csv"))
                    except Exception:
                        pass
                    continue

                print(f"\n[{run_idx}/{total_runs}] {method_label} uf={uf} seed={seed}")

                try:
                    task_cfg = _make_task_cfg(uf)
                    train_cfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
                    train_cfg["seeds"] = [seed]
                    train_cfg["lambdas"] = lambdas
                    train_cfg["steps_per_lambda"] = STEPS_FDFL if is_fdfl else STEPS_BASELINE
                    train_cfg["lr"] = LR
                    train_cfg["batch_size"] = 32 if is_fdfl else -1
                    train_cfg["decision_grad_backend"] = "spo_plus" if is_fdfl else "analytic"
                    train_cfg["device"] = args.device
                    train_cfg["model"]["hidden_dim"] = HIDDEN_DIM
                    train_cfg["log_every"] = 2

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
                    elapsed_times.append(elapsed)

                    uf_cfg = UNFAIRNESS_LEVELS[uf]
                    for df in (stage_df, iter_df):
                        if not df.empty:
                            df["method_label"] = method_label
                            df["unfairness_level"] = uf
                            df["group_bias"] = uf_cfg["group_bias"]
                            df["group_ratio"] = uf_cfg["group_ratio"]
                            df["config_name"] = config_name

                    if not stage_df.empty:
                        stage_df.to_csv(rdir / "stage_results.csv", index=False)
                        all_stage.append(stage_df)
                    if not iter_df.empty:
                        iter_df.to_csv(rdir / "iter_logs.csv", index=False)
                    with open(rdir / "run_config.json", "w") as f:
                        json.dump({"method_label": method_label, "config_name": config_name,
                                   "unfairness": uf, "seed": seed, "lambdas": lambdas,
                                   "elapsed_sec": elapsed}, f, indent=2)

                    print(f"    Done in {elapsed:.1f}s")

                except Exception as e:
                    errors.append({"method": method_label, "uf": uf, "seed": seed, "error": str(e)})
                    print(f"    ERROR: {e}")
                    traceback.print_exc()

    if all_stage:
        combined = pd.concat(all_stage, ignore_index=True)
        agg = Path(args.results_dir) / "stage_results_all.csv"
        Path(args.results_dir).mkdir(parents=True, exist_ok=True)
        combined.to_csv(agg, index=False)
        print(f"\nAggregate: {agg} ({len(combined)} rows)")

    print(f"\nDone: {run_idx - skipped - len(errors)} completed, "
          f"{skipped} skipped, {len(errors)} errors")
    if elapsed_times:
        import numpy as np
        print(f"Total: {sum(elapsed_times)/60:.1f}min (avg {np.mean(elapsed_times):.1f}s/run)")


if __name__ == "__main__":
    main()
