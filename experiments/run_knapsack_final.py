#!/usr/bin/env python3
"""Final synthetic knapsack experiment runner — full grid for INFORMS JoC submission.

Experimental grid:
  alpha          : {0.5, 2.0}
  unfairness     : {mild, medium, high}
  seeds          : [11, 22, 33, 44, 55]
  methods        : FPTO (lambda=0/0.5/1/5), SAA, WDRO,
                   FDFL-Scal (lambda=0/0.5/1/5), FDFL-PCGrad, FDFL-MGDA, FDFL-CAGrad

Note: FPTO(lambda=0) = PTO, FDFL-Scal(lambda=0) = DFL.
No validation split — train/test only.

Unfairness levels (n_items=7):
  mild   : group_bias=0.1, noise_std_lo=0.1, noise_std_hi=0.2, group_ratio=0.5 (4/3)
  medium : group_bias=0.3, noise_std_lo=0.1, noise_std_hi=0.5, group_ratio=0.5 (4/3)
  high   : group_bias=0.3, noise_std_lo=0.1, noise_std_hi=0.5, group_ratio=0.67 (5/2)

Usage:
  python experiments/run_knapsack_final.py
  python experiments/run_knapsack_final.py --dry-run
  python experiments/run_knapsack_final.py --methods FDFL-PCGrad --unfairness high
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
from experiments.configs import ALL_METHOD_CONFIGS, DEFAULT_TRAIN_CFG  # noqa: E402

# ======================================================================
# Experiment constants
# ======================================================================
SEEDS = [11, 22, 33, 44, 55]
ALPHAS = [0.5, 2.0]
STEPS_PER_LAMBDA = 20
HIDDEN_DIM = 64
N_TRAIN, N_TEST = 200, 80   # no validation
N_ITEMS = 7
N_CONSTRAINTS = 3
N_FEATURES = 5
POLY_DEGREE = 2
RESULTS_BASE = str(REPO_ROOT / "results" / "final" / "knapsack")
LAMBDAS_SWEEP = [0.0, 0.5, 1.0, 5.0]

# Unfairness level definitions
UNFAIRNESS_LEVELS = {
    "mild": {
        "group_bias": 0.1,
        "noise_std_lo": 0.1,
        "noise_std_hi": 0.2,
        "group_ratio": 0.5,
        "label": "Mild (small bias, small noise gap)",
    },
    "medium": {
        "group_bias": 0.3,
        "noise_std_lo": 0.1,
        "noise_std_hi": 0.5,
        "group_ratio": 0.5,
        "label": "Medium (larger bias + noise gap)",
    },
    "high": {
        "group_bias": 0.3,
        "noise_std_lo": 0.1,
        "noise_std_hi": 0.5,
        "group_ratio": 0.67,
        "label": "High (bias + noise gap + group imbalance)",
    },
}

# -----------------------------------------------------------------------
# Method grid — unified lambda sweep
# -----------------------------------------------------------------------
METHOD_GRID = {
    # --- Two-stage (predict-then-optimize family) ---
    "FPTO":         {"config": "FPTO",     "lambdas": LAMBDAS_SWEEP},
    # --- Data-driven optimization baselines ---
    "SAA":          {"config": "SAA",      "lambdas": [0.0]},
    "WDRO":         {"config": "WDRO",     "lambdas": [0.0]},
    # --- Integrated: scalarized ---
    "FDFL-Scal":    {"config": "FPLG",     "lambdas": LAMBDAS_SWEEP},
    # --- Integrated + MOO ---
    "FDFL-PCGrad":  {"config": "PCGrad",   "lambdas": [0.0]},
    "FDFL-MGDA":    {"config": "MGDA",     "lambdas": [0.0]},
    "FDFL-CAGrad":  {"config": "CAGrad",   "lambdas": [0.0]},
}

UNSUPPORTED_BACKENDS = {"ffo", "nce", "lancer"}


def _result_path(base_dir: str, method_label: str, alpha: float,
                 unfairness: str, seed: int) -> Path:
    tag = f"alpha_{alpha}_uf_{unfairness}"
    d = Path(base_dir) / method_label / tag / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _done_marker(result_dir: Path) -> Path:
    return result_dir / "stage_results.csv"


def _make_task_cfg(alpha: float, unfairness: str) -> dict:
    uf = UNFAIRNESS_LEVELS[unfairness]
    return {
        "name": "md_knapsack",
        "n_samples_train": N_TRAIN,
        "n_samples_val": 0,       # no validation
        "n_samples_test": N_TEST,
        "n_features": N_FEATURES,
        "n_items": N_ITEMS,
        "n_constraints": N_CONSTRAINTS,
        "scenario": "alpha_fair",
        "alpha_fair": alpha,
        "poly_degree": POLY_DEGREE,
        "group_bias": uf["group_bias"],
        "noise_std_lo": uf["noise_std_lo"],
        "noise_std_hi": uf["noise_std_hi"],
        "group_ratio": uf["group_ratio"],
        "budget_tightness": 0.5,
        "data_seed": 42,
        "fairness_type": "mad",
    }


def run_single(
    method_label: str,
    config_name: str,
    method_spec: dict,
    alpha: float,
    unfairness: str,
    seed: int,
    lambdas: list[float],
    device: str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Run a single (method, alpha, unfairness, seed) configuration."""
    task_cfg = _make_task_cfg(alpha, unfairness)

    train_cfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
    train_cfg["seeds"] = [seed]
    train_cfg["lambdas"] = lambdas
    train_cfg["steps_per_lambda"] = STEPS_PER_LAMBDA
    train_cfg["batch_size"] = -1  # overridden to 32 for FDFL methods in colab_runner
    train_cfg["decision_grad_backend"] = "spsa"
    train_cfg["device"] = device
    train_cfg["model"]["hidden_dim"] = HIDDEN_DIM
    train_cfg["log_every"] = 2

    # Apply method-specific overrides
    for k, v in method_spec.items():
        if k not in {"method", "use_dec", "use_pred", "use_fair",
                     "pred_weight_mode", "continuation", "allow_orthogonalization"}:
            train_cfg[k] = v

    cfg = {"task": task_cfg, "training": train_cfg}

    t0 = time.time()
    stage_df, iter_df = run_experiment_unified(cfg, method_configs={config_name: method_spec})
    elapsed = time.time() - t0

    # Tag results
    uf_cfg = UNFAIRNESS_LEVELS[unfairness]
    for df in (stage_df, iter_df):
        if not df.empty:
            df["method_label"] = method_label
            df["alpha_fair"] = alpha
            df["unfairness_level"] = unfairness
            df["group_bias"] = uf_cfg["group_bias"]
            df["noise_std_lo"] = uf_cfg["noise_std_lo"]
            df["noise_std_hi"] = uf_cfg["noise_std_hi"]
            df["group_ratio"] = uf_cfg["group_ratio"]
            df["config_name"] = config_name

    return stage_df, iter_df, elapsed


def main():
    parser = argparse.ArgumentParser(description="Final knapsack experiment grid")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--alphas", nargs="+", type=float, default=None)
    parser.add_argument("--unfairness", nargs="+", default=None,
                        choices=["mild", "medium", "high"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--results-dir", default=RESULTS_BASE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    alphas = args.alphas or ALPHAS
    unfairness_levels = args.unfairness or list(UNFAIRNESS_LEVELS.keys())
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
                print(f"WARNING: Unknown method '{m}'")
        if not methods:
            print("Available:", list(METHOD_GRID.keys()))
            return
    else:
        methods = METHOD_GRID

    # Filter unsupported
    filtered = {}
    for label, spec in methods.items():
        backend = str(ALL_METHOD_CONFIGS.get(spec["config"], {}).get("method", "")).lower()
        if backend in UNSUPPORTED_BACKENDS:
            print(f"  Skipping {label} (unsupported backend={backend})")
            continue
        filtered[label] = spec
    methods = filtered

    total_runs = len(methods) * len(alphas) * len(unfairness_levels) * len(seeds)

    print("=" * 70)
    print("KNAPSACK FINAL EXPERIMENT")
    print("=" * 70)
    print(f"Methods:     {list(methods.keys())}")
    print(f"Alphas:      {alphas}")
    print(f"Unfairness:  {unfairness_levels}")
    print(f"Seeds:       {seeds}")
    print(f"Val split:   NONE (train/test only)")
    print(f"Total runs:  {total_runs}")
    print(f"Results:     {args.results_dir}")
    for name in unfairness_levels:
        uf = UNFAIRNESS_LEVELS[name]
        print(f"  {name:8s}: bias={uf['group_bias']}, "
              f"noise=[{uf['noise_std_lo']},{uf['noise_std_hi']}], "
              f"ratio={uf['group_ratio']}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Would execute:")
        for method_label, grid_spec in methods.items():
            for alpha in alphas:
                for uf in unfairness_levels:
                    for seed in seeds:
                        rdir = _result_path(args.results_dir, method_label, alpha, uf, seed)
                        skip = "SKIP" if _done_marker(rdir).exists() and not args.overwrite else ""
                        print(f"  {method_label:15s} alpha={alpha} uf={uf:6s} "
                              f"seed={seed} lambdas={grid_spec['lambdas']} {skip}")
        return

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
            for uf in unfairness_levels:
                for seed in seeds:
                    run_idx += 1
                    rdir = _result_path(args.results_dir, method_label, alpha, uf, seed)

                    if _done_marker(rdir).exists() and not args.overwrite:
                        skipped += 1
                        print(f"  [{run_idx}/{total_runs}] SKIP {method_label} "
                              f"alpha={alpha} uf={uf} seed={seed}")
                        try:
                            all_stage.append(pd.read_csv(_done_marker(rdir)))
                        except Exception:
                            pass
                        continue

                    print(f"\n[{run_idx}/{total_runs}] {method_label} "
                          f"alpha={alpha} uf={uf} seed={seed}")

                    try:
                        stage_df, iter_df, elapsed = run_single(
                            method_label=method_label,
                            config_name=config_name,
                            method_spec=method_spec,
                            alpha=alpha, unfairness=uf, seed=seed,
                            lambdas=lambdas, device=args.device,
                        )
                        elapsed_times.append(elapsed)

                        if not stage_df.empty:
                            stage_df.to_csv(rdir / "stage_results.csv", index=False)
                            all_stage.append(stage_df)
                        if not iter_df.empty:
                            iter_df.to_csv(rdir / "iter_logs.csv", index=False)
                            all_iter.append(iter_df)

                        with open(rdir / "run_config.json", "w") as f:
                            json.dump({
                                "method_label": method_label, "config_name": config_name,
                                "alpha": alpha, "unfairness": uf, "seed": seed,
                                "lambdas": lambdas, "elapsed_sec": elapsed,
                            }, f, indent=2)

                        print(f"    Done in {elapsed:.1f}s")
                        if len(elapsed_times) == 1:
                            remaining = total_runs - run_idx
                            print(f"    Est. remaining: {remaining * elapsed / 60:.1f}min")

                    except Exception as e:
                        errors.append({
                            "method": method_label, "alpha": alpha,
                            "unfairness": uf, "seed": seed, "error": str(e),
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
        print(f"Total: {sum(elapsed_times)/60:.1f}min "
              f"(avg {sum(elapsed_times)/len(elapsed_times):.1f}s/run)")
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  {e['method']} alpha={e['alpha']} uf={e['unfairness']} "
                  f"seed={e['seed']}: {e['error']}")


if __name__ == "__main__":
    main()
