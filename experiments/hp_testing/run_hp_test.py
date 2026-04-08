#!/usr/bin/env python3
"""Local hyperparameter testing script for rapid iteration.

Usage:
    # Activate env first:
    #   conda activate fairdfl
    #
    # Quick single-method test (1 seed, 1 unfairness, 1 alpha):
    python experiments/hp_testing/run_hp_test.py
    #
    # Test specific methods:
    python experiments/hp_testing/run_hp_test.py --methods FPTO FDFL-Scal
    #
    # Full grid (all methods, all levels):
    python experiments/hp_testing/run_hp_test.py --full
    #
    # Dry run (print config, don't execute):
    python experiments/hp_testing/run_hp_test.py --dry-run
    #
    # Override from CLI:
    python experiments/hp_testing/run_hp_test.py --steps 100 --lr 0.005 --pred-weight fixed1

All hyperparameters can be edited in the CONFIG section below or via CLI flags.
Results go to experiments/hp_testing/results/ (gitignored).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from fair_dfl.runner import run_experiment_unified  # noqa: E402
from experiments.configs import ALL_METHOD_CONFIGS, DEFAULT_TRAIN_CFG  # noqa: E402


# ======================================================================
# CONFIG — Edit these for your HP experiments
# ======================================================================

# --- Task (knapsack problem structure) ---
TASK_CFG = {
    "n_items": 7,
    "n_budget_dims": 1,            # budget dimensions (rows of A matrix). 1 = classic knapsack
    "n_features": 5,
    "budget_tightness": 0.3,       # b = tightness * A.sum(). Lower = tighter = more competition
    "poly_degree": 2,              # polynomial feature mapping degree
    "decision_mode": "group",      # "group" = two-level group alpha-fair
    "n_samples_train": 400,        # training samples (need enough for MLP to generalize)
    "n_samples_test": 200,         # test samples
}

# --- Unfairness levels ---
UF_CONFIGS = {
    "mild":   {"group_bias": 0.2, "noise_std_lo": 0.05, "noise_std_hi": 0.10, "group_ratio": 0.5},
    "medium": {"group_bias": 0.4, "noise_std_lo": 0.05, "noise_std_hi": 0.20, "group_ratio": 0.6},
    "high":   {"group_bias": 0.6, "noise_std_lo": 0.05, "noise_std_hi": 0.30, "group_ratio": 0.75},
}

# --- Training ---
TRAIN_CFG = {
    # Core training params
    "steps_per_lambda": 70,        # gradient updates per lambda stage
    "lr": 0.001,                   # learning rate
    "lr_decay": 0.0,              # LR decay: lr_t = lr / (1 + lr_decay * t). 0 = constant
    "lr_warmup_steps": 5,          # linear LR warmup over first N steps (0 = off)
    "batch_size": 32,              # same batch size for all methods (no confound)
    "optimizer": "adamw",          # "sgd", "sgd_momentum", "adam", "adamw"
    "weight_decay": 1e-4,         # L2 reg (adam) or decoupled weight decay (adamw)

    # Decision gradient
    "decision_grad_backend": "spsa",  # "spsa", "finite_diff", "spo_plus" (LP only)

    # Alpha schedule (prediction gradient weight decay)
    # Only active for PLG/FPLG methods. Other methods use fixed1.
    "pred_weight_mode_override": None,  # Set to "fixed1" to override all methods
    "alpha_schedule": {"type": "inv_sqrt", "alpha0": 1.0, "alpha_min": 0.0},

    # Model architecture
    "hidden_dim": 64,
    "n_layers": 2,
    "activation": "relu",
    "dropout": 0.1,               # dropout rate (0.0 = off)
    "init_mode": "best_practice", # "default", "best_practice" (Kaiming He), "legacy_core"

    # Gradient control
    "grad_clip_norm": 10000.0,
    "log_every": 5,
}

# --- Experiment grid ---
ALPHAS = [0.5, 2.0]
SEEDS = [11, 22, 33]

# Quick test defaults (use --full for complete grid)
QUICK_METHODS = ["FPTO", "WDRO", "SAA"]
QUICK_ALPHAS = [2.0]
QUICK_UF = ["medium"]
QUICK_SEEDS = [11]

# Method grid (maps display name -> config name + lambda sweep)
METHOD_GRID = {
    "FPTO":        {"config": "FPTO",    "lambdas": [0.0, 0.5, 1.0, 5.0]},
    "SAA":         {"config": "SAA",     "lambdas": [0.0]},
    "WDRO":        {"config": "WDRO",    "lambdas": [0.0]},
    "FDFL-Scal":   {"config": "FDFL-Scal", "lambdas": [0.0, 0.5, 1.0, 5.0]},
    "FDFL-PCGrad": {"config": "PCGrad",  "lambdas": [0.0]},
    "FDFL-MGDA":   {"config": "MGDA",    "lambdas": [0.0]},
    "FDFL-CAGrad": {"config": "CAGrad",  "lambdas": [0.0]},
}

DECISION_GRAD_METHODS = {"FDFL-Scal", "FDFL-PCGrad", "FDFL-MGDA", "FDFL-CAGrad"}

RESULTS_DIR = str(Path(__file__).parent / "results")


# ======================================================================
# Runner
# ======================================================================

def _done(rdir: Path) -> bool:
    return (rdir / "stage_results.csv").exists()


def _save(rdir: Path, stage_df, iter_df, meta: dict):
    rdir.mkdir(parents=True, exist_ok=True)
    if not stage_df.empty:
        stage_df.to_csv(rdir / "stage_results.csv", index=False)
    if not iter_df.empty:
        iter_df.to_csv(rdir / "iter_logs.csv", index=False)
    with open(rdir / "run_config.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


_MODEL_KEYS = {"hidden_dim", "n_layers", "activation", "dropout", "batch_norm",
               "arch", "init_mode"}


def run_hp_test(
    methods: list[str] | None = None,
    alphas: list[float] | None = None,
    unfairness_levels: list[str] | None = None,
    seeds: list[int] | None = None,
    overwrite: bool = True,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run HP test grid locally."""

    alphas = alphas or QUICK_ALPHAS
    uf_levels = unfairness_levels or QUICK_UF
    seeds = seeds or QUICK_SEEDS

    selected = {k: v for k, v in METHOD_GRID.items()
                if methods is None or k in methods}

    total = len(selected) * len(alphas) * len(uf_levels) * len(seeds)

    print(f"\n{'='*60}")
    print(f"HP Testing — Local Run")
    print(f"{'='*60}")
    print(f"Methods: {list(selected.keys())}")
    print(f"Alphas: {alphas}")
    print(f"Unfairness: {uf_levels}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {total}")
    print(f"\n--- Task Config ---")
    for k, v in TASK_CFG.items():
        print(f"  {k}: {v}")
    print(f"\n--- Training Config ---")
    for k, v in TRAIN_CFG.items():
        print(f"  {k}: {v}")
    print(f"\n--- Unfairness Levels ---")
    for name in uf_levels:
        print(f"  {name}: {UF_CONFIGS[name]}")
    print(f"{'='*60}\n")

    if dry_run:
        print("DRY RUN — not executing. Exiting.")
        return pd.DataFrame()

    all_stage = []
    errors = []
    times = []
    skipped = 0
    run_idx = 0

    for method_label, grid_spec in selected.items():
        config_name = grid_spec["config"]
        lambdas = grid_spec["lambdas"]
        method_spec = copy.deepcopy(ALL_METHOD_CONFIGS[config_name])

        # Apply pred_weight_mode override if set
        pwm_override = TRAIN_CFG.get("pred_weight_mode_override")
        if pwm_override is not None:
            method_spec["pred_weight_mode"] = pwm_override

        for alpha in alphas:
            for uf_name in uf_levels:
                uf = UF_CONFIGS[uf_name]
                for seed in seeds:
                    run_idx += 1
                    rdir = Path(RESULTS_DIR) / method_label / f"alpha_{alpha}_uf_{uf_name}" / f"seed_{seed}"

                    if _done(rdir) and not overwrite:
                        skipped += 1
                        try:
                            all_stage.append(pd.read_csv(rdir / "stage_results.csv"))
                        except Exception:
                            pass
                        continue

                    print(f"  [{run_idx}/{total}] {method_label} a={alpha} uf={uf_name} s={seed} ", end="", flush=True)

                    try:
                        task_cfg = {
                            "name": "md_knapsack",
                            "n_samples_train": TASK_CFG["n_samples_train"],
                            "n_samples_val": 0,
                            "n_samples_test": TASK_CFG["n_samples_test"],
                            "n_features": TASK_CFG.get("n_features", 5),
                            "n_items": TASK_CFG["n_items"],
                            "n_budget_dims": TASK_CFG.get("n_budget_dims", 3),
                            "scenario": "alpha_fair",
                            "alpha_fair": alpha,
                            "poly_degree": TASK_CFG.get("poly_degree", 2),
                            "group_bias": uf["group_bias"],
                            "noise_std_lo": uf["noise_std_lo"],
                            "noise_std_hi": uf["noise_std_hi"],
                            "group_ratio": uf["group_ratio"],
                            "budget_tightness": TASK_CFG.get("budget_tightness", 0.5),
                            "decision_mode": TASK_CFG.get("decision_mode", "group"),
                            "data_seed": 42,
                            "fairness_type": "mad",
                        }

                        dec_backend = TRAIN_CFG.get("decision_grad_backend", "spsa")

                        train_cfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
                        train_cfg["seeds"] = [seed]
                        train_cfg["lambdas"] = lambdas
                        train_cfg["steps_per_lambda"] = TRAIN_CFG["steps_per_lambda"]
                        train_cfg["lr"] = TRAIN_CFG.get("lr", 0.001)
                        train_cfg["lr_decay"] = TRAIN_CFG.get("lr_decay", 0.0)
                        train_cfg["lr_warmup_steps"] = TRAIN_CFG.get("lr_warmup_steps", 0)
                        train_cfg["batch_size"] = TRAIN_CFG.get("batch_size", 32)
                        train_cfg["optimizer"] = TRAIN_CFG.get("optimizer", "adamw")
                        train_cfg["weight_decay"] = TRAIN_CFG.get("weight_decay", 1e-4)
                        train_cfg["decision_grad_backend"] = dec_backend
                        import torch as _torch
                        train_cfg["device"] = "cuda" if _torch.cuda.is_available() else "cpu"
                        train_cfg["grad_clip_norm"] = TRAIN_CFG.get("grad_clip_norm", 10000.0)
                        train_cfg["log_every"] = TRAIN_CFG.get("log_every", 5)
                        train_cfg["alpha_schedule"] = TRAIN_CFG.get(
                            "alpha_schedule",
                            {"type": "inv_sqrt", "alpha0": 1.0, "alpha_min": 0.0},
                        )

                        # Model config
                        train_cfg["model"]["hidden_dim"] = TRAIN_CFG.get("hidden_dim", 64)
                        train_cfg["model"]["n_layers"] = TRAIN_CFG.get("n_layers", 2)
                        train_cfg["model"]["activation"] = TRAIN_CFG.get("activation", "relu")
                        train_cfg["model"]["dropout"] = TRAIN_CFG.get("dropout", 0.1)
                        train_cfg["model"]["init_mode"] = TRAIN_CFG.get("init_mode", "best_practice")

                        # Method-specific overrides
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

                        _save(rdir, stage_df, iter_df, {
                            "method_label": method_label, "config_name": config_name,
                            "alpha": alpha, "unfairness": uf_name, "seed": seed,
                            "lambdas": lambdas, "elapsed_sec": elapsed,
                            "task_cfg": TASK_CFG, "train_cfg": TRAIN_CFG,
                        })
                        if not stage_df.empty:
                            all_stage.append(stage_df)
                        print(f"({elapsed:.1f}s)")

                        if len(times) == 1 and total > 1:
                            remaining = total - run_idx
                            print(f"    Est. remaining: {remaining * elapsed / 60:.0f}min")

                    except Exception as e:
                        errors.append({"method": method_label, "alpha": alpha,
                                       "uf": uf_name, "seed": seed, "error": str(e)})
                        print(f"ERROR: {e}")
                        traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Done: {run_idx - skipped - len(errors)} new, {skipped} skipped, {len(errors)} errors")
    if times:
        print(f"Avg: {np.mean(times):.1f}s/run, Total: {sum(times)/60:.1f}min")
    if errors:
        print("Errors:")
        for e in errors:
            print(f"  {e}")

    if all_stage:
        result = pd.concat(all_stage, ignore_index=True)
        agg_path = Path(RESULTS_DIR) / "stage_results_all.csv"
        result.to_csv(agg_path, index=False)
        print(f"\nSaved: {agg_path} ({len(result)} rows)")

        # Print summary table
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        summary_cols = ["method_label", "lambda", "alpha_fair", "unfairness_level",
                        "test_regret_normalized", "test_fairness", "test_pred_mse"]
        avail = [c for c in summary_cols if c in result.columns]
        if avail:
            print(result[avail].to_string(index=False))
        return result

    return pd.DataFrame()


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Local HP testing for knapsack experiments")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to test (default: FPTO, FDFL-Scal, SAA)")
    parser.add_argument("--full", action="store_true",
                        help="Run full grid (all methods, alphas, unfairness, 3 seeds)")
    parser.add_argument("--alphas", nargs="+", type=float, default=None)
    parser.add_argument("--unfairness", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")

    # HP overrides from CLI
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--pred-weight", choices=["fixed1", "schedule", "zero"], default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="FDFL mini-batch size (default 32)")
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-items", type=int, default=None)
    parser.add_argument("--budget", type=float, default=None,
                        help="Budget tightness (0-1)")
    parser.add_argument("--noise-hi", type=float, default=None,
                        help="Override noise_std_hi for ALL unfairness levels")

    args = parser.parse_args()

    # Apply CLI overrides
    if args.steps is not None:
        TRAIN_CFG["steps_per_lambda"] = args.steps
    if args.lr is not None:
        TRAIN_CFG["lr"] = args.lr
    if args.pred_weight is not None:
        TRAIN_CFG["pred_weight_mode_override"] = args.pred_weight
    if args.batch_size is not None:
        TRAIN_CFG["fdfl_batch_size"] = args.batch_size
    if args.n_train is not None:
        TASK_CFG["n_samples_train"] = args.n_train
    if args.n_items is not None:
        TASK_CFG["n_items"] = args.n_items
    if args.budget is not None:
        TASK_CFG["budget_tightness"] = args.budget
    if args.noise_hi is not None:
        for uf in UF_CONFIGS.values():
            uf["noise_std_hi"] = args.noise_hi

    if args.full:
        run_hp_test(
            methods=args.methods or list(METHOD_GRID.keys()),
            alphas=args.alphas or ALPHAS,
            unfairness_levels=args.unfairness or list(UF_CONFIGS.keys()),
            seeds=args.seeds or SEEDS,
            overwrite=not args.no_overwrite,
            dry_run=args.dry_run,
        )
    else:
        run_hp_test(
            methods=args.methods or QUICK_METHODS,
            alphas=args.alphas or QUICK_ALPHAS,
            unfairness_levels=args.unfairness or QUICK_UF,
            seeds=args.seeds or QUICK_SEEDS,
            overwrite=not args.no_overwrite,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
