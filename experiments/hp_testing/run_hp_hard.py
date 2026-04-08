#!/usr/bin/env python3
"""HP test with harder knapsack config — DFL should win big.

More items, tighter budget, less data, limited model capacity.
The predictor learns rough estimates but can't rank items perfectly,
so decision gradients provide large additional value.
"""

import sys, time, copy, argparse, json, traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from fair_dfl.runner import run_experiment_unified
from experiments.configs import ALL_METHOD_CONFIGS, DEFAULT_TRAIN_CFG

# ======================================================================
# HARD CONFIG — designed so DFL methods outperform prediction-only
# ======================================================================

TASK_CFG = {
    "n_items": 12,             # more items competing
    "n_budget_dims": 1,        # classic knapsack
    "n_features": 5,
    "budget_tightness": 0.15,  # very tight — only ~2 items selected
    "poly_degree": 3,          # harder function to learn
    "decision_mode": "group",
    "n_samples_train": 150,    # less data — predictor generalizes poorly
    "n_samples_val": 40,
    "n_samples_test": 100,
}

UF_CONFIGS = {
    "medium": {"group_bias": 0.4, "noise_std_lo": 0.05, "noise_std_hi": 0.20, "group_ratio": 0.6},
}

TRAIN_CFG = {
    "steps_per_lambda": 70,
    "lr": 0.001,
    "lr_decay": 0.0,
    "lr_warmup_steps": 5,
    "batch_size": 32,
    "optimizer": "adamw",
    "weight_decay": 1e-4,
    "decision_grad_backend": "spsa",
    "alpha_schedule": {"type": "inv_sqrt", "alpha0": 1.0, "alpha_min": 0.0},
    "hidden_dim": 32,          # smaller model — capacity limited
    "n_layers": 1,             # shallower
    "activation": "relu",
    "dropout": 0.1,
    "init_mode": "best_practice",
    "grad_clip_norm": 10000.0,
    "log_every": 5,
}

METHOD_GRID = {
    "FPTO":        {"config": "FPTO",      "lambdas": [0.0, 0.5, 1.0, 5.0]},
    "SAA":         {"config": "SAA",       "lambdas": [0.0]},
    "WDRO":        {"config": "WDRO",      "lambdas": [0.0]},
    "FDFL-Scal":   {"config": "FDFL-Scal", "lambdas": [0.0, 0.5, 1.0, 5.0]},
    "FDFL-PCGrad": {"config": "PCGrad",    "lambdas": [0.0]},
    "FDFL-MGDA":   {"config": "MGDA",      "lambdas": [0.0]},
    "FDFL-CAGrad": {"config": "CAGrad",    "lambdas": [0.0]},
}

RESULTS_DIR = str(Path(__file__).parent / "results_hard")


def main():
    parser = argparse.ArgumentParser(description="Hard knapsack HP test")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--alphas", nargs="+", type=float, default=[2.0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.steps:
        TRAIN_CFG["steps_per_lambda"] = args.steps

    selected = {k: v for k, v in METHOD_GRID.items()
                if args.methods is None or k in args.methods}

    total = len(selected) * len(args.alphas) * len(UF_CONFIGS) * len(args.seeds)

    print("=" * 60)
    print("HARD KNAPSACK HP TEST")
    print("=" * 60)
    print(f"Methods: {list(selected.keys())}")
    print(f"Alphas: {args.alphas}, Seeds: {args.seeds}")
    print(f"Total runs: {total}")
    print(f"\nTask (HARD): n_items={TASK_CFG['n_items']}, budget={TASK_CFG['budget_tightness']}, "
          f"n_train={TASK_CFG['n_samples_train']}, poly_deg={TASK_CFG['poly_degree']}")
    print(f"Model: hidden={TRAIN_CFG['hidden_dim']}, layers={TRAIN_CFG['n_layers']}")
    print(f"Steps: {TRAIN_CFG['steps_per_lambda']}")
    for k, v in UF_CONFIGS.items():
        snr = v['group_bias'] / v['noise_std_hi']
        print(f"  {k}: bias={v['group_bias']}, noise_hi={v['noise_std_hi']}, SNR~{snr:.1f}")
    print("=" * 60)

    if args.dry_run:
        print("DRY RUN — exiting.")
        return

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    all_stage = []
    errors = []
    times = []
    run_idx = 0

    for method_label, grid_spec in selected.items():
        config_name = grid_spec["config"]
        lambdas = grid_spec["lambdas"]
        method_spec = copy.deepcopy(ALL_METHOD_CONFIGS[config_name])

        for alpha in args.alphas:
            for uf_name, uf in UF_CONFIGS.items():
                for seed in args.seeds:
                    run_idx += 1
                    print(f"  [{run_idx}/{total}] {method_label} a={alpha} s={seed} ", end="", flush=True)

                    try:
                        task_cfg = {
                            "name": "md_knapsack",
                            "n_samples_train": TASK_CFG["n_samples_train"],
                            "n_samples_val": TASK_CFG.get("n_samples_val", 0),
                            "n_samples_test": TASK_CFG["n_samples_test"],
                            "n_features": TASK_CFG.get("n_features", 5),
                            "n_items": TASK_CFG["n_items"],
                            "n_budget_dims": TASK_CFG.get("n_budget_dims", 1),
                            "scenario": "alpha_fair",
                            "alpha_fair": alpha,
                            "poly_degree": TASK_CFG.get("poly_degree", 3),
                            "group_bias": uf["group_bias"],
                            "noise_std_lo": uf["noise_std_lo"],
                            "noise_std_hi": uf["noise_std_hi"],
                            "group_ratio": uf["group_ratio"],
                            "budget_tightness": TASK_CFG.get("budget_tightness", 0.15),
                            "decision_mode": TASK_CFG.get("decision_mode", "group"),
                            "data_seed": 42,
                            "fairness_type": "mad",
                        }

                        train_cfg = copy.deepcopy(DEFAULT_TRAIN_CFG)
                        train_cfg["seeds"] = [seed]
                        train_cfg["lambdas"] = lambdas
                        train_cfg["steps_per_lambda"] = TRAIN_CFG["steps_per_lambda"]
                        train_cfg["lr"] = TRAIN_CFG["lr"]
                        train_cfg["lr_decay"] = TRAIN_CFG["lr_decay"]
                        train_cfg["lr_warmup_steps"] = TRAIN_CFG.get("lr_warmup_steps", 0)
                        train_cfg["batch_size"] = TRAIN_CFG["batch_size"]
                        train_cfg["optimizer"] = TRAIN_CFG["optimizer"]
                        train_cfg["weight_decay"] = TRAIN_CFG["weight_decay"]
                        train_cfg["decision_grad_backend"] = TRAIN_CFG["decision_grad_backend"]
                        train_cfg["device"] = device
                        train_cfg["grad_clip_norm"] = TRAIN_CFG["grad_clip_norm"]
                        train_cfg["log_every"] = TRAIN_CFG["log_every"]
                        train_cfg["model"]["hidden_dim"] = TRAIN_CFG["hidden_dim"]
                        train_cfg["model"]["n_layers"] = TRAIN_CFG["n_layers"]
                        train_cfg["model"]["activation"] = TRAIN_CFG["activation"]
                        train_cfg["model"]["dropout"] = TRAIN_CFG["dropout"]
                        train_cfg["model"]["init_mode"] = TRAIN_CFG["init_mode"]

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

                        if not stage_df.empty:
                            all_stage.append(stage_df)
                        print(f"({elapsed:.1f}s)")

                        if len(times) == 1 and total > 1:
                            print(f"    Est. remaining: {(total - 1) * elapsed / 60:.0f}min")

                    except Exception as e:
                        errors.append({"method": method_label, "seed": seed, "error": str(e)})
                        print(f"ERROR: {e}")
                        traceback.print_exc()

    print(f"\nDone: {run_idx - len(errors)} ok, {len(errors)} errors")
    if times:
        print(f"Avg: {np.mean(times):.1f}s/run, Total: {sum(times)/60:.1f}min")

    if all_stage:
        result = pd.concat(all_stage, ignore_index=True)
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        result.to_csv(Path(RESULTS_DIR) / "stage_results_all.csv", index=False)

        cols = ["test_regret_normalized", "test_fairness", "test_pred_mse"]
        summary = result.groupby(["method_label", "lambda"])[cols].agg(["mean", "std"]).round(4)
        summary.columns = [f'{c[0].replace("test_","")[:8]}_{c[1][:4]}' for c in summary.columns]

        print("\n" + "=" * 60)
        print("RESULTS (mean +/- std over seeds)")
        print("=" * 60)
        print(summary.to_string())

        saa_mse = result[result["method_label"] == "SAA"]["test_pred_mse"].mean()
        saa_reg = result[result["method_label"] == "SAA"]["test_regret_normalized"].mean()
        print(f"\nSAA baseline: MSE={saa_mse:.4f}, Regret={saa_reg:.4f}")
        for m in ["FPTO", "WDRO", "FDFL-Scal", "FDFL-PCGrad", "FDFL-MGDA", "FDFL-CAGrad"]:
            sub = result[(result["method_label"] == m) & (result["lambda"] == 0.0)]
            if sub.empty:
                continue
            mse = sub["test_pred_mse"].mean()
            reg = sub["test_regret_normalized"].mean()
            reg_gap = (reg - saa_reg) / saa_reg * 100
            mse_gap = (mse - saa_mse) / saa_mse * 100
            print(f"  {m:15s}: Regret={reg:.4f} ({reg_gap:+.1f}% vs SAA), MSE={mse:.4f} ({mse_gap:+.1f}%)")


if __name__ == "__main__":
    main()
