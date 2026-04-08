#!/usr/bin/env python3
"""Quick healthcare HP test — verify MOO methods work with analytic gradients."""

import sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

HC_RESULTS = str(Path(__file__).parent / "results_healthcare")

def main():
    from experiments.colab_runner import run_healthcare_slice

    # Same ML HPs as knapsack test
    train_overrides = {
        "optimizer": "adamw",
        "lr": 0.001,
        "weight_decay": 1e-4,
        "init_mode": "best_practice",
        "dropout": 0.1,
        "lr_warmup_steps": 5,
        "hidden_dim": 64,
        "n_layers": 2,
    }

    task_overrides = {
        "n_sample": 500,       # small subset for quick testing (full = 0)
        "val_fraction": 0.1,   # 10% validation
    }

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=70)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--alphas", nargs="+", type=float, default=[2.0])
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--n-sample", type=int, default=500)
    args = parser.parse_args()

    task_overrides["n_sample"] = args.n_sample

    print("=" * 60)
    print("HEALTHCARE HP TEST")
    print("=" * 60)
    print(f"Steps: {args.steps}, Seeds: {args.seeds}, Alphas: {args.alphas}")
    print(f"n_sample: {task_overrides['n_sample']} (0=all 48,784)")
    print(f"Train overrides: {train_overrides}")
    print(f"Results: {HC_RESULTS}")
    print("=" * 60)

    t0 = time.time()
    results = run_healthcare_slice(
        alphas=args.alphas,
        seeds=args.seeds,
        methods=args.methods,
        results_dir=HC_RESULTS,
        steps=args.steps,
        task_overrides=task_overrides,
        train_overrides=train_overrides,
        overwrite=True,
    )
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed / 60:.1f} min")

    if not results.empty:
        cols = ["test_regret_normalized", "test_fairness", "test_pred_mse"]
        summary = results.groupby(["method_label", "lambda"])[cols].agg(["mean", "std"]).round(4)
        summary.columns = [f'{c[0].replace("test_","")[:8]}_{c[1][:4]}' for c in summary.columns]
        print("\n" + "=" * 60)
        print("RESULTS (mean +/- std over seeds)")
        print("=" * 60)
        print(summary.to_string())

        # SAA comparison
        saa_mse = results[results["method_label"] == "SAA"]["test_pred_mse"].mean()
        print(f"\nSAA MSE baseline: {saa_mse:.4f}")
        for m in ["FPTO", "WDRO", "FDFL-Scal", "FDFL-PCGrad", "FDFL-MGDA", "FDFL-CAGrad"]:
            sub = results[(results["method_label"] == m) & (results["lambda"] == 0.0)]
            if sub.empty:
                continue
            mse = sub["test_pred_mse"].mean()
            reg = sub["test_regret_normalized"].mean()
            print(f"  {m:15s}: MSE={mse:.4f} ({(mse - saa_mse) / saa_mse * 100:+.1f}% vs SAA), Regret={reg:.4f}")

        results.to_csv(Path(HC_RESULTS) / "summary.csv", index=False)


if __name__ == "__main__":
    main()
