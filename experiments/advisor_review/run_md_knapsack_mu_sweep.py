"""MD knapsack mu-sweep — verify the FDFL prediction-anchor and PCGrad
per-objective normalization fixes motivated by the advisor review.

Setup:
    Task:   per-individual 2-resource knapsack (redesigned n_train=300,
            alpha_fair=2.0), SPSA decision gradient (n_dirs=8, eps=5e-3).
    Seeds:  5 seeds (11, 22, 33, 44, 55)
    Lambdas: [0.0, 0.5, 1.0, 2.0]
    Methods (10):
        PTO group  -> FPTO, SAA, WDRO
        Static DFL -> FDFL, FDFL-0.1, FDFL-0.5, FDFL-Scal
        Dynamic    -> FPLG, PCGrad (normalize=True), MGDA

Hypothesis:
    - FDFL (mu=0)  diverges under SPSA (no prediction anchor)
    - FDFL-0.1     small anchor stabilizes training
    - FDFL-0.5, FDFL-Scal  cluster with FPLG
    - PCGrad (normalized) improves over its pre-fix baseline
    - FPLG remains the top performer

Output:
    results/advisor_review/md_knapsack_mu_sweep/
        seed_<s>/stage_results.csv
        seed_<s>/iter_logs.csv
        seed_<s>/config.json
        grand_summary.csv

Usage:
    python -m experiments.advisor_review.run_md_knapsack_mu_sweep [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from experiments.advisor_review.runner import (  # noqa: E402
    make_md_task_cfg,
    make_train_cfg,
    run_one,
)

ADV_DIR = REPO_ROOT / "results" / "advisor_review"
OUT_ROOT = ADV_DIR / "md_knapsack_mu_sweep"

METHODS = [
    "FPTO",
    "SAA",
    "WDRO",
    "FDFL",
    "FDFL-0.1",
    "FDFL-0.5",
    "FDFL-Scal",
    "FPLG",
    "PCGrad",
    "MGDA",
]
SEEDS = [11, 22, 33, 44, 55]
LAMBDAS = [0.0, 0.5, 1.0, 2.0]
STEPS = 30
N_TRAIN = 300
N_VAL = 20
N_TEST = 40
ALPHA_FAIR = 2.0


def run_seed(seed: int, overwrite: bool = False) -> tuple[pd.DataFrame, float]:
    task_cfg = make_md_task_cfg(
        n_train=N_TRAIN,
        n_val=N_VAL,
        n_test=N_TEST,
        n_features=5,
        n_resources=2,
        scenario="alpha_fair",
        alpha_fair=ALPHA_FAIR,
        poly_degree=2,
        snr=5.0,
        benefit_group_bias=0.4,
        benefit_noise_ratio=1.0,
        cost_group_bias=0.0,
        cost_noise_ratio=1.0,
        cost_mean=1.0,
        cost_std=0.2,
        budget_tightness=0.35,
        fairness_type="mad",
        group_ratio=0.5,
        decision_mode="group",
        data_seed=42,
    )
    train_cfg = make_train_cfg(
        seeds=[seed],
        lambdas=LAMBDAS,
        steps=STEPS,
        lr=5e-4,
        hidden_dim=32,
        n_layers=2,
        arch="mlp",
        decision_grad_backend="spsa",
        spsa_eps=5e-3,
        spsa_n_dirs=8,
        batch_size=-1,
    )
    sub = OUT_ROOT / f"seed_{seed}"
    t0 = time.time()
    stage_df, _, _ = run_one(
        out_dir=sub,
        task_cfg=task_cfg,
        train_cfg=train_cfg,
        methods=METHODS,
        label=f"md_mu_sweep_seed{seed}",
        overwrite=overwrite,
    )
    return stage_df, time.time() - t0


def build_grand_summary(seeds: list[int]) -> pd.DataFrame:
    """Aggregate per-seed stage_results.csv into one wide table."""
    frames: list[pd.DataFrame] = []
    for s in seeds:
        csv = OUT_ROOT / f"seed_{s}" / "stage_results.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if "seed" not in df.columns:
            df["seed"] = s
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    print("MD knapsack mu-sweep")
    print("====================")
    print(f"Output dir:  {OUT_ROOT}")
    print(f"Methods:     {METHODS}")
    print(f"Seeds:       {SEEDS}")
    print(f"Lambdas:     {LAMBDAS}")
    print(f"Steps:       {STEPS}")
    print(f"n_train:     {N_TRAIN}")
    print(f"alpha_fair:  {ALPHA_FAIR}")
    print(f"SPSA:        n_dirs=8 eps=5e-3")
    print("")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    elapsed_per_seed: list[float] = []
    for s in SEEDS:
        _, dt = run_seed(seed=s, overwrite=args.overwrite)
        elapsed_per_seed.append(dt)
        print(f"  seed {s}: {dt:.1f}s")
    total = time.time() - t_all

    grand = build_grand_summary(SEEDS)
    if not grand.empty:
        grand_csv = OUT_ROOT / "grand_summary.csv"
        grand.to_csv(grand_csv, index=False)
        print(f"\n[grand_summary] {grand_csv}  ({len(grand)} rows)")

    summary = {
        "experiment": "md_knapsack_mu_sweep",
        "methods": METHODS,
        "seeds": SEEDS,
        "lambdas": LAMBDAS,
        "steps": STEPS,
        "n_train": N_TRAIN,
        "alpha_fair": ALPHA_FAIR,
        "spsa_n_dirs": 8,
        "spsa_eps": 5e-3,
        "elapsed_sec_per_seed": elapsed_per_seed,
        "total_sec": float(total),
    }
    with open(OUT_ROOT / "grid_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary]       {OUT_ROOT / 'grid_summary.json'}")
    print(f"TOTAL: {total/60:.1f} min")


if __name__ == "__main__":
    main()
