"""MD knapsack n_train diagnostic — does adding data shrink the train→test gap?

Runs the 6-method set at n_train=200, crossed with:
  - alpha_fair ∈ {0.5, 2.0}
  - fairness_type ∈ {"mad", "bp"}   ("mad" ≈ accuracy parity, "bp" = bias parity)
Single seed, lambdas ∈ {0.0, 1.0}, 30 steps.

Output:
  results/advisor_review/hp_tuning/md_knapsack/sweep06_ntrain_diagnostic/
    n200_a0.5_mad/ n200_a0.5_bp/ n200_a2.0_mad/ n200_a2.0_bp/

Usage:
    python -m experiments.advisor_review.run_md_ntrain_diagnostic [--overwrite]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.advisor_review.runner import (  # noqa: E402
    make_md_task_cfg,
    make_train_cfg,
    run_one,
)

ADV_DIR = REPO_ROOT / "results" / "advisor_review"
OUT_ROOT = ADV_DIR / "hp_tuning" / "md_knapsack" / "sweep06_ntrain_diagnostic"

# Fixed at chosen_config except: new method list, 1 seed, 2 lambdas, n_train=200.
METHODS = ["FPTO", "FDFL", "SAA", "WDRO", "PCGrad", "FPLG"]
SEEDS = [11]
LAMBDAS = [0.0, 1.0]
STEPS = 30
N_TRAIN = 200

ALPHAS = [0.5, 2.0]
FAIRNESS_TYPES = ["mad", "bp"]  # mad ≈ accuracy parity, bp = bias parity


def run_cell(alpha: float, fairness_type: str, overwrite: bool = False) -> float:
    task_cfg = make_md_task_cfg(
        n_train=N_TRAIN,
        n_val=10,
        n_test=20,
        n_features=5,
        n_resources=2,
        scenario="alpha_fair",
        alpha_fair=alpha,
        poly_degree=2,
        snr=5.0,
        benefit_group_bias=0.4,
        benefit_noise_ratio=1.0,
        cost_group_bias=0.0,
        cost_noise_ratio=1.0,
        cost_mean=1.0,
        cost_std=0.2,
        budget_tightness=0.35,
        fairness_type=fairness_type,
        group_ratio=0.5,
        decision_mode="group",
        data_seed=42,
    )
    train_cfg = make_train_cfg(
        seeds=SEEDS,
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
    sub = OUT_ROOT / f"n{N_TRAIN}_a{alpha}_{fairness_type}"
    t0 = time.time()
    run_one(
        out_dir=sub,
        task_cfg=task_cfg,
        train_cfg=train_cfg,
        methods=METHODS,
        label=f"n{N_TRAIN}_a{alpha}_{fairness_type}",
        overwrite=overwrite,
    )
    return time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    print(f"MD n_train diagnostic — n_train={N_TRAIN}")
    print(f"Output root: {OUT_ROOT}")
    print(f"Methods:     {METHODS}")
    print(f"Seeds:       {SEEDS}")
    print(f"Lambdas:     {LAMBDAS}")
    print(f"Steps:       {STEPS}")
    print(f"Alphas:      {ALPHAS}")
    print(f"Fair types:  {FAIRNESS_TYPES}")
    print(f"Total cells: {len(ALPHAS) * len(FAIRNESS_TYPES)}")
    print("")

    t_all = time.time()
    for alpha in ALPHAS:
        for ft in FAIRNESS_TYPES:
            elapsed = run_cell(alpha, ft, overwrite=args.overwrite)
            print(f"  cell a={alpha} fair={ft}: {elapsed:.1f}s")
    print(f"\nTOTAL: {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
