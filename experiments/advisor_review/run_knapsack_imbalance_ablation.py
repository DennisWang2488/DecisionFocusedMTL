"""Track 2 — MD knapsack fairness x imbalance ablation.

See NEXT_SESSION_STATUS.md Track 2 for pinned settings.

Grid
----
- methods: AlignMO, FPTO, FDFL, SAA, WDRO, FDFL-Scal (6)
- fairness types: mad, dp, atkinson, bias_parity (4)
- benefit_group_bias x cost_group_bias: {0.0, 0.3, 0.6}^2 (9)
- alpha_fair: {0.5, 2.0} (2)
- seeds: {11, 22, 33}
- lambdas: {0.0, 0.5}   (matches hypothesis 3b "least time" config)
- n_train=200, steps_per_lambda=30, SPSA n_dirs=8, eps=1e-3

Usage:
    python -m experiments.advisor_review.run_knapsack_imbalance_ablation --benchmark
    python -m experiments.advisor_review.run_knapsack_imbalance_ablation [--alpha 0.5 2.0] [--fairness mad ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
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

ABL_DIR = REPO_ROOT / "results" / "alignmo_knapsack_imbalance"
ABL_METHODS = ["AlignMO", "FPTO", "FDFL", "SAA", "WDRO", "FDFL-Scal"]
ABL_FAIRNESS_TYPES = ["mad", "dp", "atkinson", "bias_parity"]
ABL_BIASES = [0.0, 0.3, 0.6]
ABL_ALPHAS = [0.5, 2.0]
ABL_SEEDS = [11, 22, 33]
ABL_LAMBDAS = [0.0, 0.5]
ABL_STEPS = 30
ABL_N_TRAIN = 200
ABL_N_VAL = 50
ABL_N_TEST = 100


def task_cfg(*, alpha_fair: float, fairness_type: str,
             bb: float, cb: float) -> dict:
    return make_md_task_cfg(
        n_train=ABL_N_TRAIN,
        n_val=ABL_N_VAL,
        n_test=ABL_N_TEST,
        n_features=5,
        n_resources=2,
        scenario="alpha_fair",
        alpha_fair=alpha_fair,
        poly_degree=2,
        snr=5.0,
        benefit_group_bias=bb,
        benefit_noise_ratio=1.0,
        cost_group_bias=cb,
        cost_noise_ratio=1.0,
        cost_mean=1.0,
        cost_std=0.2,
        budget_tightness=0.35,
        fairness_type=fairness_type,
        group_ratio=0.5,
        decision_mode="group",
        data_seed=42,
    )


def train_cfg(*, seeds: list[int]) -> dict:
    return make_train_cfg(
        seeds=seeds,
        lambdas=list(ABL_LAMBDAS),
        steps=ABL_STEPS,
        lr=5e-4,
        hidden_dim=32,
        n_layers=2,
        arch="mlp",
        decision_grad_backend="spsa",
        spsa_eps=1e-3,
        spsa_n_dirs=8,
        eval_train=True,
        extra={
            "lr_decay": 5e-4,
            "force_lambda_path_all_methods": True,
        },
    )


def run_cell(*, alpha_fair: float, fairness_type: str, bb: float, cb: float,
             methods: list[str] = None, seeds: list[int] = None,
             overwrite: bool = False) -> tuple:
    methods = methods or ABL_METHODS
    seeds = seeds or ABL_SEEDS
    out = (ABL_DIR / fairness_type / f"alpha_{alpha_fair}"
           / f"bb_{bb}_cb_{cb}")
    label = f"knap_abl_{fairness_type}_a{alpha_fair}_bb{bb}_cb{cb}"
    return run_one(
        out_dir=out,
        task_cfg=task_cfg(alpha_fair=alpha_fair, fairness_type=fairness_type,
                          bb=bb, cb=cb),
        train_cfg=train_cfg(seeds=seeds),
        methods=list(methods),
        label=label,
        overwrite=overwrite,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", action="store_true",
                   help="Run a single cell (alpha=2, mad, bb=0.3, cb=0.3, seed=11) to measure wall time, then exit.")
    p.add_argument("--alpha", type=float, nargs="+", default=ABL_ALPHAS)
    p.add_argument("--fairness", type=str, nargs="+", default=ABL_FAIRNESS_TYPES)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    ABL_DIR.mkdir(parents=True, exist_ok=True)

    if args.benchmark:
        print("=== Track 2 benchmark: 1 cell ===")
        t0 = time.time()
        stage_df, _, el = run_cell(alpha_fair=2.0, fairness_type="mad",
                                    bb=0.3, cb=0.3,
                                    methods=ABL_METHODS, seeds=[11],
                                    overwrite=True)
        print(f"benchmark: {el:.1f}s for {len(stage_df)} rows "
              f"(1 seed, 6 methods, 2 lambdas)")
        # Extrapolate: full ablation = 3 seeds * 4 fairness * 9 bb_cb * 2 alpha = 216 cells of same cost as this 1-seed cell * 3.
        # Approximate extrapolation: one cell_3_seeds = 3 * single_seed cost; and 72 full cells total.
        est_per_3seed = el * 3.0
        est_full = est_per_3seed * len(ABL_FAIRNESS_TYPES) * len(ABL_BIASES) ** 2 * len(ABL_ALPHAS)
        print(f"estimated full ablation wall: "
              f"~{est_full/60:.1f} min ({est_full/3600:.2f} h)")
        print(f"total wall {time.time()-t0:.1f}s")
        return

    total_cells = len(args.alpha) * len(args.fairness) * (len(ABL_BIASES) ** 2)
    print(f"=== Track 2 — knapsack fairness x imbalance ablation ===")
    print(f"cells = {len(args.alpha)} * {len(args.fairness)} * {len(ABL_BIASES)**2}"
          f" = {total_cells}")
    print(f"methods: {ABL_METHODS}")
    print("")

    summary: list[dict] = []
    t0 = time.time()
    for alpha, ft, bb, cb in product(args.alpha, args.fairness, ABL_BIASES, ABL_BIASES):
        tc = time.time()
        stage_df, _, el = run_cell(alpha_fair=alpha, fairness_type=ft,
                                    bb=bb, cb=cb, overwrite=args.overwrite)
        summary.append({
            "alpha": alpha, "fairness": ft, "bb": bb, "cb": cb,
            "elapsed_sec": float(el), "n_rows": int(len(stage_df)),
        })
        print(f"[Track2] a={alpha} {ft} bb={bb} cb={cb}: {el:.1f}s, "
              f"{len(stage_df)} rows (wall={time.time()-tc:.1f}s; cum={(time.time()-t0)/60:.1f}m)")

    elapsed = time.time() - t0
    total_rows = sum(r["n_rows"] for r in summary)
    print("")
    print(f"=== Track 2 summary: {len(summary)} cells, {total_rows} rows, "
          f"{elapsed/60:.2f} min ===")
    with open(ABL_DIR / "grid_summary.json", "w") as f:
        json.dump({"summary": summary, "grand_total_sec": float(elapsed)}, f, indent=2)


if __name__ == "__main__":
    main()
