"""AlignMO Phase 1 Go/No-Go pilot launcher.

Purpose
-------
Before any AlignMO code is written, verify that different fixed
multi-objective handlers win on different cells of
(alpha_fair x decision_grad_backend) on the healthcare task. See
``ALIGNMO_PLAN.md`` Section 4 for the decision rule (Section 4.4) and
deliverables (Section 4.5).

Grid
----
- task: healthcare (medical_resource_allocation), mad fairness only
- n_sample: 1000  (smaller than the main-text 5000 for speed)
- alpha_fair: {0.5, 1.5, 2.0, 3.0}
- decision_grad_backend:
    * analytic
    * spsa (n_dirs=8, eps=1e-3)  -- "low-noise SPSA"
- methods: FPTO, FDFL-Scal, FDFL-0.1, FDFL-0.5, FDFL, FPLG, PCGrad, MGDA
- lambdas: {0.0, 0.5, 1.0, 2.0}
- seeds: {11, 22, 33}
- steps_per_lambda: 50

Output directory
----------------
``results/pilot_alignmo/<regime>/alpha_<a>/seed_<s>/`` containing
``stage_results.csv``, ``iter_logs.csv``, and ``config.json``.

Usage
-----
    python -m experiments.advisor_review.run_alignmo_pilot [--regime analytic|spsa|both]
                                                           [--smoke]
                                                           [--n-sample 1000]
                                                           [--steps 50]
                                                           [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.advisor_review.runner import (  # noqa: E402
    make_healthcare_task_cfg,
    make_train_cfg,
    run_one,
)

# ---------------------------------------------------------------------------
# Pilot grid constants (see ALIGNMO_PLAN.md Section 4.3)
# ---------------------------------------------------------------------------

PILOT_DIR = REPO_ROOT / "results" / "pilot_alignmo"

PILOT_ALPHAS: List[float] = [0.5, 1.5, 2.0, 3.0]
PILOT_SEEDS: List[int] = [11, 22, 33]
PILOT_LAMBDAS: List[float] = [0.0, 0.5, 1.0, 2.0]
PILOT_METHODS: List[str] = [
    "FPTO",
    "FDFL-Scal",
    "FDFL-0.1",
    "FDFL-0.5",
    "FDFL",
    "FPLG",
    "PCGrad",
    "MGDA",
]
PILOT_STEPS: int = 50
PILOT_N_SAMPLE: int = 1000
PILOT_FAIRNESS_TYPE: str = "mad"

# Shared HPs (align with v2 settings for comparability)
PILOT_BUDGET_RHO: float = 0.30
PILOT_LR: float = 1e-3
PILOT_HIDDEN_DIM: int = 64
PILOT_N_LAYERS: int = 2
PILOT_TEST_FRACTION: float = 0.5
PILOT_LR_DECAY: float = 5e-4

REGIMES = {
    "analytic": {
        "decision_grad_backend": "analytic",
        "spsa_n_dirs": 4,   # unused
        "spsa_eps": 5e-3,   # unused
    },
    "spsa": {
        "decision_grad_backend": "spsa",
        "spsa_n_dirs": 8,
        "spsa_eps": 1e-3,
    },
}


def pilot_task_cfg(
    *,
    alpha_fair: float,
    split_seed: int,
    n_sample: int = PILOT_N_SAMPLE,
) -> dict:
    return make_healthcare_task_cfg(
        n_sample=int(n_sample),
        val_fraction=0.0,
        test_fraction=PILOT_TEST_FRACTION,
        alpha_fair=alpha_fair,
        fairness_type=PILOT_FAIRNESS_TYPE,
        budget_rho=PILOT_BUDGET_RHO,
        split_seed=int(split_seed),
        data_seed=42,
    )


def pilot_train_cfg(
    *,
    seeds: Iterable[int],
    regime: str,
    steps: int = PILOT_STEPS,
    lambdas: Iterable[float] = PILOT_LAMBDAS,
    force_lambda_path: bool = False,
) -> dict:
    reg = REGIMES[regime]
    extra: dict = {"lr_decay": PILOT_LR_DECAY}
    if force_lambda_path:
        extra["force_lambda_path_all_methods"] = True
    return make_train_cfg(
        seeds=list(seeds),
        lambdas=list(lambdas),
        steps=int(steps),
        lr=PILOT_LR,
        hidden_dim=PILOT_HIDDEN_DIM,
        n_layers=PILOT_N_LAYERS,
        arch="mlp",
        decision_grad_backend=reg["decision_grad_backend"],
        spsa_eps=reg["spsa_eps"],
        spsa_n_dirs=reg["spsa_n_dirs"],
        eval_train=True,
        extra=extra,
    )


def run_pilot_cell(
    *,
    regime: str,
    alpha_fair: float,
    seed: int,
    methods: Iterable[str] = PILOT_METHODS,
    steps: int = PILOT_STEPS,
    n_sample: int = PILOT_N_SAMPLE,
    lambdas: Iterable[float] = PILOT_LAMBDAS,
    out_root: Path = PILOT_DIR,
    overwrite: bool = False,
    force_lambda_path: bool = False,
) -> tuple:
    task_cfg = pilot_task_cfg(
        alpha_fair=alpha_fair, split_seed=int(seed), n_sample=n_sample
    )
    train_cfg = pilot_train_cfg(
        seeds=[int(seed)], regime=regime, steps=steps, lambdas=lambdas,
        force_lambda_path=force_lambda_path,
    )
    sub = out_root / regime / f"alpha_{alpha_fair}" / f"seed_{seed}"
    label = f"pilot_alignmo_{regime}_a{alpha_fair}_s{seed}"
    return run_one(
        out_dir=sub,
        task_cfg=task_cfg,
        train_cfg=train_cfg,
        methods=list(methods),
        label=label,
        overwrite=overwrite,
    )


def run_pilot_grid(
    *,
    regimes: Iterable[str] = ("analytic", "spsa"),
    alphas: Iterable[float] = PILOT_ALPHAS,
    seeds: Iterable[int] = PILOT_SEEDS,
    methods: Iterable[str] = PILOT_METHODS,
    steps: int = PILOT_STEPS,
    n_sample: int = PILOT_N_SAMPLE,
    lambdas: Iterable[float] = PILOT_LAMBDAS,
    out_root: Path = PILOT_DIR,
    overwrite: bool = False,
    force_lambda_path: bool = False,
) -> list[dict]:
    summary: list[dict] = []
    for regime in regimes:
        for a in alphas:
            for s in seeds:
                t0 = time.time()
                stage_df, _, elapsed = run_pilot_cell(
                    regime=regime,
                    alpha_fair=a,
                    seed=int(s),
                    methods=methods,
                    steps=steps,
                    n_sample=n_sample,
                    lambdas=lambdas,
                    out_root=out_root,
                    overwrite=overwrite,
                    force_lambda_path=force_lambda_path,
                )
                summary.append(
                    {
                        "regime": regime,
                        "alpha": float(a),
                        "seed": int(s),
                        "elapsed_sec": float(elapsed),
                        "n_rows": int(len(stage_df)),
                    }
                )
                print(
                    f"[pilot] regime={regime} alpha={a} seed={s}: "
                    f"{elapsed:.1f}s, {len(stage_df)} rows "
                    f"(cum={time.time()-t0:.1f}s)"
                )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="AlignMO Phase 1 Go/No-Go pilot.")
    p.add_argument(
        "--regime",
        choices=["analytic", "spsa", "both"],
        default="both",
        help="Which decision-gradient regime to run. Default: both.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run a single smoke-test cell (alpha=2.0, analytic, FDFL-Scal, "
        "seed=11, lambda=0.5). Overrides --regime.",
    )
    p.add_argument("--n-sample", type=int, default=PILOT_N_SAMPLE)
    p.add_argument("--steps", type=int, default=PILOT_STEPS)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run cells even if stage_results.csv already exists.",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=PILOT_DIR,
        help="Root output directory.",
    )
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        print("=== AlignMO pilot — SMOKE TEST ===")
        print("Single cell: alpha=2.0, analytic, FDFL-Scal, seed=11, lambda=0.5")
        t0 = time.time()
        stage_df, iter_df, elapsed = run_pilot_cell(
            regime="analytic",
            alpha_fair=2.0,
            seed=11,
            methods=["FDFL-Scal"],
            steps=args.steps,
            n_sample=args.n_sample,
            lambdas=[0.5],
            out_root=out_root / "smoke",
            overwrite=args.overwrite,
        )
        print(
            f"  smoke: {elapsed:.1f}s, stage_rows={len(stage_df)}, "
            f"iter_rows={len(iter_df)}"
        )
        print(f"  total wall: {time.time()-t0:.1f}s")
        return

    regimes = ["analytic", "spsa"] if args.regime == "both" else [args.regime]

    print("=== AlignMO Phase 1 pilot ===")
    print(f"Output root:     {out_root}")
    print(f"Regimes:         {regimes}")
    print(f"Alphas:          {PILOT_ALPHAS}")
    print(f"Seeds:           {PILOT_SEEDS}")
    print(f"Methods:         {PILOT_METHODS}")
    print(f"Lambdas:         {PILOT_LAMBDAS}")
    print(f"Steps/lambda:    {args.steps}")
    print(f"n_sample:        {args.n_sample}")
    total_cells = len(regimes) * len(PILOT_ALPHAS) * len(PILOT_SEEDS)
    total_stages = total_cells * len(PILOT_METHODS) * len(PILOT_LAMBDAS)
    print(f"Total cells:     {total_cells}")
    print(f"Total stages:    {total_stages}")
    print("")

    t0 = time.time()
    summary = run_pilot_grid(
        regimes=regimes,
        alphas=PILOT_ALPHAS,
        seeds=PILOT_SEEDS,
        methods=PILOT_METHODS,
        steps=args.steps,
        n_sample=args.n_sample,
        lambdas=PILOT_LAMBDAS,
        out_root=out_root,
        overwrite=args.overwrite,
    )
    elapsed = time.time() - t0

    total_rows = sum(r["n_rows"] for r in summary)
    print("")
    print("=== pilot summary ===")
    print(f"  cells:          {len(summary)}")
    print(f"  total rows:     {total_rows}")
    print(f"  total elapsed:  {elapsed:.1f}s = {elapsed/60:.2f} min")

    summary_path = out_root / "grid_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "summary": summary,
                "grand_total_sec": float(elapsed),
                "n_cells": len(summary),
                "total_rows": total_rows,
                "regimes": regimes,
                "alphas": PILOT_ALPHAS,
                "seeds": PILOT_SEEDS,
                "methods": PILOT_METHODS,
                "lambdas": PILOT_LAMBDAS,
                "steps_per_lambda": int(args.steps),
                "n_sample": int(args.n_sample),
            },
            f,
            indent=2,
        )
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
