"""Healthcare follow-up driver: post-bias_parity full-cohort experiments.

Per the empirical-followup plan (see fair-dfl/empirical-followup branch):
- Full 48,784-patient cohort (``n_sample=0``)
- No validation split (``val_fraction=0.0``); fixed step budget instead
- 50/50 train/test split
- Analytic decision gradients (the healthcare LP has them in closed form)
- 7 methods × 5 seeds × 4 lambdas per cell
- Full grid: 4 fairness types × 2 alphas = 8 cells

This module exposes the cell/grid runners as plain functions so they can be
invoked from the command line, a notebook, or a wrapper script.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

from .runner import make_healthcare_task_cfg, make_train_cfg, run_one

REPO_ROOT = Path(__file__).resolve().parents[2]
ADV_DIR = REPO_ROOT / "results" / "advisor_review"
HC_DIR = ADV_DIR / "healthcare_followup"

# ---------------------------------------------------------------------------
# Config — the "recommended training config" agreed in the empirical-followup
# planning round. Pinned here so every cell uses the same training setup.
# ---------------------------------------------------------------------------

HC_METHODS: list[str] = [
    "FPTO",
    "FDFL-Scal",
    "FPLG",
    "PCGrad",
    "MGDA",
    "SAA",
    "WDRO",
]

HC_FAIRNESS_TYPES: list[str] = ["mad", "dp", "atkinson", "bias_parity"]
HC_ALPHAS: list[float] = [0.5, 2.0]

HC_SEEDS: list[int] = [11, 22, 33, 44, 55]
HC_LAMBDAS: list[float] = [0.0, 0.5, 1.0, 5.0]
HC_STEPS: int = 70

HC_LR: float = 5e-4
HC_HIDDEN_DIM: int = 64
HC_N_LAYERS: int = 2
HC_BUDGET_RHO: float = 0.35
HC_TEST_FRACTION: float = 0.5
HC_VAL_FRACTION: float = 0.0  # no validation; fixed step budget


def hc_train_cfg(
    *,
    seeds: Iterable[int] = HC_SEEDS,
    lambdas: Iterable[float] = HC_LAMBDAS,
    steps: int = HC_STEPS,
) -> dict:
    """Standard healthcare training config (analytic decision gradients)."""
    return make_train_cfg(
        seeds=list(seeds),
        lambdas=list(lambdas),
        steps=steps,
        lr=HC_LR,
        hidden_dim=HC_HIDDEN_DIM,
        n_layers=HC_N_LAYERS,
        arch="mlp",
        decision_grad_backend="analytic",
        eval_train=True,
    )


def hc_task_cfg(
    *,
    fairness_type: str,
    alpha_fair: float,
    n_sample: int = 0,
    val_fraction: float = HC_VAL_FRACTION,
    test_fraction: float = HC_TEST_FRACTION,
    budget_rho: float = HC_BUDGET_RHO,
) -> dict:
    """Standard healthcare task config (full cohort, no validation)."""
    return make_healthcare_task_cfg(
        n_sample=n_sample,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        alpha_fair=alpha_fair,
        fairness_type=fairness_type,
        budget_rho=budget_rho,
    )


def run_healthcare_cell(
    *,
    fairness_type: str,
    alpha_fair: float,
    seeds: Iterable[int] = HC_SEEDS,
    lambdas: Iterable[float] = HC_LAMBDAS,
    steps: int = HC_STEPS,
    methods: Iterable[str] = HC_METHODS,
    out_root: Path = HC_DIR,
    overwrite: bool = False,
) -> tuple:
    """Run a single (fairness_type, alpha_fair) cell."""
    sub = out_root / fairness_type / f"alpha_{alpha_fair}"
    task_cfg = hc_task_cfg(fairness_type=fairness_type, alpha_fair=alpha_fair)
    train_cfg = hc_train_cfg(
        seeds=list(seeds), lambdas=list(lambdas), steps=steps
    )
    return run_one(
        out_dir=sub,
        task_cfg=task_cfg,
        train_cfg=train_cfg,
        methods=list(methods),
        label=f"hc_{fairness_type}_a{alpha_fair}",
        overwrite=overwrite,
    )


def run_healthcare_grid(
    *,
    fairness_types: Iterable[str] = HC_FAIRNESS_TYPES,
    alphas: Iterable[float] = HC_ALPHAS,
    seeds: Iterable[int] = HC_SEEDS,
    lambdas: Iterable[float] = HC_LAMBDAS,
    steps: int = HC_STEPS,
    methods: Iterable[str] = HC_METHODS,
    out_root: Path = HC_DIR,
    overwrite: bool = False,
) -> list[dict]:
    """Run all (fairness_type, alpha_fair) cells in the grid."""
    summary: list[dict] = []
    for ft in fairness_types:
        for a in alphas:
            t0 = time.time()
            stage_df, _, elapsed = run_healthcare_cell(
                fairness_type=ft,
                alpha_fair=a,
                seeds=seeds,
                lambdas=lambdas,
                steps=steps,
                methods=methods,
                out_root=out_root,
                overwrite=overwrite,
            )
            summary.append(
                {
                    "fairness_type": ft,
                    "alpha": a,
                    "elapsed_sec": float(elapsed),
                    "n_rows": int(len(stage_df)),
                }
            )
            print(
                f"[hc_grid] {ft} alpha={a}: {elapsed:.1f}s, {len(stage_df)} rows"
            )
    return summary
