"""Track 1 — healthcare full-cohort analytic AlignMO comparison.

Reuses the healthcare_followup_v2 variant-A fixed-method numbers at
alpha in {0.5, 2.0} (already on disk). Adds:
  - AlignMO at alpha in {0.5, 2.0}
  - 7 v2 methods + AlignMO at alpha in {1.5, 3.0}

See NEXT_SESSION_STATUS.md Track 1 for pinned settings.

Usage:
    python -m experiments.advisor_review.run_alignmo_hc_full [--overwrite]
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

from experiments.advisor_review.runner import (  # noqa: E402
    make_healthcare_task_cfg,
    make_train_cfg,
    run_one,
)

FULL_HC_DIR = REPO_ROOT / "results" / "alignmo_eval_full_hc"
FULL_HC_METHODS_7 = ["FPTO", "FDFL-Scal", "FPLG", "PCGrad", "MGDA", "SAA", "WDRO"]
FULL_HC_NEW_ALPHAS = [1.5, 3.0]
FULL_HC_EXISTING_ALPHAS = [0.5, 2.0]
FULL_HC_SEEDS = [11, 22, 33]
FULL_HC_LAMBDAS = [0.0, 0.5, 1.0, 2.0]
FULL_HC_STEPS = 70

# Matches HC v2 variant A so it joins cleanly with the existing tables.
FULL_HC_BUDGET_RHO = 0.30
FULL_HC_LR = 1e-3
FULL_HC_LR_DECAY = 5e-4
FULL_HC_HIDDEN_DIM = 64
FULL_HC_N_LAYERS = 2
FULL_HC_TEST_FRACTION = 0.5


def hc_task_cfg(*, alpha_fair: float, split_seed: int) -> dict:
    return make_healthcare_task_cfg(
        n_sample=0,                       # full cohort
        val_fraction=0.0,
        test_fraction=FULL_HC_TEST_FRACTION,
        alpha_fair=alpha_fair,
        fairness_type="mad",
        budget_rho=FULL_HC_BUDGET_RHO,
        split_seed=int(split_seed),
        data_seed=42,
    )


def hc_train_cfg(*, seed: int) -> dict:
    return make_train_cfg(
        seeds=[int(seed)],
        lambdas=list(FULL_HC_LAMBDAS),
        steps=FULL_HC_STEPS,
        lr=FULL_HC_LR,
        hidden_dim=FULL_HC_HIDDEN_DIM,
        n_layers=FULL_HC_N_LAYERS,
        arch="mlp",
        decision_grad_backend="analytic",
        eval_train=True,
        extra={
            "lr_decay": FULL_HC_LR_DECAY,
            "force_lambda_path_all_methods": True,
        },
    )


def run_cell(*, alpha: float, seed: int, methods: list[str], overwrite: bool) -> tuple:
    out = FULL_HC_DIR / f"alpha_{alpha}" / f"seed_{seed}"
    task_cfg = hc_task_cfg(alpha_fair=alpha, split_seed=seed)
    train_cfg = hc_train_cfg(seed=seed)
    label = f"alignmo_full_hc_a{alpha}_s{seed}"
    return run_one(
        out_dir=out,
        task_cfg=task_cfg,
        train_cfg=train_cfg,
        methods=methods,
        label=label,
        overwrite=overwrite,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    FULL_HC_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Track 1 — healthcare full-cohort analytic AlignMO run ===")
    print(f"Output root: {FULL_HC_DIR}")
    print(f"Methods (new-alpha cells):   {FULL_HC_METHODS_7 + ['AlignMO']}")
    print(f"Methods (reuse cells):       ['AlignMO']")
    print(f"Seeds: {FULL_HC_SEEDS}")
    print(f"Lambdas: {FULL_HC_LAMBDAS}")
    print(f"Steps/lambda: {FULL_HC_STEPS}")
    print(f"Existing alphas (AlignMO only): {FULL_HC_EXISTING_ALPHAS}")
    print(f"New alphas (full 8-method):     {FULL_HC_NEW_ALPHAS}")
    print("")

    summary: list[dict] = []
    t0 = time.time()

    # Existing alphas: AlignMO only.
    for a in FULL_HC_EXISTING_ALPHAS:
        for s in FULL_HC_SEEDS:
            tc = time.time()
            stage_df, _, el = run_cell(alpha=a, seed=s, methods=["AlignMO"], overwrite=args.overwrite)
            summary.append({"alpha": a, "seed": s, "n_methods": 1,
                            "n_rows": int(len(stage_df)), "elapsed_sec": float(el)})
            print(f"[Track1] a={a} s={s} (AlignMO only): {el:.1f}s, {len(stage_df)} rows "
                  f"(wall={time.time()-tc:.1f}s)")

    # New alphas: 7 v2 methods + AlignMO.
    full_methods = FULL_HC_METHODS_7 + ["AlignMO"]
    for a in FULL_HC_NEW_ALPHAS:
        for s in FULL_HC_SEEDS:
            tc = time.time()
            stage_df, _, el = run_cell(alpha=a, seed=s, methods=full_methods, overwrite=args.overwrite)
            summary.append({"alpha": a, "seed": s, "n_methods": len(full_methods),
                            "n_rows": int(len(stage_df)), "elapsed_sec": float(el)})
            print(f"[Track1] a={a} s={s} (8 methods): {el:.1f}s, {len(stage_df)} rows "
                  f"(wall={time.time()-tc:.1f}s)")

    elapsed = time.time() - t0
    total_rows = sum(r["n_rows"] for r in summary)
    print("")
    print(f"=== Track 1 summary: {len(summary)} cells, {total_rows} rows, {elapsed/60:.2f} min ===")

    with open(FULL_HC_DIR / "grid_summary.json", "w") as f:
        json.dump(
            {
                "summary": summary,
                "grand_total_sec": float(elapsed),
                "existing_alphas_alignmo_only": FULL_HC_EXISTING_ALPHAS,
                "new_alphas_full_methods": FULL_HC_NEW_ALPHAS,
                "methods_new_alphas": full_methods,
                "seeds": FULL_HC_SEEDS,
                "lambdas": FULL_HC_LAMBDAS,
                "steps_per_lambda": FULL_HC_STEPS,
                "n_sample": 0,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
