"""Standalone launcher for the healthcare-followup timing check.

Two phases:
  1. Smoke test  — 1 method × 1 seed × 1 lambda × 10 steps. Verifies the
     n_sample=0 / val_fraction=0 / analytic-grad / 7-method plumbing
     before committing to the longer cell.
  2. Timing cell — 7 methods × 5 seeds × 4 lambdas × 70 steps at
     fairness_type=mad, alpha_fair=2.0. This is one cell of the full grid.

Both phases write to results/advisor_review/healthcare_followup/_timing_check/.
The cell result is also placed at the canonical mad/alpha_2.0/ subdirectory so
it counts toward the full grid.

Usage:
    python -m experiments.advisor_review.timing_check_healthcare
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.advisor_review.healthcare_followup import (  # noqa: E402
    HC_DIR,
    HC_LAMBDAS,
    HC_METHODS,
    HC_SEEDS,
    HC_STEPS,
    run_healthcare_cell,
)


def smoke_test() -> float:
    """Tiny end-to-end run: 1 method, 1 seed, 1 lambda, 10 steps."""
    print("\n=== [smoke test] 1 method × 1 seed × 1 lambda × 10 steps ===")
    out = HC_DIR / "_timing_check" / "smoke"
    t0 = time.time()
    stage_df, _, elapsed = run_healthcare_cell(
        fairness_type="mad",
        alpha_fair=2.0,
        seeds=[11],
        lambdas=[0.0],
        steps=10,
        methods=["FPTO"],
        out_root=out.parent,
        overwrite=True,
    )
    # Re-locate the directory written by run_healthcare_cell so we report it
    smoke_out = out.parent / "mad" / "alpha_2.0"
    print(f"[smoke] {len(stage_df)} stage rows in {elapsed:.1f}s -> {smoke_out}")
    return elapsed


def timing_cell() -> float:
    """Full timing cell at fairness=mad, alpha=2.0 — 7 methods × 5 seeds × 4 lambdas × 70 steps."""
    print(
        f"\n=== [timing cell] {len(HC_METHODS)} methods × {len(HC_SEEDS)} seeds × "
        f"{len(HC_LAMBDAS)} lambdas × {HC_STEPS} steps ==="
    )
    print(f"  methods: {HC_METHODS}")
    print(f"  seeds:   {HC_SEEDS}")
    print(f"  lambdas: {HC_LAMBDAS}")
    t0 = time.time()
    stage_df, _, elapsed = run_healthcare_cell(
        fairness_type="mad",
        alpha_fair=2.0,
        seeds=HC_SEEDS,
        lambdas=HC_LAMBDAS,
        steps=HC_STEPS,
        methods=HC_METHODS,
        overwrite=True,
    )
    out = HC_DIR / "mad" / "alpha_2.0"
    print(f"[timing cell] {len(stage_df)} stage rows in {elapsed:.1f}s -> {out}")
    return elapsed


def main() -> None:
    print("Healthcare follow-up timing check")
    print("==================================")
    print(f"Repo root:  {REPO_ROOT}")
    print(f"Output dir: {HC_DIR}")

    t_smoke = smoke_test()
    print(f"\n[smoke total] {t_smoke:.1f}s")

    if t_smoke > 120.0:
        print(
            "[abort] Smoke test took > 120s. Investigate before running the full cell."
        )
        return

    t_cell = timing_cell()
    n_method_steps = len(HC_METHODS) * len(HC_SEEDS) * len(HC_LAMBDAS) * HC_STEPS
    print(f"\n[timing cell total] {t_cell:.1f}s for ~{n_method_steps} method-steps")
    print(f"  -> ~{(t_cell / n_method_steps) * 1000:.2f} ms per method-step")
    print(f"  -> Estimated full grid (8 cells): {(t_cell * 8) / 60:.1f} min")


if __name__ == "__main__":
    main()
