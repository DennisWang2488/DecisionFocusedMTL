"""Launch the full healthcare follow-up grid.

8 cells: 4 fairness types {mad, dp, atkinson, bias_parity} x 2 alphas {0.5, 2.0}
Each cell: 7 methods x 5 seeds x 4 lambdas (where applicable) x 70 steps

Estimated runtime ~30 min on the full 48k cohort with analytic gradients.
Outputs go to results/advisor_review/healthcare_followup/<fairness_type>/alpha_<a>/.

Usage:
    python -m experiments.advisor_review.run_healthcare_grid
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.advisor_review.healthcare_followup import (  # noqa: E402
    HC_ALPHAS,
    HC_DIR,
    HC_FAIRNESS_TYPES,
    HC_LAMBDAS,
    HC_METHODS,
    HC_SEEDS,
    HC_STEPS,
    run_healthcare_grid,
)


def main() -> None:
    print("Healthcare follow-up — full grid")
    print("================================")
    print(f"Repo root:       {REPO_ROOT}")
    print(f"Output dir:      {HC_DIR}")
    print(f"Fairness types:  {HC_FAIRNESS_TYPES}")
    print(f"Alphas:          {HC_ALPHAS}")
    print(f"Methods:         {HC_METHODS}")
    print(f"Seeds:           {HC_SEEDS}")
    print(f"Lambdas:         {HC_LAMBDAS}")
    print(f"Steps/lambda:    {HC_STEPS}")
    print(f"Total cells:     {len(HC_FAIRNESS_TYPES) * len(HC_ALPHAS)}")
    print("")

    t0 = time.time()
    summary = run_healthcare_grid(overwrite=True)
    elapsed = time.time() - t0

    print("")
    print("=== Grid summary ===")
    for row in summary:
        print(
            f"  {row['fairness_type']:>13} alpha={row['alpha']}: "
            f"{row['elapsed_sec']:>6.1f}s, {row['n_rows']} rows"
        )
    print(f"\n[grand total] {elapsed:.1f}s = {elapsed/60:.2f} min")

    # Persist summary
    summary_path = HC_DIR / "grid_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "summary": summary,
                "grand_total_sec": float(elapsed),
                "fairness_types": HC_FAIRNESS_TYPES,
                "alphas": HC_ALPHAS,
                "methods": HC_METHODS,
                "seeds": HC_SEEDS,
                "lambdas": HC_LAMBDAS,
                "steps": HC_STEPS,
            },
            f,
            indent=2,
        )
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
