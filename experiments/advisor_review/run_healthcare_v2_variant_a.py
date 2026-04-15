"""Launch the full healthcare v2 Variant A grid.

8 cells {fairness_type} x {alpha} x 3 seeds {11, 22, 33}
Each seed gets its own split_seed (coupled).
Each cell: 7 methods x (4 lambdas where applicable) x 70 steps

Estimated runtime ~14 min on full cohort with analytic gradients.

Usage:
    python -m experiments.advisor_review.run_healthcare_v2_variant_a
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

from experiments.advisor_review.healthcare_followup_v2 import (  # noqa: E402
    HC_V2_ALPHAS,
    HC_V2_DIR,
    HC_V2_FAIRNESS_TYPES,
    HC_V2_LAMBDAS,
    HC_V2_METHODS,
    HC_V2_SEEDS_A,
    HC_V2_A_STEPS,
    run_healthcare_grid_v2,
)


def main() -> None:
    print("Healthcare follow-up v2 — Variant A (baseline, 3 seeds, no val)")
    print("==================================================================")
    print(f"Output dir:      {HC_V2_DIR / 'variant_a'}")
    print(f"Fairness types:  {HC_V2_FAIRNESS_TYPES}")
    print(f"Alphas:          {HC_V2_ALPHAS}")
    print(f"Methods:         {HC_V2_METHODS}")
    print(f"Seeds:           {HC_V2_SEEDS_A}")
    print(f"Lambdas:         {HC_V2_LAMBDAS}")
    print(f"Steps/lambda:    {HC_V2_A_STEPS}")
    print(f"Total cells:     {len(HC_V2_FAIRNESS_TYPES) * len(HC_V2_ALPHAS)}")
    print(f"Runs:            {len(HC_V2_FAIRNESS_TYPES) * len(HC_V2_ALPHAS) * len(HC_V2_SEEDS_A)}")
    print("")

    t0 = time.time()
    summary = run_healthcare_grid_v2(variant="a", overwrite=True)
    elapsed = time.time() - t0

    print("")
    print("=== Variant A summary ===")
    total_rows = sum(r["n_rows"] for r in summary)
    print(f"  runs:           {len(summary)}")
    print(f"  total rows:     {total_rows}")
    print(f"  total elapsed:  {elapsed:.1f}s = {elapsed/60:.2f} min")

    summary_path = HC_V2_DIR / "variant_a" / "grid_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "variant": "a",
                "summary": summary,
                "grand_total_sec": float(elapsed),
                "n_runs": len(summary),
                "total_rows": total_rows,
            },
            f,
            indent=2,
        )
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
