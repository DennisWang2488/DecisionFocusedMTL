"""Run Variant A on seeds [44, 55] — extra seeds for the 5-seed total.

These extend the existing Variant A run (seeds 11, 22, 33) to 5 seeds
total, matching v1's seed count. The training config is Variant A's
(no val, no early stop, steps_per_lambda=70, lr_decay=5e-4), not
Variant B's. So the split_seed=44 and 55 train/test partitions that
Variant B used are re-used here under a different training setup.

Usage:
    python -m experiments.advisor_review.run_healthcare_v2_variant_a_seeds_44_55
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
    HC_V2_METHODS,
    run_healthcare_grid_v2,
)


def main() -> None:
    extra_seeds = [44, 55]
    print("Healthcare follow-up v2 — Variant A EXTRA seeds [44, 55]")
    print("=========================================================")
    print(f"Output dir:    {HC_V2_DIR / 'variant_a'}")
    print(f"Seeds:         {extra_seeds}")
    print(f"Runs:          {len(HC_V2_FAIRNESS_TYPES) * len(HC_V2_ALPHAS) * len(extra_seeds)}")
    print("")

    t0 = time.time()
    summary = run_healthcare_grid_v2(variant="a", seeds=extra_seeds, overwrite=True)
    elapsed = time.time() - t0

    print("")
    print("=== Variant A extra-seeds summary ===")
    total_rows = sum(r["n_rows"] for r in summary)
    print(f"  runs:           {len(summary)}")
    print(f"  total rows:     {total_rows}")
    print(f"  total elapsed:  {elapsed:.1f}s = {elapsed/60:.2f} min")

    summary_path = HC_V2_DIR / "variant_a" / "grid_summary_seeds_44_55.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "variant": "a",
                "extra_seeds": extra_seeds,
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
