"""Launch the full healthcare v2 Variant B grid.

8 cells {fairness_type} x {alpha} x 2 seeds {44, 55}
Each seed gets its own split_seed (coupled).
Each cell: 7 methods x (4 lambdas where applicable) x 150 steps,
with val_fraction=0.2, lr_decay=5e-3, lr_warmup_steps=5, and
per-K-step val-based early stopping.

Estimated runtime ~25 min on full cohort with analytic gradients
(longer steps_per_lambda + val eval overhead, offset by fewer seeds).

Usage:
    python -m experiments.advisor_review.run_healthcare_v2_variant_b
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
    HC_V2_SEEDS_B,
    HC_V2_B_STEPS,
    HC_V2_B_EVAL_VAL_K,
    HC_V2_B_EARLY_STOP_MIN_STEPS,
    run_healthcare_grid_v2,
)


def main() -> None:
    print("Healthcare follow-up v2 — Variant B (val + schedule + early stop, 2 seeds)")
    print("=============================================================================")
    print(f"Output dir:       {HC_V2_DIR / 'variant_b'}")
    print(f"Fairness types:   {HC_V2_FAIRNESS_TYPES}")
    print(f"Alphas:           {HC_V2_ALPHAS}")
    print(f"Methods:          {HC_V2_METHODS}")
    print(f"Seeds:            {HC_V2_SEEDS_B}")
    print(f"Lambdas:          {HC_V2_LAMBDAS}")
    print(f"Steps/lambda:     {HC_V2_B_STEPS}")
    print(f"Eval val every:   {HC_V2_B_EVAL_VAL_K} steps")
    print(f"Early stop floor: step {HC_V2_B_EARLY_STOP_MIN_STEPS}")
    print(f"Total cells:      {len(HC_V2_FAIRNESS_TYPES) * len(HC_V2_ALPHAS)}")
    print(f"Runs:             {len(HC_V2_FAIRNESS_TYPES) * len(HC_V2_ALPHAS) * len(HC_V2_SEEDS_B)}")
    print("")

    t0 = time.time()
    summary = run_healthcare_grid_v2(variant="b", overwrite=True)
    elapsed = time.time() - t0

    print("")
    print("=== Variant B summary ===")
    total_rows = sum(r["n_rows"] for r in summary)
    print(f"  runs:           {len(summary)}")
    print(f"  total rows:     {total_rows}")
    print(f"  total elapsed:  {elapsed:.1f}s = {elapsed/60:.2f} min")

    summary_path = HC_V2_DIR / "variant_b" / "grid_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "variant": "b",
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
