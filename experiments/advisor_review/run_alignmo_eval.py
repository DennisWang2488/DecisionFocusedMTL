"""AlignMO Phase 2 evaluation — run pilot grid WITH AlignMO added.

Re-uses the pilot launcher's helpers so the grid stays identical except
for the method list (pilot methods + AlignMO) and the output root. Per
ALIGNMO_PLAN.md Section 5.3 TODO 2.4.

Grid (= pilot Section 4.3 + AlignMO):
- task: healthcare, mad fairness, n_sample=1000
- alpha_fair: {0.5, 1.5, 2.0, 3.0}
- regimes: analytic, spsa (n_dirs=8, eps=1e-3)
- methods: FPTO, FDFL-Scal, FDFL-0.1, FDFL-0.5, FDFL, FPLG, PCGrad, MGDA, AlignMO
- lambdas: {0.0, 0.5, 1.0, 2.0}
- seeds: {11, 22, 33}
- steps_per_lambda: 50

Output: ``results/alignmo_eval/<regime>/alpha_<a>/seed_<s>/``.

Usage:
    python -m experiments.advisor_review.run_alignmo_eval [--regime analytic|spsa|both]
                                                          [--alignmo-only]
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

from experiments.advisor_review.run_alignmo_pilot import (  # noqa: E402
    PILOT_ALPHAS,
    PILOT_LAMBDAS,
    PILOT_METHODS,
    PILOT_N_SAMPLE,
    PILOT_SEEDS,
    PILOT_STEPS,
    run_pilot_grid,
)

EVAL_DIR = REPO_ROOT / "results" / "alignmo_eval"
EVAL_METHODS = list(PILOT_METHODS) + ["AlignMO"]


def main() -> None:
    p = argparse.ArgumentParser(description="AlignMO Phase 2 eval.")
    p.add_argument("--regime", choices=["analytic", "spsa", "both"], default="both")
    p.add_argument(
        "--alignmo-only",
        action="store_true",
        help="Only run AlignMO (the fixed methods already live in results/pilot_alignmo/).",
    )
    p.add_argument("--n-sample", type=int, default=PILOT_N_SAMPLE)
    p.add_argument("--steps", type=int, default=PILOT_STEPS)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--out-root", type=Path, default=EVAL_DIR)
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    regimes = ["analytic", "spsa"] if args.regime == "both" else [args.regime]
    methods = ["AlignMO"] if args.alignmo_only else EVAL_METHODS

    print("=== AlignMO Phase 2 evaluation ===")
    print(f"Output root: {out_root}")
    print(f"Regimes:     {regimes}")
    print(f"Alphas:      {PILOT_ALPHAS}")
    print(f"Seeds:       {PILOT_SEEDS}")
    print(f"Methods:     {methods}")
    print(f"Lambdas:     {PILOT_LAMBDAS}")
    print(f"Steps/lam:   {args.steps}")
    print(f"n_sample:    {args.n_sample}")
    total_cells = len(regimes) * len(PILOT_ALPHAS) * len(PILOT_SEEDS)
    total_stages = total_cells * len(methods) * len(PILOT_LAMBDAS)
    print(f"Cells:       {total_cells} (stages ~= {total_stages})")
    print("")

    t0 = time.time()
    summary = run_pilot_grid(
        regimes=regimes,
        alphas=PILOT_ALPHAS,
        seeds=PILOT_SEEDS,
        methods=methods,
        steps=args.steps,
        n_sample=args.n_sample,
        lambdas=PILOT_LAMBDAS,
        out_root=out_root,
        overwrite=args.overwrite,
        force_lambda_path=True,  # force AlignMO (and any other mo_method) to sweep lambda
    )
    elapsed = time.time() - t0
    total_rows = sum(r["n_rows"] for r in summary)
    print("")
    print(f"=== eval summary: {len(summary)} cells, {total_rows} rows, "
          f"{elapsed/60:.2f} min ===")

    with open(out_root / "grid_summary.json", "w") as f:
        json.dump(
            {
                "summary": summary,
                "grand_total_sec": float(elapsed),
                "regimes": regimes,
                "alphas": PILOT_ALPHAS,
                "seeds": PILOT_SEEDS,
                "methods": methods,
                "lambdas": PILOT_LAMBDAS,
                "steps_per_lambda": int(args.steps),
                "n_sample": int(args.n_sample),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
