"""Healthcare v2 follow-up: FDFL mu-variants append.

Runs the Variant A grid extended with the three new FDFL variants
(FDFL, FDFL-0.1, FDFL-0.5) alongside the existing 7 methods. The
paper-cited ``stage_results.csv`` is preserved untouched; new
outcomes are written to ``stage_results_fdfl_mu.csv`` in each cell
directory so the append is non-destructive.

Methods run (10 total):
    PTO:     FPTO, SAA, WDRO
    Static:  FDFL, FDFL-0.1, FDFL-0.5, FDFL-Scal
    Dynamic: FPLG, PCGrad (normalize=True), MGDA

Includes PCGrad with the new per-objective normalization
(mo_pcgrad_normalize=True via the updated config) to measure the
shift from the previous value.

Output (appended under the existing v2 tree):
    results/advisor_review/healthcare_followup_v2/
        variant_a/<fairness_type>/alpha_<a>/seed_<s>/
            stage_results_fdfl_mu.csv   (new, 10 methods)
            iter_logs_fdfl_mu.csv       (new, 10 methods)

Usage:
    python -m experiments.advisor_review.run_healthcare_v2_fdfl_mu
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from experiments.configs import ALL_METHOD_CONFIGS  # noqa: E402
from fair_dfl.runner import run_experiment_unified  # noqa: E402

from experiments.advisor_review.healthcare_followup_v2 import (  # noqa: E402
    HC_V2_ALPHAS,
    HC_V2_DIR,
    HC_V2_FAIRNESS_TYPES,
    HC_V2_SEEDS_A,
    hc_v2_task_cfg,
    hc_v2_train_cfg_a,
)


HC_V2_METHODS_FDFL_MU = [
    # PTO group
    "FPTO", "SAA", "WDRO",
    # Static decision-focused (with new FDFL-mu variants)
    "FDFL", "FDFL-0.1", "FDFL-0.5", "FDFL-Scal",
    # Dynamic decision-focused
    "FPLG", "PCGrad", "MGDA",
]

STAGE_CSV_NAME = "stage_results_fdfl_mu.csv"
ITER_CSV_NAME = "iter_logs_fdfl_mu.csv"
CONFIG_JSON_NAME = "config_fdfl_mu.json"


def _serialise(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def run_cell_fdfl_mu(
    *, fairness_type: str, alpha_fair: float, seed: int, overwrite: bool = False,
) -> tuple[pd.DataFrame, float]:
    cell_dir = HC_V2_DIR / "variant_a" / fairness_type / f"alpha_{alpha_fair}" / f"seed_{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    stage_csv = cell_dir / STAGE_CSV_NAME
    if stage_csv.exists() and not overwrite:
        return pd.read_csv(stage_csv), 0.0

    task_cfg = hc_v2_task_cfg(
        fairness_type=fairness_type,
        alpha_fair=alpha_fair,
        split_seed=int(seed),
        val_fraction=0.0,
    )
    train_cfg = hc_v2_train_cfg_a(seeds=[int(seed)])

    method_configs = {name: copy.deepcopy(ALL_METHOD_CONFIGS[name]) for name in HC_V2_METHODS_FDFL_MU}
    cfg = {"task": dict(task_cfg), "training": dict(train_cfg)}

    t0 = time.time()
    stage_df, iter_df = run_experiment_unified(cfg, method_configs=method_configs)
    elapsed = time.time() - t0

    if not stage_df.empty:
        stage_df.to_csv(stage_csv, index=False)
    if not iter_df.empty:
        iter_df.to_csv(cell_dir / ITER_CSV_NAME, index=False)

    with open(cell_dir / CONFIG_JSON_NAME, "w") as f:
        json.dump(
            {
                "label": f"hc_v2_a_fdflmu_{fairness_type}_a{alpha_fair}_s{seed}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_sec": float(elapsed),
                "methods": list(HC_V2_METHODS_FDFL_MU),
                "task_cfg": _serialise(task_cfg),
                "train_cfg": _serialise(train_cfg),
                "n_stage_rows": int(len(stage_df)),
                "n_iter_rows": int(len(iter_df)),
            },
            f,
            indent=2,
        )
    return stage_df, elapsed


def build_grand_summary(
    fairness_types: list[str], alphas: list[float], seeds: list[int],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ft in fairness_types:
        for a in alphas:
            for s in seeds:
                csv = HC_V2_DIR / "variant_a" / ft / f"alpha_{a}" / f"seed_{s}" / STAGE_CSV_NAME
                if csv.exists():
                    frames.append(pd.read_csv(csv))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--fairness-types", nargs="+", default=None,
                    help="Subset of fairness types to run (default: all)")
    ap.add_argument("--alphas", nargs="+", type=float, default=None,
                    help="Subset of alphas to run (default: all)")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Subset of seeds to run (default: all)")
    args = ap.parse_args()

    fairness_types = args.fairness_types or list(HC_V2_FAIRNESS_TYPES)
    alphas = args.alphas or list(HC_V2_ALPHAS)
    seeds = args.seeds or list(HC_V2_SEEDS_A)

    print("Healthcare v2 follow-up — FDFL mu variants + normalized PCGrad")
    print("==================================================================")
    print(f"Output dir:      {HC_V2_DIR / 'variant_a'}")
    print(f"Fairness types:  {fairness_types}")
    print(f"Alphas:          {alphas}")
    print(f"Methods:         {HC_V2_METHODS_FDFL_MU}")
    print(f"Seeds:           {seeds}")
    print(f"CSV name:        {STAGE_CSV_NAME}  (non-destructive append)")
    print(f"Total cells:     {len(fairness_types) * len(alphas) * len(seeds)}")
    print("")

    summary = []
    t_all = time.time()
    for ft in fairness_types:
        for a in alphas:
            for s in seeds:
                t0 = time.time()
                df, dt = run_cell_fdfl_mu(
                    fairness_type=ft, alpha_fair=a, seed=int(s), overwrite=args.overwrite,
                )
                summary.append(
                    {
                        "fairness_type": ft,
                        "alpha": float(a),
                        "seed": int(s),
                        "elapsed_sec": float(dt),
                        "n_rows": int(len(df)),
                    }
                )
                print(f"  [{ft} alpha={a} seed={s}] {len(df)} rows in {dt:.1f}s")

    total = time.time() - t_all
    print("")
    print(f"=== Summary ===")
    print(f"  cells:         {len(summary)}")
    print(f"  total rows:    {sum(r['n_rows'] for r in summary)}")
    print(f"  total elapsed: {total:.1f}s = {total/60:.2f} min")

    grand = build_grand_summary(fairness_types, alphas, seeds)
    if not grand.empty:
        grand_csv = HC_V2_DIR / "grand_summary_fdfl_mu.csv"
        grand.to_csv(grand_csv, index=False)
        print(f"[grand_summary] {grand_csv}  ({len(grand)} rows)")

    summary_path = HC_V2_DIR / "grid_summary_fdfl_mu.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "variant": "a_fdfl_mu",
                "methods": HC_V2_METHODS_FDFL_MU,
                "fairness_types": fairness_types,
                "alphas": alphas,
                "seeds": seeds,
                "summary": summary,
                "grand_total_sec": float(total),
                "n_cells": len(summary),
            },
            f,
            indent=2,
        )
    print(f"[summary]       {summary_path}")


if __name__ == "__main__":
    main()
