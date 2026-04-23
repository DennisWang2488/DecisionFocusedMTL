"""Targeted healthcare extension: raw vs normalized PCGrad on fresh seeds.

Runs the paper-style healthcare setup on a focused comparison cell:
  - fairness_type = mad
  - alpha_fair = 2.0
  - analytic decision gradients
  - full cohort
  - 70 steps per lambda

The goal is to compare the exact paper-facing healthcare regime using
five fresh seeds and both PCGrad variants side by side:
  - PCGrad-raw  : mo_pcgrad_normalize = False
  - PCGrad-norm : mo_pcgrad_normalize = True

We also include the main comparison methods for context:
  FPTO, SAA, WDRO, FDFL, FDFL-0.1, FDFL-0.5, FDFL-Scal, FPLG, MGDA

Outputs:
    results/advisor_review/healthcare_pcgrad_compare_extra/
        seed_<s>/
            stage_results.csv
            iter_logs.csv
            config.json
        grand_summary.csv
        grid_summary.json
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
    hc_v2_task_cfg,
    hc_v2_train_cfg_a,
)

ADV_DIR = REPO_ROOT / "results" / "advisor_review"
OUT_ROOT = ADV_DIR / "healthcare_pcgrad_compare_extra"

FAIRNESS_TYPE = "mad"
ALPHA_FAIR = 2.0
SEEDS = [66, 77, 88, 99, 111]

METHOD_ORDER = [
    "FPTO",
    "SAA",
    "WDRO",
    "FDFL",
    "FDFL-0.1",
    "FDFL-0.5",
    "FDFL-Scal",
    "FPLG",
    "PCGrad-raw",
    "PCGrad-norm",
    "MGDA",
]


def _serialise(obj):
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if np is not None and isinstance(obj, (np.integer,)):
        return int(obj)
    if np is not None and isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _method_configs() -> dict[str, dict]:
    names = ["FPTO", "SAA", "WDRO", "FDFL", "FDFL-0.1", "FDFL-0.5", "FDFL-Scal", "FPLG", "MGDA"]
    cfgs = {name: copy.deepcopy(ALL_METHOD_CONFIGS[name]) for name in names}

    pcgrad_raw = copy.deepcopy(ALL_METHOD_CONFIGS["PCGrad"])
    pcgrad_raw["mo_pcgrad_normalize"] = False
    cfgs["PCGrad-raw"] = pcgrad_raw

    pcgrad_norm = copy.deepcopy(ALL_METHOD_CONFIGS["PCGrad"])
    pcgrad_norm["mo_pcgrad_normalize"] = True
    cfgs["PCGrad-norm"] = pcgrad_norm
    return cfgs


def run_seed(seed: int, overwrite: bool = False) -> tuple[pd.DataFrame, float]:
    out_dir = OUT_ROOT / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_csv = out_dir / "stage_results.csv"
    if stage_csv.exists() and not overwrite:
        return pd.read_csv(stage_csv), 0.0

    task_cfg = hc_v2_task_cfg(
        fairness_type=FAIRNESS_TYPE,
        alpha_fair=ALPHA_FAIR,
        split_seed=int(seed),
        val_fraction=0.0,
    )
    train_cfg = hc_v2_train_cfg_a(seeds=[int(seed)])
    cfg = {"task": dict(task_cfg), "training": dict(train_cfg)}
    method_configs = _method_configs()

    t0 = time.time()
    stage_df, iter_df = run_experiment_unified(cfg, method_configs=method_configs)
    elapsed = time.time() - t0

    if not stage_df.empty:
        stage_df.to_csv(stage_csv, index=False)
    if not iter_df.empty:
        iter_df.to_csv(out_dir / "iter_logs.csv", index=False)

    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {
                "label": f"healthcare_pcgrad_compare_extra_seed{seed}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_sec": float(elapsed),
                "methods": METHOD_ORDER,
                "task_cfg": _serialise(task_cfg),
                "train_cfg": _serialise(train_cfg),
                "n_stage_rows": int(len(stage_df)),
                "n_iter_rows": int(len(iter_df)),
            },
            f,
            indent=2,
        )
    return stage_df, elapsed


def build_grand_summary(seeds: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for s in seeds:
        csv = OUT_ROOT / f"seed_{s}" / "stage_results.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if "seed" not in df.columns:
            df["seed"] = s
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    print("Healthcare targeted extension: raw vs normalized PCGrad")
    print("=======================================================")
    print(f"Output dir:      {OUT_ROOT}")
    print(f"Fairness type:   {FAIRNESS_TYPE}")
    print(f"Alpha:           {ALPHA_FAIR}")
    print(f"Seeds:           {SEEDS}")
    print(f"Methods:         {METHOD_ORDER}")
    print("")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = []
    t_all = time.time()
    for seed in SEEDS:
        df, elapsed = run_seed(seed=seed, overwrite=args.overwrite)
        summary.append({
            "seed": int(seed),
            "elapsed_sec": float(elapsed),
            "n_rows": int(len(df)),
        })
        print(f"  [seed {seed}] {len(df)} rows in {elapsed:.1f}s")

    total = time.time() - t_all
    grand = build_grand_summary(SEEDS)
    if not grand.empty:
        grand_csv = OUT_ROOT / "grand_summary.csv"
        grand.to_csv(grand_csv, index=False)
        print(f"\n[grand_summary] {grand_csv} ({len(grand)} rows)")

    with open(OUT_ROOT / "grid_summary.json", "w") as f:
        json.dump(
            {
                "experiment": "healthcare_pcgrad_compare_extra",
                "fairness_type": FAIRNESS_TYPE,
                "alpha_fair": ALPHA_FAIR,
                "seeds": SEEDS,
                "methods": METHOD_ORDER,
                "summary": summary,
                "total_sec": float(total),
            },
            f,
            indent=2,
        )
    print(f"[summary]       {OUT_ROOT / 'grid_summary.json'}")
    print(f"TOTAL: {total/60:.2f} min")


if __name__ == "__main__":
    main()
