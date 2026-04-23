"""Aggregate mu-sweep results: MD knapsack (Exp 1) and healthcare (Exp 2).

Produces a compact markdown-friendly table of test_regret (mean ± std
across seeds) per method, at lambda=0 (the no-fairness budget). Also
emits a method-level flag line that makes the hypothesized pattern
easy to read off:

    FDFL (mu=0) >> 0.2  -> divergent?
    FDFL-0.1 < 0.2      -> stabilized?
    FDFL-0.5, FDFL-Scal clustered near FPLG?
    PCGrad normalized vs. previous value (prints both if available)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ADV_DIR = REPO_ROOT / "results" / "advisor_review"

MD_GRAND = ADV_DIR / "md_knapsack_mu_sweep" / "grand_summary.csv"
HC_GRAND = ADV_DIR / "healthcare_followup_v2" / "grand_summary_fdfl_mu.csv"


def _summarize(df: pd.DataFrame, metric: str, group_by_fair: bool = False) -> pd.DataFrame:
    keys = ["method"]
    if group_by_fair and "fairness_type" in df.columns:
        keys.append("fairness_type")
    if "alpha_fair" in df.columns:
        keys.append("alpha_fair")
    sub = df[df["lambda"] == 0.0] if "lambda" in df.columns else df
    agg = sub.groupby(keys)[metric].agg(["mean", "std", "count"]).reset_index()
    agg["mean_pm_std"] = agg.apply(
        lambda r: f"{r['mean']:.4f} \u00b1 {r['std']:.4f}  (n={int(r['count'])})",
        axis=1,
    )
    return agg


def _print_table(df: pd.DataFrame, title: str, metric: str) -> None:
    print(f"\n=== {title} ===")
    if df.empty:
        print("  (no data yet)")
        return
    print(f"  metric: {metric}")
    agg = _summarize(df, metric=metric)
    for _, r in agg.iterrows():
        print(f"  {r['method']:12s}  {r['mean_pm_std']}")


def main() -> None:
    print("mu sweep analysis")
    print("=================")

    if MD_GRAND.exists():
        md = pd.read_csv(MD_GRAND)
        metric = "test_regret_normalized" if "test_regret_normalized" in md.columns else "test_regret"
        _print_table(md, f"MD knapsack  ({MD_GRAND.name})", metric=metric)
    else:
        print(f"\n[missing] {MD_GRAND}")

    if HC_GRAND.exists():
        hc = pd.read_csv(HC_GRAND)
        if "fairness_type" in hc.columns:
            hc = hc[hc["fairness_type"] == "mad"]
        if "alpha_fair" in hc.columns:
            hc = hc[hc["alpha_fair"] == 2.0]
        metric = "test_regret_normalized" if "test_regret_normalized" in hc.columns else "test_regret"
        _print_table(hc, f"Healthcare mad a=2  ({HC_GRAND.name})", metric=metric)
    else:
        print(f"\n[missing] {HC_GRAND}")


if __name__ == "__main__":
    main()
