"""Analysis for the healthcare follow-up v2 grid.

Differences from v1 analyze_healthcare_grid.py:
- Per-seed subdirectory layout (split_seed coupled to seed)
- Loads both variants A and B
- Includes train metrics alongside test
- Adds cross-fairness aggregation: rank + ratio normalisation
- Adds constrained fairness selection (fairness-best lambda under a
  regret-slack constraint) so fairness-heavy methods don't always
  show up at lambda=0

Usage:
    python -m experiments.advisor_review.analyze_healthcare_v2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.advisor_review.healthcare_followup_v2 import (  # noqa: E402
    HC_V2_ALPHAS,
    HC_V2_DIR,
    HC_V2_FAIRNESS_TYPES,
    HC_V2_METHODS,
    HC_V2_SEEDS_A,
    HC_V2_SEEDS_B,
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def variant_dir(variant: str) -> Path:
    return HC_V2_DIR / f"variant_{variant.lower().strip()}"


def cell_path(variant: str, fairness_type: str, alpha: float, seed: int) -> Path:
    return (
        variant_dir(variant)
        / fairness_type
        / f"alpha_{alpha}"
        / f"seed_{seed}"
        / "stage_results.csv"
    )


def load_variant(variant: str) -> pd.DataFrame:
    """Load all (fairness_type, alpha, seed) cells for one variant."""
    variant = variant.lower().strip()
    seeds = HC_V2_SEEDS_A if variant == "a" else HC_V2_SEEDS_B
    frames: list[pd.DataFrame] = []
    for ft in HC_V2_FAIRNESS_TYPES:
        for a in HC_V2_ALPHAS:
            for s in seeds:
                path = cell_path(variant, ft, a, s)
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                df["variant"] = variant
                df["fairness_type"] = ft
                df["alpha_fair"] = float(a)
                df["split_seed"] = int(s)
                frames.append(df)
    if not frames:
        raise RuntimeError(f"No data for variant {variant}")
    return pd.concat(frames, ignore_index=True)


def load_both_variants() -> pd.DataFrame:
    """Load both variants A and B, concatenated."""
    frames = []
    for v in ("a", "b"):
        try:
            frames.append(load_variant(v))
        except RuntimeError as exc:
            print(f"[skip variant {v}] {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _train_test_gap(df: pd.DataFrame) -> pd.Series:
    return (
        df["test_regret_normalized"] - df["train_regret_normalized"]
    ).fillna(0.0)


def aggregate_cell_v2(df_cell: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std across seeds, per (method, lambda). Include train metrics."""
    return (
        df_cell.groupby(["method", "lambda"], dropna=False)
        .agg(
            train_reg_n_mean=("train_regret_normalized", "mean"),
            train_reg_n_std=("train_regret_normalized", "std"),
            test_reg_n_mean=("test_regret_normalized", "mean"),
            test_reg_n_std=("test_regret_normalized", "std"),
            train_fair_mean=("train_fairness", "mean"),
            test_fair_mean=("test_fairness", "mean"),
            test_fair_std=("test_fairness", "std"),
            train_mse_mean=("train_pred_mse", "mean"),
            test_mse_mean=("test_pred_mse", "mean"),
            n_seeds=("split_seed", "nunique"),
            nan_steps=("nan_or_inf_steps", "sum"),
            explode_steps=("exploding_steps", "sum"),
        )
        .round(5)
    )


def best_pareto_per_method_v2(df_cell: pd.DataFrame) -> pd.DataFrame:
    """For each method, pick the lambda with the lowest mean test_regret_normalized.

    Parallels v1's selection rule. Good for answering "which method wins
    regret at this (fairness, alpha)?"
    """
    rows = []
    for method in df_cell["method"].unique():
        sub = (
            df_cell[df_cell["method"] == method]
            .groupby("lambda")
            .agg(
                test_reg_n=("test_regret_normalized", "mean"),
                test_reg_n_std=("test_regret_normalized", "std"),
                test_fair=("test_fairness", "mean"),
                test_fair_std=("test_fairness", "std"),
                train_reg_n=("train_regret_normalized", "mean"),
                train_fair=("train_fairness", "mean"),
                early_stop_step=("early_stop_step", "mean"),
            )
        )
        if sub.empty:
            continue
        best_lam = float(sub["test_reg_n"].idxmin())
        best = sub.loc[best_lam]
        rows.append(
            {
                "method": method,
                "best_lambda": best_lam,
                "test_reg_n": float(best["test_reg_n"]),
                "test_reg_n_std": float(best["test_reg_n_std"]),
                "test_fair": float(best["test_fair"]),
                "test_fair_std": float(best["test_fair_std"]),
                "train_reg_n": float(best["train_reg_n"]),
                "train_fair": float(best["train_fair"]),
                "train_test_gap": float(best["test_reg_n"] - best["train_reg_n"]),
                "early_stop_step": float(best["early_stop_step"]),
            }
        )
    return pd.DataFrame(rows).round(5).sort_values("test_reg_n")


def best_fair_per_method_constrained(
    df_cell: pd.DataFrame, regret_slack: float = 0.10
) -> pd.DataFrame:
    """For each method, pick the lambda that minimises test_fairness subject to
    regret not exceeding (1 + slack) * FPTO_lam_0_regret.

    This is the "fairness-best operating point" — avoids the
    pathological selection of lam=0 for fairness-heavy methods.
    """
    # FPTO lambda=0 regret as constraint reference (per-cell)
    fpto_lam_0 = df_cell[(df_cell["method"] == "fpto") & (df_cell["lambda"] == 0.0)]
    if fpto_lam_0.empty:
        baseline_reg = float("nan")
    else:
        baseline_reg = float(fpto_lam_0["test_regret_normalized"].mean())
    regret_budget = baseline_reg * (1.0 + regret_slack) if baseline_reg == baseline_reg else float("inf")

    rows = []
    for method in df_cell["method"].unique():
        sub = (
            df_cell[df_cell["method"] == method]
            .groupby("lambda")
            .agg(
                test_reg_n=("test_regret_normalized", "mean"),
                test_fair=("test_fairness", "mean"),
                test_fair_std=("test_fairness", "std"),
            )
        )
        if sub.empty:
            continue
        feasible = sub[sub["test_reg_n"] <= regret_budget]
        if feasible.empty:
            # Fall back to the lambda with lowest regret (method can't meet the constraint)
            best_lam = float(sub["test_reg_n"].idxmin())
        else:
            best_lam = float(feasible["test_fair"].idxmin())
        best = sub.loc[best_lam]
        rows.append(
            {
                "method": method,
                "best_lambda_fair_constrained": best_lam,
                "test_reg_n_at_fair_best": float(best["test_reg_n"]),
                "test_fair_at_fair_best": float(best["test_fair"]),
                "test_fair_std_at_fair_best": float(best["test_fair_std"]),
                "regret_budget": float(regret_budget),
                "baseline_regret_fpto_lam0": baseline_reg,
            }
        )
    return pd.DataFrame(rows).round(5)


def grand_summary_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Combine per-cell best Pareto points across all cells, with variant."""
    rows = []
    for (variant, ft, a), cell in df.groupby(["variant", "fairness_type", "alpha_fair"]):
        bp = best_pareto_per_method_v2(cell)
        bp.insert(0, "variant", variant)
        bp.insert(1, "fairness_type", ft)
        bp.insert(2, "alpha", a)
        bfc = best_fair_per_method_constrained(cell)
        bp = bp.merge(bfc, on="method", how="left")
        rows.append(bp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Cross-fairness aggregation
# ---------------------------------------------------------------------------


def cross_fairness_rank(
    df: pd.DataFrame,
    metric: str = "test_fair_at_fair_best",
    variant: str | None = None,
) -> pd.DataFrame:
    """Rank methods 1-7 within each (fairness_type, alpha) cell on fairness.

    Lower fairness = better = rank 1. Uses average tiebreak.
    Aggregates mean rank and std rank across the 4 fairness types per alpha.

    Expects a grand_summary-style frame with columns including
    ``test_fair_at_fair_best`` and ``method``.
    """
    if variant is not None:
        df = df[df["variant"] == variant]
    rank_rows = []
    for (v, ft, a), cell in df.groupby(["variant", "fairness_type", "alpha"]):
        values = cell[metric].values
        ranks = rankdata(values, method="average")
        for m, r, val in zip(cell["method"].values, ranks, values):
            rank_rows.append(
                {
                    "variant": v,
                    "fairness_type": ft,
                    "alpha": a,
                    "method": m,
                    "rank": float(r),
                    "value": float(val),
                }
            )
    rank_df = pd.DataFrame(rank_rows)
    if rank_df.empty:
        return pd.DataFrame()

    agg = (
        rank_df.groupby(["variant", "method", "alpha"])
        .agg(
            mean_rank=("rank", "mean"),
            std_rank=("rank", "std"),
            min_rank=("rank", "min"),
            max_rank=("rank", "max"),
            borda_sum=("rank", "sum"),
        )
        .round(3)
    )
    return agg.reset_index()


def cross_fairness_ratio(
    df_cells: pd.DataFrame,
    baseline_method: str = "fpto",
    baseline_lam: float = 0.0,
    variant: str | None = None,
) -> pd.DataFrame:
    """Per-seed ratio normalisation of fairness.

    For each (fairness_type, alpha, split_seed), compute the baseline
    fairness (baseline_method at baseline_lam on the SAME seed) and
    divide all methods' fairness by it. Pair seeds, then average.

    Uses the stage-level df (all rows, not grand_summary).
    """
    if variant is not None:
        df_cells = df_cells[df_cells["variant"] == variant]

    rows = []
    for (v, ft, a, s), cell in df_cells.groupby(
        ["variant", "fairness_type", "alpha_fair", "split_seed"]
    ):
        baseline = cell[
            (cell["method"] == baseline_method)
            & (cell["lambda"] == baseline_lam)
        ]
        if baseline.empty:
            continue
        baseline_fair = float(baseline["test_fairness"].mean())
        if not np.isfinite(baseline_fair) or abs(baseline_fair) < 1e-9:
            continue
        for method in cell["method"].unique():
            sub = cell[cell["method"] == method]
            if sub.empty:
                continue
            # Take fairness at the lambda minimising regret (matches v1 selection)
            best_row = sub.loc[sub["test_regret_normalized"].idxmin()]
            rows.append(
                {
                    "variant": v,
                    "fairness_type": ft,
                    "alpha": a,
                    "split_seed": s,
                    "method": method,
                    "rel_fair_vs_baseline": float(best_row["test_fairness"]) / baseline_fair,
                    "baseline_fair": baseline_fair,
                }
            )

    rel_df = pd.DataFrame(rows)
    if rel_df.empty:
        return pd.DataFrame()

    agg = (
        rel_df.groupby(["variant", "method", "alpha", "fairness_type"])
        .agg(
            rel_fair_mean=("rel_fair_vs_baseline", "mean"),
            rel_fair_std=("rel_fair_vs_baseline", "std"),
            n_seeds=("split_seed", "nunique"),
        )
        .round(4)
    )
    return agg.reset_index()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def health_report_v2(df: pd.DataFrame) -> dict:
    out = {
        "n_rows_total": int(len(df)),
        "n_variants": int(df["variant"].nunique()),
        "n_cells": int(df.groupby(["variant", "fairness_type", "alpha_fair"]).ngroups),
        "n_methods": int(df["method"].nunique()),
        "n_split_seeds": int(df["split_seed"].nunique()),
        "nan_or_inf_steps_total": int(df["nan_or_inf_steps"].sum()),
        "exploding_steps_total": int(df["exploding_steps"].sum()),
        "max_train_test_gap": float(_train_test_gap(df).max()),
        "median_train_test_gap": float(_train_test_gap(df).median()),
    }
    if "early_stop_enabled" in df.columns:
        out["rows_with_early_stop_applied"] = int(
            df[df["early_stop_enabled"] == 1]["early_stop_applied"].sum()
        )
        out["rows_with_early_stop_enabled"] = int((df["early_stop_enabled"] == 1).sum())
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("Healthcare v2 grid analysis")
    print("=" * 40)
    df = load_both_variants()
    if df.empty:
        raise RuntimeError("No v2 data found. Run variants a and b first.")
    print(f"loaded: {len(df)} rows, {df['variant'].nunique()} variants")

    health = health_report_v2(df)
    print("\n=== Health ===")
    for k, v in health.items():
        print(f"  {k}: {v}")

    HC_V2_DIR.mkdir(parents=True, exist_ok=True)
    with open(HC_V2_DIR / "health.json", "w") as f:
        json.dump(health, f, indent=2)

    # Grand summary
    gs = grand_summary_v2(df)
    gs.to_csv(HC_V2_DIR / "grand_summary.csv", index=False)
    print(f"\ngrand_summary: {len(gs)} rows -> grand_summary.csv")

    # Cross-fairness rank (on fairness-constrained Pareto point)
    rank_agg = cross_fairness_rank(gs, metric="test_fair_at_fair_best")
    rank_agg.to_csv(HC_V2_DIR / "cross_fairness_rank.csv", index=False)
    print(f"\n=== Cross-fairness rank (fairness-constrained Pareto point) ===")
    print(rank_agg.to_string(index=False))

    # Cross-fairness ratio vs FPTO lam=0
    ratio_fpto = cross_fairness_ratio(df, baseline_method="fpto", baseline_lam=0.0)
    ratio_fpto.to_csv(HC_V2_DIR / "cross_fairness_ratio_fpto.csv", index=False)
    print(f"\n=== Cross-fairness ratio vs FPTO lam=0 ===")
    if not ratio_fpto.empty:
        pivot = ratio_fpto.pivot_table(
            index=["variant", "method"],
            columns=["fairness_type", "alpha"],
            values="rel_fair_mean",
        ).round(3)
        print(pivot.to_string())

    # Cross-fairness ratio vs SAA lam=0
    ratio_saa = cross_fairness_ratio(df, baseline_method="saa", baseline_lam=0.0)
    ratio_saa.to_csv(HC_V2_DIR / "cross_fairness_ratio_saa.csv", index=False)

    # Per-cell summaries (train + test)
    for (variant, ft, a), cell in df.groupby(["variant", "fairness_type", "alpha_fair"]):
        agg = aggregate_cell_v2(cell)
        out_dir = variant_dir(variant) / ft / f"alpha_{a}"
        out_dir.mkdir(parents=True, exist_ok=True)
        agg.to_csv(out_dir / "summary_by_method_lambda.csv")
        best = best_pareto_per_method_v2(cell)
        best.to_csv(out_dir / "best_pareto_per_method.csv", index=False)

    print(f"\n[outputs] {HC_V2_DIR}")


if __name__ == "__main__":
    main()
