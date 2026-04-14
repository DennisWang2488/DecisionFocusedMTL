"""Analysis utilities for the healthcare follow-up grid.

After ``run_healthcare_grid`` finishes, run this module to compute
publication-style summary tables (one per cell + a grand-summary across cells)
and write them to ``results/advisor_review/healthcare_followup/``.

Reads ``stage_results.csv`` from each cell, aggregates across seeds, and
extracts each method's Pareto-best operating point (the lambda achieving the
lowest mean ``test_regret_normalized``).

Usage:
    python -m experiments.advisor_review.analyze_healthcare_grid
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.advisor_review.healthcare_followup import (  # noqa: E402
    HC_ALPHAS,
    HC_DIR,
    HC_FAIRNESS_TYPES,
    HC_METHODS,
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def cell_path(fairness_type: str, alpha: float) -> Path:
    return HC_DIR / fairness_type / f"alpha_{alpha}" / "stage_results.csv"


def load_cell(fairness_type: str, alpha: float) -> pd.DataFrame | None:
    path = cell_path(fairness_type, alpha)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["fairness_type"] = fairness_type
    df["alpha_fair"] = float(alpha)
    return df


def load_grid(
    fairness_types: Iterable[str] = HC_FAIRNESS_TYPES,
    alphas: Iterable[float] = HC_ALPHAS,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ft in fairness_types:
        for a in alphas:
            df = load_cell(ft, a)
            if df is not None:
                frames.append(df)
    if not frames:
        raise RuntimeError("No cell results found.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

def aggregate_cell(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std across seeds, per (method, lambda)."""
    return (
        df.groupby(["method", "lambda"], dropna=False)
        .agg(
            train_reg_n_mean=("train_regret_normalized", "mean"),
            train_reg_n_std=("train_regret_normalized", "std"),
            test_reg_n_mean=("test_regret_normalized", "mean"),
            test_reg_n_std=("test_regret_normalized", "std"),
            train_fair_mean=("train_fairness", "mean"),
            train_fair_std=("train_fairness", "std"),
            test_fair_mean=("test_fairness", "mean"),
            test_fair_std=("test_fairness", "std"),
            train_mse_mean=("train_pred_mse", "mean"),
            test_mse_mean=("test_pred_mse", "mean"),
            n_seeds=("seed", "count"),
            nan_steps=("nan_or_inf_steps", "sum"),
            explode_steps=("exploding_steps", "sum"),
        )
        .round(5)
    )


def best_pareto_per_method(df: pd.DataFrame) -> pd.DataFrame:
    """For each method, pick the lambda with the lowest mean test_regret_normalized."""
    rows = []
    for method in df["method"].unique():
        sub = (
            df[df["method"] == method]
            .groupby("lambda")
            .agg(
                test_reg_n=("test_regret_normalized", "mean"),
                test_reg_n_std=("test_regret_normalized", "std"),
                test_fair=("test_fairness", "mean"),
                test_fair_std=("test_fairness", "std"),
                train_reg_n=("train_regret_normalized", "mean"),
                train_fair=("train_fairness", "mean"),
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
            }
        )
    return pd.DataFrame(rows).round(5).sort_values("test_reg_n")


def grand_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Best Pareto point per method per (fairness_type, alpha) cell."""
    rows = []
    for (ft, a), cell in df.groupby(["fairness_type", "alpha_fair"]):
        cell_summary = best_pareto_per_method(cell)
        cell_summary.insert(0, "fairness_type", ft)
        cell_summary.insert(1, "alpha", a)
        rows.append(cell_summary)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------------

def health_report(df: pd.DataFrame) -> dict:
    out = {
        "n_rows_total": int(len(df)),
        "n_cells": int(df.groupby(["fairness_type", "alpha_fair"]).ngroups),
        "n_methods": int(df["method"].nunique()),
        "n_seeds": int(df["seed"].nunique()),
        "n_lambdas_used": int(df["lambda"].nunique()),
        "nan_or_inf_steps_total": int(df["nan_or_inf_steps"].sum()),
        "exploding_steps_total": int(df["exploding_steps"].sum()),
        "max_train_test_gap": float(
            (df["test_regret_normalized"] - df["train_regret_normalized"]).max()
        ),
        "median_train_test_gap": float(
            (df["test_regret_normalized"] - df["train_regret_normalized"]).median()
        ),
    }
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Healthcare grid analysis")
    print("========================")
    df = load_grid()
    print(f"loaded: {len(df)} rows, "
          f"{df.groupby(['fairness_type','alpha_fair']).ngroups} cells")
    health = health_report(df)
    print("\n=== Health ===")
    for k, v in health.items():
        print(f"  {k}: {v}")

    out_dir = HC_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-cell aggregates
    cell_summaries = {}
    for (ft, a), cell in df.groupby(["fairness_type", "alpha_fair"]):
        agg = aggregate_cell(cell)
        sub_dir = out_dir / ft / f"alpha_{a}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        agg.to_csv(sub_dir / "summary_by_method_lambda.csv")
        best = best_pareto_per_method(cell)
        best.to_csv(sub_dir / "best_pareto_per_method.csv", index=False)
        cell_summaries[(ft, a)] = {
            "agg": agg,
            "best": best,
        }

    # Grand summary across cells
    gs = grand_summary(df)
    gs.to_csv(out_dir / "grand_summary.csv", index=False)
    print("\n=== Grand summary (best Pareto point per method per cell) ===")
    print(gs.to_string(index=False))

    # Health JSON
    with open(out_dir / "health.json", "w") as f:
        json.dump(health, f, indent=2)

    print(f"\n[outputs] {out_dir}/grand_summary.csv, health.json, per-cell summary CSVs")


if __name__ == "__main__":
    main()
