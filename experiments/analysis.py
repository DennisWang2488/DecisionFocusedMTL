"""Result loading, summary tables, and data exploration."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .configs import ALPHA_VALUES as _DEFAULT_ALPHAS


# ---------------------------------------------------------------------------
# Load results from CSV
# ---------------------------------------------------------------------------
def load_results(results_dir: str = "results") -> tuple[pd.DataFrame, pd.DataFrame]:
    stage_path = os.path.join(results_dir, "stage_results_full.csv")
    iter_path = os.path.join(results_dir, "iter_logs_full.csv")

    assert os.path.exists(stage_path), f"No stage CSV at {stage_path}"
    stage_df = pd.read_csv(stage_path)
    iter_df = pd.read_csv(iter_path) if os.path.exists(iter_path) else pd.DataFrame()

    print(f"Stage results: {len(stage_df)} rows")
    print(f"Iteration logs: {len(iter_df)} rows")
    print(f"Methods present: {sorted(stage_df['config_name'].unique().tolist())}")
    print(f"Alphas present:  {sorted(stage_df['alpha_fair'].unique().tolist())}")
    return stage_df, iter_df


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def summary_table(stage_df: pd.DataFrame,
                  alpha_values: list[float] | None = None) -> pd.DataFrame | None:
    alpha_values = alpha_values or _DEFAULT_ALPHAS
    if len(stage_df) == 0 or "test_regret" not in stage_df.columns:
        print("No results available for summary table.")
        return None

    has_norm = "test_regret_normalized" in stage_df.columns

    agg_dict = {
        "regret_mean": ("test_regret", "mean"),
        "regret_std": ("test_regret", "std"),
    }
    if has_norm:
        agg_dict["norm_regret_mean"] = ("test_regret_normalized", "mean")
        agg_dict["norm_regret_std"] = ("test_regret_normalized", "std")
    agg_dict.update({
        "fairness_mean": ("test_fairness", "mean"),
        "fairness_std": ("test_fairness", "std"),
        "pred_mse_mean": ("test_pred_mse", "mean"),
        "pred_mse_std": ("test_pred_mse", "std"),
        "n_runs": ("test_regret", "count"),
    })

    summary = (
        stage_df
        .groupby(["config_name", "alpha_fair"])
        .agg(**agg_dict)
        .reset_index()
        .sort_values(["alpha_fair", "regret_mean"])
    )

    for alpha in alpha_values:
        print(f"\n{'=' * 100}")
        print(f"Results for alpha_fair = {alpha}")
        print(f"{'=' * 100}")
        sub = summary[summary["alpha_fair"] == alpha].copy()

        display_cols = ["config_name", "regret_mean", "regret_std"]
        col_names = ["Method", "Regret (mean)", "Regret (std)"]
        if has_norm:
            display_cols += ["norm_regret_mean", "norm_regret_std"]
            col_names += ["NormRegret (mean)", "NormRegret (std)"]
        display_cols += ["fairness_mean", "fairness_std",
                         "pred_mse_mean", "pred_mse_std", "n_runs"]
        col_names += ["Fairness (mean)", "Fairness (std)",
                      "Pred MSE (mean)", "Pred MSE (std)", "N"]

        sub_display = sub[display_cols].copy()
        sub_display.columns = col_names
        print(sub_display.to_string(index=False, float_format="%.6f"))

    return summary


# ---------------------------------------------------------------------------
# Select best lambda per (method, alpha) using Pareto-proxy score
# ---------------------------------------------------------------------------
def select_best_lambda(stage_df: pd.DataFrame,
                       alpha_values: list[float] | None = None) -> pd.DataFrame:
    alpha_values = alpha_values or _DEFAULT_ALPHAS
    has_norm = "test_regret_normalized" in stage_df.columns

    best_rows = []
    for alpha in alpha_values:
        sub = stage_df[stage_df["alpha_fair"] == alpha]
        for method_name in sub["config_name"].unique():
            msub = sub[sub["config_name"] == method_name]
            agg_cols = {
                "regret": ("test_regret", "mean"),
                "fairness": ("test_fairness", "mean"),
                "pred_mse": ("test_pred_mse", "mean"),
            }
            if has_norm:
                agg_cols["norm_regret"] = ("test_regret_normalized", "mean")
            avg = msub.groupby("lambda").agg(**agg_cols).reset_index()
            if len(avg) > 0:
                avg["combined"] = (avg["regret"] / max(avg["regret"].max(), 1e-12) +
                                   avg["fairness"] / max(avg["fairness"].max(), 1e-12))
                best = avg.loc[avg["combined"].idxmin()]
                row = {"alpha": alpha, "method": method_name,
                       "best_lambda": best["lambda"],
                       "regret": best["regret"], "fairness": best["fairness"],
                       "pred_mse": best["pred_mse"]}
                if has_norm:
                    row["norm_regret"] = best["norm_regret"]
                best_rows.append(row)

    return pd.DataFrame(best_rows)


# ---------------------------------------------------------------------------
# Data exploration
# ---------------------------------------------------------------------------
def explore_data(data_csv: str):
    assert os.path.exists(data_csv), f"Data file not found at {data_csv}"
    df = pd.read_csv(data_csv)

    print(f"Dataset shape: {df.shape}")
    print(f"Number of patients: {len(df):,}")
    print(f"Number of features: {df.shape[1]}")
    print(f"\nColumn types:\n{df.dtypes.value_counts()}")

    # Race distribution
    print("\n" + "=" * 50)
    print("Race Distribution")
    print("=" * 50)
    race_counts = df["race"].value_counts().sort_index()
    for race_val, count in race_counts.items():
        label = "White" if race_val == 0 else "Black"
        print(f"  Race={race_val} ({label}): n={count:,} ({100 * count / len(df):.1f}%)")

    # Benefit distribution by race
    print("\n" + "=" * 50)
    print("Benefit Distribution by Race")
    print("=" * 50)
    for race_val in sorted(df["race"].unique()):
        label = "White" if race_val == 0 else "Black"
        subset = df[df["race"] == race_val]
        b = subset["benefit"]
        print(f"  Race={race_val} ({label}): mean={b.mean():.4f}, "
              f"std={b.std():.4f}, min={b.min():.4f}, max={b.max():.4f}")

    # Cost distribution by race
    if "cost_t" in df.columns:
        print("\n" + "=" * 50)
        print("Cost Distribution by Race")
        print("=" * 50)
        for race_val in sorted(df["race"].unique()):
            label = "White" if race_val == 0 else "Black"
            subset = df[df["race"] == race_val]
            c = subset["cost_t"]
            print(f"  Race={race_val} ({label}): mean={c.mean():.2f}, "
                  f"std={c.std():.2f}, min={c.min():.2f}, max={c.max():.2f}")

    return df
