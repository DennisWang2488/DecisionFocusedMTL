"""Plot functions for experiment results — each function produces one figure."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .configs import ALPHA_VALUES as _DEFAULT_ALPHAS, COLOR_MAP, MARKER_MAP


def _get_style(method_name: str):
    color = COLOR_MAP.get(method_name, "#333333")
    marker = MARKER_MAP.get(method_name, "o")
    return color, marker


# ---------------------------------------------------------------------------
# Pareto: Regret vs Fairness
# ---------------------------------------------------------------------------
def plot_pareto_regret_vs_fairness(stage_df: pd.DataFrame,
                                   alpha_values: list[float] | None = None,
                                   results_dir: str = "results"):
    alpha_values = alpha_values or _DEFAULT_ALPHAS
    if len(stage_df) == 0 or "test_regret" not in stage_df.columns:
        print("No data for pareto_regret_vs_fairness.")
        return

    fig, axes = plt.subplots(1, len(alpha_values), figsize=(8 * len(alpha_values), 6))
    if len(alpha_values) == 1:
        axes = [axes]

    for ax_idx, alpha in enumerate(alpha_values):
        ax = axes[ax_idx]
        sub = stage_df[stage_df["alpha_fair"] == alpha]

        for method_name in sub["config_name"].unique():
            msub = sub[sub["config_name"] == method_name]
            color, marker = _get_style(method_name)

            ax.scatter(msub["test_regret"], msub["test_fairness"],
                       c=color, marker=marker, s=50, alpha=0.6,
                       label=method_name, edgecolors="white", linewidths=0.5)
            ax.scatter(msub["test_regret"].mean(), msub["test_fairness"].mean(),
                       c=color, marker=marker, s=150, alpha=1.0,
                       edgecolors="black", linewidths=1.5)

        ax.set_xlabel("Test Regret (lower is better)", fontsize=12)
        ax.set_ylabel("Test Fairness Loss / MAD (lower is better)", fontsize=12)
        ax.set_title(f"Pareto Frontier: alpha = {alpha}", fontsize=14)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "pareto_regret_vs_fairness.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved pareto_regret_vs_fairness.png")


# ---------------------------------------------------------------------------
# Pareto: Normalized Regret vs Fairness
# ---------------------------------------------------------------------------
def plot_pareto_normalized(stage_df: pd.DataFrame,
                           alpha_values: list[float] | None = None,
                           results_dir: str = "results"):
    alpha_values = alpha_values or _DEFAULT_ALPHAS
    if len(stage_df) == 0 or "test_regret_normalized" not in stage_df.columns:
        print("Normalized regret column not available; skipping.")
        return

    fig, axes = plt.subplots(1, len(alpha_values), figsize=(8 * len(alpha_values), 6))
    if len(alpha_values) == 1:
        axes = [axes]

    for ax_idx, alpha in enumerate(alpha_values):
        ax = axes[ax_idx]
        sub = stage_df[stage_df["alpha_fair"] == alpha]

        for method_name in sub["config_name"].unique():
            msub = sub[sub["config_name"] == method_name]
            color, marker = _get_style(method_name)

            ax.scatter(msub["test_regret_normalized"], msub["test_fairness"],
                       c=color, marker=marker, s=50, alpha=0.6,
                       label=method_name, edgecolors="white", linewidths=0.5)
            ax.scatter(msub["test_regret_normalized"].mean(), msub["test_fairness"].mean(),
                       c=color, marker=marker, s=150, alpha=1.0,
                       edgecolors="black", linewidths=1.5)

        ax.set_xlabel("Normalized Test Regret (lower is better)", fontsize=12)
        ax.set_ylabel("Test Fairness Loss / MAD (lower is better)", fontsize=12)
        ax.set_title(f"Pareto Frontier (Normalized Regret): alpha = {alpha}", fontsize=14)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "pareto_norm_regret_vs_fairness.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved pareto_norm_regret_vs_fairness.png")


# ---------------------------------------------------------------------------
# Pareto: Regret vs Prediction MSE
# ---------------------------------------------------------------------------
def plot_pareto_regret_vs_mse(stage_df: pd.DataFrame,
                               alpha_values: list[float] | None = None,
                               results_dir: str = "results"):
    alpha_values = alpha_values or _DEFAULT_ALPHAS
    if len(stage_df) == 0 or "test_regret" not in stage_df.columns:
        print("No data for pareto_regret_vs_mse.")
        return

    fig, axes = plt.subplots(1, len(alpha_values), figsize=(8 * len(alpha_values), 6))
    if len(alpha_values) == 1:
        axes = [axes]

    for ax_idx, alpha in enumerate(alpha_values):
        ax = axes[ax_idx]
        sub = stage_df[stage_df["alpha_fair"] == alpha]

        for method_name in sub["config_name"].unique():
            msub = sub[sub["config_name"] == method_name]
            color, marker = _get_style(method_name)

            ax.scatter(msub["test_regret"], msub["test_pred_mse"],
                       c=color, marker=marker, s=50, alpha=0.6,
                       label=method_name, edgecolors="white", linewidths=0.5)
            ax.scatter(msub["test_regret"].mean(), msub["test_pred_mse"].mean(),
                       c=color, marker=marker, s=150, alpha=1.0,
                       edgecolors="black", linewidths=1.5)

        ax.set_xlabel("Test Regret (lower is better)", fontsize=12)
        ax.set_ylabel("Test Prediction MSE (lower is better)", fontsize=12)
        ax.set_title(f"Regret vs Prediction MSE: alpha = {alpha}", fontsize=14)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "pareto_regret_vs_mse.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved pareto_regret_vs_mse.png")


# ---------------------------------------------------------------------------
# 3D Pareto: Regret vs Fairness vs MSE
# ---------------------------------------------------------------------------
def plot_pareto_3d(stage_df: pd.DataFrame,
                   alpha_values: list[float] | None = None,
                   results_dir: str = "results"):
    alpha_values = alpha_values or _DEFAULT_ALPHAS
    if len(stage_df) == 0 or "test_regret" not in stage_df.columns:
        print("No data for pareto_3d.")
        return

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8 * len(alpha_values), 7))

    for ax_idx, alpha in enumerate(alpha_values):
        ax = fig.add_subplot(1, len(alpha_values), ax_idx + 1, projection="3d")
        sub = stage_df[stage_df["alpha_fair"] == alpha]

        for method_name in sub["config_name"].unique():
            msub = sub[sub["config_name"] == method_name]
            color, marker = _get_style(method_name)

            ax.scatter(msub["test_regret"], msub["test_fairness"], msub["test_pred_mse"],
                       c=color, marker=marker, s=40, alpha=0.7, label=method_name)

        ax.set_xlabel("Regret", fontsize=9)
        ax.set_ylabel("Fairness (MAD)", fontsize=9)
        ax.set_zlabel("Pred MSE", fontsize=9)
        ax.set_title(f"3D Pareto: alpha = {alpha}", fontsize=12)
        ax.legend(fontsize=6, loc="upper left", ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "pareto_3d.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved pareto_3d.png")


# ---------------------------------------------------------------------------
# Lambda sweep
# ---------------------------------------------------------------------------
def plot_lambda_sweep(stage_df: pd.DataFrame,
                      alpha_values: list[float] | None = None,
                      results_dir: str = "results"):
    alpha_values = alpha_values or _DEFAULT_ALPHAS
    if len(stage_df) == 0 or "lambda" not in stage_df.columns:
        print("No lambda column; skipping lambda sweep plot.")
        return

    has_norm = "test_regret_normalized" in stage_df.columns
    metrics = [("test_regret", "Regret"), ("test_fairness", "Fairness (MAD)"),
               ("test_pred_mse", "Prediction MSE")]
    if has_norm:
        metrics.insert(1, ("test_regret_normalized", "Normalized Regret"))

    n_cols = len(metrics)
    fig, axes = plt.subplots(len(alpha_values), n_cols, figsize=(6 * n_cols, 5 * len(alpha_values)))
    if len(alpha_values) == 1:
        axes = axes.reshape(1, -1)

    for row_idx, alpha in enumerate(alpha_values):
        sub = stage_df[stage_df["alpha_fair"] == alpha]

        for col_idx, (metric, metric_label) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            grouped = (sub.groupby(["config_name", "lambda"])[metric]
                       .agg(["mean", "std"]).reset_index())

            for method_name in grouped["config_name"].unique():
                msub = grouped[grouped["config_name"] == method_name]
                color, _ = _get_style(method_name)
                lam = msub["lambda"].values
                mu = msub["mean"].values
                sigma = msub["std"].values
                ax.plot(lam, mu, "-o",
                        color=color, label=method_name, markersize=4, linewidth=1.5)
                ax.fill_between(lam, mu - sigma, mu + sigma, color=color, alpha=0.15)

            ax.set_xlabel("Lambda (fairness weight)", fontsize=10)
            ax.set_ylabel(metric_label, fontsize=10)
            ax.set_title(f"{metric_label} vs Lambda (alpha={alpha})", fontsize=11)
            ax.legend(fontsize=6, ncol=3)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "lambda_sweep.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved lambda_sweep.png")


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def plot_training_curves(iter_df: pd.DataFrame,
                         alpha_values: list[float] | None = None,
                         results_dir: str = "results"):
    alpha_values = alpha_values or _DEFAULT_ALPHAS
    if len(iter_df) == 0 or "loss_dec" not in iter_df.columns:
        print("No iteration data for training curves.")
        return

    plot_alpha = 2.0 if 2.0 in iter_df["alpha_fair"].unique() else iter_df["alpha_fair"].iloc[0]
    sub_iter = iter_df[iter_df["alpha_fair"] == plot_alpha]

    if "lambda" in sub_iter.columns:
        available_lambdas = sorted(sub_iter["lambda"].unique())
        plot_lambda = 0.2 if 0.2 in available_lambdas else available_lambdas[-1]
        sub_iter = sub_iter[sub_iter["lambda"] == plot_lambda]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    loss_cols = [("loss_dec", "Decision Regret"), ("loss_pred", "Prediction Loss"),
                 ("loss_fair", "Fairness Loss")]

    for ax_idx, (col, label) in enumerate(loss_cols):
        ax = axes[ax_idx]
        if col not in sub_iter.columns:
            continue

        for method_name in sub_iter["config_name"].unique():
            msub = sub_iter[sub_iter["config_name"] == method_name]
            grouped = msub.groupby("iter")[col].agg(["mean", "std"])
            color, _ = _get_style(method_name)
            iters = grouped.index.values
            mu = grouped["mean"].values
            sigma = grouped["std"].values

            ax.plot(iters, mu, color=color, label=method_name, linewidth=1.2)
            ax.fill_between(iters, np.clip(mu - sigma, 0, None), mu + sigma,
                            color=color, alpha=0.15)

        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"{label} (alpha={plot_alpha}, lambda={plot_lambda})", fontsize=11)
        ax.legend(fontsize=6, ncol=3)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved training_curves.png")


# ---------------------------------------------------------------------------
# Gradient conflict analysis
# ---------------------------------------------------------------------------
def plot_gradient_conflict(iter_df: pd.DataFrame,
                           alpha_values: list[float] | None = None,
                           results_dir: str = "results"):
    alpha_values = alpha_values or _DEFAULT_ALPHAS
    if len(iter_df) == 0 or "cos_dec_pred" not in iter_df.columns:
        print("No gradient conflict data.")
        return

    plot_alpha = 2.0 if 2.0 in iter_df["alpha_fair"].unique() else iter_df["alpha_fair"].iloc[0]
    sub_iter = iter_df[iter_df["alpha_fair"] == plot_alpha]

    if "lambda" in sub_iter.columns:
        available_lambdas = sorted(sub_iter["lambda"].unique())
        plot_lambda = 0.2 if 0.2 in available_lambdas else available_lambdas[-1]
        sub_iter = sub_iter[sub_iter["lambda"] == plot_lambda]

    cos_pairs = [
        ("cos_dec_pred", "cos(g_dec, g_pred)"),
        ("cos_dec_fair", "cos(g_dec, g_fair)"),
        ("cos_pred_fair", "cos(g_pred, g_fair)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax_idx, (col, label) in enumerate(cos_pairs):
        ax = axes[ax_idx]
        if col not in sub_iter.columns:
            continue

        for method_name in sub_iter["config_name"].unique():
            msub = sub_iter[sub_iter["config_name"] == method_name]
            grouped = msub.groupby("iter")[col].mean()
            color, _ = _get_style(method_name)
            ax.plot(grouped.index.values, grouped.values, color=color,
                    label=method_name, linewidth=1.2)

        ax.axhline(y=0, color="black", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"{label} (alpha={plot_alpha})", fontsize=11)
        ax.set_ylim(-1.1, 1.1)
        ax.legend(fontsize=6, ncol=3)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "gradient_conflict.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved gradient_conflict.png")


# ---------------------------------------------------------------------------
# Per-alpha comparison (bar chart of best-lambda results)
# ---------------------------------------------------------------------------
def plot_per_alpha_comparison(stage_df: pd.DataFrame,
                              alpha_values: list[float] | None = None,
                              results_dir: str = "results"):
    alpha_values = alpha_values or _DEFAULT_ALPHAS
    if len(stage_df) == 0 or "test_regret" not in stage_df.columns:
        print("No data for per-alpha comparison.")
        return

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

    best_df = pd.DataFrame(best_rows)
    if len(best_df) == 0:
        print("No best-lambda rows to plot.")
        return

    bar_metrics = [("regret", "Best Regret")]
    if has_norm:
        bar_metrics.append(("norm_regret", "Best Normalized Regret"))
    bar_metrics += [("fairness", "Best Fairness (MAD)"), ("pred_mse", "Best Pred MSE")]

    n_bar_plots = len(bar_metrics)
    fig, axes = plt.subplots(1, n_bar_plots, figsize=(6 * n_bar_plots, 6))

    methods_list = sorted(best_df["method"].unique())
    x = np.arange(len(methods_list))
    width = 0.35

    for ax_idx, (metric, label) in enumerate(bar_metrics):
        ax = axes[ax_idx]
        for i, alpha in enumerate(alpha_values):
            sub = best_df[best_df["alpha"] == alpha]
            vals = []
            for m in methods_list:
                row = sub[sub["method"] == m]
                vals.append(row[metric].values[0] if len(row) > 0 else 0)
            offset = (i - 0.5) * width
            ax.bar(x + offset, vals, width, label=f"alpha={alpha}", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(methods_list, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "per_alpha_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved per_alpha_comparison.png")


# ---------------------------------------------------------------------------
# Convenience: generate all plots
# ---------------------------------------------------------------------------
def plot_all(stage_df: pd.DataFrame, iter_df: pd.DataFrame,
             alpha_values: list[float] | None = None,
             results_dir: str = "results"):
    os.makedirs(results_dir, exist_ok=True)
    plot_pareto_regret_vs_fairness(stage_df, alpha_values, results_dir)
    plot_pareto_normalized(stage_df, alpha_values, results_dir)
    plot_pareto_regret_vs_mse(stage_df, alpha_values, results_dir)
    plot_pareto_3d(stage_df, alpha_values, results_dir)
    plot_lambda_sweep(stage_df, alpha_values, results_dir)
    plot_training_curves(iter_df, alpha_values, results_dir)
    plot_gradient_conflict(iter_df, alpha_values, results_dir)
    plot_per_alpha_comparison(stage_df, alpha_values, results_dir)
