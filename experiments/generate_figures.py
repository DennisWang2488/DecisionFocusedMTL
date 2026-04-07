#!/usr/bin/env python3
"""Generate publication-quality figures from final experiment results.

Produces:
  Figure 1: Pareto front (healthcare) — 2D projections of 3-objective tradeoffs
  Figure 2: Cosine similarity heatmap (knapsack) — gradient conflict analysis
  Figure 3: Method ranking summary — compact visual across conditions

Usage:
  python experiments/generate_figures.py
  python experiments/generate_figures.py --healthcare-dir results/final/healthcare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# ======================================================================
# Publication-quality matplotlib settings
# ======================================================================

def _setup_matplotlib():
    """Configure matplotlib for INFORMS journal publication."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "text.usetex": False,  # Set True if LaTeX is available
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })
    return plt


# Method visual properties — keyed by method_label
METHOD_COLORS = {
    "FPTO": "#1f77b4", "SAA": "#e6550d", "WDRO": "#756bb1",
    "FDFL-Scal": "#ff7f0e",
    "FDFL-PCGrad": "#2ca02c", "FDFL-MGDA": "#d62728", "FDFL-CAGrad": "#9467bd",
}

METHOD_MARKERS = {
    "FPTO": "o", "SAA": "D", "WDRO": "p", "FDFL-Scal": "^",
    "FDFL-PCGrad": "*", "FDFL-MGDA": "h", "FDFL-CAGrad": "d",
}

# Categorization for legend grouping
METHOD_CATEGORY = {
    "FPTO": "Two-stage", "SAA": "Data-driven", "WDRO": "Data-driven",
    "FDFL-Scal": "FDFL-Scal",
    "FDFL-PCGrad": "FDFL-MOO", "FDFL-MGDA": "FDFL-MOO", "FDFL-CAGrad": "FDFL-MOO",
}


def _load_results(results_dir: str) -> pd.DataFrame:
    p = Path(results_dir)
    csvs = sorted(p.rglob("stage_results.csv"))
    if not csvs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)


def _load_iter_logs(results_dir: str) -> pd.DataFrame:
    p = Path(results_dir)
    agg = p / "iter_logs_all.csv"
    if agg.exists():
        return pd.read_csv(agg)
    csvs = sorted(p.rglob("iter_logs.csv"))
    if not csvs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)


# ======================================================================
# Figure 1: Pareto Front (Healthcare)
# ======================================================================

def generate_pareto_figure(df: pd.DataFrame, output_dir: Path):
    """2D Pareto front projections for healthcare experiment."""
    plt = _setup_matplotlib()

    if df.empty:
        print("  Skipping Pareto figure (no data)")
        return

    # Ensure lambda column
    if "lambda" not in df.columns:
        df["lambda"] = 0.0
    df["lam"] = df["lambda"].fillna(0.0)

    alphas = sorted(df["alpha_fair"].unique())
    projections = [
        ("test_regret_normalized", "test_fairness",
         "Norm. Decision Regret", "Pred. Fairness Violation"),
        ("test_regret_normalized", "test_pred_mse",
         "Norm. Decision Regret", "Prediction MSE"),
    ]

    fig, axes = plt.subplots(len(alphas), len(projections),
                             figsize=(3.5 * len(projections), 3.0 * len(alphas)),
                             squeeze=False)

    for row, alpha in enumerate(alphas):
        sub = df[df["alpha_fair"] == alpha]
        # Each point is a (method_label, lambda) pair, averaged across seeds
        method_means = sub.groupby(["method_label", "lam"]).agg(
            test_regret_normalized=("test_regret_normalized", "mean"),
            test_fairness=("test_fairness", "mean"),
            test_pred_mse=("test_pred_mse", "mean"),
        ).reset_index()

        for col, (x_metric, y_metric, x_label, y_label) in enumerate(projections):
            ax = axes[row, col]

            plotted_labels = set()
            for _, mrow in method_means.iterrows():
                method = mrow["method_label"]
                lam = mrow["lam"]
                x_val = mrow[x_metric]
                y_val = mrow[y_metric]

                if np.isnan(x_val) or np.isnan(y_val):
                    continue

                color = METHOD_COLORS.get(method, "#333333")
                marker = METHOD_MARKERS.get(method, "o")
                # Label includes lambda for methods with lambda sweep
                if lam > 0 and method in ("FPTO", "FDFL-Scal"):
                    display = f"{method} (lam={lam})"
                elif lam == 0 and method == "FPTO":
                    display = "PTO"
                elif lam == 0 and method == "FDFL-Scal":
                    display = "DFL"
                else:
                    display = method

                show_label = display if display not in plotted_labels else None
                plotted_labels.add(display)

                ax.scatter(x_val, y_val, c=color, marker=marker,
                           s=60, edgecolors="black", linewidths=0.5,
                           label=show_label, zorder=5)

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f"$\\alpha = {alpha}$")

            if col == len(projections) - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left",
                          frameon=True, framealpha=0.9, fontsize=6)

    fig.suptitle("Healthcare: Pareto Front Projections", fontsize=11, y=1.02)
    plt.tight_layout()

    out_path = output_dir / "fig1_pareto_healthcare.pdf"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  Figure 1 saved to {out_path}")


# ======================================================================
# Figure 2: Cosine Similarity Heatmap (Knapsack)
# ======================================================================

def generate_cosine_figure(df: pd.DataFrame, iter_df: pd.DataFrame, output_dir: Path):
    """Gradient cosine similarity figure for knapsack experiment."""
    plt = _setup_matplotlib()

    # Use stage-level cosine similarities (averaged over training)
    if df.empty:
        print("  Skipping cosine figure (no data)")
        return

    cos_cols = ["avg_cos_dec_pred", "avg_cos_dec_fair", "avg_cos_pred_fair"]
    available_cos = [c for c in cos_cols if c in df.columns]

    if not available_cos:
        print("  Skipping cosine figure (no cosine columns)")
        return

    cos_display = {
        "avg_cos_dec_pred": r"$\cos(\nabla_{dec}, \nabla_{pred})$",
        "avg_cos_dec_fair": r"$\cos(\nabla_{dec}, \nabla_{fair})$",
        "avg_cos_pred_fair": r"$\cos(\nabla_{pred}, \nabla_{fair})$",
    }

    alphas = sorted(df["alpha_fair"].unique())
    uf_levels = ["mild", "medium", "high"]
    if "unfairness_level" in df.columns:
        uf_levels = [u for u in uf_levels if u in df["unfairness_level"].unique()]
    else:
        print("  Skipping cosine figure (no unfairness_level column)")
        return

    # Focus on MOO methods that actually compute gradient conflicts
    moo_methods = ["FDFL-PCGrad", "FDFL-MGDA", "FDFL-CAGrad"]
    moo_methods = [m for m in moo_methods if m in df["method_label"].unique()]

    if not moo_methods:
        print("  Skipping cosine figure (no MOO methods in data)")
        return

    n_cos = len(available_cos)
    fig, axes = plt.subplots(len(alphas), n_cos,
                             figsize=(3.5 * n_cos, 3.0 * len(alphas)),
                             squeeze=False)

    for row, alpha in enumerate(alphas):
        for col, cos_col in enumerate(available_cos):
            ax = axes[row, col]

            # Build heatmap: rows = methods, cols = unfairness levels
            data_matrix = []
            method_labels = []
            for method in moo_methods:
                row_vals = []
                for uf in uf_levels:
                    sub = df[
                        (df["alpha_fair"] == alpha)
                        & (df["unfairness_level"] == uf)
                        & (df["method_label"] == method)
                    ]
                    val = sub[cos_col].mean() if not sub.empty and cos_col in sub.columns else np.nan
                    row_vals.append(val)
                data_matrix.append(row_vals)
                method_labels.append(METHOD_DISPLAY.get(method, method))

            matrix = np.array(data_matrix)
            im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

            # Annotate cells
            for i in range(len(method_labels)):
                for j in range(len(uf_levels)):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        color = "white" if abs(val) > 0.5 else "black"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=7, color=color)

            ax.set_xticks(range(len(uf_levels)))
            ax.set_xticklabels([u.capitalize() for u in uf_levels])
            ax.set_yticks(range(len(method_labels)))
            ax.set_yticklabels(method_labels)
            ax.set_title(f"{cos_display.get(cos_col, cos_col)}\n$\\alpha={alpha}$",
                         fontsize=8)

    # Colorbar
    fig.colorbar(im, ax=axes, shrink=0.6, label="Cosine similarity")
    fig.suptitle("Knapsack: Gradient Conflict Analysis", fontsize=11, y=1.02)
    plt.tight_layout()

    out_path = output_dir / "fig2_cosine_knapsack.pdf"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  Figure 2 saved to {out_path}")


# ======================================================================
# Figure 3: Method Ranking Summary
# ======================================================================

def generate_ranking_figure(hc_df: pd.DataFrame, kn_df: pd.DataFrame, output_dir: Path):
    """Compact method ranking visualization across all conditions."""
    plt = _setup_matplotlib()

    metrics = ["test_regret_normalized", "test_fairness", "test_pred_mse"]
    metric_display = {
        "test_regret_normalized": "Regret",
        "test_fairness": "Fair. Viol.",
        "test_pred_mse": "Pred. MSE",
    }

    def _compute_ranks(df: pd.DataFrame, condition_cols: list[str]) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        if "lambda" not in df.columns:
            df["lambda"] = 0.0
        df["row_key"] = df["method_label"] + " lam=" + df["lambda"].fillna(0).astype(str)
        ranks_list = []
        for keys, grp in df.groupby(condition_cols):
            method_means = grp.groupby("row_key")[metrics].mean()
            for metric in metrics:
                ranked = method_means[metric].rank(ascending=True)
                for method, rank in ranked.items():
                    ranks_list.append({
                        "method": method, "metric": metric, "rank": rank,
                    })
        if not ranks_list:
            return pd.DataFrame()
        ranks_df = pd.DataFrame(ranks_list)
        return ranks_df.groupby(["method", "metric"])["rank"].mean().reset_index()

    hc_cond = ["alpha_fair"]
    if "hidden_dim" in hc_df.columns:
        hc_cond.append("hidden_dim")
    hc_ranks = _compute_ranks(hc_df, hc_cond)

    kn_cond = ["alpha_fair"]
    if "unfairness_level" in kn_df.columns:
        kn_cond.append("unfairness_level")
    kn_ranks = _compute_ranks(kn_df, kn_cond)

    if hc_ranks.empty and kn_ranks.empty:
        print("  Skipping ranking figure (no data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7, 4), sharey=True)

    for ax, ranks, title in [(axes[0], hc_ranks, "Healthcare"),
                              (axes[1], kn_ranks, "Knapsack")]:
        if ranks.empty:
            ax.set_title(f"{title} (no data)")
            continue

        # Pivot to method x metric matrix
        pivot = ranks.pivot(index="method", columns="metric", values="rank")
        # Reorder methods
        ordered_methods = [m for m in [
            "PTO", "FPTO_0.5", "FPTO_1.0", "FPTO_5.0",
            "SAA", "WDRO",
            "FDFL-Scal_0.5", "FDFL-Scal_1.0", "FDFL-Scal_5.0",
            "FDFL-PCGrad", "FDFL-MGDA", "FDFL-CAGrad",
        ] if m in pivot.index]
        pivot = pivot.reindex(ordered_methods)

        ordered_metrics = [m for m in metrics if m in pivot.columns]
        pivot = pivot[ordered_metrics]

        im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto",
                       vmin=1, vmax=len(ordered_methods))

        # Annotate
        for i in range(len(ordered_methods)):
            for j in range(len(ordered_metrics)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=7, fontweight="bold" if val <= 3 else "normal")

        ax.set_xticks(range(len(ordered_metrics)))
        ax.set_xticklabels([metric_display.get(m, m) for m in ordered_metrics],
                           rotation=30, ha="right")
        ax.set_yticks(range(len(ordered_methods)))
        ax.set_yticklabels([METHOD_DISPLAY.get(m, m) for m in ordered_methods])
        ax.set_title(title)

    fig.colorbar(im, ax=axes, shrink=0.6, label="Average rank (1 = best)")
    fig.suptitle("Method Ranking Across Experiments", fontsize=11, y=1.02)
    plt.tight_layout()

    out_path = output_dir / "fig3_ranking_summary.pdf"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  Figure 3 saved to {out_path}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--healthcare-dir", default=str(REPO_ROOT / "results" / "final" / "healthcare"))
    parser.add_argument("--knapsack-dir", default=str(REPO_ROOT / "results" / "final" / "knapsack"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "final" / "figures"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    hc_df = _load_results(args.healthcare_dir)
    kn_df = _load_results(args.knapsack_dir)
    kn_iter = _load_iter_logs(args.knapsack_dir)

    if not hc_df.empty:
        print(f"  Healthcare: {len(hc_df)} stage rows")
    if not kn_df.empty:
        print(f"  Knapsack:   {len(kn_df)} stage rows")
    if not kn_iter.empty:
        print(f"  Knapsack iter logs: {len(kn_iter)} rows")

    print("\nGenerating figures...")
    generate_pareto_figure(hc_df, output_dir)
    generate_cosine_figure(kn_df, kn_iter, output_dir)
    generate_ranking_figure(hc_df, kn_df, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
