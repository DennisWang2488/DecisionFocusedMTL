"""Plots for the healthcare follow-up v2 grid.

New plots vs v1:
- Per-cell Pareto plot with TRAIN (dashed) and TEST (solid) overlaid
- Cross-fairness rank heatmap (method x (ft, alpha), coloured by rank)
- Cross-fairness ratio heatmap (vs FPTO lambda=0)
- Variant A vs Variant B side-by-side comparison
- Variant B early-stop-step histogram

Usage:
    python -m experiments.advisor_review.plot_healthcare_v2
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.advisor_review.analyze_healthcare_v2 import (  # noqa: E402
    aggregate_cell_v2,
    cross_fairness_rank,
    cross_fairness_ratio,
    grand_summary_v2,
    load_both_variants,
    load_variant,
    variant_dir,
)
from experiments.advisor_review.healthcare_followup_v2 import (  # noqa: E402
    HC_V2_ALPHAS,
    HC_V2_DIR,
    HC_V2_FAIRNESS_TYPES,
)
from experiments.configs import COLOR_MAP, MARKER_MAP  # noqa: E402


PARETO_CURVE_METHODS = {"fpto", "fdfl-scal", "fplg"}


def _color(method: str) -> str:
    canon = {k.lower(): k for k in COLOR_MAP}
    return COLOR_MAP.get(canon.get(method, method), "#888888")


def _marker(method: str) -> str:
    canon = {k.lower(): k for k in MARKER_MAP}
    return MARKER_MAP.get(canon.get(method, method), "o")


# ---------------------------------------------------------------------------
# Per-cell Pareto plot with train + test overlay
# ---------------------------------------------------------------------------


def plot_cell_with_train_overlay(
    df_cell: pd.DataFrame,
    fairness_type: str,
    alpha: float,
    variant: str,
    out_path: Path,
) -> None:
    agg = aggregate_cell_v2(df_cell).reset_index()

    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    for method in agg["method"].unique():
        sub = agg[agg["method"] == method].sort_values("lambda")
        color = _color(method)
        marker = _marker(method)

        # TEST points (solid line)
        if method in PARETO_CURVE_METHODS and len(sub) > 1:
            ax.errorbar(
                sub["test_fair_mean"],
                sub["test_reg_n_mean"],
                xerr=sub["test_fair_std"],
                yerr=sub["test_reg_n_std"],
                color=color,
                marker=marker,
                linestyle="-",
                linewidth=1.5,
                markersize=8,
                capsize=2,
                label=method,
                alpha=0.9,
            )
        else:
            ax.errorbar(
                sub["test_fair_mean"],
                sub["test_reg_n_mean"],
                xerr=sub["test_fair_std"],
                yerr=sub["test_reg_n_std"],
                color=color,
                marker=marker,
                linestyle="None",
                markersize=11,
                capsize=2,
                label=method,
                alpha=0.95,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.6,
            )

        # TRAIN points (dashed line, no errorbars to reduce clutter)
        if method in PARETO_CURVE_METHODS and len(sub) > 1:
            ax.plot(
                sub["train_fair_mean"],
                sub["train_reg_n_mean"],
                color=color,
                marker=marker,
                markerfacecolor="none",
                linestyle="--",
                linewidth=1.0,
                markersize=6,
                alpha=0.55,
            )
        else:
            ax.plot(
                sub["train_fair_mean"],
                sub["train_reg_n_mean"],
                color=color,
                marker=marker,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.2,
                linestyle="None",
                markersize=9,
                alpha=0.55,
            )

    ax.set_xlabel(f"fairness ({fairness_type})  [solid=test, dashed=train]")
    ax.set_ylabel("regret (normalised)")
    ax.set_title(f"Healthcare v2 [{variant}] — {fairness_type}, alpha={alpha}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


def plot_all_cells_for_variant(variant: str) -> None:
    df = load_variant(variant)
    plots_dir = variant_dir(variant) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for ft in HC_V2_FAIRNESS_TYPES:
        for a in HC_V2_ALPHAS:
            cell = df[(df["fairness_type"] == ft) & (df["alpha_fair"] == a)]
            if cell.empty:
                continue
            plot_cell_with_train_overlay(
                cell, ft, a, variant,
                plots_dir / f"pareto_{ft}_alpha_{a}.png",
            )


def plot_grid_for_variant(variant: str) -> None:
    df = load_variant(variant)
    fig, axes = plt.subplots(
        len(HC_V2_FAIRNESS_TYPES),
        len(HC_V2_ALPHAS),
        figsize=(12.0, 18.0),
    )
    for i, ft in enumerate(HC_V2_FAIRNESS_TYPES):
        for j, a in enumerate(HC_V2_ALPHAS):
            ax = axes[i][j]
            cell = df[(df["fairness_type"] == ft) & (df["alpha_fair"] == a)]
            if cell.empty:
                ax.set_title(f"{ft}, alpha={a}\n(no data)")
                continue
            agg = aggregate_cell_v2(cell).reset_index()
            for method in agg["method"].unique():
                sub = agg[agg["method"] == method].sort_values("lambda")
                color = _color(method)
                marker = _marker(method)
                if method in PARETO_CURVE_METHODS and len(sub) > 1:
                    ax.errorbar(
                        sub["test_fair_mean"], sub["test_reg_n_mean"],
                        xerr=sub["test_fair_std"], yerr=sub["test_reg_n_std"],
                        color=color, marker=marker, linestyle="-",
                        markersize=6, linewidth=1.2, capsize=2,
                        label=method if (i == 0 and j == 0) else None,
                        alpha=0.9,
                    )
                    ax.plot(
                        sub["train_fair_mean"], sub["train_reg_n_mean"],
                        color=color, marker=marker, markerfacecolor="none",
                        linestyle="--", linewidth=0.9, markersize=5, alpha=0.5,
                    )
                else:
                    ax.errorbar(
                        sub["test_fair_mean"], sub["test_reg_n_mean"],
                        xerr=sub["test_fair_std"], yerr=sub["test_reg_n_std"],
                        color=color, marker=marker, linestyle="None",
                        markersize=9, capsize=2,
                        label=method if (i == 0 and j == 0) else None,
                        markerfacecolor=color, markeredgecolor="black",
                        markeredgewidth=0.5, alpha=0.95,
                    )
                    ax.plot(
                        sub["train_fair_mean"], sub["train_reg_n_mean"],
                        color=color, marker=marker, markerfacecolor="none",
                        markeredgecolor=color, markeredgewidth=1.0,
                        linestyle="None", markersize=7, alpha=0.5,
                    )
            ax.set_title(f"{ft}, alpha={a}")
            ax.grid(alpha=0.3)
            if i == len(HC_V2_FAIRNESS_TYPES) - 1:
                ax.set_xlabel(f"fairness ({ft})")
            if j == 0:
                ax.set_ylabel("regret (norm.)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"Healthcare v2 — Variant {variant.upper()} — train (dashed) / test (solid)",
                 y=1.03, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path = variant_dir(variant) / "plots" / "pareto_grid_8cells.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved grid] {out_path}")


# ---------------------------------------------------------------------------
# Cross-fairness heatmap
# ---------------------------------------------------------------------------


def plot_cross_fairness_rank_heatmap(out_path: Path) -> None:
    df = load_both_variants()
    gs = grand_summary_v2(df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for idx, variant in enumerate(["a", "b"]):
        ax = axes[idx]
        sub = gs[gs["variant"] == variant]
        if sub.empty:
            ax.set_title(f"Variant {variant.upper()}\n(no data)")
            continue
        pivot = sub.pivot_table(
            index="method",
            columns=["fairness_type", "alpha"],
            values="test_fair_at_fair_best",
            aggfunc="mean",
        )
        # Rank within each column
        ranks = pivot.rank(axis=0, method="average")
        im = ax.imshow(ranks.values, cmap="RdYlGn_r", vmin=1, vmax=7)
        ax.set_xticks(range(len(ranks.columns)))
        col_labels = [f"{ft}\na={a}" for ft, a in ranks.columns]
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(ranks.index)))
        ax.set_yticklabels(ranks.index, fontsize=9)
        ax.set_title(f"Variant {variant.upper()} — fairness rank heatmap\n(lower rank = fairer)")
        for i in range(len(ranks.index)):
            for j in range(len(ranks.columns)):
                val = ranks.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            color="black", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.7, label="rank (1=best, 7=worst)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved rank heatmap] {out_path}")


def plot_cross_fairness_ratio_heatmap(out_path: Path) -> None:
    df = load_both_variants()
    ratio = cross_fairness_ratio(df, baseline_method="fpto", baseline_lam=0.0)
    if ratio.empty:
        print("  [skip] ratio heatmap — no data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for idx, variant in enumerate(["a", "b"]):
        ax = axes[idx]
        sub = ratio[ratio["variant"] == variant]
        if sub.empty:
            ax.set_title(f"Variant {variant.upper()}\n(no data)")
            continue
        pivot = sub.pivot_table(
            index="method",
            columns=["fairness_type", "alpha"],
            values="rel_fair_mean",
        )
        im = ax.imshow(pivot.values, cmap="RdYlGn_r",
                       norm=plt.matplotlib.colors.LogNorm(vmin=0.01, vmax=10))
        ax.set_xticks(range(len(pivot.columns)))
        col_labels = [f"{ft}\na={a}" for ft, a in pivot.columns]
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_title(f"Variant {variant.upper()} — fairness / FPTO_lam0\n(log scale, <1 fairer, >1 worse)")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color="black", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved ratio heatmap] {out_path}")


# ---------------------------------------------------------------------------
# Train vs test gap bar chart
# ---------------------------------------------------------------------------


def plot_train_test_gap_bar(out_path: Path) -> None:
    df = load_both_variants()
    gs = grand_summary_v2(df)
    fig, ax = plt.subplots(figsize=(10, 5))
    methods = sorted(gs["method"].unique())
    width = 0.35
    x = np.arange(len(methods))
    for idx, variant in enumerate(["a", "b"]):
        sub = gs[gs["variant"] == variant]
        if sub.empty:
            continue
        means = []
        stds = []
        for m in methods:
            mdf = sub[sub["method"] == m]
            means.append(mdf["train_test_gap"].mean() if not mdf.empty else np.nan)
            stds.append(mdf["train_test_gap"].std() if not mdf.empty else np.nan)
        offset = -width / 2 if variant == "a" else width / 2
        ax.bar(
            x + offset, means, width, yerr=stds, capsize=2,
            label=f"Variant {variant.upper()}",
            color="steelblue" if variant == "a" else "indianred",
            alpha=0.85,
        )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("mean train->test gap (test_reg_n - train_reg_n)")
    ax.set_title("Train-to-test generalization gap across cells, by method")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved gap bar] {out_path}")


# ---------------------------------------------------------------------------
# Variant B early-stop histogram
# ---------------------------------------------------------------------------


def plot_early_stop_histogram(out_path: Path) -> None:
    df = load_variant("b")
    sub = df[df["early_stop_enabled"] == 1]
    if sub.empty:
        print("  [skip] early stop histogram — variant b has no data")
        return
    methods = sorted(sub["method"].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(2.8 * len(methods), 3.8), sharey=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, m in zip(axes, methods):
        rows = sub[sub["method"] == m]
        ax.hist(rows["early_stop_step"].values, bins=15, color=_color(m), alpha=0.85)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("early_stop_step")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("count")
    fig.suptitle("Variant B — early_stop_step distribution per method", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved early stop hist] {out_path}")


# ---------------------------------------------------------------------------
# Variant A vs Variant B side-by-side
# ---------------------------------------------------------------------------


def plot_variant_compare(out_path: Path) -> None:
    df = load_both_variants()
    gs = grand_summary_v2(df)

    fig, axes = plt.subplots(
        len(HC_V2_FAIRNESS_TYPES), len(HC_V2_ALPHAS),
        figsize=(12, 18), sharex=False, sharey=False,
    )
    for i, ft in enumerate(HC_V2_FAIRNESS_TYPES):
        for j, a in enumerate(HC_V2_ALPHAS):
            ax = axes[i][j]
            sub = gs[(gs["fairness_type"] == ft) & (gs["alpha"] == a)]
            if sub.empty:
                ax.set_title(f"{ft}, alpha={a}\n(no data)")
                continue
            for variant, marker_edge in [("a", "black"), ("b", "red")]:
                v_sub = sub[sub["variant"] == variant]
                for _, row in v_sub.iterrows():
                    m = row["method"]
                    ax.errorbar(
                        row["test_fair"], row["test_reg_n"],
                        xerr=row["test_fair_std"], yerr=row["test_reg_n_std"],
                        color=_color(m), marker=_marker(m),
                        markersize=10 if variant == "a" else 8,
                        markerfacecolor=_color(m),
                        markeredgecolor=marker_edge,
                        markeredgewidth=1.2 if variant == "a" else 2.0,
                        linestyle="None",
                        capsize=2,
                        alpha=0.9,
                    )
                    ax.annotate(
                        f"{m[:4]}_{variant}",
                        (row["test_fair"], row["test_reg_n"]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=6, alpha=0.7,
                    )
            ax.set_title(f"{ft}, alpha={a}\n(black=A, red=B)")
            ax.set_xlabel(f"fairness ({ft})")
            ax.set_ylabel("regret (norm.)")
            ax.grid(alpha=0.3)
    fig.suptitle("Variant A (no val) vs Variant B (val + early stop)", y=1.01, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved variant compare] {out_path}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main() -> None:
    print("Healthcare v2 plots")
    print("=" * 40)
    # Per-variant plots
    for v in ["a", "b"]:
        try:
            plot_all_cells_for_variant(v)
            plot_grid_for_variant(v)
        except RuntimeError as e:
            print(f"[skip variant {v}] {e}")
    # Cross-fairness + compare
    cf_dir = HC_V2_DIR / "cross_fairness" / "plots"
    cf_dir.mkdir(parents=True, exist_ok=True)
    plot_cross_fairness_rank_heatmap(cf_dir / "rank_heatmap.png")
    plot_cross_fairness_ratio_heatmap(cf_dir / "ratio_heatmap.png")
    plot_train_test_gap_bar(cf_dir / "train_test_gap_bar.png")
    plot_variant_compare(cf_dir / "variant_a_vs_b.png")
    # Variant B diagnostic
    try:
        plot_early_stop_histogram(variant_dir("b") / "plots" / "early_stop_histogram.png")
    except RuntimeError as e:
        print(f"[skip early stop histogram] {e}")


if __name__ == "__main__":
    main()
