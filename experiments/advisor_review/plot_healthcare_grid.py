"""Pareto-frontier plots for the healthcare follow-up grid.

For each (fairness_type, alpha) cell, draw a single figure with:
  x-axis: test_fairness (the per-fairness-type measure being optimised)
  y-axis: test_regret_normalized
  - FPTO / FDFL-Scal / FPLG appear as Pareto curves (lambda sweep)
  - MGDA / PCGrad / SAA / WDRO appear as single points
  - Error bars: +/- 1 std across the 5 seeds
  - Colours / markers from experiments.configs.COLOR_MAP / MARKER_MAP

Also draws an 8-panel grid figure showing all cells side-by-side.

Usage:
    python -m experiments.advisor_review.plot_healthcare_grid
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.advisor_review.analyze_healthcare_grid import (  # noqa: E402
    aggregate_cell,
    load_cell,
    load_grid,
)
from experiments.advisor_review.healthcare_followup import (  # noqa: E402
    HC_ALPHAS,
    HC_DIR,
    HC_FAIRNESS_TYPES,
)
from experiments.configs import COLOR_MAP, MARKER_MAP  # noqa: E402


# Methods that have a meaningful lambda sweep (use_fair=True, no mo_method).
PARETO_CURVE_METHODS = {"fpto", "fdfl-scal", "fplg"}
# Methods rendered as single Pareto points.
SINGLE_POINT_METHODS = {"mgda", "pcgrad", "saa", "wdro"}


def _resolve_color(method_name: str) -> str:
    """COLOR_MAP keys are the canonical method labels; method_name in the CSV
    is lowercase. Map lowercase -> canonical label."""
    canonical = {k.lower(): k for k in COLOR_MAP}
    return COLOR_MAP.get(canonical.get(method_name, method_name), "#888888")


def _resolve_marker(method_name: str) -> str:
    canonical = {k.lower(): k for k in MARKER_MAP}
    return MARKER_MAP.get(canonical.get(method_name, method_name), "o")


def plot_cell(
    fairness_type: str,
    alpha: float,
    out_path: Path,
    title: str | None = None,
) -> None:
    df = load_cell(fairness_type, alpha)
    if df is None:
        print(f"  [skip] no data for {fairness_type}/alpha_{alpha}")
        return
    agg = aggregate_cell(df).reset_index()

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for method in agg["method"].unique():
        sub = agg[agg["method"] == method].sort_values("lambda")
        color = _resolve_color(method)
        marker = _resolve_marker(method)

        if method in PARETO_CURVE_METHODS and len(sub) > 1:
            ax.errorbar(
                sub["test_fair_mean"],
                sub["test_reg_n_mean"],
                xerr=sub["test_fair_std"],
                yerr=sub["test_reg_n_std"],
                color=color,
                marker=marker,
                linestyle="-",
                markersize=8,
                linewidth=1.5,
                capsize=3,
                label=method,
                alpha=0.85,
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
                capsize=3,
                label=method,
                alpha=0.95,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.6,
            )

    ax.set_xlabel(f"test fairness ({fairness_type})")
    ax.set_ylabel("test regret (normalized)")
    if title is None:
        title = f"Healthcare — fairness={fairness_type}, alpha={alpha}"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


def plot_grid(out_path: Path) -> None:
    """8-panel grid: rows = fairness types, columns = alphas."""
    df_all = load_grid()
    n_rows = len(HC_FAIRNESS_TYPES)
    n_cols = len(HC_ALPHAS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6.0 * n_cols, 4.5 * n_rows), sharey=False
    )
    if n_rows == 1:
        axes = [axes]

    for i, ft in enumerate(HC_FAIRNESS_TYPES):
        for j, a in enumerate(HC_ALPHAS):
            ax = axes[i][j] if n_rows > 1 else axes[j]
            cell = df_all[(df_all["fairness_type"] == ft) & (df_all["alpha_fair"] == a)]
            if cell.empty:
                ax.set_title(f"{ft}, alpha={a}\n(no data)")
                continue
            agg = aggregate_cell(cell).reset_index()
            for method in agg["method"].unique():
                sub = agg[agg["method"] == method].sort_values("lambda")
                color = _resolve_color(method)
                marker = _resolve_marker(method)
                if method in PARETO_CURVE_METHODS and len(sub) > 1:
                    ax.errorbar(
                        sub["test_fair_mean"], sub["test_reg_n_mean"],
                        xerr=sub["test_fair_std"], yerr=sub["test_reg_n_std"],
                        color=color, marker=marker, linestyle="-",
                        markersize=6, linewidth=1.2, capsize=2,
                        label=method if (i == 0 and j == 0) else None,
                        alpha=0.85,
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
            ax.set_title(f"{ft}, alpha={a}")
            ax.grid(alpha=0.3)
            if i == n_rows - 1:
                ax.set_xlabel(f"test fairness ({ft})")
            if j == 0:
                ax.set_ylabel("test regret (norm.)")
    # One legend across the whole figure
    handles, labels = axes[0][0].get_legend_handles_labels() if n_rows > 1 else axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved grid] {out_path}")


def main() -> None:
    print("Healthcare grid plots")
    print("=====================")
    plots_dir = HC_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for ft in HC_FAIRNESS_TYPES:
        for a in HC_ALPHAS:
            plot_cell(
                ft, a,
                out_path=plots_dir / f"pareto_{ft}_alpha_{a}.png",
            )
    plot_grid(out_path=plots_dir / "pareto_grid_8cells.png")
    print(f"[done] plots written to {plots_dir}")


if __name__ == "__main__":
    main()
