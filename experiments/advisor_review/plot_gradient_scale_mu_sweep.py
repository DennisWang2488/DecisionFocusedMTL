"""Gradient-scale diagnostics figure for the FDFL mu sweep.

Reads iter_logs from the MD knapsack mu sweep (Exp 1) and the
healthcare v2 FDFL mu runs (Exp 2), and plots grad_norm_dec vs.
grad_norm_pred on a shared log-y axis for FDFL at mu in
{0, 0.1, 0.5, 1}.

Top row:    MD knapsack (SPSA, n_train=300, alpha=2, mad)       — scale explosion
Bottom row: healthcare  (analytic gradients, full cohort)       — contrast (flat)

Output:
    results/advisor_review/md_knapsack_mu_sweep/
        fig_gradient_scale_mu_sweep.png
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

ADV_DIR = REPO_ROOT / "results" / "advisor_review"
MD_DIR = ADV_DIR / "md_knapsack_mu_sweep"
HC_DIR = ADV_DIR / "healthcare_followup_v2"
OUT_PNG = MD_DIR / "fig_gradient_scale_mu_sweep.png"

# FDFL family: (label, method_name, colour)
MU_VARIANTS = [
    (r"$\mu=0$",   "fdfl",      "#1f77b4"),
    (r"$\mu=0.1$", "fdfl-0.1",  "#2ca02c"),
    (r"$\mu=0.5$", "fdfl-0.5",  "#ff7f0e"),
    (r"$\mu=1$",   "fdfl-scal", "#d62728"),
]


def _load_md_iter_logs() -> pd.DataFrame:
    """Load MD knapsack iter_logs across seeds."""
    frames = []
    for sub in sorted(MD_DIR.glob("seed_*")):
        csv = sub / "iter_logs.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_hc_iter_logs() -> pd.DataFrame:
    """Load healthcare Variant A FDFL mu iter_logs.

    We use the mad+alpha=2.0 cells to match the paper-section setup.
    """
    frames = []
    for ft in ["mad"]:
        for a in [2.0]:
            for sub in sorted((HC_DIR / "variant_a" / ft / f"alpha_{a}").glob("seed_*")):
                csv = sub / "iter_logs_fdfl_mu.csv"
                if csv.exists():
                    df = pd.read_csv(csv)
                    df["fairness_type"] = ft
                    df["alpha_fair"] = a
                    frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _plot_row(ax_dec, ax_pred, df: pd.DataFrame, title_prefix: str) -> None:
    """Plot grad_norm_dec and grad_norm_pred for each mu variant vs. iter."""
    if df.empty:
        for ax in (ax_dec, ax_pred):
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    # Filter train-phase rows only (no val_check)
    if "stage_type" in df.columns:
        df = df[df["stage_type"].fillna("train") == "train"]

    # Average across seeds & lambdas for a median-like summary, plus 5-95% band
    for label, method, colour in MU_VARIANTS:
        sub = df[df["method"].str.lower() == method]
        if sub.empty:
            continue
        for ax, col in [(ax_dec, "grad_norm_dec"), (ax_pred, "grad_norm_pred")]:
            if col not in sub.columns:
                continue
            by_iter = sub.groupby("iter")[col]
            med = by_iter.median()
            q05 = by_iter.quantile(0.05)
            q95 = by_iter.quantile(0.95)
            ax.plot(med.index, med.values, color=colour, label=label, linewidth=1.8)
            ax.fill_between(
                med.index, np.maximum(q05.values, 1e-12), np.maximum(q95.values, 1e-12),
                color=colour, alpha=0.15, linewidth=0,
            )

    ax_dec.set_yscale("log")
    ax_pred.set_yscale("log")
    ax_dec.set_title(f"{title_prefix}: $\\|\\nabla_\\theta L_{{dec}}\\|$")
    ax_pred.set_title(f"{title_prefix}: $\\|\\nabla_\\theta L_{{pred}}\\|$")
    ax_dec.set_xlabel("iteration")
    ax_pred.set_xlabel("iteration")
    ax_dec.set_ylabel("gradient norm (log)")
    ax_dec.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_pred.grid(True, which="both", linestyle=":", alpha=0.5)


def main() -> None:
    md_df = _load_md_iter_logs()
    hc_df = _load_hc_iter_logs()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    _plot_row(axes[0, 0], axes[0, 1], md_df,
              "MD knapsack (SPSA, $n=300$, $\\alpha=2$)")
    _plot_row(axes[1, 0], axes[1, 1], hc_df,
              "Healthcare (analytic, full cohort, $\\alpha=2$)")

    # Shared legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 1.04), frameon=False)

    fig.suptitle(
        "FDFL gradient-norm scale across $\\mu \\in \\{0, 0.1, 0.5, 1\\}$",
        y=1.08, fontsize=12,
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"[saved] {OUT_PNG}")
    plt.close(fig)


if __name__ == "__main__":
    main()
