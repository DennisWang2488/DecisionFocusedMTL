"""Generate paper-ready figures for healthcare v2 Variant A (5 seeds).

Outputs go to ``results/advisor_review/healthcare_followup_v2/paper/figures/``:
- ``fig_pareto_healthcare.pdf`` — 8-panel Pareto grid (train dashed, test solid)
- ``fig_gradient_healthcare.pdf`` — bar chart of pairwise gradient cosines
  (dec-pred, dec-fair, pred-fair) per method, at lambda=0 on the MAD cell
  across alpha in {0.5, 2.0}

Matches the style expected by the IJOC paper's Plots/ directory.
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
    load_variant,
)
from experiments.advisor_review.healthcare_followup_v2 import (  # noqa: E402
    HC_V2_ALPHAS,
    HC_V2_DIR,
    HC_V2_FAIRNESS_TYPES,
)
from experiments.configs import COLOR_MAP, MARKER_MAP  # noqa: E402


PARETO_CURVE_METHODS = {"fpto", "fdfl-scal", "fplg"}
METHOD_DISPLAY = {
    "fpto": "FPTO",
    "fdfl-scal": "FDFL-Scal",
    "fplg": "FPLG",
    "pcgrad": "FDFL-PCGrad",
    "mgda": "FDFL-MGDA",
    "saa": "SAA",
    "wdro": "WDRO",
}
FAIRNESS_DISPLAY = {
    "mad": "MAD",
    "dp": "DP",
    "atkinson": "Atkinson",
    "bias_parity": "BiasParity",
}


def _color(method: str) -> str:
    canon = {k.lower(): k for k in COLOR_MAP}
    return COLOR_MAP.get(canon.get(method, method), "#888888")


def _marker(method: str) -> str:
    canon = {k.lower(): k for k in MARKER_MAP}
    return MARKER_MAP.get(canon.get(method, method), "o")


# ---------------------------------------------------------------------------
# Figure 1: Pareto grid (8 panels)
# ---------------------------------------------------------------------------

def fig_pareto_healthcare(out_path: Path) -> None:
    df = load_variant("a")
    fig, axes = plt.subplots(
        len(HC_V2_FAIRNESS_TYPES),
        len(HC_V2_ALPHAS),
        figsize=(11.0, 14.0),
    )
    for i, ft in enumerate(HC_V2_FAIRNESS_TYPES):
        for j, a in enumerate(HC_V2_ALPHAS):
            ax = axes[i][j]
            cell = df[(df["fairness_type"] == ft) & (df["alpha_fair"] == a)]
            if cell.empty:
                ax.set_title(f"{FAIRNESS_DISPLAY[ft]}, $\\alpha$={a}")
                continue
            agg = aggregate_cell_v2(cell).reset_index()
            for method in agg["method"].unique():
                sub = agg[agg["method"] == method].sort_values("lambda")
                color = _color(method)
                marker = _marker(method)
                label = METHOD_DISPLAY.get(method, method)
                if method in PARETO_CURVE_METHODS and len(sub) > 1:
                    ax.errorbar(
                        sub["test_fair_mean"], sub["test_reg_n_mean"],
                        xerr=sub["test_fair_std"], yerr=sub["test_reg_n_std"],
                        color=color, marker=marker, linestyle="-",
                        markersize=6, linewidth=1.3, capsize=2,
                        label=label if (i == 0 and j == 0) else None,
                        alpha=0.9,
                    )
                    # Train overlay (open markers, dashed)
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
                        label=label if (i == 0 and j == 0) else None,
                        markerfacecolor=color, markeredgecolor="black",
                        markeredgewidth=0.5, alpha=0.95,
                    )
                    ax.plot(
                        sub["train_fair_mean"], sub["train_reg_n_mean"],
                        color=color, marker=marker, markerfacecolor="none",
                        markeredgecolor=color, markeredgewidth=1.0,
                        linestyle="None", markersize=7, alpha=0.5,
                    )
            ax.set_title(f"{FAIRNESS_DISPLAY[ft]}, $\\alpha$={a}", fontsize=11)
            ax.grid(alpha=0.3)
            if i == len(HC_V2_FAIRNESS_TYPES) - 1:
                ax.set_xlabel(f"Test {FAIRNESS_DISPLAY[ft]} violation")
            if j == 0:
                ax.set_ylabel("Normalised regret")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=7, fontsize=9,
        frameon=False, bbox_to_anchor=(0.5, 1.005),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path} and .png")


# ---------------------------------------------------------------------------
# Figure 2: Gradient cosines
# ---------------------------------------------------------------------------

def _collect_cosines(fairness_type: str = "mad") -> pd.DataFrame:
    """Load per-iteration cosines from all seeds' iter_logs at stage_idx=0."""
    base = HC_V2_DIR / "variant_a" / fairness_type
    rows = []
    for alpha in HC_V2_ALPHAS:
        adir = base / f"alpha_{alpha}"
        if not adir.exists():
            continue
        for seed_dir in sorted(adir.glob("seed_*")):
            iter_csv = seed_dir / "iter_logs.csv"
            if not iter_csv.exists():
                continue
            df = pd.read_csv(iter_csv)
            if "stage_type" in df.columns:
                df = df[df["stage_type"] == "train"]
            # stage_idx=0 corresponds to lambda=0 (first lambda in the sweep)
            df = df[df["stage_idx"] == 0]
            # Drop baselines (no end-to-end training) and FPTO (no decision gradient,
            # so dec cosines are trivially 0 and not informative for the plot)
            df = df[~df["method"].isin(["saa", "wdro", "fpto"])]
            df = df.dropna(subset=["cos_dec_pred", "cos_dec_fair", "cos_pred_fair"])
            for _, r in df.iterrows():
                rows.append(
                    {
                        "alpha": float(alpha),
                        "seed": int(seed_dir.name[len("seed_"):]),
                        "method": r["method"],
                        "cos_dec_pred": float(r["cos_dec_pred"]),
                        "cos_dec_fair": float(r["cos_dec_fair"]),
                        "cos_pred_fair": float(r["cos_pred_fair"]),
                    }
                )
    return pd.DataFrame(rows)


def fig_gradient_healthcare(out_path: Path) -> None:
    raw = _collect_cosines(fairness_type="mad")
    if raw.empty:
        raise RuntimeError("No cosine data found.")
    # per-run average: mean over iterations within each (method, seed, alpha)
    per_run = (
        raw.groupby(["alpha", "method", "seed"])
        .agg(
            cos_dec_pred=("cos_dec_pred", "mean"),
            cos_dec_fair=("cos_dec_fair", "mean"),
            cos_pred_fair=("cos_pred_fair", "mean"),
        )
        .reset_index()
    )
    # per-method mean +/- std across seeds
    stats = (
        per_run.groupby(["alpha", "method"])
        .agg(
            dp_mean=("cos_dec_pred", "mean"),
            dp_std=("cos_dec_pred", "std"),
            df_mean=("cos_dec_fair", "mean"),
            df_std=("cos_dec_fair", "std"),
            pf_mean=("cos_pred_fair", "mean"),
            pf_std=("cos_pred_fair", "std"),
        )
        .reset_index()
    )
    # Fixed method order for both panels
    method_order = ["fdfl-scal", "fplg", "pcgrad", "mgda"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for ax, alpha in zip(axes, HC_V2_ALPHAS):
        sub = stats[stats["alpha"] == alpha].set_index("method").reindex(method_order)
        x = np.arange(len(sub))
        width = 0.25
        # Pair 1: dec vs pred
        ax.bar(
            x - width,
            sub["dp_mean"].values,
            width,
            yerr=sub["dp_std"].values,
            label=r"$\cos(\nabla_{\mathrm{dec}}, \nabla_{\mathrm{pred}})$",
            color="#1f77b4",
            capsize=3,
        )
        # Pair 2: dec vs fair
        ax.bar(
            x,
            sub["df_mean"].values,
            width,
            yerr=sub["df_std"].values,
            label=r"$\cos(\nabla_{\mathrm{dec}}, \nabla_{\mathrm{fair}})$",
            color="#ff7f0e",
            capsize=3,
        )
        # Pair 3: pred vs fair
        ax.bar(
            x + width,
            sub["pf_mean"].values,
            width,
            yerr=sub["pf_std"].values,
            label=r"$\cos(\nabla_{\mathrm{pred}}, \nabla_{\mathrm{fair}})$",
            color="#2ca02c",
            capsize=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_DISPLAY[m] for m in method_order], rotation=15, fontsize=9)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        ax.set_title(rf"$\alpha={alpha}$", fontsize=12)
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylim(-0.6, 1.05)
    axes[0].set_ylabel("Cosine similarity")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, fontsize=10,
        frameon=False, bbox_to_anchor=(0.5, 1.04),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path} and .png")

    # Also dump the raw cosine summary for the text numbers
    stats.to_csv(out_path.parent / "gradient_cosine_summary.csv", index=False)
    print(f"  [saved] gradient_cosine_summary.csv")


def main() -> None:
    paper_dir = HC_V2_DIR / "paper"
    fig_dir = paper_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_pareto_healthcare(fig_dir / "fig_pareto_healthcare.pdf")
    fig_gradient_healthcare(fig_dir / "fig_gradient_healthcare.pdf")
    print(f"[done] {fig_dir}")


if __name__ == "__main__":
    main()
