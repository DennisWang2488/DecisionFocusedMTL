"""Aggregate the AlignMO Phase 1 pilot and emit all Section 4.5 deliverables.

Produces, under ``results/pilot_alignmo/``:

- ``grand_summary.csv``      — one row per (method, alpha, regime, seed, lambda)
- ``per_cell_winners.csv``   — one row per (alpha, regime)
- ``diagnostic_profile.csv`` — medians of diagnostic EMAs per (alpha, regime, method)
- ``fig_per_cell_winners.png`` — heatmap, 4 rows (alpha) x 2 cols (regime)
- ``GO_NO_GO.md``            — memo with verdict per Section 4.4 decision rule

Decision rule (commit BEFORE looking at data, see ALIGNMO_PLAN.md 4.4):
    # distinct methods winning >=1 cell (of 8 cells)
        >= 4  : GO
        2-3   : CAUTIOUS GO
        == 1  : NO GO

Usage:
    python -m experiments.advisor_review.aggregate_alignmo_pilot
    python -m experiments.advisor_review.aggregate_alignmo_pilot --pilot-dir results/pilot_alignmo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PILOT_DIR_DEFAULT = REPO_ROOT / "results" / "pilot_alignmo"

REGIMES = ("analytic", "spsa")
METRIC = "test_regret_normalized"

DIAG_ITER_COLS = ["cos_dec_pred", "cos_dec_fair", "cos_pred_fair",
                  "grad_norm_dec", "grad_norm_pred", "grad_norm_fair"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _iter_cells(pilot_dir: Path):
    for regime in REGIMES:
        regime_dir = pilot_dir / regime
        if not regime_dir.is_dir():
            continue
        for alpha_dir in sorted(regime_dir.glob("alpha_*")):
            try:
                alpha = float(alpha_dir.name.split("_", 1)[1])
            except ValueError:
                continue
            for seed_dir in sorted(alpha_dir.glob("seed_*")):
                try:
                    seed = int(seed_dir.name.split("_", 1)[1])
                except ValueError:
                    continue
                yield regime, alpha, seed, seed_dir


def load_grand_summary(pilot_dir: Path) -> pd.DataFrame:
    """Walk the pilot directory tree and aggregate stage_results + diagnostic EMAs.

    For each (regime, alpha, seed), opens stage_results.csv and appends
    columns `regime` and `alpha`. Also computes per-stage medians of the
    iter-log diagnostics (pairwise cosines, per-objective grad norms) and
    merges them in.
    """
    rows: List[pd.DataFrame] = []
    for regime, alpha, seed, seed_dir in _iter_cells(pilot_dir):
        stage_csv = seed_dir / "stage_results.csv"
        iter_csv = seed_dir / "iter_logs.csv"
        if not stage_csv.exists():
            continue
        stage = pd.read_csv(stage_csv)
        stage["regime"] = regime
        stage["alpha"] = alpha

        if iter_csv.exists():
            it = pd.read_csv(iter_csv)
            keys = ["method", "seed", "stage_idx", "lambda"]
            diag_cols = [c for c in DIAG_ITER_COLS if c in it.columns]
            if diag_cols:
                agg_med = (
                    it.groupby(keys)[diag_cols]
                    .median()
                    .reset_index()
                    .rename(columns={c: f"iter_med_{c}" for c in diag_cols})
                )
                stage = stage.merge(agg_med, on=keys, how="left")
                if {"grad_norm_dec", "grad_norm_pred", "grad_norm_fair"}.issubset(diag_cols):
                    # log-scale ratios: r_dp = log(|g_d|/|g_p|), r_df = log(|g_d|/|g_f|)
                    eps = 1e-12
                    g_d = it["grad_norm_dec"].clip(lower=eps)
                    g_p = it["grad_norm_pred"].clip(lower=eps)
                    g_f = it["grad_norm_fair"].clip(lower=eps)
                    it["_r_dp"] = np.log(g_d) - np.log(g_p)
                    it["_r_df"] = np.log(g_d) - np.log(g_f)
                    rag = (
                        it.groupby(keys)[["_r_dp", "_r_df"]]
                        .median()
                        .reset_index()
                        .rename(columns={"_r_dp": "iter_med_r_dp",
                                         "_r_df": "iter_med_r_df"})
                    )
                    stage = stage.merge(rag, on=keys, how="left")
        rows.append(stage)

    if not rows:
        raise RuntimeError(f"No stage_results.csv found under {pilot_dir}")
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-cell winners (TODO 1.5)
# ---------------------------------------------------------------------------

def build_per_cell_winners(grand: pd.DataFrame) -> pd.DataFrame:
    """For each (alpha, regime) cell, pick
    best_method = argmin over methods of (min over lambda of mean-over-seeds of METRIC).
    """
    rows = []
    for (alpha, regime), cell in grand.groupby(["alpha", "regime"]):
        per_method = []
        for method, mdf in cell.groupby("method"):
            per_lam = mdf.groupby("lambda")[METRIC].agg(["mean", "std", "count"])
            if per_lam.empty or per_lam["mean"].isna().all():
                continue
            best_lam = float(per_lam["mean"].idxmin())
            r = per_lam.loc[best_lam]
            per_method.append({
                "method": method,
                "best_lambda": best_lam,
                "mean": float(r["mean"]),
                "std": float(r["std"]) if not np.isnan(r["std"]) else 0.0,
                "n_seeds": int(r["count"]),
            })
        if not per_method:
            continue
        pm = pd.DataFrame(per_method).sort_values("mean", ascending=True).reset_index(drop=True)
        best = pm.iloc[0]
        runner = pm.iloc[1] if len(pm) > 1 else None
        gap = float(runner["mean"] - best["mean"]) if runner is not None else float("nan")
        seed_std = float(best["std"]) if best["std"] > 0 else float("nan")
        ratio = gap / seed_std if (seed_std and not np.isnan(seed_std) and seed_std > 0) else float("nan")
        rows.append({
            "alpha": alpha,
            "regime": regime,
            "best_method": best["method"],
            "best_regret_mean": best["mean"],
            "best_regret_std": best["std"],
            "best_lambda": best["best_lambda"],
            "runner_up_method": runner["method"] if runner is not None else None,
            "runner_up_regret_mean": float(runner["mean"]) if runner is not None else float("nan"),
            "gap": gap,
            "gap_seed_std_ratio": ratio,
        })
    return pd.DataFrame(rows).sort_values(["regime", "alpha"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Diagnostic profile (TODO 1.6)
# ---------------------------------------------------------------------------

def build_diagnostic_profile(grand: pd.DataFrame) -> pd.DataFrame:
    """One row per (alpha, regime, method): medians of the diagnostic EMAs
    (pairwise cosines + log-scale ratios) across seeds and lambdas.
    """
    diag_cols = [c for c in grand.columns
                 if c.startswith("iter_med_") or c.startswith("avg_cos_")]
    if not diag_cols:
        return pd.DataFrame()
    grp = grand.groupby(["alpha", "regime", "method"])[diag_cols].median().reset_index()
    return grp.sort_values(["regime", "alpha", "method"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Heatmap figure (TODO 1.7)
# ---------------------------------------------------------------------------

def plot_per_cell_winners(winners: pd.DataFrame, out_png: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    alphas = sorted(winners["alpha"].unique())
    regimes = [r for r in REGIMES if r in set(winners["regime"])]
    methods = sorted(winners["best_method"].unique())
    cmap = plt.get_cmap("tab10")
    color_of = {m: cmap(i % 10) for i, m in enumerate(methods)}

    fig, ax = plt.subplots(figsize=(max(5, 2 + 1.6 * len(regimes)),
                                     max(4, 1 + 1.2 * len(alphas))))
    for i, a in enumerate(alphas):
        for j, r in enumerate(regimes):
            sub = winners[(winners["alpha"] == a) & (winners["regime"] == r)]
            if sub.empty:
                continue
            m = sub.iloc[0]["best_method"]
            gap = sub.iloc[0]["gap"]
            ratio = sub.iloc[0]["gap_seed_std_ratio"]
            color = color_of[m]
            ax.add_patch(plt.Rectangle((j, len(alphas) - 1 - i), 1, 1,
                                       facecolor=color, edgecolor="black"))
            label = f"{m}\n(gap={gap:.3f},\nratio={ratio:.2f})"
            ax.text(j + 0.5, len(alphas) - 1 - i + 0.5, label,
                    ha="center", va="center", fontsize=8)
    ax.set_xlim(0, len(regimes))
    ax.set_ylim(0, len(alphas))
    ax.set_xticks([x + 0.5 for x in range(len(regimes))])
    ax.set_xticklabels(regimes)
    ax.set_yticks([y + 0.5 for y in range(len(alphas))])
    ax.set_yticklabels([f"alpha={a}" for a in reversed(alphas)])
    ax.set_xlabel("decision-gradient regime")
    ax.set_title("Per-cell winners (Phase 1 pilot)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color_of[m], edgecolor="black", label=m)
               for m in methods]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
              borderaxespad=0, title="winner")
    ax.set_aspect("equal")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# GO / NO-GO memo (TODO 1.8)
# ---------------------------------------------------------------------------

def verdict_from_winners(winners: pd.DataFrame) -> tuple[str, int, str]:
    n_distinct = int(winners["best_method"].nunique())
    if n_distinct >= 4:
        return "GREEN", n_distinct, "GO — Phase 2 as planned."
    if n_distinct >= 2:
        return "YELLOW", n_distinct, ("CAUTIOUS GO — Phase 2 with tightened framing "
                                       "(AlignMO routes between 2-3 empirically separated regimes).")
    return "RED", n_distinct, "NO GO — drop AlignMO, reposition around the dominant fixed method."


def write_go_no_go(winners: pd.DataFrame, out_md: Path) -> None:
    banner, n, recommendation = verdict_from_winners(winners)
    lines = []
    lines.append(f"# GO / NO-GO memo — AlignMO Phase 1 pilot")
    lines.append("")
    lines.append(f"**Verdict:** {banner}")
    lines.append(f"**Distinct cell-winners (out of {len(winners)} cells):** {n}")
    lines.append(f"**Recommendation:** {recommendation}")
    lines.append("")
    lines.append("## Per-cell table")
    lines.append("")
    lines.append("| regime | alpha | winner | best_lambda | mean | std | runner_up | gap | gap/seed_std |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, r in winners.iterrows():
        lines.append(
            f"| {r['regime']} | {r['alpha']} | {r['best_method']} | {r['best_lambda']:.2f} | "
            f"{r['best_regret_mean']:.4f} | {r['best_regret_std']:.4f} | "
            f"{r['runner_up_method']} | {r['gap']:.4f} | {r['gap_seed_std_ratio']:.2f} |"
        )
    lines.append("")
    lines.append("## Decision rule (ALIGNMO_PLAN.md Section 4.4)")
    lines.append("")
    lines.append("- >= 4 distinct winners: GREEN (GO).")
    lines.append("- 2-3 distinct winners: YELLOW (CAUTIOUS GO).")
    lines.append("- 1 distinct winner: RED (NO GO, reposition).")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- Cells with `gap_seed_std_ratio` < 1 are within seed noise; ")
    lines.append("  escalate to 5 seeds per Section 4.7 before calling a tight win.")
    lines.append("- A secondary diagnostic check lives in `diagnostic_profile.csv`.")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pilot-dir", type=Path, default=PILOT_DIR_DEFAULT)
    args = p.parse_args()

    pilot_dir = Path(args.pilot_dir)
    print(f"[aggregate] pilot_dir = {pilot_dir}")

    grand = load_grand_summary(pilot_dir)
    grand_csv = pilot_dir / "grand_summary.csv"
    grand.to_csv(grand_csv, index=False)
    print(f"[aggregate] grand_summary: {len(grand)} rows -> {grand_csv}")

    winners = build_per_cell_winners(grand)
    winners_csv = pilot_dir / "per_cell_winners.csv"
    winners.to_csv(winners_csv, index=False)
    print(f"[aggregate] per_cell_winners: {len(winners)} rows -> {winners_csv}")

    diag = build_diagnostic_profile(grand)
    diag_csv = pilot_dir / "diagnostic_profile.csv"
    diag.to_csv(diag_csv, index=False)
    print(f"[aggregate] diagnostic_profile: {len(diag)} rows -> {diag_csv}")

    fig_png = pilot_dir / "fig_per_cell_winners.png"
    plot_per_cell_winners(winners, fig_png)
    print(f"[aggregate] figure -> {fig_png}")

    go_md = pilot_dir / "GO_NO_GO.md"
    write_go_no_go(winners, go_md)
    banner, n, _ = verdict_from_winners(winners)
    print(f"[aggregate] verdict = {banner} ({n} distinct winners) -> {go_md}")


if __name__ == "__main__":
    main()
