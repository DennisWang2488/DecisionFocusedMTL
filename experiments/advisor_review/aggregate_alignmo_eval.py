"""Aggregate the AlignMO Phase 2 eval and produce the vs-pilot comparison.

Deliverables (see ALIGNMO_PLAN.md Section 5.3 TODO 2.4b and 5.4):

- ``results/alignmo_eval/grand_summary.csv`` — AlignMO stages joined
  with the pilot's fixed-method stages (source = pilot or eval).
- ``results/alignmo_eval/per_cell_alignmo_vs_best.csv`` — one row per
  (alpha, regime): best fixed method, AlignMO score, gap, seed-std
  ratio, do-no-harm flag, do-good flag.
- ``results/alignmo_eval/mode_trace.csv`` — AlignMO per-mode fraction
  per (alpha, regime), averaged across seeds and lambdas.
- ``results/alignmo_eval/acceptance.md`` — Phase 2 acceptance memo.

Usage:
    python -m experiments.advisor_review.aggregate_alignmo_eval
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

from experiments.advisor_review.aggregate_alignmo_pilot import (  # noqa: E402
    load_grand_summary as load_pilot_grand,
    METRIC,
)

EVAL_DIR = REPO_ROOT / "results" / "alignmo_eval"
PILOT_DIR = REPO_ROOT / "results" / "pilot_alignmo"

ALIGNMO_MODES = ("scalarized", "projected", "anchored", "anchored_projected")


def best_over_methods(df: pd.DataFrame, method_set) -> pd.DataFrame:
    rows = []
    for (alpha, regime), cell in df.groupby(["alpha", "regime"]):
        cell = cell[cell["method"].isin(method_set)]
        if cell.empty:
            continue
        per_method = []
        for method, mdf in cell.groupby("method"):
            per_lam = mdf.groupby("lambda")[METRIC].agg(["mean", "std"])
            if per_lam.empty or per_lam["mean"].isna().all():
                continue
            best_lam = float(per_lam["mean"].idxmin())
            r = per_lam.loc[best_lam]
            per_method.append({
                "method": method,
                "best_lambda": best_lam,
                "mean": float(r["mean"]),
                "std": float(r["std"]) if not np.isnan(r["std"]) else 0.0,
            })
        if not per_method:
            continue
        pm = pd.DataFrame(per_method).sort_values("mean").reset_index(drop=True)
        best = pm.iloc[0]
        rows.append({
            "alpha": alpha, "regime": regime,
            "best_method": best["method"],
            "best_mean": best["mean"],
            "best_std": best["std"],
            "best_lambda": best["best_lambda"],
        })
    return pd.DataFrame(rows)


def mode_trace_from_iters(eval_dir: Path) -> pd.DataFrame:
    rows = []
    for regime_dir in sorted(eval_dir.glob("*")):
        if not regime_dir.is_dir() or regime_dir.name not in ("analytic", "spsa"):
            continue
        regime = regime_dir.name
        for alpha_dir in sorted(regime_dir.glob("alpha_*")):
            alpha = float(alpha_dir.name.split("_", 1)[1])
            mode_counts = {m: 0 for m in ALIGNMO_MODES}
            total = 0
            for seed_dir in sorted(alpha_dir.glob("seed_*")):
                it_csv = seed_dir / "iter_logs.csv"
                if not it_csv.exists():
                    continue
                it = pd.read_csv(it_csv)
                col = "mode_this_step"
                if col not in it.columns:
                    continue
                for m in ALIGNMO_MODES:
                    mode_counts[m] += int((it[col] == m).sum())
                total += int(it[col].notna().sum())
            if total == 0:
                continue
            row = {"alpha": alpha, "regime": regime, "n_logged_iters": total}
            for m in ALIGNMO_MODES:
                row[f"frac_{m}"] = mode_counts[m] / total
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["regime", "alpha"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    p.add_argument("--pilot-dir", type=Path, default=PILOT_DIR)
    args = p.parse_args()

    eval_dir, pilot_dir = Path(args.eval_dir), Path(args.pilot_dir)

    pilot = load_pilot_grand(pilot_dir)
    pilot["source"] = "pilot"
    eval_df = load_pilot_grand(eval_dir)  # same loader — same layout
    eval_df["source"] = "eval"
    grand = pd.concat([pilot, eval_df], ignore_index=True)
    grand_csv = eval_dir / "grand_summary.csv"
    grand.to_csv(grand_csv, index=False)
    print(f"[agg] grand_summary: {len(grand)} rows -> {grand_csv}")

    fixed_methods = sorted(pilot["method"].unique())
    alignmo_methods = ["alignmo"]
    assert "alignmo" in set(eval_df["method"]), "alignmo missing from eval_dir"

    best_fixed = best_over_methods(pilot, fixed_methods).rename(columns={
        "best_method": "best_fixed",
        "best_mean": "best_fixed_mean",
        "best_std": "best_fixed_std",
        "best_lambda": "best_fixed_lambda",
    })
    best_align = best_over_methods(eval_df, alignmo_methods).rename(columns={
        "best_method": "alignmo_method",
        "best_mean": "alignmo_mean",
        "best_std": "alignmo_std",
        "best_lambda": "alignmo_lambda",
    })
    cmp = best_fixed.merge(best_align, on=["alpha", "regime"], how="outer")
    cmp["gap"] = cmp["alignmo_mean"] - cmp["best_fixed_mean"]
    cmp["ratio_gap_over_fixed_std"] = cmp["gap"] / cmp["best_fixed_std"].replace(0, np.nan)
    # Do no harm: AlignMO within 1 seed-std of the best fixed handler.
    cmp["do_no_harm"] = cmp["gap"] <= cmp["best_fixed_std"]
    # Do good: AlignMO strictly better than best fixed (negative gap beyond std).
    cmp["do_good"] = cmp["gap"] < -cmp["best_fixed_std"]
    cmp = cmp.sort_values(["regime", "alpha"]).reset_index(drop=True)
    cmp_csv = eval_dir / "per_cell_alignmo_vs_best.csv"
    cmp.to_csv(cmp_csv, index=False)
    print(f"[agg] per-cell comparison -> {cmp_csv}")

    modes = mode_trace_from_iters(eval_dir)
    modes_csv = eval_dir / "mode_trace.csv"
    modes.to_csv(modes_csv, index=False)
    print(f"[agg] mode_trace -> {modes_csv}")

    # Acceptance memo.
    n_cells = len(cmp)
    no_harm = int(cmp["do_no_harm"].sum())
    good = int(cmp["do_good"].sum())

    # Average-rank side-criterion (TODO 5.4 alt).
    rank_rows = []
    all_methods = fixed_methods + alignmo_methods
    for (alpha, regime), cell_all in grand.groupby(["alpha", "regime"]):
        per_method_min = []
        for method in all_methods:
            mdf = cell_all[cell_all["method"] == method]
            if mdf.empty:
                continue
            per_lam = mdf.groupby("lambda")[METRIC].mean()
            if per_lam.empty:
                continue
            per_method_min.append({"method": method, "mean": float(per_lam.min())})
        pm = pd.DataFrame(per_method_min).sort_values("mean").reset_index(drop=True)
        pm["rank"] = pm["mean"].rank(method="min")
        for _, row in pm.iterrows():
            rank_rows.append({"alpha": alpha, "regime": regime,
                              "method": row["method"], "rank": row["rank"]})
    rank_df = pd.DataFrame(rank_rows)
    avg_rank = rank_df.groupby("method")["rank"].mean().sort_values().reset_index()
    avg_rank.to_csv(eval_dir / "avg_rank.csv", index=False)
    alignmo_rank = float(avg_rank[avg_rank["method"] == "alignmo"]["rank"].iloc[0])
    best_avg_method = str(avg_rank.iloc[0]["method"])
    lowest_rank = alignmo_rank == float(avg_rank["rank"].min())

    lines: List[str] = []
    lines.append("# AlignMO Phase 2 acceptance memo")
    lines.append("")
    lines.append(f"**Cells:** {n_cells}  ")
    lines.append(f"**Do-no-harm cells (AlignMO ≤ best fixed + 1 seed-std):** {no_harm}/{n_cells}")
    lines.append(f"**Do-good cells (AlignMO strictly < best fixed − 1 seed-std):** {good}/{n_cells}")
    lines.append(f"**Lowest avg rank overall:** {best_avg_method} "
                 f"(AlignMO avg rank = {alignmo_rank:.2f})")
    lines.append("")
    lines.append("**Phase 2 acceptance criteria (Section 5.4):**")
    lines.append(f"- Do-no-harm on every cell: **{'PASS' if no_harm == n_cells else 'FAIL'}**")
    lines.append(f"- Do-good on ≥ 2 cells **OR** lowest avg rank: "
                 f"**{'PASS' if (good >= 2 or lowest_rank) else 'FAIL'}**")
    lines.append(f"- Mode-trace diversity (≥ 2 modes used anywhere): "
                 f"**{'PASS' if ((modes.drop(columns=['alpha','regime','n_logged_iters']) > 0.05).sum(axis=1).max() >= 2) else 'FAIL'}**")
    lines.append("")
    lines.append("## Per-cell table (AlignMO vs best fixed handler)")
    lines.append("")
    lines.append("| regime | alpha | best_fixed (λ) | fixed mean | fixed std | AlignMO (λ) | AlignMO mean | gap | gap/std | do-no-harm | do-good |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, r in cmp.iterrows():
        lines.append(
            f"| {r['regime']} | {r['alpha']} | "
            f"{r['best_fixed']} ({r['best_fixed_lambda']:.2f}) | "
            f"{r['best_fixed_mean']:.4f} | {r['best_fixed_std']:.4f} | "
            f"AlignMO ({r['alignmo_lambda']:.2f}) | {r['alignmo_mean']:.4f} | "
            f"{r['gap']:+.4f} | "
            f"{(r['ratio_gap_over_fixed_std'] if pd.notna(r['ratio_gap_over_fixed_std']) else float('nan')):+.2f} | "
            f"{'✓' if r['do_no_harm'] else '✗'} | "
            f"{'✓' if r['do_good'] else '✗'} |"
        )
    lines.append("")
    lines.append("## AlignMO mode fractions (per cell, averaged across seeds & λ)")
    lines.append("")
    lines.append("| regime | alpha | scalarized | projected | anchored | anchored_projected |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for _, r in modes.iterrows():
        lines.append(
            f"| {r['regime']} | {r['alpha']} | "
            f"{r['frac_scalarized']:.2f} | {r['frac_projected']:.2f} | "
            f"{r['frac_anchored']:.2f} | {r['frac_anchored_projected']:.2f} |"
        )
    lines.append("")
    lines.append("## Average rank (lower is better)")
    lines.append("")
    lines.append("| method | avg_rank |")
    lines.append("| --- | --- |")
    for _, r in avg_rank.iterrows():
        lines.append(f"| {r['method']} | {r['rank']:.2f} |")
    (eval_dir / "acceptance.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[agg] acceptance memo -> {eval_dir / 'acceptance.md'}")


if __name__ == "__main__":
    main()
