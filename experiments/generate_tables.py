#!/usr/bin/env python3
"""Generate publication-ready LaTeX tables from final experiment results.

Produces:
  Table 1: Healthcare results (rows = method x lambda, panels for alpha)
  Table 2: Knapsack results (rows = method x lambda, panels for alpha x unfairness)
  Table 3: Cross-experiment method ranking summary

The new method grid uses:
  FPTO with lambda in {0, 0.5, 1, 5}  — lambda=0 is PTO
  FDFL-Scal with lambda in {0, 0.5, 1, 5}  — lambda=0 is DFL
  SAA, WDRO  — no lambda
  FDFL-PCGrad, FDFL-MGDA, FDFL-CAGrad  — no lambda

Usage:
  python experiments/generate_tables.py
  python experiments/generate_tables.py --healthcare-dir results/final/healthcare
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# ======================================================================
# Row keys: (method_label, lambda) combinations in display order
# ======================================================================
# Each entry: (method_label, lambda, display_name, group_tag_for_midrule)
ROW_SPEC = [
    ("FPTO",        0.0, r"FPTO ($\lambda=0$) $\equiv$ PTO",   "two_stage"),
    ("FPTO",        0.5, r"FPTO ($\lambda=0.5$)",               "two_stage"),
    ("FPTO",        1.0, r"FPTO ($\lambda=1$)",                 "two_stage"),
    ("FPTO",        5.0, r"FPTO ($\lambda=5$)",                 "two_stage"),
    ("SAA",         0.0, "SAA",                                  "data_driven"),
    ("WDRO",        0.0, "WDRO",                                 "data_driven"),
    ("FDFL-Scal",   0.0, r"FDFL-Scal ($\lambda=0$) $\equiv$ DFL", "scalarized"),
    ("FDFL-Scal",   0.5, r"FDFL-Scal ($\lambda=0.5$)",          "scalarized"),
    ("FDFL-Scal",   1.0, r"FDFL-Scal ($\lambda=1$)",            "scalarized"),
    ("FDFL-Scal",   5.0, r"FDFL-Scal ($\lambda=5$)",            "scalarized"),
    ("FDFL-PCGrad", 0.0, "FDFL-PCGrad",                         "moo"),
    ("FDFL-MGDA",   0.0, "FDFL-MGDA",                           "moo"),
    ("FDFL-CAGrad", 0.0, "FDFL-CAGrad",                         "moo"),
]

# Metrics
METRICS = {
    "test_regret_normalized": ("Norm. Regret", "lower"),
    "test_fairness":          ("Pred. Fair. Viol.", "lower"),
    "test_pred_mse":          ("Pred. MSE", "lower"),
}


def _load_results(results_dir: str) -> pd.DataFrame:
    p = Path(results_dir)
    agg = p / "stage_results_all.csv"
    if agg.exists():
        return pd.read_csv(agg)
    csvs = sorted(p.rglob("stage_results.csv"))
    if not csvs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)


def _fmt(mean: float, std: float, bold: bool = False, fmt: str = ".4f") -> str:
    if np.isnan(mean):
        return "--"
    s = f"{mean:{fmt}} {{\\scriptsize$\\pm${std:{fmt}}}}"
    return f"\\textbf{{{s}}}" if bold else s


def _find_best(agg: pd.DataFrame, metric: str, direction: str) -> set:
    """Return set of (method_label, lambda) tuples that achieve the best mean."""
    col = f"{metric}_mean"
    if col not in agg.columns or agg[col].isna().all():
        return set()
    best = agg[col].min() if direction == "lower" else agg[col].max()
    hits = agg[agg[col] == best]
    return set(zip(hits["method_label"], hits["lam"]))


def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = list(METRICS.keys())
    rows = []
    for keys, grp in df.groupby(group_cols, sort=False):
        if isinstance(keys, (str, float, int)):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for col in metric_cols:
            vals = grp[col].dropna()
            row[f"{col}_mean"] = vals.mean() if len(vals) > 0 else np.nan
            row[f"{col}_std"] = vals.std() if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


# ======================================================================
# Table 1: Healthcare
# ======================================================================

def generate_healthcare_table(df: pd.DataFrame, output_dir: Path) -> str:
    if df.empty:
        return "% No healthcare results"

    # Ensure lambda column
    if "lambda" not in df.columns:
        df["lambda"] = 0.0
    df["lam"] = df["lambda"].fillna(0.0)

    alphas = sorted(df["alpha_fair"].unique())
    n_met = len(METRICS)
    col_spec = "l" + "|".join(["c" * n_met] * len(alphas))

    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Healthcare resource allocation. Mean $\pm$ std over 5 seeds, "
        r"best hidden dim selected per (method, $\alpha$). Best in \textbf{bold}.}",
        r"\label{tab:healthcare}", r"\small",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    # Headers
    h1 = [""]
    for alpha in alphas:
        h1.append(r"\multicolumn{" + str(n_met) + r"}{c}{$\alpha=" + f"{alpha}" + r"$}")
    lines.append(" & ".join(h1) + r" \\")

    h2 = ["Method"]
    for _ in alphas:
        for _, (name, _) in METRICS.items():
            h2.append(name)
    lines.append(" & ".join(h2) + r" \\")
    lines.append(r"\midrule")

    # If hidden_dim exists, pick best per (method, alpha, lambda) by min regret
    if "hidden_dim" in df.columns:
        best_hd = (df.groupby(["method_label", "alpha_fair", "lam", "hidden_dim"])
                   ["test_regret_normalized"].mean().reset_index())
        idx = best_hd.groupby(["method_label", "alpha_fair", "lam"])["test_regret_normalized"].idxmin()
        best_hd = best_hd.loc[idx, ["method_label", "alpha_fair", "lam", "hidden_dim"]]
        df = df.merge(best_hd, on=["method_label", "alpha_fair", "lam", "hidden_dim"])

    prev_group = None
    for method_label, lam, display, group in ROW_SPEC:
        # Add midrule between groups
        if prev_group is not None and group != prev_group:
            lines.append(r"\midrule")
        prev_group = group

        parts = [display]
        for alpha in alphas:
            sub = df[(df["method_label"] == method_label) & (df["alpha_fair"] == alpha)
                     & (np.isclose(df["lam"], lam))]
            agg = _aggregate(sub, ["method_label", "lam"])

            # Compute best across all methods for this alpha
            alpha_df = df[df["alpha_fair"] == alpha]
            alpha_agg = _aggregate(alpha_df, ["method_label", "lam"])
            best_set = {m: _find_best(alpha_agg, m, d) for m, (_, d) in METRICS.items()}

            for metric_key, (_, direction) in METRICS.items():
                if agg.empty:
                    parts.append("--")
                    continue
                mean = agg.iloc[0].get(f"{metric_key}_mean", np.nan)
                std = agg.iloc[0].get(f"{metric_key}_std", 0.0)
                is_bold = (method_label, lam) in best_set.get(metric_key, set())
                parts.append(_fmt(mean, std, bold=is_bold))

        lines.append(" & ".join(parts) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    table = "\n".join(lines)
    out = output_dir / "table1_healthcare.tex"
    out.write_text(table, encoding="utf-8")
    print(f"  Table 1 -> {out}")
    return table


# ======================================================================
# Table 2: Knapsack
# ======================================================================

def generate_knapsack_table(df: pd.DataFrame, output_dir: Path) -> str:
    if df.empty:
        return "% No knapsack results"

    if "lambda" not in df.columns:
        df["lambda"] = 0.0
    df["lam"] = df["lambda"].fillna(0.0)

    alphas = sorted(df["alpha_fair"].unique())
    uf_levels = ["mild", "medium", "high"]
    if "unfairness_level" in df.columns:
        uf_levels = [u for u in uf_levels if u in df["unfairness_level"].unique()]

    n_met = len(METRICS)
    lines = []

    for alpha in alphas:
        col_spec = "l" + "|".join(["c" * n_met] * len(uf_levels))
        lines += [
            r"\begin{table}[htbp]", r"\centering",
            r"\caption{Knapsack, $\alpha=" + f"{alpha}" + r"$. "
            r"Mean $\pm$ std over 5 seeds. Best in \textbf{bold}.}",
            r"\label{tab:knapsack_a" + f"{alpha}".replace(".", "") + "}",
            r"\small",
            r"\begin{tabular}{" + col_spec + "}",
            r"\toprule",
        ]

        h1 = [""]
        for uf in uf_levels:
            h1.append(r"\multicolumn{" + str(n_met) + r"}{c}{" + uf.capitalize() + "}")
        lines.append(" & ".join(h1) + r" \\")

        h2 = ["Method"]
        for _ in uf_levels:
            for _, (name, _) in METRICS.items():
                h2.append(name)
        lines.append(" & ".join(h2) + r" \\")
        lines.append(r"\midrule")

        alpha_df = df[df["alpha_fair"] == alpha]

        prev_group = None
        for method_label, lam, display, group in ROW_SPEC:
            if prev_group is not None and group != prev_group:
                lines.append(r"\midrule")
            prev_group = group

            parts = [display]
            for uf in uf_levels:
                sub = alpha_df[
                    (alpha_df["method_label"] == method_label)
                    & (alpha_df["unfairness_level"] == uf)
                    & (np.isclose(alpha_df["lam"], lam))
                ]
                agg = _aggregate(sub, ["method_label", "lam"])

                panel_df = alpha_df[alpha_df["unfairness_level"] == uf]
                panel_agg = _aggregate(panel_df, ["method_label", "lam"])
                best_set = {m: _find_best(panel_agg, m, d) for m, (_, d) in METRICS.items()}

                for metric_key, (_, direction) in METRICS.items():
                    if agg.empty:
                        parts.append("--")
                        continue
                    mean = agg.iloc[0].get(f"{metric_key}_mean", np.nan)
                    std = agg.iloc[0].get(f"{metric_key}_std", 0.0)
                    is_bold = (method_label, lam) in best_set.get(metric_key, set())
                    parts.append(_fmt(mean, std, bold=is_bold))

            lines.append(" & ".join(parts) + r" \\")

        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    table = "\n".join(lines)
    out = output_dir / "table2_knapsack.tex"
    out.write_text(table, encoding="utf-8")
    print(f"  Table 2 -> {out}")
    return table


# ======================================================================
# Table 3: Summary ranking
# ======================================================================

def generate_summary_table(hc_df: pd.DataFrame, kn_df: pd.DataFrame,
                           output_dir: Path) -> str:
    summary_metrics = ["test_regret_normalized", "test_fairness", "test_pred_mse"]
    metric_short = {"test_regret_normalized": "Regret", "test_fairness": "Fair.",
                    "test_pred_mse": "MSE"}

    def _ranks(df, condition_cols):
        if df.empty:
            return {}
        if "lambda" not in df.columns:
            df["lambda"] = 0.0
        df["lam"] = df["lambda"].fillna(0.0)
        df["row_key"] = df["method_label"] + "_lam" + df["lam"].astype(str)
        result = {}
        for keys, grp in df.groupby(condition_cols):
            means = grp.groupby("row_key")[summary_metrics].mean()
            for metric in summary_metrics:
                ranked = means[metric].rank(ascending=True)
                for key, rank in ranked.items():
                    result.setdefault(key, {}).setdefault(metric, []).append(rank)
        return {k: {m: np.mean(v) for m, v in mv.items()} for k, mv in result.items()}

    hc_cond = ["alpha_fair"]
    if "hidden_dim" in hc_df.columns:
        hc_cond.append("hidden_dim")
    hc_ranks = _ranks(hc_df, hc_cond)

    kn_cond = ["alpha_fair"]
    if "unfairness_level" in kn_df.columns:
        kn_cond.append("unfairness_level")
    kn_ranks = _ranks(kn_df, kn_cond)

    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Average rank across configurations (1 = best).}",
        r"\label{tab:summary}", r"\small",
        r"\begin{tabular}{l|ccc|ccc}", r"\toprule",
        r" & \multicolumn{3}{c|}{Healthcare} & \multicolumn{3}{c}{Knapsack} \\",
    ]
    h = ["Method"] + [metric_short[m] for m in summary_metrics] * 2
    lines.append(" & ".join(h) + r" \\")
    lines.append(r"\midrule")

    prev_group = None
    for method_label, lam, display, group in ROW_SPEC:
        if prev_group is not None and group != prev_group:
            lines.append(r"\midrule")
        prev_group = group
        row_key = f"{method_label}_lam{lam}"
        parts = [display]
        for metric in summary_metrics:
            val = hc_ranks.get(row_key, {}).get(metric, np.nan)
            parts.append(f"{val:.1f}" if not np.isnan(val) else "--")
        for metric in summary_metrics:
            val = kn_ranks.get(row_key, {}).get(metric, np.nan)
            parts.append(f"{val:.1f}" if not np.isnan(val) else "--")
        lines.append(" & ".join(parts) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table = "\n".join(lines)
    out = output_dir / "table3_summary.tex"
    out.write_text(table, encoding="utf-8")
    print(f"  Table 3 -> {out}")
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--healthcare-dir", default=str(REPO_ROOT / "results" / "final" / "healthcare"))
    parser.add_argument("--knapsack-dir", default=str(REPO_ROOT / "results" / "final" / "knapsack"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "final" / "tables"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hc_df = _load_results(args.healthcare_dir)
    kn_df = _load_results(args.knapsack_dir)
    print(f"Healthcare: {len(hc_df)} rows | Knapsack: {len(kn_df)} rows")

    generate_healthcare_table(hc_df, output_dir)
    generate_knapsack_table(kn_df, output_dir)
    generate_summary_table(hc_df, kn_df, output_dir)
    print(f"\nAll tables in {output_dir}/")


if __name__ == "__main__":
    main()
