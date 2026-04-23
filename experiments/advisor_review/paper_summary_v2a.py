"""Generate paper-ready summary tables for the healthcare v2 Variant A experiment.

Produces the following in ``results/advisor_review/healthcare_followup_v2/paper/``:

- ``summary_for_paper.md`` — markdown writeup with setup, tables, observations
- ``table1_regret.md`` and ``table1_regret.tex`` — test_regret_normalized per
  (method, cell), mean +/- std across seeds
- ``table2_fairness.md`` and ``table2_fairness.tex`` — test_fairness per
  (method, cell), mean +/- std across seeds
- ``table3_best_lambda.md`` and ``table3_best_lambda.tex`` — Pareto-best
  lambda per (method, cell)
- ``summary_regret_by_method.csv`` — per-method summary (mean, std, best/worst cell)

Usage:
    python -m experiments.advisor_review.paper_summary_v2a
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.advisor_review.analyze_healthcare_v2 import (  # noqa: E402
    load_variant,
)
from experiments.advisor_review.healthcare_followup_v2 import (  # noqa: E402
    HC_V2_ALPHAS,
    HC_V2_DIR,
    HC_V2_FAIRNESS_TYPES,
)

METHOD_ORDER = ["fpto", "fdfl-scal", "fplg", "pcgrad", "mgda", "saa", "wdro"]
METHOD_DISPLAY = {
    "fpto": "FPTO",
    "fdfl-scal": "FDFL-Scal",
    "fplg": "FPLG",
    "pcgrad": "PCGrad",
    "mgda": "MGDA",
    "saa": "SAA",
    "wdro": "WDRO",
}
FAIRNESS_DISPLAY = {
    "mad": "MAD",
    "dp": "DP",
    "atkinson": "Atkinson",
    "bias_parity": "BiasParity",
}


def _best_lambda_row(
    df_cell: pd.DataFrame,
    metric_col: str = "test_regret_normalized",
) -> pd.DataFrame:
    """For each method in the cell, pick the lambda whose mean metric_col
    is lowest across seeds. Return aggregate (mean, std) at that lambda.
    """
    rows = []
    for method in df_cell["method"].unique():
        method_df = df_cell[df_cell["method"] == method]
        per_lam = (
            method_df.groupby("lambda")
            .agg(
                test_reg_n_mean=("test_regret_normalized", "mean"),
                test_reg_n_std=("test_regret_normalized", "std"),
                test_fair_mean=("test_fairness", "mean"),
                test_fair_std=("test_fairness", "std"),
                train_reg_n_mean=("train_regret_normalized", "mean"),
                train_fair_mean=("train_fairness", "mean"),
                n=("test_regret_normalized", "count"),
            )
        )
        if per_lam.empty:
            continue
        # Pareto-best = lambda with lowest mean test_regret_normalized
        best_lam = float(per_lam["test_reg_n_mean"].idxmin())
        r = per_lam.loc[best_lam]
        rows.append(
            {
                "method": method,
                "best_lambda": best_lam,
                "test_reg_n_mean": float(r["test_reg_n_mean"]),
                "test_reg_n_std": float(r["test_reg_n_std"]),
                "test_fair_mean": float(r["test_fair_mean"]),
                "test_fair_std": float(r["test_fair_std"]),
                "train_reg_n_mean": float(r["train_reg_n_mean"]),
                "train_fair_mean": float(r["train_fair_mean"]),
                "n_seeds": int(r["n"]),
            }
        )
    return pd.DataFrame(rows)


def build_grand_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, fairness, alpha) with mean/std regret and fairness."""
    rows = []
    for (ft, a), cell in df.groupby(["fairness_type", "alpha_fair"]):
        cell_best = _best_lambda_row(cell)
        cell_best["fairness_type"] = ft
        cell_best["alpha"] = a
        rows.append(cell_best)
    return pd.concat(rows, ignore_index=True)


def pivot_regret(gt: pd.DataFrame) -> pd.DataFrame:
    """Methods x (fairness, alpha), cells = "mean +/- std" strings."""
    rows = []
    for m in METHOD_ORDER:
        msub = gt[gt["method"] == m]
        if msub.empty:
            continue
        row = {"method": METHOD_DISPLAY[m]}
        for ft in HC_V2_FAIRNESS_TYPES:
            for a in HC_V2_ALPHAS:
                cell = msub[(msub["fairness_type"] == ft) & (msub["alpha"] == a)]
                if cell.empty:
                    row[(FAIRNESS_DISPLAY[ft], f"α={a}")] = "—"
                else:
                    mu = float(cell["test_reg_n_mean"].iloc[0])
                    sd = float(cell["test_reg_n_std"].iloc[0])
                    row[(FAIRNESS_DISPLAY[ft], f"α={a}")] = f"{mu:.4f} ± {sd:.4f}"
        rows.append(row)
    # Flatten the MultiIndex columns
    df = pd.DataFrame(rows)
    df.columns = (
        ["method"]
        + [f"{ft}, {al}" for ft in ["MAD", "DP", "Atkinson", "BiasParity"] for al in ["α=0.5", "α=2.0"]]
    )
    return df


def pivot_fairness(gt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in METHOD_ORDER:
        msub = gt[gt["method"] == m]
        if msub.empty:
            continue
        row = {"method": METHOD_DISPLAY[m]}
        for ft in HC_V2_FAIRNESS_TYPES:
            for a in HC_V2_ALPHAS:
                cell = msub[(msub["fairness_type"] == ft) & (msub["alpha"] == a)]
                if cell.empty:
                    row[f"{FAIRNESS_DISPLAY[ft]}, α={a}"] = "—"
                else:
                    mu = float(cell["test_fair_mean"].iloc[0])
                    sd = float(cell["test_fair_std"].iloc[0])
                    if abs(mu) >= 1.0 or abs(sd) >= 1.0:
                        row[f"{FAIRNESS_DISPLAY[ft]}, α={a}"] = f"{mu:.3f} ± {sd:.3f}"
                    else:
                        row[f"{FAIRNESS_DISPLAY[ft]}, α={a}"] = f"{mu:.4f} ± {sd:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def pivot_best_lambda(gt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in METHOD_ORDER:
        msub = gt[gt["method"] == m]
        if msub.empty:
            continue
        row = {"method": METHOD_DISPLAY[m]}
        for ft in HC_V2_FAIRNESS_TYPES:
            for a in HC_V2_ALPHAS:
                cell = msub[(msub["fairness_type"] == ft) & (msub["alpha"] == a)]
                if cell.empty:
                    row[f"{FAIRNESS_DISPLAY[ft]}, α={a}"] = "—"
                else:
                    row[f"{FAIRNESS_DISPLAY[ft]}, α={a}"] = f"{float(cell['best_lambda'].iloc[0]):.1f}"
        rows.append(row)
    return pd.DataFrame(rows)


def df_to_markdown_pipe(df: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown pipe table (no tabulate dependency)."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, sep] + rows)


def df_to_latex_booktabs(
    df: pd.DataFrame,
    caption: str,
    label: str,
    bold_winners: list[tuple[str, str]] | None = None,
) -> str:
    """Render a Pandas DataFrame to a booktabs LaTeX table with multicolumn headers.

    Assumes columns are like 'MAD, α=0.5' / 'DP, α=2.0' etc. (8 columns).
    Optionally bolds the winning cell per column, given a list of (method, col).
    """
    method_col = df.columns[0]
    data_cols = list(df.columns[1:])

    # Group columns by prefix (before the comma)
    group_of_col = [c.split(",")[0].strip() for c in data_cols]
    groups: list[tuple[str, int]] = []
    cur = None
    count = 0
    for g in group_of_col:
        if g == cur:
            count += 1
        else:
            if cur is not None:
                groups.append((cur, count))
            cur = g
            count = 1
    if cur is not None:
        groups.append((cur, count))

    n_cols = len(data_cols)
    col_spec = "l" + "c" * n_cols

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Group header row
    header_row = ["Method"]
    for grp_name, grp_span in groups:
        if grp_span == 1:
            header_row.append(grp_name)
        else:
            header_row.append(rf"\multicolumn{{{grp_span}}}{{c}}{{{grp_name}}}")
    lines.append(" & ".join(header_row) + r" \\")

    # Subheader row (alpha)
    sub_row = [""]
    for c in data_cols:
        parts = c.split(",")
        alpha_label = parts[1].strip() if len(parts) > 1 else ""
        # Escape the greek alpha for LaTeX
        alpha_label = alpha_label.replace("α", r"$\alpha$")
        sub_row.append(alpha_label)
    lines.append(" & ".join(sub_row) + r" \\")
    lines.append(r"\midrule")

    def _latex_cell(val: str) -> str:
        # Convert the ± char and unicode α to LaTeX math.
        return str(val).replace("±", r"$\pm$").replace("α", r"$\alpha$")

    # Data rows
    for _, row in df.iterrows():
        cells = [str(row[method_col])]
        for c in data_cols:
            val = _latex_cell(row[c])
            if bold_winners and (row[method_col], c) in bold_winners:
                cells.append(rf"\textbf{{{val}}}")
            else:
                cells.append(val)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def find_winners_by_col(df: pd.DataFrame, lower_is_better: bool = True) -> list[tuple[str, str]]:
    """For each data column, find the row (method) with the lowest mean value.

    Returns a list of (method_display, col_name) pairs to bold.
    Parses 'mean ± std' strings.
    """
    method_col = df.columns[0]
    data_cols = list(df.columns[1:])
    winners = []
    for c in data_cols:
        best_val = np.inf if lower_is_better else -np.inf
        best_method = None
        for _, row in df.iterrows():
            s = str(row[c])
            if s == "—":
                continue
            try:
                mu = float(s.split("±")[0].strip())
            except ValueError:
                continue
            if (lower_is_better and mu < best_val) or (not lower_is_better and mu > best_val):
                best_val = mu
                best_method = row[method_col]
        if best_method is not None:
            winners.append((best_method, c))
    return winners


def main() -> None:
    df = load_variant("a")
    n_seeds = df["split_seed"].nunique()
    print(f"loaded Variant A: {len(df)} rows, {n_seeds} seeds")

    out_dir = HC_V2_DIR / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grand table: (method, fairness, alpha) x stats
    gt = build_grand_table(df)
    gt.to_csv(out_dir / "grand_table_variant_a.csv", index=False)

    # Table 1: regret
    t1 = pivot_regret(gt)
    t1.to_csv(out_dir / "table1_regret.csv", index=False)

    w1 = find_winners_by_col(t1, lower_is_better=True)
    md1 = t1.pipe(df_to_markdown_pipe)
    with open(out_dir / "table1_regret.md", "w", encoding="utf-8") as f:
        f.write(f"## Table 1: test_regret_normalized, Variant A, {n_seeds} seeds\n\n")
        f.write(md1 + "\n")
    tex1 = df_to_latex_booktabs(
        t1,
        caption=(
            f"Test regret (normalised by oracle objective) on the healthcare task "
            f"across 4 fairness types and 2 welfare curvatures $\\alpha$. "
            f"Mean $\\pm$ std across {n_seeds} seeds, each with its own train/test "
            f"split. Lower is better. Bold marks the best method per column."
        ),
        label="tab:healthcare_regret",
        bold_winners=w1,
    )
    with open(out_dir / "table1_regret.tex", "w", encoding="utf-8") as f:
        f.write(tex1 + "\n")

    # Table 2: fairness
    t2 = pivot_fairness(gt)
    t2.to_csv(out_dir / "table2_fairness.csv", index=False)

    w2 = find_winners_by_col(t2, lower_is_better=True)
    md2 = t2.pipe(df_to_markdown_pipe)
    with open(out_dir / "table2_fairness.md", "w", encoding="utf-8") as f:
        f.write(f"## Table 2: test_fairness (raw units per fairness type), Variant A, {n_seeds} seeds\n\n")
        f.write(md2 + "\n")
    tex2 = df_to_latex_booktabs(
        t2,
        caption=(
            f"Test fairness at the Pareto-best $\\lambda$ (the lambda minimising "
            f"test regret). Units differ per fairness type (MAD on per-group MSE, "
            f"DP and BiasParity on benefit scale, Atkinson as a unit-free ratio). "
            f"Mean $\\pm$ std across {n_seeds} seeds. Lower is fairer."
        ),
        label="tab:healthcare_fairness",
        bold_winners=w2,
    )
    with open(out_dir / "table2_fairness.tex", "w", encoding="utf-8") as f:
        f.write(tex2 + "\n")

    # Table 3: best lambda
    t3 = pivot_best_lambda(gt)
    t3.to_csv(out_dir / "table3_best_lambda.csv", index=False)
    md3 = t3.pipe(df_to_markdown_pipe)
    with open(out_dir / "table3_best_lambda.md", "w", encoding="utf-8") as f:
        f.write(f"## Table 3: Pareto-best lambda per method, Variant A, {n_seeds} seeds\n\n")
        f.write(md3 + "\n")
    tex3 = df_to_latex_booktabs(
        t3,
        caption=(
            "Pareto-best $\\lambda$ per method per cell, where Pareto-best means "
            "the $\\lambda$ value in $\\{0, 0.5, 1, 2\\}$ that minimises test "
            "regret. Methods whose training is $\\lambda$-independent (MOO: MGDA, "
            "PCGrad; baselines: SAA, WDRO) always show $\\lambda = 0.0$."
        ),
        label="tab:healthcare_best_lambda",
    )
    with open(out_dir / "table3_best_lambda.tex", "w", encoding="utf-8") as f:
        f.write(tex3 + "\n")

    # Per-method mean summary
    per_method = (
        gt.groupby("method")
        .agg(
            mean_regret=("test_reg_n_mean", "mean"),
            min_regret=("test_reg_n_mean", "min"),
            max_regret=("test_reg_n_mean", "max"),
            mean_fair=("test_fair_mean", "mean"),
        )
        .round(4)
        .sort_values("mean_regret")
    )
    per_method.to_csv(out_dir / "per_method_summary.csv")

    # Rewrite alpha for stdout (Windows cp1252 can't encode the greek letter)
    def _ascii(s: str) -> str:
        return s.replace("α", "a").replace("±", "+/-")

    print("\n=== Table 1: test_regret_normalized ===")
    print(_ascii(t1.to_string(index=False)))
    print("\n=== Table 2: test_fairness ===")
    print(_ascii(t2.to_string(index=False)))
    print("\n=== Table 3: Pareto-best lambda ===")
    print(_ascii(t3.to_string(index=False)))
    print("\n=== Per-method summary ===")
    print(per_method.to_string())
    print(f"\n[outputs] {out_dir}")


if __name__ == "__main__":
    main()
