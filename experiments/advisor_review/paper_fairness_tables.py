"""Generate per-fairness-type tables in the user's 2-panel IJOC style.

Produces one LaTeX file per fairness type (dp, atkinson, bias_parity)
matching the exact format of ``E:/IJOC-paper/Tables/table_healthcare.tex``
after the user's edits: 2 panels (alpha=0.5 and alpha=2.0), each a
`l ccc` tabular with Norm. Regret / Fair. Viol. / Pred. MSE columns,
values in math mode as ``$mu \\pm sd$``, winners in ``$\\mathbf{...}$``.

Writes to ``results/advisor_review/healthcare_followup_v2/paper/ijoc_tables/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.advisor_review.analyze_healthcare_v2 import load_variant  # noqa: E402
from experiments.advisor_review.healthcare_followup_v2 import HC_V2_DIR  # noqa: E402


# Row order matches the user's MAD table.
ROW_ORDER: list[tuple[str, str, float | None]] = [
    # (method_key_in_csv, display_label, lambda_or_None_for_single_row)
    ("fpto", r"PTO ($\equiv$ FPTO, $\lambda\!=\!0$)", 0.0),
    ("fpto", r"FPTO ($\lambda\!=\!0.5$)", 0.5),
    ("fpto", r"FPTO ($\lambda\!=\!1$)", 1.0),
    ("fpto", r"FPTO ($\lambda\!=\!2$)", 2.0),
    ("__MIDRULE__", "", None),
    ("saa", r"SAA", 0.0),
    ("wdro", r"WDRO", 0.0),
    ("__MIDRULE__", "", None),
    ("fdfl-scal", r"DFL ($\equiv$ FDFL-Scal, $\lambda\!=\!0$)", 0.0),
    ("fdfl-scal", r"FDFL-Scal ($\lambda\!=\!0.5$)", 0.5),
    ("fdfl-scal", r"FDFL-Scal ($\lambda\!=\!1$)", 1.0),
    ("fdfl-scal", r"FDFL-Scal ($\lambda\!=\!2$)", 2.0),
    ("__MIDRULE__", "", None),
    ("fplg", r"FPLG ($\lambda\!=\!0$)", 0.0),
    ("fplg", r"FPLG ($\lambda\!=\!0.5$)", 0.5),
    ("fplg", r"FPLG ($\lambda\!=\!1$)", 1.0),
    ("fplg", r"FPLG ($\lambda\!=\!2$)", 2.0),
    ("__MIDRULE__", "", None),
    ("pcgrad", r"FDFL-PCGrad", 0.0),
    ("mgda", r"FDFL-MGDA", 0.0),
]


def _fmt(mu: float, sd: float, best_mu: float, ndp: int = 4) -> str:
    """Format a value as $mu \\pm sd$, bolded if it ties the best."""
    if ndp == 4:
        s = f"{mu:.4f} \\pm {sd:.4f}"
    elif ndp == 3:
        s = f"{mu:.3f} \\pm {sd:.3f}"
    else:
        s = f"{mu:.{ndp}f} \\pm {sd:.{ndp}f}"
    if abs(mu - best_mu) < 1e-12:
        return f"$\\mathbf{{{s}}}$"
    return f"${s}$"


def _cell_stats(df_cell: pd.DataFrame) -> pd.DataFrame:
    """Mean and std across seeds per (method, lambda) for one cell."""
    return (
        df_cell.groupby(["method", "lambda"])
        .agg(
            reg_mean=("test_regret_normalized", "mean"),
            reg_std=("test_regret_normalized", "std"),
            fair_mean=("test_fairness", "mean"),
            fair_std=("test_fairness", "std"),
            mse_mean=("test_pred_mse", "mean"),
            mse_std=("test_pred_mse", "std"),
            n=("split_seed", "count"),
        )
        .reset_index()
    )


def _get_row(stats: pd.DataFrame, method: str, lam: float) -> pd.Series | None:
    match = stats[(stats["method"] == method) & (stats["lambda"] == lam)]
    if match.empty:
        return None
    return match.iloc[0]


def _panel(
    stats: pd.DataFrame,
    reg_decimals: int = 4,
    fair_decimals: int | None = None,
    mse_decimals: int = 3,
) -> str:
    """Render one panel as `\\begin{tabular}...\\end{tabular}`."""
    # First pass: gather each row's raw values so we can identify column winners.
    rows_data: list[tuple[str, float, float, float, float, float, float]] = []
    for method, label, lam in ROW_ORDER:
        if method == "__MIDRULE__":
            rows_data.append(("__MIDRULE__", 0, 0, 0, 0, 0, 0))
            continue
        r = _get_row(stats, method, lam)
        if r is None:
            continue
        rows_data.append(
            (
                label,
                float(r["reg_mean"]),
                float(r["reg_std"]),
                float(r["fair_mean"]),
                float(r["fair_std"]),
                float(r["mse_mean"]),
                float(r["mse_std"]),
            )
        )

    # Find column winners (minimum mean) across data rows only.
    data_rows = [r for r in rows_data if r[0] != "__MIDRULE__"]
    best_reg = min(r[1] for r in data_rows)
    best_fair = min(r[3] for r in data_rows)
    best_mse = min(r[5] for r in data_rows)

    # Decide fairness display decimals based on the scale.
    if fair_decimals is None:
        max_fair = max(abs(r[3]) for r in data_rows)
        if max_fair >= 10:
            fair_decimals = 3
        elif max_fair >= 1:
            fair_decimals = 3
        else:
            fair_decimals = 4

    lines = [
        r"\begin{tabular}{l ccc}",
        r"\toprule",
        r"Method & Norm.\ Regret & Fair.\ Viol. & Pred.\ MSE \\",
        r"\midrule",
    ]
    for row in rows_data:
        if row[0] == "__MIDRULE__":
            lines.append(r"\midrule")
            continue
        label, rm, rs, fm, fs, mm, ms = row
        reg_str = _fmt(rm, rs, best_reg, ndp=reg_decimals)
        fair_str = _fmt(fm, fs, best_fair, ndp=fair_decimals)
        mse_str = _fmt(mm, ms, best_mse, ndp=mse_decimals)
        lines.append(
            f"{label:<44} & {reg_str} & {fair_str} & {mse_str} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def build_table(
    fairness_type: str,
    fairness_display: str,
    label: str,
    fair_decimals: int | None = None,
) -> str:
    df = load_variant("a")
    cell = df[df["fairness_type"] == fairness_type]
    if cell.empty:
        raise RuntimeError(f"No data for fairness_type={fairness_type}")

    cell_a = cell[cell["alpha_fair"] == 0.5]
    cell_b = cell[cell["alpha_fair"] == 2.0]
    stats_a = _cell_stats(cell_a)
    stats_b = _cell_stats(cell_b)

    panel_a = _panel(stats_a, fair_decimals=fair_decimals)
    panel_b = _panel(stats_b, fair_decimals=fair_decimals)

    n_seeds = int(cell["split_seed"].nunique())

    caption = (
        f"Healthcare resource allocation results with {fairness_display} as "
        f"the fairness violation. Mean $\\pm$ std.\\ over {n_seeds} seeds, each "
        f"using its own train/test partition of the 48{{,}}784-patient cohort. "
        f"Best value per column per panel is \\textbf{{bold}}; all metrics are "
        f"lower-is-better. FPTO, SAA, and WDRO train the same predictive model "
        f"regardless of $\\alpha$; only their normalized regret varies with "
        f"$\\alpha$. DFL-family methods use the decision gradient, so all three "
        f"metrics depend on $\\alpha$."
    )

    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{5pt}",
            "",
            r"\smallskip",
            r"\textit{Panel A: $\alpha = 0.5$}",
            "",
            r"\smallskip",
            panel_a,
            "",
            r"\bigskip",
            r"\textit{Panel B: $\alpha = 2$}",
            "",
            r"\smallskip",
            panel_b,
            "",
            r"\end{table}",
            "",
        ]
    )


def main() -> None:
    out_dir = HC_V2_DIR / "paper" / "ijoc_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        (
            "dp",
            r"MAD of per-group mean predictions (demographic parity)",
            "tab:healthcare-dp",
            "table_healthcare_dp.tex",
        ),
        (
            "atkinson",
            r"Atkinson index over per-group MSE (inequality aversion $\epsilon=0.5$)",
            "tab:healthcare-atk",
            "table_healthcare_atk.tex",
        ),
        (
            "bias_parity",
            r"MAD of per-group mean residuals (bias parity / calibration first moment)",
            "tab:healthcare-bp",
            "table_healthcare_bp.tex",
        ),
    ]

    for ft, display, label, filename in specs:
        tex = build_table(ft, display, label)
        (out_dir / filename).write_text(tex, encoding="utf-8")
        print(f"[saved] {out_dir / filename}")


if __name__ == "__main__":
    main()
