"""Score method differentiation for an HP sweep stage.

For each sweep cell, compute three scalars that tell us at a glance whether
methods are separating in the right direction:

* ``regret_spread``  : range of mean(test_regret) across methods
* ``mse_spread``     : range of mean(test_pred_mse) across methods
* ``regret_rank_gap``: mean(FPTO test_regret) - mean(DFL test_regret)
                       (positive == "DFL beats FPTO on regret" — the
                        regime we want)
* ``parity_signal``  : 1 if both spreads are non-trivial AND
                       ``regret_rank_gap > 0`` else 0

These are deliberately crude — they exist so that HP tuning is one
``python -m experiments.advisor_review.diff_score <sweep_dir>`` away from
a numerical answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def score_csv(stage_csv: Path) -> dict:
    df = pd.read_csv(stage_csv)
    if df.empty:
        return {"file": str(stage_csv), "rows": 0}
    g = df.groupby("method")
    test_regret = g["test_regret"].mean()
    test_mse = g["pred_mse"].mean() if "pred_mse" in df else g["test_pred_mse"].mean()
    out = {
        "file": str(stage_csv.name),
        "rows": int(len(df)),
        "methods": ",".join(sorted(test_regret.index.astype(str))),
        "regret_min": float(test_regret.min()),
        "regret_max": float(test_regret.max()),
        "regret_spread": float(test_regret.max() - test_regret.min()),
        "mse_min": float(test_mse.min()),
        "mse_max": float(test_mse.max()),
        "mse_spread": float(test_mse.max() - test_mse.min()),
    }
    if "fpto" in test_regret.index and "dfl" in test_regret.index:
        out["regret_gap_fpto_minus_dfl"] = float(
            test_regret.loc["fpto"] - test_regret.loc["dfl"]
        )
    return out


def score_dir(parent: Path) -> pd.DataFrame:
    rows = []
    for sub in sorted(Path(parent).iterdir()):
        if not sub.is_dir():
            continue
        csv = sub / "stage_results.csv"
        if not csv.exists():
            continue
        rec = score_csv(csv)
        rec["subdir"] = sub.name
        rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m experiments.advisor_review.diff_score <sweep_dir>")
        sys.exit(1)
    parent = Path(sys.argv[1])
    df = score_dir(parent)
    if df.empty:
        print(f"(no stage_results.csv files found under {parent})")
    else:
        cols = ["subdir", "regret_min", "regret_max", "regret_spread",
                "mse_min", "mse_max", "mse_spread", "regret_gap_fpto_minus_dfl"]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].round(3).to_string(index=False))
