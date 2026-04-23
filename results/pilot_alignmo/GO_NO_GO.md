# GO / NO-GO memo — AlignMO Phase 1 pilot

**Verdict:** GREEN
**Distinct cell-winners (out of 8 cells):** 4
**Recommendation:** GO — Phase 2 as planned.

## Per-cell table

| regime | alpha | winner | best_lambda | mean | std | runner_up | gap | gap/seed_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analytic | 0.5 | fpto | 2.00 | 0.0501 | 0.0013 | fplg | 0.0097 | 7.30 |
| analytic | 1.5 | pcgrad | 0.00 | 0.0140 | 0.0000 | fpto | 0.0020 | 51.99 |
| analytic | 2.0 | fplg | 2.00 | 0.1525 | 0.0070 | pcgrad | 0.0000 | 0.00 |
| analytic | 3.0 | fdfl-0.5 | 2.00 | 2.5161 | 0.4649 | fdfl-scal | 0.0272 | 0.06 |
| spsa | 0.5 | fpto | 2.00 | 0.0501 | 0.0013 | pcgrad | 0.0046 | 3.50 |
| spsa | 1.5 | fpto | 2.00 | 0.0160 | 0.0004 | pcgrad | 0.0007 | 2.03 |
| spsa | 2.0 | fpto | 2.00 | 0.1748 | 0.0084 | pcgrad | 0.0123 | 1.46 |
| spsa | 3.0 | fpto | 2.00 | 2.9215 | 0.7834 | pcgrad | 0.6402 | 0.82 |

## Decision rule (ALIGNMO_PLAN.md Section 4.4)

- >= 4 distinct winners: GREEN (GO).
- 2-3 distinct winners: YELLOW (CAUTIOUS GO).
- 1 distinct winner: RED (NO GO, reposition).

## Notes

- Cells with `gap_seed_std_ratio` < 1 are within seed noise; 
  escalate to 5 seeds per Section 4.7 before calling a tight win.
- A secondary diagnostic check lives in `diagnostic_profile.csv`.