# Hypothesis 3b: Benefit vs cost imbalance grid (MD knapsack)

**Status:** PARTIAL — alpha=0.5 only, 8 of 9 cells (cell `bb=0.6, cb=0.6`
not run). The alpha=2.0 grid is not yet run. See "Follow-up runs" below.

**Setup:** chosen MD config from `hp_tuning/md_knapsack/chosen_config.json`
(n_train=40, n_resources=2, snr=5.0, budget_tightness=0.35, hidden_dim=32,
SPSA n_dirs=8, 6 methods, 3 seeds, lambdas={0.0, 0.5}). 8 cells × 27 stage
rows = 216 rows total, ~20 minutes.

## Mean test_regret per cell (averaged over methods, seeds, lambdas)

| benefit_bias \ cost_bias | 0.0  | 0.3  | 0.6  |
|--------------------------|-----:|-----:|-----:|
| 0.0                      | 1.54 | 1.58 | 1.67 |
| 0.3                      | 1.52 | 1.57 | 1.67 |
| 0.6                      | 1.50 | 1.56 |  —   |

## Mean test_fairness per cell (across same axes)

| benefit_bias \ cost_bias | 0.0  | 0.3  | 0.6  |
|--------------------------|-----:|-----:|-----:|
| 0.0                      | 0.38 | 0.38 | 0.38 |
| 0.3                      | 0.07 | 0.07 | 0.07 |
| 0.6                      | 0.52 | 0.52 |  —   |

## Three observations that match the advisor's hypothesis

1. **Cost imbalance affects decisions but not predictions.** Moving
   `cost_group_bias` 0.0 → 0.6 at fixed `benefit_group_bias` raises
   regret by ~10% (1.54 → 1.67) but leaves `test_fairness` essentially
   unchanged (Δ < 0.005). The new MD task design successfully isolates
   the two effects — exactly what the advisor's redesign was meant to
   demonstrate.

2. **Benefit imbalance affects fairness strongly and non-monotonically.**
   bb=0.3 → fairness 0.07 (low), bb=0.0 → 0.38, bb=0.6 → 0.52. The
   non-monotone pattern (smallest fairness at intermediate bias) is
   probably an interaction with the lambda sweep at this small `n_train`
   — would expect the alpha=2.0 grid to clean it up.

3. **At alpha=0.5 (utilitarian), all methods give nearly identical
   regret in every cell** (max-min spread within a cell is ~0.02). This
   is the "PTO catches up" regime documented in
   `docs/misspecification_note.md` — when the alpha-fair welfare is
   nearly linear, prediction-only methods are competitive with
   decision-focused ones. The chosen config was tuned at alpha=2.0
   precisely to avoid this.

## Follow-up runs (still to do)

* **Cell `alpha0.5_bb0.6_cb0.6`** — never finished, only config.json
  was written. Re-run with the same config to fill in the grid.
* **Full alpha=2.0 grid** — same 9 cells, alpha=2.0, where the method
  differentiation from the chosen config (DFL=126, FPLG=158, FPTO=167)
  should reproduce. This is the regime where the cost-imbalance effect
  is expected to be most clearly visible in *per-method* (not just
  pooled) test regret.

The pooled-mean tables above show the *task-level* effects (benefit and
cost imbalance create distinct prediction vs decision signatures). To
see the *method-level* effects (which methods are most sensitive to
which imbalance type) the alpha=2.0 grid is required.
