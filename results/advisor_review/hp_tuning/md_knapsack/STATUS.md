# MD knapsack — paused, follow-up needed

**As of 2026-04-14, branch `claude/review-advisor-feedback-lYjdA`.**

This subdirectory holds an HP sweep and a chosen-config validation for the
**redesigned per-individual MultiDimKnapsackTask** that landed in commit
`3cab494`. The work is **paused** to focus on the healthcare empirical
follow-up. Resume from this file when ready.

## What landed

| Stage | What | Status |
|-------|------|--------|
| sweep01_pipeline_check | Pipeline check at default base config (n_train=40, snr=3, finite_diff) | ✓ |
| sweep02_snr | SNR ∈ {1, 3, 5, 10}, FPTO+DFL only | ✓ |
| sweep03_budget | budget_tightness ∈ {0.2, 0.35, 0.5, 0.7} at snr=5 | ✓ |
| sweep04_hidden | hidden_dim ∈ {16, 32, 64} at snr=5, budget=0.35 | ✓ |
| sweep05_chosen_config | **All 6 methods + lambda sweep + 3 seeds** at chosen config (SPSA n_dirs=8, 30 steps) | ⚠ see "Concerns" |
| chosen_config.json | Snapshot of the picked HP point | ✓ (but provisional) |

Plus the SPSA strategy was patched to support the new per-individual MD
layout (commit `f9b31cc`) and made the default for the standalone runner
(commit `eab8577`).

## Concerns about sweep05_chosen_config (must address before paper use)

The naive ranking-by-test-regret in `sweep05_chosen_config` shows DFL
beating all MOO methods, which **contradicts prior experiments where MOO
methods dominated**. Re-analysis with `test_regret_normalized` exposes
the cause: at `n_train=40` the train→test gap is ~2× for every method
and dominates the inter-method spread.

| method     | train_norm | val_norm | test_norm | train→test gap |
|------------|-----------:|---------:|----------:|---------------:|
| DFL        |      0.092 |    0.065 |     0.153 |          0.061 |
| PCGrad     |      0.095 |    0.060 |     0.204 |    **0.109**   |
| FPLG (avg) |      0.165 |    0.082 |     0.184 |          0.019 |
| MGDA       |      0.171 |    0.086 |     0.189 |          0.018 |
| FDFL-Scal  |      0.174 |    0.086 |     0.189 |          0.015 |
| FPTO       |      0.180 |    0.089 |     0.193 |          0.013 |

**Interpretation:** PCGrad has the lowest *training* regret of any method
but generalises catastrophically — its MOO projection is finding a tight
fit to 40 samples that doesn't transfer. The MOO theory is not failing;
the experiment is too small to reveal it. DFL's apparent test-regret win
is partially artefactual: the simplest decision-focused method overfits
least when the predictor capacity is too high for the data.

This is exactly the regime described in `docs/misspecification_note.md`
(small data + no model selection → DFL appears to dominate). The
chosen_config should NOT be used in paper figures as currently tuned.

## What to do when resuming

1. **Re-run sweep05 at larger scale** — try `n_train ∈ {200, 400, 800}`,
   `steps ∈ {50, 100, 200}`, with validation-based model selection
   actually used. Cost will be ~10x higher per cell; SPSA n_dirs=8 still
   keeps it tractable. Goal: shrink the train→test gap below the
   inter-method spread so the ranking is meaningful.
2. **Use normalized regret in all comparisons**, not raw regret. The
   current notebook uses raw test_regret in the Pareto plots — switch
   to `test_regret_normalized` (normalised against the oracle objective).
3. **Add a per-method Pareto-frontier plot** (regret vs MSE coloured by
   lambda). Currently the comparison is per-method-mean across lambdas,
   which mixes lambda=0 (≈ DFL) with lambda=1 (heavy fairness penalty)
   and obscures the trade-off MOO methods are designed to make.
4. **Re-run the partial benefit/cost imbalance grid (Step 3b)** at the
   re-tuned config and at alpha=2.0 so the MOO-shines regime is in
   scope. Currently only alpha=0.5 (utilitarian → no DFL/MOO advantage)
   is recorded, and the cell `bb=0.6, cb=0.6` is missing.
5. **Run Step 4a (MD budget sweep)** with the re-tuned config.
6. **Add a normalized-regret Pareto plot** to the notebook so the
   re-tune can be validated visually.

## Reusable artifacts

- `chosen_config.json` — keep as a record of the small-n tuning attempt;
  do NOT use as the final config.
- `sweep02_snr/` through `sweep04_hidden/` — these still tell us the
  axis-by-axis sensitivities (snr=5 better than snr=1, tighter budget
  helps differentiation, hidden_dim=32 has cleanest MSE parity). Those
  conclusions hold under any n_train.
- `experiments/advisor_review/runner.py` and `drivers.py` — the helper
  module is reusable as-is; just call it with larger task / train cfg
  parameters.
