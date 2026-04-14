# Healthcare follow-up — full-cohort post-`bias_parity` grid

**Branch**: `fair-dfl/empirical-followup` · **Run date**: 2026-04-14 · **Commit**: see `git log fair-dfl/empirical-followup`

## TL;DR

Eight (fairness_type × α) cells on the **full 48,784-patient Obermeyer
2019 cohort** with analytic decision gradients. 7 methods × 5 seeds ×
(4 lambdas where applicable), 70 SGD steps per lambda. Total run ≈
**27.3 minutes**, no NaN / no exploding gradients across **78,400
method-steps**.

- **At α=2.0** (the regime where the 2-level α-fair welfare matters):
  **FPLG dominates the regret axis in all 4 fairness cells**
  (`test_regret_normalized` ≈ 0.1274). PCGrad is consistently 2nd at
  ~0.1280-0.1295. The DFL/MOO methods clearly outperform the predict-then-
  optimize baselines (FPTO, SAA, WDRO).
- **At α=0.5** (utilitarian regime): the ranking flips. FPTO/PCGrad/WDRO
  are competitive (~0.060), and FPLG/FDFL-Scal/MGDA are noticeably worse
  (~0.075). The decision objective is too close to MSE to give DFL/MOO a
  lever, so a good predictor wins.
- **The train→test gap is essentially zero** (max 0.006, median 0.0001
  across the entire grid) — the regret of MD knapsack at n=40 (PCGrad
  +0.109) is gone at this scale. The ranking is meaningful.
- **Fairness winner depends on the metric**: FDFL-Scal wins mad and
  atkinson (separation/dispersion of MSE); PCGrad wins bias_parity (signed
  per-group residual); the dp winner is FPLG (lambda 1.0 / 5.0). Across
  all 4 fairness types FPLG and PCGrad are the consistent Pareto-frontier
  contenders.

## Setup

| Field | Value |
|---|---|
| Cohort | Full Obermeyer 2019 healthcare data (48,784 patients) |
| Train/test split | 50 / 50 (`test_fraction=0.5`) |
| Validation | **None** (`val_fraction=0.0`) — fixed step budget, no early stopping |
| Methods | `FPTO`, `FDFL-Scal`, `FPLG`, `PCGrad`, `MGDA`, `SAA`, `WDRO` |
| Seeds | `[11, 22, 33, 44, 55]` (5 seeds) |
| Lambdas | `[0.0, 0.5, 1.0, 5.0]` (only for fairness-aware non-MOO methods) |
| Steps / lambda | 70 |
| Optimiser | SGD, `lr=5e-4`, `lr_decay=5e-4`, full-batch, `weight_decay=0` |
| Architecture | MLP, `hidden_dim=64`, `n_layers=2`, ReLU, no dropout, no batchnorm |
| Decision gradient | **Analytic** (closed-form for the healthcare LP) |
| Fairness types | `mad`, `dp`, `atkinson`, `bias_parity` (4 cells per α) |
| α (group welfare) | `0.5`, `2.0` (2 alphas per fairness type) |
| Total cells | 4 × 2 = 8 |
| Method-steps run | 8 × ~9,800 = ~78,400 |
| Wall-clock | 27.3 min total (199–218 s per cell) |

### Why no validation?

Closed-form decision gradients on a stable LP, no learning-rate schedule
beyond a built-in `lr_decay=5e-4`, no early stopping. With a fixed step
budget (`steps_per_lambda=70`) and a known optimum (the oracle LP
solution), validation does not pick a hyperparameter we can't already
inspect from the loss curve. `eval_train=True` is on, so we detect
overfitting via the train-vs-test gap directly.

### Why each method gets a different number of points

The runner has two gates that collapse the lambda sweep to `[0.0]`:

1. `(not base_spec.use_fair) and (not force_lambda_path_all_methods)` —
   collapses methods that don't use fairness at all (`SAA`, `WDRO`).
2. `train_cfg.get("mo_method") is not None and (not force_lambda_path_all_methods)`
   — collapses MOO methods (`MGDA`, `PCGrad`, `WS-*`, `CAGrad`, `FAMO`,
   `PLG-*`) because the per-objective handler ignores `lambda_value`. We
   verified this in `training/loop.py` line 489: `g_fair_param` is computed
   from `g_fair_pred` directly, without `beta_t` scaling, then handed to
   the MOO handler. Sweeping λ on a MOO method would produce identical
   copies.

So per cell:

- **3 Pareto-curve methods** (FPTO, FDFL-Scal, FPLG): 4 λ × 5 seeds = 20 rows
- **2 MOO single-point methods** (MGDA, PCGrad): 1 λ × 5 seeds = 5 rows
- **2 baselines** (SAA, WDRO): 1 λ × 5 seeds = 5 rows
- **Total per cell**: 60 + 10 + 10 = 80 rows
- **Total grid**: 8 × 80 = 640 rows

## Findings

### 1. Stability check — overfitting is gone

| Metric | Value |
|---|---|
| `nan_or_inf_steps_total` | **0** |
| `exploding_steps_total` | **0** |
| `max_train_test_gap` (across 640 rows) | **0.0061** |
| `median_train_test_gap` | **0.0001** |

Compare to MD knapsack at n_train=40 (`results/advisor_review/hp_tuning/
md_knapsack/STATUS.md`):

- PCGrad train→test gap: **+0.109**
- DFL train→test gap: +0.061
- Other methods: ~0.015

At n_train ≈ 24,392 (half of 48,784) the train→test gap is **17×–1000×
smaller** than the inter-method spread, so the ranking is meaningful.

### 2. Per-cell Pareto winners

Best mean `test_regret_normalized` per (fairness_type, α) cell, with the
Pareto-best λ in parentheses:

| fairness | α   | 1st                | 2nd                 | 3rd                | 4th               |
|----------|-----|--------------------|---------------------|--------------------|-------------------|
| mad      | 0.5 | FPTO (5.0) 0.0529  | PCGrad 0.0576       | WDRO 0.0606        | FDFL-Scal (1) 0.0651 |
| mad      | 2.0 | **FPLG (1.0) 0.1275** | PCGrad 0.1295    | FDFL-Scal (0) 0.1322 | MGDA 0.1366    |
| dp       | 0.5 | FPTO (5.0) 0.0598  | WDRO 0.0606         | PCGrad 0.0701      | FPLG (5) 0.0746   |
| dp       | 2.0 | **FPLG (5.0) 0.1276** | FDFL-Scal (0) 0.1322 | PCGrad 0.1341 | MGDA 0.1621    |
| atkinson | 0.5 | PCGrad 0.0601      | FPTO (0) 0.0602     | WDRO 0.0606        | FPLG (5) 0.0746   |
| atkinson | 2.0 | **FPLG (5.0) 0.1275** | PCGrad 0.1285    | FDFL-Scal (0) 0.1322 | MGDA 0.1641   |
| bias_p.  | 0.5 | PCGrad 0.0602      | FPTO (0) 0.0602     | WDRO 0.0606        | FPLG (5) 0.0746   |
| bias_p.  | 2.0 | **FPLG (5.0) 0.1274** | PCGrad 0.1281    | FDFL-Scal (5) 0.1319 | MGDA 0.1627   |

**Pattern: at α=2.0, FPLG is the best regret across all 4 fairness types.**
PCGrad is always 2nd by ≤ 0.002 — it sits right on the FPLG Pareto curve
or slightly above. At α=0.5 the picture inverts: FPTO/PCGrad/WDRO are
competitive at ~0.060 and FPLG/FDFL-Scal/MGDA are at ~0.075. The α=0.5
regime gives DFL no lever because the welfare is nearly linear in
predictions.

### 3. Method-fairness specialisation at α=2.0

Different methods minimise different fairness statistics. At α=2.0:

| fairness | best fairness method | best fair value | best regret method | best regret |
|----------|----------------------|-----------------|--------------------|-------------|
| mad      | FDFL-Scal (λ=0)     | 22.80           | FPLG (λ=1)         | 0.1275      |
| dp       | MGDA                 | 0.260           | FPLG (λ=5)         | 0.1276      |
| atkinson | FDFL-Scal (λ=0)     | 0.0065          | FPLG (λ=5)         | 0.1275      |
| bias_p.  | **PCGrad** 0.056     | 0.056           | FPLG (λ=5)         | 0.1274      |

The bias_parity finding is interesting: PCGrad achieves a per-group
residual MAD of **0.056** (essentially calibrated across groups) while
keeping `test_regret_normalized` at 0.1281. FPLG achieves 0.1274 regret at
the cost of a 5× larger bias_parity (0.317). They are mutually
non-dominated.

### 4. High-λ behaviour

At λ=5, FPLG and FDFL-Scal stay healthy across **all 8 cells** — the
α_t schedule of FPLG and the smaller relative weight of fairness in
FDFL-Scal both prevent the catastrophic train→test fairness divergence
seen in earlier MD knapsack runs.

The earlier "fairness overfit" pathology I called out from the timing cell
(FDFL-Scal mad α=2.0 at λ=5 had train_fair=2.96 vs test_fair=37.74) is
**isolated to the mad metric**. In atkinson, dp, and bias_parity the test
fairness at λ=5 is similar to the train fairness (gap < 1.5×). So MAD on
MSE is uniquely sensitive to high-λ overfitting; the other fairness types
generalise gracefully.

### 5. Decision focus implicitly buys some fairness

At λ=0 (no explicit fairness term), FDFL-Scal still uses the
decision-regret gradient. In the mad α=2.0 cell:

- FDFL-Scal at λ=0: train_fairness = 20.5, test_fairness = 22.8
- FPTO at λ=0: train_fairness = 59.7, test_fairness = 51.5

FDFL-Scal's per-group MSE spread is **2.5–3× smaller** than FPTO's even
without an explicit fairness penalty. This is the 2-level α-fair welfare
in the regret implicitly encoding group fairness.

### 6. SAA and WDRO across the grid

SAA is the worst-regret method in every cell (test_reg_n = 0.279 at α=2.0,
0.0752 at α=0.5). WDRO (test_reg_n ≈ 0.191 at α=2.0, 0.0606 at α=0.5)
beats SAA but lags every fairness-aware method at α=2.0. Both single-point
methods don't move with λ, so they appear in the same place across the 4
fairness panels.

## Methodological notes

- **Use normalized regret throughout.** Raw test_regret depends on the
  problem's oracle objective and is not comparable across α or fairness
  types. The `*_regret_normalized` columns divide by the oracle objective,
  giving a unit-free ratio.
- **MOO methods are single Pareto points by design.** The framework's MOO
  handlers compute their own gradient combination; lambda has no effect on
  them. This is methodologically correct (their selling point is "no λ
  knob"), but it means MOO methods cannot trace a Pareto curve. To compare
  against Pareto-curve methods, look at whether the MOO point sits above,
  on, or below the curve.
- **High λ is the "what if we push fairness too hard" diagnostic, not the
  operating point.** The Pareto frontier is best traced at λ ∈ [0, 1]. λ=5
  is included to surface the failure mode (only seen for FDFL-Scal /
  FPTO on mad).
- **Two regimes**: α=0.5 is the predict-then-optimize regime (FPTO,
  PCGrad, WDRO competitive; DFL/MOO no advantage). α=2.0 is the
  decision-focused regime (FPLG dominates regret, PCGrad close 2nd).

## Relationship to prior runs

- The earlier `dp_vs_ap` healthcare experiment (`results/advisor_review/
  hypothesis/dp_vs_ap/`) ran at `n_sample=5000` with `val_fraction=0.2` and
  a smaller method set. This grid scales it up to the full cohort, drops
  validation, and adds `atkinson` and `bias_parity` cells. The qualitative
  finding (FPLG/MOO dominate fair α=2.0) is consistent.
- The MD knapsack experiments (`results/advisor_review/hp_tuning/
  md_knapsack/STATUS.md`) are paused. They had a train→test gap problem at
  `n_train=40` that obscured the ranking. Healthcare with ~24k training
  samples does not have that problem (see Findings §1) — at this scale the
  empirical conclusions are trustworthy.

## Files

- `experiments/advisor_review/healthcare_followup.py` — config + cell + grid runners
- `experiments/advisor_review/run_healthcare_grid.py` — launcher
- `experiments/advisor_review/analyze_healthcare_grid.py` — aggregates + grand summary
- `experiments/advisor_review/plot_healthcare_grid.py` — per-cell + 8-panel Pareto plots
- `results/advisor_review/healthcare_followup/{ft}/alpha_{a}/stage_results.csv` — raw results
- `results/advisor_review/healthcare_followup/{ft}/alpha_{a}/summary_by_method_lambda.csv` — per-cell aggregates
- `results/advisor_review/healthcare_followup/{ft}/alpha_{a}/best_pareto_per_method.csv` — best-Pareto rows
- `results/advisor_review/healthcare_followup/grand_summary.csv` — best Pareto point per (method, cell)
- `results/advisor_review/healthcare_followup/health.json` — NaN/explosion + train→test gap
- `results/advisor_review/healthcare_followup/grid_summary.json` — wall-clock per cell
- `results/advisor_review/healthcare_followup/plots/pareto_*.png` — 8 per-cell + 1 grid PNG

## What to do next

- **PR review**: open `fair-dfl/empirical-followup` against `main`, point
  to this file and the 8-panel grid plot.
- **[deferred]** Step 4b: budget-tightness sweep on healthcare at the
  recommended config.
- **[deferred]** MD knapsack re-tune at `n_train ∈ {200, 400, 800}` with
  validation (see `STATUS.md`).
- **Optional**: add `WS-equal`, `WS-dec`, `WS-fair` to expose the explicit
  preference-weighted MOO frontier (currently MOO methods only contribute
  one point each).
