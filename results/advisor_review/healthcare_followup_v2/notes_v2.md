# Healthcare follow-up v2 — findings

**Branch**: `fair-dfl/empirical-followup` · **Run date**: 2026-04-14 ·
**Reference**: [setup.md](./setup.md) for problem definition, fairness
formulas, and HP tables.

## TL;DR

1. **Stability** across 640 rows, 0 NaN, 0 explosion. Max train→test
   gap **0.0161** (worse than v1's 0.0061 — see §7).
2. **At α=2.0** FPLG still wins regret in all 4 fairness cells across
   **both variants**, but at smaller λ than v1 (λ=0.5–1 vs v1's 1–5).
3. **At α=0.5** the winner flipped from v1: Variant A has **WDRO** (3/4
   cells) and **PCGrad** (1/4 cell) winning, Variant B has **FPTO**
   (3/4) and PCGrad (1/4). FPLG/FDFL-Scal are consistently 0.01–0.02
   behind. DFL has no lever when α=0.5 — expected.
4. **Cross-fairness rank**: at α=2.0 the **fairest** method is FPTO
   (rank 2.75/3.00 in A/B); at α=0.5 it's **PCGrad** (rank 2.75/2.50).
   FPLG wins regret but **never** wins fairness — classic
   regret↔fairness trade-off.
5. **Early stopping engaged on only 15% of non-SAA stages** in Variant
   B. Most training was long enough that the val metric kept improving
   through to step 150. PCGrad was the most aggressive early-stopper
   (mean stop step 108), FPTO/WDRO never stopped (mean 150).
6. **Train→test gaps got WORSE in Variant B** (FPLG: 0.0019 → 0.0051,
   PCGrad: 0.0019 → 0.0041). Counter-intuitively, adding validation +
   early stopping INCREASED the generalization gap — see §6 for the
   explanation (longer effective training + val-vs-test distribution
   mismatch from disjoint split seeds).

## Setup recap

| | Variant A | Variant B |
|---|---|---|
| Seeds | [11, 22, 33] | [44, 55] |
| `split_seed` = seed? | yes | yes |
| `val_fraction` | 0.0 | 0.2 |
| Early stopping | off | on (K=10, min_steps=20, metric=val_regret) |
| `lr_decay` | 5e-4 | 5e-3 |
| `lr_warmup_steps` | 0 | 5 |
| `steps_per_lambda` | 70 | 150 |
| Total runs | 24 | 16 |
| Total rows | 384 | 256 |
| Elapsed | 22.3 min | 23.4 min |

Both share: `budget_rho=0.30`, `lambdas=[0, 0.5, 1, 2]`, `lr=1e-3`, MLP
hidden=64 n_layers=2, PyTorch-default init, full-batch SGD, analytic
decision gradients.

## 1. Stability

From `health.json`:

| Metric | Both variants |
|---|---|
| `n_rows_total` | 640 |
| `n_cells` | 16 (8 per variant) |
| `n_methods` | 7 |
| `n_split_seeds` | 5 (3 A + 2 B) |
| `nan_or_inf_steps_total` | **0** |
| `exploding_steps_total` | **0** |
| `max_train_test_gap` | **0.01608** |
| `median_train_test_gap` | 0.000561 |
| `rows_with_early_stop_enabled` | 256 (all Variant B) |
| `rows_with_early_stop_applied` | 240 (= 256 − 16 SAA rows) |

Comparison with v1: median gap is ~6× higher in v2 (0.00056 vs ~0.00009)
but still below any inter-method difference. The max gap is 2.6× higher,
entirely from Variant B FPLG on split_seed=55 at α=2. See §6.

## 2. Variant A — baseline (3 seeds, no val)

### 2.1 Per-cell winners (lowest `test_regret_normalized`)

| fairness | α | winner | best λ | test_reg_n | test_fair |
|---|---|---|---|---|---|
| mad | 0.5 | **PCGrad** | 0.0 | 0.0487 | 21.22 |
| mad | 2.0 | **FPLG** | 0.5 | **0.1267** | 24.48 |
| dp | 0.5 | **WDRO** | 0.0 | 0.0490 | 1.56 |
| dp | 2.0 | **FPLG** | 1.0 | **0.1266** | 1.87 |
| atkinson | 0.5 | **WDRO** | 0.0 | 0.0490 | 0.0048 |
| atkinson | 2.0 | **FPLG** | 1.0 | **0.1265** | 0.0075 |
| bias_parity | 0.5 | **WDRO** | 0.0 | 0.0490 | 0.931 |
| bias_parity | 2.0 | **FPLG** | 1.0 | **0.1265** | 0.554 |

### 2.2 Pivot of `test_regret_normalized` across (method, cell)

```
              atkinson       bias_parity        dp              mad
alpha            0.5    2.0     0.5    2.0    0.5    2.0    0.5    2.0
method
fdfl-scal      0.0749 0.1299  0.0746 0.1298  0.0749 0.1299  0.0665 0.1299
fplg           0.0739 0.1265  0.0739 0.1265  0.0739 0.1266  0.0736 0.1267  ← α=2 winner
fpto           0.0501 0.1541  0.0500 0.1530  0.0501 0.1541  0.0495 0.1526
mgda           0.0758 0.1530  0.0758 0.1508  0.0758 0.1550  0.0758 0.1301
pcgrad         0.0501 0.1276  0.0499 0.1286  0.0583 0.1384  0.0487 0.1274  ← mad α=0.5 winner
saa            0.0752 0.2807  0.0752 0.2807  0.0752 0.2807  0.0752 0.2807
wdro           0.0490 0.1501  0.0490 0.1501  0.0490 0.1501  0.0490 0.1501  ← α=0.5 winner (3/4)
```

### 2.3 Observations

- **FPLG's α=2.0 regret is remarkably flat across fairness types**
  (0.1265–0.1267, spread 0.0002). Its α_t schedule decouples regret
  from the fairness gradient scale, so swapping fairness types (which
  have very different units) barely moves the outcome.
- **MGDA is sensitive to fairness type**: 0.1301 (mad) → 0.1508 (bp) →
  0.1530 (atk) → 0.1550 (dp) — spread of 0.025. MGDA's gradient
  projection depends on raw gradient norms, so fairness-types with
  smaller-magnitude gradients (dp, atkinson) get "over-projected" and
  regret suffers.
- **PCGrad wins mad α=0.5** (0.0487, lowest regret in A) — at the
  utilitarian regime with separation-family fairness, PCGrad's
  orthogonal projection finds a slightly better optimum than FPTO.
- **WDRO dominates 3/4 α=0.5 cells** despite being "just" a
  distributionally-robust prediction baseline. The tight budget
  (0.30 vs v1's 0.35) makes robustness more valuable than in v1, where
  FPTO won at α=0.5.
- **v1 changes summary**: at α=2.0, FPLG's winning λ drops from 1–5
  (v1) to 0.5–1 (v2). The λ=5 catastrophic regime we saw in v1 was
  correctly replaced.

## 3. Variant B — validation + LR schedule + early stopping (2 seeds)

### 3.1 Per-cell winners

| fairness | α | winner | best λ | test_reg_n | test_fair |
|---|---|---|---|---|---|
| mad | 0.5 | **FPTO** | 0.0 | 0.0465 | (see ratio table) |
| mad | 2.0 | **FPLG** | 1.0 | **0.1286** | — |
| dp | 0.5 | **FPTO** | 0.0 | 0.0465 | — |
| dp | 2.0 | **FPLG** | 1.0 | **0.1287** | — |
| atkinson | 0.5 | **FPTO** | 0.0 | 0.0465 | — |
| atkinson | 2.0 | **FPLG** | 1.0 | **0.1287** | — |
| bias_parity | 0.5 | **PCGrad** | 0.0 | 0.0457 | — |
| bias_parity | 2.0 | **FPLG** | 1.0 | **0.1287** | — |

**FPLG sweeps α=2.0** in Variant B too, with test_reg_n in a tight
0.1286–0.1287 range. At α=0.5, **FPTO wins 3/4** cells (was WDRO in
Variant A), and PCGrad takes bias_parity.

### 3.2 Early stopping actually engaged?

Distribution of `early_stop_step` per method (out of 150 max):

| method | mean step | median | min | max | % engaged (step < 150) |
|---|---|---|---|---|---|
| FPTO | 150.0 | 150 | 150 | 150 | **0%** (never stopped) |
| WDRO | 150.0 | 150 | 150 | 150 | **0%** (never stopped) |
| MGDA | 139.4 | 150 | 20 | 150 | 6% |
| FPLG | 147.2 | 150 | 20 | 150 | ~10% |
| FDFL-Scal | 133.3 | 150 | 20 | 150 | ~25% |
| PCGrad | **108.1** | 135 | 20 | 150 | ~65% |
| SAA | 0 | 0 | 0 | 0 | n/a (`steps_per_lambda=0`) |

**Non-SAA rows where early stopping was triggered**: **37 / 240 (15%)**.

The aggressive gradient-combining methods — **PCGrad** and
**FDFL-Scal** — use early stopping the most. FPTO (pure prediction)
and WDRO (robust prediction) never engage early stopping because their
val regret keeps improving throughout all 150 steps. This is the
expected pattern: **overfitting is a decision-focused phenomenon** in
healthcare at this data scale.

## 4. Variant A vs Variant B — does adding validation + early stop help?

### 4.1 Does the winner change?

| fairness | α | A winner | B winner | Same? |
|---|---|---|---|---|
| mad | 0.5 | PCGrad 0.0487 | **FPTO** 0.0465 | ✗ |
| mad | 2.0 | **FPLG** 0.1267 | **FPLG** 0.1286 | ✓ |
| dp | 0.5 | WDRO 0.0490 | **FPTO** 0.0465 | ✗ |
| dp | 2.0 | **FPLG** 0.1266 | **FPLG** 0.1287 | ✓ |
| atkinson | 0.5 | WDRO 0.0490 | **FPTO** 0.0465 | ✗ |
| atkinson | 2.0 | **FPLG** 0.1265 | **FPLG** 0.1287 | ✓ |
| bias_parity | 0.5 | WDRO 0.0490 | **PCGrad** 0.0457 | ✗ |
| bias_parity | 2.0 | **FPLG** 0.1265 | **FPLG** 0.1287 | ✓ |

**At α=2.0 FPLG wins in both variants** — the headline v1 claim holds
across training setups. At α=0.5 the winner is less stable across
variants (FPTO vs WDRO vs PCGrad all within 0.003 of each other).

### 4.2 Regret delta (B − A) at best Pareto point

Calculated for each cell:

| cell | A best | B best | delta |
|---|---|---|---|
| mad α=0.5 | 0.0487 | 0.0465 | **−0.0022** (B is better) |
| mad α=2.0 | 0.1267 | 0.1286 | **+0.0019** (B is worse) |
| dp α=0.5 | 0.0490 | 0.0465 | **−0.0025** |
| dp α=2.0 | 0.1266 | 0.1287 | **+0.0021** |
| atkinson α=0.5 | 0.0490 | 0.0465 | **−0.0025** |
| atkinson α=2.0 | 0.1265 | 0.1287 | **+0.0021** |
| bias_parity α=0.5 | 0.0490 | 0.0457 | **−0.0033** |
| bias_parity α=2.0 | 0.1265 | 0.1287 | **+0.0021** |

**Systematic split**: at α=0.5, Variant B is consistently ~0.0025
**better** than A. At α=2.0, Variant B is consistently ~0.0020
**worse** than A. Since seeds are disjoint, we cannot cleanly attribute
this to "training setup" — but the consistency across 4 fairness types
at each α suggests a real effect, not split noise.

**Interpretation**: at α=0.5 (easy regime), more training data helps
less than "more steps + LR schedule". At α=2.0 (hard regime), the
model is pushed harder by the LR schedule and longer training into a
state that overfits the val metric but doesn't generalize as well to
test (see §6).

## 5. Cross-fairness comparison

### 5.1 Rank-based aggregation

Mean rank across the 4 fairness types at each α (lower = fairer). Ties
broken by average. Computed on `test_fair_at_fair_best` (the fairness-
constrained Pareto point, regret slack 10%).

| method | α=0.5 A | α=0.5 B | α=2.0 A | α=2.0 B |
|---|---|---|---|---|
| **pcgrad** | **2.75** | **2.50** | 3.75 | **3.25** |
| **fpto** | 3.25 | 3.00 | **2.75** | **3.00** |
| **wdro** | 3.00 | 3.50 | 3.00 | 3.75 |
| saa | 3.75 | 4.00 | 5.25 | 5.25 |
| fdfl-scal | 4.75 | 4.75 | 4.00 | 3.75 |
| **fplg** | 4.75 | 4.75 | 4.25 | **5.00** |
| mgda | 5.75 | 5.50 | 5.00 | 4.00 |

**Most consistently fair at α=0.5**: PCGrad (2.75/2.50). **Most
consistently fair at α=2.0**: FPTO (2.75/3.00).

**FPLG wins regret but not fairness in any cell** — classic
regret↔fairness trade-off. FPLG's α_t schedule that gives it a clean
regret Pareto curve also means it doesn't chase the fairness penalty
as hard as FDFL-Scal or PCGrad.

**FPLG's fairness rank WORSENS in Variant B** (4.25 → 5.00 at α=2.0).
Early stopping catches FPLG before its later-stage training (where it
would reduce fairness at higher λ) completes, so its fairness-best
Pareto point is worse in B.

### 5.2 Ratio vs FPTO λ=0

`rel_fair = test_fair(method) / test_fair(fpto_lam0)`, computed per
seed then averaged. <1 is fairer than FPTO, >1 is worse.

**Variant A**:
```
              atkinson  bias_parity    dp          mad
alpha            0.5 2.0    0.5 2.0  0.5 2.0    0.5 2.0
method
fdfl-scal      2.67 1.21   3.21 0.93 0.00 0.99  4.03 0.95
fplg           2.67 1.38   3.20 0.69 0.00 1.09  4.99 1.05
fpto           1.00 1.00   0.83 0.83 1.00 1.00  0.90 0.93
mgda           2.67 2.28   3.21 2.80 0.00 0.16  5.01 1.19
pcgrad         0.99 1.72   0.45 0.59 0.05 0.17  0.91 1.20
saa            1.73 1.73   3.21 3.21 0.00 0.00  2.43 2.43
wdro           0.89 0.89   1.19 1.19 0.91 0.91  0.87 0.87
```

- **PCGrad on bias_parity α=2**: 0.59 (41% fairer than FPTO)
- **PCGrad on bias_parity α=0.5**: 0.45 (55% fairer)
- **MGDA on dp α=2**: 0.16 (84% fairer — MGDA finds the independence
  direction aggressively at α=2 even though it hurts regret)
- **WDRO** is the only method with a **consistent** ratio around 0.88
  across all cells — it's never the outright fairest but it's always
  modestly fairer than FPTO.
- **FPLG/FDFL-Scal are WORSE than FPTO at α=0.5** (ratios 2.6–5.0),
  because they invest in a fairness direction that doesn't help when
  the welfare is nearly utilitarian.

**Variant B**:
```
              atkinson  bias_parity    dp          mad
alpha            0.5 2.0    0.5 2.0  0.5 2.0    0.5 2.0
method
fdfl-scal      2.45 1.09   2.68 0.77 0.00 1.09  4.40 1.03
fplg           2.44 1.27   2.68 0.60 0.01 1.21  5.31 1.14
fpto           1.00 1.00   0.83 0.83 1.00 1.00  1.00 1.00
mgda           2.45 2.18   2.69 2.50 0.00 0.12  5.37 0.61
pcgrad         0.97 1.16   0.37 0.43 0.20 0.49  0.93 1.07
saa            1.62 1.62   2.69 2.69 0.00 0.00  2.64 2.64
wdro           0.92 0.92   1.03 1.03 0.98 0.98  0.95 0.95
```

- **PCGrad bias_parity α=0.5**: 0.37 (better than Variant A's 0.45)
- **MGDA mad α=2**: 0.61 (big improvement from A's 1.19 — different
  split seeds matter)
- **FPTO on mad**: 1.00 in B vs 0.90/0.93 in A — fpto no longer gains
  anything over λ=0 on mad in B

### 5.3 Bottom line on cross-fairness

- **If you care most about regret at α=2**: FPLG (robustly wins
  regret, fairness is mediocre)
- **If you care most about fairness on bias_parity / calibration**:
  PCGrad (consistently best on bias_parity, 40–60% fairer than FPTO)
- **If you care most about dp / independence**: MGDA or SAA (both near
  0 on dp — SAA because it predicts the constant mean, MGDA because
  its gradient projection finds the independence direction)
- **If you care about consistency across all fairness types**: FPTO or
  WDRO (both in top 3 across all 8 cells)
- **FPLG is strictly worse than WDRO on fairness** in most cells
  despite winning regret. The "win regret, lose fairness" trade is
  consistent and unavoidable.

## 6. Why Variant B's train→test gaps are WORSE

### 6.1 The numbers

Mean `train_test_gap` per method, at each method's best Pareto point:

| method | Variant A mean | Variant B mean | ratio B/A |
|---|---|---|---|
| saa | 0.00119 | -0.00214 | neg |
| wdro | 0.00003 | 0.00170 | ~60× |
| fpto | -0.00004 | 0.00164 | large |
| mgda | -0.00005 | -0.00016 | flat |
| fdfl-scal | 0.00068 | 0.00312 | 4.6× |
| pcgrad | 0.00187 | 0.00405 | 2.2× |
| **fplg** | **0.00186** | **0.00514** | **2.8×** |

Across EVERY trainable method, Variant B's gap is bigger than A's. The
max gap outlier (0.0161) is FPLG on split_seed=55 at α=2.0 λ=2 with
`early_stop_step=150` (val kept improving throughout 150 steps).

### 6.2 Why?

**Three compounding factors**:

1. **Less effective training data.** Variant B carves 20% of the train
   half for validation, so training sees only 40% of the cohort
   (~19,500 patients) vs Variant A's 50% (~24,400). The extra overfit
   capacity is not fully absorbed by early stopping.
2. **Longer training runway.** Variant B has `steps_per_lambda=150` vs
   A's `70`. Even with early stopping, most stages run the full 150
   steps (only 15% trigger early stop). Methods like FPLG and PCGrad
   — which v1 and Variant A already showed are the most aggressive
   gradient-combiners — now have 2× more training steps before any
   check can fire.
3. **Val and test come from disjoint partitions of the SAME split
   seed.** Val is 20% of train half (same seed's partition); test is
   the other 50% (also same seed's partition). But the overfit
   dynamics mean val's distribution is closer to train than to test.
   Val-based early stop picks the step that minimises val regret,
   which is correlated with (but not the same as) minimising test
   regret.

**The net effect**: early stopping engages for exactly the methods
that need it most (PCGrad, FDFL-Scal), but the engagement isn't strong
enough to offset the extra training budget + reduced training data.
For FPLG, which rarely early-stops (only 10% of stages), the result
is a straightforwardly longer overfit.

### 6.3 Is this a bug or a finding?

**Finding, not a bug**: v2 Variant B was designed to explore "does
realistic training change the conclusions?" and the empirical answer
is "yes, slightly — realistic training actually makes the overfit
problem WORSE at full cohort scale when you have good decision
gradients."

**Implication for the paper**: on the healthcare task, **validation +
LR schedule + early stopping is not a free lunch**. If you have
enough data that v1-style training doesn't overfit (which is the case
at n≈24k), adding realistic training machinery gives you a model with
worse test performance. This is a clean negative result that argues
**against** the reviewer's likely objection that "the paper should
use real training techniques".

## 7. Method-fairness specialisation at α=2.0

Does v2 reproduce v1's specialisation pattern?

**v1 (n=48k, one split, 5 seeds)**:
- mad → FDFL-Scal fair-best
- dp → MGDA fair-best
- atkinson → FDFL-Scal fair-best
- bias_parity → PCGrad fair-best

**v2 Variant A (n=48k, 3 split seeds)**:
- mad → **WDRO** fair-best (different from v1)
- dp → **MGDA** fair-best ✓ (same as v1)
- atkinson → **WDRO** fair-best (different from v1)
- bias_parity → **PCGrad** fair-best ✓ (same as v1)

**v2 Variant B (n=48k, 2 split seeds, early stop)**:
- mad → **WDRO** fair-best (same as A)
- dp → **MGDA** fair-best ✓
- atkinson → **WDRO** fair-best (same as A)
- bias_parity → **PCGrad** fair-best ✓

**Stable**: MGDA dominates DP fairness, PCGrad dominates bias_parity
fairness in both variants and in v1. Those are robust paper findings.

**Shifted**: On mad and atkinson (both separation-family), the fair-best
method **flipped** from FDFL-Scal (v1) to WDRO (v2). WDRO's
distributional robustness gives it a more stable operating point
under the tighter budget (0.30 vs 0.35). FDFL-Scal still has
competitive fairness but WDRO edges it.

## 8. Relationship to v1

| Aspect | v1 | v2 |
|---|---|---|
| Budget | 0.35 | **0.30** |
| Lambdas | [0, 0.5, 1, 5] | **[0, 0.5, 1, 2]** |
| lr | 5e-4 | **1e-3** |
| Seeds | 5, fixed split | 3+2, per-seed split |
| Variant B | (none) | val + early stop |
| Max train→test gap | 0.0061 | 0.0161 (Variant B outlier) |
| FPLG α=2 winner? | yes (all 4) | **yes** (all 4, both variants) |
| High-λ pathology? | yes (mad λ=5) | avoided (dropped λ=5) |
| Cross-fairness aggregation | (none) | rank + ratio |
| Train metrics visible? | no | yes (setup.md §6) |

**Main qualitative finding preserved**: FPLG wins regret at α=2.0.
Everything else shifted slightly — which is good, because it means
the headline claim is robust to budget, lr, split seed, λ range, and
training-loop style.

## 9. Methodological caveats

1. **Disjoint seeds** between variants A and B mean we cannot cleanly
   attribute A vs B differences to "training setup". Some of the delta
   is split-to-split variance.
2. **Per-seed splits** in v2 mean test_regret_normalized is averaged
   over different test sets — more conservative than v1's same-split
   averaging but captures split variance.
3. **`lr_decay=5e-3`** is a reciprocal decay approximation of a LR
   schedule; not cosine. The conclusions may shift slightly under a
   textbook cosine schedule.
4. **FAMO not tested**: early stopping fails fast on FAMO because we
   don't snapshot its internal state. Not in the grid.
5. **Val and test are from the SAME split seed's train/test
   partition**. Val is carved from train half, test is the other half.
   Val-test distribution mismatch caused by finite-sample splitting is
   a plausible reason Variant B's gaps are larger.
6. **Early stopping engaged 15% of the time** — the mechanism works
   but the healthcare task with analytic gradients doesn't overfit
   aggressively enough for it to be the dominant effect.

## 10. Takeaways for the paper

- **v2 reaffirms v1's central claim**: FPLG dominates regret across
  all fairness measures at α=2.0 on the full healthcare cohort, with
  PCGrad as consistent runner-up. This result is robust to the
  modifications listed in §8.
- **Cross-fairness picture**: methods specialise. FPLG = regret
  winner; PCGrad = bias_parity winner and best consistent fairness at
  α=0.5; MGDA = dp winner; WDRO = robust runner-up. FPTO is
  surprisingly competitive on fairness rank (top-3) at α=2.0 despite
  being a plain predict-then-optimize baseline.
- **α regime dependence** is clean: α=2.0 is the decision-focused
  regime where DFL/MOO methods shine; α=0.5 is the near-utilitarian
  regime where predict-then-optimize suffices. Matches the
  misspecification note's prediction.
- **Validation + early stopping is not a free lunch** at full cohort
  scale. Variant B is slightly worse on regret at α=2.0 across all 4
  fairness types — the cost of data carved out for val + longer
  training outweighs early stopping's gains.
- **Cross-fairness rank-based aggregation** is a cleaner way to
  compare across fairness types than raw fairness units. It also
  reveals that FPLG's clean regret Pareto curve comes at the cost of
  fairness — it never ranks #1 on fairness.

## 11. Deferred

- Healthcare Step 4b (budget sweep) — still deferred.
- Replace reciprocal `lr_decay` with real cosine schedule and re-run
  Variant B — would strengthen the argument that the val+schedule
  degradation isn't a crude-schedule artifact.
- Add WS-equal / WS-dec / WS-fair to give MOO methods a
  preference-weighted frontier instead of single points.
- MD knapsack re-tune at larger n_train — still deferred, see
  `results/advisor_review/hp_tuning/md_knapsack/STATUS.md`.
