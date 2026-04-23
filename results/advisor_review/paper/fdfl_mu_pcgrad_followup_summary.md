# FDFL-mu and Normalized-PCGrad Follow-up Summary

Date: 2026-04-21  
Branch: `fair-dfl/empirical-followup`

## Executive summary

The follow-up confirmed the original diagnosis.

1. On **MD knapsack**, adding a fixed prediction-loss weight `mu` does
   help FDFL materially. The no-anchor variant (`mu=0`) is unstable
   under SPSA, while small positive anchors (`mu=0.1`, `0.5`) restore
   performance to the same band as the better scalarized methods.
2. On **MD knapsack**, **PCGrad normalization helps**, but it does not
   fully rescue PCGrad in the SPSA regime. The method improves relative
   to the earlier poor baseline, but it still trails the anchored
   scalarized methods.
3. On **healthcare**, where decision gradients are **analytic** rather
   than SPSA-based, the FDFL `mu` variants are nearly identical in test
   regret. This means the prediction anchor is not doing useful work in
   that regime, so there is no strong reason to turn it on for standard
   healthcare FDFL runs.
4. On **healthcare**, **PCGrad normalization does help**: normalized
   PCGrad improves over the prior paper-era value and becomes the best
   mean-regret method in the MAD / `alpha=2` slice of the follow-up.

## What we originally found

The original experiments suggested a sharp contrast between tasks:

- **Healthcare**: end-to-end methods were stable and competitive.
  FPLG was strongest at `alpha=2`, and PCGrad was often a close
  runner-up.
- **MD knapsack**: several end-to-end methods behaved much worse,
  especially FDFL without an explicit prediction anchor and the
  MOO handlers under SPSA decision gradients.

At that point, the working explanation was that the knapsack setting was
simply "harder." The advisor-review diagnosis made the mechanism more
precise.

## Diagnosis before the fix

The diagnosis identified two distinct failure modes in the SPSA-based
MD knapsack setting:

1. **FDFL without a prediction anchor**  
   When `mu=0`, FDFL optimizes decision regret plus fairness but has no
   direct prediction-loss term keeping the predictor calibrated. Under
   noisy SPSA gradients, this allows the predictor to drift.

2. **PCGrad with severe gradient-scale imbalance**  
   The decision gradient norm could be on the order of thousands, while
   the prediction gradient norm was around order 1. In that regime,
   PCGrad's geometry is dominated by the decision objective's scale, so
   projection no longer reflects a balanced multi-objective trade-off.

These findings motivated two code changes:

- fixed numeric `pred_weight_mode` support so FDFL can run at
  `mu in {0.1, 0.5, ...}`
- per-objective normalization inside PCGrad before conflict projection

## What we changed

### 1. Fixed-`mu` support for FDFL

Both `_pred_weight()` implementations now accept numeric strings, so
`pred_weight_mode="0.1"` or `"0.5"` works as a fixed prediction-loss
weight.

New method entries:

- `FDFL-0.1`
- `FDFL-0.5`

Interpretation:

- `FDFL` = no prediction anchor (`mu=0`)
- `FDFL-0.1`, `FDFL-0.5` = anchored FDFL
- `FDFL-Scal` = standard scalarized setting (`mu=1`)
- `FPLG` = dynamic prediction anchor via `alpha_t`

### 2. Normalized PCGrad

PCGrad now optionally:

- L2-normalizes each objective gradient before projection
- performs pairwise conflict projection in normalized space
- rescales the final direction by the mean of original norms

This keeps the step size on a realistic scale while avoiding domination
by the largest raw gradient.

The default experiment config now turns this on for `PCGrad`.

## Results: MD knapsack

Diagnostic slice: `lambda=0`, `alpha=2`, 5 seeds.

| Method | Test regret normalized |
|---|---:|
| FDFL | 0.8384 +- 0.3560 |
| FDFL-0.1 | 0.1970 +- 0.0037 |
| FDFL-0.5 | 0.1864 +- 0.0045 |
| FDFL-Scal | 0.1748 +- 0.0069 |
| FPLG | 0.1914 +- 0.0041 |
| PCGrad (normalized) | 0.3545 +- 0.3618 |
| MGDA | 0.3301 +- 0.3747 |
| FPTO | 0.2028 +- 0.0030 |
| WDRO | 0.2028 +- 0.0030 |
| SAA | 0.2051 +- 0.0000 |

### Interpretation

- **Yes, `mu` helped FDFL.**  
  This is the clearest result of the follow-up. Plain FDFL at `mu=0`
  is unstable, while even `mu=0.1` largely fixes the problem.

- **Yes, PCGrad normalization helped relative to the earlier diagnosis,**
  but **not enough**.  
  Normalized PCGrad is no longer the catastrophic outlier it once
  appeared to be, but it is still substantially worse than anchored
  scalarized methods like `FDFL-0.5`, `FDFL-Scal`, and `FPLG`.

- **Main lesson for MD knapsack:**  
  the first-order fix is to restore a prediction anchor. Normalization
  addresses gradient scale, but it does not fully eliminate the damage
  caused by noisy SPSA decision gradients interacting with projection-
  based MOO updates.

## Results: healthcare

Diagnostic slice used in the follow-up summary:
MAD fairness, `alpha=2`, all appended cells from the full-cohort run.

| Method | Test regret normalized |
|---|---:|
| FDFL | 0.1025 +- 0.0270 |
| FDFL-0.1 | 0.1018 +- 0.0268 |
| FDFL-0.5 | 0.1023 +- 0.0275 |
| FDFL-Scal | 0.1028 +- 0.0282 |
| FPLG | 0.1017 +- 0.0267 |
| PCGrad (normalized) | 0.0945 +- 0.0424 |

### Interpretation

- **`mu` does not matter much on healthcare.**  
  All FDFL variants are effectively tied. This is consistent with the
  analytic-gradient regime: the decision gradient is already stable and
  informative, so the prediction anchor is not needed to prevent drift.

- **So for healthcare, it is reasonable to keep `mu` off for ordinary
  FDFL runs.**  
  There is no strong empirical benefit from adding a fixed prediction
  anchor in this setting.

- **PCGrad normalization does help on healthcare.**  
  Normalized PCGrad improved relative to the paper-era result and is the
  best mean-regret method in this follow-up slice.

## Final takeaways

### What the follow-up supports

1. **Keep the fixed-`mu` feature.**  
   It is justified by a real instability in SPSA-based settings.

2. **Use `mu>0` when decision gradients are noisy or approximate.**  
   For SPSA-style tasks like MD knapsack, this is a meaningful
   stabilization tool.

3. **Do not force `mu>0` on healthcare.**  
   In the analytic-gradient setting, the results are flat across `mu`,
   so the extra anchor is unnecessary for the standard FDFL variants.

4. **Keep PCGrad normalization as the default.**  
   It is helpful in the analytic-gradient healthcare setting and is a
   principled correction for objective-scale imbalance.

### What the follow-up does NOT support

1. **It does not show that normalized PCGrad fully solves SPSA
   knapsack.**  
   The method still trails the anchored scalarized methods there.

2. **It does not imply that `mu` should always be turned on.**  
   The benefit is regime-dependent; it is clearly useful under noisy
   approximate gradients and mostly irrelevant under analytic gradients.

## Recommended paper / advisor narrative

The cleanest narrative is:

- The original gap between healthcare and MD knapsack was not just
  "task difficulty."
- The follow-up identifies a concrete mechanism:
  - on SPSA tasks, FDFL needs a prediction anchor and PCGrad suffers from
    scale imbalance plus noisy gradients
  - on analytic-gradient tasks, the anchor is unnecessary, but
    normalization still helps PCGrad
- Therefore:
  - **fixed `mu` is a justified feature for robust FDFL**
  - **normalized PCGrad is a justified default**
  - but **projection-based MOO remains less reliable than anchored
    scalarization under SPSA**

## Key files

- MD mu sweep:
  `results/advisor_review/md_knapsack_mu_sweep/`
- Healthcare append:
  `results/advisor_review/healthcare_followup_v2/`
- Gradient diagnostic figure:
  `results/advisor_review/md_knapsack_mu_sweep/fig_gradient_scale_mu_sweep.png`
- Insertable LaTeX section:
  `results/advisor_review/paper/fdfl_mu_followup_section.tex`
