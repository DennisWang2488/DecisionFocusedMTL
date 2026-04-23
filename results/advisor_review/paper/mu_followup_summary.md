# FDFL-mu / normalized-PCGrad follow-up summary

Date: 2026-04-21  
Branch: `fair-dfl/empirical-followup`

## Executive summary

The follow-up addressed two separate hypotheses from the MD knapsack diagnosis:

1. **FDFL instability under SPSA** came from having **no prediction anchor**
   when `mu = 0`.
2. **PCGrad instability** came from **raw gradient-scale imbalance**, where the
   decision gradient could dominate the prediction/fairness gradients.

The empirical results support the first hypothesis strongly and the second one
only partially:

- On **MD knapsack**, adding a fixed prediction weight `mu > 0` clearly
  stabilizes FDFL.
- On **healthcare**, varying `mu` has very little effect, consistent with the
  fact that the decision gradients are analytic and already well behaved.
- **Normalized PCGrad** does **not** rescue MD knapsack under SPSA.
- **Normalized PCGrad** also does **not** improve the exact paper-comparable
  healthcare MAD / `alpha=2` cell; the earlier apparent improvement came from
  accidentally averaging across the whole append grid rather than comparing the
  same cell.

## What we originally found

Before this follow-up, the working story was:

- **Healthcare**: end-to-end methods worked well with analytic gradients.
  FPLG was the strongest regret method at `alpha = 2`, with PCGrad close
  behind.
- **MD knapsack**: scalarized methods were better behaved, while FDFL and
  projection-based MOO methods could become unstable or much worse.

The diagnosis from the gradient logs was:

- In **MD knapsack**, the SPSA decision gradient could be noisy and extremely
  large.
- For **FDFL with `mu = 0`**, there was no prediction-loss anchor, so the
  predictor could drift.
- For **PCGrad**, the projection geometry could be dominated by the decision
  gradient norm instead of actual directional conflict.

## What we changed

Two code changes were implemented:

1. **Fixed-`mu` support for FDFL**
   - `pred_weight_mode` now accepts numeric strings such as `"0.1"` or `"0.5"`.
   - New configs: `FDFL-0.1`, `FDFL-0.5`.

2. **Per-objective gradient normalization in PCGrad**
   - Each objective gradient is normalized before projection.
   - The final direction is re-scaled by the mean original norm.
   - `PCGrad` now defaults to `mo_pcgrad_normalize=True`.

## What we found after the fix

### 1. MD knapsack: fixed `mu` clearly helps FDFL

Exact results from:
`results/advisor_review/md_knapsack_mu_sweep/seed_*/stage_results.csv`

At `lambda = 0`:

| Method | Mean normalized test regret | Std |
|---|---:|---:|
| FDFL (`mu=0`) | 0.8384 | 0.3560 |
| FDFL-0.1 | 0.1970 | 0.0037 |
| FDFL-0.5 | 0.1864 | 0.0045 |
| FDFL-Scal (`mu=1`) | 0.1748 | 0.0069 |
| FPLG | 0.1914 | 0.0041 |
| PCGrad (normalized) | 0.3545 | 0.3618 |

Interpretation:

- The **no-anchor FDFL variant is genuinely unstable**.
- Even a **small anchor** (`mu=0.1`) fixes most of the problem.
- `mu=0.5` and `mu=1` land in the same performance band as the best
  scalarized methods.
- **Normalized PCGrad is still weak** in the SPSA regime.

The same pattern holds across the whole lambda sweep:

| Method | `lambda=0` | `0.5` | `1.0` | `2.0` |
|---|---:|---:|---:|---:|
| FDFL | 0.8384 | 0.5184 | 0.8390 | 0.7421 |
| FDFL-0.1 | 0.1970 | 0.1967 | 0.1968 | 0.1965 |
| FDFL-0.5 | 0.1864 | 0.1856 | 0.1865 | 0.1860 |
| FDFL-Scal | 0.1748 | 0.1760 | 0.1752 | 0.1755 |
| FPLG | 0.1914 | 0.1730 | 0.1514 | 0.1420 |
| PCGrad | 0.3545 | n/a | n/a | n/a |

This is strong evidence that the original FDFL failure mode in MD knapsack was
primarily a **missing prediction anchor** problem.

### 2. Healthcare: `mu` is mostly irrelevant

Exact results from:
`results/advisor_review/healthcare_followup_v2/variant_a/mad/alpha_2.0/seed_*/stage_results_fdfl_mu.csv`

The paper-comparable slice is **MAD fairness, `alpha=2.0`**.

| Method | Best lambda in append run | Mean normalized test regret | Std |
|---|---:|---:|---:|
| FDFL (`mu=0`) | 1.0 | 0.1282 | 0.0022 |
| FDFL-0.1 | 0.0 | 0.1282 | 0.0017 |
| FDFL-0.5 | 0.0 | 0.1294 | 0.0019 |
| FDFL-Scal (`mu=1`) | 0.0 | 0.1307 | 0.0019 |
| FPLG | 0.5 | 0.1279 | 0.0019 |

Interpretation:

- All FDFL `mu` variants are in a **tight cluster**.
- This supports the expected story: with **analytic decision gradients**, the
  anchor is not what determines performance.
- So from a modeling perspective, it is reasonable to keep the standard FDFL
  behavior off by default in healthcare-style settings and treat fixed `mu`
  mainly as a stabilization tool for approximate-gradient settings.

### 3. Healthcare: normalized PCGrad did **not** improve the paper-comparable cell

This point needs extra care.

An earlier quick summary reported:

- `PCGrad = 0.0945`

That number is **not** the correct apples-to-apples comparison to the paper
table. It came from averaging over the full append grid rather than isolating
the exact paper-comparable cell.

The correct comparison for **MAD fairness, `alpha=2.0`** is:

| Method | Last paper version | New append run | Delta |
|---|---:|---:|---:|
| PCGrad | 0.1282 +- 0.0019 | 0.1333 +- 0.0035 | +0.0051 |

So in the exact healthcare cell that corresponds to the paper table:

- **PCGrad did not improve**
- it appears to be **slightly worse**

This means we should **not** claim that normalized PCGrad improved the
healthcare result unless we are explicitly talking about a different aggregate
or a different comparison target.

## Correct interpretation of the follow-up

### Supported strongly

- Fixed `mu` is a useful new feature because it fixes a real instability in
  **SPSA / approximate-gradient** settings.
- On MD knapsack, the original FDFL problem was not just “knapsack is hard”;
  it was specifically that **`mu = 0` leaves the model unanchored**.

### Supported weakly / partially

- Gradient normalization is a reasonable PCGrad modification in principle.
- But in this follow-up it **does not rescue PCGrad on MD knapsack**, and in
  the exact paper-comparable healthcare cell it **does not improve the result**.

## Recommended story going forward

1. **Keep fixed-`mu` support in the codebase.**
   - It is empirically justified.
   - It gives a clean way to study how much prediction anchoring is needed.

2. **Keep normalized PCGrad as an implementation option/default, but do not
   oversell it from this experiment.**
   - The current follow-up does not support a strong empirical claim that it
     improves the main healthcare paper result.
   - It also does not solve the SPSA knapsack problem.

3. **Main paper-level takeaway from this follow-up**
   - The MD knapsack failure mode is best understood as an
     **approximate-gradient instability** problem.
   - A small fixed prediction-loss anchor is sufficient to suppress it.

## Files / directories

- MD mu sweep:
  `results/advisor_review/md_knapsack_mu_sweep/`
- Healthcare mu append:
  `results/advisor_review/healthcare_followup_v2/`
- Gradient diagnostics figure:
  `results/advisor_review/md_knapsack_mu_sweep/fig_gradient_scale_mu_sweep.png`
- Standalone LaTeX write-up:
  `results/advisor_review/paper/fdfl_mu_followup_section.tex`
