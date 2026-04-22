# Healthcare targeted extension: raw vs normalized PCGrad

Date: 2026-04-21  
Branch: `fair-dfl/empirical-followup`

## Purpose

Re-check the exact paper-style healthcare regime with **five fresh seeds**
to answer a focused question:

> Does normalized PCGrad beat the original raw PCGrad on healthcare?

Setup:

- Task: healthcare resource allocation
- Fairness type: `mad`
- Welfare curvature: `alpha = 2.0`
- Full cohort
- Analytic decision gradients
- Variant-A style training: 70 steps per lambda, no validation / early stop
- Fresh seeds: `66, 77, 88, 99, 111`

Methods included:

- FPTO, SAA, WDRO
- FDFL, FDFL-0.1, FDFL-0.5, FDFL-Scal
- FPLG
- MGDA
- PCGrad-raw (`mo_pcgrad_normalize = False`)
- PCGrad-norm (`mo_pcgrad_normalize = True`)

Results directory:

- `results/advisor_review/healthcare_pcgrad_compare_extra/`

## Main result

On these five fresh seeds, **raw PCGrad is better than normalized PCGrad**
in the exact paper-style healthcare cell.

### Best-by-lambda summary

| Method | Best lambda | Mean normalized test regret | Std |
|---|---:|---:|---:|
| FDFL | 1.0 | 0.1286 | 0.0016 |
| FDFL-0.1 | 0.0 | 0.1285 | 0.0010 |
| FDFL-0.5 | 0.0 | 0.1297 | 0.0011 |
| FDFL-Scal | 0.0 | 0.1309 | 0.0012 |
| FPLG | 0.5 | 0.1282 | 0.0013 |
| **PCGrad-raw** | 0.0 | **0.1291** | 0.0016 |
| **PCGrad-norm** | 0.0 | **0.1369** | 0.0052 |
| MGDA | 0.0 | 0.1304 | 0.0019 |
| FPTO | 0.5 | 0.1519 | 0.0019 |
| WDRO | 0.0 | 0.1532 | 0.0019 |
| SAA | 0.0 | 0.2791 | 0.0013 |

### PCGrad head-to-head

| Variant | Lambda | Mean normalized test regret | Std |
|---|---:|---:|---:|
| PCGrad-raw | 0.0 | 0.1291 | 0.0016 |
| PCGrad-norm | 0.0 | 0.1369 | 0.0052 |

Difference:

- `PCGrad-norm - PCGrad-raw = +0.0078`

So on this targeted extension, normalization is clearly worse.

## Interpretation

This targeted rerun changes the current conclusion about healthcare:

- We **should not claim** that normalized PCGrad improves healthcare.
- In the exact paper-style setting, the fresh-seed evidence goes the other way:
  **raw PCGrad is better**.

Combined with the earlier follow-up confusion, the safest interpretation is:

1. The earlier low aggregate number for PCGrad (`~0.0945`) was not the right
   apples-to-apples comparison for the paper cell.
2. On the exact healthcare cell, the effect of normalization is **not robust**.
3. Therefore normalization should be treated as an **exploratory implementation
   variant**, not as a result we highlight in the paper.

## What this means for `mu`

The `mu` story is more stable:

- On healthcare, the FDFL-family results remain tightly clustered
  (`0.1282` to `0.1309` in this extension).
- So the main message still holds:
  **fixed `mu` matters in noisy SPSA settings, but not much in analytic-gradient
  healthcare.**

## Recommendation

### For the codebase

- Keep fixed-`mu` support.
- It solves a real MD-knapsack instability.

### For PCGrad normalization

- Keep it only if you want it as an optional engineering knob.
- Do **not** present it as a healthcare improvement based on the current data.

### For the paper

- Emphasize the `mu` ablation and the SPSA-instability diagnosis.
- Treat normalized PCGrad as either:
  - a negative / mixed ablation result, or
  - an implementation detail that is not central to the paper's claims.
