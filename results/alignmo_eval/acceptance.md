# AlignMO Phase 2 acceptance memo

**Cells:** 8  
**Do-no-harm cells (AlignMO ≤ best fixed + 1 seed-std):** 5/8
**Do-good cells (AlignMO strictly < best fixed − 1 seed-std):** 3/8
**Lowest avg rank overall:** fpto (AlignMO avg rank = 3.00)

**Phase 2 acceptance criteria (Section 5.4):**
- Do-no-harm on every cell: **FAIL**
- Do-good on ≥ 2 cells **OR** lowest avg rank: **PASS**
- Mode-trace diversity (≥ 2 modes used anywhere): **PASS**

## Per-cell table (AlignMO vs best fixed handler)

| regime | alpha | best_fixed (λ) | fixed mean | fixed std | AlignMO (λ) | AlignMO mean | gap | gap/std | do-no-harm | do-good |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analytic | 0.5 | fpto (2.00) | 0.0501 | 0.0013 | AlignMO (0.00) | 0.0633 | +0.0133 | +10.00 | ✗ | ✗ |
| analytic | 1.5 | pcgrad (0.00) | 0.0140 | 0.0000 | AlignMO (0.00) | 0.0143 | +0.0003 | +8.47 | ✗ | ✗ |
| analytic | 2.0 | fplg (2.00) | 0.1525 | 0.0070 | AlignMO (0.50) | 0.1562 | +0.0037 | +0.52 | ✓ | ✗ |
| analytic | 3.0 | fdfl-0.5 (2.00) | 2.5161 | 0.4649 | AlignMO (0.00) | 4.5770 | +2.0609 | +4.43 | ✗ | ✗ |
| spsa | 0.5 | fpto (2.00) | 0.0501 | 0.0013 | AlignMO (1.00) | 0.0485 | -0.0015 | -1.17 | ✓ | ✓ |
| spsa | 1.5 | fpto (2.00) | 0.0160 | 0.0004 | AlignMO (1.00) | 0.0149 | -0.0011 | -3.13 | ✓ | ✓ |
| spsa | 2.0 | fpto (2.00) | 0.1748 | 0.0084 | AlignMO (0.50) | 0.1655 | -0.0093 | -1.10 | ✓ | ✓ |
| spsa | 3.0 | fpto (2.00) | 2.9215 | 0.7834 | AlignMO (0.50) | 2.7113 | -0.2102 | -0.27 | ✓ | ✗ |

## AlignMO mode fractions (per cell, averaged across seeds & λ)

| regime | alpha | scalarized | projected | anchored | anchored_projected |
| --- | --- | --- | --- | --- | --- |
| analytic | 0.5 | 0.20 | 0.00 | 0.23 | 0.57 |
| analytic | 1.5 | 0.20 | 0.00 | 0.37 | 0.43 |
| analytic | 2.0 | 0.52 | 0.36 | 0.04 | 0.08 |
| analytic | 3.0 | 0.65 | 0.07 | 0.28 | 0.00 |
| spsa | 0.5 | 0.20 | 0.00 | 0.74 | 0.06 |
| spsa | 1.5 | 0.20 | 0.00 | 0.74 | 0.06 |
| spsa | 2.0 | 0.20 | 0.00 | 0.74 | 0.06 |
| spsa | 3.0 | 0.20 | 0.00 | 0.74 | 0.06 |

## Average rank (lower is better)

| method | avg_rank |
| --- | --- |
| fpto | 2.50 |
| alignmo | 3.00 |
| fplg | 3.25 |
| pcgrad | 3.75 |
| fdfl-scal | 5.12 |
| fdfl-0.5 | 5.50 |
| fdfl-0.1 | 6.25 |
| fdfl | 7.00 |
| mgda | 8.62 |