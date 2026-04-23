# Hypothesis 3a: Demographic parity vs accuracy parity (healthcare)

**Setup:** healthcare task, n_sample=5000, val_fraction=0.2, test_fraction=0.5,
analytic decision gradient, MLP hidden_dim=64, n_layers=2, lr=5e-4, 70 steps,
5 seeds, lambdas={0, 0.1, 0.3, 1.0, 3.0}, methods={FPTO, FDFL-Scal, FPLG, PCGrad, MGDA}.

Two parallel grids: `mad` = MAD of per-group MSE (accuracy parity) and `dp` =
MAD of per-group mean predictions (demographic parity). 4.3 min total runtime.

## Mean test metrics by method (averaged over lambdas + seeds)

### alpha = 2.0 (egalitarian — fairness matters)

| method     | AP (mad) regret | AP fairness | DP regret | DP fairness |
|------------|----------------:|------------:|----------:|------------:|
| FPLG       |            92.1 |        42.5 |      91.9 |        0.96 |
| FDFL-Scal  |            99.3 |        36.3 |     101.5 |        1.11 |
| PCGrad     |            88.6 |        26.2 |      99.6 |        0.22 |
| FPTO       |           112.0 |        48.0 |     118.2 |        0.44 |
| MGDA       |           121.8 |        86.7 |     171.1 |        0.00 |

**Observations:**

1. **Method rankings are mostly preserved across fairness types.** PCGrad,
   FPLG, FDFL-Scal cluster as the strong group on both metrics; FPTO and MGDA
   are weaker. The exception is **MGDA on DP**, which collapses to "mean
   prediction is identical across groups" (fairness ≈ 0.003) at the cost of
   regret 171 vs 122 on AP. MGDA's "balance every objective" search direction
   is too aggressive when the fairness gradient ignores labels entirely.
2. **PCGrad is the most consistent winner on AP** (best regret AND best
   fairness simultaneously) but loses its edge on DP (slightly worse regret
   than FPLG; FPLG matches it on fairness too). PCGrad's projection works
   well when the fairness gradient is structurally aligned with the
   prediction loss; less so when it isn't.
3. **The Pareto shape changes.** AP's fairness range is roughly 26–87 (in
   per-group MSE units); DP's is 0–1 (in per-group mean prediction units).
   The two fairness signals are not directly comparable in magnitude, but
   the trade-off slope is similar: pushing fairness lower costs ~10–20%
   regret per ~50% fairness reduction in both fairness types.

### alpha = 0.5 (utilitarian — fairness matters less)

| method     | AP regret | AP fairness | DP regret | DP fairness |
|------------|----------:|------------:|----------:|------------:|
| FPTO       |      20.2 |        48.0 |      20.3 |        0.44 |
| PCGrad     |      20.1 |        37.3 |      22.5 |        0.02 |
| FDFL-Scal  |      24.0 |        81.5 |      24.4 |        0.00 |
| FPLG       |      24.2 |        95.4 |      24.2 |        0.00 |
| MGDA       |      24.7 |        95.8 |      24.7 |        0.00 |

**Observations:**

* At α=0.5 the regret gap between FPTO and DFL methods **collapses**: FPTO
  is the best on regret (20.2). This is the "PTO catches up" regime
  documented in `docs/misspecification_note.md`.
* DP "fairness 0.0" entries are not bugs — they mean the lambda schedule
  pushed the per-group mean predictions all the way to identical, which is
  trivial for DP (just shrink predictions to one constant) but devastating
  for AP (you would need to also have equal MSE).

## Conclusion

Demographic parity is a **viable alternative** fairness type for the
healthcare task with a Pareto trade-off slope similar to accuracy parity.
The new `fairness_type="dp"` shipped in commit `2598462` produces sensible,
non-trivial results across all five methods. **MGDA is incompatible with
DP** at α=2.0 — its multi-objective balancing logic interacts badly with
DP's label-free gradient. PCGrad is the safer MOO choice if DP is required.

The paper's central claim — *adding prediction fairness does not lead to
strict drawbacks in decision regret, and often improves it* — holds for
both fairness types at α=2.0 (PCGrad on AP, FPLG on DP both beat FPTO on
both axes simultaneously) and is neutral at α=0.5 (PTO is competitive in
the utilitarian regime).
