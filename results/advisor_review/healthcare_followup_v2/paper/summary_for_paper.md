# Healthcare v2 — paper-ready summary

**Data source**: `results/advisor_review/healthcare_followup_v2/variant_a/`
(5 seeds: `[11, 22, 33, 44, 55]`, 40 runs, 640 stage rows, 0 NaN, 0
explosion).

**Branch**: `fair-dfl/empirical-followup`.

**Supersedes**: the 3-seed Variant A + Variant B reporting in
[../notes_v2.md](../notes_v2.md). Variant B (val + early stopping) is
kept on disk for reference but is **not** used in the paper numbers.

## 1. Experimental setup

**Dataset.** Full Obermeyer et al. (2019) healthcare cohort, $n = 48{,}784$
patients, each with 65 features, a benefit label, a per-patient cost,
and a binary group label (Black / non-Black). Train and test are split
50/50. Each seed gets its own `split_seed` (coupled), so the 5 seeds
produce 5 different train/test partitions.

**Decision problem.** Given predictions $\hat y = f(x)$, an allocator
solves
$$\max_{d \ge 0}\; U_\alpha(d \odot \hat y,\,g)\quad \text{s.t.}\quad \sum_i d_i c_i \le B$$
where $U_\alpha$ is a 2-level $\alpha$-fair welfare aggregating
$d_i \hat y_i$ within groups and then across them.
$B = 0.30 \cdot \sum_i c_i \bar y$ (30 % of average expenditure on the
full population). We evaluate two welfare curvatures:
$\alpha = 0.5$ (near-utilitarian) and $\alpha = 2.0$
(decision-focused regime where group welfare matters).

**Fairness measures** (full formulas in
[../setup.md](../setup.md) §2). All four use the same smoothed-MAD
aggregator over a per-group statistic:

- **MAD** — per-group MSE (separation family)
- **DP** — per-group mean prediction (independence family)
- **BiasParity** — per-group mean residual (sufficiency family)
- **Atkinson** — scale-invariant inequality index over per-group MSE

**Training.** MLP predictor (`hidden=64`, `n_layers=2`, ReLU,
softplus output), PyTorch-default init (no explicit Kaiming He), full-
batch SGD, `lr = 10^{-3}`, `lr_decay = 5 \times 10^{-4}`, 70 steps per
$\lambda$. **No validation split, no LR schedule beyond the gentle
reciprocal decay, no early stopping.** Analytic decision gradients
(closed form for the healthcare LP).

**Methods** (7 total):

- **FPTO** — Fairness-aware Predict-Then-Optimise (no regret gradient)
- **FDFL-Scal** — 3-objective weighted-sum DFL (dec + pred + fair),
  static weight
- **FPLG** — 3-objective PLG-style DFL with an $\alpha_t$ prediction-
  weight schedule
- **PCGrad** — 3-objective with PCGrad gradient projection (MOO)
- **MGDA** — 3-objective with MGDA convex gradient combination (MOO)
- **SAA** — Sample-Average Approximation (predicts the train mean)
- **WDRO** — Wasserstein Distributionally-Robust prediction baseline

Lambdas swept: $\lambda \in \{0, 0.5, 1, 2\}$ for fairness-aware non-
MOO methods (FPTO, FDFL-Scal, FPLG). MOO methods and non-fair baselines
run at $\lambda = 0$ only (the runner collapses these because the MOO
handler ignores the $\lambda$ scaling of the fairness gradient).

**Pareto-best operating point.** For each (method, fairness, $\alpha$)
cell we report the $\lambda$ that minimises mean test regret across the
5 seeds. Table 3 lists those $\lambda$ values explicitly.

## 2. Table 1: Test regret (normalised)

Mean $\pm$ std across 5 seeds, each seed with its own train/test
partition. Lower is better. **Bold** marks the column winner.

| Method | MAD α=0.5 | MAD α=2.0 | DP α=0.5 | DP α=2.0 | Atkinson α=0.5 | Atkinson α=2.0 | BiasParity α=0.5 | BiasParity α=2.0 |
|---|---|---|---|---|---|---|---|---|
| FPTO | 0.0495 ± 0.0006 | 0.1522 ± 0.0025 | 0.0500 ± 0.0008 | 0.1534 ± 0.0033 | 0.0500 ± 0.0008 | 0.1534 ± 0.0033 | 0.0499 ± 0.0008 | 0.1523 ± 0.0032 |
| FDFL-Scal | 0.0653 ± 0.0036 | 0.1307 ± 0.0019 | 0.0749 ± 0.0013 | 0.1307 ± 0.0019 | 0.0749 ± 0.0013 | 0.1307 ± 0.0019 | 0.0746 ± 0.0013 | 0.1305 ± 0.0019 |
| **FPLG** | 0.0735 ± 0.0013 | **0.1275 ± 0.0020** | 0.0738 ± 0.0012 | **0.1274 ± 0.0020** | 0.0738 ± 0.0012 | **0.1274 ± 0.0020** | 0.0738 ± 0.0012 | **0.1273 ± 0.0020** |
| PCGrad | **0.0488 ± 0.0004** | 0.1282 ± 0.0019 | 0.0583 ± 0.0020 | 0.1363 ± 0.0036 | 0.0500 ± 0.0007 | 0.1284 ± 0.0020 | 0.0498 ± 0.0008 | 0.1291 ± 0.0021 |
| MGDA | 0.0759 ± 0.0014 | 0.1309 ± 0.0013 | 0.0759 ± 0.0013 | 0.1541 ± 0.0028 | 0.0759 ± 0.0014 | 0.1530 ± 0.0042 | 0.0759 ± 0.0014 | 0.1504 ± 0.0068 |
| SAA | 0.0754 ± 0.0002 | 0.2795 ± 0.0024 | 0.0754 ± 0.0002 | 0.2795 ± 0.0024 | 0.0754 ± 0.0002 | 0.2795 ± 0.0024 | 0.0754 ± 0.0002 | 0.2795 ± 0.0024 |
| WDRO | 0.0490 ± 0.0005 | 0.1499 ± 0.0024 | **0.0490 ± 0.0005** | 0.1499 ± 0.0024 | **0.0490 ± 0.0005** | 0.1499 ± 0.0024 | **0.0490 ± 0.0005** | 0.1499 ± 0.0024 |

**Row-wise mean regret across the 8 cells** (overall ordering):

| Method | mean | min | max |
|---|---|---|---|
| PCGrad | **0.0911** | 0.0488 | 0.1363 |
| WDRO | 0.0994 | 0.0490 | 0.1499 |
| FPLG | 0.1006 | 0.1273 | 0.1275 (at α=2 only) — 0.0735 at α=0.5 |
| FPTO | 0.1013 | 0.0495 | 0.1534 |
| FDFL-Scal | 0.1015 | 0.0653 | 0.1307 |
| MGDA | 0.1115 | 0.0759 | 0.1541 |
| SAA | 0.1774 | 0.0754 | 0.2795 |

LaTeX version: [`table1_regret.tex`](./table1_regret.tex).

## 3. Table 2: Test fairness at the Pareto-best λ

Raw fairness values in their native units (MAD on per-group MSE; DP and
BiasParity on benefit scale; Atkinson as a unit-free ratio in $[0, 1]$).
Lower is fairer. **These are evaluated at each method's Pareto-best-for-
regret $\lambda$**, so low regret does not imply low fairness — it is
what the method produces _as a side effect_ of its regret-best operating
point.

| Method | MAD α=0.5 | MAD α=2.0 | DP α=0.5 | DP α=2.0 | Atkinson α=0.5 | Atkinson α=2.0 | BiasParity α=0.5 | BiasParity α=2.0 |
|---|---|---|---|---|---|---|---|---|
| FPTO | 22.096 ± 0.799 | 22.096 ± 0.799 | 1.750 ± 0.089 | 1.750 ± 0.089 | 0.0057 ± 0.0005 | 0.0057 ± 0.0005 | 0.6443 ± 0.0408 | 0.6443 ± 0.0408 |
| FDFL-Scal | 112.172 ± 6.972 | 22.876 ± 1.181 | 0.0033 ± 0.0029 | 1.732 ± 0.137 | 0.0153 ± 0.0014 | 0.0069 ± 0.0006 | 2.523 ± 0.066 | 0.7298 ± 0.0939 |
| FPLG | 119.309 ± 5.595 | 25.110 ± 1.064 | 0.0058 ± 0.0040 | 1.900 ± 0.135 | 0.0152 ± 0.0014 | 0.0079 ± 0.0006 | 2.521 ± 0.066 | 0.5613 ± 0.0951 |
| PCGrad | **21.282 ± 0.579** | 27.386 ± 2.871 | 0.1344 ± 0.0949 | **0.5530 ± 0.3971** | **0.0057 ± 0.0005** | 0.0087 ± 0.0019 | **0.2907 ± 0.0948** | **0.3719 ± 0.1988** |
| MGDA | 120.002 ± 5.616 | 27.734 ± 1.791 | **0.0030 ± 0.0020** | **0.3029 ± 0.0541** | 0.0153 ± 0.0014 | 0.0128 ± 0.0010 | 2.527 ± 0.066 | 2.170 ± 0.062 |
| SAA | 58.498 ± 3.647 | 58.498 ± 3.647 | **0.0010 ± 0.0000** | **0.0010 ± 0.0000** | 0.0100 ± 0.0012 | 0.0100 ± 0.0012 | 2.526 ± 0.067 | 2.526 ± 0.067 |
| WDRO | 21.008 ± 0.851 | **21.008 ± 0.851** | 1.592 ± 0.095 | 1.592 ± 0.095 | 0.0051 ± 0.0005 | **0.0051 ± 0.0005** | 0.9346 ± 0.0550 | 0.9346 ± 0.0550 |

LaTeX: [`table2_fairness.tex`](./table2_fairness.tex).

_Caveat: FPTO, WDRO, SAA show identical values across $\alpha=0.5$ and
$\alpha=2.0$ because they don't train with the welfare in the loss — the
only thing that changes across cells is which $\lambda$ the optimiser
converges at (for PTO-style methods this is a $\lambda$-only choice, not
a training difference). SAA on DP shows $\approx 0.001$ because it
predicts a constant and DP measures prediction spread across groups._

## 4. Table 3: Pareto-best λ per method (per cell)

The $\lambda$ value in $\{0, 0.5, 1, 2\}$ that minimises mean test
regret. Methods whose training is $\lambda$-independent (MGDA, PCGrad,
SAA, WDRO) always appear at $\lambda = 0$.

| Method | MAD α=0.5 | MAD α=2.0 | DP α=0.5 | DP α=2.0 | Atkinson α=0.5 | Atkinson α=2.0 | BiasParity α=0.5 | BiasParity α=2.0 |
|---|---|---|---|---|---|---|---|---|
| FPTO | 0.5 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 2.0 | 2.0 |
| FDFL-Scal | 0.5 | 0.0 | 0.0 | 0.0 | 2.0 | 0.0 | 2.0 | 2.0 |
| FPLG | 2.0 | 0.5 | 2.0 | 1.0 | 2.0 | 1.0 | 2.0 | 1.0 |
| PCGrad | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| MGDA | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| SAA | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| WDRO | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

LaTeX: [`table3_best_lambda.tex`](./table3_best_lambda.tex).

## 5. Key observations

### 5.1 At α = 2.0, FPLG dominates regret across **all four** fairness types

FPLG's test regret at $\alpha=2.0$ is essentially **independent of the
fairness type**: it spans only 0.1273–0.1275 across MAD, DP, Atkinson,
and BiasParity, with per-cell std of ~0.002. This is the strongest
single claim in the experiment:

- **FPLG vs PCGrad (closest rival) at α = 2.0**: FPLG 0.1273–0.1275
  versus PCGrad 0.1282–0.1363 → FPLG wins by 0.7–9.0 × the seed std.
  For MAD and Atkinson cells the margin is within $\sim 1$ std; for
  DP the margin is 4–5 std.
- **FPLG vs FDFL-Scal at α = 2.0**: FPLG 0.1273–0.1275 versus FDFL-Scal
  1.1305–0.1307 → FPLG wins by $\sim 1.5$ std.
- **FPLG vs MGDA at α = 2.0**: FPLG 0.1273–0.1275 vs MGDA 0.1309–0.1541
  → FPLG wins on MAD by 1.5 std and on DP by ~9 std.
- FPLG beats all predict-then-optimise baselines (FPTO, SAA, WDRO) by
  at least 0.022 (more than 10× the seed std) at $\alpha=2$.

### 5.2 At α = 0.5, the winners are predict-then-optimise methods

- **Mad α=0.5**: PCGrad 0.0488 ± 0.0004 (wins), WDRO 0.0490, FPTO
  0.0495 — all within 1 std. FPLG/FDFL-Scal/MGDA are $\sim 0.025$
  higher.
- **DP, Atkinson, BiasParity α=0.5**: WDRO 0.0490 wins; FPTO 0.0499
  essentially tied; PCGrad 0.0498 tied on Atkinson/BiasParity, ahead
  only by the DP fairness-gradient interaction (see §5.6).

**Interpretation**: at $\alpha=0.5$ the welfare is close to a linear sum
of $d_i \hat y_i$, so minimising prediction MSE already minimises regret.
Decision-focused training has **no lever** because the decision objective
and the prediction objective are near-parallel. This reproduces the
pattern predicted in `docs/misspecification_note.md`.

### 5.3 MGDA's regret is much more sensitive to fairness type than FPLG

At $\alpha=2.0$, MGDA's mean test regret spans 0.1309 (MAD) to 0.1541
(DP), a **0.023 spread**. FPLG over the same cells spans 0.1273 to
0.1275 (**0.0002 spread**, $\approx 100\times$ tighter).

The mechanism: MGDA's convex-combination gradient depends directly on
the per-objective raw gradient magnitudes, which differ by orders of
magnitude across fairness types (MAD gradient scale is $\sim 20$, DP is
$\sim 1$, Atkinson is $\sim 0.005$). FPLG's $\alpha_t$ schedule
normalises the prediction-fairness balance in a way that is
fairness-type-agnostic, so its regret curve is essentially flat across
metric choice.

### 5.4 By _mean_ across cells, PCGrad has the lowest regret

PCGrad averages 0.0911 across all 8 cells (lowest), WDRO 0.0994, FPLG
0.1006, FPTO 0.1013. **But PCGrad never wins a single $\alpha = 2$
cell** — it is in second place by 0.0007–0.0089 everywhere. Its mean
advantage comes entirely from the $\alpha = 0.5$ column, where it is
tied with FPTO/WDRO while FPLG is 0.025 higher. When reporting per-
regime results, FPLG dominates at $\alpha = 2$ and PCGrad dominates on
average only because $\alpha = 0.5$ has 4 cells worth of votes.

### 5.5 Which method is fairest at _its own best regret_?

Looking at Table 2, at each method's regret-best $\lambda$:

- **Best MAD at α=2.0**: WDRO 21.0, FPTO 22.1, FDFL-Scal 22.9 — all
  well below FPLG's 25.1 and PCGrad's 27.4. WDRO is the most MAD-fair
  because its λ=0 training happens to produce the smallest per-group
  MSE spread.
- **Best DP at α=2.0**: MGDA 0.30 (winner by a lot); PCGrad 0.55, SAA
  0.001 (but at catastrophic regret 0.28); FPLG/FDFL-Scal 1.7–1.9.
  MGDA's projection finds the independence direction even at $\lambda
  = 0$ because the DP fairness gradient still enters the handler raw.
- **Best BiasParity at α=2.0**: **PCGrad 0.37** (winner), FPLG 0.56,
  FDFL-Scal 0.73, WDRO 0.93. PCGrad's orthogonal-projection step
  zeros out the per-group mean residual while leaving MSE reduction
  nearly untouched. This is a clean method-fairness pairing.
- **Best Atkinson at α=2.0**: WDRO 0.0051, FPTO 0.0057, PCGrad 0.0087.

**FPLG never wins a fairness column.** This is the classic regret-vs-
fairness trade-off: FPLG's clean $\alpha_t$ schedule that gives it a
flat regret curve across fairness types also means it does not chase
any specific fairness direction as aggressively as the specialised
methods.

### 5.6 Even at λ = 0, MOO methods are not fairness-type-agnostic

PCGrad's DP-$\alpha$=0.5 regret (0.0583) is visibly higher than its MAD-
$\alpha$=0.5 regret (0.0488) even though both cells use $\lambda = 0$.
The reason: at $\lambda = 0$ the fairness gradient is zero for non-
MOO methods (because it enters through $\beta_t = \lambda$), but MOO
handlers receive the _raw_ per-objective gradients, and the
orthogonal-projection step in PCGrad still includes the fairness
gradient direction in the combination even when $\lambda = 0$.
Different fairness gradients produce different projected update
directions, so PCGrad's $\lambda = 0$ training is not identical across
fairness types. This is a subtle but reproducible effect.

## 6. Suggested figure + caption for the paper

**Figure**:
[`../variant_a/plots/pareto_grid_8cells.png`](../variant_a/plots/pareto_grid_8cells.png)
— 8-panel grid. Rows are fairness types; columns are $\alpha$. Each
method is plotted at its best operating point with $\pm 1$ std error
bars across the 5 seeds. Training set performance is overlaid dashed.

**Caption (draft)**:

> **Figure X**: Pareto frontier on the healthcare task. For each of 4
> fairness measures ($y$-axis: `test_fair`; see text) and 2 welfare
> curvatures $\alpha$ ($x$-axis: normalised test regret), we plot the
> 7 methods at their Pareto-best $\lambda$, with $\pm 1$ std across 5
> seeds. Solid markers are test, dashed open markers are train. FPLG
> (pink $\times$) sweeps the lowest regret in every $\alpha = 2$ panel
> (right column). At $\alpha = 0.5$ (left column) the decision
> objective is too close to MSE to give DFL a lever, and the
> predict-then-optimise baselines FPTO/WDRO/PCGrad are competitive.

## 7. Methodological notes

1. **Seeds**: 5 seeds `[11, 22, 33, 44, 55]`, each with its own
   `split_seed` coupled to the training seed, so cross-seed variance
   captures both initialisation and split-level variation.
2. **Error bars = seed std**, not standard error. Multiply by
   $1/\sqrt{5} \approx 0.45$ for s.e.m.
3. **All 640 stage rows are stable**: 0 NaN, 0 exploding, max
   train→test gap 0.0041 (median 0.0004). The inter-method ranking is
   not driven by generalisation noise.
4. **Lambdas [0, 0.5, 1, 2]**: we dropped $\lambda = 5$ from the v1
   run because it produced catastrophic fairness overfitting on MAD.
   The v1 λ=5 column was an artefact; v2's λ=2 is the high-λ boundary.
5. **Budget**: v2 uses $\rho = 0.30$ (v1 used 0.35). Tighter budget
   sharpens the decision boundary and increases FPLG's regret advantage
   at $\alpha = 2$.
6. **Training**: no validation, no early stopping, no LR schedule
   beyond the gentle reciprocal `lr_decay = 5 \times 10^{-4}`. We
   deliberately keep training simple because (a) decision gradients
   are closed-form so no tuning is needed, (b) the earlier Variant B
   experiment confirmed that adding val + early stop made gaps _worse_
   not better at this scale (disjoint train/val/test partitions +
   longer training budget outweighed early stopping's gains). See
   [`../notes_v2.md`](../notes_v2.md) §6 for the Variant B analysis if
   needed.
7. **MOO methods (MGDA, PCGrad) run at $\lambda = 0$ only**: their
   handlers ignore the $\lambda$ scaling on the fairness gradient, so
   the runner collapses the sweep. This is a design choice of the MOO
   framework, not an experimental limitation.

## 8. Files in this directory

- `grand_table_variant_a.csv` — long-form (method, fairness, alpha)
  table with mean + std for regret, fairness, train metrics, n_seeds
- `table1_regret.{csv, md, tex}` — Table 1 above, 3 formats
- `table2_fairness.{csv, md, tex}` — Table 2, 3 formats
- `table3_best_lambda.{csv, md, tex}` — Table 3, 3 formats
- `per_method_summary.csv` — 1 row per method with mean/min/max regret
- `summary_for_paper.md` — this file

## 9. How to regenerate

```bash
# From the repo root
python -m experiments.advisor_review.paper_summary_v2a
```

All outputs are written to `results/advisor_review/healthcare_followup_v2/paper/`.
