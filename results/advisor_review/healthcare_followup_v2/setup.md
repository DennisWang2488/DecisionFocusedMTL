# Healthcare follow-up v2 — experimental setup and methodology

**Branch**: `fair-dfl/empirical-followup`
**Run date**: 2026-04-14
**Previous run**: v1 at `results/advisor_review/healthcare_followup/` (commit `1355939`)

This document pins down the **problem definition**, **fairness measures**
(with formulas and semantic interpretation), **training configurations**
for the two variants, and the **cross-fairness comparison methodology**.
Read this before the `notes_v2.md` findings.

## 1. Task definition

### 1.1 Data

Full Obermeyer 2019 healthcare dataset, $n = 48{,}784$ patients, each with:

- A feature vector $x_i \in \mathbb{R}^{p}$ (65 clinical/demographic features)
- A benefit label $y_i \in \mathbb{R}_{\ge 0}$ (predicted benefit from
  extended care)
- A per-patient cost $c_i > 0$
- A group label $g_i \in \{g_1, g_2\}$ (Black vs non-Black, from the
  Obermeyer paper)

We use a 50/50 train/test split (`test_fraction=0.5`). In Variant B we
additionally carve 20% of the train split into a validation set. Each
training seed gets its own `split_seed`, so seed 11 trains/tests on a
different partition than seed 22.

### 1.2 Decision problem (LP)

Given a predictor $f: \mathbb{R}^{p} \to \mathbb{R}_{\ge 0}$ producing
predictions $\hat{y}_i = f(x_i)$, the allocator solves:

$$
\begin{aligned}
\max_{d \in \mathbb{R}^{n}_{\ge 0}} \quad & U_\alpha(d \odot \hat{y}, g) \\
\text{s.t.} \quad & \sum_{i=1}^{n} d_i c_i \;\le\; B, \qquad d_i \ge 0
\end{aligned}
$$

where $d_i \ge 0$ is the allocated dose for patient $i$, and $B$ is the
budget. We set $B = \rho \cdot \sum_i c_i \cdot \bar{y}$ with
$\rho = 0.30$ (v1 used 0.35). $\bar{y}$ is the mean benefit over the
training set — the budget represents 30 % of an "average-cost
expenditure" on the whole population.

### 1.3 Welfare function: 2-level $\alpha$-fair welfare

The welfare functional aggregates over patients within a group and then
over groups. For group $g$ with members $I_g = \{i : g_i = g\}$ and
allocation utility $u_i(d_i) = d_i \hat{y}_i$:

**Inner aggregation** (per-group utility):
$$
U_g(d, \hat{y}) \;=\; \frac{1}{\alpha}\Big(\sum_{i \in I_g} d_i \hat{y}_i\Big)^{\alpha}
$$

**Outer aggregation** (across groups):
$$
U_\alpha(d, \hat{y}, g) \;=\; \sum_{g \in \mathcal{G}} U_g(d, \hat{y})
$$

- **$\alpha = 0.5$** → near-utilitarian; the welfare is roughly linear in
  total allocated benefit, and group boundaries barely matter.
- **$\alpha = 2.0$** → the inner aggregation becomes a **convex**
  function of group utility, so the outer sum penalises concentration
  within a group. To maximise, the LP prefers _spreading_ benefit to
  groups that currently have less, which is the "fair" behaviour. This
  is the regime where decision-focused training has a lever over MSE-
  only training.

### 1.4 Decision regret

For a predictor $f$, the decision regret on split $S$ is:

$$
R(f; S) \;=\; U_\alpha(d^\star(y), y, g) \;-\; U_\alpha(d^\star(\hat{y}), y, g)
$$

where $d^\star(\cdot)$ is the LP optimum given the predictions. Normalised
regret divides by the oracle objective:

$$
\tilde{R}(f; S) \;=\; \frac{R(f; S)}{|U_\alpha(d^\star(y), y, g)|}
$$

so it's unit-free and comparable across seeds (even when each seed has a
different test set).

## 2. Fairness measures

All four fairness measures have the same **outer aggregator** — a
smoothed mean absolute deviation (MAD) over per-group statistics — so
the only thing that differs between them is which per-group statistic
$s_g$ enters the aggregator.

### 2.1 Common aggregator

For per-group statistics $\{s_1, \ldots, s_G\}$, smoothing $\varepsilon
= 10^{-6}$, the smoothed-MAD loss is:

$$
L_{\text{agg}}(\{s_g\}) \;=\; \frac{1}{G}\sum_{g=1}^{G} \sqrt{\big(s_g - \bar{s}\big)^2 + \varepsilon},
\qquad \bar{s} = \frac{1}{G}\sum_g s_g
$$

The square root inside guarantees a Lipschitz-continuous loss that's
differentiable everywhere. $\varepsilon$ prevents zero-gradient pathologies
at $s_g = \bar{s}$.

### 2.2 MAD on per-group MSE (separation family)

Per-group statistic is the prediction MSE restricted to that group:

$$
s_g^{\text{mad}} \;=\; \text{MSE}_g(f) \;=\; \frac{1}{|I_g|}\sum_{i \in I_g}\big(f(x_i) - y_i\big)^2
$$

$$
\boxed{L_{\text{mad}}(f) \;=\; \frac{1}{G}\sum_{g=1}^{G}\sqrt{(\text{MSE}_g - \overline{\text{MSE}})^2 + \varepsilon}}
$$

**Family**: _separation_ (equalised errors). This is the "equality of
mistake magnitudes across groups" objective — also known as balanced
error parity in the fairness literature.

**What it does well**: catches any systematic under-performance of the
predictor on a protected group, whether it's from variance or bias.

**Failure mode**: rewards degenerate flat predictors. A constant
predictor $f \equiv c$ that perfectly equalises per-group MSE (because
each group has the same y variance) would score a perfect 0 on MAD.
Always read MAD together with the overall MSE.

### 2.3 Demographic parity on predictions (independence family)

Per-group statistic is the mean prediction in that group:

$$
s_g^{\text{dp}} \;=\; \mu_g(f) \;=\; \frac{1}{|I_g|}\sum_{i \in I_g} f(x_i)
$$

$$
\boxed{L_{\text{dp}}(f) \;=\; \frac{1}{G}\sum_{g=1}^{G}\sqrt{(\mu_g - \bar{\mu})^2 + \varepsilon}}
$$

**Family**: _independence_ $\hat{Y} \perp G$. The prediction
distribution should not depend on group membership in the mean.

**What it does well**: penalises any classifier whose _output_ is
systematically different between groups, regardless of ground truth.
Useful when you want "the model treats everyone the same on average".

**Failure mode**: penalises _accurate_ predictors when the ground-truth
label distribution genuinely differs between groups. In the healthcare
data, the mean true benefit actually differs between groups, so a
correctly-calibrated predictor will have $\mu_1 \ne \mu_2$ and score
non-zero on DP — even though it's not discriminating.

### 2.4 Bias parity on residuals (sufficiency family)

Per-group statistic is the mean signed residual in that group:

$$
s_g^{\text{bp}} \;=\; b_g(f) \;=\; \frac{1}{|I_g|}\sum_{i \in I_g}\big(f(x_i) - y_i\big)
$$

$$
\boxed{L_{\text{bp}}(f) \;=\; \frac{1}{G}\sum_{g=1}^{G}\sqrt{(b_g - \bar{b})^2 + \varepsilon}}
$$

**Family**: _sufficiency / calibration first moment_. $\mathbb{E}[Y \mid
\hat{Y}, G] = \mathbb{E}[Y \mid \hat{Y}]$. If the predictor is
well-calibrated within each group, $b_g = 0$ for all $g$, and
$L_{\text{bp}} = 0$.

**What it does well**: catches systematic over- or under-prediction per
group. If the model is uniformly optimistic about group $g_1$ and
pessimistic about $g_2$, bias parity will catch it even when the _magnitude_
of errors (MSE) is the same.

**Failure mode**: ignores variance entirely. A group can have huge
per-sample errors (say, errors are $\pm 100$ with equal probability)
that average to zero residual, and still score $b_g = 0$. Always read bias
parity together with MSE.

### 2.5 Atkinson index on per-group MSE

Per-group statistic is again the MSE, but the aggregator is **different**
— Atkinson is a scale-invariant inequality index. With inequality
aversion $\epsilon = 0.5$:

$$
\text{EDE}_\epsilon(\{s_g\}) \;=\; \left(\frac{1}{G}\sum_{g=1}^{G} s_g^{1-\epsilon}\right)^{\frac{1}{1-\epsilon}},
\qquad \bar{s} = \frac{1}{G}\sum_g s_g
$$

$$
\boxed{L_{\text{atk}}(f) \;=\; 1 - \frac{\text{EDE}_\epsilon(\{\text{MSE}_g\})}{\overline{\text{MSE}}}}
$$

EDE is the "equally-distributed equivalent" — the per-group value that,
if all groups had it, would produce the same aggregate welfare as the
actual distribution. The ratio $\text{EDE} / \bar{s}$ is between 0 (max
inequality) and 1 (perfect equality), so the Atkinson index is in $[0,
1]$.

**Family**: a _concave inequality index_ applied to per-group MSE. Has
the same "separation" goal as MAD (equalising per-group errors), but
using a different aggregator.

**What it does well**: scale-invariant. Multiplying all MSEs by a
constant doesn't change Atkinson — it measures the _shape_ of the
distribution, not the magnitude. Robust to overall model quality.

**Failure mode**: with $\epsilon = 0.5$ (weak aversion to inequality),
Atkinson is insensitive to small differences. A predictor with MSE =
{0.1, 0.11} has nearly the same Atkinson as one with MSE = {100, 110}.
If you care about absolute differences, use MAD or variance instead.

### 2.6 Why different fairness families disagree

MAD, DP, bias_parity, and Atkinson measure different things. In
particular:

- **Separation** (MAD, Atkinson): equalise error _magnitude_ per group
- **Independence** (DP): equalise prediction _distribution_ per group
- **Sufficiency** (bias_parity): equalise per-group _signed error_
  (calibration first moment)

These families are **provably incompatible** in general (Chouldechova
2017, Kleinberg–Mullainathan–Raghavan 2016): you cannot satisfy all three
simultaneously unless either (a) the groups have identical label
distributions, or (b) the predictor is perfect. Our healthcare data has
clearly different per-group label distributions, so trade-offs between
these families are inevitable and methods end up specialising in one
family or another.

## 3. Hyperparameters — Variant A

Baseline run: 3 seeds, no validation, no early stopping.

| HP | Value | Notes |
|---|---|---|
| `n_sample` | 0 (full 48,784 cohort) | |
| `test_fraction` | 0.5 | 50/50 train/test |
| `val_fraction` | **0.0** | no validation split in A |
| `split_seed` | **= seed** (per-seed coupling) | each seed gets a different partition |
| `budget_rho` | **0.30** | was 0.35 in v1 |
| `lambdas` | **[0.0, 0.5, 1.0, 2.0]** | was [0, 0.5, 1, 5] in v1 |
| `steps_per_lambda` | 70 | same as v1 |
| `optimizer` | `sgd` | full batch, no momentum |
| `lr` | **1e-3** | was 5e-4 in v1 |
| `lr_decay` | 5e-4 | gentle reciprocal decay, $\text{lr}_t = \text{lr}_0 / (1 + \text{lr\_decay} \cdot t)$ |
| `batch_size` | -1 (full) | |
| `weight_decay` | 0 | no L2 |
| Architecture | MLP, `hidden_dim=64`, `n_layers=2`, ReLU | |
| `init_mode` | `"default"` | PyTorch default (Kaiming uniform with $a = \sqrt{5}$). Not explicit Kaiming He. |
| Post-processor | softplus | ensures $\hat{y} > 0$ |
| `decision_grad_backend` | `analytic` | closed form for healthcare LP |
| Fairness types | {mad, dp, atkinson, bias_parity} | 4 per $\alpha$ |
| $\alpha_{\text{fair}}$ | {0.5, 2.0} | 2 alphas per fairness type |
| Seeds | **[11, 22, 33]** | 3 seeds |
| Methods | FPTO, FDFL-Scal, FPLG, PCGrad, MGDA, SAA, WDRO | same 7 as v1 |
| Total cells | 4 × 2 = 8 | |
| Total runs | 8 × 3 = 24 | one run per cell per seed |

## 4. Hyperparameters — Variant B

Realistic training: validation + LR schedule + early stopping. Uses the
disjoint seed set [44, 55]. Same cohort, fairness types, alphas, and
methods as Variant A. **Differences from A**:

| HP | Variant A | Variant B | Why |
|---|---|---|---|
| `val_fraction` | 0.0 | **0.2** | 20% of train becomes val for early stopping |
| `steps_per_lambda` | 70 | **150** | longer to give early stop room to engage |
| `lr_decay` | 5e-4 | **5e-3** | 10× stronger reciprocal decay; LR halves at step ~200 |
| `lr_warmup_steps` | 0 | **5** | linear warmup over first 5 steps |
| `eval_val_every_k_steps` | (off) | **10** | evaluate val metrics every 10 training steps |
| `early_stop_metric` | — | `"val_regret"` | selection metric for snapshot |
| `early_stop_min_steps` | — | **20** | hard floor; don't stop before step 20 |
| Seeds | [11, 22, 33] | **[44, 55]** | disjoint from A |
| Total cells | 8 | 8 | |
| Total runs | 24 | 16 | 2 seeds instead of 3 |

### 4.1 Early stopping mechanism

The training loop (`src/fair_dfl/training/loop.py::train_single_stage`)
has been extended with a per-stage val-based early-stop mechanism:

1. **Val check every K steps** (starting from `early_stop_min_steps`):
   - Evaluate the predictor on the validation split via
     `eval_split_medical`
   - Record `val_regret`, `val_fairness`, `val_pred_mse` as a new
     `stage_type="val_check"` row in `iter_logs.csv`
2. **Best snapshot tracking**:
   - If the current val metric is better than the running best, save
     a deep copy of `predictor.state_dict()`, `optimizer.state_dict()`,
     and `dual_lambda`
   - Record `best_step = t + 1`
3. **At end of stage**:
   - If a best snapshot exists, restore it (`load_state_dict` + dual
     lambda restore)
   - Record `early_stop_step` in the stage row
4. **Scope**: per-lambda-stage. Methods with `continuation=True` (like
   FPLG) restart the next stage from the restored best state. This means
   each stage converges to its best validation point before passing the
   state to the next stage — a cleaner semantic than "restart at final
   step of previous stage".

**Caveats**:
- FAMO is explicitly unsupported (the handler has internal state $\xi,
  \ell_{\text{prev}}$ that we'd need to snapshot; not implemented).
  We assert fail-fast in the loop if the grid ever includes FAMO.
- Medical task only (the helper `eval_split_medical` is task-specific).
- The mechanism is **gated off** when `val_fraction=0` or
  `eval_val_every_k_steps=0`, so Variant A runs are unaffected.

### 4.2 Choice of `lr_decay=5e-3` instead of cosine

The existing schedule infrastructure is a reciprocal decay $\text{lr}_t
= \text{lr}_0 / (1 + \text{lr\_decay} \cdot t)$. At `lr_decay=5e-3`:

- Step 0: $\text{lr}_0 = 10^{-3}$
- Step 50: $\text{lr}_0 / 1.25 = 8 \times 10^{-4}$
- Step 200: $\text{lr}_0 / 2 = 5 \times 10^{-4}$
- Step 500: $\text{lr}_0 / 3.5 = 2.86 \times 10^{-4}$
- Step 1000: $\text{lr}_0 / 6 = 1.67 \times 10^{-4}$

This is not a textbook cosine schedule, but combined with early
stopping it accomplishes the same goal: the model converges faster to
the val plateau, and early stopping catches it before the LR-induced
noise reintroduces training-set overfit.

## 5. Seed assignment and train/test splits

v1 used one fixed `split_seed = 2` across all 5 training seeds
`[11, 22, 33, 44, 55]`. The model saw the same train/test partition with
different initialisations. For v2 we **couple** the split seed to the
training seed:

| Variant | Training seeds | Split seeds | Train/test partitions |
|---|---|---|---|
| A | 11, 22, 33 | 11, 22, 33 | 3 different |
| B | 44, 55 | 44, 55 | 2 different |

**Implications**:

- Cross-seed mean regret is now a mean over different test sets, not
  the same test set with different initialisations. This is more
  conservative and captures split-level variability, which the v1
  numbers did not.
- `test_regret_normalized` is still per-seed (each seed's denominator
  is the oracle objective on its own test set), so comparing normalised
  regret across seeds is valid.
- Variant A and Variant B use disjoint seed sets, so they operate on
  non-overlapping train/test partitions. This is **more honest** about
  generalisation (we're not testing the same data twice) but means
  direct variant comparison is a distribution-level comparison, not a
  paired one.

## 6. Cross-fairness comparison methodology

v1 reported per-cell winners, which was hard to read across the 8 cells
because `mad ~ 22`, `dp ~ 1.7`, `atkinson ~ 0.007`, and `bias_parity ~
0.3` all have different units. v2 adds two unit-free aggregations:

### 6.1 Rank aggregation (primary)

For each of the 8 (fairness_type, alpha) cells independently, rank the
7 methods 1–7 on `test_fairness` at their Pareto-best operating point
(using `scipy.stats.rankdata(method='average')` to handle ties). Then:

- **Mean rank per method per $\alpha$**: average rank over the 4
  fairness types at that alpha. Range $[1, 7]$.
- **Std rank per method per $\alpha$**: consistency across fairness
  types. Low std = method is consistently in the same rank.
- **Borda count** (equivalent): sum of ranks across cells; a method
  with the lowest sum "wins".

This is the standard non-parametric way to compare methods across
incommensurable scales in fair-ML benchmarks (see Demšar 2006).

### 6.2 Ratio normalisation (secondary)

For each (fairness_type, alpha, seed), normalise fairness by two
baselines:

- **FPTO λ=0** (unconstrained fitted reference): `rel_fair_vs_fpto =
  test_fair(method) / test_fair(fpto_lam_0)` — same seed. Range $(0,
  \infty)$; <1 is better than FPTO, >1 is worse.
- **SAA λ=0** (feature-ignoring ceiling): `rel_fair_vs_saa =
  test_fair(method) / test_fair(saa_lam_0)` — same seed. SAA predicts
  the train mean constant, so this measures how much of the unfairness
  is coming from using features at all.

**Pair seeds, then average**: ratios are computed per seed then averaged
across seeds (not average-then-divide).

### 6.3 Constrained fairness selection

v1's "best Pareto per method" selected the lambda with the lowest
`test_regret_normalized`. This biases against fairness-heavy methods
(they always show up at $\lambda = 0$, their least-fair point).

For the cross-fairness comparison, we add a second selector:

$$
\lambda^\star_{\text{fair}}(m) = \underset{\lambda}{\arg\min}\; \text{fair}_\lambda(m)
\quad \text{s.t.} \quad \tilde{R}_\lambda(m) \le (1 + \text{slack}) \cdot \tilde{R}_0(\text{fpto})
$$

with `slack = 0.10`. This picks the fairest operating point that the
method can reach without losing more than 10% regret vs unconstrained
FPTO. Used in the fairness-rank heatmaps.

## 7. Execution

### 7.1 Dependencies

Same as v1: Python 3.13, `numpy`, `pandas`, `torch`, `cvxpy`, `MOSEK`
(or `CLARABEL`/`SCS` fallback). No new dependencies for v2.

### 7.2 Reproduce

```bash
# Training-loop modification + v2 infrastructure must be checked out
# (commit TBD on fair-dfl/empirical-followup)

# Variant A: 3 seeds, no val, ~20 min
python -m experiments.advisor_review.run_healthcare_v2_variant_a

# Variant B: 2 seeds, val + early stop, ~22 min
python -m experiments.advisor_review.run_healthcare_v2_variant_b

# Post-process: aggregates + cross-fairness + plots
python -m experiments.advisor_review.analyze_healthcare_v2
python -m experiments.advisor_review.plot_healthcare_v2
```

### 7.3 Directory layout (post-run)

```
results/advisor_review/healthcare_followup_v2/
├── setup.md                        # ← this file
├── notes_v2.md                     # findings with actual numbers
├── variant_a/
│   ├── grand_summary.csv
│   ├── health.json
│   ├── grid_summary.json
│   └── {fairness}/alpha_{α}/seed_{s}/
│       ├── stage_results.csv
│       ├── iter_logs.csv
│       ├── config.json
│       ├── best_pareto_per_method.csv
│       └── summary_by_method_lambda.csv
├── variant_b/
│   └── (same layout as variant_a)
├── cross_fairness/
│   ├── rank_matrix.csv
│   ├── ratio_matrix.csv
│   └── plots/
│       ├── rank_heatmap.png
│       ├── ratio_heatmap.png
│       └── variant_a_vs_b.png
└── plots/
    ├── variant_a/pareto_{ft}_alpha_{α}.png    # train + test overlay
    ├── variant_a/pareto_grid_8cells.png
    ├── variant_b/pareto_{ft}_alpha_{α}.png
    ├── variant_b/pareto_grid_8cells.png
    └── variant_b/early_stop_histogram.png
```

## 8. Key differences from v1

1. **Budget**: 0.35 → 0.30 (tighter)
2. **Lambdas**: [0, 0.5, 1, 5] → [0, 0.5, 1, 2] (drop λ=5 divergence)
3. **LR**: 5e-4 → 1e-3 (2×)
4. **Seeds**: 5 fixed-split → (3 varied-split A) + (2 varied-split B)
5. **Variant B** is entirely new: val + LR decay + early stopping
6. **Cross-fairness aggregation** is new (rank + ratio)
7. **Train metrics** are now surfaced alongside test (visually overlaid
   in Pareto plots)
8. **Early-stopping mechanism** is a new, backward-compatible feature of
   the training loop (`eval_val_every_k_steps` config key)
