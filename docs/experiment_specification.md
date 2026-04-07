# Experiment Specification: Decision-Focused Multi-Task Fair Learning

**Document purpose:** Reference for writing the Experiments and Results sections of the INFORMS JoC paper. Contains all parameters, design choices, mathematical definitions, and rationale.

---

## 1. Experimental Design Overview

We evaluate our framework on two experiments:

1. **Healthcare Resource Allocation** (real data, Obermeyer et al. 2019)
2. **Synthetic Multi-Dimensional Knapsack** (controlled unfairness)

Both share the same method comparison and metric framework. The healthcare experiment tests on a real-world dataset with inherent group disparities; the knapsack experiment isolates unfairness mechanisms under controlled conditions across three severity levels.

### 1.1 Optimization Framework

Both experiments follow the same **predict-then-decide** pipeline. A parametric model f_theta: X -> R^n predicts item benefits y-hat from features x. The predicted benefits are passed to a downstream optimization solver to produce a decision d:

```
d*(y-hat) = argmax_{d in D} U(d, y-hat)
```

where D is the feasible set (knapsack constraints or budget constraint) and U is the alpha-fair utility function. The true quality of the decision is evaluated using the **true** benefits y:

```
obj_true  = U(d*(y),    y)   [oracle decision on true benefits]
obj_pred  = U(d*(y-hat), y)  [predicted decision evaluated on true benefits]
```

Regret = obj_true - obj_pred >= 0.

### 1.2 Alpha-Fairness Utility

Both experiments use the **alpha-fairness** family of utility functions (Mo & Walrand 2000):

```
U_alpha(u) = sum_i u_i^(1-alpha) / (1-alpha),   alpha != 1
           = sum_i log(u_i),                      alpha = 1
```

where u_i = r_i * d_i is the utility of item/patient i (benefit r_i times allocation d_i).

- **alpha -> 0:** approaches sum of utilities (utilitarian / max total welfare)
- **alpha = 0.5:** moderate inequality aversion
- **alpha = 1:** proportional fairness (Nash bargaining)
- **alpha = 2:** strong inequality aversion (closer to Rawlsian max-min)
- **alpha -> inf:** max-min fairness (Rawlsian)

We test alpha in {0.5, 2.0} to bracket the policy-relevant range between efficiency-leaning and equity-leaning objectives.

### 1.3 Methods Compared

| Method | Category | Objectives Used | Lambda | Description |
|--------|----------|----------------|--------|-------------|
| FPTO (lambda=0) = **PTO** | Two-stage | pred | 0 | Predict-then-optimize, no fairness |
| FPTO (lambda=0.5,1,5) | Two-stage | pred + fair | 0.5, 1, 5 | Fair predict-then-optimize |
| SAA | Data-driven baseline | -- | -- | Sample average approximation, no ML |
| WDRO | Data-driven baseline | pred (robust) | -- | Wasserstein DRO |
| FDFL-Scal (lambda=0) = **DFL** | Integrated | dec + pred | 0 | Decision-focused learning, no fairness |
| FDFL-Scal (lambda=0.5,1,5) | Integrated | dec + pred + fair | 0.5, 1, 5 | Scalarized fair DFL |
| FDFL-PCGrad | Integrated + MOO | dec + pred + fair | -- | PCGrad gradient combination |
| FDFL-MGDA | Integrated + MOO | dec + pred + fair | -- | MGDA gradient combination |
| FDFL-CAGrad | Integrated + MOO | dec + pred + fair | -- | CAGrad gradient combination |

**Method hierarchy:**
- **FPTO** (Fair Predict-Then-Optimize): trains on prediction MSE + lambda * fairness penalty. At lambda=0 this is standard PTO.
- **FDFL-Scal** (Fair Decision-Focused Learning, Scalarized): trains on decision regret + prediction MSE (via PLG schedule) + lambda * fairness. At lambda=0 this is DFL. This is the proposed FPLG method.
- **FDFL-{MOO}**: same as FDFL-Scal but uses a MOO gradient combination rule (PCGrad / MGDA / CAGrad) instead of scalarization. Lambda is not a separate parameter — the MOO handler determines the gradient blend.

**Key insight:** FPTO and FDFL-Scal sweep lambda in {0, 0.5, 1, 5}, producing a Pareto approximation. At lambda=0, FPTO = PTO and FDFL-Scal = DFL. MOO methods do not require lambda — they determine objective weighting algorithmically from gradient geometry.

### 1.4 Three Objectives

| Objective | Symbol | Formula | Differentiable? |
|-----------|--------|---------|----------------|
| Prediction MSE | L_pred | mean((f(x) - y)^2) | Yes (analytic) |
| Prediction fairness | L_fair | MAD of per-group MSEs (see §6) | Yes (smooth approx.) |
| Decision regret | L_dec | max(obj_true - obj_pred, 0) | Via SPSA or KKT |

---

## 2. Experiment 1: Healthcare Resource Allocation

### 2.1 Dataset

- **Source:** Obermeyer et al. (2019), *Science* — "Dissecting racial bias in an algorithm used to manage the health of populations"
- **Population:** 48,784 patients (we use the full dataset, not the 5,000-sample subset from prior work)
- **Data file:** `data/data_processed.csv`
- **Train/test split:** 50% / 50% (no validation split)
- **Split seeds:** data_seed=42, split_seed=2

### 2.2 Target Variable

The model predicts **total healthcare costs in the target year** (`cost_t`), expressed in dollars. This is the standard proxy used in the original Obermeyer et al. (2019) paper: higher predicted cost -> higher resource allocation priority.

> Note: The original paper documents that using cost as a proxy for health need introduces racial bias because, at equal health need, Black patients historically had lower healthcare costs due to reduced access. This is the core unfairness we address.

### 2.3 Features

Total feature count: **~130 features** across the following groups:

| Feature group | Examples | Count (approx.) |
|---------------|----------|-----------------|
| Demographics | `dem_age`, `dem_female`, all `dem_*` (excl. race) | ~10 |
| Comorbidities | `gagne_sum_tm1`, Elixhauser indices, Romano indices | ~40 |
| Prior costs | All `cost_*` (excl. `cost_t`, `cost_avoidable_t`) | ~15 |
| Lab results | Test counts, abnormal flags (`*_tests_tm1`, `*-low/high/normal_tm1`) | ~60 |
| Medications | Lasix-related features (`lasix_*`) | ~5 |

Race is explicitly excluded from the feature set — fairness is imposed through the fairness loss, not by including group membership.

### 2.4 Group Definition

- **Groups:** Defined by the `race` column (binarized: Black vs. non-Black in the main experiment, matching Obermeyer et al.)
- **Group sizes (approximate, 50% train split):** ~11% Black, ~89% non-Black, consistent with the original cohort
- **Fairness target:** Equal prediction MSE across racial groups (disparity reduction, not demographic parity in allocations)

### 2.5 Decision Problem

**Objective:** Allocate healthcare program enrollment resources across patients to maximize alpha-fair welfare:

```
max_{d >= 0}  U_alpha(r * d)     [element-wise product r_i * d_i]
subject to:   sum_i d_i <= B
```

where:
- r_i = predicted cost for patient i (the model output)
- d_i = fraction of resources allocated to patient i
- B = total budget = `budget_rho * sum(min(cost_t, cap))`, with `budget_rho = 0.35`

**Budget rationale:** `budget_rho = 0.35` means the program can serve approximately 35% of the total need (sum of capped costs). This reflects a realistic resource-constrained setting where the algorithm must prioritize.

**Decision mode:** Group-level allocation (`decision_mode="group"`). Within each racial group, resources are allocated proportionally to individual predicted costs. The group-level aggregate allocation is the decision variable. This reduces the problem size from n_patients to n_groups while preserving the alpha-fair welfare objective.

### 2.6 Decision Gradient

- **Backend:** Analytic (KKT-based implicit differentiation)
- The group-level allocation problem has a closed-form KKT structure that enables exact computation of d∂L_dec/∂y-hat via the envelope theorem
- No finite differences or SPSA required for healthcare

### 2.7 Experimental Grid

| Factor | Values | Count |
|--------|--------|-------|
| Method | 7 methods (13 method-lambda configs) | 7 |
| Alpha | {0.5, 2.0} | 2 |
| Hidden dim | 64 | 1 |
| Seeds | {11, 22, 33, 44, 55} | 5 |
| **Total runs** | 7 × 2 × 1 × 5 | **70** |

Each run with lambda-sweep methods (FPTO, FDFL-Scal) produces 4 stage rows (one per lambda). Other methods produce 1 stage row.

---

## 3. Experiment 2: Synthetic Multi-Dimensional Knapsack

### 3.1 Problem Formulation

**Optimization problem (group-level alpha-fairness, matching healthcare):**

The knapsack uses the same two-level group alpha-fair objective as the healthcare experiment. Items are partitioned into groups (group 0 = majority, group 1 = minority). The decision solver maximizes:

```
Inner level:  G_k = sum_{i in group k} (r_i * d_i)^{1-alpha} / (1-alpha)   [per-group alpha-fair utility]
Outer level:  Phi = sum_k G_k^{1-alpha} / (1-alpha)                        [alpha-fair across groups]
```

subject to:
```
A d <= b          [resource constraints]
d_i >= 1e-6       [positivity for power objectives]
```

where:
- r in R^7: true item benefits (unknown at decision time; predicted by the model)
- d in R^7: fractional allocation per item
- A in R^{3 x 7}: resource consumption matrix, A_ij ~ Uniform[0.5, 1.5]
- b in R^3: resource capacities, b_k = 0.5 * sum_j A_kj (budget_tightness = 0.5)
- Both inner and outer levels use the **same alpha** (matching healthcare)

**Group-level objective:** The two-level structure ensures that the solver accounts for group welfare balance. Without it, the objective is purely item-level and does not "see" group membership — making the decision gradient group-blind and preventing FDFL methods from differentiating themselves from FPTO (see §3.1.1).

**CVXPY implementation:** For alpha < 1, the two-level objective is expressed directly in CVXPY (DCP-compliant: power of affine is concave, power of concave with 0 < p < 1 is concave). For alpha >= 1, item-level atoms are used in the solver (see §3.1.1 for why). The numpy evaluator always uses the full two-level formula for all alpha values.

**Parameter:** `decision_mode="group"` (default). Set to `"item"` for the original item-level formulation.

#### 3.1.1 Note: Alpha=2 Collapse

At alpha=2 specifically, the two-level group formulation is **mathematically equivalent** to the item-level formulation. This is because the inner-level reciprocal (G_k = 1/sum(1/y_i)) and the outer-level power (G_k^{-1}) cancel exactly:

```
Phi = -sum_k 1/G_k = -sum_k sum_{i in k} 1/y_i = -sum_i 1/y_i = item-level objective
```

This means:
- **alpha=0.5:** Group objective is genuinely different from item-level → creates method differentiation
- **alpha=2.0:** Group objective collapses to item-level → methods behave similarly to the item-level case

For alpha >= 1, the two-level objective is also not DCP-expressible in CVXPY (the outer power of a reciprocal-sum expression violates DCP rules). Since it's equivalent to item-level at alpha=2 anyway, the solver uses item-level atoms.

**Why 7 items:** Small enough for SPSA to be computationally fast (~96 solver calls/step), but large enough with 2 groups (4/3 or 5/2 split) to create meaningful group allocation tradeoffs.

### 3.2 Data Generation

**Feature-to-benefit mapping** (following PyEPO's nonlinear generation scheme):

```
x_i ~ N(0, I_5)                          [5-dim features per sample]
W_d ~ N(0, 1/d),  d in {1,2},  W_d in R^{5 x 7}   [weight matrices, fixed per seed]
raw = x @ W_1 + x^2 @ W_2                [degree-2 polynomial]
y_raw = raw + group_shift + noise
y = softplus(y_raw) + 0.05               [ensures y > 0]
```

where:
- `group_shift[i] = +group_bias` if item i in group 0, `-group_bias` if in group 1
- `noise[i] ~ N(0, noise_std_lo)` for group 0 items, `N(0, noise_std_hi)` for group 1
- `softplus(x) = log(1 + exp(x))`, the +0.05 floor ensures strict positivity

**Weight matrix scaling** (1/d): higher-degree terms have smaller coefficients, making the cubic terms a smaller perturbation on the linear signal. This creates a nonlinear mapping that an MLP with standard initialization can learn but not trivially fit.

**Training/test samples:** 200 train, 80 test, no validation split.

### 3.3 Unfairness Mechanisms

Three independent sources of unfairness are combined:

1. **Group bias** (`group_bias`): Systematic mean shift in benefit values.
   - Majority group (group 0): benefits shifted up by +group_bias → over-represented in allocation
   - Minority group (group 1): benefits shifted down by -group_bias → under-represented
   - Represents a structural advantage for the majority group

2. **Noise heterogeneity** (`noise_std_lo`, `noise_std_hi`): Differential prediction difficulty.
   - Majority group: low noise → easier to predict accurately
   - Minority group: high noise → harder to predict → more prediction error
   - Represents data quality disparities (less historical data for minority group)

3. **Group size imbalance** (`group_ratio`): Fraction of items in the majority group.
   - Larger majority → minority items have fewer representatives
   - Exacerbates fairness violations because MAD is sensitive to group size asymmetry

### 3.4 Three Unfairness Levels

| Level | group_bias | noise_std_lo | noise_std_hi | group_ratio | Group split (n=7) | Primary mechanism |
|-------|-----------|-------------|-------------|------------|-------------------|-------------------|
| **Mild** | 0.1 | 0.1 | 0.2 | 0.5 | 4 / 3 | Small bias, small noise gap |
| **Medium** | 0.3 | 0.1 | 0.5 | 0.5 | 4 / 3 | Larger bias + noise gap |
| **High** | 0.3 | 0.1 | 0.5 | 0.67 | 5 / 2 | Bias + noise gap + group imbalance |

The progression isolates unfairness mechanisms:
- **Mild → Medium:** Bias triples (0.1→0.3), noise gap widens (0.2→0.5), group sizes unchanged
- **Medium → High:** Adds group size imbalance (4/3→5/2) while keeping bias and noise fixed

The simultaneous increase is intentional: in practice, data quality, representation, and historical advantage are correlated. Disentangling each mechanism is a secondary analysis.

### 3.5 Decision Gradient

The alpha-fair knapsack objective does not satisfy CVXPY's DPP requirements, preventing the use of differentiable solvers (cvxpylayers, fold-opt). We use **SPSA** (Simultaneous Perturbation Stochastic Approximation):

**SPSA gradient estimator:**
For each sample b in the batch:
1. Sample Rademacher vector: Delta_b in {-1, +1}^n (each element iid)
2. Solve: d_+ = d*(y-hat_b + eps * Delta_b), d_- = d*(y-hat_b - eps * Delta_b)
3. Compute: regret_+ = max(obj_true_b - U(d_+, y_b), 0), regret_- = max(obj_true_b - U(d_-, y_b), 0)
4. Gradient estimate: g_b = (regret_+ - regret_-) / (2 * eps * Delta_b)   [element-wise]

**Parameters:** eps = 5e-3 (perturbation magnitude), n_dirs = 1 (single random direction per step).

**Unbiasedness:** The SPSA estimator is an unbiased estimator of the gradient of the expected regret w.r.t. the prediction, provided the regret function is smooth in y-hat. The Rademacher perturbation satisfies E[1/Delta_j^2] = 1, so the estimator correctly scales the gradient.

**Computational cost:**
- SPSA: bsz × (1 + 2 × n_dirs) solver calls per step = **96 calls/step** (batch=32, n_dirs=1)
- Element-wise FD: bsz × (1 + 2 × dim) = 32 × 15 = **480 calls/step** (dim=7)
- **Speedup: ~5x** vs. finite differences, independent of problem dimension

**Solver chain (in order of preference):**
1. MOSEK (requires license; most accurate for power objectives)
2. CLARABEL (no license; modern power-cone solver, CVXPY default since v1.3)
3. ECOS (fast but can return inaccurate solutions for tight-budget power objectives)
4. SCS (last resort; less accurate but always terminates)

**Reference:** Spall, J.C. (1992). "Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation." IEEE Trans. Automatic Control, 37(3), 332-341.

### 3.6 Experimental Grid

| Factor | Values | Count |
|--------|--------|-------|
| Method | 7 methods (13 method-lambda configs) | 7 |
| Alpha | {0.5, 2.0} | 2 |
| Unfairness | {mild, medium, high} | 3 |
| Seeds | {11, 22, 33, 44, 55} | 5 |
| **Total runs** | 7 × 2 × 3 × 5 | **210** |

---

## 4. Training Configuration (Shared)

### 4.1 Prediction Model

All methods (except SAA) use the same MLP architecture:

| Parameter | Healthcare | Knapsack |
|-----------|-----------|----------|
| Architecture | MLP | MLP |
| Input dim | ~130 features | 5 features |
| Hidden dim | 64 | 64 |
| Hidden layers | 2 | 2 |
| Activation | ReLU | ReLU |
| Dropout | 0.0 | 0.0 |
| Batch normalization | No | No |
| Weight initialization | PyTorch default (Kaiming uniform) | same |
| Output activation | Softplus | Softplus |
| Output dim | n_groups (2) | n_items (7) |

**Output activation (Softplus):** ensures predictions are strictly positive before being passed to the alpha-fair optimizer, which requires positive benefits.

### 4.2 Optimization

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | Adam | default betas (0.9, 0.999) |
| Learning rate | 5×10⁻⁴ | |
| LR schedule | Inverse time decay | lr(t) = lr₀ / (1 + lr_decay × t) |
| LR decay coefficient | 5×10⁻⁴ | |
| Batch size (HC) | Full batch (~24,000) | Global budget constraint requires all patients |
| Batch size (KN, non-FDFL) | Full batch (200) | FPTO/SAA/WDRO use full training set |
| Batch size (KN, FDFL) | 32 | Mini-batch limits SPSA solver calls per step |
| Gradient clip norm | 10,000 | Safety; rarely active |
| Exploding threshold | 1,000,000 | Step skipped if gradient norm exceeds this |
| Steps per lambda (HC) | 70 | Per lambda stage; 4 stages → 280 total steps |
| Steps per lambda (KN) | 200 | Per lambda stage; 4 stages → 800 total steps |

**Lambda values:** {0.0, 0.5, 1.0, 5.0}. These were chosen to span a practical range:
- 0.0: fairness-blind baseline
- 0.5, 1.0: moderate fairness emphasis
- 5.0: strong fairness emphasis (where accuracy-fairness tradeoff is visible)
Values beyond 5.0 showed excessive accuracy degradation in pilot experiments.

### 4.3 Prediction Weight Schedule (alpha_t)

For FDFL methods, prediction and decision gradients are combined using a decaying prediction weight:

```
alpha(t) = max(alpha_min, alpha0 / sqrt(t + 1))
alpha0 = 1.0,  alpha_min = 0.0
```

This is the PLG (Prediction-then-Learning-to-Generalize) schedule from [CITE]. Early training emphasizes prediction accuracy (warm start); as t increases, the weight shifts toward decision quality. At convergence, alpha(t) → 0 and the method is effectively pure decision-focused learning.

### 4.4 Warmstart (FDFL Methods)

All FDFL methods use a warmstart period:

- **Duration:** 25% of steps_per_lambda (10 steps for knapsack, 18 steps for healthcare)
- **During warmstart:** method runs as FPTO (prediction + fairness gradients only, no decision gradient)
- **After warmstart:** switches to full method specification (adds decision gradient)
- **Rationale:** Starting from a reasonable predictor prevents the SPSA gradient estimator from operating in a degenerate high-regret region where gradient estimates are noisy and uninformative

### 4.5 Lambda Sweep (Continuation Mode)

For FPTO and FDFL-Scal, training proceeds through lambda stages [0.0, 0.5, 1.0, 5.0] sequentially:

- **Continuation:** model weights carry over from one lambda stage to the next
- **Effect:** traces an approximate Pareto frontier (regret vs. fairness) more efficiently than independent restarts from scratch
- **Interpretation:** each stage is a fine-tuning of the previous model with a stronger fairness constraint

MOO methods (PCGrad, MGDA, CAGrad) do not use a lambda sweep — they handle all three objectives jointly within a single training run.

---

## 5. Method Implementation Details

### 5.1 SAA (Sample Average Approximation)

- **No neural network training.** The "prediction" for every test point is the **training set mean**: y-hat(x) = mean(y_train)
- Represents a no-ML baseline: what happens if we use population statistics rather than individual predictions?
- The decision is then made by solving the optimization problem with these constant predictions
- Steps = 0 (no gradient descent)

### 5.2 WDRO (Wasserstein Distributionally Robust Optimization)

Our WDRO implementation trains the same MLP with an importance-weighted MSE objective:

```
L_WDRO = mean_i(L_i) + epsilon * std_i(L_i)
```

where L_i = (f(x_i) - y_i)^2 is the per-sample squared error. This is equivalent to optimizing the mean plus epsilon times the standard deviation of the loss distribution, which provides a first-order approximation to the Wasserstein DRO ball objective.

- **DRO epsilon:** 0.1 (controls robustness radius; chosen as a round number in [0.05, 0.2] range standard in the DRO literature)
- Upweights high-loss samples, providing worst-case robustness without explicit group labels
- Does **not** use a fairness loss term — robustness is distributional, not group-specific

### 5.3 PCGrad (Yu et al. 2020, NeurIPS)

Resolves gradient conflicts by projecting each objective's gradient onto the normal plane of conflicting gradients:

For each pair (i, j) where cos(g_i, g_j) < 0:
```
g_i <- g_i - (g_i · g_j / ||g_j||²) * g_j
```
Final direction: sum of projected gradients. The projection is applied to the "running" gradient while using the original g_i for conflict detection (preventing cascading modifications).

Reference: Yu, T., et al. (2020). "Gradient Surgery for Multi-Task Learning." NeurIPS.

### 5.4 MGDA (Sener & Koltun 2018, NeurIPS)

Finds the minimum-norm update direction that does not increase any objective:

```
min_{w in Delta^m}  ||sum_i w_i * g_i||²
```

where Delta^m is the m-dimensional simplex. This is solved as a QP:

```
min_{w}  w^T M w,   M_ij = g_i · g_j
subject to: sum w_i = 1, w_i >= 0
```

Solved via SLSQP (scipy), maxiter=200, ftol=1e-12. Falls back to equal weights (w_i = 1/m) if optimization fails.

Reference: Sener, O. & Koltun, V. (2018). "Multi-Task Learning as Multi-Objective Optimization." NeurIPS.

### 5.5 CAGrad (Liu et al. 2021, ICLR)

Computes a conflict-averse update direction that stays close to the mean gradient while reducing the worst-case objective increase:

```
g0 = (1/m) sum_i g_i                         [mean gradient]
min_{w in Delta^m}  w^T b + c * ||g0|| * sqrt(w^T M w)
    where b_i = g_i · g0,  M_ij = g_i · g_j
Final direction: g0 + G^T w
```

- **Conflict-aversion coefficient c = 0.5** (default from original paper)
  - c = 0: reduces to mean gradient (no conflict aversion)
  - c → ∞: approaches MGDA behavior (maximum conflict aversion)
- Solved via SLSQP, same settings as MGDA

Reference: Liu, B., et al. (2021). "Conflict-Averse Gradient Descent for Multi-task Learning." ICLR.

---

## 6. Metrics

### 6.1 Normalized Decision Regret (primary)

```
regret_i = max(obj_true_i - obj_pred_i, 0)
regret_normalized_i = regret_i / (|obj_true_i| + |obj_pred_i| + 1e-8)
Regret = mean_i(regret_normalized_i)
```

where:
- obj_true_i = U_alpha(d*(y_i), y_i): objective achieved by the **oracle** decision (using true benefits)
- obj_pred_i = U_alpha(d*(y-hat_i), y_i): objective achieved by the **predicted** decision, evaluated on **true** benefits

The normalization `|obj_true| + |obj_pred|` accounts for scale variation across alpha values and problem instances (alpha-fair objectives can be negative for alpha > 1). Lower is better.

> Note: We also track `regret_normalized_true = regret / |obj_true|` (normalized by oracle only) as a secondary metric. The main tables use the symmetric normalization above.

### 6.2 Prediction Fairness Violation (primary)

Mean Absolute Deviation (MAD) of per-group prediction MSEs:

```
mse_g = mean_{i in group g} (f(x_i) - y_i)²
gap_g = mse_g - mean_g(mse_g)
L_fair = mean_g(sqrt(gap_g² + 1e-6))
```

The `sqrt(gap² + eps)` smoothing makes the loss differentiable near zero (subgradient at 0 is 0 without smoothing, which blocks gradient flow). Lower is better.

**Why MAD:** Symmetric (agnostic to which group has higher MSE), smooth, and additive over groups. Alternative: max-group-gap is non-smooth; Atkinson index requires a reference distribution.

### 6.3 Prediction MSE (secondary)

```
MSE = mean_i ||f(x_i) - y_i||²
```

Reported to diagnose accuracy-fairness-regret tradeoffs. A method can achieve good regret by learning decision-relevant features at the cost of overall MSE.

### 6.4 Statistical Reporting

- All metrics reported as **mean ± standard deviation** over 5 random seeds
- Seeds control: training data shuffle order, model weight initialization, and (for knapsack) SPSA perturbation directions
- The knapsack data generation (A matrix, W matrices) is fixed by `data_seed=42` across all method runs, so all methods see the same underlying problem instance
- No formal significance testing is performed; the 5-seed std provides a visual uncertainty indicator

---

## 7. Computational Details

### 7.1 Parallelization Strategy

| Worker | Experiment | Grid | Est. Colab Time |
|--------|-----------|------|-----------|
| A | Healthcare (all) | 7 methods × 2 alpha × 5 seeds = 70 runs | ~30-45 min |
| B | Knapsack alpha=0.5 | 7 methods × 3 uf × 5 seeds = 105 runs | ~60-90 min |
| C | Knapsack alpha=2.0 | 7 methods × 3 uf × 5 seeds = 105 runs | ~60-90 min |

Healthcare uses analytic gradients (fast). Knapsack FDFL methods use SPSA with batch=32; baseline methods (FPTO, SAA, WDRO) use full batch (200 samples, no solver calls during training).

### 7.2 Computational Environment

| Component | Value |
|-----------|-------|
| Platform | Google Colab (free tier / Pro) |
| GPU | T4 (16 GB VRAM) — healthcare uses GPU; knapsack uses CPU (CVXPY not GPU-compatible) |
| Python | 3.12 |
| PyTorch | 2.x |
| CVXPY | 1.4+ |
| Solvers | MOSEK 10.x (licensed), CLARABEL (bundled with CVXPY 1.4+), ECOS, SCS |

**Note on GPU usage:** The knapsack task is CPU-bound due to CVXPY solver calls. Using GPU for PyTorch forward/backward passes provides minimal speedup over the solver bottleneck. Healthcare can use GPU since it has analytic gradients.

### 7.3 Checkpoint and Resume

- Each run saves to: `results/final/{experiment}/{method}/alpha_{a}_uf_{u}/seed_{s}/stage_results.csv`
- The checkpoint check (`_done()`) reads for the presence of `stage_results.csv`
- Re-running any notebook cell automatically skips completed runs
- Results directories are versioned (`knapsack_v2/`) when problem parameters change, so old results are not overwritten

### 7.4 Reproducibility

| Component | Seed / Value |
|-----------|-------------|
| Knapsack data generation (A, W) | `data_seed = 42` |
| Healthcare train/test split | `data_seed = 42`, `split_seed = 2` |
| Model weight initialization | PyTorch default; controlled by training seed |
| Training seeds | {11, 22, 33, 44, 55} |
| SPSA perturbation directions | Controlled by training seed via `np.random.default_rng(seed)` |
| Framework | PyTorch 2.x, CVXPY 1.4+ |

All random state is seeded before each run. The knapsack constraint matrix A and weight matrices W are generated once from `data_seed` and shared across all methods, ensuring methods face identical problem instances.

---

## 8. Design Rationale

### 8.1 Why No Validation Split?

1. All lambda values are reported as separate rows in the results table — no model selection is performed using a held-out set
2. Training uses fixed steps (no early stopping criterion)
3. MOO methods do not have a lambda hyperparameter to tune
4. Maximizes training data (especially for the 48,784-patient healthcare dataset)

### 8.2 Why 5 Seeds for All Methods?

Standard for INFORMS JoC computational experiments. The SPSA-based decision gradient makes FDFL methods fast enough to run all 5 seeds within Colab session limits. Previous implementations using element-wise finite differences limited FDFL to 2 seeds due to solver cost.

### 8.3 Why Alpha in {0.5, 2.0}?

- **alpha=0.5:** Moderate inequality aversion — favors efficiency with some equity concern. Corresponds to a concave but relatively flat utility function.
- **alpha=2.0:** Strong inequality aversion — closer to Rawlsian max-min fairness. Methods that ignore fairness perform notably worse at alpha=2.0 since the objective penalizes low allocations more severely.
- These two values are sufficient to show how the fairness-accuracy tradeoff sharpens with stronger equity preferences.

### 8.4 Why MAD Fairness (Not Gap or Atkinson)?

- **MAD** is symmetric (no reference group), differentiable (with smoothing), and decomposable across groups
- **Group gap** (max_group - min_group MSE) is non-smooth, sensitive to a single extreme group, and directional
- **Atkinson index** requires specifying an inequality-aversion parameter (an additional hyperparameter)
- MAD is the simplest fairness metric that is both group-agnostic and smoothly differentiable

### 8.5 Why Continuation Mode for Lambda Sweep?

- Sequential lambda training (0 → 0.5 → 1 → 5) traces the Pareto frontier while reusing computation
- Each stage fine-tunes the previous model — a natural warm start for increasing fairness pressure
- Independent restarts from scratch are more expensive and produce disconnected Pareto points
- Continuation is a standard approach in multi-objective scalarization (e.g., penalty parameter methods in constrained optimization)

### 8.6 Why SPSA Instead of Finite Differences?

Element-wise finite differences require 2 × n_items solver calls per sample per gradient step. At n_items=7, batch=32:
- FD cost: 32 × 15 = 480 calls/step
- SPSA cost: 32 × 3 = 96 calls/step (~5× faster)

More importantly, SPSA cost is **dimension-independent** — scaling to larger n_items does not increase the per-step cost beyond the solver's own scaling.

The eps=5e-3 perturbation magnitude was chosen to balance:
- Too small: solver output is effectively constant → gradient estimate is pure noise
- Too large: the gradient estimate is biased (finite-difference bias of O(eps²))
- 5e-3 is in the range where the alpha-fair objective changes measurably but the perturbation is small relative to typical prediction magnitudes (~O(1) after softplus)

---

## 9. Output Structure

### 9.1 Table Layout

**Table 1 (Healthcare Results):**
- Rows: 13 method-lambda configurations
- Columns: Norm. Regret (↓), Pred. Fairness Viol. (↓), Pred. MSE (↓)
- Panels: alpha=0.5 and alpha=2.0
- Format: mean ± std (bold = best per column per panel)

**Table 2 (Knapsack Results):**
- Same column structure as Table 1
- Panels: Mild / Medium / High unfairness level
- Separate table for each alpha value (two tables total)

**Table 3 (Method Summary):**
- Average rank across all experimental conditions
- Columns: Avg. Regret Rank, Avg. Fairness Rank, Avg. MSE Rank

### 9.2 Figure Layout

1. **Pareto front (Healthcare):** 2D scatter of regret vs. fairness violation. Each point = one method-lambda combination. Subplots per alpha. FDFL methods should trace a frontier extending to better fairness than FPTO at similar regret.
2. **Regret by unfairness level (Knapsack):** Grouped bar chart. Methods on x-axis, bars colored by unfairness level. Shows how regret gaps grow as unfairness increases.
3. **Gradient conflict heatmap (Knapsack):** Pairwise cosine similarities among (grad_dec, grad_pred, grad_fair) for each MOO method and unfairness level. Motivates the need for MOO gradient combination.

---

## 10. File Inventory

| File | Purpose |
|------|---------|
| `experiments/run_healthcare_final.py` | Local healthcare experiment runner |
| `experiments/run_knapsack_final.py` | Local knapsack experiment runner |
| `experiments/colab_runner.py` | Shared module for Colab workers (all overrides) |
| `experiments/configs.py` | Method registry and training defaults |
| `experiments/generate_tables.py` | LaTeX table generator |
| `experiments/generate_figures.py` | Publication figure generator |
| `notebooks/worker_a_healthcare.ipynb` | Colab worker: healthcare |
| `notebooks/worker_b_knapsack_alpha05.ipynb` | Colab worker: knapsack alpha=0.5 |
| `notebooks/worker_c_knapsack_alpha20.ipynb` | Colab worker: knapsack alpha=2.0 |
| `notebooks/results.ipynb` | Aggregation + analysis + figure/table export |
| `src/fair_dfl/tasks/md_knapsack.py` | Knapsack task: data generation, solver, SPSA |
| `src/fair_dfl/tasks/medical_resource_allocation.py` | Healthcare task: data loading, KKT gradient |
| `src/fair_dfl/decision/strategies/spsa.py` | SPSA decision gradient strategy |
| `src/fair_dfl/algorithms/mo_handler.py` | PCGrad / MGDA / CAGrad implementations |
| `src/fair_dfl/training/loop.py` | Unified training loop (skip_regret, warmstart) |
| `src/fair_dfl/losses.py` | MSE loss, MAD fairness loss (with smoothing) |
