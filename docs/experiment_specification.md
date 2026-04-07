# Experiment Specification: Decision-Focused Multi-Task Fair Learning

**Document purpose:** Reference for writing the Experiments and Results sections of the INFORMS JoC paper. Contains all parameters, design choices, and rationale.

---

## 1. Experimental Design Overview

We evaluate our framework on two experiments:

1. **Healthcare Resource Allocation** (real data, Obermeyer et al. 2019)
2. **Synthetic Multi-Dimensional Knapsack** (controlled unfairness)

Both share the same method comparison and metric framework. The healthcare experiment tests on a real-world dataset with inherent group disparities; the knapsack experiment isolates unfairness mechanisms under controlled conditions.

### Methods Compared

| Method | Category | Objectives Used | Lambda | Description |
|--------|----------|----------------|--------|-------------|
| FPTO (lambda=0) = **PTO** | Two-stage | pred | 0 | Predict-then-optimize (no fairness) |
| FPTO (lambda=0.5,1,5) | Two-stage | pred + fair | 0.5, 1, 5 | Fair predict-then-optimize |
| SAA | Data-driven | -- | -- | Sample average approximation (no ML model) |
| WDRO | Data-driven | pred (robust) | -- | Wasserstein distributionally robust optimization |
| FDFL-Scal (lambda=0) = **DFL** | Integrated | dec + pred | 0 | Decision-focused learning (no fairness) |
| FDFL-Scal (lambda=0.5,1,5) | Integrated | dec + pred + fair | 0.5, 1, 5 | Scalarized fair decision-focused learning |
| FDFL-PCGrad | Integrated + MOO | dec + pred + fair | -- | PCGrad multi-objective handler |
| FDFL-MGDA | Integrated + MOO | dec + pred + fair | -- | MGDA multi-objective handler |
| FDFL-CAGrad | Integrated + MOO | dec + pred + fair | -- | CAGrad multi-objective handler |

**Key insight:** FPTO and FDFL-Scal each sweep lambda in {0, 0.5, 1, 5}. At lambda=0, FPTO reduces to PTO (no fairness), and FDFL-Scal reduces to DFL (no prediction fairness). MOO methods do not require a lambda parameter — they determine objective weighting algorithmically.

### Three Objectives

1. **Prediction loss** (pred): Mean squared error between predicted and true values
2. **Prediction fairness** (fair): Mean absolute deviation (MAD) of group-wise MSE — measures disparity in prediction quality across groups
3. **Decision regret** (dec): Normalized difference between the optimal objective value (using true parameters) and the achieved objective value (using predicted parameters)

---

## 2. Experiment 1: Healthcare Resource Allocation

### Dataset

- **Source:** Obermeyer et al. (2019) — algorithmic bias in healthcare risk prediction
- **Full dataset:** 48,784 patients (not the 5,000-sample subset from prior work)
- **Data file:** `data/data_processed.csv`
- **Split:** 50% train / 50% test (no validation split)
- **Split seeds:** data_seed=42, split_seed=2

### Features

The prediction model uses the following feature groups:
- **Demographics:** All `dem_*` columns (excluding race)
- **Comorbidities:** `gagne_sum_tm1`, Elixhauser indices (`*_elixhauser_tm1`), Romano indices (`*_romano_tm1`)
- **Prior costs:** All `cost_*` columns (excluding `cost_t` and `cost_avoidable_t`)
- **Lab results:** Test counts (`*_tests_tm1`), abnormal flags (`*-low_tm1`, `*-high_tm1`, `*-normal_tm1`)
- **Medications:** Lasix-related features (`lasix_*`)

### Group Definition

- Groups are defined by the **race** column in the dataset
- Fairness is measured as disparity in prediction MSE between racial groups

### Decision Problem

- **Objective:** Allocate healthcare resources to patients to maximize alpha-fairness welfare
- **Decision mode:** Group-level allocation (`decision_mode="group"`)
- **Budget:** Dynamic, computed as `budget = budget_rho * sum(capped_costs)`, with `budget_rho = 0.35`
- **Alpha-fairness values:** alpha in {0.5, 2.0}
  - alpha=0.5: moderate inequality aversion (closer to utilitarian)
  - alpha=2.0: strong inequality aversion (closer to Rawlsian)

### Decision Gradient

- **Backend:** Analytic (KKT-based implicit differentiation)
- Healthcare has a closed-form solution structure, enabling exact gradient computation

### Experimental Grid

| Factor | Values | Count |
|--------|--------|-------|
| Method | 7 methods (13 method-lambda configs) | 7 |
| Alpha | {0.5, 2.0} | 2 |
| Hidden dim | 64 | 1 |
| Seeds | {11, 22, 33, 44, 55} | 5 |
| **Total runs** | 7 x 2 x 1 x 5 | **70** |

Each run with lambda-sweep methods (FPTO, FDFL-Scal) produces 4 stage rows (one per lambda value). Other methods produce 1 stage row.

---

## 3. Experiment 2: Synthetic Multi-Dimensional Knapsack

### Problem Formulation

- **Type:** Multi-dimensional knapsack with alpha-fairness utility
- **Items:** 20 items with group labels
- **Constraints:** 3 knapsack constraints
- **Constraint matrix:** A ~ Uniform[0.5, 1.5], shape (3, 20)
- **Budget:** b = 0.3 * A.sum(axis=1) (budget_tightness = 0.3, tight budget forces selective allocation)

### Data Generation

**Feature-to-cost mapping** (PyEPO-style nonlinear):
- Features: z_i ~ N(0, I_5) for each sample
- True costs: degree-3 polynomial mapping
  - For each degree d in {1, 2, 3}: W_d ~ N(0, 1/d), shape (5, 20)
  - raw = sum_{d=1}^{3} z^d @ W_d
- Positive transform: y = softplus(raw + group_shift + noise) + 0.05

**Unfairness mechanisms** (three parameters):

1. **Group bias** (`group_bias`): Additive mean shift
   - Group 0 items: shifted by +group_bias
   - Group 1 items: shifted by -group_bias

2. **Noise heterogeneity** (`noise_std_lo`, `noise_std_hi`): Differential noise variance
   - Group 0 items: noise ~ N(0, noise_std_lo)
   - Group 1 items: noise ~ N(0, noise_std_hi)

3. **Group size imbalance** (`group_ratio`): Fraction of items in group 0
   - n_group0 = round(group_ratio * n_items)
   - n_group1 = n_items - n_group0

### Three Unfairness Levels

| Level | group_bias | noise_std_lo | noise_std_hi | group_ratio | Group split | Unfairness source |
|-------|-----------|-------------|-------------|------------|------------|-------------------|
| **Mild** | 0.1 | 0.1 | 0.3 | 0.5 | 10/10 | Small bias, small noise gap |
| **Medium** | 0.3 | 0.1 | 0.8 | 0.6 | 12/8 | Larger bias + noise gap + mild imbalance |
| **High** | 0.5 | 0.1 | 1.0 | 0.75 | 15/5 | Large bias + noise gap + strong imbalance |

The progression increases all three unfairness mechanisms simultaneously:
- Mild -> Medium: increases mean shift (0.1 -> 0.3), noise gap (0.3 -> 0.8), adds mild group imbalance (50/50 -> 60/40)
- Medium -> High: further increases bias (0.3 -> 0.5), noise gap (0.8 -> 1.0), and strong imbalance (60/40 -> 75/25)

### Decision Gradient

- **Backend:** SPSA (Simultaneous Perturbation Stochastic Approximation)
- The multi-dimensional knapsack solver (CVXPY with ECOS/SCS) does not provide analytic gradients
- SPSA perturbs all prediction dimensions simultaneously with a random Rademacher vector, requiring only 2 solver calls per sample per direction (vs 2 * dim for element-wise finite differences)
- Perturbation step size: eps = 5e-3, n_dirs = 1
- Reference: Spall (1992), IEEE Trans. Automatic Control

### Experimental Grid

| Factor | Values | Count |
|--------|--------|-------|
| Method | 7 methods (13 method-lambda configs) | 7 |
| Alpha | {0.5, 2.0} | 2 |
| Unfairness | {mild, medium, high} | 3 |
| Seeds | {11, 22, 33, 44, 55} | 5 |
| **Total runs** | 7 x 2 x 3 x 5 | **210** |

---

## 4. Training Configuration (Shared)

### Prediction Model

| Parameter | Healthcare | Knapsack |
|-----------|-----------|----------|
| Architecture | MLP | MLP |
| Hidden dimensions | 64 | 64 |
| Hidden layers | 2 | 2 |
| Activation | ReLU | ReLU |
| Dropout | 0.0 | 0.0 |
| Batch normalization | No | No |
| Weight initialization | Default (PyTorch) | Default |
| Post-processing | Softplus (healthcare) | Softplus |

### Optimization

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 0.0005 | |
| LR decay | 0.0005 | lr(t) = lr / (1 + lr_decay * t) |
| Batch size (HC) | Full batch | Required for global budget constraint |
| Batch size (KN, non-FDFL) | Full batch (-1) | n_train = 200 |
| Batch size (KN, FDFL) | 32 | Mini-batch limits SPSA solver calls |
| Gradient clip norm | 10,000 | Safety threshold |
| Exploding threshold | 1,000,000 | Step skipped if norm exceeds this |
| Steps per lambda (HC) | 70 | Per lambda stage |
| Steps per lambda (KN) | 40 | Per lambda stage |

### Prediction Weight Schedule (alpha_t)

For methods using both prediction and decision gradients (PLG-style), the prediction loss weight decays over training:

- Schedule type: Inverse square root
- Formula: `alpha(t) = max(alpha_min, alpha0 / sqrt(t + 1))`
- alpha0 = 1.0, alpha_min = 0.0
- Early training emphasizes prediction accuracy; later training emphasizes decision quality

### Warmstart

- Applied to: FPLG-based methods (FDFL-Scal, FDFL-PCGrad, FDFL-MGDA, FDFL-CAGrad)
- Warmstart fraction: 25% of steps_per_lambda
- During warmstart: method runs as FPTO (prediction + fairness only, no decision gradient)
- After warmstart: switches to full method specification
- Rationale: Provides a reasonable initial predictor before introducing decision gradients

### Lambda Sweep (Continuation Mode)

For FPTO and FDFL-Scal, training proceeds through lambda stages [0.0, 0.5, 1.0, 5.0] sequentially with **continuation** — the model state carries over from one lambda to the next, rather than reinitializing.

---

## 5. Method Implementation Details

### SAA (Sample Average Approximation)

- **No training:** The "prediction" is simply the mean of training labels
- Steps = 0; evaluation uses `pred(x) = mean(y_train)` for all inputs
- Serves as a no-ML baseline: what happens if we use population statistics directly

### WDRO (Wasserstein Distributionally Robust Optimization)

- Trains the same MLP predictor as other methods, but with importance-weighted MSE
- **DRO epsilon:** 0.1
- Per-sample loss: `L_i = (pred_i - true_i)^2`
- DRO weights: `w_i = 1 + epsilon * (L_i - mean(L)) / std(L)`
- Effective loss: `mean(L) + epsilon * std(L)`
- Gradients are multiplied by the DRO weights, upweighting high-loss samples
- This provides a form of worst-case robustness without explicit group information

### PCGrad (Yu et al. 2020)

- For each objective gradient g_i, check for conflict with all other g_j
- Conflict detected when: `cos(g_i, g_j) < 0`
- If conflict: project g_i onto the normal plane of g_j
  - `g_i = g_i - (g_i . g_j / ||g_j||^2) * g_j`
- Uses original g_i for conflict detection but applies projection to running state
- Final direction: sum of all projected gradients

### MGDA (Sener & Koltun 2018)

- Finds the minimum-norm point in the convex hull of objective gradients
- Solves: `min_{lambda in simplex} ||sum_i lambda_i * g_i||^2`
- Equivalent QP: `min lambda^T M lambda` where `M_ij = g_i . g_j`
- Solved via SLSQP (scipy), maxiter=200, ftol=1e-12
- Falls back to equal weights if optimization fails

### CAGrad (Liu et al. ICLR 2021)

- Computes mean gradient g_0 = (1/m) sum g_i
- Finds weights w that solve:
  - `min_{w in simplex} w^T b + c * ||g_0|| * sqrt(w^T M w)`
  - where b_i = g_i . g_0 and M = G G^T
- **Conflict-aversion coefficient c = 0.5** (default)
  - c=0: reduces to mean gradient
  - c large: approaches MGDA-like behavior
- Final direction: g_0 + G^T w
- Solved via SLSQP, same settings as MGDA

### SPSA Decision Gradient (Knapsack)

Decision-focused methods on the knapsack problem require differentiating through the CVXPY solver. Since the alpha-fair knapsack objective violates CVXPY's DPP requirements (preventing cvxpylayers / fold-opt), we use SPSA:

- **Perturbation:** Rademacher vector Delta where each Delta_j ~ {-1, +1} uniformly
- **Gradient estimate:** For each sample b:
  - Solve knapsack with pred[b] + eps * Delta[b] -> obj_plus
  - Solve knapsack with pred[b] - eps * Delta[b] -> obj_minus
  - grad[b, j] = (regret_plus - regret_minus) / (2 * eps * Delta[b, j])
- **Cost per step:** bsz * (1 + 2 * n_dirs) solver calls = 96 with batch=32, n_dirs=1
- **vs element-wise FD:** Would require bsz * (1 + 2 * dim) = 1,312 calls (dim=20)

This achieves a ~14x reduction in solver calls per step while providing an unbiased gradient estimator. The SPSA cost is independent of problem dimension, making it crucial for scaling to n_items=20.

---

## 6. Metrics

### Primary Metrics (reported in tables)

1. **Normalized Decision Regret** (lower is better)
   - `regret = max(obj_true - obj_pred, 0) / (|obj_true| + |obj_pred| + epsilon)`
   - Where obj_true uses the oracle decision (true parameters), obj_pred uses the predicted parameters
   - Averaged across test samples

2. **Prediction Fairness Violation** (lower is better)
   - Type: Mean Absolute Deviation (MAD) of group MSEs
   - For each group g: `mse_g = mean((pred_g - true_g)^2)`
   - Deviations: `gap_g = mse_g - mean(mse_all_groups)`
   - Loss: `mean_g(sqrt(gap_g^2 + smoothing))` where smoothing = 1e-6
   - Measures disparity in prediction quality across groups

3. **Prediction MSE** (lower is better)
   - Standard mean squared error between predictions and true values

### Gradient Diagnostics (knapsack only)

- **Cosine similarity** between gradient pairs:
  - cos(grad_dec, grad_pred) — decision vs prediction alignment
  - cos(grad_dec, grad_fair) — decision vs fairness alignment
  - cos(grad_pred, grad_fair) — prediction vs fairness alignment
- Averaged over training iterations; captures gradient conflict dynamics

---

## 7. Computational Details

### Parallelization Strategy

Experiments are split across 3 parallel Colab workers:

| Worker | Experiment | Grid | Est. Time |
|--------|-----------|------|-----------|
| A | Healthcare (all) | 7 methods x 2 alpha x 5 seeds = 70 runs | ~30-45 min |
| B | Knapsack alpha=0.5 | 7 methods x 3 uf x 5 seeds = 105 runs | ~30-60 min |
| C | Knapsack alpha=2.0 | 7 methods x 3 uf x 5 seeds = 105 runs | ~30-60 min |

Healthcare uses analytic gradients (fast). Knapsack uses SPSA decision gradients with mini-batch, split across 2 workers by alpha.

### Checkpoint/Resume

- Each run saves results to: `{experiment}/{method}/alpha_{a}_hd_{h}/seed_{s}/stage_results.csv`
- Existing results are automatically skipped on re-run
- No experiment needs to be rerun for analysis changes

### Reproducibility

| Parameter | Value |
|-----------|-------|
| Data seed | 42 |
| Split seed | 2 (healthcare) |
| Training seeds | {11, 22, 33, 44, 55} |
| Model init seed | 13,579 + seed * 101 + 1 |
| Framework | PyTorch |
| Solver | CVXPY (ECOS primary, SCS fallback) |

---

## 8. Design Rationale

### Why no validation split?

1. All lambda values are reported as separate table rows (no model selection via validation)
2. Fixed training steps — no early stopping
3. MOO methods do not use lambda at all
4. Maximizes training data for the full Obermeyer dataset

### Why 5 seeds for all methods?

Standard for INFORMS JoC computational experiments. Provides reliable mean +/- std estimates while keeping total compute manageable. SPSA-based decision gradients make FDFL methods fast enough to run all 5 seeds (previously limited to 2 seeds with element-wise finite differences).

### Why alpha in {0.5, 2.0}?

- alpha=0.5: Moderate inequality aversion — favors efficiency with some equity concern
- alpha=2.0: Strong inequality aversion — closer to max-min fairness (Rawlsian)
- These bracket the most policy-relevant range for resource allocation

### Why MAD fairness (not gap or Atkinson)?

- MAD (mean absolute deviation of group MSEs) is smooth, differentiable, and symmetric
- Agnostic to which group has higher MSE (unlike "gap" which is directional)
- The smoothing parameter (1e-6) ensures differentiability near zero

### Why continuation mode for lambda sweep?

- Lambda stages run sequentially with model state carried over
- This traces an approximate Pareto frontier more efficiently than independent restarts
- Each stage refines the previous solution rather than starting from scratch

---

## 9. Expected Result Structure

### Table Layout

**Table 1 (Healthcare):**
- Rows: 13 method-lambda configurations
- Columns: Norm. Regret, Pred. Fair. Viol., Pred. MSE
- Panels: alpha=0.5 and alpha=2.0
- Best values bolded per column per panel

**Table 2 (Knapsack):**
- Rows: 13 method-lambda configurations
- Columns: Same 3 metrics
- Panels: Mild / Medium / High unfairness
- Separate table per alpha value

**Table 3 (Summary):**
- Average method rank across all conditions
- Columns: Regret rank, Fairness rank, MSE rank (per experiment)

### Figure Layout

1. **Pareto front (Healthcare):** 2D scatter of regret vs fairness, each point is a method-lambda combo, subplots per alpha
2. **Regret by unfairness (Knapsack):** Grouped bar chart, methods on x-axis, bars colored by unfairness level
3. **Gradient conflict (Knapsack):** Heatmap of pairwise gradient cosine similarities, rows = MOO methods, columns = unfairness levels

---

## 10. File Inventory

| File | Purpose |
|------|---------|
| `experiments/run_healthcare_final.py` | Healthcare experiment runner |
| `experiments/run_knapsack_final.py` | Knapsack experiment runner |
| `experiments/colab_runner.py` | Shared module for Colab workers |
| `experiments/configs.py` | Method registry and defaults |
| `experiments/generate_tables.py` | LaTeX table generator |
| `experiments/generate_figures.py` | Publication figure generator |
| `notebooks/worker_a_healthcare.ipynb` | Colab worker: healthcare |
| `notebooks/worker_b_knapsack_alpha05.ipynb` | Colab worker: knapsack alpha=0.5 |
| `notebooks/worker_c_knapsack_alpha20.ipynb` | Colab worker: knapsack alpha=2.0 |
| `notebooks/results.ipynb` | Aggregation + analysis + export |
| `src/fair_dfl/tasks/md_knapsack.py` | Knapsack task implementation |
| `src/fair_dfl/tasks/medical_resource_allocation.py` | Healthcare task implementation |
| `src/fair_dfl/algorithms/mo_handler.py` | MOO gradient handlers |
| `src/fair_dfl/training/loop.py` | Unified training loop |
| `src/fair_dfl/losses.py` | Loss functions (MSE, fairness) |
