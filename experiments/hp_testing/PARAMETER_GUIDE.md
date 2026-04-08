# Comprehensive Parameter Guide

This guide covers every tunable parameter in the experiment framework — what it does, where it flows, and when to change it.

---

## How Parameters Flow

```
Notebook/Script Config
    |
    v
TASK_OVERRIDES ──> task_cfg dict ──> MultiDimKnapsackTask (data generation + solver)
TRAIN_OVERRIDES ──> train_cfg dict ──> train_single_stage() (training loop)
    |                                       |
    |                 method_spec (from ALL_METHOD_CONFIGS)
    |                       |
    v                       v
    run_experiment_unified(cfg, method_configs)
```

**Three layers:**
1. **Task config** — controls the optimization problem (what data is generated, how solver works)
2. **Train config** — controls learning (optimizer, steps, gradients, model architecture)
3. **Method spec** — controls which objectives are active (decision, prediction, fairness)

---

## 1. Task-Level Parameters (`TASK_OVERRIDES` / `task_cfg`)

These define the knapsack problem structure and data generation.

### Problem Structure

| Parameter | Default | What It Does | When to Change |
|-----------|---------|-------------|----------------|
| `n_items` | 7 | Number of items in the knapsack | More items = harder problem, more solver time |
| `n_constraints` | 3 | Number of resource constraints (rows of A matrix) | More constraints = tighter feasible region |
| `budget_tightness` | 0.5 | Fraction of total capacity: `b = tightness * A.sum(axis=1)` | **Lower = tighter budget = fewer items selected = more threshold effects**. Try 0.25-0.3 for sharper differentiation |
| `n_features` | 5 | Dimension of input features x | More features = more complex mapping to learn |
| `poly_degree` | 2 | Polynomial feature mapping degree. Benefits = sum_{d=1}^{deg} x^d @ W_d | Higher = more nonlinear signal. 2 is usually sufficient |
| `scenario` | "alpha_fair" | `"alpha_fair"` or `"lp"`. Controls solver objective | LP has vertex solutions (threshold effects), alpha-fair is smooth |
| `alpha_fair` | 2.0 | Alpha parameter in alpha-fair utility: sum(d_i^{1-alpha}/(1-alpha)) | Lower alpha (0.1-0.5) = closer to linear. Higher (2-5) = more concave, more spreading. **All values produce smooth allocations** |
| `decision_mode` | "group" | `"group"` = two-level group alpha-fair (like healthcare). `"item"` = item-level | Use "group" for fairness experiments |
| `fairness_type` | "mad" | Fairness metric: "mad" (mean absolute deviation of group MSEs) | Only "mad" is currently used |

### Data Generation

| Parameter | Default | What It Does | When to Change |
|-----------|---------|-------------|----------------|
| `n_samples_train` | 200 | Training set size | **Critical for learning.** With 5 features, poly_degree=2, need at least 200-500 for MLP to generalize |
| `n_samples_val` | 0 | Validation set size (0 = no validation) | Set >0 if you want early stopping |
| `n_samples_test` | 80 | Test set size | 80-200 is fine for evaluation |
| `data_seed` | 42 | RNG seed for data generation (A matrix, W matrices, group assignment) | Fixed at 42. Same problem structure across seeds |
| `group_bias` | per UF level | Constant shift: group_0 gets +bias, group_1 gets -bias | **Larger bias = groups have more different true benefits.** Need bias > noise_std for signal to be learnable |
| `noise_std_lo` | per UF level | Noise std for majority group (group_0) | Keep low (0.05-0.1) |
| `noise_std_hi` | per UF level | Noise std for minority group (group_1) | **This is the #1 knob for difficulty.** If noise_std_hi > signal, MLP cannot learn and all methods collapse. Try 0.1-0.3 |
| `group_ratio` | per UF level | Fraction of items in majority group (group_0). E.g., 0.5 = equal split, 0.67 = 2:1 | Higher ratio = smaller minority = harder fairness problem |

### Key Insight: Signal-to-Noise Ratio

The benefit for item j is:
```
y_j = softplus(x @ W_1[:,j] + x^2 @ W_2[:,j] + group_shift_j + noise_j)
```

Where `noise_j ~ N(0, noise_std_j^2)`.

**If noise_std_hi > ~0.3, the MLP predicts the mean (= SAA).** This is because the Bayes-optimal MSE predictor IS the conditional mean, and when noise dominates, the conditional mean is close to the unconditional mean.

Rule of thumb: `group_bias / noise_std_hi > 1.5` for learnable signal.

---

## 2. Training Parameters (`TRAIN_OVERRIDES` / `train_cfg`)

These control how the model learns.

### Core Training

| Parameter | Default | What It Does | When to Change |
|-----------|---------|-------------|----------------|
| `steps_per_lambda` | 70 | Number of gradient updates per lambda stage | **Critical.** With `pred_weight_mode="schedule"`, alpha_t decays as 1/sqrt(t+1). At step 70, alpha_t=0.12. At step 200, alpha_t=0.07. **More steps + schedule = less prediction signal = worse learning**. Either use fewer steps OR use `fixed1` |
| `lr` | 0.0005 | Initial learning rate | With SGD: 0.0005-0.005. With Adam: 0.001-0.01 |
| `lr_decay` | 0.0005 | LR decay: lr_t = lr / (1 + lr_decay * t) | 0 = constant LR. 0.0005 = gentle decay |
| `batch_size` | -1 | Samples per gradient update. -1 = full batch | Full batch for baselines (FPTO, SAA, WDRO). Mini-batch for FDFL (controlled by fdfl_batch_size) |
| `fdfl_batch_size` | 32 | Mini-batch size for decision-gradient methods | Each sample needs solver calls. Smaller = fewer solver calls but noisier. 16-64 is typical |
| `optimizer` | "sgd" | "sgd" or "adam" | Adam can help with noisy gradients but may overfit on small datasets |

### Prediction Weight / Alpha Schedule (THE KEY PARAMETER)

| Parameter | Default | What It Does | When to Change |
|-----------|---------|-------------|----------------|
| `pred_weight_mode` | per method | Controls how prediction gradient is weighted over time | See detailed explanation below |
| `alpha_schedule` | `{type: "inv_sqrt", alpha0: 1.0, alpha_min: 0.0}` | Schedule config for mode="schedule" | Only active when pred_weight_mode="schedule" |

#### `pred_weight_mode` — Detailed Explanation

This parameter determines how much the MSE prediction loss contributes to the gradient at each step. It's set per-method in `ALL_METHOD_CONFIGS` and can be overridden.

**Three modes:**

##### `"fixed1"` (constant, no decay)
```
pred_weight(t) = 1.0   for all t
```
- MSE gradient has full strength at every step
- Used by: FPTO, SAA, WDRO (baselines that don't use decision gradients)
- **Best for learning** when you need the model to actually reduce MSE
- **Recommended override for FDFL methods** when you see null differentiation

##### `"schedule"` (decaying via alpha_t)
```
pred_weight(t) = alpha_schedule(t)
```
With default `inv_sqrt`: `alpha(t) = 1.0 / sqrt(t + 1)`

| Step | alpha_t | Prediction gradient weight |
|------|---------|--------------------------|
| 0 | 1.000 | Full strength |
| 10 | 0.302 | 30% |
| 20 | 0.218 | 22% |
| 40 | 0.156 | 16% |
| 70 | 0.119 | 12% |
| 100 | 0.100 | 10% |
| 200 | 0.071 | 7% |
| 400 | 0.050 | 5% |

- Used by: FPLG, PLG, all MOO methods (PCGrad, MGDA, CAGrad, FAMO)
- The theoretical motivation: as training progresses, shift from "learn good predictions" to "learn good decisions"
- **THE PROBLEM**: on small datasets with high noise, the model needs ALL 200 steps to learn predictions. By step 100, prediction gradient is nearly off, and the model stops improving MSE
- **THIS IS WHY 70 STEPS WORKS BUT 200 DOESN'T** — at 70 steps, alpha_t never drops below 0.12

##### `"zero"` (no prediction gradient)
```
pred_weight(t) = 0.0   for all t
```
- Pure decision-focused: only decision regret gradient (and optionally fairness)
- Used by: DFL, FDFL (the "no prediction" variants)
- Rarely useful in practice — the decision gradient is too noisy alone

#### Available Schedule Types

Set via `alpha_schedule.type`:

| Schedule | Formula | Params | Behavior |
|----------|---------|--------|----------|
| `inv_sqrt` | `alpha0 / sqrt(t+1)` | `alpha0`, `alpha_min` | Slow decay, default |
| `sigmoid_decay` | `alpha_min + (alpha_max - alpha_min) / (1 + exp((t - midpoint) / temperature))` | `alpha_max`, `alpha_min`, `midpoint`, `temperature` | S-curve: fast near midpoint |
| `paper_decay` | `(1 + exp((t-c)/temperature))^{-kappa}` | `c`, `kappa`, `temperature` | Threshold-like drop at step c |
| `poly_decay` | `alpha_min + (alpha_max - alpha_min) * (1 - t/horizon)^power` | `alpha_max`, `alpha_min`, `power`, `horizon` | Polynomial decay to 0 at horizon |
| `constant` | `value` | `value` | No decay |

**Practical recommendation:** Either use `"fixed1"` or use `constant` schedule with value=0.5. The `inv_sqrt` schedule is too aggressive for small datasets.

### Decision Gradient

| Parameter | Default | What It Does | When to Change |
|-----------|---------|-------------|----------------|
| `decision_grad_backend` | "spsa" | How decision gradients are estimated | "spsa" = 2 solver calls/sample (fast, noisy). "finite_diff" = 2*n_items solver calls/sample (accurate, slow). "spo_plus" = LP only |
| `decision_grad_fd_eps` | 1e-3 | Perturbation size for finite_diff | Larger = less variance but more bias. 1e-3 is good default |

### Model Architecture

Set inside `train_cfg["model"]` dict. In TRAIN_OVERRIDES, these are auto-routed:

| Parameter | Default | What It Does | When to Change |
|-----------|---------|-------------|----------------|
| `hidden_dim` | 64 | Hidden layer width | 32-128 typical. Larger = more capacity but risk overfitting on small n |
| `n_layers` | 2 | Number of hidden layers | 1-3. More layers rarely helps on small synthetic data |
| `activation` | "relu" | Activation function | "relu", "tanh", "gelu" |
| `dropout` | 0.0 | Dropout rate | 0.0-0.3. Can help with overfitting |
| `batch_norm` | false | Batch normalization | Usually not needed for synthetic |
| `arch` | "mlp" | Architecture type | "mlp" is the only tested option for knapsack |
| `init_mode` | "default" | Weight initialization | "default" = PyTorch default |

### Gradient Control

| Parameter | Default | What It Does | When to Change |
|-----------|---------|-------------|----------------|
| `grad_clip_norm` | 10000.0 | Max gradient L2 norm (clip if exceeded) | Lower (1.0-10.0) if you see NaN/exploding. 10000 = effectively no clipping |
| `explode_threshold` | 1e12 | If loss exceeds this, skip the step | Safety net for numerical issues |
| `fairness_smoothing` | 1e-6 | Epsilon added to fairness denominator | Prevents division by zero in MAD |
| `log_every` | 5 | Log metrics every N steps | Lower for debugging, higher for speed |

### Lambda (Fairness Penalty Weight)

| Parameter | Default | What It Does | When to Change |
|-----------|---------|-------------|----------------|
| `lambdas` | [0.0, 0.5, 1.0, 5.0] | Lambda values to sweep (one stage per lambda) | The Pareto frontier parameter. lambda=0 = no fairness. lambda=5 = heavy fairness |
| `lambda_train` | 0.0 | Which lambda value to use during training (vs all being evaluated) | Usually 0.0 — fairness is tested at each lambda stage |

### Advanced (MOO handlers)

| Parameter | Default | What It Does |
|-----------|---------|-------------|
| `mo_method` | None | Multi-objective handler: "pcgrad", "mgda", "cagrad", "famo", "weighted_sum" |
| `mo_weights` | {} | Weights for weighted_sum handler: {"decision_regret": 0.33, "pred_loss": 0.33, "pred_fairness": 0.33} |
| `mo_cagrad_c` | 0.5 | CAGrad convergence parameter (0-1) |
| `mo_famo_gamma` | 1e-3 | FAMO task-balancing gamma |
| `mo_famo_w_lr` | 0.025 | FAMO weight learning rate |
| `warmstart_fraction` | 0.25 | For FPLG/PLG: fraction of steps to run as FPTO before enabling decision gradient |
| `continuation` | per method | Whether to continue from previous lambda stage's weights |

---

## 3. Method Specs (`ALL_METHOD_CONFIGS`)

Each method declares which objectives it uses. You normally don't edit these unless experimenting with new variants.

| Method | pred_weight_mode | use_dec | use_pred | use_fair | Notes |
|--------|-----------------|---------|----------|----------|-------|
| **FPTO** | fixed1 | No | Yes | Yes | Predict + fairness only. The "PTO" baseline |
| **DFL** | zero | Yes | No | No | Pure decision-focused, no prediction signal |
| **FDFL** | zero | Yes | No | Yes | Decision + fairness, no prediction signal |
| **PLG** | schedule | Yes | Yes | No | Prediction-guided DFL (alpha_t schedule) |
| **FPLG** | schedule | Yes | Yes | Yes | Full 3-objective (this is "FDFL-Scal" in experiments) |
| **SAA** | fixed1 | No | Yes | No | Predicts the training mean. No training loop |
| **WDRO** | fixed1 | No | Yes | No | Distributionally robust, no decision gradient |
| **PCGrad** | schedule | Yes | Yes | Yes | PCGrad MOO handler |
| **MGDA** | schedule | Yes | Yes | Yes | MGDA MOO handler |
| **CAGrad** | schedule | Yes | Yes | Yes | CAGrad MOO handler |
| **FAMO** | schedule | Yes | Yes | Yes | FAMO auto-weighting |

**Key observation:** ALL FDFL/MOO methods use `schedule` mode by default. This means their prediction gradient decays with alpha_t. This is the root cause of the convergence issue at high step counts.

---

## 4. Recommended HP Configurations

### Configuration A: Debug (verify learning works)
```python
TRAIN_CFG = {
    "steps_per_lambda": 70,
    "lr": 0.0005,
    "pred_weight_mode_override": None,  # use method defaults
    "decision_grad_backend": "spsa",
    "fdfl_batch_size": 32,
}
```
**Why:** 70 steps keeps alpha_t > 0.12 throughout. Fast to run.

### Configuration B: Fix the schedule (recommended)
```python
TRAIN_CFG = {
    "steps_per_lambda": 200,
    "lr": 0.005,
    "pred_weight_mode_override": "fixed1",  # constant prediction weight
    "decision_grad_backend": "spsa",
    "fdfl_batch_size": 32,
}
```
**Why:** fixed1 prevents alpha_t decay. 200 steps with full prediction signal means the model actually learns. Higher LR compensates for the now-constant weighting.

### Configuration C: Low noise + more data
```python
TASK_CFG = {
    "n_samples_train": 400,
    "budget_tightness": 0.3,  # tighter
    ...
}
UF_CONFIGS = {
    "medium": {"group_bias": 0.5, "noise_std_lo": 0.05, "noise_std_hi": 0.2, "group_ratio": 0.6},
}
TRAIN_CFG = {
    "steps_per_lambda": 200,
    "lr": 0.005,
    "pred_weight_mode_override": "fixed1",
    ...
}
```
**Why:** SNR > 2.5 (group_bias 0.5 / noise 0.2), 400 samples for generalization, tight budget for competition.

---

## 5. CLI Quick Reference

```bash
# Quick test (3 methods, 1 seed, ~30s)
python experiments/hp_testing/run_hp_test.py

# Test fixed1 override
python experiments/hp_testing/run_hp_test.py --pred-weight fixed1 --steps 200

# Test higher LR
python experiments/hp_testing/run_hp_test.py --lr 0.005

# Reduce noise globally
python experiments/hp_testing/run_hp_test.py --noise-hi 0.15

# More training data
python experiments/hp_testing/run_hp_test.py --n-train 400

# Full grid (all methods, all configs, 3 seeds — ~30min on CPU)
python experiments/hp_testing/run_hp_test.py --full

# Combine everything
python experiments/hp_testing/run_hp_test.py --pred-weight fixed1 --steps 200 --lr 0.005 --n-train 400 --noise-hi 0.2

# Specific methods only
python experiments/hp_testing/run_hp_test.py --methods FPTO FDFL-Scal SAA --seeds 11 22 33
```
