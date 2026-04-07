# LP Knapsack Experiment

Decision-focused multi-task fair learning on a linear-program knapsack.
This experiment uses SPO+ decision gradients and is designed to show
clear separation between methods — unlike the alpha-fair variant where
the smooth concave objective absorbs prediction errors.

## Why LP?

The alpha-fair knapsack has strictly concave utility, producing smooth
allocations that change gradually with predictions. All methods achieve
similar regret because even poor predictions lead to near-optimal
decisions.

The LP knapsack `max r^T d, Ad <= b, 0 <= d <= 1` has **vertex
solutions** — most items are fully selected (d=1) or fully excluded
(d=0). A prediction error that swaps two items' rankings causes one to
be selected and another excluded, generating large regret. This is the
classic setting where decision-focused learning (DFL) outperforms
predict-then-optimize (PTO).

## Problem Formulation

```
max  r^T d
s.t. A d <= b          (3 resource constraints)
     0 <= d_i <= 1     (bounded allocation per item)
```

- **Items:** 10, split into two groups (majority/minority)
- **Constraints:** A in R^{3x10}, A_ij ~ Uniform[0.5, 1.5]
- **Budget:** b = 0.3 * A.sum(axis=1) (tight: ~3 of 10 items selected)
- **Features:** 5-dim, z ~ N(0, I_5)
- **Benefit mapping:** degree-2 polynomial: raw = z @ W_1 + z^2 @ W_2
- **Positive transform:** y = softplus(raw + group_shift + noise) + 0.05
- **Training samples:** 80 (small data regime)
- **Test samples:** 80

## Decision Gradient: SPO+

SPO+ (Elmachtoub & Grigas 2022) provides a convex surrogate loss for LP
decision regret:

```
c_spo  = 2 * r_hat - r           (SPO+ cost vector, can be negative)
d_spo  = argmax_{d in D} c_spo^T d
d_star = argmax_{d in D} r^T d   (oracle decision)

Loss:     L = max(c_spo^T d_spo - r^T d_star, 0)
Gradient: g = 2 * (d_spo - d_star)   (subgradient w.r.t. r_hat)
```

**Cost:** 2 solver calls per sample per step (vs SPSA's 3, vs
finite-diff's 2*dim+1 = 21).

Registered as `decision_grad_backend="spo_plus"` in the strategy factory.

## Fairness Approach

The LP objective is fairness-unaware. Fairness is injected through the
multi-task learning framework:

### During Training (3 objectives)

1. **Decision regret** (SPO+ loss) — learn to predict item rankings correctly
2. **Prediction MSE** — standard regression loss
3. **Prediction fairness** (MAD of per-group MSE) — equalize prediction
   quality across item groups so minority items are not systematically
   misranked

MOO methods (PCGrad, MGDA, CAGrad) handle the three-way tradeoff
without manual lambda tuning. FDFL-Scal sweeps lambda in {0, 0.5, 1, 5}.

### During Evaluation (3 decision-level metrics)

These appear in `stage_results.csv` as `test_decision_*`:

| Metric | Column | What it measures |
|--------|--------|-----------------|
| Allocation gap | `decision_alloc_gap` | \|mean(d[group0]) - mean(d[group1])\| |
| Selection rate gap | `decision_selection_gap` | \|frac(d>0.5 in group0) - frac(d>0.5 in group1)\| |
| Welfare gap | `decision_welfare_gap` | \|mean(r*d for group0) - mean(r*d for group1)\| |

**Paper narrative:** prediction fairness is a differentiable proxy.
Methods that achieve prediction fairness also achieve decision fairness
because LP linearity ensures unbiased predictions lead to unbiased
item selection.

## Unfairness Levels

| Level | group_bias | noise_std_lo | noise_std_hi | group_ratio | Group split |
|-------|-----------|-------------|-------------|------------|-------------|
| mild | 0.2 | 0.05 | 0.5 | 0.5 | 5/5 |
| medium | 0.4 | 0.05 | 1.0 | 0.65 | 7/3 |
| high | 0.6 | 0.05 | 1.5 | 0.75 | 8/2 |

Minority group items have: lower mean benefit (negative bias), higher
noise (harder to predict), and fewer representatives (group imbalance).

## Methods Compared

| Method | Category | Decision gradient | Lambda |
|--------|----------|------------------|--------|
| FPTO (lambda=0) = PTO | Two-stage | none | 0 |
| FPTO (lambda=0.5,1,5) | Two-stage | none | sweep |
| SAA | Baseline | none | -- |
| WDRO | Baseline | none | -- |
| FDFL-Scal (lambda=0) = DFL | Integrated | SPO+ | 0 |
| FDFL-Scal (lambda=0.5,1,5) | Integrated | SPO+ | sweep |
| FDFL-PCGrad | Integrated + MOO | SPO+ | -- |
| FDFL-MGDA | Integrated + MOO | SPO+ | -- |
| FDFL-CAGrad | Integrated + MOO | SPO+ | -- |

## Training Configuration

| Parameter | Baselines (FPTO/SAA/WDRO) | FDFL methods |
|-----------|--------------------------|--------------|
| Steps per lambda | 200 | 80 |
| Batch size | Full (80) | 32 |
| Learning rate | 0.002 | 0.002 |
| Decision gradient | none (analytic) | SPO+ |
| Hidden dim | 64 | 64 |
| Hidden layers | 2 | 2 |
| Lambda values | {0, 0.5, 1, 5} | {0, 0.5, 1, 5} or MOO |

## Running

### Local

```bash
# Full grid: 7 methods x 3 unfairness x 5 seeds = 105 runs
python experiments/lp_knapsack/run_lp_knapsack.py

# Dry run
python experiments/lp_knapsack/run_lp_knapsack.py --dry-run

# Subset
python experiments/lp_knapsack/run_lp_knapsack.py \
  --methods FDFL-Scal FPTO \
  --unfairness mild \
  --seeds 11 22
```

### Colab

```python
# In notebook setup cell:
from experiments.colab_runner import *

LP_RESULTS = os.path.join(DRIVE_ROOT, 'results', 'final', 'lp_knapsack')
os.makedirs(LP_RESULTS, exist_ok=True)

# Run all methods for one unfairness level
run_lp_knapsack_slice(
    unfairness_levels=['mild'],
    results_dir=LP_RESULTS,
)

# Or with overrides
run_lp_knapsack_slice(
    unfairness_levels=['mild'],
    results_dir=LP_RESULTS,
    steps=100,
    task_overrides={"n_items": 15, "budget_tightness": 0.25},
    train_overrides={"lr": 0.001, "hidden_dim": 32},
)

# Check progress
show_progress(LP_RESULTS, 'LP Knapsack')
```

### Notebook Override Parameters

All parameters are controllable from the notebook without editing source:

**task_overrides:**
- `n_items`, `n_constraints`, `n_features`
- `n_samples_train`, `n_samples_test`
- `poly_degree`, `budget_tightness`

**train_overrides:**
- `lr`, `lr_decay`, `grad_clip_norm`, `log_every`
- `decision_grad_backend` (default `"spo_plus"`)
- `fdfl_batch_size` (default 32)
- `hidden_dim`, `n_layers`, `activation`, `dropout` (model architecture)

**unfairness_configs:**
- Override the default `LP_UNFAIRNESS_LEVELS` dict entirely

## Expected Results

With LP + SPO+, you should see:

1. **SAA has highest regret** — constant predictions lead to wrong item
   rankings
2. **FDFL methods have lowest regret** — SPO+ learns decision-relevant
   prediction structure (which items are on the selection boundary)
3. **FPTO sits in between** — good predictions but doesn't focus on
   decision-relevant features
4. **Lambda sweep traces a Pareto frontier** — higher lambda = lower
   fairness violation at the cost of slightly higher regret
5. **MOO methods achieve good regret AND fairness** without manual
   lambda tuning

## File Inventory

| File | Purpose |
|------|---------|
| `run_lp_knapsack.py` | Standalone experiment runner (local) |
| `README.md` | This guide |
| `../../src/fair_dfl/decision/strategies/spo_plus.py` | SPO+ strategy |
| `../../src/fair_dfl/tasks/md_knapsack.py` | LP knapsack task (scenario="lp") |
| `../../experiments/colab_runner.py` | `run_lp_knapsack_slice()` for Colab |

## References

- Elmachtoub, A. N. & Grigas, P. (2022). "Smart 'Predict, then
  Optimize'." *Management Science*, 68(1), 9-26.
- Spall, J. C. (1992). "Multivariate Stochastic Approximation Using a
  Simultaneous Perturbation Gradient Approximation." *IEEE Trans.
  Automatic Control*, 37(3), 332-341.
