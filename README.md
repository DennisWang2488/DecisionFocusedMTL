# Fair Decision-Focused Learning (Fair DFL)

Multi-objective decision-focused learning framework with fairness constraints.
Supports medical resource allocation and portfolio optimization tasks,
multiple predictor architectures, decision gradient backends, and
multi-objective optimization (MOO) methods.

## Quick Start

```bash
# List all available methods
python run_methods.py --list-methods

# Quick test with 500 samples
python run_methods.py --methods PLG FPLG --n-sample 500

# Run specific methods at specific alpha values
python run_methods.py --methods MGDA PCGrad --alphas 0.5 2.0

# Run everything (all 23 methods, all alpha values)
python run_methods.py --all

# Dry run — show plan without executing
python run_methods.py --methods PLG FPLG --dry-run

# Small-sample sanity check (n=1000, 3 fairness types, 3 alphas)
./scripts/run_sample.sh --dry-run
```

Results are saved as CSVs in `results/` by default.

The root-level scripts `run_methods.py`, `run_ablation.py`, `analysis.py`, `plotting.py`, and `configs.py`
are compatibility entrypoints. Their implementations now live under `experiments/`.

## Project Structure

```
colab_upload/
    README.md                   # Project overview and usage
    CHANGELOG.md                # High-level change history
    Makefile                    # Common developer workflows
    pyproject.toml              # Packaging, pytest, and lint config

    run_methods.py              # Thin compatibility wrapper
    run_ablation.py             # Thin compatibility wrapper
    analysis.py                 # Thin compatibility wrapper
    plotting.py                 # Thin compatibility wrapper
    configs.py                  # Thin compatibility wrapper

    docs/                       # Supplementary project documents
        CODE_REVIEW.md
        DFL_MultiPortfolio_Fairness_Spec.md

    experiments/                # Experiment-facing code grouped by function
        __init__.py
        configs.py              # Method registry, defaults, plot styling
        colab_runner.py         # Shared runner for Colab workers (healthcare + knapsack + LP)
        run_methods.py          # Main CLI experiment runner
        run_ablation.py         # Ablation experiment runner
        run_healthcare_final.py # Healthcare experiment (INFORMS JoC)
        run_knapsack_final.py   # Alpha-fair knapsack experiment (INFORMS JoC)
        generate_tables.py      # LaTeX table generator
        generate_figures.py     # Publication figure generator
        analysis.py             # Result loading and summary helpers
        plotting.py             # Plot generation
        lp_knapsack/            # LP knapsack experiment with SPO+ (see below)

    data/
        data_processed.csv      # Medical resource allocation dataset (48,784 patients)
        mosek.lic               # MOSEK solver license

    notebooks/
        run.ipynb               # Run experiments from notebook
        analysis.ipynb          # Analyze results and generate plots
        no_fairness.ipynb       # No-fairness variant experiments
        archive/
            medical_experiment_colab.ipynb  # Superseded monolith (kept for reference)

    scripts/
        run_sample.sh           # Small-sample sanity check (n=1000)

    results/                    # All experiment outputs
        stage_results_full.csv  # Per-lambda-stage metrics
        iter_logs_full.csv      # Per-iteration diagnostics
        *.png                   # Generated plots
        no_fairness/            # No prediction fairness experiments
        no_decision_fairness/   # No decision fairness experiments
        sample/                 # Small-sample test results

    src/fair_dfl/
        runner.py               # Top-level experiment dispatch
        schedules.py            # Learning rate and alpha schedules
        losses.py               # Loss functions (MSE, fairness)
        metrics.py              # Cosine similarity, L2 norm, orthogonal projection

        models/                 # Predictor architectures
            registry.py         # build_predictor(), PredictorHandle
            architectures.py    # MLP, ResNetTabular, FTTransformer
            initialization.py   # Weight init modes (default, best_practice, legacy)
            postprocessing.py   # PostProcessor (softplus, relu, exp, none)

        decision/               # Decision gradient strategies
            factory.py          # build_decision_gradient(), DecisionGradientComputer
            interface.py        # DecisionResult, DecisionGradientStrategy ABC
            strategies/
                analytic.py     # Analytic (KKT-based) gradients
                finite_diff.py  # Finite difference approximation
                spsa.py         # SPSA (simultaneous perturbation)
                spo_plus.py     # SPO+ (LP surrogate, Elmachtoub & Grigas 2022)

        training/               # Unified training loop
            loop.py             # train_single_stage(), run_method_seed(), run_methods()
            eval.py             # eval_split(), evaluate_model()
            method_spec.py      # MethodSpec dataclass, resolve_method_spec()

        tasks/                  # Optimization tasks
            base.py             # BaseTask, SplitData, TaskData
            md_knapsack.py      # Multi-dim knapsack (LP + alpha-fair scenarios)
            medical_resource_allocation.py
            resource_allocation.py
            portfolio_qp.py
            portfolio_qp_simplex.py
            portfolio_qp_multi_constraint.py
            pyepo_synthetic.py

        algorithms/             # MOO handlers, gradient utilities, and legacy trainers
            mo_handler.py       # WeightedSum, PCGrad, MGDA, CAGrad, FAMO, PLG variants
            torch_utils.py      # Gradient manipulation utilities
            core_methods.py     # Legacy core trainer (used by runner.py old path)

    tests/                     # Lightweight regression and semantics checks
        test_losses.py
        test_medical_gradients.py
        test_method_semantics.py
        test_mo_handlers.py
```

## Methods

### Base Methods

| Method | Objectives | Description |
|--------|-----------|-------------|
| FPTO   | pred+fair | Fair predict-then-optimize |
| DFL    | dec       | Decision-focused learning |
| FDFL   | dec+fair  | Fair DFL |
| PLG    | dec+pred  | Predict-and-Learn with Gradients |
| FPLG   | dec+pred+fair | Fair PLG (full 3-objective) |

### MOO Methods (Multi-Objective Optimization)

| Method | Handler | Description |
|--------|---------|-------------|
| WS-equal / WS-dec / WS-fair | weighted_sum | Weighted sum with different weight profiles |
| MGDA   | mgda    | Multiple-gradient descent |
| PCGrad | pcgrad  | Projecting conflicting gradients |
| CAGrad | cagrad  | Conflict-averse gradient descent |
| FAMO   | famo    | Fast adaptive MOO |

### Baselines

| Method | Description |
|--------|-------------|
| SAA    | Sample average approximation (no training) |
| WDRO   | Wasserstein distributionally robust optimization |
| PTO    | Predict-then-optimize (no fairness, no decision gradient) |

### No-Fairness Variants

PCGrad-nf, MGDA-nf, CAGrad-nf: 2-objective (dec+pred) versions of MOO methods.

## Configuration

Method configs in `experiments/configs.py` declare:
- **Objective flags**: `use_dec`, `use_pred`, `use_fair`
- **Prediction weight mode**: `pred_weight_mode` (`fixed1`, `schedule`, `zero`)
- **Continuation**: whether to carry model state across lambda stages
- **Decision gradient backend**: `decision_grad_backend` (analytic, finite_diff, spsa, spo_plus)
- **MOO handler**: `mo_method` (for multi-objective gradient combination)

Training defaults in `DEFAULT_TRAIN_CFG`:
- `lambdas`: fairness penalty weights for Pareto sweep `[0.0, 0.5]`
- `seeds`: `[11, 22, 33]`
- `steps_per_lambda`: `70`
- `model`: `{"arch": "mlp", "hidden_dim": 64, "n_layers": 2}`

### Predictor Architectures

Configure via the `model` key in training config:

```python
# MLP (default)
"model": {"arch": "mlp", "hidden_dim": 64, "n_layers": 2, "activation": "relu"}

# ResNet for tabular data
"model": {"arch": "resnet_tabular", "hidden_dim": 128, "n_blocks": 3}

# Feature Tokenizer Transformer
"model": {"arch": "ft_transformer", "d_token": 64, "n_heads": 4, "n_layers": 2}
```

### Decision Gradient Backends

Set `decision_grad_backend` in method config:

| Backend | Strategy | Use Case |
|---------|----------|----------|
| `analytic` (default) | KKT-based implicit diff | Tasks with closed-form solutions |
| `finite_diff` | Finite differences | Any task with a solver |
| `spsa` | Simultaneous perturbation (Spall 1992) | Black-box solvers, dim-independent cost |
| `spo_plus` | SPO+ convex surrogate (Elmachtoub & Grigas 2022) | LP tasks only, 2 calls/sample |

## CLI Options

```
python run_methods.py [options]

--methods NAME [NAME ...]   Methods to run (case-insensitive)
--all                       Run all 23 methods
--alphas FLOAT [FLOAT ...]  Alpha-fairness values (default: 0.5 2.0)
--fairness-type TYPE        Prediction-side fairness metric: mad, gap, atkinson (default: mad)
--n-sample INT              Number of patients, 0=all (default: 0)
--data-csv PATH             Path to data CSV (default: data/data_processed.csv)
--results-dir DIR           Output directory (default: results/)
--seeds INT [INT ...]       Override seeds
--lambdas FLOAT [FLOAT ...] Override lambda values
--steps INT                 Override steps_per_lambda
--device DEVICE             Override device (cuda/cpu)
--overwrite                 Overwrite existing results
--dry-run                   Show plan without executing
--list-methods              List all methods and exit
```

## Output Files

- `results/stage_results_full.csv`: Per-lambda-stage metrics (regret, fairness, MSE)
- `results/iter_logs_full.csv`: Per-iteration training diagnostics (gradient norms, cosines, losses)

## Notebooks

- **`notebooks/run.ipynb`**: Run experiments interactively
- **`notebooks/analysis.ipynb`**: Load results and generate comparison plots
- **`notebooks/no_fairness.ipynb`**: No-fairness variant experiments

## LP Knapsack Experiment (INFORMS JoC)

The `experiments/lp_knapsack/` directory contains a self-contained LP knapsack
experiment designed to demonstrate decision-focused multi-task fair learning
with strong method differentiation.

**Problem:** `max r^T d,  A d <= b,  0 <= d <= 1` — LP solutions are at
vertices (most items are fully selected or fully excluded), so prediction
quality directly determines which items are chosen.

**Decision gradient:** SPO+ (Elmachtoub & Grigas 2022) — a convex surrogate
for LP regret. Only 2 solver calls per sample per gradient step.

**Fairness:** Prediction-level MAD (mean absolute deviation of per-group MSE)
during training. Decision-level metrics (allocation gap, selection rate gap,
welfare gap) for evaluation.

```bash
# Local run
python experiments/lp_knapsack/run_lp_knapsack.py
python experiments/lp_knapsack/run_lp_knapsack.py --dry-run
python experiments/lp_knapsack/run_lp_knapsack.py --methods FDFL-Scal --unfairness high

# Colab (in notebook cell)
from experiments.colab_runner import *
run_lp_knapsack_slice(unfairness_levels=['mild'], results_dir=LP_RESULTS)
```

See [`experiments/lp_knapsack/README.md`](experiments/lp_knapsack/README.md)
for the full experimental specification.

## Layout Rationale

- `src/fair_dfl/`: reusable library code
- `experiments/`: experiment orchestration, configs, and analysis utilities
- `docs/`: review/spec/reference documents
- `data/`, `results/`, `notebooks/`: inputs, outputs, and exploratory work
- root wrappers: preserve existing commands while keeping implementation files grouped by function
