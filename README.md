# Fair Decision-Focused Learning (Fair DFL)

Multi-objective decision-focused learning framework with fairness constraints.
Supports medical resource allocation and portfolio optimization tasks,
multiple predictor architectures, decision gradient backends, and
multi-objective optimization (MOO) methods.

## Quick Start

```bash
# Install in editable mode (with dev tools)
make install-dev

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

## Development

```bash
make install-dev          # Install package + pytest, ruff, mypy
make test                 # Run all tests (55 tests)
make test-fast            # Skip slow tests
make test-grads           # Gradient correctness tests only
make lint                 # Ruff linter
make typecheck            # Mypy type checks
make help                 # Show all available targets
```

Tests run without an editable install — `pyproject.toml` sets `pythonpath = ["src"]`
so pytest resolves `fair_dfl` directly from source.

## Project Structure

```
├── pyproject.toml              # Package metadata, dependencies, tool config
├── Makefile                    # Dev commands: install, test, lint, typecheck
├── CHANGELOG.md                # Change log
├── CODE_REVIEW.md              # Code review notes
├── DFL_MultiPortfolio_Fairness_Spec.md  # Design specification
│
├── configs.py                  # Method registry, training defaults, plot styling
├── run_methods.py              # CLI experiment runner
├── run_ablation.py             # Ablation experiment runner
├── analysis.py                 # Result analysis utilities
├── plotting.py                 # Plot generation
│
├── data/
│   └── data_processed.csv      # Medical resource allocation dataset (48,784 patients)
│
├── notebooks/
│   ├── run.ipynb               # Run experiments from notebook
│   ├── analysis.ipynb          # Analyze results and generate plots
│   ├── no_fairness.ipynb       # No-fairness variant experiments
│   ├── ablation/               # Ablation experiment notebooks
│   └── archive/
│       └── medical_experiment_colab.ipynb  # Superseded monolith (kept for reference)
│
├── scripts/
│   └── run_sample.sh           # Small-sample sanity check (n=1000)
│
├── results/                    # All experiment outputs
│   ├── stage_results_full.csv  # Per-lambda-stage metrics
│   ├── iter_logs_full.csv      # Per-iteration diagnostics
│   ├── *.png                   # Generated plots
│   ├── no_fairness/            # No prediction fairness experiments
│   └── no_decision_fairness/   # No decision fairness experiments
│
├── tests/
│   ├── test_losses.py          # Finite-difference gradient checks for all losses
│   ├── test_medical_gradients.py  # Medical task gradient correctness
│   ├── test_method_semantics.py   # Method config, MOO payload, warmstart
│   └── test_mo_handlers.py     # Multi-objective handler tests
│
└── src/fair_dfl/               # Main package
    ├── __init__.py             # Lazy-loaded entry point (run_experiment)
    ├── runner.py               # Top-level experiment dispatch
    ├── config.py               # Internal configuration
    ├── losses.py               # Loss functions (MSE, group fairness variants)
    ├── metrics.py              # Cosine similarity, L2 norm, orthogonal projection
    ├── schedules.py            # Learning rate and alpha schedules
    │
    ├── models/                 # Predictor architectures
    │   ├── registry.py         # build_predictor(), PredictorHandle
    │   ├── architectures.py    # MLP, ResNetTabular, FTTransformer
    │   ├── initialization.py   # Weight init modes (default, best_practice, legacy)
    │   └── postprocessing.py   # PostProcessor (softplus, relu, exp, none)
    │
    ├── decision/               # Decision gradient strategies
    │   ├── factory.py          # build_decision_gradient(), DecisionGradientComputer
    │   ├── interface.py        # DecisionResult, DecisionGradientStrategy ABC
    │   └── strategies/
    │       ├── analytic.py     # Analytic (KKT-based) gradients
    │       ├── finite_diff.py  # Finite difference approximation
    │       ├── fold_opt.py     # FFO (solver unrolling)
    │       ├── nce.py          # Noise-contrastive estimation
    │       ├── lancer.py       # LANCER surrogate
    │       ├── cvxpylayers.py  # CvxpyLayers (placeholder)
    │       └── torch_autograd.py  # Torch autograd (placeholder)
    │
    ├── training/               # Unified training loop
    │   ├── loop.py             # train_single_stage(), run_method_seed(), run_methods()
    │   ├── eval.py             # eval_split(), evaluate_model()
    │   └── method_spec.py      # MethodSpec dataclass, resolve_method_spec()
    │
    ├── tasks/                  # Optimization tasks
    │   ├── base.py             # BaseTask, SplitData, TaskData
    │   ├── medical_resource_allocation.py
    │   ├── resource_allocation.py
    │   ├── portfolio_qp.py
    │   ├── portfolio_qp_simplex.py
    │   ├── portfolio_qp_multi_constraint.py
    │   └── pyepo_synthetic.py  # PyEPO tasks (optional dependency)
    │
    ├── algorithms/             # MOO handlers and legacy trainers
    │   ├── mo_handler.py       # WeightedSum, PCGrad, MGDA, CAGrad, FAMO, PLG
    │   ├── torch_utils.py      # Gradient manipulation utilities
    │   ├── core_methods.py     # Legacy core trainer (runner.py old path)
    │   └── advanced_methods.py # Legacy advanced trainer (runner.py old path)
    │
    └── advanced/               # FFO/NCE/LANCER implementation details
        ├── predictors.py       # Advanced predictor utilities
        ├── nce.py              # NCE solution pool
        ├── lancer.py           # LANCER trainer
        └── ffolayer_local/     # FFO layer implementation
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

### Advanced Decision Gradient Methods

| Method | Backend | Description |
|--------|---------|-------------|
| FFO    | ffo     | Fold-and-optimize (solver unrolling) |
| NCE    | nce     | Noise-contrastive estimation |
| LANCER | lancer  | Learned surrogate gradients |

### MOO Methods (Multi-Objective Optimization)

| Method | Handler | Description |
|--------|---------|-------------|
| WS-equal / WS-dec / WS-fair | weighted_sum | Weighted sum with different weight profiles |
| MGDA   | mgda    | Multiple-gradient descent |
| PCGrad | pcgrad  | Projecting conflicting gradients |
| CAGrad | cagrad  | Conflict-averse gradient descent |
| FAMO   | famo    | Fast adaptive MOO |
| PLG-FP | plg_fp  | Nested PLG (fairness primary) |
| PLG-PP | plg_pp  | Nested PLG (prediction primary) |

### Baselines

| Method | Description |
|--------|-------------|
| SAA    | Sample average approximation (no training) |
| WDRO   | Wasserstein distributionally robust optimization |
| PTO    | Predict-then-optimize (no fairness, no decision gradient) |

### No-Fairness Variants

PCGrad-nf, MGDA-nf, CAGrad-nf: 2-objective (dec+pred) versions of MOO methods.

## Configuration

Method configs in `configs.py` declare:
- **Objective flags**: `use_dec`, `use_pred`, `use_fair`
- **Prediction weight mode**: `pred_weight_mode` (`fixed1`, `schedule`, `zero`)
- **Continuation**: whether to carry model state across lambda stages
- **Decision gradient backend**: `decision_grad_backend` (for FFO/NCE/LANCER)
- **MOO handler**: `mo_method` (for multi-objective gradient combination)

Training defaults in `DEFAULT_TRAIN_CFG`:
- `lambdas`: fairness penalty weights for Pareto sweep `[0.0, 0.05, 0.2, 0.5]`
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
| `ffo` | Solver unrolling | Differentiable solvers |
| `nce` | Noise-contrastive estimation | Black-box solvers |
| `lancer` | Learned surrogate | Black-box solvers |

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

Each result row includes provenance metadata: `run_id`, `timestamp_utc`, `git_commit`,
`python_version`, and `has_validation` for traceability.

## Notebooks

- **`notebooks/run.ipynb`**: Run experiments interactively
- **`notebooks/analysis.ipynb`**: Load results and generate comparison plots
- **`notebooks/no_fairness.ipynb`**: No-fairness variant experiments
- **`notebooks/ablation/`**: Ablation experiment notebooks

## Requirements

- Python >= 3.10
- PyTorch >= 2.0, NumPy >= 1.24, pandas >= 2.0, cvxpy >= 1.3, scipy >= 1.10, matplotlib >= 3.7
- Optional: `cvxpylayers`, `pyepo`, `threadpoolctl` (install via `pip install -e ".[advanced]"`)
- Dev tools: `pytest`, `ruff`, `mypy` (install via `pip install -e ".[dev]"`)
