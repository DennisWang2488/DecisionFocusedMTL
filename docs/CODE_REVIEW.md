# Code Review: Fair Decision-Focused Learning (FDFL) Repository

**Reviewer**: Claude (automated)
**Date**: 2026-03-22
**Scope**: Full repository review — bugs, quality issues, and architectural recommendations

---

## Part 1: Bugs

### BUG-1: Fairness type mismatch between medical and non-medical code paths

**Files**: `src/fair_dfl/losses.py:96-120`
**Severity**: High — will crash at runtime

The medical task (`medical_resource_allocation.py`) supports three fairness types: `"gap"`, `"mad"`, `"atkinson"`.
However, the shared `group_fairness_loss_and_grad()` function in `losses.py` (used by ALL non-medical tasks) only handles `"mad"` and `"generalized_entropy"/"ge"`.

Any non-medical task configured with `fairness_type="gap"` or `"atkinson"` will raise:
```
ValueError: Unknown fairness_type: gap
```

**Fix**: Either add `"gap"` and `"atkinson"` support to `losses.py`, or add a clear mapping between the two naming conventions.

---

### BUG-2: Numerical overflow in `softplus_with_grad` sigmoid

**File**: `src/fair_dfl/losses.py:10`
**Severity**: Medium — silent NaN/Inf in gradients

```python
sigmoid = 1.0 / (1.0 + np.exp(-z))  # overflows when z << 0
```

When `z` contains very large negative values, `np.exp(-z)` overflows to `inf`, producing `sigmoid = 0.0` (correct value) but potentially triggering floating-point warnings or NaN in edge cases. The softplus *value* computation on lines 7-9 already uses the numerically stable `max(z,0) + log1p(exp(-|z|))` pattern, but the gradient doesn't.

**Fix**: Use the stable form:
```python
sigmoid = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
```
Or equivalently reuse `exp_term`:
```python
sigmoid = 1.0 / (1.0 + exp_term) * np.where(z >= 0, 1.0, np.exp(z) / np.exp(z))
```

---

### BUG-3: `_apply_subset_fraction` subsets the test set

**File**: `src/fair_dfl/runner.py:68-74`
**Severity**: High — silently corrupts experimental results

```python
train_idx = choose_idx(data.train.x.shape[0])
val_idx = choose_idx(data.val.x.shape[0])
test_idx = choose_idx(data.test.x.shape[0])  # <-- should NOT be subsetted
```

Subsampling the test set changes evaluation metrics and makes results **non-comparable** across runs with different `train_subset_fraction` values. In research experiments, the test set should remain fixed to ensure fair comparison. Only `train` (and optionally `val`) should be subsetted.

Additionally, lines 76-84 also subset the medical task's internal `_splits["test"]`, compounding the issue.

**Fix**: Remove `test_idx` subsampling; keep `test` split untouched.

---

### BUG-4: `generate_data` ignores the `seed` parameter

**File**: `src/fair_dfl/tasks/medical_resource_allocation.py:113-114`
**Severity**: Medium — violates interface contract

```python
def generate_data(self, seed: int) -> TaskData:
    _ = seed  # explicitly discarded!
```

The `BaseTask` interface defines `generate_data(seed)`, but the medical task ignores it, using `self.data_seed` and `self.split_seed` from the constructor instead. This means:
- Callers who expect `seed` to control randomness get silently deterministic behavior
- The same data is generated regardless of what seed is passed
- `runner.py:264` calls `task.generate_data(seed=int(task_cfg.get("data_seed", 42)))` — the seed argument is useless

**Fix**: Either use the `seed` parameter, or document why it's intentionally ignored and consider removing it from the interface for this subclass.

---

### BUG-5: Dead `replace` logic in `sample_batch`

**File**: `src/fair_dfl/tasks/medical_resource_allocation.py:772-775`
**Severity**: Low — dead code, no functional impact

```python
if batch_size <= 0 or batch_size >= n:
    return s                          # returns early when batch_size >= n
replace = n < batch_size              # always False (since we only reach here when batch_size < n)
idx = rng.choice(n, size=batch_size, replace=replace)
```

The `replace` variable is always `False`. The condition `n < batch_size` can never be `True` because the early return on line 772 already handles `batch_size >= n`.

**Fix**: Remove the dead `replace` variable, hardcode `replace=False`.

---

### BUG-6: Misleading legacy config name

**File**: `src/fair_dfl/models/registry.py:49-53`
**Severity**: Low — confusing but not crash-inducing

```python
"mlp_2x64_softplus": {
    "arch": "mlp",
    "hidden_dim": 64,
    "n_layers": 2,
    "activation": "relu",  # <-- NOT softplus!
},
```

The name says "softplus" but the activation is "relu". Anyone using this legacy config by name will get unexpected behavior.

**Fix**: Either rename to `"mlp_2x64_relu"` or change the activation to `"softplus"`.

---

### BUG-7: Duplicate metric keys in stage results

**File**: `src/fair_dfl/training/loop.py:625-626`
**Severity**: Low — wastes space, confuses analysis

```python
"nan_or_inf_steps": nan_or_inf_steps,
"nan_steps": nan_or_inf_steps,  # same value, different key
```

Both keys track the exact same counter. Downstream analysis code may use either key inconsistently.

**Fix**: Remove one of the two keys (keep `"nan_or_inf_steps"` as it's more descriptive).

---

## Part 2: Quality Issues

### QI-1: Zero test coverage

No test files exist anywhere in the repository (`**/*test*` returns nothing).

**Impact**: Every change is a leap of faith. Gradient computations, solver logic, and fairness metrics are mathematically intricate — bugs in these are nearly impossible to catch by visual inspection alone.

**What to test (priority order)**:
1. **Gradient correctness** — finite-difference checks for every loss/gradient function in `losses.py`, `medical_resource_allocation.py` (the analytic Jacobians, VJP, etc.)
2. **Solver correctness** — `_solve_group`, `_solve_alpha_fair` produce feasible solutions (budget constraint satisfied, non-negative allocations)
3. **Fairness metrics** — known-answer tests for `_fair_loss_and_grad_gap`, `_fair_loss_and_grad_mad`, `_fair_loss_and_grad_atkinson`
4. **Training loop smoke tests** — 5-step training runs that don't crash, produce finite losses
5. **Config/method registry** — all 23 methods in `ALL_METHOD_CONFIGS` can be resolved without errors

---

### QI-2: Heavy reliance on private `_splits` attribute

**Files**: `training/loop.py`, `training/eval.py`, `runner.py`

The training loop, evaluation module, and runner all directly access `task._splits` (a "private" attribute by Python convention). This creates:
- Tight coupling between training code and the medical task's internal representation
- Fragile code that breaks if the medical task's storage format changes
- `isinstance(task, MedicalResourceAllocationTask)` checks scattered throughout the codebase

**Suggestion**: Add a public API to `BaseTask` for accessing splits, or unify the medical and non-medical data flow so both go through `TaskData`.

---

### QI-3: Massive code duplication in solver internals

**File**: `src/fair_dfl/tasks/medical_resource_allocation.py`

Three methods — `_solve_group` (line 186), `_solve_group_grad_jacobian` (line 230), and `_solve_group_vjp` (line 283) — all independently compute the same intermediate terms:

```python
term_s_all = (np.power(np.clip(c, 1e-12, None), -1.0/alpha) * np.power(np.clip(b, 1e-12, None), 1.0/alpha)) ** (1.0 - alpha)
term_h_all = np.power(np.clip(c, 1e-12, None), (alpha-1.0)/alpha) * np.power(np.clip(b, 1e-12, None), (1.0-alpha)/alpha)
s_k = np.add.reduceat(term_s_all[sort_order], group_start_indices)
h_k = np.add.reduceat(term_h_all[sort_order], group_start_indices)
# ... psi_k, xi, phi_all computation ...
```

This block is copy-pasted 3 times (~30 lines each). A change to the math must be applied in all three places.

**Suggestion**: Extract a `_GroupSolverContext` helper (or simple dict-returning function) that computes these shared intermediates once.

---

### QI-4: `_build_task` is an unmaintainable if-else chain

**File**: `src/fair_dfl/runner.py:105-267`

Adding a new task requires:
1. Adding a new import
2. Adding a new `if name == "..."` block (20+ lines of boilerplate)
3. Knowing which `bind_context` keys to pass

This is 160+ lines of near-identical code.

**Suggestion**: Use a task registry pattern (similar to `models/registry.py`). Each task class registers itself with its name and a `from_config(cfg) -> (task, data)` class method.

---

### QI-5: `isinstance` checks everywhere instead of polymorphism

**Files**: `training/loop.py` (lines 266, 281, 306, 496, 589), `training/eval.py` (line 87), `runner.py` (line 76)

The training loop is littered with:
```python
if isinstance(task, MedicalResourceAllocationTask):
    # medical-specific path
else:
    # generic path
```

This makes it very hard to add a new task type without modifying the training loop.

**Suggestion**: Unify the `BaseTask` interface so that:
- All tasks provide `compute_batch(pred, true, **ctx)` with the same signature
- All tasks handle their own batching via `sample_batch(split, batch_size, rng)`
- The training loop doesn't need to know which task type it's running

---

## Part 3: Architectural Recommendations for Research Maintainability

### AR-1: Add a proper experiment configuration system

**Current state**: Configs are built programmatically in `configs.py` and `run_methods.py` via dicts and CLI args. There's no way to save/load/diff/reproduce a complete experiment config.

**Recommendation**: Use a structured config system like [Hydra](https://hydra.cc/) or at minimum YAML/JSON config files that capture the *complete* experiment state. This enables:
- **Reproducibility**: `python run.py --config experiments/table2_mad_alpha2.yaml`
- **Experiment tracking**: configs are version-controlled alongside results
- **Ablation studies**: override specific keys without touching code
- **Diffing**: `diff experiment_a.yaml experiment_b.yaml` to understand what changed

Example structure:
```
configs/
  base.yaml            # shared defaults
  tasks/
    medical.yaml
    portfolio.yaml
  methods/
    fplg.yaml
    moo_pcgrad.yaml
  experiments/
    table1.yaml         # overrides for paper Table 1
    ablation_fairness.yaml
```

---

### AR-2: Separate "what to run" from "how to run"

**Current state**: `run_methods.py` mixes CLI parsing, experiment orchestration, CSV I/O, and deduplication logic in one 344-line file.

**Recommendation**: Split into layers:
```
cli.py                  # CLI argument parsing only
experiment.py           # Orchestrates task + method + seeds
io.py                   # CSV reading/writing/deduplication
```

This makes it easy to:
- Run experiments from a notebook (import `experiment.py`, skip CLI)
- Swap CSV output for a database or MLflow
- Add new CLI commands without touching experiment logic

---

### AR-3: Use a results database instead of flat CSVs

**Current state**: Results are appended to `stage_results_full.csv` and `iter_logs_full.csv`. Deduplication is done by checking `(task, method, seed, lambda, stage_idx)` tuples in `run_methods.py`.

**Problems**:
- CSVs grow unbounded and become slow to load
- No metadata about *when* or *how* a result was generated
- Hard to query "show me all FPLG results with alpha_fair=2.0 across all fairness types"
- Concurrent writes can corrupt the file

**Recommendation**: Use SQLite (zero-config, single file) or DuckDB for structured storage. Alternatively, adopt a lightweight experiment tracker like [MLflow](https://mlflow.org/) or [Weights & Biases](https://wandb.ai/). At minimum, add:
- A unique `run_id` to each experiment
- Timestamps
- Git commit hash
- Full config snapshot

---

### AR-4: Add a plugin/registry system for tasks

**Current state**: Adding a new task requires modifying 3+ files (`runner.py`, `tasks/__init__.py`, potentially `training/loop.py`).

**Recommendation**: Registry pattern where tasks self-register:
```python
@register_task("medical_resource_allocation")
class MedicalResourceAllocationTask(BaseTask):
    @classmethod
    def from_config(cls, cfg: dict) -> tuple[BaseTask, TaskData]:
        ...
```

Then `_build_task` becomes:
```python
def _build_task(cfg):
    return TASK_REGISTRY[cfg["name"]].from_config(cfg)
```

---

### AR-5: Unify the medical vs. non-medical code paths

**Current state**: The training loop has two parallel code paths — one for `MedicalResourceAllocationTask` and one for everything else. This is the single biggest source of complexity and potential bugs.

**Root cause**: The medical task uses 1D predictions with extra context (cost, race), while other tasks use 2D predictions with groups in `TaskData`.

**Recommendation**: Define a unified `BatchResult` dataclass:
```python
@dataclass
class BatchResult:
    x: np.ndarray
    y: np.ndarray
    context: dict[str, np.ndarray]  # {"cost": ..., "race": ..., "groups": ...}
```

And a unified `compute_batch` on `BaseTask`:
```python
def compute_batch(self, pred, true, context, need_grads, fairness_smoothing) -> dict:
    ...
```

This eliminates all `isinstance` checks in the training loop.

---

### AR-6: Add gradient unit tests with finite-difference verification

This is the single highest-impact quality improvement for a research codebase.

**What to test**: Every function that returns both a loss and a gradient:
- `losses.py`: `mse_loss_and_grad`, `group_mse_mad_loss_and_grad`, `group_mse_generalized_entropy_loss_and_grad`
- `medical_resource_allocation.py`: `_decision_regret_and_grad`, `_solve_group_vjp`, `_solve_jacobian`, `_solve_group_grad_jacobian`, `_fair_loss_and_grad_*`
- `torch_utils.py`: `merge_guided_dec_pred_gradient`

**Pattern**:
```python
def test_grad_mse():
    pred = np.random.randn(10, 5)
    true = np.random.randn(10, 5)
    loss, grad = mse_loss_and_grad(pred, true)
    # Finite difference check
    eps = 1e-5
    for i in range(pred.size):
        pred_plus = pred.copy().ravel()
        pred_plus[i] += eps
        pred_minus = pred.copy().ravel()
        pred_minus[i] -= eps
        loss_plus, _ = mse_loss_and_grad(pred_plus.reshape(pred.shape), true)
        loss_minus, _ = mse_loss_and_grad(pred_minus.reshape(pred.shape), true)
        fd_grad = (loss_plus - loss_minus) / (2 * eps)
        assert abs(grad.ravel()[i] - fd_grad) < 1e-4, f"Gradient mismatch at index {i}"
```

---

### AR-7: Add type hints and a `py.typed` marker

**Current state**: Some functions have type hints, many don't. The `Dict`, `List`, `Tuple` imports from `typing` are used inconsistently with the modern `dict`, `list`, `tuple` syntax.

**Recommendation**:
- Run `mypy` in strict mode and fix errors incrementally
- Use modern Python 3.10+ syntax (`dict[str, Any]` instead of `Dict[str, Any]`)
- This catches bugs at "edit time" rather than "3-hour experiment crashes at hour 2" time

---

### AR-8: Add a `Makefile` or `justfile` for common workflows

Research projects benefit enormously from documented, one-command workflows:

```makefile
test:           ## Run all tests
    pytest tests/ -v

test-grads:     ## Run gradient correctness tests only
    pytest tests/test_gradients.py -v

lint:           ## Type check + style
    mypy src/ && ruff check src/

quick:          ## Small-sample smoke test (5 min)
    python run_methods.py --methods FPLG FPTO --n-sample 500 --steps 20

full:           ## Full paper reproduction (~hours)
    python run_methods.py --all --n-sample 0

plot:           ## Generate all figures from latest results
    python -c "from plotting import *; ..."
```

---

### AR-9: Add a `pyproject.toml` and make the package installable

**Current state**: No `pyproject.toml`, `setup.py`, or `requirements.txt` visible. Dependencies are implicit.

**Recommendation**:
```toml
[project]
name = "fair-dfl"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "pandas>=2.0",
    "cvxpy>=1.3",
    "scipy>=1.10",
    "matplotlib>=3.7",
]

[project.optional-dependencies]
dev = ["pytest", "mypy", "ruff"]
advanced = ["cvxpylayers", "pyepo"]
```

This enables:
- `pip install -e .` for development
- `pip install -e ".[dev]"` for development with test/lint tools
- Clear documentation of what's needed to run the code

---

### AR-10: Consider a notebook-first analysis workflow

**Current state**: `analysis.py` and `plotting.py` contain functions that are likely called from notebooks (Colab, given the repo name), but there's no notebook in the repo.

**Recommendation**: Include a canonical analysis notebook (`analysis.ipynb`) that:
1. Loads results from the standard CSV/DB location
2. Generates all paper tables and figures
3. Includes interactive exploration (filter by method, alpha, fairness type)
4. Is version-controlled so reviewers can reproduce figures

---

## Summary: Priority Matrix

| Priority | Item | Type | Effort |
|----------|------|------|--------|
| **P0** | BUG-3: Test set subsampling | Bug | 5 min |
| **P0** | QI-1: Add gradient unit tests | Quality | 1-2 days |
| **P0** | BUG-1: Fairness type mismatch | Bug | 30 min |
| **P1** | AR-5: Unify medical vs non-medical paths | Architecture | 2-3 days |
| **P1** | BUG-2: Sigmoid overflow | Bug | 5 min |
| **P1** | AR-9: `pyproject.toml` + dependencies | Infrastructure | 1 hour |
| **P1** | AR-6: Finite-diff gradient tests | Quality | 1 day |
| **P2** | AR-1: Structured config system | Architecture | 1-2 days |
| **P2** | QI-3: Deduplicate solver internals | Quality | 2-3 hours |
| **P2** | AR-4: Task registry | Architecture | 3-4 hours |
| **P2** | QI-5: Remove `isinstance` checks | Quality | 1-2 days |
| **P3** | AR-2: Separate CLI/orchestration/IO | Architecture | 1 day |
| **P3** | AR-3: Results database | Infrastructure | 1 day |
| **P3** | BUG-4: `generate_data` seed ignored | Bug | 15 min |
| **P3** | AR-7: Type hints + mypy | Quality | 1-2 days |
| **P3** | AR-8: Makefile | Infrastructure | 30 min |
| **P4** | BUG-5: Dead `replace` code | Bug | 2 min |
| **P4** | BUG-6: Misleading config name | Bug | 2 min |
| **P4** | BUG-7: Duplicate metric key | Bug | 2 min |
| **P4** | QI-4: `_build_task` refactor | Quality | 2-3 hours |
| **P4** | AR-10: Analysis notebook | Documentation | 2-3 hours |
