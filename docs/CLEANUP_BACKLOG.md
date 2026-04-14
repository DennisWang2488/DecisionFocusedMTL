# Cleanup Backlog

This file records cleanup items identified during code review passes that were intentionally **deferred** — not because they're wrong, but because they fall outside the scope of the review at hand (refactor instead of deletion, wider blast radius than agreed, kept on user instruction, etc.).

Each entry lists *what*, *why deferred*, and *how to safely tackle it later*. Use this as the starting point for future cleanup sessions so you don't re-derive the same analysis from scratch.

Last updated: 2026-04-14, after the second review pass.

---

## 1. `core_methods.py` legacy trainer

**File**: `src/fair_dfl/algorithms/core_methods.py` (1094 LOC)

**Why deferred**: The legacy `run_experiment()` path in `runner.py` calls `run_core_methods()` from this file, but no notebook, experiment script, or test imports `fair_dfl.run_experiment` — they all use `run_experiment_unified`. So `run_core_methods()`, `train_single_seed()`, `CORE_METHOD_SPECS`, `METHOD_ALIASES`, `DEFAULT_METHODS`, and `PUBLIC_METHODS` are provably dead. The catch: `loop.py:413` still imports `_finite_diff_decision_grad` from this file as a fallback when the user has not configured a `DecisionGradientComputer`. Removing the dead path requires touching `runner.py`, `algorithms/__init__.py`, `fair_dfl/__init__.py`, and trimming `core_methods.py` itself to keep just the FD function. Wider blast radius than the current safe-deletion scope.

**How to tackle later**:
1. Confirm again that no caller imports `fair_dfl.run_experiment` (grep `from fair_dfl import run_experiment` and `import fair_dfl` followed by `.run_experiment`).
2. Strip `core_methods.py` down to just `_finite_diff_decision_grad` and its helper `_softplus_np`. Drop everything else: `MethodSpec`, `BASE_METHOD_SPECS`, `METHOD_SPECS`, `PUBLIC_METHODS`, `METHOD_ALIASES`, `_MLP2x64Softplus`, `_combine_gradients_with_mo_handler`, `_combine_gradients`, `_train_single_stage`, `train_single_seed`, `run_core_methods`. ~1000 LOC removed.
3. Delete `runner.py:run_experiment()` (currently lines 309–347 approximately) and its supporting `DEFAULT_METHODS`/`PUBLIC_METHODS` constants. Drop the `from .algorithms.core_methods import ...` line at the top.
4. Trim `algorithms/__init__.py`: remove `METHOD_SPECS`, `METHOD_ALIASES`, `REVERSE_ALIASES`, `run_core_methods` from `__all__` and `_EXPORT_MAP`.
5. Update `fair_dfl/__init__.py` to expose `run_experiment_unified` as the public entry point. If you want backward compatibility, alias `run_experiment = run_experiment_unified` with a deprecation warning.
6. After this is done, the items below (#5 `resource_allocation.py`, the `finite_diff.py` isinstance check) become much easier to remove together.

**Net delta**: ~−1000 LOC, no behavior change.

---

## 2. NestedPLG handler dedup

**File**: `src/fair_dfl/algorithms/mo_handler.py` lines 734–928

**Why deferred**: Behavior-preserving refactor, not a deletion. Three classes — `NestedPLGFairPrimaryHandler`, `NestedPLGDecPrimaryHandler`, `NestedPLGPredPrimaryHandler` — are 95 % identical. They share the same constructor, the same kappa decay formula, the same diagnostic computation, and the same `compute_direction` structure. They differ only in which gradient fills the (primary, guide) roles in step 1 and step 2. Total duplication: ~130 LOC.

**How to tackle later**:
1. Extract a `_NestedPLGBase` class with the constructor, `extra_logs()`, and a `compute_direction()` template that calls two abstract methods: `_step1_assignment(grads) -> (primary, guide)` and `_step2_assignment(grads, d_step1) -> (primary, guide)`.
2. Re-implement each of the three existing classes as a thin subclass that just provides the assignment hooks (~15 LOC each).
3. There is also a latent bug to fix in passing: `_build_active_moo_payload` at `loop.py:153` only checks for `NestedPLGFairPrimaryHandler` and `NestedPLGPredPrimaryHandler`, missing `NestedPLGDecPrimaryHandler`. After introducing the base class, change the check to `isinstance(mo_handler, _NestedPLGBase)`.
4. There are no existing tests for the NestedPLG handlers (verified by grep). Add at least one test in `tests/test_mo_handlers.py` that instantiates each variant with known inputs and verifies they produce *different* outputs from each other.

**Net delta**: ~−130 LOC, no behavior change (plus one latent bug fixed).

---

## 3. MOO config base pattern

**File**: `experiments/configs.py` lines 68–106

**Why deferred**: Refactor, not deletion. 11 of the MOO method configs (`WS-equal`, `WS-dec`, `WS-fair`, `MGDA`, `PCGrad`, `CAGrad`, `FAMO`, `PLG-DP`, `PLG-FP`, `PLG-PP`, plus `FDFL-Scal`) share an identical base: `method=fplg`, `use_dec=True`, `use_pred=True`, `use_fair=True`, `pred_weight_mode=fixed1`, `continuation=True`, `allow_orthogonalization=True`. They differ only in `mo_method` and (sometimes) `mo_weights`.

**How to tackle later**:
1. Define `_MOO_BASE = {...}` with the seven shared fields.
2. Build each variant by merging: `"MGDA": {**_MOO_BASE, "mo_method": "mgda"}` etc.
3. Define a separate `_NF_BASE` for the no-fairness variants (`PCGrad-nf`, `MGDA-nf`, `CAGrad-nf`) which use `method=plg`, `use_fair=False`.
4. The test `tests/test_method_semantics.py` already validates that all configs resolve correctly, so it will catch any mistake in the merge.

**Net delta**: ~−40 LOC, no behavior change. Pure readability/maintainability win.

---

## 4. Portfolio QP variants

**Files**:
- `src/fair_dfl/tasks/portfolio_qp.py` (169 LOC)
- `src/fair_dfl/tasks/portfolio_qp_simplex.py` (213 LOC)
- `src/fair_dfl/tasks/portfolio_qp_multi_constraint.py` (237 LOC)

**Why deferred**: User instruction. No active experiment uses these tasks (grep `"name": "portfolio_qp` across `experiments/` returns no hits), but the user wants to keep them as reference implementations.

**How to tackle later**: Re-evaluate only if the file count grows further or if maintenance cost becomes visible (e.g. someone breaks them while refactoring `losses.py` and nobody notices because no test exercises them). If kept long-term, consider extracting a shared `BasePortfolioTask` because `generate_data()` is ~95 % identical across all three variants. But again — only if they stay alive.

---

## 5. `resource_allocation.py`

**File**: `src/fair_dfl/tasks/resource_allocation.py`

**Why deferred**: User said "only PyEPO" in the second review pass, and we honored that. After Phase 1 of the second review deletes `pyepo_synthetic.py` (which was the only consumer of `ResourceAllocationTask` as a parent class), this file becomes truly orphaned. It is still referenced by `core_methods.py` and `decision/strategies/finite_diff.py` via `isinstance(task, ResourceAllocationTask)` checks, but no active experiment ever creates an instance.

**How to tackle later**: Delete this file together with item #1 (`core_methods.py` legacy trainer). When `core_methods.py` is trimmed, the `isinstance` check there goes away. The matching `isinstance` check in `decision/strategies/finite_diff.py` should also be removed at the same time. After both are gone, this file has no references and can be deleted.

**Net delta**: ~−210 LOC. Bundle it with item #1.

---

## 6. Training loop simplification

**File**: `src/fair_dfl/training/loop.py` (894 LOC, `train_single_stage` ~542 lines, cyclomatic complexity ~30)

**Why deferred**: Largest complexity hotspot but riskiest to refactor. Grew from 490 → 542 lines between the first and second review passes. Contains 7 scattered `isinstance(task, MedicalResourceAllocationTask)` checks that violate the task interface abstraction. No integration test covers a full healthcare run end-to-end, so any structural change risks silent behavior drift that would only surface in published numbers.

**How to tackle later**:
1. **Pre-work**: write an integration test that runs one healthcare seed and one MD knapsack seed for ≤20 steps and asserts on final regret/MSE values within tight tolerance. This is a precondition — do not refactor without it.
2. Extract `_compute_decision_gradient()` from the dispatch at lines ~387–408. Three branches collapse into one call.
3. Extract `_optimizer_step()` from lines ~533–593 (nan/inf handling, gradient clipping, FAMO post-step, dual lambda update). ~60 lines moved.
4. Extract a `TaskAdapter` (or extend `BaseTask`) with `sample_batch(split, size, rng)` and `compute_output(pred, true, **ctx)` methods so the 7 medical isinstance checks can become a single polymorphic call.
5. Extract `_iteration_logger` for the per-step diagnostics block (~44 lines).
6. Run the integration test before and after each extraction. Commit each extraction separately.
7. Target: `train_single_stage()` ~280–300 lines (a 45 % reduction), CC ~12–15.

**Net delta**: 0 LOC overall (restructure, not deletion), but huge clarity win and removes the medical task coupling.

---

## 7. Experiment runner deduplication

**Files**:
- `experiments/run_methods.py`
- `experiments/run_ablation.py`
- `experiments/run_healthcare_final.py`
- `experiments/run_knapsack_final.py`
- `experiments/colab_runner.py`

**Why deferred**: Refactor, not deletion. ~150 LOC of shared boilerplate is duplicated across these runners: path/sys.path setup, argparse construction, `load_existing_csv()` / `append_and_save()`, run-metadata tagging (run_id, timestamp, git commit, python version), `compute_full_batch_size()`, dry-run output formatting, done-marker checking. `colab_runner.py` additionally duplicates method grid definition and task config builders.

**How to tackle later**:
1. Create `experiments/_runners_common.py` with: `load_existing_csv`, `append_and_save`, `make_run_metadata`, `build_common_argparser`, `setup_path`, `done_marker_path`, `compute_full_batch_size` (move from `configs.py`).
2. Update each runner to import from this module instead of redefining. Keep each runner's *unique* logic (task-specific config, lambda iteration order, etc.) in the runner itself.
3. For `colab_runner.py`, reuse the same task config builders that `run_healthcare_final.py` and `run_knapsack_final.py` already use. The method grid is already in `experiments/configs.py:ALL_METHOD_CONFIGS` — point at it.
4. Run the existing experiments end-to-end after each runner is updated, to confirm no regression in CSV format / metadata fields.

**Net delta**: ~−150 LOC.

---

## 8. Root-level wrapper files

**Files**:
- `run_methods.py` (8 lines, forwards to `experiments.run_methods:main`)
- `run_ablation.py` (8 lines, forwards to `experiments.run_ablation:main`)
- `analysis.py` (3 lines, re-exports `experiments.analysis`)
- `configs.py` (3 lines, re-exports `experiments.configs`)
- `plotting.py` (3 lines, re-exports `experiments.plotting`)

**Why deferred**: Total cost is 26 LOC and they're referenced by `Makefile` lines 28 and 31 (`make quick` and `make full`). They exist as backward-compatibility shims so `python run_methods.py ...` from the repo root still works. Zero maintenance cost as long as the forwarding stays accurate.

**How to tackle later**: Re-evaluate only if (a) the wrappers start to drift from what they forward to (e.g. arguments diverge), or (b) the Makefile entries are updated to call `python -m experiments.run_methods` directly, in which case the wrappers become genuinely unused. Don't delete them just to remove files.

---

## 9. `legacy-md-knapsack` branch

**Why deferred**: User instruction. The branch is fully merged into main (zero unique commits) but the user wants to keep it as a reference snapshot of the pre-redesign MD knapsack layout, in case anything regresses in the new per-individual / multi-resource design.

**How to tackle later**: Delete (`git branch -D legacy-md-knapsack`) once the new MD knapsack design has been validated end-to-end across all the planned experiments and the user no longer needs the snapshot for comparison. The git history preserves the commits regardless.

---

## 10. Medical task fairness method dedup

**File**: `src/fair_dfl/tasks/medical_resource_allocation.py` lines ~537–697

**Why deferred**: Not a clean drop-in replacement. The task has its own `_fair_loss_and_grad_mad()`, `_fair_loss_and_grad_gap()`, `_fair_loss_and_grad_atkinson()`, and `_fair_loss_and_grad_dp()` static methods. They are mathematically equivalent to the corresponding functions in `losses.py`, but operate on **1D individual-level arrays** indexed by `race` group ID, while `losses.py` operates on **2D batch×group arrays**. The shape conventions differ enough that you can't just `from losses import ...` and call them.

**How to tackle later**:
1. Add `individual_mse_mad_loss_and_grad`, `individual_mse_gap_loss_and_grad`, `individual_mse_atkinson_loss_and_grad`, `individual_pred_mean_dp_loss_and_grad` to `losses.py`. These take `(pred, true, groups)` as three 1D arrays.
2. Add tests in `tests/test_losses.py` that verify each new function produces identical output to the medical task's static methods on the same inputs.
3. Have the medical task's `_compute_fairness` dispatch to the shared `losses.py` versions instead of its own static methods.
4. Delete the now-unused static methods from the medical task (~50 LOC freed).

**Risk**: moderate. `tests/test_medical_gradients.py` specifically tests the medical task's gradient outputs end-to-end, so any divergence will be caught — but the fix may be subtle if there's an indexing bug.

**Net delta**: ~−50 LOC plus better test coverage of the loss functions.

---

## How to use this file

When starting a new cleanup session:

1. Read this file first.
2. Pick items by scope: deletions are safer than refactors; items with explicit "How to tackle later" instructions can be picked off one at a time.
3. Items marked "deferred on user instruction" need explicit user re-confirmation — don't act on them just because they're documented here.
4. Update this file at the end of each session: cross out completed items, add new ones, tighten the "How to tackle later" instructions if you learned something new during the work.
