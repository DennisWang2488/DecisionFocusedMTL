# Changelog

codex resume 019ce170-90a1-76b1-9fb2-fb04219c1b36

All notable changes made today in this repository.

## 2026-03-23

### Result provenance metadata (Claude)

#### Added
- Every result row now includes: `run_id`, `timestamp_utc`, `git_commit`, `python_version`, `has_validation`.
- This enables tracing results back to exact code versions and distinguishing validation-less runs.

### Additional bug fixes (Claude)

#### Fixed
- `generate_data(seed)` on `MedicalResourceAllocationTask` now respects the `seed` parameter instead of silently ignoring it. When a caller passes a different seed, it overrides `data_seed` only; `split_seed` is left unchanged so callers can vary data generation independently of the train/val/test split. (BUG-4)
- Renamed misleading legacy config `mlp_2x64_softplus` to `mlp_2x64_relu` (old name kept as alias for backward compatibility). The activation was always `relu`, not `softplus`. (BUG-6)
- Removed duplicate `nan_steps` metric key from stage results; only `nan_or_inf_steps` is kept. (BUG-7)
- Converted `fair_dfl.training`, `fair_dfl.algorithms`, and `fair_dfl.tasks` package exports to lazy imports so lightweight unit tests do not eagerly pull in torch / PyEPO side effects during collection.

### Unreleased - Bug-fix pass on experiment semantics and reporting

#### Fixed
- No-fairness MOO variants now pass only their active objectives into the MOO handler, so `PCGrad-nf`, `MGDA-nf`, and `CAGrad-nf` run as true 2-objective methods instead of carrying a zero fairness objective through the optimizer.
- `train_subset_fraction` no longer subsets the test split in the unified runner; test evaluation stays fixed across subset experiments.
- Empty validation splits now record score columns as `NaN` rather than `0.0`, which preserves the distinction between "no validation split" and a genuine zero metric.
- Normalized test-regret metrics are now written even when validation is absent.
- Stage-level solver-call and decision-time totals now include work from iterations that were skipped due to NaN/Inf gradients.
- `run_methods.py` append/dedup logic now tolerates older CSV schemas when new dedup keys such as `fairness_type` are present only in fresh runs.
- Shared fairness losses for non-medical tasks now support `gap` and `atkinson` in addition to `mad` and `ge`.
- `softplus_with_grad()` now uses a numerically stable sigmoid computation for large-magnitude logits.

#### Changed
- The default training config now skips redundant lambda sweeps for methods that do not use fairness unless a method explicitly opts back into the full lambda path.

#### Cleaned Up
- Removed dead replacement logic from medical-task mini-batch sampling.

## 2026-03-12

### Commit `7ecb6e7` - Add unified training and experiment tooling

#### Added
- Unified experiment stack:
  - `src/fair_dfl/models/` for predictor architectures, init, postprocessing, and registry.
  - `src/fair_dfl/decision/` for pluggable decision-gradient strategies (analytic, finite-diff, FFO, NCE, LANCER, placeholders).
  - `src/fair_dfl/training/` for unified method spec resolution, training loop, and evaluation.
- New runner entrypoint `run_experiment_unified()` in `src/fair_dfl/runner.py`.
- New experiment/config utilities:
  - `configs.py`
  - `run_methods.py`
  - `analysis.py`
  - `plotting.py`
- Documentation and artifacts:
  - `README.md`
  - notebooks and result folders for fairness ablations.

#### Changed
- Extended methods and aliases to include new baselines and integrations (including SAA/WDRO and advanced backends).
- Added task-level decision interface methods across task implementations for generic gradient strategies.
- Updated notebooks and generated result plots/tables for the new workflow.

#### Fixed
- PTO lambda semantics (lambda path constrained as intended in new config flow).
- SAA evaluation path for non-medical tasks (override prediction supported in shared evaluation path).
- WDRO per-sample weighting for multi-output predictions.
- Notebook syntax issue in results-loading print lines.
- Unified-loop integration issues:
  - Reused stateful decision-gradient backends across lambda stages.
  - Avoided redundant per-step solver passes by letting strategies reuse precomputed task outputs.
  - Restored LANCER warm-start behavior in the unified path.

### Commit `8bc8ca5` - Reorganize project layout and add fairness-type CLI option

#### Added
- Small-sample runner script: `scripts/run_sample.sh`.
- Fairness metric switch in CLI:
  - `--fairness-type {mad,gap,atkinson}` in `run_methods.py`.
  - `fairness_type` propagated into task config and tagged in output rows.

#### Changed
- Repository layout cleanup:
  - Dataset moved to `data/data_processed.csv`.
  - Notebooks moved to `notebooks/` with archive under `notebooks/archive/`.
  - Fairness-ablation result folders moved under `results/no_fairness/` and `results/no_decision_fairness/`.
- README updated to reflect the new structure and CLI options.
- `src/fair_dfl/algorithms/__init__.py` docs clarified for unified vs legacy paths.

#### Notes
- Existing historical CSVs without `fairness_type` are not backward-migrated automatically when appending.
- Local tooling file `.claude/settings.local.json` remains intentionally untracked.
