# Changelog

codex resume 019ce170-90a1-76b1-9fb2-fb04219c1b36

All notable changes made today in this repository.

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
