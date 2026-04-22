# Changelog

codex resume 019ce170-90a1-76b1-9fb2-fb04219c1b36

All notable changes made today in this repository.

## 2026-04-21

### Advisor-review follow-up: FDFL mu parameter + PCGrad normalization (Claude)

#### Motivation
MD knapsack diagnosis confirmed two failure modes under SPSA decision
gradients: (1) FDFL (mu=0) diverges because there is no prediction
anchor — the SPSA gradient is noisy and without MSE regularization the
predictor drifts; (2) PCGrad oscillates because the decision-regret
gradient magnitude (~3000) dwarfs the prediction/fairness gradients
(~1), so the projection geometry is dominated by a single objective.

#### Added
- `pred_weight_mode` now accepts any numeric string (e.g. `"0.1"`,
  `"0.5"`) as a fixed prediction weight (mu).  Named keyword modes
  (`zero`, `fixed1`, `schedule`) take priority; anything else is
  parsed as `float`.  A non-numeric string still raises `ValueError`.
  Implemented in both `src/fair_dfl/training/loop.py` and
  `src/fair_dfl/algorithms/core_methods.py`.
- `PCGradHandler(normalize=True)` — per-objective gradient
  normalization before pairwise conflict projection, with rescaling by
  the mean of the original norms to preserve an objective-scale step
  size.  Enabled by `mo_pcgrad_normalize=True` in train config.
- New method configs `FDFL-0.1` and `FDFL-0.5` (static dec+pred+fair,
  mu ∈ {0.1, 0.5}).
- Method taxonomy in `experiments/configs.py` reorganized into three
  groups: PTO, static decision-focused, dynamic decision-focused.
  `README.md` method tables updated to match.

#### Changed
- `PCGrad` config now sets `mo_pcgrad_normalize: True` by default.
  No separate un-normalized variant is kept going forward.
- `PCGrad-nf` also updated to normalize.

#### Experiments (Exp 1 + 2 + 3)
- **Exp 1 — MD knapsack mu sweep**
  (`results/advisor_review/md_knapsack_mu_sweep/`):
  10 methods × 5 seeds × λ ∈ {0, 0.5, 1, 2} × 30 steps, SPSA
  n_dirs=8 eps=5e-3, n_train=300, alpha_fair=2.0, fairness=mad.
  Launcher: `experiments/advisor_review/run_md_knapsack_mu_sweep.py`.
  Key finding: FDFL (mu=0) normalized regret ≈ 1.0 (diverged);
  FDFL-0.1 snaps to ~0.19; FDFL-0.5/Scal cluster with FPLG near
  the top performers.

- **Exp 2 — Healthcare FDFL-mu append**
  (`results/advisor_review/healthcare_followup_v2/variant_a/**/stage_results_fdfl_mu.csv`):
  10 methods × 4 fairness_types × 2 alphas × 5 seeds × 70 steps,
  analytic gradients, full cohort.  Non-destructive — paper-cited
  `stage_results.csv` is untouched.
  Launcher: `experiments/advisor_review/run_healthcare_v2_fdfl_mu.py`.
  Grand summary: `grand_summary_fdfl_mu.csv` (1120 rows).
  Key finding (mad, alpha=2): FDFL mu ∈ {0, 0.1, 0.5, 1} all cluster
  at test_regret_normalized ≈ 0.102 ± 0.003 (analytic grads don't
  need the anchor). PCGrad (normalized) = 0.0945, meaningfully below
  the cluster — shift > 0.005, paper table should be reviewed.

- **Exp 3 — Gradient diagnostics figure**
  (`results/advisor_review/md_knapsack_mu_sweep/fig_gradient_scale_mu_sweep.png`):
  2×2 panel: grad_norm_dec vs grad_norm_pred on log-y axis for FDFL
  mu ∈ {0, 0.1, 0.5, 1}, top row MD knapsack (SPSA, scale explosion),
  bottom row healthcare (analytic, flat).
  Script: `experiments/advisor_review/plot_gradient_scale_mu_sweep.py`.

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
