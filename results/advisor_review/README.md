# Advisor review experiment results

Empirical follow-up to the four advisor-feedback code changes shipped in
commits `2598462` and `3cab494` on `claude/review-advisor-feedback-lYjdA`
(now also on `main`).

## Layout

```
hp_tuning/
    md_knapsack/                # Step 2 — find a method-differentiating regime
        sweep_<timestamp>/      # one subdirectory per sweep step
            stage_results.csv
            iter_logs.csv
            config.json
        chosen_config.json      # the final picked config

hypothesis/
    dp_vs_ap/                   # Step 3a — demographic parity vs accuracy parity (healthcare)
        ap/, dp/                # parallel CSVs for the two fairness types
        notes.md
    benefit_cost_imbalance/     # Step 3b — 3x3 grid on the new MD task
        cell_b<bb>_c<cb>/       # one folder per (benefit_bias, cost_bias) cell
        notes.md
    train_test_gap/             # Step 3c — analysis of healthcare CSVs (no new runs)
        analysis.md

budget_sweep/
    md_small/                   # Step 4a — budget tightness sweep on small MD
        budget_<value>/
    healthcare_full/            # Step 4b — local-scale (n_sample=5000) healthcare sweep
        budget_<value>/

figures/                        # all final PNGs from the visualization notebook
```

Every run writes a `config.json` alongside its CSVs capturing the exact
task config, training config, method list, seeds, lambdas, and timestamp.

## Conventions

- Methods used throughout: `FPTO`, `DFL`, `FDFL-Scal`, `FPLG`, `PCGrad`, `MGDA`
  (one representative from each family).
- All experiments are **local-scale**: healthcare uses `n_sample <= 5000`,
  MD uses `n_train <= 240`. The full-cohort 48k healthcare appendix runs
  are explicitly out of scope for this task and will happen on Colab.
- Stage-level CSVs from this codebase always carry `train_*`, `val_*`,
  `test_*` regret/MSE/fairness columns (see commit `2598462`) and, for MD,
  per-group `*_benefit_mean_r{j}`, `*_cost_mean_r{j}`, `*_decision_mean_r{j}`
  diagnostics (see commit `3cab494`).
