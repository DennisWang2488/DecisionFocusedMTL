# AlignMO next-session status

Lightweight tracker for the three tracks the user approved on 2026-04-23.
The canonical plan lives in `ALIGNMO_PLAN.md`; this file is a working
checklist for the in-progress scale-up work.

Updated as tasks land.

---

## Track 1 — Healthcare full-cohort analytic AlignMO comparison

**Goal.** Reproduce the n=1000 pilot/eval AlignMO-vs-fixed comparison at
the full 48,784-patient cohort to check whether the SPSA-regime
advantage carries over when gradients are clean and n is large.

**Pinned settings (user, 2026-04-23).**
- task: `medical_resource_allocation`, full cohort (`n_sample=0`),
  fairness `mad` only, analytic backend only.
- methods: 7 v2 methods (`FPTO, FDFL-Scal, FPLG, PCGrad, MGDA, SAA, WDRO`)
  + `AlignMO` (8 total). **Drop** `FDFL-0.1`, `FDFL-0.5`, `FDFL`.
- α ∈ {0.5, 1.5, 2.0, 3.0}; seeds {11, 22, 33}; λ ∈ {0.0, 0.5, 1.0, 2.0};
  `steps_per_lambda=70`; `force_lambda_path_all_methods=True` so PCGrad,
  MGDA, AlignMO do the same λ sweep.
- Reuse `results/advisor_review/healthcare_followup_v2/variant_a/mad/`
  for the 7 fixed methods at α∈{0.5, 2.0}, seeds {11, 22, 33}.

**TODO.**
- [ ] T1.1 Write `experiments/advisor_review/run_alignmo_hc_full.py`:
      run AlignMO at α∈{0.5, 2.0} + (7 methods + AlignMO) at α∈{1.5, 3.0}.
      Output under `results/alignmo_eval_full_hc/<alpha>/seed_<s>/`.
- [ ] T1.2 Execute the launcher; log to
      `results/alignmo_eval_full_hc/run_log.txt`.
- [ ] T1.3 Aggregate: union existing v2 numbers (α∈{0.5, 2.0}, seeds
      {11, 22, 33}) with the fresh α∈{1.5, 3.0} runs; emit
      `grand_summary.csv` + `acceptance_full_hc.md` (same shape as
      `results/alignmo_eval/acceptance.md`).

**SPSA full-cohort healthcare: deferred** (user, 2026-04-23). Not this
session.

---

## Track 2 — Knapsack fairness×imbalance ablation

**Goal.** Complete the "adding fairness helps" story: show how each
method behaves as we vary the two independent imbalance axes on the MD
knapsack task, across all four prediction fairness measures. This is a
*separate ablation*, not the main AlignMO-vs-fixed comparison; its
method list is intentionally minimal.

**Pinned settings (user, 2026-04-23).**
- task: `md_knapsack`, `n_train=200`, SPSA backend (`n_dirs=8,
  eps=1e-3`), `steps_per_lambda=30`, λ ∈ {0.0, 0.5} (least-time config).
- methods (user-approved, 6): `AlignMO, FPTO, FDFL, SAA, WDRO, FDFL-Scal`.
- fairness types (all 4): `mad, dp, atkinson, bias_parity`.
- α ∈ {0.5, 2.0}. Seeds {11, 22, 33}.
- (benefit_group_bias, cost_group_bias) ∈ {0.0, 0.3, 0.6}² — 9 cells,
  "almost fair / mild / most unfair" ≈ {0.0, 0.3, 0.6} as in the existing
  hypothesis 3b grid.
- **Existing hypothesis 3b cells are NOT reused** (different method set,
  mad-only, λ∈{0, 0.5} but α=0.5 only, missing `bb=0.6 cb=0.6`); we
  rerun the full ablation fresh so it is internally self-consistent.

**Budget note.** 6 methods × 3 seeds × 2 λ × 30 steps × 9 imbalance ×
4 fairness × 2 α ≈ 2592 stage rows. Need to **benchmark one cell first**
(T2.1) before committing to the full sweep; if per-cell wall is too
high at n=200, propose a reduction to the user.

**TODO.**
- [ ] T2.1 Benchmark 1 cell (`α=2.0, mad, bb=0.3, cb=0.3, seed=11`,
      6 methods, 2 λ, SPSA, n=200, 30 steps). Record wall time.
- [ ] T2.2 If benchmark ≤ ~20 min for full sweep, write
      `experiments/advisor_review/run_knapsack_imbalance_ablation.py`.
      Else, negotiate with user on scope reduction.
- [ ] T2.3 Execute the ablation; log to
      `results/alignmo_knapsack_imbalance/run_log.txt`.
- [ ] T2.4 Aggregate into `grand_summary.csv` and write
      `imbalance_alignmo.md` with per-(fairness, α) 3×3 tables of
      regret & fairness, plus a per-method ranking.

---

## Track 3 — FOLD-opt gradient backend exploration

**Goal.** "See if fold-opt has better gradients than SPSA" in the regime
where SPSA historically blew up on the knapsack task (the same
instability that motivated the µ-anchor in FDFL and the normalize
option in PCGrad).

**Pinned settings (user, 2026-04-23).**
- Small MD knapsack, `n_train=50` (very small so gradient dynamics are
  readable).
- Compare: per-objective gradient norms + regret trajectory on
  `{AlignMO, FDFL, PCGrad}` at α=2.0, MAD, SPSA vs fold-opt vs analytic
  (analytic may not be available for MD knapsack; will confirm while
  reading the repo).
- **Read the fold-opt codebase first** (user-requested) before wiring.

**TODO.**
- [ ] T3.1 Read `H:\myREPO\Fairness-Decision-Focused-Loss\fold-opt-package`
      README + main module; produce a concrete integration plan:
      (a) public API / entry point; (b) what inputs fold-opt needs
      (decision, objective, constraints); (c) expected output shape
      (is it per-sample parameter gradients? or per-prediction gradients?);
      (d) licensing / import considerations; (e) estimated effort to wire
      as `decision_grad_backend="fold_opt"` in
      `src/fair_dfl/decision/factory.py`.
- [ ] T3.2 Report back to user with T3.1 plan (no integration work yet).
- [ ] T3.3 If user approves: add `FoldOptDecisionGradient` under
      `src/fair_dfl/decision/` following the SPSA/analytic pattern; add
      to the factory; add a minimal test.
- [ ] T3.4 Write `experiments/advisor_review/run_fold_opt_smoke.py`:
      n_train=50, MD knapsack, α=2, MAD, SPSA vs fold-opt on
      {AlignMO, FDFL, PCGrad}, 3 seeds, small λ grid, 30 steps. Log
      per-step gradient norms and final test regret.
- [ ] T3.5 Plot/diff: `fig_fold_opt_vs_spsa.png` (grad norms over
      training) + `fold_opt_report.md` (headline table, one paragraph).

---

## Execution log (append as work lands)

- [ ] ~~task not started~~
