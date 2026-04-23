# AlignMO — Plan and Execution Log

> **Canonical work plan for the AlignMO contribution to the IJOC submission.**
> Every implementing agent that touches this file must keep **Section 10
> (Status Log)** and **Section 2 (Current Status)** in sync with what was
> actually done. Rules for updating the file are in Section 8.

---

## 1. Context

**Project.** `E:\Codes\DecisionFocusedMTL\`, branch
`fair-dfl/empirical-followup`. The repo backs an IJOC submission titled
"End-to-End Fairness Optimization with Decision Focused Multi-Task Fair
Learning" (working title), currently at `results/advisor_review/paper/draft_v2.tex`.

**Where we are in the review cycle.** The advisor-review pass identified
that the paper's empirical story ("MOO helps / hurts conditionally on
gradient regime") is currently an *observation* rather than an
*artifact*. To convert it into a non-incremental IJOC contribution, we
are adding a new adaptive multi-objective gradient handler called
**AlignMO** that routes between **four modes** (the cross product of two
independent binary decisions: normalize? project?), online, using two
cheap diagnostics (pairwise gradient cosines and log-scale ratios).
Target: a new Section 4 in the manuscript + reproducible code +
empirical evidence that AlignMO dominates each fixed mode.

**What AlignMO is, in one paragraph.** A `MultiObjectiveGradientHandler`
subclass (same base class as `PCGradHandler`, `MGDAHandler`,
`CAGradHandler` in `src/fair_dfl/algorithms/mo_handler.py`). At each
step it receives per-objective gradients `g_dec, g_pred, g_fair`, updates
EMAs of pairwise cosines `c_dp, c_df, c_pf` and log-scale ratios
`r_dp, r_df`, and makes two **independent** binary decisions:

- **Decision A (scale).** If `max(|r_dp|, |r_df|) > τ_scale` → normalize
  each gradient to unit norm, substitute `μ_eff = max(μ, μ_floor)`, and
  multiply the final direction by the mean of the original norms.
  Otherwise: use raw gradients and `μ_eff = μ`.
- **Decision B (direction).** If `min(c_dp, c_df, c_pf) < τ_conflict` →
  apply PCGrad projection to the (possibly normalized) weighted
  gradients. Otherwise: weighted sum.

The two decisions compose into **four named modes**:

| Scale | Direction | Mode name | Direction formula |
| --- | --- | --- | --- |
| balanced | compatible | `scalarized` | `g_dec + μ·g_pred + λ·g_fair` |
| balanced | conflict | `projected` | `PCGrad(g_dec, μ·g_pred, λ·g_fair)` |
| imbalanced | compatible | `anchored` | `s̄ · (ĝ_dec + μ_eff·ĝ_pred + λ·ĝ_fair)` |
| imbalanced | conflict | `anchored_projected` | `s̄ · PCGrad(ĝ_dec, μ_eff·ĝ_pred, λ·ĝ_fair)` |

where `ĝ = g/‖g‖` and `s̄ = mean(‖g_dec‖, ‖g_pred‖, ‖g_fair‖)`.

Defaults: `τ_conflict=-0.1, τ_scale=2.0, μ_floor=0.1, β_ema=0.9,
T_warmup=10`. Note: `τ_align` is no longer needed — the alignment
decision is now a single threshold (`τ_conflict`). Warmup forces the
`scalarized` fallback.

**Relevant prior pieces in the codebase.**
- Closed-form group α-fair decision: `src/fair_dfl/tasks/medical_resource_allocation.py:198` (`_solve_group`) + Jacobian + O(n) VJP. Sign-bug already patched and tested for α∈{0.3,…,5.0}.
- Existing MOO handlers: `src/fair_dfl/algorithms/mo_handler.py` (WeightedSum, PCGrad-with-optional-normalize, MGDA, CAGrad, FAMO, PLG3).
- Training loop: `src/fair_dfl/training/loop.py` (handler is called in `compute_direction`; diagnostics land in `iter_logs.csv`).
- Method registry: `experiments/configs.py` (`ALL_METHOD_CONFIGS`). Add `"AlignMO"` here.
- Decision-gradient backend factory: `src/fair_dfl/decision/factory.py` (`"analytic"` vs `"spsa"` etc). Healthcare supports both via `solve_decision` / `evaluate_objective`.
- Test suite: `tests/` (currently 108 passing tests; AlignMO must stay green).

---

## 2. Current Status

**Phase.** 🟢 Phase 2 — AlignMO refactored to the 4-mode /
2-binary-decision framework (TODO 2.1d done); 121 tests pass
(including 6 new cases in TODO 2.3c). Eval sweep now running.

**Decision gate.** Satisfied. `results/pilot_alignmo/GO_NO_GO.md`
records a **GREEN** verdict (`4` distinct cell-winners out of `8`), so
Phase 2 may proceed as planned under Section 5.

**⚠️ Important refinement (2026-04-23).** The initial AlignMO
implementation routes by priority `pcgrad > anchored > scalarized`,
which is incorrect when both scale imbalance AND cosine conflict fire:
it applies PCGrad on raw (unnormalized) gradients, reproducing the
exact knapsack failure mode we identified in the paper. The refactor
decomposes the routing into **two independent binary decisions**
(normalize? project?) and composes them. Detail in Section 1; new
TODOs in Section 5.3 (2.1d, 2.3c). **Do NOT run the evaluation sweep
(TODO 2.4) on the current handler — refactor first.**

**Most recent action.** Stopped the background full-cohort healthcare
`spsa` backend-sanity run at the user's request. Summarized the completed
full-cohort `analytic` backend-sanity run (seed `1`, `lambdas={0,1}`):
`PCGrad` wins at `alpha=0.5` and `alpha=1.5`, `FDFL-0.1` narrowly wins
at `alpha=2.0` over `FPLG`, and `FPTO` wins again at `alpha=3.0` while
several decision-focused methods exhibit severe instability / blow-up.
Full-cohort `finite_diff` remains intentionally deferred; if revisited,
only do it as a very small-sample control (for example `n_sample=100`).

**Next action.** Advisor decision needed on Phase 2 acceptance.
`results/alignmo_eval/acceptance.md` shows:
- **Do-good: 3/8 cells** (strict wins on SPSA α∈{0.5, 1.5, 2.0});
- **Do-no-harm: 5/8 cells** — fails on analytic α=0.5 (+0.013, 10σ),
  α=1.5 (+0.0003, 8σ over a near-zero std), α=3.0 (+2.06, 4.4σ,
  a clear divergence);
- **Avg rank: AlignMO 3.00 (2nd) vs FPTO 2.50 (1st)**; AlignMO beats
  FPLG, PCGrad, all FDFL-* and MGDA;
- **Mode trace: all 4 modes used** (analytic low-α → anchored_projected
  dominant; SPSA all α → anchored dominant 0.74).
Two forks: (A) run TODO 2.7 sensitivity (vary `tau_conflict`/
`tau_scale` ±50%) before Phase 3 to see if defaults move the
analytic-α=3.0 cell out of blow-up; or (B) reframe Phase 2 claim as
"AlignMO strictly dominates on the realistic SPSA backend + holds
best-or-near-best avg rank on analytic" and move to Phase 3 (Section
4 draft).

---

## 3. Goals

**Headline.** Ship an adaptive multi-objective gradient handler, AlignMO,
that (i) Pareto-dominates each fixed handler (FDFL-Scal, FDFL-μ-anchored,
PCGrad, MGDA, FPLG) on the healthcare and knapsack benchmarks, and (ii)
whose mode selection is interpretable via a mode-trace figure.

**Supporting.**
- Add a new Section 4 ("AlignMO: Adaptive Multi-Objective Gradient
  Handling") to `draft_v2.tex`.
- Preserve all existing empirical results in Sections 5.1–5.2.
- Demonstrate robustness: defaults shared across both tasks; no per-task
  threshold tuning.

**Non-goals.**
- No new closed-form derivation.
- No new tasks beyond the two already in the paper.
- No new theoretical propositions (IJOC venue; theorems are explicitly
  deferred).
- No changes to the training loop's public API.

---

## 4. Phase 1 — Go/No-Go Pilot (START HERE)

### 4.1 Purpose

Before any AlignMO code is written, verify that **different fixed
handlers win on different cells** of (α × gradient regime) on the
healthcare task. If one handler wins everywhere, AlignMO's adaptive
framing collapses and the paper must be repositioned.

### 4.2 Scope

- **Task.** Healthcare only.
- **Subsample.** `n_sample = 1000` (smaller than the main-text 5000 for
  speed; the relative ranking of methods should be preserved at this
  size).
- **Fairness type.** `mad` only.
- **Code used.** No new handler code. Only the existing methods
  registered in `experiments/configs.py`.

### 4.3 Grid

| Axis | Levels | Notes |
| --- | --- | --- |
| `alpha_fair` | {0.5, 1.5, 2.0, 3.0} | α=1.5 and 3.0 are new cells; α=3 is now safe post closed-form fix. |
| `decision_grad_backend` | `analytic`; `spsa` with `n_dirs=8, eps=1e-3` | Two regimes on the SAME task isolates regime from task. (High-noise SPSA deferred unless analytic+low-SPSA show monotone winners.) |
| `method` | `FPTO, FDFL-Scal, FDFL-0.1, FDFL-0.5, FDFL, FPLG, PCGrad, MGDA` | 8 methods. `FPTO` is the non-decision-focused baseline. |
| `lambdas` | {0.0, 0.5, 1.0, 2.0} | Reuse existing sweep logic; report best-λ per cell. |
| `seeds` | {11, 22, 33} | 3 seeds for speed; escalate only if gaps are within noise. |
| `steps_per_lambda` | 50 | vs 70 in main text. |

Total stage rows: 4 × 2 × 8 × 3 × 4 = **768 stages**. Expected wall-clock
on CPU: analytic ≈ 1.5 h, SPSA-low ≈ 2 h. Under 4 h end-to-end.

### 4.4 Decision rule (commit to this BEFORE looking at the data)

| # of distinct methods that win ≥1 cell (out of 8 cells) | Verdict | Next step |
| --- | --- | --- |
| ≥ 4 | ✅ GO | Phase 2 as planned |
| 2–3 | ⚠️ CAUTIOUS GO | Phase 2 with tightened framing ("AlignMO routes between the 2–3 empirically separated regimes") |
| 1 | ❌ NO GO | Drop AlignMO. Reposition the paper around the dominant fixed method. Abort this plan. |

A secondary check — *can the diagnostics drive the adaptation?* — runs
after the verdict. Write `results/pilot_alignmo/diagnostic_profile.csv`
with median `c_dp, c_df, c_pf, r_dp, r_df` per (α, regime). If
diagnostics do NOT separate cells whose winners differ, Phase 2 still
runs but the framing weakens (we'd be claiming AlignMO routes correctly
*by construction*, not by observation).

### 4.5 Deliverables

Under `results/pilot_alignmo/`:
- `grand_summary.csv` — one row per (method, α, regime, seed, λ). Must
  include the diagnostic EMAs that already exist in `iter_logs.csv`
  (aggregated per stage).
- `per_cell_winners.csv` — one row per (α, regime): `best_method,
  best_regret_mean, best_regret_std, runner_up_method, gap,
  gap_seed_std_ratio`.
- `diagnostic_profile.csv` — one row per (α, regime, method): medians of
  the diagnostic EMAs.
- `fig_per_cell_winners.png` — 4×2 heatmap, colored by winning method.
- `GO_NO_GO.md` — single-page memo with the verdict, counts, and either
  a green/yellow/red banner and the reason.

### 4.6 TODO checklist — Phase 1

- [x] **1.1** Write `experiments/advisor_review/run_alignmo_pilot.py`.
  Reuse `run_healthcare_v2_variant_a.py` as a template (same data
  loading, same runner harness). Parameterize the grid from Section 4.3.
  Make `n_sample=1000`, `steps_per_lambda=50` configurable from CLI.
- [x] **1.2** Smoke-test on 1 cell (`α=2.0, analytic, FDFL-Scal, seed=11,
  λ=0.5`). Confirm it writes a valid `stage_results.csv`. Time it.
- [x] **1.3** Run the full grid. Recommended order: analytic regime first
  (fast failure surface), then SPSA. Log to `results/pilot_alignmo/run_log.txt`.
- [x] **1.4** Aggregate stage rows into `grand_summary.csv` using an
  aggregation script patterned on `paper_summary_v2a.py`.
- [x] **1.5** Build `per_cell_winners.csv`. For each (α, regime) cell:
  pick `best_method = argmin over methods of (min over λ of test_regret_normalized_mean)`.
  Record gap and seed-std ratio.
- [x] **1.6** Build `diagnostic_profile.csv` from the per-step EMAs
  already logged in `iter_logs.csv`.
- [x] **1.7** Render `fig_per_cell_winners.png`: 4 rows (α), 2 cols
  (regime), cell colored by winning method. Use a shared legend.
- [x] **1.8** Write `GO_NO_GO.md` memo: count distinct winners, apply
  decision rule from Section 4.4, call the verdict, and list the
  per-cell results in a compact table.
- [x] **1.9** Update this file's Section 2 with the verdict; append to
  Section 10 status log.

### 4.7 Risks and escalations for Phase 1

- **Risk:** SPSA on healthcare has never been exercised at n=1000.
  Mitigation: smoke-test (TODO 1.2) catches this early.
- **Risk:** Gaps within one seed-std (no clear winner). Mitigation:
  escalate ambiguous cells to 5 seeds before calling no-go.
- **Risk:** FDFL (μ=0) diverges on SPSA. Expected. Record as NaN; do
  not let it invalidate other methods' rankings.

---

## 5. Phase 2 — Build AlignMO (CONDITIONAL on pilot verdict)

*Do not start Phase 2 until Phase 1's GO_NO_GO memo is GREEN (or YELLOW
with an explicit decision from the advisor).*

### 5.1 Subtasks

| # | Task | Primary file(s) |
| --- | --- | --- |
| 2.1 | Implement `AlignMOHandler` | `src/fair_dfl/algorithms/mo_handler.py` |
| 2.2 | Register `"alignmo"` method and thread hyperparameters | `experiments/configs.py`, `src/fair_dfl/training/loop.py` |
| 2.3 | Unit tests | `tests/test_mo_handlers.py` |
| 2.4 | Evaluation script over the same pilot grid, now including AlignMO | `experiments/advisor_review/run_alignmo_eval.py` |
| 2.5 | Mode-trace figure + comparison table | `experiments/advisor_review/plot_alignmo_mode_trace.py`, `paper_alignmo_tables.py` |
| 2.6 | Cross-task evaluation on knapsack (n=300 default) | Extend the existing knapsack launcher |

### 5.2 AlignMOHandler contract (for TODO 2.1 / REVISED for 2.1d)

```
class AlignMOHandler(MultiObjectiveGradientHandler):
    """
    Adaptive multi-objective gradient handler for decision-focused learning.

    At each step, maintains EMAs of pairwise cosines c_dp, c_df, c_pf and
    log-scale ratios r_dp, r_df over the last ~1/(1-beta_ema) steps, and
    makes TWO INDEPENDENT BINARY DECISIONS that compose into 4 modes:

        Decision A — normalize?
            imbalanced := max(|r_dp|, |r_df|) > tau_scale
            If imbalanced: normalize each g_j to unit norm,
                           mu_eff = max(mu, mu_floor),
                           post_scale = mean(||g_dec||, ||g_pred||, ||g_fair||)
            Else:          use raw g_j, mu_eff = mu, post_scale = 1.0

        Decision B — project?
            conflict := min(c_dp, c_df, c_pf) < tau_conflict
            If conflict: direction = PCGrad(g_dec, mu_eff*g_pred, lam*g_fair)
                         (applied to the possibly-normalized gradients)
            Else:        direction = g_dec + mu_eff*g_pred + lam*g_fair
                         (applied to the possibly-normalized gradients)

        Return post_scale * direction.

    The cross product yields four modes, logged for interpretability:
        (balanced, compatible) = "scalarized"
        (balanced, conflict)   = "projected"
        (imbalanced, compatible) = "anchored"
        (imbalanced, conflict)   = "anchored_projected"

    Cold start (t < T_warmup) forces mode = "scalarized".

    Parameters
    ----------
    tau_conflict : float, default -0.1   # min pairwise cosine threshold
    tau_scale : float, default 2.0       # max |log norm ratio| threshold
    mu_floor : float, default 0.1        # prediction anchor under imbalance
    beta_ema : float, default 0.9        # EMA decay
    T_warmup : int, default 10           # steps before trusting diagnostics

    NOTE: tau_align is REMOVED. The alignment decision is now a single
    threshold (tau_conflict), because the 2-binary-decision framing does
    not need a hysteresis band — EMA smoothing handles near-boundary
    thrashing.

    Logged diagnostics (self._last_diag):
        mode_this_step,                      # one of 4 strings above
        regime_scale ∈ {"balanced", "imbalanced"},
        regime_direction ∈ {"compatible", "conflict"},
        c_dp, c_df, c_pf, r_dp, r_df,
        mu_eff_used, post_scale_used,
        n_projections (nonzero only when projection fires),
        n_mode_switches_so_far
    """
```

### 5.3 TODO checklist — Phase 2

*Items 2.1a–2.3b were completed against the earlier 3-mode-exclusive
contract. Items 2.1d and 2.3c below are the refactor required to move
to the 4-mode / 2-binary-decision framework. All subsequent items
(2.4+) run on the refactored handler.*

- [x] **2.1a** ~~Implement `AlignMOHandler`~~ (done; needs refactor via 2.1d)
- [x] **2.1b** ~~Log diagnostics to `iter_logs.csv`~~ (done; needs extended logging via 2.1d)
- [x] **2.2a** ~~Register `"alignmo"` in factory~~
- [x] **2.2b** ~~Add `"AlignMO"` to `ALL_METHOD_CONFIGS`~~
- [x] **2.2c** ~~Thread `alignmo_*` kwargs through training-loop config~~
- [x] **2.3a** ~~Unit tests (6 cases)~~
- [x] **2.3b** ~~`pytest tests/ -x -q` green at ≥ 114 tests~~
- [ ] **2.1d** ⚠️ **(NEW, BLOCKING)** Refactor `AlignMOHandler.compute_direction`
  from 3-mode-exclusive (`pcgrad > anchored > scalarized`) to
  2-binary-decision composition per the revised contract in 5.2.
  Structure the body as:
  ```
  if step < T_warmup:
      mode = "scalarized"; normalize = False; project = False
  else:
      normalize = max(|r_dp|, |r_df|) > tau_scale
      project   = min(c_dp, c_df, c_pf) < tau_conflict
      mode = {
         (False, False): "scalarized",
         (False, True):  "projected",
         (True,  False): "anchored",
         (True,  True):  "anchored_projected",
      }[(normalize, project)]
  # apply normalization if needed, then projection if needed
  ```
  Remove `tau_align` from `__init__` and all call sites. Remove
  `mo_alignmo_tau_align` from logged diagnostics. Add
  `regime_scale`, `regime_direction`, `post_scale_used` to diagnostics.
- [ ] **2.3c** ⚠️ **(NEW, BLOCKING)** Unit tests covering the 4-mode
  framework. Six new tests (on top of the 6 that already pass):
  (i) balanced + compatible → scalarized with no projection, no
  normalization;
  (ii) balanced + conflict → projected, direction equals PCGrad on raw
  gradients;
  (iii) imbalanced + compatible → anchored, direction is the unit-sum
  scaled by mean_norm;
  (iv) **imbalanced + conflict → anchored_projected, direction is
  PCGrad applied to unit-norm gradients, scaled by mean_norm** (this
  is the case the original handler mis-routed);
  (v) mode-switch logging: when a run traverses cells (A,B) → (A,¬B)
  → (¬A,¬B), the 4 fields `regime_scale`, `regime_direction`,
  `mode_this_step`, `n_mode_switches_so_far` should all be correct;
  (vi) removing `tau_align` from config does not break existing
  instantiation paths. Rerun `pytest tests/ -x -q`; confirm ≥ 120 tests.
- [ ] **2.4a** Write `run_alignmo_eval.py` = pilot grid + AlignMO added.
  **Do not run before 2.1d and 2.3c are green.**
- [ ] **2.4b** Execute the eval and aggregate into
  `results/alignmo_eval/grand_summary.csv`.
- [ ] **2.5a** `fig_alignmo_mode_trace.png`: for each (α, regime), show
  the fraction of training steps AlignMO spent in each of the 4 modes,
  averaged over seeds. **Now shows 4 mode bars per cell, not 3.**
- [ ] **2.5b** `table_alignmo_vs_best_fixed.tex`: per-cell regret for
  AlignMO vs the pilot's per-cell winner, with win/tie/loss.
- [ ] **2.6** Extend to the knapsack task (`n_train=300`, α=2, MAD, SPSA).
  Rerun the main knapsack configuration with AlignMO added; save to
  `results/alignmo_eval/knapsack/`. This is the cell where
  `anchored_projected` is expected to appear — report its fraction
  explicitly.
- [ ] **2.7** Sensitivity ablation: vary `tau_conflict` and `tau_scale`
  by ±50%; rerun a small sub-grid (α=2, both regimes, 3 seeds) to show
  AlignMO is not overly sensitive. Document in
  `results/alignmo_eval/sensitivity.md`.
- [ ] **2.7b** λ-selection ablation per
  `experiments/advisor_review/LAMBDA_ABLATION_NOTE.md`. Compares
  AlignMO (main, fixed λ sweep) vs AlignMO-free (MGDA-style auto λ)
  vs AlignMO-FAMO on Pareto coverage and operating-point utility.
  Expected outcome: main dominates on ≥2/3 fairness budgets.
- [ ] **2.7c** Projection-primitive ablation. Inside AlignMO's
  `projected` and `anchored_projected` cells, swap the default (PCGrad)
  for each of {MGDA, CAGrad, FAMO, Nash-MTL}. MGDA/CAGrad/FAMO are
  already in `mo_handler.py`; **Nash-MTL needs a new ~200-line
  implementation** (see contract below). Grid: healthcare n=1000,
  4 α × 2 regime × 3 seeds × 3 λ = 72 runs per primitive, plus the
  same on knapsack n=300 SPSA cells. Report metrics: (i) per-cell
  regret vs PCGrad, (ii) catastrophic-failure rate on SPSA (PCGrad's
  1/9 benchmark), (iii) mode-trace stability. Deliverable:
  `results/alignmo_eval/projection_primitive_ablation.md` with
  decision on default. If one primitive strictly dominates PCGrad,
  switch AlignMO's default and update Section 4 of the paper to
  name it. If roughly tied, keep PCGrad and document the comparison.
  **Prerequisite:** 2.1d + 2.3c done, 2.4b run at least once with
  PCGrad default.

  **Nash-MTL implementation contract.** Given gradients
  $g_1, \ldots, g_m$, Nash-MTL seeks a direction $d$ that maximizes
  the product (geometric mean) of per-objective projections, not the
  sum:
  $$\max_{d:\|d\|\le 1} \prod_i g_i^\top d = \max_{d:\|d\|\le 1} \sum_i \log(g_i^\top d).$$
  Dual form: $d = \frac{\sum_i \alpha_i g_i}{\|\sum_i \alpha_i g_i\|}$
  where $\alpha_i$ are Nash bargaining weights satisfying
  $\alpha_i \cdot g_i^\top d = 1$ for all $i$. Fixed-point iteration
  converges in 2–3 steps for $m=3$ objectives. Alternative: solve as
  a small SOCP via `cvxpy` (overkill for $m=3$ but robust).
  Reference: Navon et al 2022 (ICML), "Multi-Task Learning as a
  Bargaining Game". Implementation goes in `mo_handler.py` as a new
  `NashMTLHandler(MultiObjectiveGradientHandler)` subclass. Expose
  as `mo_method="nash_mtl"` in the registry. Add unit tests mirroring
  `TestPCGradHandler`'s structure.

- [ ] **2.8** Update this file's Section 2; append to status log.

*Theory track (optional, runs in parallel with Phase 2 once 2.4b lands):*

- [ ] **2.9** Execute the prompt in
  `experiments/advisor_review/THEOREM_PROMPT.md` to produce
  `results/advisor_review/paper/appendix_theorem.tex` (≤2 pages,
  subsequential Pareto-stationarity with M-estimation bridge lemma).
  Hand off to theory-oriented collaborator or dedicated session.

### 5.4 Phase 2 acceptance criteria

Before declaring Phase 2 complete:
- [ ] 108+ tests pass.
- [ ] **Do-no-harm:** AlignMO within 1 seed-std of the per-cell best
  fixed handler on every (α, regime) cell.
- [ ] **Do-good:** AlignMO strictly dominates the per-cell best on ≥ 2
  cells OR has the lowest average rank across all cells.
- [ ] Mode-trace figure visibly shows AlignMO picking different modes
  for different (α, regime) pairs (not collapsing to one mode).
- [ ] Defaults unchanged across healthcare and knapsack; no per-task
  tuning.

---

## 6. Phase 3 — Paper draft (Section 4)

*Start after Phase 2 acceptance criteria pass.*

### 6.1 Where it lands

Insert a new Section 4 in `draft_v2.tex` between the current
"Fair Decision Focused Learning Algorithm" section (Section 4 in current
numbering, which will become Section 3.4 or remain as Section 4's
Subsection 4.1) and the "Numerical Experiments" section. Renumber
existing sections accordingly.

### 6.2 Section skeleton

```
\section{AlignMO: Adaptive Multi-Objective Gradient Handling}
\label{sec:alignmo}

  \subsection{Motivation}          % ~0.5 pages
  \subsection{Two online diagnostics} % ~0.5 pages, 1 figure ref
  \subsection{Three modes}         % ~0.5 pages, 1 small table
  \subsection{The AlignMO algorithm} % ~0.5-1 page incl. pseudocode box
  \subsection{Computational cost}  % ~0.25 pages
  \subsection{Relation to existing handlers}  % ~0.25 pages
  \subsection{Empirical evaluation}  % ~0.75 pages, 1 table + 1 figure
  \subsection{Limitations}         % ~0.25 pages
```

Target total: **2.5–3 rendered pages**.

### 6.3 TODO checklist — Phase 3

- [ ] **3.1** Draft `results/advisor_review/paper/section_alignmo.tex`
  per the skeleton in 6.2, populated with concrete numbers from
  Section 5's evaluation.
- [ ] **3.2** Add `\input{section_alignmo}` in `draft_v2.tex` at the
  right location; renumber references.
- [ ] **3.3** Produce algorithm environment: pseudocode from Section 5.2,
  typeset with `algpseudocode`.
- [ ] **3.4** Build LaTeX tables from `table_alignmo_vs_best_fixed.tex`
  and figure references for `fig_alignmo_mode_trace.png`.
- [ ] **3.5** Re-read Section 5 (old experiments section) and update any
  claims that are now weaker than AlignMO's (e.g., "FDFL-PCGrad achieves
  the lowest regret at α=0.5" → "AlignMO achieves the lowest regret at
  α=0.5, routing to PCGrad mode as predicted by Figure …").
- [ ] **3.6** Update Abstract and Introduction contribution bullets to
  include AlignMO as a headline contribution.
- [ ] **3.7** Update this file's Section 2 and status log.

---

## 7. Acceptance criteria (whole project)

- [ ] Phase 1 GO_NO_GO verdict is GREEN or YELLOW with explicit advisor
  go-ahead.
- [ ] Phase 2 eval shows AlignMO meets both do-no-harm and do-good
  criteria (Section 5.4).
- [ ] `pytest tests/ -x -q` passes (≥114 tests).
- [ ] `draft_v2.tex` compiles with AlignMO section included, no unresolved
  `??` references, no blue scaffolding text in the new section.
- [ ] Abstract + intro + conclusion mention AlignMO consistently.

---

## 8. Operating rules for the implementing agent

1. **Read the whole file before starting.** Sections 1–7 set context;
   Section 4 or 5 tells you what to do next depending on phase.
2. **Work TODOs in order within a phase.** Do not jump ahead to a later
   phase without written permission (advisor's call, recorded in the
   log).
3. **After every meaningful action, update this file:**
   - Update Section 2 (*Current Status*) in place: current phase, most
     recent action, next action.
   - Prepend a dated entry to Section 10 (*Status Log*). Format:
     ```
     ### YYYY-MM-DD HH:MM — <one-line title>
     **Done.** <1–3 sentences on what was accomplished.>
     **Next.** <1 sentence on the immediate next step.>
     **Notes / risks.** <optional; any findings, surprises, blockers.>
     **Artifacts.** <optional; list new files produced, with paths.>
     ```
4. **Never delete log entries.** Append or prepend, but preserve history.
5. **Never modify** `_solve_group`, `_solve_group_grad_jacobian`, or
   `_solve_group_vjp` in `medical_resource_allocation.py`. These are
   post-fix, post-test, and out of scope.
6. **Never modify** the main-text empirical tables in `draft_v2.tex`
   (Tables A.1, A.2, A.3). They are paper-cited.
7. **If a TODO is blocked**, add it to Section 9 (*Open Questions /
   Risks*) with a one-line description and a proposed resolution. Do
   not silently skip it.
8. **Default seeds:** 11, 22, 33 unless the task requires more. Keep
   additional seeds explicit in the log entry.
9. **Commit discipline:** commit after each top-level TODO completes
   (e.g., "phase1: pilot launcher"), never amend. Never push without
   asking.
10. **Scope discipline:** if a task in the codebase seems to need
    fixing but isn't in the checklist, add it to Section 9 and
    continue with the current TODO. Don't yak-shave.

---

## 9. Open questions / risks

*Add items here as they arise. Each item: short title, one-line
description, proposed resolution, owner.*

- **Healthcare SPSA sanity probe.** Pilot Phase 1 found `FPTO` winning
  all healthcare SPSA cells, which is weaker than expected from prior
  intuition. Proposed resolution: run a one-seed full-cohort backend
  comparison (`analytic` vs `finite_diff` vs `spsa`) at `mad`,
  alphas `{0.5,1.5,2.0,3.0}`, `steps=70`, `lambdas={0,1}` from the
  CUDA-capable `main` env before over-interpreting the pilot's SPSA
  story. Owner: implementing agent.

- **Is PCGrad the right primitive for the "project" branch?** AlignMO
  currently uses PCGrad (Yu et al 2020) as its projection primitive.
  Newer MOO methods exist — Nash-MTL (Navon et al 2022), FAMO (Liu
  et al 2024), DB-MTL (Lin et al 2023) — with different theoretical
  profiles. Two questions: (a) Should AlignMO's project-branch use a
  newer handler instead of PCGrad? (b) Should we offer variants
  `AlignMO-Nash`, `AlignMO-FAMO`, etc., and benchmark them? Proposed
  resolution: **after TODO 2.1d refactor lands**, run a small ablation
  (one α, one regime, 3 seeds) swapping PCGrad for CAGrad (already in
  codebase) and MGDA in the project-branch. If one variant strictly
  dominates PCGrad on our setting, switch the default. Document in
  `results/alignmo_eval/projection_primitive_ablation.md`. Owner:
  implementing agent. Raised 2026-04-23.

- **Lambda auto-selection ablation.** Whether `λ` should be user-
  specified or auto-selected MGDA-style. Design note and protocol
  in `experiments/advisor_review/LAMBDA_ABLATION_NOTE.md`.
  Corresponding TODO: 2.7b. Owner: implementing agent. Raised 2026-04-23.

- **Theorem track.** Subsequential Pareto-stationarity proposition
  with an M-estimation bridge lemma for non-decomposable fairness.
  Prompt and scope in `experiments/advisor_review/THEOREM_PROMPT.md`.
  Corresponding TODO: 2.9. Owner: theory collaborator or dedicated
  session. Raised 2026-04-23.

---

## 10. Status Log

*Prepend new entries. Append-only below this line.*

### 2026-04-23 13:05 — TODO 2.4 done: Phase 2 eval aggregated (mixed verdict)
**Done.** Background sweep (`bgl2yt97j`) wrote 96 AlignMO stage rows
under `results/alignmo_eval/` (4 lambdas × 3 seeds × 4 alphas × 2
regimes) in 55 min. Wrote
`experiments/advisor_review/aggregate_alignmo_eval.py`: joins AlignMO
stages with the pilot's fixed-method stages and produces
`grand_summary.csv` (720 rows), `per_cell_alignmo_vs_best.csv`,
`mode_trace.csv`, `avg_rank.csv`, and a one-page `acceptance.md` memo.
**Phase 2 acceptance read:**
- Do-no-harm on every cell: **FAIL (5/8)**. AlignMO is within 1 seed-std
  of best fixed on analytic α=2.0 and all 4 SPSA cells, but loses on
  analytic α=0.5 (+0.013, 10σ), α=1.5 (+0.0003 over a ~0 std, 8σ),
  α=3.0 (+2.06, 4.4σ — a clear divergence).
- Do-good on ≥2 cells OR lowest avg rank: **PASS** (3/8 strict wins,
  all SPSA α∈{0.5, 1.5, 2.0}).
- Mode-trace diversity: **PASS** (all four modes fire — SPSA is
  `anchored`-dominant 0.74, analytic-low-α is `anchored_projected`-
  dominant 0.43–0.57, analytic-α=2 is `scalarized`/`projected` mix,
  analytic-α=3 is `scalarized` 0.65 + `anchored` 0.28).
- Overall avg rank: AlignMO 3.00 (2nd); FPTO 2.50 (1st); AlignMO beats
  FPLG, PCGrad, all FDFL-*, and MGDA.
**Headline read.** AlignMO cleanly dominates the realistic noisy
(SPSA) backend and is competitive-to-best on analytic α∈{1.5, 2.0}, but
breaks down at analytic α=3.0 (a numerically fragile regime for *all*
decision-focused methods). Stale 3-mode eval artefacts were
overwritten in place by the refactored run.
**Next.** Advisor call: (A) TODO 2.7 sensitivity first, or (B) reframe
the Section 4 claim to "AlignMO dominates the SPSA regime + near-best
avg rank on analytic" and proceed to Phase 3.
**Notes / risks.** The analytic-α=3.0 blow-up looks like it comes
from the `anchored` mode contributing 28% of steps there while
`anchored_projected` contributes 0% — so the handler is correctly
identifying scale imbalance but the rescaling-by-mean-norm step may
be over-amplifying an already-large decision gradient. Worth probing
with `tau_scale` sensitivity (TODO 2.7).
**Artifacts.**
- `experiments/advisor_review/aggregate_alignmo_eval.py`
- `results/alignmo_eval/grand_summary.csv` (720 rows, pilot + eval)
- `results/alignmo_eval/per_cell_alignmo_vs_best.csv`
- `results/alignmo_eval/mode_trace.csv`
- `results/alignmo_eval/avg_rank.csv`
- `results/alignmo_eval/acceptance.md`

### 2026-04-23 12:10 — TODO 2.1d + 2.3c done: 4-mode refactor, 121 tests
**Done.** Refactored `AlignMOHandler.compute_direction` in
`src/fair_dfl/algorithms/mo_handler.py` from the 3-mode-exclusive
priority logic (`pcgrad > anchored > scalarized`) to the
2-binary-decision composition per the revised Section 5.2 contract:
independently evaluate `normalize = |r| > tau_scale` and
`project = min(c) < tau_conflict`, then compose into
`{scalarized, projected, anchored, anchored_projected}`. Warmup forces
`(False, False) → scalarized`. Removed `tau_align` as a live parameter
(kept as a deprecated, ignored kwarg for backward compat); dropped
`mo_alignmo_tau_align` from `configs.py`, `loop.py`, and
`core_methods.py`. Added diagnostic fields `regime_scale`,
`regime_direction`, `post_scale_used` per contract. Updated
`tests/test_mo_handlers.py::TestAlignMOHandler` — refreshed the
original 6 cases (renamed mode strings: `pcgrad → projected`,
`anchored_stabilized → anchored`) and added 6 new cases per 2.3c:
(i) balanced+compatible ⇒ scalarized with numeric sum-check,
(ii) balanced+conflict ⇒ PCGrad-on-raw numerical match,
(iii) imbalanced+compatible ⇒ `mean_norm × (u_dec + mu_eff·u_pred +
lam·u_fair)` numerical match,
(iv) **imbalanced+conflict ⇒ `mean_norm × PCGrad(unit grads)`** (the
cell the old handler mis-routed on knapsack),
(v) 3-regime traversal with correct `regime_scale / regime_direction /
mode_this_step / n_mode_switches_so_far`,
(vi) deprecated `tau_align=…` kwarg still accepted without raising.
`pytest tests/ -x -q`: **121 passed** (was 115; +6 new).
**Next.** TODO 2.4b: background AlignMO-only eval sweep
(`results/alignmo_eval/run_log.txt`) writes 96 stage rows to
`results/alignmo_eval/`; aggregate and join with
`results/pilot_alignmo/grand_summary.csv` to produce Phase 2
acceptance numbers (do-no-harm / do-good).
**Notes / risks.** The earlier `--alignmo-only` run (against the stale
3-mode handler) was discarded. The relaunched sweep runs the
refactored 4-mode handler with `force_lambda_path_all_methods=True`
so AlignMO sweeps the full {0, 0.5, 1, 2} λ grid alongside the fixed
methods.
**Artifacts.**
- `src/fair_dfl/algorithms/mo_handler.py` (AlignMOHandler 4-mode refactor)
- `src/fair_dfl/training/loop.py`, `src/fair_dfl/algorithms/core_methods.py` (drop tau_align)
- `experiments/configs.py` (drop tau_align; AlignMO entry)
- `tests/test_mo_handlers.py` (6 refreshed + 6 new test cases)
- `experiments/advisor_review/run_alignmo_pilot.py`,
  `experiments/advisor_review/run_alignmo_eval.py`
  (force_lambda_path wiring)

### 2026-04-23 — Added TODO 2.7c: projection-primitive ablation
**Done.** Added TODO 2.7c in Section 5.3 for a projection-primitive
ablation inside the `projected` and `anchored_projected` cells.
Compares PCGrad (current default) with MGDA, CAGrad, FAMO, and
Nash-MTL. First three are already in the codebase; Nash-MTL requires
~200-line new implementation — contract documented inline.
**Next.** No change to immediate next action (still awaiting
completion of 2.4 eval sweep with PCGrad default). 2.7c runs
*after* 2.4b produces baseline numbers.
**Notes / risks.**
- If Nash-MTL strictly dominates, we may want to rebrand AlignMO's
  project-branch default. This is a "deferred" design decision — we
  won't decide until the ablation data is in.
- Nash-MTL's scale-invariance might make it a natural fit for the
  `anchored_projected` cell even better than PCGrad+normalize.
  Worth watching in the ablation.

### 2026-04-23 — Framing refinement: 3-mode → 4-mode / 2-binary-decision
**Done.** User review of the original 3-mode contract found that the
exclusive routing `pcgrad > anchored > scalarized` misbehaves when
BOTH scale-imbalance and directional-conflict fire: it applies PCGrad
on raw (unnormalized) gradients, reproducing the exact knapsack SPSA
failure mode the paper documents. Reframed as two independent binary
decisions (normalize? project?) whose cross product is 4 modes
(`scalarized`, `projected`, `anchored`, `anchored_projected`).
Updated Section 1 (Context), Section 2 (Current Status), Section 5.2
(Handler contract), Section 5.3 (inserted new blocking TODOs 2.1d,
2.3c, 2.7b, 2.9), Section 9 (open questions).
**Next.** Phase 2 TODO 2.1d — refactor `AlignMOHandler.compute_direction`
to the 2-binary-decision composition. DO NOT run the eval sweep until
2.1d + 2.3c are green.
**Notes / risks.**
- `τ_align` is removed (single threshold `τ_conflict` now suffices).
- 2.3c must include a unit test for the `anchored_projected` case,
  which was previously mis-routed.
- New sibling artifacts now referenced from the plan:
  `LAMBDA_ABLATION_NOTE.md` (λ sweep vs auto; TODO 2.7b) and
  `THEOREM_PROMPT.md` (subsequential Pareto-stationarity with
  M-estimation bridge lemma; TODO 2.9).
- **Open question for advisor:** whether PCGrad is the right projection
  primitive or whether Nash-MTL / FAMO / CAGrad should be used in the
  project-branch. Raised as open question in Section 9.
**Artifacts.**
- `experiments/advisor_review/ALIGNMO_PLAN.md` (this file, extensive edits)
- `experiments/advisor_review/LAMBDA_ABLATION_NOTE.md` (new)
- `experiments/advisor_review/THEOREM_PROMPT.md` (new)

### 2026-04-23 11:06 — TODO 2.1–2.3 done: AlignMO wired + 115 tests green
**Done.** `AlignMOHandler` (already present in
`src/fair_dfl/algorithms/mo_handler.py:255`) now registered as
`mo_method="alignmo"` in both dispatchers
(`src/fair_dfl/algorithms/core_methods.py:539`,
`src/fair_dfl/training/loop.py:112`). Both sites call
`mo_handler.set_step_context(mu=alpha_t, lam=beta_t)` before
`compute_direction` (duck-typed, no impact on other handlers).
`mo_alignmo_*` kwargs (tau_align/tau_conflict/tau_scale/mu_floor/
beta_ema/T_warmup) threaded from `train_cfg`. Added `"AlignMO"` to
`ALL_METHOD_CONFIGS` with the Section 5.2 defaults plus color/marker
styling. Added `TestAlignMOHandler` to `tests/test_mo_handlers.py` with
7 cases (warmup→scalarized, aligned→scalarized, conflict→pcgrad,
scale-imbalance→anchored with mu_floor, NaN→finite, EMA persistence +
mode-switch counting, contract-log-keys). `pytest tests/ -q`: **115
passed** (was 108). End-to-end sanity: invoked `AlignMO` via the pilot
launcher on (analytic, α=2.0, seed=11, n_sample=500, 20 steps) — stage
row written, per-iter `mode_this_step`, `c_dp`, `r_dp`, `n_projections`
logged; routed to `scalarized` for all four logged iters on that cell
(consistent with the pilot's finding that analytic α=2.0 has aligned
gradients with moderate scale ratio).
**Next.** TODO 2.4: write `run_alignmo_eval.py` (pilot grid with
AlignMO added) and execute into `results/alignmo_eval/`.
**Notes / risks.** The existing `AlignMOHandler` contract matches
Section 5.2 exactly, so no implementation changes were needed — only
wiring. `_build_active_moo_payload` in `loop.py` may drop
pred_loss/pred_fairness when `iter_spec.use_pred`/`use_fair` are False;
AlignMO's `compute_direction` handles missing keys by substituting
zeros, so that path is still safe.
**Artifacts.**
- `src/fair_dfl/training/loop.py` (registration + set_step_context)
- `src/fair_dfl/algorithms/core_methods.py` (registration + set_step_context)
- `experiments/configs.py` (AlignMO method entry + styling)
- `tests/test_mo_handlers.py` (TestAlignMOHandler)
- `results/pilot_alignmo/smoke_alignmo/analytic/alpha_2.0/seed_11/` (smoke artefacts)


### 2026-04-23 10:52 — CUDA env confirmed; backend probe widened to all pilot alphas
**Done.** Verified the desktop thread's default Python is CPU-only, but
`conda run -n main python` has CUDA access on the RTX 3080. Also
verified the healthcare task officially supports `analytic` and
`finite_diff`; updated the planned backend sanity probe from a single
assumed `alpha=2.0` cell to the full pilot alpha grid
`{0.5,1.5,2.0,3.0}` at seed `1`.
**Next.** Write and launch the backend-sanity runner under
`experiments/advisor_review/` for seed `1`, alphas
`{0.5,1.5,2.0,3.0}`, backends `{analytic, finite_diff, spsa}`,
`lambdas={0,1}`, and full-cohort healthcare via `conda run -n main`.
**Notes / risks.** This probe is diagnostic and outside the ordered
Phase 2 checklist; do not confuse it with the AlignMO build/eval path.

### 2026-04-23 11:02 — Backend-sanity launcher written with finite-diff guardrail
**Done.** Wrote
`experiments/advisor_review/run_healthcare_backend_sanity.py` to run the
requested one-seed healthcare backend comparison using the advisor-review
runner harness. The launcher targets the full pilot alpha grid,
`lambdas={0,1}`, full cohort by default, and `device="cuda"`, and it
explicitly blocks full-cohort `finite_diff` unless
`--allow-expensive-fd` is passed because the current healthcare FD path
would require O(`n_train`) decision solves per step.
**Next.** Launch the full-cohort `analytic` + `spsa` runs from
`conda run -n main`, then decide whether to force expensive
full-cohort `finite_diff` or switch that control to a smaller cohort.
**Artifacts.**
- `experiments/advisor_review/run_healthcare_backend_sanity.py`

### 2026-04-23 10:44 — Analytic backend sanity done; SPSA left running; finite-diff deferred
**Done.** Ran the full-cohort healthcare backend-sanity launcher in the
CUDA-capable `main` env for `analytic`; all four alphas completed for
seed `1` and wrote outputs under
`results/pilot_alignmo/backend_sanity/analytic/`. Started the matching
full-cohort `spsa` run; after an interrupted foreground turn, confirmed
the worker process is still alive and accruing CPU in the background.
User explicitly chose not to run full-cohort `finite_diff`; only revisit
finite-diff later as a very small-sample control if still needed.
**Next.** Monitor the background `spsa` backend-sanity run and notify
the user when it finishes.
**Artifacts.**
- `results/pilot_alignmo/backend_sanity/analytic/alpha_0.5/seed_1/stage_results.csv`
- `results/pilot_alignmo/backend_sanity/analytic/alpha_1.5/seed_1/stage_results.csv`
- `results/pilot_alignmo/backend_sanity/analytic/alpha_2.0/seed_1/stage_results.csv`
- `results/pilot_alignmo/backend_sanity/analytic/alpha_3.0/seed_1/stage_results.csv`

### 2026-04-23 11:09 — Background SPSA sanity run terminated; analytic result summarized
**Done.** Stopped the background full-cohort healthcare `spsa`
backend-sanity processes at the user's request, leaving no active
backend-sanity Python/Conda worker. Summarized the completed
full-cohort `analytic` run for seed `1`: `PCGrad` is best at
`alpha=0.5` (`0.0488`) and `alpha=1.5` (`0.0123`), `FDFL-0.1` is
best at `alpha=2.0` (`0.1286`) with `FPLG` a near-tie, and `FPTO`
is best at `alpha=3.0` (`2.0690`) while several decision-focused
methods explode to enormous normalized regret values.
**Next.** Resume Phase 2, TODO 2.1a: implement `AlignMOHandler`.
**Notes / risks.** The analytic backend-sanity run strengthens the
adaptation story at low/mid `alpha`, but high-`alpha` healthcare remains
numerically fragile even with clean gradients.

### 2026-04-23 10:38 — Added ad hoc backend-sanity probe requested by user
**Done.** Recorded an out-of-checklist diagnostic in Section 9 to
investigate the surprising healthcare SPSA result from Phase 1. The
probe will compare `analytic`, `finite_diff`, and `spsa` on the full
healthcare cohort with `fairness_type="mad"`, assumed `alpha=2.0`,
`seed=1`, `steps=70`, `lambdas={0,1}` using the CUDA-capable `main`
Conda env if available.
**Next.** Launch the backend comparison and inspect whether
`finite_diff` behaves closer to `analytic` or to `spsa`.
**Notes / risks.** This is outside the Phase 2 checklist; keep it
diagnostic and do not let it sprawl into new method code.

### 2026-04-23 10:15 — TODOs 1.4-1.9 done: pilot aggregated, verdict GREEN
**Done.** Ran `experiments/advisor_review/aggregate_alignmo_pilot.py`
successfully and generated all Phase 1 deliverables under
`results/pilot_alignmo/`: `grand_summary.csv` (`624` rows),
`per_cell_winners.csv` (`8` rows), `diagnostic_profile.csv` (`64`
rows), `fig_per_cell_winners.png`, and `GO_NO_GO.md`. The memo reports
`4` distinct cell winners (`fpto`, `pcgrad`, `fplg`, `fdfl-0.5`) across
the `8` pilot cells, which triggers a **GREEN / GO** verdict under the
Section 4.4 rule.
**Next.** Phase 2, TODO 2.1a: implement `AlignMOHandler` in
`src/fair_dfl/algorithms/mo_handler.py`, then thread its diagnostics to
`iter_logs.csv` per Section 5.2.
**Notes / risks.** Three cells have `gap_seed_std_ratio < 1`
(`analytic, alpha=2.0`; `analytic, alpha=3.0`; `spsa, alpha=3.0`), so
the pilot is a clean GO overall but not every local winner is sharp.
**Artifacts.**
- `experiments/advisor_review/aggregate_alignmo_pilot.py`
- `results/pilot_alignmo/grand_summary.csv`
- `results/pilot_alignmo/per_cell_winners.csv`
- `results/pilot_alignmo/diagnostic_profile.csv`
- `results/pilot_alignmo/fig_per_cell_winners.png`
- `results/pilot_alignmo/GO_NO_GO.md`

### 2026-04-23 10:05 — TODO 1.3 closed; aggregation work started
**Done.** Updated the plan after the full Phase 1 pilot finished
successfully. Verified the raw pilot is complete at
`results/pilot_alignmo/`: analytic `12/12`, SPSA `12/12`, `24` cells,
`624` stage rows, total elapsed `16024.0s`, with the final summary
written to `grid_summary.json` and `run_log.txt` at 09:48:53 EDT.
**Next.** TODO 1.4: build `grand_summary.csv` from the completed stage
outputs, using `aggregate_alignmo_pilot.py` and fixing it if needed.
**Notes / risks.** An untracked `aggregate_alignmo_pilot.py` is already
present in the worktree; treat it as candidate in-progress work rather
than overwrite it blindly.

### 2026-04-23 08:44 — TODO 1.3 progress check: 22/24 cells done
**Done.** Inspected the live Phase 1 pilot outputs while the launcher
was still running. Confirmed analytic is complete (12/12 cells) and
SPSA is at 10/12 completed cells; the only unfinished work appears to
be `alpha=3.0` for seeds 22 and 33. Verified the Python process is
still accruing CPU time, so the run has not exited.
**Next.** Let TODO 1.3 finish, then verify the final two SPSA cells and
start TODO 1.4 to build `grand_summary.csv`.
**Notes / risks.** `results/pilot_alignmo/run_log.txt` appears stale
because `run_alignmo_pilot.py` prints progress only after each
`(regime, alpha, seed)` cell and redirected stdout is buffered; lack of
new log lines does not by itself prove a hang.

### 2026-04-23 05:21 — TODO 1.1 + 1.2 done: pilot launcher + smoke test
**Done.** Wrote `experiments/advisor_review/run_alignmo_pilot.py`
parameterizing the Section 4.3 grid (4 alphas × 2 regimes × 8 methods
× 4 lambdas × 3 seeds = 768 stages at `n_sample=1000`,
`steps_per_lambda=50`, `fairness_type="mad"`, `budget_rho=0.30`,
`lr=1e-3`, hidden 64×2, SPSA `n_dirs=8, eps=1e-3`). Verified imports
and ran the one-cell smoke test (α=2.0, analytic, FDFL-Scal, seed=11,
λ=0.5): 1 stage row + 10 iter rows in 1.5 s; `iter_logs.csv` contains
per-step `cos_dec_pred/dec_fair/pred_fair` and `grad_norm_*`, and
`stage_results.csv` has the aggregated `avg_cos_*` + grad-norm
quantiles. Those are the diagnostics needed for Section 4.5's
`diagnostic_profile.csv`.
**Next.** TODO 1.3: run the full grid with
`python -m experiments.advisor_review.run_alignmo_pilot --regime both`,
piping to `results/pilot_alignmo/run_log.txt`.
**Notes / risks.** Budget estimate based on smoke: analytic ≈ 1.5 s
per (method, seed, λ) ⇒ analytic half ≈ 9–10 min for 384 stages;
SPSA ~3–5× slower; whole pilot likely well under the 4 h estimate in
Section 4.3. CVXPY ortools-version warning appears on import but does
not block execution.
**Artifacts.**
- `experiments/advisor_review/run_alignmo_pilot.py`
- `results/pilot_alignmo/smoke/analytic/alpha_2.0/seed_11/stage_results.csv`
- `results/pilot_alignmo/smoke/analytic/alpha_2.0/seed_11/iter_logs.csv`

### 2026-04-23 — Plan document created
**Done.** Consolidated the pilot + AlignMO build + paper-section plan
into this file (`experiments/advisor_review/ALIGNMO_PLAN.md`). Fixed the
closed-form sign bug for `_solve_group` at α>1, α≠2 (already staged in
working tree pre-session); added 20 new regression tests covering
α∈{0.3, 0.5, 0.8, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0}; 108 tests pass.
**Next.** Phase 1, TODO 1.1 — write the pilot launcher.
**Notes / risks.** Pilot is the go/no-go gate. Do not start Phase 2
before the GO_NO_GO memo is written and reviewed.
**Artifacts.**
- `experiments/advisor_review/ALIGNMO_PLAN.md` (this file)
- `tests/test_medical_gradients.py::TestSolveGroupOptimality` (already
  present pre-session; verified 20/20 parametrized tests pass)
- `src/fair_dfl/tasks/medical_resource_allocation.py` (closed-form sign
  fix already applied pre-session; currently uncommitted in working
  tree)
