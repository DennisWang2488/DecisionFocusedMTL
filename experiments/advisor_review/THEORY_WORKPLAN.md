# AlignMO Theory Workplan

## Purpose

This file tracks the **next research edits and proof steps** for the AlignMO theory appendix after the first review round.

It is intentionally short and operational: the goal is to decide what to edit now and what proof work to do next, in order.

## Current theorem stance

As of this revision:

- the main theorem is a **conditional convergent-run Pareto-stationarity sanity check**;
- it should be read at the **oracle/population-fairness** level;
- the scalarized uniqueness/rate result is a **secondary corollary**;
- the subsequential theorem remains a **planned upgrade**, not a finished theorem.

## Modification plan

### 1. Scope cleanup

Goal: keep all theory-facing documents on the same theorem statement.

Tasks:
- update `theory_decisions.md` to reflect the current theorem stance;
- update `THEOREM_PROMPT.md` so it no longer asks for a theorem stronger than the current draft supports;
- keep `appendix_theorem.tex` and `theorem_notes.md` aligned with those decisions.

Status: completed.

### 2. Fairness-bridge cleanup

Goal: resolve the mismatch between the \(n\to\infty\) bridge lemma and the \(t\to\infty\) perturbation language in the theorem.

Two acceptable outcomes:
- **Outcome A:** keep the main theorem at the oracle/population-fairness level;
- **Outcome B:** prove a fixed-sample control such as \(\|\xi_t\|\le \varepsilon_n\) on the relevant trajectory/neighborhood.

Current default: **Outcome A**, unless Outcome B becomes genuinely provable.

### 3. Subsequental-upgrade preparation

Goal: prepare the route from the convergent-run theorem to a local subsequential theorem.

Tasks:
- formalize the EMA tracking lemma;
- define the switched set-valued field \(D(\theta)\);
- define the differential inclusion \(F(\theta) = -\operatorname{clco} D(\theta)\);
- verify the bounded perturbed-solution conditions;
- identify the exact local Lyapunov condition needed mode-by-mode.

Expected result:
- either a local subsequential theorem under explicit Lyapunov assumptions;
- or a precise note saying which mode blocks the upgrade.

### 4. Bonus corollary refinement

Goal: only after the first three items are stable, tighten the scalarized-commitment corollary.

Possible next additions:
- Robbins-Monro \(O(1/t)\)-type local rate;
- slightly cleaner statement separating uniqueness from rate.

This is lower priority than the fairness cleanup and subsequential groundwork.

## Immediate next proof task

The single most important next proof task is:

> decide whether the paper theorem stays purely oracle/population-fairness, or whether a fixed-sample perturbation control can actually be proved from the group-MAD bridge analysis.

Until that is settled, further subsequential-upgrade work is useful only as notes, not as final appendix text.
