# Prompt: theorem cleanup and upgrade path for AlignMO

> **Status:** revised prompt document. Use this after reading
> `results/advisor_review/paper/appendix_theorem.tex` and
> `results/advisor_review/paper/theorem_notes.md`.
>
> **Purpose:** this is no longer a “just write the theorem” prompt.
> It is a focused execution prompt for stabilizing the theorem statement
> around what is actually proved, then preparing the next upgrade step.

---

## Context you are stepping into

You are working on a short theoretical appendix for an IJOC submission on
**AlignMO**, an adaptive multi-objective gradient handler for
decision-focused learning.

The repository already contains:
- a current theorem draft in `results/advisor_review/paper/appendix_theorem.tex`;
- working notes in `results/advisor_review/paper/theorem_notes.md`;
- a decision memo in `experiments/advisor_review/theory_decisions.md`.

Read those files first. They reflect the current state of the theory
better than older planning language elsewhere in the repo.

The key correction from recent review is:

> the local group-MAD bridge lemma is an \(n\to\infty\) approximation
> statement, while the main theorem had been using a \(t\to\infty\)
> along-training perturbation statement.

Those are different limits. So the current main theorem should be treated
as an **oracle/population-fairness theorem** unless a separate fixed-sample
perturbation result is proved.

---

## What the current draft already supports

The theorem draft currently supports the following main result:

> For fixed user-specified \((\mu,\lambda)\), if an oracle/population-fairness
> AlignMO run converges to a limit away from routing boundaries, then that
> limit is Pareto-stationary for
> \((L_{\mathrm{dec}},L_{\mathrm{pred}},L_{\mathrm{fair}})\).

There is also a stronger but clearly secondary corollary:

> if AlignMO eventually commits to `scalarized` and the scalarized objective
> is locally strongly convex and smooth, then the local scalarized limit is
> unique, and exact fixed-step gradient descent gives a local linear rate.

Do not silently upgrade those claims unless you genuinely close the proof gap.

---

## Immediate task

Your job is to help with **one of two tightly scoped theory tasks**:

### Task A: Fairness-bridge cleanup (highest priority)

Resolve the mismatch between:
- the local \(n\to\infty\) group-MAD bridge lemma, and
- the \(t\to\infty\) perturbation assumption used by the main theorem.

Concretely, decide and write clearly which of the following is supportable:

1. **Oracle theorem only.**
   Keep the main theorem entirely at the population/oracle-fairness level,
   and relegate the empirical group-MAD bridge to a remark or future-work note.

2. **Fixed-sample perturbation control.**
   Replace the old \(\xi_t\to 0\) story by a statement the bridge can actually support for a fixed dataset,
   for example a uniform bound of the form
   \[
   \|\xi_t\| \le \varepsilon_n
   \]
   on the training trajectory or on a compact neighborhood.

If you cannot prove (2) cleanly, do not force it. Strengthen option (1)
and make the scope explicit.

### Task B: Subsequental-upgrade prerequisites (only after Task A is clean)

Prepare the route from the current convergent-run theorem to:

> every convergent subsequence whose limit lies away from routing boundaries
> and fairness kinks has a Pareto-stationary limit.

The intended route is:
- EMA tracking lemma;
- switched set-valued field \(D(\theta)\);
- differential inclusion \(F(\theta) = -\operatorname{clco} D(\theta)\);
- Benaim-Hofbauer-Sorin Theorem 3.6 for internally chain transitive limit sets;
- BHS Proposition 3.27 / Corollary 3.28 to exclude nonstationary limit sets via a local Lyapunov function.

Your job here is to verify exactly which pieces can be written rigorously
with the current assumptions, and which pieces would require stronger
mode-specific assumptions.

---

## What NOT to do

- Do not pretend the bridge lemma already gives a fixed-sample training-time statement if it does not.
- Do not claim global subsequential convergence unless the BHS route is genuinely closed.
- Do not turn the projected branch into a theorem by burying all the content in an opaque assumption and then calling the result “standard.”
- Do not spend time on rates, uniqueness, or continuation in \(\lambda\) unless specifically asked. Those remain secondary.

---

## References to use

1. `results/advisor_review/paper/appendix_theorem.tex`
2. `results/advisor_review/paper/theorem_notes.md`
3. `results/advisor_review/paper/draft_v2.tex` Appendix A material on non-decomposability
4. Van der Vaart, *Asymptotic Statistics*, Chapter 5
5. Benaim, Hofbauer, Sorin, *Stochastic Approximations and Differential Inclusions*
6. Liu et al. 2021 (CAGrad) for overall proof style
7. Yu et al. 2020 (PCGrad) only as a primitive-local reference, not as the whole theorem architecture

---

## Deliverables

Depending on which task you work on, produce one or more of:

- edits to `results/advisor_review/paper/appendix_theorem.tex`;
- edits to `results/advisor_review/paper/theorem_notes.md`;
- a short note in `results/advisor_review/paper/theorem_notes.md` titled
  `Fairness bridge status` or `Subsequential upgrade status` explaining:
  - what now works,
  - what still does not,
  - what assumption would be needed next.

---

## Success criteria

- The main theorem statement matches what is actually justified.
- The role of the empirical group-MAD bridge is described honestly.
- The next proof step is clearer after the work than before.

If you conclude that the fairness bridge still does not justify any fixed-sample theorem-facing perturbation statement, say so explicitly and keep the main theorem at the oracle/population level.
