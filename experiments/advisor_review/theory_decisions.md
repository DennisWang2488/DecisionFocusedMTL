# AlignMO Theory Decisions

## 1. Current theorem target

The main theorem should currently target a **conditional convergent-run Pareto-stationarity sanity check for a fixed user-specified pair \((\mu,\lambda)\)**, stated at the population/oracle-fairness level.

In words:

> Under standard smoothness, bounded-iterate, and Robbins-Monro step-size assumptions, if an AlignMO run on the population objectives converges to a limit away from routing boundaries, then that limit is Pareto-stationary for \((L_{\mathrm{dec}}, L_{\mathrm{pred}}, L_{\mathrm{fair}})\).

This is weaker than the original subsequential target, but it is the strongest theorem that is currently defendable without papering over the fairness-bridge mismatch or the switched-dynamics difficulty.

## 2. Planned upgrade path

The intended next upgrade is still:

> every convergent subsequence whose limit lies away from routing boundaries and fairness kinks has a Pareto-stationary limit.

But this should now be treated as a **Phase B theory task**, not as the immediate theorem we claim to have finished. The prerequisite is cleaning up the fairness statement so the theorem and the algorithm live on the same time scale.

## 3. Reference class anchor

Use **CAGrad + differential-inclusion / stochastic-approximation machinery** as the proof architecture anchor, not MGDA-style fixed-combiner theory and not a PCGrad-specific theorem.

Concretely:
- CAGrad is the right intuition for a short “sanity-check convergence” result.
- Benaim-Hofbauer-Sorin is the right reference class for the subsequential-upgrade route once we move from a single committed mode to switched dynamics.
- PCGrad remains only a branch-local primitive, not the architecture for the whole proof.

## 4. Bridge lemma commitment

**Yes, include the bridge lemma, but downgrade its role.**

The bridge lemma should currently be presented as:
- a local \(n\to\infty\) result for group-MAD fairness on margin-separated compacta;
- useful for explaining the non-decomposability issue and for motivating future fixed-sample control;
- **not** yet the ingredient that closes the main theorem for the fixed-sample training algorithm.

Immediate decision:
- the main theorem is oracle/population-fairness unless and until we prove a fixed-sample perturbation statement such as \(\|\xi_t\|\le \varepsilon_n\) along the training trajectory.

## 5. Projection primitive handling

Keep the theorem **parametric in a projection/conflict-resolution primitive \(P\)**.

The current projected branches should be handled via assumptions like:
- \(P\) is continuous or outer-semicontinuous on nondegenerate inputs;
- zeros of the projected field imply Pareto-stationarity;
- any stronger Lyapunov property needed for the subsequential upgrade is primitive-specific.

Do not overclaim more than this until the project-branch primitive is finalized.

## 6. Scope cap

Keep the appendix capped at **2 pages** for the main theorem plus the short scalarized corollary.

If the subsequential upgrade cannot be expressed cleanly within that budget, it should stay in working notes rather than force scope creep in the paper.

## 7. Explicit non-goals for the main theorem

The main theorem should explicitly not claim:
- global subsequential convergence unless the BHS route is actually closed;
- finite-sample empirical-fairness convergence;
- uniqueness of the limit;
- convergence rates;
- distance to the Pareto front;
- continuation in \(\lambda\);
- SPSA bias/variance theory.

## 8. Stronger bonus result to keep

Keep a short **scalarized-commitment corollary** as a bonus:
- if AlignMO eventually commits to `scalarized`,
- and the scalarized objective is locally strongly convex and smooth,
- then the local scalarized limit is unique;
- with exact fixed-step gradient descent, local linear convergence follows.

This should remain clearly secondary to the main theorem.

## 9. Immediate proof priorities

The next theory tasks should be done in this order:

1. **Fairness cleanup.**
   Decide whether the main theorem stays oracle/population-fairness, or whether we can prove a fixed-sample perturbation control that actually justifies the algorithm-facing statement.

2. **EMA lemma, final form.**
   Turn the EMA stabilization argument into a short formal lemma.

3. **Subsequential route prerequisites.**
   Define the switched set-valued field, verify the perturbed-solution conditions, and identify the exact local Lyapunov assumption needed mode-by-mode.

4. **Only then** tighten the scalarized uniqueness/rate corollary further.

## 10. Open sub-questions for the paper team

- Do we want the paper theorem to stay explicitly oracle/population-fairness, with empirical fairness discussed only in a remark?
- Is the theorem meant to cover only group-MAD fairness, or should we leave room for the individual-MAD analogue later?
- How much theorem qualification is acceptable for the projected branch if the final primitive remains unresolved?
- If the subsequential upgrade only works locally under a Lyapunov condition, is that still worth carrying in the paper, or should it stay in notes?

## Bottom line

The right stance now is:
- **main theorem:** conditional convergent-run Pareto-stationarity at the oracle/population-fairness level;
- **next proof goal:** clean the fairness bridge / perturbation story before pushing harder on subsequential convergence;
- **bonus theorem:** keep the scalarized-commitment uniqueness/rate corollary, but do not let it drive the paper’s main theory framing.
