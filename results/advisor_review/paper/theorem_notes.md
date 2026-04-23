# AlignMO Theorem Notes

## Current polished appendix target

The appendix now has two deliberately separated layers:

1. **Main proposition.**
   A conditional convergent-run result: if an oracle/population-fairness AlignMO run converges, then its limit is Pareto-stationary.

2. **Stronger scalarized-commitment corollary.**
   If the mode eventually commits to `scalarized`, then local strong convexity of
   \[
   V_{\mathrm{scal}} = L_{\mathrm{dec}}+\mu L_{\mathrm{pred}}+\lambda L_{\mathrm{fair}}
   \]
   gives uniqueness of the scalarized branch's local limit, and exact fixed-step gradient descent gives a local linear rate.

This is the split we want: the main theorem stays close to the algorithm and venue expectations, while the stronger local result is clearly labeled as conditional.

## Highest-priority issue from review

The biggest current gap is the **time-scale mismatch in the fairness bridge**.

- The bridge lemma is an \(n\to\infty\) statement about empirical-to-population approximation.
- The main theorem had been using an along-training condition like \(\xi_t\to 0\), which is a \(t\to\infty\) statement.

For a fixed dataset, the first does not imply the second.

So the main theorem should currently be interpreted as an **oracle/population-fairness theorem**.
The empirical group-MAD bridge is still useful context, but it does not yet close the loop needed for the main proposition on the actual fixed-sample algorithm.

## Fairness bridge status

### What we currently have

The current bridge lemma is a **local asymptotic approximation statement**:

- on a margin-separated compact set away from fairness kinks,
- the empirical group-MAD objective and gradient approach their population counterparts as \(n\to\infty\).

This is useful for:
- explaining why MAD fairness is theoretically harder than decomposable penalties;
- motivating a population-level theorem;
- identifying what kind of fixed-sample perturbation statement might be possible later.

### What we do not currently have

We do **not** currently have a theorem-facing statement of the form:

\[
\text{along the actual fixed-sample training trajectory, } \|\xi_t\| \to 0
\quad\text{or}\quad
\|\xi_t\| \le \varepsilon_n.
\]

That missing step is exactly why the main theorem cannot honestly be presented yet as a theorem for the actual empirical group-MAD algorithm.

### Current safe conclusion

The safe conclusion is:

> keep the main theorem at the oracle/population-fairness level, and treat the local bridge lemma as context plus future-work motivation.

### What would be needed to go beyond that

To recover a fixed-sample theorem-facing statement, we would need one of:

1. **Uniform fixed-sample control on the relevant region.**
   Prove that on a compact neighborhood containing the training trajectory,
   \[
   \sup_{\theta\in\mathcal K}\|\nabla F_n(\theta)-\nabla F(\theta)\| \le \varepsilon_n
   \]
   with high probability, and then feed that into the theorem as a perturbation bound.

2. **Trajectory-restricted control.**
   Prove the weaker but sufficient statement that
   \[
   \|\nabla F_n(\theta_t)-\nabla F(\theta_t)\| \le \varepsilon_n
   \]
   only along the iterates actually visited by training.

3. **A nonsmooth population-subgradient route.**
   Replace the smooth local bridge by a Clarke-subgradient or nonsmooth M-estimation argument that works through kink crossings.

Option (1) is probably the cleanest theorem-facing route.
Option (2) may be easier but is algorithm/path dependent.
Option (3) is the most general, but it is also the furthest from a short IJOC appendix.

### Immediate implication for our proof order

Until one of the three items above is proved, the correct order is:

1. state the main theorem as oracle/population-fairness;
2. work on the subsequential-upgrade logic at that population level;
3. revisit fixed-sample empirical fairness only after the switched-dynamics proof route is clearer.

## Fixed-sample perturbation control: formal subproblem

The empirical-fairness theorem-facing subproblem can now be stated precisely.

### Desired theorem-facing statement

We would like a statement of the following form.

Let
\[
\xi_t := \nabla F_n(\theta_t) - \nabla F(\theta_t),
\]
where \(F_n\) is the empirical group-MAD fairness term and \(F\) is the target population fairness term.
Then on a high-probability event and on the region actually visited by training, prove either
\[
\sup_{t\ge 0}\|\xi_t\| \le \varepsilon_n
\qquad\text{with } \varepsilon_n\to 0,
\]
or at least
\[
\sup_{\theta\in\mathcal K}\|\nabla F_n(\theta)-\nabla F(\theta)\| \le \varepsilon_n
\]
for a compact \(\mathcal K\) containing the training trajectory.

This is enough to replace the old informal \(\xi_t\to 0\) story by a fixed-sample small-perturbation theorem.

### Most plausible route

The most plausible route is **uniform fixed-sample control on a margin-separated compact neighborhood**.

Concretely, pick a compact \(\mathcal K\) such that:
- the training trajectory remains in \(\mathcal K\);
- \(\mathcal K\) stays away from fairness kinks:
  \[
  \inf_{\theta\in\mathcal K}\min_k |m_k(\theta)-\mu(\theta)| \ge \gamma > 0.
  \]

Then try to prove, with high probability,
\[
\sup_{\theta\in\mathcal K}\|\nabla F_n(\theta)-\nabla F(\theta)\|
\le
2\Delta_n
\]
provided the sign-stability condition \(2\delta_n<\gamma\) holds, where
\[
\delta_n := \max_k \sup_{\theta\in\mathcal K}|m_{k,n}(\theta)-m_k(\theta)|,
\qquad
\Delta_n := \max_k \sup_{\theta\in\mathcal K}\|\nabla m_{k,n}(\theta)-\nabla m_k(\theta)\|.
\]

If this inequality lands, the theorem-facing perturbation can be taken as
\[
\varepsilon_n = 2\Delta_n
\]
on the event \(2\delta_n<\gamma\). This tends to zero whenever the groupwise losses and gradients converge uniformly and the value deviation is small enough to preserve the sign pattern.

### Why this route is attractive

This route is attractive because:
- it matches the current local bridge lemma;
- it stays within smooth calculus on a fixed sign region;
- it plugs directly into the theorem as a deterministic small perturbation on a high-probability event;
- it avoids having to prove anything path-adaptive about the iterates themselves.

### What assumptions it needs

To make that route work, we likely need:

1. **Compact trajectory containment.**
   The iterates remain in a deterministic compact \(\mathcal K\).

2. **Margin from fairness kinks.**
   The whole relevant region satisfies a strict sign margin:
   \[
   \inf_{\theta\in\mathcal K}\min_k |m_k(\theta)-\mu(\theta)| \ge \gamma > 0.
   \]

3. **Uniform law of large numbers for values and gradients.**
   For each group \(k\),
   \[
   \sup_{\theta\in\mathcal K}|m_{k,n}(\theta)-m_k(\theta)| \to 0,
   \qquad
   \sup_{\theta\in\mathcal K}\|\nabla m_{k,n}(\theta)-\nabla m_k(\theta)\| \to 0.
   \]

4. **Differentiability of the prediction model and bounded derivatives on \(\mathcal K\).**
   This is what makes the gradient ULLN plausible.

### Likely first proof step

The likely first proof step was not a probability theorem. It was an algebraic lemma:

> On a sign-stable compact set, \(\nabla F_n-\nabla F\) can be decomposed into a finite linear combination of
> \((\nabla m_{k,n}-\nabla m_k)\), \((m_{k,n}-m_k)\), and \((\nabla \mu_n-\nabla \mu)\).

Once that is written down explicitly, the probability part reduces to whichever uniform convergence theorem we are willing to assume.

### Algebraic lemma we should now try to use

Let
\[
\delta_n := \max_k \sup_{\theta\in\mathcal K}|m_{k,n}(\theta)-m_k(\theta)|,
\qquad
\Delta_n := \max_k \sup_{\theta\in\mathcal K}\|\nabla m_{k,n}(\theta)-\nabla m_k(\theta)\|.
\]
Assume the population margin condition
\[
\gamma := \inf_{\theta\in\mathcal K}\min_k |m_k(\theta)-\mu(\theta)| > 0.
\]

Then two deterministic facts hold.

1. **Sign stability from value control.**
   Since
   \[
   |\mu_n(\theta)-\mu(\theta)|
   \le
   \frac1K\sum_{k=1}^K |m_{k,n}(\theta)-m_k(\theta)|
   \le \delta_n,
   \]
   we have
   \[
   |(m_{k,n}(\theta)-\mu_n(\theta))-(m_k(\theta)-\mu(\theta))|
   \le 2\delta_n.
   \]
   Therefore if \(2\delta_n < \gamma\), then the empirical and population sign patterns agree on all of \(\mathcal K\):
   \[
   \operatorname{sign}(m_{k,n}(\theta)-\mu_n(\theta))
   =
   \operatorname{sign}(m_k(\theta)-\mu(\theta)).
   \]

2. **Gradient control once signs are frozen.**
   On that event, writing the common sign as \(s_k(\theta)\in\{-1,+1\}\),
   \[
   \nabla F_n(\theta)
   =
   \frac1K\sum_{k=1}^K s_k(\theta)\bigl(\nabla m_{k,n}(\theta)-\nabla \mu_n(\theta)\bigr),
   \]
   \[
   \nabla F(\theta)
   =
   \frac1K\sum_{k=1}^K s_k(\theta)\bigl(\nabla m_k(\theta)-\nabla \mu(\theta)\bigr).
   \]
   Hence
   \[
   \nabla F_n(\theta)-\nabla F(\theta)
   =
   \frac1K\sum_{k=1}^K s_k(\theta)
   \Bigl((\nabla m_{k,n}(\theta)-\nabla m_k(\theta))
   -
   (\nabla \mu_n(\theta)-\nabla \mu(\theta))\Bigr).
   \]
   Since
   \[
   \|\nabla \mu_n(\theta)-\nabla \mu(\theta)\|
   \le
   \frac1K\sum_{k=1}^K \|\nabla m_{k,n}(\theta)-\nabla m_k(\theta)\|
   \le \Delta_n,
   \]
   it follows that
   \[
   \sup_{\theta\in\mathcal K}\|\nabla F_n(\theta)-\nabla F(\theta)\|
   \le 2\Delta_n.
   \]

This is stronger and cleaner than the earlier vague decomposition statement:
- \(\delta_n\) is used to keep the sign pattern fixed;
- once signs are fixed, the actual gradient error is controlled by \(\Delta_n\).

So the fixed-sample perturbation branch reduces to:

> can we justify, on a theorem-acceptable compact set \(\mathcal K\), both
> \(2\delta_n<\gamma\) and \(\Delta_n\to 0\) with high probability?

### Main failure mode

The most likely point of failure is the **margin assumption**.

If the actual training trajectory can approach or cross a fairness kink, then:
- the sign pattern changes;
- the smooth gradient representation of \(F_n\) and \(F\) changes face;
- the uniform gradient-control argument above breaks.

That is the clearest reason to keep this subproblem separate from the current main theorem.

### Current recommendation

Treat this subproblem as a **research branch**, not as a prerequisite for polishing the current appendix.

The next concrete step on this branch should be:

1. decide whether there is a theorem-acceptable compact region \(\mathcal K\) on which the population margin \(\gamma>0\) is realistic;
2. identify what assumptions on the prediction model class would justify \(\Delta_n\to 0\) uniformly on \(\mathcal K\);
3. only then decide whether the fixed-sample perturbation branch is strong enough to re-enter the paper theorem.

## Current priority: subsequential-upgrade roadmap

The next proof target is to upgrade the main result from

> "if the full run converges, its limit is Pareto-stationary"

to

> "every convergent subsequence has a Pareto-stationary limit."

That is the right next move because it strengthens the paper's main theoretical story without forcing the main theorem to carry rate or uniqueness claims.

### Target upgraded statement

For fixed user-specified \((\mu,\lambda)\), under bounded iterates, Robbins-Monro steps,
diagnostic continuity, and either oracle/population fairness or a separately established fixed-sample perturbation control, every accumulation point
\(\theta^\star\) of AlignMO that lies away from routing boundaries and fairness kinks is
Pareto-stationary for \((L_{\mathrm{dec}},L_{\mathrm{pred}},L_{\mathrm{fair}})\).

Equivalently: every convergent subsequence \(\theta_{t_k}\to\theta^\star\) with an off-boundary,
off-kink limit converges to a Pareto-stationary point.

### Primary reference class for the upgrade

The cleanest formal route is now:

1. **Benaim-Hofbauer-Sorin Theorem 3.6**:
   the limit set of a bounded perturbed solution to a differential inclusion is internally chain transitive.
   Source: [Benaim, Hofbauer, Sorin 2005/2006](https://perso.imj-prg.fr/sylvain-sorin/wp-content/uploads/sorin-pub/Stochastic%20Approximations%20and%20Differential%20Inclusions.pdf), Theorem 3.6.

2. **Benaim-Hofbauer-Sorin Proposition 3.27 / Corollary 3.28**:
   if a Lyapunov function exists for a candidate stationary set and its value range has empty interior,
   then every internally chain transitive set lies inside that stationary set.

This is the right reference class because our obstacle is no longer "does a limit exist?";
it is "what can the cluster set look like when the rule switches between modes?"

## What the current draft actually proves

The strongest statement that looks cleanly provable **right now** is:

> For fixed user-specified \((\mu,\lambda)\), if an oracle/population-fairness AlignMO run converges to a limit \(\theta^\star\), and \(\theta^\star\) is away from the mode-selection boundaries, then \(\theta^\star\) is Pareto-stationary for \((L_{\mathrm{dec}}, L_{\mathrm{pred}}, L_{\mathrm{fair}})\).

This is weaker than the original subsequential target in `experiments/advisor_review/THEOREM_PROMPT.md`, but it is materially closer to a proof we can defend in a short IJOC appendix.

## Why the prompt needed correction

Three prompt claims are too strong as written:

1. `scalarized` / `anchored` are **not** automatically common-descent directions for every objective.
   A positive weighted sum of gradients is a descent direction for the weighted scalarization, not necessarily for each objective individually.

2. `anchored` is **not literally SGD on a scalar objective**.
   The field
   \[
   \widehat g_{\mathrm{dec}} + \mu_{\mathrm{eff}}\widehat g_{\mathrm{pred}} + \lambda \widehat g_{\mathrm{fair}}
   \]
   is generally not the gradient of any globally defined scalar potential.

3. PCGrad's paper does **not** give the exact general Pareto-stationarity result the prompt informally attributes to it.
   The safe way to handle the projected branch is to assume a projection primitive \(P\) whose zero set is Pareto-stationary, then instantiate that assumption for whichever primitive the paper finally adopts.

4. The proof uses continuity of the committed mode field at the limit point.
   This is automatic for the raw scalarized branch, but not automatic for normalized branches because
   \(g/\|g\|\) is generally discontinuous at \(g=0\). So the theorem must either:
   - assume continuity of any mode field active near the limit point; or
   - impose a nondegeneracy condition ensuring normalized gradients are either zero or bounded away from zero near that limit.

These corrections are what drive the current proof architecture.

## Current proof architecture

### Lemma 1: EMA stabilization at a convergent limit

Let \(\phi(\theta)\) be any bounded continuous diagnostic and
\[
e_t = \beta e_{t-1} + (1-\beta)\phi(\theta_t), \qquad 0<\beta<1.
\]
If \(\theta_t \to \theta^\star\), then \(e_t \to \phi(\theta^\star)\).

This gives eventual mode commitment whenever the diagnostic values at \(\theta^\star\) have nonzero margin from the thresholds \(\tau_{\mathrm{scale}}\) and \(\tau_{\mathrm{conflict}}\).

### Lemma 2: Zero of scalarized / anchored field implies Pareto-stationarity

If
\[
g_{\mathrm{dec}}(\theta^\star) + \mu g_{\mathrm{pred}}(\theta^\star) + \lambda g_{\mathrm{fair}}(\theta^\star)=0
\]
with \(\mu,\lambda>0\), then \(\theta^\star\) is Pareto-stationary after normalizing the coefficients to the simplex.

If
\[
\widehat g_{\mathrm{dec}}(\theta^\star) + \mu_{\mathrm{eff}}\widehat g_{\mathrm{pred}}(\theta^\star) + \lambda \widehat g_{\mathrm{fair}}(\theta^\star)=0,
\]
then
\[
\frac{1}{\|g_{\mathrm{dec}}(\theta^\star)\|}g_{\mathrm{dec}}(\theta^\star)
+ \frac{\mu_{\mathrm{eff}}}{\|g_{\mathrm{pred}}(\theta^\star)\|}g_{\mathrm{pred}}(\theta^\star)
+ \frac{\lambda}{\|g_{\mathrm{fair}}(\theta^\star)\|}g_{\mathrm{fair}}(\theta^\star)=0
\]
whenever the three gradients are nonzero, so again a positive convex combination of raw gradients vanishes. If one gradient is already zero, Pareto-stationarity is immediate.

### Lemma 3: Bridge for group-MAD fairness away from kinks

Write the empirical group MSEs as \(m_{k,n}(\theta)\), their population counterparts as \(m_k(\theta)\), and \(\mu_n(\theta)=K^{-1}\sum_k m_{k,n}(\theta)\), \(\mu(\theta)=K^{-1}\sum_k m_k(\theta)\).

On any compact set \(\mathcal K\) such that
\[
\inf_{\theta\in\mathcal K}\min_k |m_k(\theta)-\mu(\theta)| \ge \gamma > 0,
\]
uniform convergence of \(m_{k,n}\) and \(\nabla m_{k,n}\) to \(m_k\) and \(\nabla m_k\) implies
\[
\sup_{\theta\in\mathcal K}|F_n(\theta)-F(\theta)| = o_p(1), \qquad
\sup_{\theta\in\mathcal K}\|\nabla F_n(\theta)-\nabla F(\theta)\| = o_p(1).
\]

This is the cleanest version of the bridge lemma for the current appendix. It avoids full non-smooth M-estimation machinery by working on margin-separated compacta. If we later need the exact kink statement, we can add a citation-backed Clarke-subgradient remark.

### Proposition: convergent-tail Pareto stationarity

Once the mode commits, the update has the form
\[
\theta_{t+1} = \theta_t - \eta_t \{ d_{m^\star}(\theta_t) + \xi_t \},
\]
where \(d_{m^\star}\) is the committed mode's direction field and \(\xi_t\to 0\) is the fairness-bridge perturbation.

If \(\theta_t\to \theta^\star\) and \(d_{m^\star}(\theta^\star)\neq 0\), continuity gives a unit vector \(v\) and \(c>0\) such that
\[
\langle d_{m^\star}(\theta_t)+\xi_t, v\rangle \ge c
\]
for all large \(t\). Then
\[
\langle \theta_{t+1}-\theta_t, v\rangle \le -c\eta_t,
\]
and summing contradicts convergence because \(\sum_t \eta_t = \infty\).

Hence \(d_{m^\star}(\theta^\star)=0\), and Lemma 2 plus the projected-primitive assumption turn that into Pareto-stationarity.

## Remaining gap to the original target

The gap between the current draft and the original target is:

> upgrading from "if the full run converges, its limit is Pareto-stationary" to "every convergent subsequence converges to a Pareto-stationary point."

That upgrade is plausible, but it is not free. It needs one of:

- standard stochastic-approximation / asymptotic-pseudo-trajectory machinery for piecewise continuous vector fields; or
- a stronger recurrence argument showing that any cluster point away from mode boundaries attracts an actual tail, not just sparse visits.

This is the main open proof-design decision left after the current draft.

## New observation: EMA tracking may remove the memory obstacle

The EMA state is less problematic than it first looked. If a diagnostic is a bounded Lipschitz function
\(\phi(\theta_t)\) and the EMA obeys
\[
e_t=\beta e_{t-1}+(1-\beta)\phi(\theta_t), \qquad 0<\beta<1,
\]
then the tracking error
\[
\delta_t:=e_t-\phi(\theta_t)
\]
satisfies
\[
\delta_t
=
\beta\delta_{t-1}
+ \beta\bigl(\phi(\theta_{t-1})-\phi(\theta_t)\bigr).
\]
If \(\phi\) is Lipschitz and \(\|\theta_t-\theta_{t-1}\|=O(\eta_t)\) with \(\eta_t\to 0\), then
\[
\|\phi(\theta_{t-1})-\phi(\theta_t)\| = O(\eta_t),
\]
so the stable linear recursion above implies \(\delta_t\to 0\).

This matters because it means:

- the EMA diagnostics asymptotically agree with the instantaneous diagnostics computed from \(\theta_t\);
- along any convergent subsequence \(\theta_{t_k}\to\theta^\star\), the EMA state at the same times also converges to the diagnostic value at \(\theta^\star\);
- the switching rule can therefore be analyzed asymptotically using a set-valued mode map on \(\theta\) alone.

So the true subsequential obstacle is not "EMA memory" anymore. The real obstacle is proving that the cluster set of the switched recursion is forced into Pareto-stationary points rather than a larger internally chain transitive set.

## Revised route to the subsequential theorem

The cleanest upgrade path now looks like this.

### Step A: Replace EMA routing by an asymptotically equivalent instantaneous router

Define the instantaneous diagnostics
\[
\phi(\theta)=\bigl(c_{dp}(\theta),c_{df}(\theta),c_{pf}(\theta),r_{dp}(\theta),r_{df}(\theta)\bigr)
\]
and let \(\mathcal M(\theta)\) be the set of active modes determined by the threshold rule applied to \(\phi(\theta)\), with multiple active modes only on switching boundaries.

By the EMA tracking lemma above, the actual AlignMO update is an asymptotically vanishing perturbation of the switched recursion
\[
\theta_{t+1}=\theta_t-\eta_t d_t, \qquad d_t\in D(\theta_t),
\]
where
\[
D(\theta):=\{d_m(\theta): m\in \mathcal M(\theta)\}.
\]

What still needs to be checked carefully:

- upper semicontinuity of \(\mathcal M(\theta)\) away from routing boundaries;
- local boundedness of the direction fields \(d_m\);
- whether we need \(\operatorname{clco} D(\theta)\) for the formal differential-inclusion theorem even though AlignMO itself picks a single branch.

### Step B: Use differential-inclusion stochastic-approximation machinery

Let \(F(\theta)=-\operatorname{clco} D(\theta)\), or the smallest closed convex set-valued extension if needed for upper semicontinuity.
Then the recursion is in the Benaim-Hofbauer-Sorin differential-inclusion template:
- bounded iterates;
- vanishing step sizes with \(\sum_t \eta_t=\infty\);
- vanishing perturbation from the fairness bridge and EMA tracking.

Under the standard closed-graph / compact-convex-value assumptions on \(F\), the interpolated process should be a perturbed solution of
\[
\dot\theta \in F(\theta),
\]
and the limit set of \(\{\theta_t\}\) should be internally chain transitive for this differential inclusion.

This is the most promising way to recover a genuine subsequential statement. Reference:
- Benaim, Hofbauer, Sorin (2005), limit set of a bounded perturbed solution to a differential inclusion is internally chain transitive.

Concrete proof obligation:

- write the interpolated AlignMO path as a **bounded perturbed solution** of
  \[
  \dot\theta \in F(\theta),
  \]
  with perturbations coming from:
  1. EMA tracking error;
  2. empirical-to-population fairness bridge;
  3. any small discontinuity regularization near routing boundaries.

This is the step where we either successfully invoke BHS cleanly, or learn that the theorem statement must explicitly exclude a small boundary neighborhood.

### Step C: Rule out nonstationary internally chain transitive sets locally

This is now the real burden. A sufficient local condition near any off-boundary limit point \(\theta^\star\) is:

> In the single committed mode active near \(\theta^\star\), there exists a \(C^1\) Lyapunov function \(V_m\) such that \( \langle \nabla V_m(\theta), d_m(\theta)\rangle > 0 \) for every non-Pareto-stationary \(\theta\) in that neighborhood, and the Pareto-stationary set has no interval of \(V_m\)-values.

Then Benaim-Hofbauer-Sorin Proposition 3.27 / Corollary 3.28 style arguments imply that any internally chain transitive set in that neighborhood must lie inside the stationary set.

### What this means mode-by-mode

- `scalarized`: good. Use \(V_{\mathrm{scal}}=L_{\mathrm{dec}}+\mu L_{\mathrm{pred}}+\lambda L_{\mathrm{fair}}\).
- `anchored`: plausible if we strengthen the compatibility branch to guarantee
  \[
  \langle \nabla V_{\mathrm{anch}}(\theta), d_{\mathrm{anch}}(\theta)\rangle > 0
  \]
  away from stationary points, with \(V_{\mathrm{anch}}=L_{\mathrm{dec}}+\mu_{\mathrm{eff}}L_{\mathrm{pred}}+\lambda L_{\mathrm{fair}}\). This is automatic if the compatible region enforces nonnegative pairwise cosines; with the current \(\tau_{\mathrm{conflict}}=-0.1\), we need an explicit positivity assumption.
- `projected` / `anchored_projected`: this becomes primitive-specific. We need the chosen \(P\) to come with either:
  1. a strict Lyapunov function, or
  2. a theorem that any internally chain transitive set of the induced flow lies in the Pareto-stationary set.

This is the pressure point in the full subsequential theorem. It is also why a Nash-style primitive may end up easier to theorize than raw PCGrad.

### Current best-form upgraded theorem

Given the present state of the argument, the best honest upgraded theorem is probably:

> Every accumulation point of AlignMO that lies in a neighborhood where one active mode admits a \(C^1\) Lyapunov function satisfying the BHS Proposition 3.27 conditions is Pareto-stationary.

This is weaker than a fully global subsequential theorem, but materially stronger than the current convergent-run proposition, and it matches the proof tools we actually have.

### Minimal checklist to complete the subsequential upgrade

1. **EMA lemma, final form.**
   Turn the EMA tracking observation into a short formal lemma with proof.

2. **Set-valued map definition.**
   Define \(D(\theta)\) and \(F(\theta)=-\operatorname{clco}D(\theta)\) precisely.

3. **Perturbed-solution verification.**
   Show the interpolated AlignMO trajectory is a bounded perturbed solution in the BHS sense.

4. **ICT conclusion.**
   Cite BHS Theorem 3.6 to conclude the limit set is internally chain transitive.

5. **Local Lyapunov exclusion.**
   For each relevant mode region, verify the assumptions of BHS Proposition 3.27 / Corollary 3.28.

6. **Translate back to Pareto stationarity.**
   Show the zero/stationary set of that local mode field implies Pareto-stationarity of the original three-objective problem.

If Step 5 fails for some mode, that tells us exactly where the theorem needs to be qualified.

## Rate and uniqueness: best route is a strengthened local corollary

If we want to explore **rate of convergence** and/or **uniqueness of the limit**, the cleanest route is not to overload the main subsequential theorem. The clean route is:

1. keep the main theorem in the current "standard assumptions / asymptotic sanity-check" style;
2. add a separate **local committed-mode corollary** under stronger assumptions.

This separates the two jobs:
- the main theorem explains why adaptive routing does not break first-order asymptotics;
- the corollary explains what extra structure would buy us uniqueness or a rate.

### My preference

For the **main paper theorem**, I prefer:

> keep assumptions closer to the algorithm and accept the weaker theorem.

Reason:
- it is a better fit for IJOC;
- it avoids making the paper look as if the core contribution is a strong-convexity theorem;
- it keeps the assumptions interpretable from the implementation and diagnostics.

For the **optional extension**, I prefer:

> add a strengthened local corollary for uniqueness / rate, clearly labeled as conditional.

That gives us the best of both worlds.

## Candidate strengthened corollary template

The right abstract object is the committed mode field \(d_m\), not necessarily a scalar objective. Let \(m^\star\) be the asymptotic mode.

Assume:

1. **Eventual commitment.**
   There exists \(T\) such that for all \(t\ge T\), AlignMO stays in one mode \(m^\star\).

2. **Local exactness.**
   Near the candidate limit \(\theta^\star\), the bridge/estimation perturbation is small:
   \[
   \theta_{t+1}=\theta_t-\eta_t\bigl(d_{m^\star}(\theta_t)+\xi_t\bigr),
   \qquad \xi_t\to 0.
   \]

3. **Local Lipschitzness.**
   The field \(d_{m^\star}\) is \(L\)-Lipschitz on a neighborhood \(U\) of \(\theta^\star\).

4. **Strong monotonicity / local contraction.**
   There exists \(\alpha>0\) such that for all \(\theta,\vartheta\in U\),
   \[
   \langle d_{m^\star}(\theta)-d_{m^\star}(\vartheta),\ \theta-\vartheta\rangle
   \ge \alpha \|\theta-\vartheta\|^2.
   \]

This is the right strengthening because it covers:
- `scalarized` mode when the scalarized objective is locally strongly convex;
- `anchored` mode if the normalized field is locally strongly monotone;
- `projected` modes if the chosen primitive induces a locally strongly monotone effective field.

## What these assumptions buy us

### Uniqueness of the local limit

Under local strong monotonicity, \(d_{m^\star}\) has **at most one zero** in \(U\).

Proof sketch:
\[
0=\langle d_{m^\star}(\theta)-d_{m^\star}(\vartheta),\theta-\vartheta\rangle
\ge \alpha\|\theta-\vartheta\|^2
\]
for any two zeros \(\theta,\vartheta\), hence \(\theta=\vartheta\).

So if the run enters \(U\) and stays there, the limit in that basin is unique.

### Linear rate with constant step size

If \(\xi_t\equiv 0\) and we use a small constant step size \(\eta\), then
\[
\|\theta_{t+1}-\theta^\star\|^2
\le
\bigl(1-2\eta\alpha+\eta^2L^2\bigr)\|\theta_t-\theta^\star\|^2.
\]
Hence for
\[
0<\eta<\frac{2\alpha}{L^2},
\]
the iteration is a contraction, giving a **local linear rate**.

If \(\xi_t\neq 0\) but is summable or decays sufficiently fast, the same argument gives linear convergence up to a vanishing perturbation term.

### \(O(1/t)\)-style rates with Robbins-Monro steps

If we keep the classical diminishing schedule
\[
\eta_t = \frac{c}{t},
\]
then under local strong monotonicity and bounded perturbation moments, the natural target is an **\(O(1/t)\)** rate in squared distance or objective gap, not a linear rate.

This is more compatible with the current stochastic-approximation framing, but it requires more bookkeeping and the statement is less sharp than the constant-step local linear result.

## Best mode-specific statements

### Scalarized mode

This is the cleanest branch for uniqueness/rates.

If
\[
V_{\mathrm{scal}}(\theta)
=
L_{\mathrm{dec}}(\theta)+\mu L_{\mathrm{pred}}(\theta)+\lambda L_{\mathrm{fair}}(\theta)
\]
is locally strongly convex and smooth near \(\theta^\star\), then:
- \(\theta^\star\) is the unique local minimizer in that neighborhood;
- constant-step gradient descent gives local linear convergence;
- \(1/t\)-steps give the usual \(O(1/t)\)-type decay.

This is the strongest candidate for a polished corollary.

### Anchored mode

This branch can still support uniqueness/rates, but only through the field \(d_{\mathrm{anch}}\), not through a scalar potential in general.

So the right strengthened assumption is:
- local Lipschitzness and strong monotonicity of \(d_{\mathrm{anch}}\).

This is mathematically clean, but it is less interpretable than strong convexity of a scalar objective.

### Projected / anchored-projected modes

These are the hardest branches.

To get uniqueness or rates here, we would need one of:
- a projection primitive \(P\) whose induced field is locally strongly monotone; or
- a specific primitive (likely Nash-style rather than PCGrad) with an existing local stability result we can reuse.

This is the clearest reason to keep the main theorem projection-parametric while making any rate/uniqueness corollary either:
- scalarized-only, or
- conditional on a strong monotonicity property of the committed projected field.

## Recommended theorem hierarchy

If we decide to carry both a main theorem and a stronger extension, the clean hierarchy is:

1. **Main theorem.**
   Subsequential convergence / cluster-point Pareto-stationarity under standard assumptions.

2. **Corollary A (local uniqueness).**
   Under eventual commitment and local strong monotonicity of the committed mode field, the local limit is unique.

3. **Corollary B (local rate).**
   Under the same assumptions plus an explicit step-size regime:
   - constant step: local linear rate;
   - \(1/t\)-steps: \(O(1/t)\)-style rate.

This is much cleaner than trying to make the main theorem carry all three claims at once.

## Practical recommendation

The next theory pass should prioritize the subsequential upgrade in the order above:

1. formalize the EMA tracking lemma;
2. define the set-valued switched field and verify the BHS perturbed-solution conditions;
3. identify the exact Lyapunov assumption needed mode-by-mode, with scalarized first.

Only after that should we tighten the scalarized rate/uniqueness corollary further.
