# Discussion prompt: theory of the AlignMO algorithm

> **Purpose.** This is a *thinking* / *debating* prompt, **not** an
> execution prompt. The goal is to surface the tensions, tradeoffs,
> and design decisions around the theoretical framing of AlignMO
> *before* anyone writes a theorem. The companion file
> `THEOREM_PROMPT.md` is the execution prompt and should be run
> AFTER this discussion yields concrete choices.

> **Format.** Hand this to a Claude session, a theory-leaning
> collaborator, or use it yourself as a structured brainstorm.
> Time horizon: 1–2 focused sessions (60–90 minutes each).

---

## 1. Read first

Before engaging with the discussion questions, read:

1. `experiments/advisor_review/ALIGNMO_PLAN.md` — Sections 1 (Context)
   and 3 (Goals), to understand what AlignMO does and what the paper
   is claiming.
2. `experiments/advisor_review/THEOREM_PROMPT.md` — the companion
   execution prompt, to see what theorem is currently "on the table."
3. `results/advisor_review/paper/draft_v2.tex` — Appendix A (lines
   1200–1373), the half-written non-decomposability argument.
4. For reference class: skim the introductions (not the full proofs)
   of Désidéri 2012 (MGDA), Yu et al 2020 (PCGrad) Proposition 1,
   Sener & Koltun 2018, and Navon et al 2022 (Nash-MTL).

---

## 2. What we're trying to decide in this discussion

We have a target theorem on the table: **subsequential convergence to
a Pareto-stationary point of $(L_{\text{dec}}, L_{\text{pred}}, L_{\text{fair}})$ under
standard assumptions, with an M-estimation bridge lemma for the
non-decomposable fairness penalty.** See THEOREM_PROMPT.md.

The question this discussion addresses is: **is that the right target,
and is it positioned correctly for IJOC?**

Specifically we need to decide:

1. Should the target theorem be stronger, weaker, or differently framed?
2. Which proof architecture is cleanest given our design choices
   (4-mode framework, PCGrad-vs-Nash-MTL choice, λ as user-specified)?
3. Is the bridge lemma essential, or is there a way to avoid the
   non-decomposable fairness issue?
4. Does the choice of projection primitive (2.7c ablation) affect
   what theorem we should prove?

These are design decisions, not proof steps. They need advisor or
senior-researcher input, not mechanical theorem writing.

---

## 3. The constraints that shape the decision

### Venue constraint

IJOC is a computing journal. Theorems are accepted but not required;
reviewers are looking for algorithmic + empirical contributions
supported by *sanity-check* theory, not foundational results. The
reference bar for "acceptable IJOC theory" is:
- Désidéri 2012 (MGDA convergence): half a page, Lipschitz + bounded.
- Yu et al 2020 (PCGrad Prop 1): one page, under specific regime.
- Liu et al 2021 (CAGrad): 2 pages, standard assumptions.

Theory longer than ~2 pages risks shifting reviewer perception toward
"this is a theory paper" and prompts questions ("what's the rate?",
"tight constants?", "comparison with other MOO?") that we do not want
to answer.

### Substantive constraint

AlignMO's contribution is the **routing mechanism**, not a new
combiner. We cannot truthfully claim novelty in the convergence of
any individual mode (each mode is SGD on some objective, or reuses
a known combiner's guarantee). The theorem can only be about *the
composition* — that switching between modes does not break
convergence.

This constrains what a theorem can say. It cannot compete with MGDA's
cleanness because our problem has 4 modes and a mode-switch dimension
MGDA doesn't have.

### Data constraint

The fairness penalty $F$ is MAD-based and **non-decomposable**
(shared empirical center $\overline{MSE}(\theta)$). Any SGD / M-estimation
argument on $F$ requires either (a) bridging to a decomposable
surrogate, or (b) a genuinely non-decomposable concentration bound.

Option (a) is our current plan (bridge lemma). Option (b) would be a
contribution in itself but is MS-OR-scope, not IJOC-scope.

---

## 4. Discussion topics (the meat of the session)

Work through these in order. Spend 5–10 minutes on each. Write down
conclusions as you go — the deliverable of the session is Section 6
below.

### 4.1 Is subsequential Pareto-stationarity the right target?

- **Weaker target (safer):** "Under the additional assumption that
  the mode eventually commits, AlignMO reduces to SGD on a fixed
  scalarized objective and inherits convergence to a stationary
  point of that objective." This is nearly trivial and lets us
  skip the mode-commit argument entirely.

- **Current target (Pareto-stationary subsequence):** What
  THEOREM_PROMPT.md targets. Requires a mode-commit argument and a
  bridge lemma. ~1.5 pages.

- **Stronger target (rate / distance-to-front):** Would require
  Polyak–Łojasiewicz-type assumptions and substantial technical
  work. 3+ pages. Risk: reviewers will ask "why this rate, not a
  better one?"

Which is the right target? Argue for each. Consider: what does the
advisor actually want to see in a meeting? What does an IJOC
reviewer actually need to sign off?

### 4.2 Can we avoid the bridge lemma?

Options:
1. **Bridge to decomposable surrogate** (current plan). M-estimation
   machinery; ~0.5 page of the theorem.
2. **Switch to a decomposable fairness measure** (e.g., expected
   pairwise squared difference between groups, which *is*
   decomposable). But this changes the paper's main contribution —
   MAD is currently the reported fairness.
3. **Treat $F$ as a black-box Lipschitz function of $\theta$** and
   avoid touching its internal structure. Cleaner but gives less
   insight.

Which of these does the advisor prefer? Does option 3 buy us enough
to skip the bridge? (Probably not, because SGD convergence on a
non-decomposable criterion is itself a research question.)

### 4.3 Does the 4-mode framework help or hurt the theorem?

The 2-binary-decision framing gives us exactly 4 modes with clean
closed forms for the direction. This should make "each mode is SGD
on some effective objective" *easier* to state. Specifically:
- `scalarized`: SGD on $L_{\text{dec}} + \mu L_{\text{pred}} + \lambda L_{\text{fair}}$.
- `projected`: SGD on the same, but with PCGrad's projected direction,
  so reduce-to-Yu-et-al's-Prop-1.
- `anchored`: SGD on $L_{\text{dec}} + \mu_{\text{eff}} L_{\text{pred}} + \lambda L_{\text{fair}}$ with
  normalized gradients (scaled by mean_norm).
- `anchored_projected`: combine projected + normalized.

Does the independence of the two binary decisions make the proof
simpler? I think yes, because we can analyze the two "stabilization
transformations" (normalize, project) independently.

Debate: is there a cleaner way to state "these 4 modes all do SGD on
some effective objective"?

### 4.4 How does the projection-primitive choice (2.7c) affect the theorem?

If we end up using Nash-MTL instead of PCGrad in the `projected`
branch:
- Nash-MTL has cleaner Pareto-stationarity (stronger than PCGrad's
  Prop 1).
- The theorem gets *easier* because we don't have to invoke PCGrad's
  narrow sufficient condition.
- The bridge lemma is unchanged.

If we stay with PCGrad:
- We need to invoke Yu et al 2020 Prop 1 under its sufficient condition
  (all pairwise cosines non-positive).
- AlignMO only routes to the `projected` mode when this condition
  holds (by construction of the mode selector). This is actually a
  selling point.

Decision: should we write the theorem assuming Nash-MTL as the
primitive? That way the theorem matches our likely final design choice.
Or should we write it PCGrad-specific and update later?

Recommendation to discuss: write the theorem *parametrically* in the
projection primitive. Define "a projection primitive $P$ satisfies
(P1), (P2), …" and show Pareto-stationarity for any $P$ satisfying
those. PCGrad and Nash-MTL both satisfy; MGDA also satisfies. This is
elegant and doesn't commit to 2.7c's outcome.

### 4.5 What about the EMA diagnostics themselves?

The mode-commit argument relies on EMA diagnostics (cosines, log-
ratios) converging to their values at the limit point $\theta^*$. But
EMAs have their own dynamics. Is there a subtle issue?

- If gradients are continuous in $\theta$ (follows from Lipschitz
  gradient assumption A1), then $(\cos, \log\text{ratio})$ are
  continuous functions of $\theta$ away from $\|g\| = 0$ critical
  points.
- As $\theta_t \to \theta^*$, the diagnostic values converge (as
  deterministic functions of $\theta$), and the EMA integrates these
  with geometric weight $\beta^t$.
- If $\beta < 1$ fixed, EMA doesn't fully converge to
  "diagnostic at $\theta^*$" but to an exponentially weighted average
  around a neighborhood. This is sufficient for mode-commit if the
  mode boundary is not right at $\theta^*$.

Debate: do we need $\beta \to 1$ (decaying EMA) for a clean statement,
or is fixed $\beta$ fine? The current design uses fixed $\beta = 0.9$.

### 4.6 The λ-sweep trajectory

Currently the theorem is for a single $(\mu, \lambda)$. In practice
we do a $\lambda$-sweep with warm-starts (FPLG-style). Does the
theorem say anything about the sweep?

The natural extension: if each stage's $\theta$ at the end of the
previous $\lambda$ is used as the warm-start for the next, then the
whole trajectory traces a "quasi-continuation" of Pareto-stationary
points for different $\lambda$. A theorem about this would be
**continuation** in λ: "as λ varies along the sweep, AlignMO's
trajectory stays within $O(\Delta\lambda)$ of the Pareto-stationary
curve."

This is probably out of scope (continuation theorems are a separate
research area) but worth noting as "future work" and explicitly
NOT claiming.

### 4.7 Negative results worth proving?

Sometimes a "negative" theorem strengthens a paper:
- "PCGrad alone does not converge to Pareto-stationarity in our
  setting [specific counterexample]."
- "MGDA alone fails on SPSA gradients because … [specific
  characterization]."

Debate: is any of these worth ~0.5 page? It would motivate AlignMO
directly. But it also invites "then prove MGDA works in the
non-SPSA case," which expands scope.

Probably skip. Mention as empirical observation, not theorem.

### 4.8 What about the decision gradient itself?

The decision gradient $\nabla L_{\text{dec}}$ comes from either:
- Analytic VJP (closed form we derived, via the post-fix `_solve_group`).
- SPSA (biased + variance estimator).

The theorem implicitly treats $\nabla L_{\text{dec}}$ as if it's the true
gradient. Under SPSA this is false — SPSA has both bias (O(ε²)) and
variance. Does this affect the Pareto-stationarity argument?

Possible answers:
1. **Treat SPSA as a known-bias-known-variance gradient oracle.** Add
   an assumption: "the decision gradient estimator has bias $b$ and
   variance $V$." The theorem's conclusion becomes: subsequential
   convergence to a neighborhood of size $O(b + \sqrt{V}/T)$ around a
   Pareto-stationary point. Adds complexity.
2. **Defer SPSA analysis to empirical claims.** The theorem only covers
   the analytic case; for SPSA we have empirical evidence.

Option 2 is consistent with IJOC scope. Option 1 is more ambitious
and tips toward MS OR.

### 4.9 Scope of the appendix

Is 2 pages enough? Breakdown:
- 0.3 pages: assumptions.
- 0.3 pages: bridge lemma statement + sketch.
- 0.4 pages: main proposition statement.
- 0.8 pages: proof (three-step architecture).
- 0.2 pages: remark on "AlignMO routes to PCGrad only where PCGrad's
  assumptions hold" as structural advantage.

This fits in 2 pages if we're disciplined. 3 pages if we include the
bridge lemma proof in full.

Debate: can we get the bridge lemma to 0.3 pages by pure reference
to M-estimation machinery, or do we need to write out the details?

---

## 5. Things to explicitly consider and debate

- **The reviewer's mental model.** What does an IJOC reviewer expect
  when they see a theorem in a computing-algorithm paper? Usually:
  *one* clean statement, *one* proof, *no* rate claims, *no* tight
  constants. Does our proposed theorem fit this template?

- **The "we're not the first" factor.** Nash-MTL, FAMO, CAGrad all
  proved some convergence statement. Ours needs to be comparable but
  *also* distinguishable. What distinguishes ours?
  - Candidate: "*AlignMO is the first MOO method whose convergence
    theorem explicitly covers the non-decomposable-fairness case.*"
    That's the bridge lemma as the distinguishing feature.
  - Candidate: "*AlignMO is the first adaptive MOO method whose
    theorem explicitly covers mode-switching.*" That's the mode-commit
    argument as the distinguishing feature.
  - Use both if we want to be maximally defensible.

- **What we do NOT want from reviewers.**
  - "But what's the rate?" — avoid by not stating a rate.
  - "Why these assumptions?" — avoid by using *exactly* the Désidéri
    / Sener-Koltun / Liu CAGrad assumptions, cited by name.
  - "How does this compare to Nash-MTL's convergence?" — answer
    honestly: our Pareto-stationarity is equivalent to Nash-MTL's in
    the single-mode limit, and strictly more general under mode-switching.

- **Integration with Appendix A.** Appendix A currently has the
  value-level bridge argument (non-decomposability → centered
  surrogate → decomposable). This needs to be rolled into the bridge
  lemma or kept as a preliminary. Which is cleaner?

---

## 6. Deliverable of the discussion

A **decision document**, ~1 page long, written at the end of the
discussion. Save to
`experiments/advisor_review/theory_decisions.md`. Must include:

1. **Target theorem statement** (in words, not LaTeX): what
   specifically we're going to prove.
2. **Reference class anchor**: one of {MGDA, PCGrad Prop 1, CAGrad},
   whose proof architecture we're copying. Named explicitly.
3. **Bridge lemma commitment**: yes or no, and if yes, how detailed.
4. **Projection primitive handling**: parametric over $P$, or
   committed to PCGrad/Nash.
5. **Scope cap**: 2 pages or 3 pages.
6. **Explicit non-goals list**: rate, uniqueness, distance-to-front,
   continuation in λ, SPSA bias treatment.
7. **Open sub-questions for the paper team** (things we can't decide
   in this discussion without more input).

This document is what gets handed to whoever executes the theorem
prompt next (per TODO 2.9 in ALIGNMO_PLAN.md).

---

## 7. Things NOT to do in this discussion

- Do not start writing LaTeX proof fragments. The point is to
  decide WHAT to prove, not HOW.
- Do not expand scope to "what about regret bounds?" or
  "what about generalization?". Those are separate research
  questions outside this paper.
- Do not get stuck on notation. Notation can be fixed later;
  it's the architecture that matters.
- Do not attempt to prove that AlignMO is *better* than PCGrad /
  MGDA / Nash-MTL theoretically. That's not achievable with the
  theorem we're targeting.

---

## 8. One-line version

> "We're deciding whether to prove subsequential Pareto-stationarity
> with a bridge lemma for non-decomposable fairness — or whether the
> venue constraints (IJOC) + design choices (4-mode, Nash-MTL
> candidate, λ as user-specified) push us toward a weaker, stronger,
> or differently-framed theorem. The output is a 1-page decisions
> document that unblocks `TODO 2.9` (theorem execution)."
