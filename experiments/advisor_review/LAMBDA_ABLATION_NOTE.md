# Ablation note: Should λ be auto-selected?

> **Status:** design note, not yet executed. Referenced from `ALIGNMO_PLAN.md`.
> Target section in the paper: discussion paragraph in Section 4
> (AlignMO design rationale) + optional ablation table in Section 5.

## 1. The question

AlignMO currently has two numerical knobs (`μ`, gradient normalization)
and one preference knob (`λ`, fairness weight). `μ` is auto-managed
(substituted by `μ_floor` when scale imbalance is detected); `λ` is
user-specified and swept over `{0, 0.5, 1, 2}` to trace the
fairness/efficiency frontier.

Some MOO methods (MGDA, FAMO) make **all** weights internal/adaptive.
The question is whether AlignMO should do the same for `λ`.

## 2. The answer, with reasoning

**Keep `λ` user-specified. Document this as a deliberate design choice,
not an oversight.**

The reasoning distinguishes two categories of weight:

| Weight | Category | Role | Auto? |
| --- | --- | --- | --- |
| `μ` (prediction anchor) | Numerical stabilizer | Prevents decision-gradient dominance under noisy decision gradients (SPSA). Has no semantic content; the user has no "opinion" about μ beyond "don't let training diverge." | Yes — promote to `μ_floor` when needed. |
| `λ` (fairness weight) | Preference | Encodes "how much prediction fairness is worth to me relative to efficiency." Different stakeholders, different regulatory regimes, and different deployment contexts have different answers. | **No** — auto-selection hides a value judgment inside an optimization heuristic. |

The MGDA/FAMO reference class auto-weights all objectives because those
methods assume the objectives are **commensurable prediction tasks of
similar importance**. In fairness-constrained decision making, `λ` is
not a computational quantity; it's an ethical trade-off. Removing user
control over `λ` reduces the method's fitness for real deployment.

Moreover, MGDA-with-auto-`λ` produces **one Pareto point** (the
min-norm direction), not a curve. Practitioners often want the full
fairness/efficiency curve to present to stakeholders. Fixed `λ` with
a sweep gives exactly that.

## 3. The ablation protocol

To empirically justify the design choice, run this ablation alongside
the main AlignMO evaluation. If the pilot (Phase 1) is GREEN, this
ablation becomes Phase 2 TODO 2.7b.

### 3.1 Methods compared

| Name | How `λ` (and `μ`) are chosen | Expected behavior |
| --- | --- | --- |
| **AlignMO (main)** | `λ` user-specified over sweep; `μ` auto (stabilizer) | Produces a fairness/efficiency Pareto curve |
| **AlignMO-free** | `λ, μ` both found by MGDA min-norm over `{L_dec, L_pred, L_fair}` at each step; projection per AlignMO's mode selector | Produces one Pareto point per run |
| **AlignMO-FAMO** | `λ, μ` via FAMO's loss-decay adaptation; projection per AlignMO | Produces one Pareto point per run with different bias |

All three share AlignMO's mode selector (scalarized / projected /
anchored / anchored-projected per the 4-mode framing). The only
difference is how objective weights are chosen.

### 3.2 Grid

- Task: healthcare at `n=5000`, full cohort if time permits.
- `α ∈ {0.5, 2.0}` (reuse main-text setting).
- Gradient regime: analytic.
- Seeds: 5.
- Training steps: 70 per stage.
- For AlignMO (main): `λ ∈ {0, 0.5, 1, 2}` sweep.
- For AlignMO-free / AlignMO-FAMO: no sweep; one run.

### 3.3 Metrics

Primary:
- **Pareto coverage:** the convex hull of (test regret, test fairness)
  points each method produces. Measure area dominated.
- **Practitioner utility at a target fairness level:** for a fixed
  fairness violation budget `F_max ∈ {20, 30, 50}` (MAD units on the
  healthcare task), what is the best regret achievable by each method?

Secondary:
- Total wall-clock: AlignMO (main) pays 4× for the λ sweep;
  AlignMO-free pays 1×. Is the curve worth the 4× cost?
- Location of AlignMO-free's single point: does it land near a useful
  operating point, or does it sit somewhere no stakeholder would pick?

### 3.4 Expected outcomes and decision rule

| Outcome | Conclusion |
| --- | --- |
| AlignMO (main) Pareto-dominates AlignMO-free on ≥ 2 of 3 fairness budgets | **Confirms the design choice.** Write up in Section 4 discussion. |
| AlignMO-free lands on the Pareto curve AlignMO traces but at only one (unpredictable) point | **Confirms design choice + gives a nice figure.** The ablation becomes a selling point for fixed-λ operation. |
| AlignMO-free strictly dominates AlignMO on some fairness budgets | **Rethink.** Possibly AlignMO-free is a better default; reposition the paper to use it. Unlikely outcome but worth stating the rule. |

### 3.5 Writeup implications

If the expected outcome holds:

In **Section 4.x ("AlignMO design rationale")**, add a paragraph:

> AlignMO auto-manages the prediction-anchor `μ` (a numerical stabilizer
> with no semantic content) but preserves `λ` as a user-specified
> preference knob. This deliberate asymmetry reflects the distinction
> between computational trade-offs and ethical ones: fairness-vs-
> efficiency is a stakeholder choice, not a property of the
> optimization landscape. Appendix X documents an ablation in which `λ`
> is also auto-selected via MGDA-style min-norm; the resulting single
> Pareto point lands unpredictably across cells, in contrast to the
> controllable trade-off curve produced by the `λ`-sweep protocol.

In **Section 5.x ("AlignMO evaluation")**, add a sub-table:

```
Method              | Pareto curve    | Operating-point utility
AlignMO (main)      | ✓ (4 points)    | best at F_max=20, 30, 50
AlignMO-free        | ✗ (1 point)     | worse at 2/3 targets
AlignMO-FAMO        | ✗ (1 point)     | worse at 3/3 targets
```

In the **Appendix**, include the full-grid numbers and the exact
operating-point computation.

## 4. Implementation notes for the ablation

The ablation can reuse the infrastructure AlignMO is built on:

- `AlignMOHandler` already computes per-objective gradients for the
  mode selector. Adding AlignMO-free only requires swapping the
  weighted-sum step for an MGDA min-norm step (reuse `MGDAHandler`).
- FAMO is already available in `mo_handler.py`; wrap it similarly.
- Same CSV schema, same evaluation pipeline. No new tasks or metrics.

Estimated effort: **3 days** after AlignMO main implementation lands.

## 5. Non-goals

- No new theoretical analysis of MGDA or FAMO. The ablation is
  empirical only.
- No deep dive into the philosophy of "should fairness weights be
  auto-set." A one-paragraph discussion in the paper body is enough.
- No user study. Practitioner utility is measured computationally
  (via the operating-point protocol), not via human preferences.

## 6. When to run this ablation

After AlignMO main Phase 2 completes and the headline evaluation shows
AlignMO competes against the fixed handlers. Before Phase 3 (paper
writeup), so the ablation figure can land in the draft.
