# Note on the "DFL shines" regime and paper framing

**Status:** advisor feedback; documentation only — no code change.

## What the advisor flagged

When DFL/MOO methods are compared to a vanilla two-stage PTO baseline, the gap
between the two depends sharply on **how capable the predictor is** and
**whether validation-based model selection is used**:

* **DFL shines (small gap regime).** When the predictor is *less* capable
  (small model, no/limited validation, harder synthetic task with low SNR,
  or high model misspecification) DFL and MOO methods clearly dominate PTO
  on decision regret. The intuition: when the predictor cannot fit ground
  truth well, baking the decision objective into training matters because
  the residual prediction error gets routed through the loss that actually
  matters at deployment time.
* **PTO catches up (capable model + validation).** With a sufficiently
  capable predictor *and* validation-based model selection, the residual
  prediction error shrinks to a regime where PTO becomes competitive (and
  in many cases statistically indistinguishable from DFL) on decision
  regret, while having a much simpler training pipeline.

Showing only the "DFL shines" regime in the paper risks looking like
**p-hacking**: it would suggest the empirical advantage is more universal
than it actually is.

## Plan for the paper

1. **Main paper** — keep the strong DFL-shines results that motivate the
   approach. They are real, the regime is realistic for many practical
   settings (limited data, no held-out validation, hard non-linear targets),
   and they are the cleanest way to demonstrate the *mechanism* of the
   integrated loss.
2. **Appendix** — also present the **capable model + validation** results
   (e.g. healthcare with the full 48k patient cohort, MLP predictor, val
   split active, lambda chosen on val) for transparency. The story there is
   "with enough data and proper model selection, PTO is competitive — but
   our methods still match it on decision regret while explicitly trading
   off prediction fairness, which PTO cannot do."
3. **Discussion / limitations** — explicitly call out the regime dependence
   so reviewers do not have to reverse-engineer it. The paper's central
   claim — *"adding prediction fairness does not lead to strict drawbacks
   in decision regret, and often improves it"* — is preserved by both
   regimes; only the magnitude of the DFL-vs-PTO gap shifts.

## Operational checklist for new experiments

When running the healthcare experiment:

* If the result is intended for the **main paper**, use a small / limited
  predictor or omit validation-based model selection so the DFL effect is
  visible. Document the choice in the experiment config.
* If the result is intended for the **appendix transparency table**, use a
  full-capacity MLP predictor *with* val-based model selection enabled.
* Always log both train and test regret/MSE/fairness — the new
  `train_regret` / `train_pred_mse` / `train_fairness` columns in the
  stage CSV (added by the train-eval changes alongside this note) make it
  easy to spot a generalisation gap that would otherwise hide regime
  effects.

## Why no code knob for "misspecification mode"

A flag like `--misspec on/off` would just be a flag for the model size /
validation toggle. The advisor explicitly recommended **not** adding such a
knob — it would hide the methodological choice instead of forcing each
experiment script to make it explicitly. The capability gap is already
controlled by existing parameters (`model.hidden_dim`, `model.n_layers`,
`val_fraction`, model selection logic). Pick them per experiment; do not
flag them.
