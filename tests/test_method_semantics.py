"""Experiment semantics tests: method configs, MOO payloads, and edge cases."""

import numpy as np
import pytest

from fair_dfl.training.method_spec import MethodSpec, resolve_method_spec
from fair_dfl.training.loop import _build_active_moo_payload, _active_spec, _pred_weight


class TestMethodSpecResolution:
    def test_all_configs_resolve(self):
        """Every method in ALL_METHOD_CONFIGS should resolve without error."""
        from configs import ALL_METHOD_CONFIGS
        for name, cfg in ALL_METHOD_CONFIGS.items():
            spec = resolve_method_spec(cfg)
            assert isinstance(spec, MethodSpec), f"Failed to resolve {name}"

    def test_nf_variants_have_no_fairness(self):
        """No-fairness variants should have use_fair=False."""
        from configs import ALL_METHOD_CONFIGS
        for name, cfg in ALL_METHOD_CONFIGS.items():
            if name.endswith("-nf"):
                spec = resolve_method_spec(cfg)
                assert not spec.use_fair, f"{name} should have use_fair=False"

    def test_saa_wdro_no_decision(self):
        """SAA and WDRO should not use decision gradients."""
        from configs import ALL_METHOD_CONFIGS
        for name in ["SAA", "WDRO"]:
            spec = resolve_method_spec(ALL_METHOD_CONFIGS[name])
            assert not spec.use_dec, f"{name} should have use_dec=False"


class TestActiveSpecWarmstart:
    def test_warmstart_disables_decision(self):
        base = MethodSpec(
            use_dec=True, use_pred=True, use_fair=True,
            pred_weight_mode="schedule", continuation=True,
            allow_orthogonalization=True,
        )
        warmstart_spec = _active_spec(base, iter_idx=0, warmstart_steps=10)
        assert not warmstart_spec.use_dec
        assert warmstart_spec.use_pred

    def test_after_warmstart_uses_base(self):
        base = MethodSpec(
            use_dec=True, use_pred=True, use_fair=True,
            pred_weight_mode="schedule", continuation=True,
            allow_orthogonalization=True,
        )
        post_warmstart = _active_spec(base, iter_idx=10, warmstart_steps=10)
        assert post_warmstart.use_dec


class TestPredWeight:
    def test_zero_mode(self):
        assert _pred_weight("zero", t=5, alpha_schedule_cfg={}) == 0.0

    def test_fixed1_mode(self):
        assert _pred_weight("fixed1", t=5, alpha_schedule_cfg={}) == 1.0

    def test_schedule_mode(self):
        cfg = {"type": "inv_sqrt", "alpha0": 1.0, "alpha_min": 0.0}
        w = _pred_weight("schedule", t=0, alpha_schedule_cfg=cfg)
        assert w == pytest.approx(1.0)
        w_later = _pred_weight("schedule", t=100, alpha_schedule_cfg=cfg)
        assert w_later < w  # should decay

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            _pred_weight("bogus", t=0, alpha_schedule_cfg={})


class TestBuildActiveMOOPayload:
    def _make_spec(self, use_dec=True, use_pred=True, use_fair=True):
        return MethodSpec(
            use_dec=use_dec, use_pred=use_pred, use_fair=use_fair,
            pred_weight_mode="schedule", continuation=True,
            allow_orthogonalization=False,
        )

    def test_all_active(self):
        from fair_dfl.algorithms.mo_handler import WeightedSumHandler
        handler = WeightedSumHandler(weights={})
        spec = self._make_spec()
        out = {"loss_pred": 1.0, "loss_dec": 2.0, "loss_fair": 3.0}
        grads, losses = _build_active_moo_payload(
            iter_spec=spec, out=out,
            g_dec_param=np.ones(5), g_pred_param=np.ones(5) * 2, g_fair_param=np.ones(5) * 3,
            mo_handler=handler,
        )
        assert set(grads.keys()) == {"pred_loss", "decision_regret", "pred_fairness"}
        assert set(losses.keys()) == {"pred_loss", "decision_regret", "pred_fairness"}

    def test_nf_excludes_fairness(self):
        from fair_dfl.algorithms.mo_handler import PCGradHandler
        handler = PCGradHandler()
        spec = self._make_spec(use_fair=False)
        out = {"loss_pred": 1.0, "loss_dec": 2.0, "loss_fair": 0.0}
        grads, losses = _build_active_moo_payload(
            iter_spec=spec, out=out,
            g_dec_param=np.ones(5), g_pred_param=np.ones(5) * 2, g_fair_param=np.zeros(5),
            mo_handler=handler,
        )
        assert "pred_fairness" not in grads
        assert "pred_fairness" not in losses
        assert set(grads.keys()) == {"pred_loss", "decision_regret"}

    def test_pred_only_method(self):
        from fair_dfl.algorithms.mo_handler import WeightedSumHandler
        handler = WeightedSumHandler(weights={})
        spec = self._make_spec(use_dec=False, use_fair=False)
        out = {"loss_pred": 1.0, "loss_dec": 0.0, "loss_fair": 0.0}
        grads, losses = _build_active_moo_payload(
            iter_spec=spec, out=out,
            g_dec_param=np.zeros(5), g_pred_param=np.ones(5), g_fair_param=np.zeros(5),
            mo_handler=handler,
        )
        assert set(grads.keys()) == {"pred_loss"}
