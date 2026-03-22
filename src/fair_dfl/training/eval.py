"""Unified evaluation functions for all methods and task types."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from ..models import PredictorHandle
from ..tasks.base import BaseTask, SplitData
from ..tasks.medical_resource_allocation import MedicalResourceAllocationTask


def eval_split(
    task: BaseTask,
    predictor: PredictorHandle,
    split: SplitData,
    fairness_smoothing: float,
    override_pred: np.ndarray | None = None,
) -> Dict[str, float]:
    """Evaluate a predictor on a non-medical task split."""
    if override_pred is not None:
        raw_pred = override_pred
    else:
        raw_pred = predictor.predict_numpy(split.x)
    out = task.compute(
        raw_pred=raw_pred,
        true=split.y,
        need_grads=False,
        fairness_smoothing=fairness_smoothing,
    )
    metrics = {
        "regret": float(out["loss_dec"]),
        "pred_mse": float(out["loss_pred"]),
        "fairness": float(out["loss_fair"]),
        "solver_calls_eval": float(out.get("solver_calls", 0)),
        "decision_ms_eval": float(out.get("decision_ms", 0.0)),
    }
    for key, metric_name in [
        ("loss_dec_normalized", "regret_normalized"),
        ("loss_dec_normalized_true", "regret_normalized_true"),
        ("loss_dec_normalized_pred_obj", "regret_normalized_pred_obj"),
    ]:
        if key in out:
            metrics[metric_name] = float(out[key])
    return metrics


def eval_split_medical(
    task: MedicalResourceAllocationTask,
    predictor: PredictorHandle,
    split_name: str,
    fairness_smoothing: float,
    override_pred: np.ndarray | None = None,
) -> Dict[str, float]:
    """Evaluate a predictor on a medical task split."""
    if override_pred is not None:
        return task.evaluate_split(
            split=split_name, pred=override_pred, fairness_smoothing=fairness_smoothing,
        )
    split = task._splits[split_name]
    pred = predictor.predict_numpy(split.x).reshape(-1)
    return task.evaluate_split(
        split=split_name, pred=pred, fairness_smoothing=fairness_smoothing,
    )


_EMPTY_METRICS: Dict[str, float] = {
    "regret": 0.0, "pred_mse": 0.0, "fairness": 0.0,
    "solver_calls_eval": 0.0, "decision_ms_eval": 0.0,
}


def evaluate_model(
    task: BaseTask,
    predictor: PredictorHandle,
    data: Any,
    fairness_smoothing: float,
    saa_override_val: np.ndarray | None = None,
    saa_override_test: np.ndarray | None = None,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Evaluate on both val and test splits, dispatching to the correct evaluator.

    If the val split is empty (val_fraction=0.0), val_metrics returns zeros.
    """
    if isinstance(task, MedicalResourceAllocationTask):
        val_split = task._splits.get("val")
        if val_split is not None and val_split.x.shape[0] > 0:
            val_metrics = eval_split_medical(
                task=task, predictor=predictor, split_name="val",
                fairness_smoothing=fairness_smoothing, override_pred=saa_override_val,
            )
        else:
            val_metrics = dict(_EMPTY_METRICS)
        test_metrics = eval_split_medical(
            task=task, predictor=predictor, split_name="test",
            fairness_smoothing=fairness_smoothing, override_pred=saa_override_test,
        )
    else:
        if data.val is not None and data.val.x.shape[0] > 0:
            val_metrics = eval_split(
                task=task, predictor=predictor, split=data.val,
                fairness_smoothing=fairness_smoothing, override_pred=saa_override_val,
            )
        else:
            val_metrics = dict(_EMPTY_METRICS)
        test_metrics = eval_split(
            task=task, predictor=predictor, split=data.test,
            fairness_smoothing=fairness_smoothing, override_pred=saa_override_test,
        )
    return val_metrics, test_metrics
