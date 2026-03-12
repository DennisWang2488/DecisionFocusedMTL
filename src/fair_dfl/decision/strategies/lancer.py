"""LANCER decision gradient strategy.

Uses a learned MLP surrogate to approximate the decision objective
and provide gradients for the predictor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ...advanced.lancer import LancerConfig, LancerTrainer
from ...tasks.base import BaseTask
from ...tasks.medical_resource_allocation import MedicalResourceAllocationTask
from ..interface import DecisionGradientStrategy, DecisionResult


class LancerStrategy(DecisionGradientStrategy):
    """LANCER: Learned Approximation via Contrastive surrogate.

    Trains an MLP surrogate to approximate the decision objective.
    The surrogate provides differentiable gradients for the predictor
    without requiring solver differentiation.
    """

    def __init__(
        self,
        z_dim: int = 1,
        device: torch.device | None = None,
        lancer_cfg: dict | None = None,
    ) -> None:
        self._z_dim = z_dim
        self._device = device or torch.device("cpu")
        self._cfg = LancerConfig(**(lancer_cfg or {}))
        self._trainer: LancerTrainer | None = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._trainer is None:
            self._trainer = LancerTrainer(
                z_dim=self._z_dim, device=self._device, cfg=self._cfg,
            )
            self._initialized = True

    def warm_start(self, module: torch.nn.Module, x_all: torch.Tensor, y_all: torch.Tensor) -> None:
        """Warm-start the predictor surrogate from training data."""
        self._ensure_initialized()
        assert self._trainer is not None
        self._trainer.warm_start_predictor(module, x_all, y_all)

    def compute(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        task: BaseTask,
        need_grads: bool = True,
        fairness_smoothing: float = 1e-6,
        pred_tensor: torch.Tensor | None = None,
        **ctx: Any,
    ) -> DecisionResult:
        self._ensure_initialized()
        assert self._trainer is not None

        task_output = ctx.get("task_output")
        if task_output is not None:
            out = task_output
            base_solver_calls = 0
            base_decision_ms = 0.0
        elif isinstance(task, MedicalResourceAllocationTask):
            out = task.compute_batch(
                raw_pred=pred, true=true,
                cost=ctx["cost"], race=ctx["race"],
                need_grads=True, fairness_smoothing=fairness_smoothing,
            )
            base_solver_calls = int(out.get("solver_calls", 0))
            base_decision_ms = float(out.get("decision_ms", 0.0))
        else:
            out = task.compute(
                raw_pred=pred, true=true,
                need_grads=False, fairness_smoothing=fairness_smoothing,
            )
            base_solver_calls = int(out.get("solver_calls", 0))
            base_decision_ms = float(out.get("decision_ms", 0.0))

        if not need_grads:
            return DecisionResult(
                loss_dec=float(out["loss_dec"]),
                grad_dec=np.zeros_like(pred),
                solver_calls=base_solver_calls,
                decision_ms=base_decision_ms,
                task_output=out,
            )

        # Update surrogate
        z_pred_np = pred.reshape(-1, 1)
        z_true_np = np.asarray(true, dtype=float).reshape(-1, 1)
        f_hat = np.full(z_pred_np.shape[0], float(out["loss_dec"]), dtype=float)
        surrogate_loss = self._trainer.update_surrogate(z_pred_np, z_true_np, f_hat)

        # Get LANCER gradient
        if pred_tensor is None:
            pred_tensor = torch.as_tensor(pred, dtype=torch.float32, device=self._device)
        z_true_t = torch.as_tensor(z_true_np, dtype=torch.float32, device=self._device)
        g_lancer, pred_loss = self._trainer.predictor_grad(
            pred_tensor.reshape(-1, 1), z_true_t,
        )
        total_surrogate_loss = float(surrogate_loss + pred_loss)
        grad_dec = g_lancer.reshape(pred.shape)

        return DecisionResult(
            loss_dec=float(out["loss_dec"]),
            grad_dec=grad_dec,
            solver_calls=base_solver_calls,
            decision_ms=base_decision_ms,
            task_output=out,
            extra={"lancer_surrogate_loss": total_surrogate_loss},
        )

    def reset(self) -> None:
        pass  # Keep surrogate across stages

    def supports_task(self, task: BaseTask) -> bool:
        return isinstance(task, MedicalResourceAllocationTask)
