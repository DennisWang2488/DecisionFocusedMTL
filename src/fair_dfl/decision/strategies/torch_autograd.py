"""Torch autograd decision gradient strategy (placeholder).

For tasks where the solver is implemented in differentiable PyTorch,
gradients can be computed via standard torch autograd.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...tasks.base import BaseTask
from ..interface import DecisionGradientStrategy, DecisionResult


class TorchAutogradStrategy(DecisionGradientStrategy):
    """Placeholder for future torch-autograd-based decision gradients.

    Use this when the optimization solver is fully implemented in
    differentiable PyTorch operations (no external solver calls).
    """

    def compute(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        task: BaseTask,
        need_grads: bool = True,
        fairness_smoothing: float = 1e-6,
        **ctx: Any,
    ) -> DecisionResult:
        raise NotImplementedError(
            "TorchAutogradStrategy is a placeholder for future use. "
            "Implement a differentiable solver in PyTorch and override this method."
        )

    def supports_task(self, task: BaseTask) -> bool:
        return False
