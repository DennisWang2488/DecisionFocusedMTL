"""CvxpyLayers decision gradient strategy (placeholder).

For tasks that define a CVXPY problem, uses cvxpylayers to compute
differentiable gradients through the convex optimization layer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...tasks.base import BaseTask
from ..interface import DecisionGradientStrategy, DecisionResult


class CvxpyLayersStrategy(DecisionGradientStrategy):
    """Placeholder for cvxpylayers-based decision gradients.

    Requires the task to define a build_cvxpy_problem() method that
    returns (problem, parameters, variables) for the cvxpylayers wrapper.
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
            "CvxpyLayersStrategy is a placeholder. "
            "Implement build_cvxpy_problem() on the task and complete this strategy."
        )

    def supports_task(self, task: BaseTask) -> bool:
        return hasattr(task, "build_cvxpy_problem")
