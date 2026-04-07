"""Factory for building DecisionGradientComputer instances."""

from __future__ import annotations

from typing import Any, Dict

import torch

from ..tasks.base import BaseTask
from .interface import DecisionGradientStrategy, DecisionResult
from .strategies.analytic import AnalyticStrategy
from .strategies.finite_diff import FiniteDiffStrategy


class DecisionGradientComputer:
    """Facade wrapping a DecisionGradientStrategy.

    Provides the primary interface for the training loop to compute
    decision gradients regardless of the backend strategy.
    """

    def __init__(self, strategy: DecisionGradientStrategy) -> None:
        self.strategy = strategy

    def compute(self, pred, true, task, **ctx) -> DecisionResult:
        return self.strategy.compute(pred=pred, true=true, task=task, **ctx)

    def reset(self) -> None:
        self.strategy.reset()

    @property
    def name(self) -> str:
        return self.strategy.name


def build_decision_gradient(
    train_cfg: Dict[str, Any],
    task: BaseTask,
    device: torch.device | None = None,
) -> DecisionGradientComputer:
    """Build a DecisionGradientComputer from training config.

    Reads 'decision_grad_backend' from train_cfg:
        "analytic"    -> AnalyticStrategy (default)
        "finite_diff" -> FiniteDiffStrategy
        "ffo"         -> FoldOptStrategy
        "nce"         -> NCEStrategy
        "lancer"      -> LancerStrategy
        "cvxpylayers" -> CvxpyLayersStrategy
        "autograd"    -> TorchAutogradStrategy
    """
    backend = str(train_cfg.get("decision_grad_backend", "analytic")).strip().lower()

    if backend == "analytic":
        strategy: DecisionGradientStrategy = AnalyticStrategy()

    elif backend == "finite_diff":
        eps = float(train_cfg.get("decision_grad_fd_eps", 1e-3))
        strategy = FiniteDiffStrategy(eps=eps)

    elif backend == "spsa":
        from .strategies.spsa import SPSAStrategy
        eps = float(train_cfg.get("decision_grad_spsa_eps", 5e-3))
        n_dirs = int(train_cfg.get("decision_grad_spsa_n_dirs", 1))
        strategy = SPSAStrategy(eps=eps, n_dirs=n_dirs)

    elif backend == "ffo":
        from .strategies.fold_opt import FoldOptStrategy
        ffo_cfg = dict(train_cfg.get("ffo", {}))
        budget = getattr(task, "budget", 2500.0)
        strategy = FoldOptStrategy(
            ffo_cfg=ffo_cfg,
            budget=float(budget),
            device=device or torch.device("cpu"),
        )

    elif backend == "nce":
        import numpy as np
        from .strategies.nce import NCEStrategy
        nce_cfg = dict(train_cfg.get("nce", {}))
        strategy = NCEStrategy(
            pool_size=int(nce_cfg.get("pool_size", 32)),
            solve_ratio=float(nce_cfg.get("solve_ratio", 1.0)),
            refresh_interval=int(nce_cfg.get("refresh_interval", 1)),
        )

    elif backend == "lancer":
        from .strategies.lancer import LancerStrategy
        lancer_cfg = dict(train_cfg.get("lancer", {}))
        strategy = LancerStrategy(
            z_dim=1,
            device=device or torch.device("cpu"),
            lancer_cfg=lancer_cfg,
        )

    elif backend == "cvxpylayers":
        from .strategies.cvxpylayers import CvxpyLayersStrategy
        strategy = CvxpyLayersStrategy()

    elif backend == "autograd":
        from .strategies.torch_autograd import TorchAutogradStrategy
        strategy = TorchAutogradStrategy()

    else:
        raise ValueError(
            f"Unknown decision_grad_backend: {backend!r}. "
            f"Options: analytic, finite_diff, spsa, ffo, nce, lancer, cvxpylayers, autograd"
        )

    if not strategy.supports_task(task):
        raise ValueError(
            f"Decision gradient strategy {strategy.name!r} does not support "
            f"task {type(task).__name__!r}."
        )

    return DecisionGradientComputer(strategy)
