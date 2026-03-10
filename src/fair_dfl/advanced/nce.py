"""NCE solution-pool surrogate used by the nce method."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class NCELossOutput:
    loss: float
    grad_pred: np.ndarray
    pool_size: int


class NCESolutionPool:
    """
    PyEPO-style contrastive pool over candidate decisions.
    """

    def __init__(self, pool_size: int, sense: str = "maximize") -> None:
        self.pool = deque(maxlen=max(int(pool_size), 1))
        sense_l = str(sense).lower().strip()
        if sense_l not in {"maximize", "minimize"}:
            raise ValueError(f"Unsupported sense: {sense}")
        self.sense = sense_l

    def update(self, decision: np.ndarray) -> None:
        d = np.asarray(decision, dtype=float).reshape(-1)
        if not np.all(np.isfinite(d)):
            return
        self.pool.append(d.copy())

    def _stack_pool(self, dim: int) -> np.ndarray:
        if not self.pool:
            return np.zeros((0, dim), dtype=float)
        arr = np.asarray(self.pool, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != dim:
            return np.zeros((0, dim), dtype=float)
        return arr

    def loss_and_grad(
        self,
        pred_score: np.ndarray,
        true_decision: np.ndarray,
    ) -> NCELossOutput:
        c = np.asarray(pred_score, dtype=float).reshape(-1)
        w_true = np.asarray(true_decision, dtype=float).reshape(-1)
        pool = self._stack_pool(c.size)
        if pool.shape[0] == 0:
            zero = np.zeros_like(c)
            return NCELossOutput(loss=0.0, grad_pred=zero, pool_size=0)

        obj_true = float(c @ w_true)
        obj_pool = pool @ c

        mean_pool = pool.mean(axis=0)
        if self.sense == "maximize":
            loss = float(np.mean(obj_pool - obj_true))
            grad = mean_pool - w_true
        else:
            loss = float(np.mean(obj_true - obj_pool))
            grad = w_true - mean_pool

        return NCELossOutput(loss=loss, grad_pred=grad, pool_size=int(pool.shape[0]))
