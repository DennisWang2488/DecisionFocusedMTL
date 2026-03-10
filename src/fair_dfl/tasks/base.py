"""Task layer for data-driven optimization experiments used by the 8 active methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

import numpy as np


@dataclass
class SplitData:
    x: np.ndarray
    y: np.ndarray


@dataclass
class TaskData:
    train: SplitData
    val: SplitData
    test: SplitData
    groups: np.ndarray
    meta: Dict[str, np.ndarray | float]


class BaseTask:
    name: str
    n_outputs: int

    def generate_data(self, seed: int) -> TaskData:
        raise NotImplementedError

    def compute(
        self,
        raw_pred: np.ndarray,
        true: np.ndarray,
        need_grads: bool,
        fairness_smoothing: float = 1e-6,
    ) -> Dict[str, np.ndarray | float]:
        raise NotImplementedError


def add_bias_column(x: np.ndarray) -> np.ndarray:
    """Append a bias column of ones to a 2D feature matrix."""
    if x.ndim != 2:
        raise ValueError("add_bias_column expects a 2D array.")
    ones = np.ones((x.shape[0], 1), dtype=x.dtype)
    return np.concatenate([x, ones], axis=1)
