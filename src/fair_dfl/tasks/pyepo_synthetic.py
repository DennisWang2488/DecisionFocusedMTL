"""PyEPO-backed synthetic task adapters for smoke testing core methods."""

from dataclasses import dataclass

import numpy as np
from pyepo.data.knapsack import genData as gen_knapsack_data
from pyepo.data.portfolio import genData as gen_portfolio_data

from .base import SplitData, TaskData
from .portfolio_qp import PortfolioQPTask
from .portfolio_qp_simplex import PortfolioQPSimplexTask
from .resource_allocation import ResourceAllocationTask


@dataclass
class PyEPONonlinearKnapsackTask(ResourceAllocationTask):
    pyepo_dim: int = 1
    pyepo_deg: int = 2
    pyepo_noise_width: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        self.name = "pyepo_nonlinear_knapsack"

    def generate_data(self, seed: int) -> TaskData:
        total = self.n_samples_train + self.n_samples_val + self.n_samples_test
        weights, x, c = gen_knapsack_data(
            num_data=total,
            num_features=self.n_features,
            num_items=self.n_items,
            dim=self.pyepo_dim,
            deg=self.pyepo_deg,
            noise_width=self.pyepo_noise_width,
            seed=seed,
        )
        y = c.astype(float)

        groups = np.zeros(self.n_items, dtype=int)
        groups[self.n_items // 2 :] = 1
        group_shift = np.where(groups == 0, -self.group_bias, self.group_bias)

        rng = np.random.default_rng(seed + 97)
        y = y + group_shift[None, :] + rng.normal(scale=self.noise_std, size=y.shape)
        y = np.clip(y, 1e-6, None)

        costs = np.asarray(weights, dtype=float)
        if costs.ndim == 2:
            costs = costs.mean(axis=0)
        costs = np.clip(costs, 1e-6, None)

        i_train = self.n_samples_train
        i_val = i_train + self.n_samples_val
        return TaskData(
            train=SplitData(x=x[:i_train], y=y[:i_train]),
            val=SplitData(x=x[i_train:i_val], y=y[i_train:i_val]),
            test=SplitData(x=x[i_val:], y=y[i_val:]),
            groups=groups,
            meta={"costs": costs},
        )


@dataclass
class PyEPOPortfolioQPTask(PortfolioQPTask):
    pyepo_deg: int = 2
    pyepo_noise_level: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.name = "pyepo_portfolio_qp"

    def generate_data(self, seed: int) -> TaskData:
        total = self.n_samples_train + self.n_samples_val + self.n_samples_test
        sigma, x, r = gen_portfolio_data(
            num_data=total,
            num_features=self.n_features,
            num_assets=self.n_assets,
            deg=self.pyepo_deg,
            noise_level=self.pyepo_noise_level,
            seed=seed,
        )
        y = r.astype(float)

        groups = np.zeros(self.n_assets, dtype=int)
        groups[self.n_assets // 2 :] = 1
        group_shift = np.where(groups == 0, -self.group_bias, self.group_bias)

        rng = np.random.default_rng(seed + 197)
        y = y + group_shift[None, :] + rng.normal(scale=self.noise_std, size=y.shape)

        i_train = self.n_samples_train
        i_val = i_train + self.n_samples_val
        return TaskData(
            train=SplitData(x=x[:i_train], y=y[:i_train]),
            val=SplitData(x=x[i_train:i_val], y=y[i_train:i_val]),
            test=SplitData(x=x[i_val:], y=y[i_val:]),
            groups=groups,
            meta={"sigma": np.asarray(sigma, dtype=float)},
        )


@dataclass
class PyEPOPortfolioQPSimplexTask(PortfolioQPSimplexTask):
    pyepo_deg: int = 2
    pyepo_noise_level: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.name = "pyepo_portfolio_qp_simplex"

    def generate_data(self, seed: int) -> TaskData:
        total = self.n_samples_train + self.n_samples_val + self.n_samples_test
        sigma, x, r = gen_portfolio_data(
            num_data=total,
            num_features=self.n_features,
            num_assets=self.n_assets,
            deg=self.pyepo_deg,
            noise_level=self.pyepo_noise_level,
            seed=seed,
        )
        y = r.astype(float)

        groups = np.zeros(self.n_assets, dtype=int)
        groups[self.n_assets // 2 :] = 1
        group_shift = np.where(groups == 0, -self.group_bias, self.group_bias)

        rng = np.random.default_rng(seed + 197)
        y = y + group_shift[None, :] + rng.normal(scale=self.noise_std, size=y.shape)

        i_train = self.n_samples_train
        i_val = i_train + self.n_samples_val
        return TaskData(
            train=SplitData(x=x[:i_train], y=y[:i_train]),
            val=SplitData(x=x[i_train:i_val], y=y[i_train:i_val]),
            test=SplitData(x=x[i_val:], y=y[i_val:]),
            groups=groups,
            meta={"sigma": np.asarray(sigma, dtype=float)},
        )
