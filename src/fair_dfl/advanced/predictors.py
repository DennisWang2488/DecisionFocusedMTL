"""Predictor helpers for advanced methods (ffo, nce, lancer)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d_in = in_dim
        for _ in range(max(n_layers, 1)):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.ReLU())
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class PredictorHandle:
    family: str
    module: nn.Module
    device: torch.device

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xb = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            out = self.module(xb)
        return out.detach().cpu().numpy()

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.module.parameters()

    def train(self) -> None:
        self.module.train()

    def eval(self) -> None:
        self.module.eval()


def build_predictor(
    family: str,
    input_dim: int,
    output_dim: int,
    seed: int,
    device: torch.device,
    mlp_hidden_dim: int = 64,
    mlp_layers: int = 2,
) -> PredictorHandle:
    torch.manual_seed(seed)
    fam = family.lower().strip()
    if fam == "linear":
        module: nn.Module = nn.Linear(input_dim, output_dim)
    elif fam == "mlp":
        module = _MLP(input_dim, output_dim, hidden_dim=int(mlp_hidden_dim), n_layers=int(mlp_layers))
    else:
        raise ValueError(f"Unsupported predictor family: {family}")
    module.to(device)
    return PredictorHandle(family=fam, module=module, device=device)


def flatten_param_grads(module: nn.Module) -> np.ndarray:
    grads = []
    for p in module.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p).reshape(-1))
        else:
            grads.append(p.grad.detach().reshape(-1))
    if not grads:
        return np.zeros(1, dtype=float)
    return torch.cat(grads).detach().cpu().numpy()
