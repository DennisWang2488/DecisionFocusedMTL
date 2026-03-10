"""LANCER surrogate trainer used by the lancer method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn


class LancerSurrogate(nn.Module):
    """
    MLP surrogate W that predicts decision loss from (z_pred, z_true).
    """

    def __init__(self, z_dim: int, hidden_dim: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = z_dim
        for _ in range(max(1, n_layers)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()

    def forward(self, z_pred: torch.Tensor, z_true: torch.Tensor) -> torch.Tensor:
        diff = torch.square(z_true - z_pred)
        return self.net(diff).reshape(-1)

    def forward_theta_step(self, z_pred: torch.Tensor, z_true: torch.Tensor) -> torch.Tensor:
        return self.forward(z_pred, z_true).mean()


@dataclass
class LancerConfig:
    c_epochs_init: int = 5
    c_lr_init: float = 0.005
    c_max_iter: int = 1
    lancer_max_iter: int = 1
    c_nbatch: int = 128
    lancer_nbatch: int = 128
    use_replay_buffer: bool = False
    z_regul: float = 0.0
    hidden_dim: int = 64
    n_layers: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4


class LancerTrainer:
    def __init__(self, z_dim: int, device: torch.device, cfg: LancerConfig) -> None:
        self.cfg = cfg
        self.device = device
        self.surrogate = LancerSurrogate(z_dim=z_dim, hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers).to(device)
        self.optimizer = torch.optim.Adam(
            self.surrogate.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )
        self._buf_z_pred: list[np.ndarray] = []
        self._buf_z_true: list[np.ndarray] = []
        self._buf_f_hat: list[np.ndarray] = []

    def warm_start_predictor(
        self,
        predictor: nn.Module,
        x: torch.Tensor,
        z_true: torch.Tensor,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> None:
        ep = int(self.cfg.c_epochs_init if epochs is None else epochs)
        if ep <= 0:
            return
        optimizer = torch.optim.Adam(predictor.parameters(), lr=float(self.cfg.c_lr_init if lr is None else lr))
        mse = nn.MSELoss()
        predictor.train()
        for _ in range(ep):
            pred = predictor(x)
            loss = mse(pred, z_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _push_buffer(self, z_pred: np.ndarray, z_true: np.ndarray, f_hat: np.ndarray) -> None:
        if self.cfg.use_replay_buffer:
            self._buf_z_pred.append(z_pred.copy())
            self._buf_z_true.append(z_true.copy())
            self._buf_f_hat.append(f_hat.copy())
            # bound memory for stability
            if len(self._buf_z_pred) > 64:
                self._buf_z_pred.pop(0)
                self._buf_z_true.pop(0)
                self._buf_f_hat.pop(0)
        else:
            self._buf_z_pred = [z_pred.copy()]
            self._buf_z_true = [z_true.copy()]
            self._buf_f_hat = [f_hat.copy()]

    def update_surrogate(self, z_pred: np.ndarray, z_true: np.ndarray, f_hat: np.ndarray) -> float:
        z_pred = np.asarray(z_pred, dtype=float)
        z_true = np.asarray(z_true, dtype=float)
        f_hat = np.asarray(f_hat, dtype=float).reshape(-1)
        self._push_buffer(z_pred, z_true, f_hat)
        z_pred_all = np.concatenate(self._buf_z_pred, axis=0)
        z_true_all = np.concatenate(self._buf_z_true, axis=0)
        f_hat_all = np.concatenate(self._buf_f_hat, axis=0)

        z_pred_t = torch.as_tensor(z_pred_all, dtype=torch.float32, device=self.device)
        z_true_t = torch.as_tensor(z_true_all, dtype=torch.float32, device=self.device)
        f_hat_t = torch.as_tensor(f_hat_all, dtype=torch.float32, device=self.device)

        self.surrogate.train()
        last_loss = 0.0
        iters = max(int(self.cfg.lancer_max_iter), 1)
        for _ in range(iters):
            pred = self.surrogate(z_pred_t, z_true_t)
            loss = self.surrogate.loss_fn(pred, f_hat_t)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            last_loss = float(loss.detach().cpu().item())
        return last_loss

    def predictor_grad(self, z_pred: torch.Tensor, z_true: torch.Tensor) -> tuple[np.ndarray, float]:
        self.surrogate.eval()
        z_pred_local = z_pred.detach().clone().requires_grad_(True)
        z_true_local = z_true.detach()
        surrogate_loss = self.surrogate.forward_theta_step(z_pred_local, z_true_local)
        grad = torch.autograd.grad(surrogate_loss, z_pred_local, retain_graph=False, create_graph=False)[0]
        return grad.detach().cpu().numpy(), float(surrogate_loss.detach().cpu().item())
