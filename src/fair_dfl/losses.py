"""Prediction/fairness loss utilities shared by task implementations."""

import numpy as np


def softplus_with_grad(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = np.maximum(z, 0.0)
    exp_term = np.exp(-np.abs(z))
    value = positive + np.log1p(exp_term)
    sigmoid = 1.0 / (1.0 + np.exp(-z))
    return value, sigmoid


def mse_loss_and_grad(pred: np.ndarray, true: np.ndarray) -> tuple[float, np.ndarray]:
    diff = pred - true
    loss = float(np.mean(diff * diff))
    grad = (2.0 / pred.size) * diff
    return loss, grad


def group_mse_mad_loss_and_grad(
    pred: np.ndarray,
    true: np.ndarray,
    groups: np.ndarray,
    smoothing: float = 1e-6,
) -> tuple[float, np.ndarray]:
    unique_groups = np.unique(groups)
    errors = (pred - true) ** 2
    group_mse = []
    for g in unique_groups:
        mask = groups == g
        group_mse.append(errors[:, mask].mean())
    group_mse_arr = np.asarray(group_mse, dtype=float)
    mean_mse = float(group_mse_arr.mean())
    gap = group_mse_arr - mean_mse
    smooth_abs = np.sqrt(gap * gap + smoothing)
    loss = float(smooth_abs.mean())

    # d(loss)/d(mse_g) = (1/G) * (phi'(gap_g) - mean_h phi'(gap_h))
    dphi = gap / smooth_abs
    dloss_dmse = (dphi - dphi.mean()) / max(len(unique_groups), 1)

    grad = np.zeros_like(pred)
    for idx, g in enumerate(unique_groups):
        mask = groups == g
        denom = pred.shape[0] * int(mask.sum())
        if denom == 0:
            continue
        grad[:, mask] = dloss_dmse[idx] * (2.0 * (pred[:, mask] - true[:, mask]) / float(denom))
    return loss, grad


def group_mse_generalized_entropy_loss_and_grad(
    pred: np.ndarray,
    true: np.ndarray,
    groups: np.ndarray,
    alpha: float = 2.0,
    eps: float = 1e-8,
) -> tuple[float, np.ndarray]:
    unique_groups = np.unique(groups)
    errors = (pred - true) ** 2
    group_mse = []
    for g in unique_groups:
        mask = groups == g
        group_mse.append(errors[:, mask].mean())
    group_mse_arr = np.asarray(group_mse, dtype=float)

    mu = float(np.mean(group_mse_arr))
    mu = max(mu, eps)
    ratio = np.clip(group_mse_arr / mu, eps, None)
    n_groups = max(len(unique_groups), 1)

    if abs(alpha - 1.0) < 1e-12:
        loss = float(np.mean(ratio * np.log(ratio)))
        a = np.log(ratio) + 1.0
        da = float(np.mean(ratio * a))
        dloss_dmse = (a - da) / (n_groups * mu)
    elif abs(alpha) < 1e-12:
        loss = float(-np.mean(np.log(ratio)))
        dloss_dmse = (1.0 - 1.0 / ratio) / (n_groups * mu)
    else:
        moment = float(np.mean(ratio**alpha))
        loss = float((moment - 1.0) / (alpha * (alpha - 1.0)))
        dloss_dmse = (ratio ** (alpha - 1.0) - moment) / ((alpha - 1.0) * n_groups * mu)

    grad = np.zeros_like(pred)
    for idx, g in enumerate(unique_groups):
        mask = groups == g
        denom = pred.shape[0] * int(mask.sum())
        if denom == 0:
            continue
        grad[:, mask] = dloss_dmse[idx] * (2.0 * (pred[:, mask] - true[:, mask]) / float(denom))
    return loss, grad


def group_fairness_loss_and_grad(
    pred: np.ndarray,
    true: np.ndarray,
    groups: np.ndarray,
    fairness_type: str = "mad",
    smoothing: float = 1e-6,
    ge_alpha: float = 2.0,
) -> tuple[float, np.ndarray]:
    mode = str(fairness_type).strip().lower()
    if mode == "mad":
        return group_mse_mad_loss_and_grad(
            pred=pred,
            true=true,
            groups=groups,
            smoothing=smoothing,
        )
    if mode in {"generalized_entropy", "ge"}:
        return group_mse_generalized_entropy_loss_and_grad(
            pred=pred,
            true=true,
            groups=groups,
            alpha=ge_alpha,
            eps=max(float(smoothing), 1e-12),
        )
    raise ValueError(f"Unknown fairness_type: {fairness_type}")
