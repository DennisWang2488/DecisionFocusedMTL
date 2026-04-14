"""Prediction/fairness loss utilities shared by task implementations."""

import numpy as np


def softplus_with_grad(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = np.maximum(z, 0.0)
    exp_term = np.exp(-np.abs(z))
    value = positive + np.log1p(exp_term)
    sigmoid = np.empty_like(value)
    pos_mask = z >= 0.0
    sigmoid[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
    exp_z = np.exp(z[~pos_mask])
    sigmoid[~pos_mask] = exp_z / (1.0 + exp_z)
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


def group_mse_gap_loss_and_grad(
    pred: np.ndarray,
    true: np.ndarray,
    groups: np.ndarray,
    smoothing: float = 1e-6,
) -> tuple[float, np.ndarray]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return 0.0, np.zeros_like(pred)

    errors = (pred - true) ** 2
    if len(unique_groups) == 2:
        g0, g1 = unique_groups[0], unique_groups[1]
        m0, m1 = groups == g0, groups == g1
        n0, n1 = int(m0.sum()), int(m1.sum())
        if n0 == 0 or n1 == 0:
            return 0.0, np.zeros_like(pred)
        mse0 = float(errors[:, m0].mean())
        mse1 = float(errors[:, m1].mean())
        gap = mse0 - mse1
        loss = float(np.sqrt(gap * gap + smoothing))
        coeff = gap / max(loss, 1e-12)
        grad = np.zeros_like(pred)
        grad[:, m0] = coeff * 2.0 * (pred[:, m0] - true[:, m0]) / float(pred.shape[0] * n0)
        grad[:, m1] = -coeff * 2.0 * (pred[:, m1] - true[:, m1]) / float(pred.shape[0] * n1)
        return loss, grad

    group_mse = []
    for g in unique_groups:
        mask = groups == g
        group_mse.append(errors[:, mask].mean())
    group_mse_arr = np.asarray(group_mse, dtype=float)
    mean_mse = float(group_mse_arr.mean())
    gap = group_mse_arr - mean_mse
    smooth_abs = np.sqrt(gap * gap + smoothing)
    loss = float(smooth_abs.mean())

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


def group_pred_mean_dp_loss_and_grad(
    pred: np.ndarray,
    true: np.ndarray,  # noqa: ARG001 - kept for API symmetry with other fairness losses
    groups: np.ndarray,
    smoothing: float = 1e-6,
) -> tuple[float, np.ndarray]:
    """Demographic parity on predictions: MAD of per-group mean predictions.

    Loss
    ----
    L = mean_g sqrt( (mu_g - mu_bar)^2 + smoothing )
        where mu_g    = mean(pred | group=g)        (per-group mean prediction)
              mu_bar  = mean_g(mu_g)                (mean of per-group means)

    Unlike ``group_mse_*`` losses (which equalise per-group MSE i.e. accuracy
    parity), this targets demographic parity: it penalises differences in the
    average predicted benefit across groups regardless of label values.
    """
    unique_groups = np.unique(groups)
    K = len(unique_groups)
    if K < 2:
        return 0.0, np.zeros_like(pred)

    group_means = np.zeros(K, dtype=np.float64)
    group_sizes = np.zeros(K, dtype=np.float64)
    for idx, g in enumerate(unique_groups):
        mask = groups == g
        n_g = float(mask.sum())
        group_sizes[idx] = n_g
        group_means[idx] = float(pred[:, mask].mean()) if n_g > 0 else 0.0

    mean_of_means = float(group_means.mean())
    dev = group_means - mean_of_means                          # (K,)
    smooth_abs = np.sqrt(dev * dev + smoothing)                # (K,)
    loss = float(smooth_abs.mean())

    # d(loss)/d(mu_g) via chain rule (identical to MAD form):
    #   d(loss)/d(mu_g) = (1/K) * (dev_g / smooth_abs_g - mean_h(dev_h / smooth_abs_h))
    dphi = dev / smooth_abs                                    # (K,)
    dloss_dmu = (dphi - dphi.mean()) / float(K)                # (K,)

    # d(mu_g)/d(pred[b, i]) = 1 / (B * n_g)  if i in g else 0
    grad = np.zeros_like(pred)
    B = float(pred.shape[0])
    for idx, g in enumerate(unique_groups):
        mask = groups == g
        n_g = group_sizes[idx]
        if n_g == 0:
            continue
        grad[:, mask] = dloss_dmu[idx] / (B * n_g)
    return loss, grad


def group_mse_atkinson_loss_and_grad(
    pred: np.ndarray,
    true: np.ndarray,
    groups: np.ndarray,
    smoothing: float = 1e-6,
    epsilon: float = 0.5,
) -> tuple[float, np.ndarray]:
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    if n_groups < 2:
        return 0.0, np.zeros_like(pred)

    errors = (pred - true) ** 2
    group_mse = np.zeros(n_groups, dtype=np.float64)
    group_sizes = np.zeros(n_groups, dtype=np.float64)
    group_masks = []
    for idx, g in enumerate(unique_groups):
        mask = groups == g
        group_masks.append(mask)
        n_g = float(mask.sum())
        group_sizes[idx] = n_g
        group_mse[idx] = max(float(errors[:, mask].mean()), smoothing) if n_g > 0 else smoothing

    grand_mean = float(np.mean(group_mse))
    grand_mean_safe = max(grand_mean, 1e-12)

    if abs(epsilon - 1.0) < 1e-12:
        log_mse = np.log(group_mse)
        geomean = float(np.exp(np.mean(log_mse)))
        loss = max(1.0 - geomean / grand_mean_safe, 0.0)

        dloss_dmse = np.zeros(n_groups, dtype=np.float64)
        for idx in range(n_groups):
            d_geomean = geomean / (float(n_groups) * group_mse[idx])
            d_grand = 1.0 / float(n_groups)
            dloss_dmse[idx] = -(d_geomean * grand_mean_safe - geomean * d_grand) / (grand_mean_safe ** 2)
    else:
        one_minus_eps = 1.0 - epsilon
        powered = np.power(group_mse, one_minus_eps)
        mean_powered = float(np.mean(powered))
        mean_powered_safe = max(mean_powered, 1e-12)
        ede = mean_powered_safe ** (1.0 / one_minus_eps)
        loss = max(1.0 - ede / grand_mean_safe, 0.0)

        dloss_dmse = np.zeros(n_groups, dtype=np.float64)
        for idx in range(n_groups):
            d_ede = (1.0 / float(n_groups)) * ede / mean_powered_safe * (group_mse[idx] ** (-epsilon))
            d_grand = 1.0 / float(n_groups)
            dloss_dmse[idx] = -(d_ede * grand_mean_safe - ede * d_grand) / (grand_mean_safe ** 2)

    grad = np.zeros_like(pred)
    for idx, mask in enumerate(group_masks):
        denom = pred.shape[0] * group_sizes[idx]
        if denom == 0:
            continue
        grad[:, mask] = dloss_dmse[idx] * 2.0 * (pred[:, mask] - true[:, mask]) / denom
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
    if mode == "gap":
        return group_mse_gap_loss_and_grad(
            pred=pred,
            true=true,
            groups=groups,
            smoothing=smoothing,
        )
    if mode == "mad":
        return group_mse_mad_loss_and_grad(
            pred=pred,
            true=true,
            groups=groups,
            smoothing=smoothing,
        )
    if mode == "atkinson":
        return group_mse_atkinson_loss_and_grad(
            pred=pred,
            true=true,
            groups=groups,
            smoothing=smoothing,
        )
    if mode in {"demographic_parity", "dp"}:
        return group_pred_mean_dp_loss_and_grad(
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
