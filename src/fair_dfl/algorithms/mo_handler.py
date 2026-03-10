"""Multi-objective gradient handler implementations for Decision-Focused Learning.

Provides abstract base class and eight concrete strategies:
  - WeightedSumHandler: normalized weighted sum of objective gradients
  - PCGradHandler: projecting away conflicting gradient components (Yu et al. 2020)
  - MGDAHandler: minimum-norm point in the convex hull (Sener & Koltun 2018)
  - CAGradHandler: conflict-averse gradient descent (Liu et al. ICLR 2021)
  - PLGHandler3Obj: prediction-loss-guided 3-objective extension for DFL
  - FAMOHandler: fast adaptive multitask optimization (Liu et al. NeurIPS 2023)
  - NestedPLGFairPrimaryHandler: nested PLG with fairness-primary step (PLG-FP)
  - NestedPLGPredPrimaryHandler: nested PLG with prediction-primary step (PLG-PP)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from ..metrics import cosine, l2_norm, project_orthogonal


class MultiObjectiveGradientHandler(ABC):
    """Abstract base for multi-objective gradient combination strategies."""

    @abstractmethod
    def compute_direction(
        self,
        grads: Dict[str, np.ndarray],
        losses: Dict[str, float],
        step: int,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        """Return combined gradient direction. Same shape as each input grad (1D, flattened params)."""

    @abstractmethod
    def extra_logs(self) -> Dict[str, float]:
        """Diagnostics from last call.

        Must include:
        - mo_grad_norm_{name}: per-objective gradient norms
        - mo_cos_{name1}_{name2}: pairwise cosine similarities
        - direction_alignment_with_dec_regret: cosine(output, g_dec)
        - stationarity_proxy: min_{lambda in Delta} ||sum lambda_i g_i||
        """

    # ------------------------------------------------------------------
    # Shared diagnostic helpers
    # ------------------------------------------------------------------

    def _compute_common_diagnostics(
        self,
        grads: Dict[str, np.ndarray],
        direction: np.ndarray,
    ) -> Dict[str, float]:
        """Compute standard diagnostics shared across all handlers."""
        diag: Dict[str, float] = {}
        names = sorted(grads.keys())

        # Per-objective gradient norms.
        for name in names:
            diag[f"mo_grad_norm_{name}"] = l2_norm(grads[name])

        # Pairwise cosine similarities.
        for n1, n2 in combinations(names, 2):
            diag[f"mo_cos_{n1}_{n2}"] = cosine(grads[n1], grads[n2])

        # Alignment of output direction with decision regret gradient.
        if "decision_regret" in grads:
            diag["direction_alignment_with_dec_regret"] = cosine(direction, grads["decision_regret"])
        else:
            diag["direction_alignment_with_dec_regret"] = 0.0

        # Stationarity proxy: min over simplex of ||sum lambda_i g_i||.
        diag["stationarity_proxy"] = _stationarity_proxy(list(grads.values()))

        return diag


# ======================================================================
# Utility: stationarity proxy via grid search over the simplex
# ======================================================================

def _simplex_grid(m: int, n_per_dim: int = 200) -> np.ndarray:
    """Generate ~n_per_dim uniformly spaced points on the (m-1)-simplex."""
    if m == 1:
        return np.array([[1.0]])
    if m == 2:
        t = np.linspace(0.0, 1.0, n_per_dim)
        return np.column_stack([t, 1.0 - t])
    if m == 3:
        pts: List[np.ndarray] = []
        n = max(int(np.sqrt(n_per_dim)), 10)
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                pts.append(np.array([i / n, j / n, k / n]))
        return np.array(pts)
    # General case: random sampling.
    rng = np.random.default_rng(0)
    raw = rng.exponential(size=(n_per_dim, m))
    return raw / raw.sum(axis=1, keepdims=True)


def _stationarity_proxy(grad_list: List[np.ndarray]) -> float:
    """min_{lambda in simplex} ||sum lambda_i g_i||, grid search."""
    m = len(grad_list)
    if m == 0:
        return 0.0
    G = np.stack([g.ravel() for g in grad_list], axis=0)  # (m, d)
    lambdas = _simplex_grid(m, n_per_dim=200)  # (K, m)
    # Vectorized: combined = lambdas @ G  -> (K, d)
    combined = lambdas @ G
    norms = np.sqrt(np.sum(combined * combined, axis=1) + 1e-12)
    return float(np.min(norms))


# ======================================================================
# 1. WeightedSumHandler
# ======================================================================

class WeightedSumHandler(MultiObjectiveGradientHandler):
    """Normalized weighted sum of per-objective gradients."""

    def __init__(self, weights: Dict[str, float]) -> None:
        self._weights = dict(weights) if weights else {}
        self._last_diag: Dict[str, float] = {}

    def compute_direction(
        self,
        grads: Dict[str, np.ndarray],
        losses: Dict[str, float],
        step: int,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        names = sorted(grads.keys())
        # Resolve weights: default to equal if not specified.
        raw_w = {n: self._weights.get(n, 1.0) for n in names}
        total_w = sum(raw_w.values())
        if total_w < 1e-12:
            total_w = 1.0
        norm_w = {n: raw_w[n] / total_w for n in names}

        dim = next(iter(grads.values())).shape[0]
        direction = np.zeros(dim, dtype=float)
        for n in names:
            direction += norm_w[n] * grads[n].ravel()

        self._last_diag = self._compute_common_diagnostics(grads, direction)
        for n in names:
            self._last_diag[f"mo_ws_weight_{n}"] = norm_w[n]
        return direction

    def extra_logs(self) -> Dict[str, float]:
        return dict(self._last_diag)


# ======================================================================
# 2. PCGradHandler
# ======================================================================

class PCGradHandler(MultiObjectiveGradientHandler):
    """PCGrad: project away conflicting components (Yu et al. 2020)."""

    def __init__(self) -> None:
        self._last_diag: Dict[str, float] = {}

    def compute_direction(
        self,
        grads: Dict[str, np.ndarray],
        losses: Dict[str, float],
        step: int,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        names = sorted(grads.keys())
        m = len(names)
        flat_grads = {n: grads[n].ravel().copy() for n in names}

        n_projections = 0
        n_pairs = 0

        # For each g_i, project away conflicting components from all g_j.
        # Per Yu et al. 2020 Algorithm 1: conflict check uses the ORIGINAL g_i
        # (not the running-projected state), but the projection is applied to
        # the running state.  This ensures the decision to project is not
        # influenced by earlier projections within the same i-loop.
        projected = {}
        for i, ni in enumerate(names):
            gi_orig = flat_grads[ni]          # original gradient, read-only
            gi_running = flat_grads[ni].copy()  # running state for sequential projection
            for j, nj in enumerate(names):
                if i == j:
                    continue
                n_pairs += 1
                gj = flat_grads[nj]
                # Conflict check on ORIGINAL g_i (not gi_running).
                cos_ij = cosine(gi_orig, gj)
                if cos_ij < 0.0:
                    # Project the running state onto the normal plane of g_j.
                    dot_ij = float(np.dot(gi_running, gj))
                    dot_jj = float(np.dot(gj, gj)) + 1e-12
                    gi_running = gi_running - (dot_ij / dot_jj) * gj
                    n_projections += 1
            projected[ni] = gi_running

        # Sum projected gradients.
        dim = next(iter(grads.values())).shape[0]
        direction = np.zeros(dim, dtype=float)
        for n in names:
            direction += projected[n]

        conflict_fraction = float(n_projections) / max(n_pairs, 1)

        self._last_diag = self._compute_common_diagnostics(grads, direction)
        self._last_diag["mo_pcgrad_n_projections"] = float(n_projections)
        self._last_diag["mo_pcgrad_conflict_fraction"] = conflict_fraction
        return direction

    def extra_logs(self) -> Dict[str, float]:
        return dict(self._last_diag)


# ======================================================================
# 3. MGDAHandler
# ======================================================================

class MGDAHandler(MultiObjectiveGradientHandler):
    """MGDA: minimum-norm point in convex hull via SLSQP."""

    def __init__(self) -> None:
        self._last_diag: Dict[str, float] = {}

    def compute_direction(
        self,
        grads: Dict[str, np.ndarray],
        losses: Dict[str, float],
        step: int,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        names = sorted(grads.keys())
        m = len(names)
        G = np.stack([grads[n].ravel() for n in names], axis=0)  # (m, d)

        # Gram matrix for the QP: min_{lambda} lambda^T M lambda
        # where M_ij = g_i . g_j
        M = G @ G.T  # (m, m)

        lambdas = _solve_mgda_qp(M, m)

        direction = (lambdas @ G).ravel()

        self._last_diag = self._compute_common_diagnostics(grads, direction)
        for i, n in enumerate(names):
            self._last_diag[f"mo_mgda_lambda_{n}"] = float(lambdas[i])
        self._last_diag["mo_mgda_min_norm"] = l2_norm(direction)
        return direction

    def extra_logs(self) -> Dict[str, float]:
        return dict(self._last_diag)


def _solve_mgda_qp(M: np.ndarray, m: int) -> np.ndarray:
    """Solve min_{lambda in simplex} lambda^T M lambda via SLSQP.

    Falls back to equal weights if optimization fails.
    """
    def objective(lam: np.ndarray) -> float:
        return float(lam @ M @ lam)

    def grad_objective(lam: np.ndarray) -> np.ndarray:
        return 2.0 * (M @ lam)

    x0 = np.ones(m) / m
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}
    bounds = [(0.0, 1.0)] * m

    try:
        result = minimize(
            objective,
            x0,
            jac=grad_objective,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-12},
        )
        if result.success:
            lam = np.clip(result.x, 0.0, 1.0)
            lam /= lam.sum() + 1e-12
            return lam
    except Exception:
        pass

    # Fallback: equal weights.
    return np.ones(m) / m


# ======================================================================
# 4. CAGradHandler
# ======================================================================

class CAGradHandler(MultiObjectiveGradientHandler):
    """CAGrad: Conflict-Averse Gradient descent (Liu et al. ICLR 2021).

    Given per-objective gradients g_1, ..., g_m, computes the mean gradient
    g_0 = (1/m) sum_i g_i, then finds a weight vector w on the simplex that
    solves:

        min_{w in Delta_m}  w^T (G g_0) + c * ||g_0|| * sqrt(w^T M w)

    where G is the (m x d) matrix of per-objective gradients stacked as rows,
    M = G G^T is the (m x m) Gram matrix, and c >= 0 is the conflict-aversion
    coefficient.

    The final update direction is: g_0 + sum_i w_i * g_i  (i.e. g_0 + G^T w).

    When c = 0 the objective collapses to the mean gradient; as c grows the
    solution approaches MGDA-like behaviour.
    """

    def __init__(self, c: float = 0.5) -> None:
        if c < 0.0:
            raise ValueError(f"Conflict-aversion parameter c must be >= 0, got {c}")
        self._c = c
        self._last_diag: Dict[str, float] = {}

    def compute_direction(
        self,
        grads: Dict[str, np.ndarray],
        losses: Dict[str, float],
        step: int,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        names = sorted(grads.keys())
        m = len(names)
        G = np.stack([grads[n].ravel() for n in names], axis=0)  # (m, d)

        # Mean gradient g_0 = (1/m) sum_i g_i.
        g0 = G.mean(axis=0)  # (d,)
        g0_norm = float(np.sqrt(np.dot(g0, g0) + 1e-12))

        # Gram matrix M = G G^T  (m x m).
        M = G @ G.T

        # Vector of dot products b_i = g_i . g_0  (m,).
        b = G @ g0

        # Solve the CAGrad sub-problem for simplex weights.
        w = _solve_cagrad_qp(M, b, g0_norm, self._c, m)

        # g_w = sum_i w_i g_i = G^T w  (d,).
        g_w = G.T @ w  # (d,)

        # Final direction: g_0 + g_w.
        direction = g0 + g_w

        # ---- diagnostics ----
        self._last_diag = self._compute_common_diagnostics(grads, direction)
        self._last_diag["mo_cagrad_c"] = self._c
        self._last_diag["mo_cagrad_g0_norm"] = g0_norm
        self._last_diag["mo_cagrad_gw_norm"] = float(l2_norm(g_w))
        self._last_diag["mo_cagrad_direction_norm"] = float(l2_norm(direction))
        for i, n in enumerate(names):
            self._last_diag[f"mo_cagrad_w_{n}"] = float(w[i])
        return direction

    def extra_logs(self) -> Dict[str, float]:
        return dict(self._last_diag)


def _solve_cagrad_qp(
    M: np.ndarray,
    b: np.ndarray,
    g0_norm: float,
    c: float,
    m: int,
) -> np.ndarray:
    """Solve the CAGrad sub-problem on the simplex via SLSQP.

    Minimises:
        f(w) = w^T b  +  c * g0_norm * sqrt(w^T M w)

    subject to  w in Delta_m  (simplex: w >= 0, sum w_i = 1).

    Parameters
    ----------
    M : (m, m) Gram matrix  G G^T.
    b : (m,)  vector of dot products  g_i . g_0.
    g0_norm : ||g_0||.
    c : conflict-aversion coefficient.
    m : number of objectives.

    Returns
    -------
    w : (m,) optimal simplex weights.  Falls back to equal weights on failure.
    """

    def objective(w: np.ndarray) -> float:
        quad = float(w @ M @ w)
        return float(w @ b) + c * g0_norm * np.sqrt(max(quad, 1e-24))

    def grad_objective(w: np.ndarray) -> np.ndarray:
        Mw = M @ w  # (m,)
        quad = float(w @ Mw)
        sqrt_quad = np.sqrt(max(quad, 1e-24))
        # d/dw [w^T b] = b
        # d/dw [sqrt(w^T M w)] = M w / sqrt(w^T M w)
        return b + c * g0_norm * (Mw / sqrt_quad)

    x0 = np.ones(m) / m
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}
    bounds = [(0.0, 1.0)] * m

    try:
        result = minimize(
            objective,
            x0,
            jac=grad_objective,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-12},
        )
        if result.success:
            w = np.clip(result.x, 0.0, 1.0)
            w /= w.sum() + 1e-12
            return w
    except Exception:
        pass

    # Fallback: equal weights.
    return np.ones(m) / m


# ======================================================================
# 5. PLGHandler3Obj
# ======================================================================

class PLGHandler3Obj(MultiObjectiveGradientHandler):
    """Prediction-Loss-Guided 3-objective handler for DFL.

    Step 1: MGDA on primary objectives (decision_regret, pred_fairness) -> d_primary
    Step 2: Add orthogonal guiding component from pred_loss gradient.
    Step 3: Decay kappa over training steps.
    Falls back to full 3-obj MGDA if ||d_primary|| < epsilon.
    """

    def __init__(
        self,
        kappa_0: float = 1.0,
        kappa_decay: float = 0.01,
        primary_objectives: Tuple[str, ...] = ("decision_regret", "pred_fairness"),
        guiding_objectives: Tuple[str, ...] = ("pred_loss",),
    ) -> None:
        self._kappa_0 = kappa_0
        self._kappa_decay = kappa_decay
        self._primary = primary_objectives
        self._guiding = guiding_objectives
        self._last_diag: Dict[str, float] = {}

    def compute_direction(
        self,
        grads: Dict[str, np.ndarray],
        losses: Dict[str, float],
        step: int,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        names = sorted(grads.keys())
        kappa_t = self._kappa_0 / (1.0 + self._kappa_decay * step)
        fallback_used = 0.0

        # Sentinel values — overwritten on the relevant code paths.
        dim = next(iter(grads.values())).shape[0]
        d_primary: np.ndarray = np.zeros(dim, dtype=float)
        d_primary_norm: float = 0.0
        guiding_component_norm: float = 0.0

        # Identify primary and guiding gradients.
        primary_names = [n for n in self._primary if n in grads]
        guiding_names = [n for n in self._guiding if n in grads]

        if len(primary_names) < 1:
            # No primary objectives found — fall back to equal-weight sum.
            direction = np.zeros(dim, dtype=float)
            for n in names:
                direction += grads[n].ravel()
            direction /= max(len(names), 1)
            fallback_used = 1.0
        else:
            # Step 1: MGDA on primary objectives.
            m_p = len(primary_names)
            G_primary = np.stack([grads[n].ravel() for n in primary_names], axis=0)
            M_primary = G_primary @ G_primary.T
            lambdas_primary = _solve_mgda_qp(M_primary, m_p)
            d_primary = (lambdas_primary @ G_primary).ravel()
            d_primary_norm = l2_norm(d_primary)

            if d_primary_norm < epsilon:
                # Fallback: full m-objective MGDA over all gradients.
                m_all = len(names)
                G_all = np.stack([grads[n].ravel() for n in names], axis=0)
                M_all = G_all @ G_all.T
                lambdas_all = _solve_mgda_qp(M_all, m_all)
                direction = (lambdas_all @ G_all).ravel()
                fallback_used = 1.0
            else:
                # Step 2: Add orthogonal guiding component.
                if guiding_names:
                    g_guide = np.zeros(dim, dtype=float)
                    for gn in guiding_names:
                        g_guide += grads[gn].ravel()
                    g_guide /= max(len(guiding_names), 1)

                    # Orthogonal component: g_guide - proj_{d_primary}(g_guide)
                    g_orth = project_orthogonal(g_guide, d_primary)
                    guiding_component_norm = l2_norm(kappa_t * g_orth)
                    direction = d_primary + kappa_t * g_orth
                else:
                    direction = d_primary

        self._last_diag = self._compute_common_diagnostics(grads, direction)
        self._last_diag["mo_plg_kappa_t"] = kappa_t
        self._last_diag["mo_plg_primary_norm"] = d_primary_norm
        self._last_diag["mo_plg_guiding_component_norm"] = guiding_component_norm
        self._last_diag["mo_plg_fallback_used"] = fallback_used
        return direction

    def extra_logs(self) -> Dict[str, float]:
        return dict(self._last_diag)


# ======================================================================
# 6. FAMOHandler (Liu et al., NeurIPS 2023)
# ======================================================================

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    x_shifted = x - np.max(x)
    e = np.exp(x_shifted)
    return e / (e.sum() + 1e-12)


def _softmax_jacobian(z: np.ndarray) -> np.ndarray:
    """Jacobian of softmax: J_ij = z_i * (delta_ij - z_j).

    Parameters
    ----------
    z : (k,) softmax output

    Returns
    -------
    J : (k, k) Jacobian matrix
    """
    return np.diag(z) - np.outer(z, z)


class FAMOHandler(MultiObjectiveGradientHandler):
    """FAMO: Fast Adaptive Multitask Optimization (Liu et al., NeurIPS 2023).

    Maintains learnable logit weights xi in R^k whose softmax z determines
    the per-objective weighting.  After each parameter update the caller
    invokes ``update_weights(new_losses)`` so that FAMO can adjust the logits
    based on per-objective loss improvements.

    Because the DFL pipeline provides per-objective gradients rather than a
    single autograd-backed combined loss, the update rule is adapted:

    * ``compute_direction`` uses z = softmax(xi) to form a weighted sum of
      the individual objective gradients and caches the current losses.
    * ``update_weights`` computes delta = log(old_losses) - log(new_losses),
      back-propagates through the softmax analytically via its Jacobian, and
      updates the logits with an optional weight-decay term (gamma).

    Parameters
    ----------
    n_tasks : int
        Number of objectives (must match the number of keys in *grads*).
    gamma : float
        L2 regularization / weight-decay coefficient on the logits.
    w_lr : float
        Learning rate for the logit update.
    min_loss : float
        Floor applied to loss values before taking the log (prevents -inf).
    """

    # Canonical objective ordering — must match the sorted key order used
    # everywhere else in this file.
    _OBJ_NAMES: Tuple[str, ...] = ("decision_regret", "pred_fairness", "pred_loss")

    def __init__(
        self,
        n_tasks: int = 3,
        gamma: float = 1e-3,
        w_lr: float = 0.025,
        min_loss: float = 1e-8,
    ) -> None:
        self._n_tasks = n_tasks
        self._gamma = gamma
        self._w_lr = w_lr
        self._min_loss = min_loss

        # Logits initialised to zero -> uniform softmax weights.
        self._xi: np.ndarray = np.zeros(n_tasks, dtype=float)

        # Cache: losses from the most recent ``compute_direction`` call.
        self._prev_losses: Optional[np.ndarray] = None

        self._last_diag: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def compute_direction(
        self,
        grads: Dict[str, np.ndarray],
        losses: Dict[str, float],
        step: int,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        """Weighted gradient direction using FAMO softmax weights.

        Also stores the current losses so that ``update_weights`` can later
        compute the per-objective improvement.
        """
        names = sorted(grads.keys())
        m = len(names)

        # Resize logits if the number of tasks changed (defensive).
        if m != self._n_tasks:
            self._n_tasks = m
            self._xi = np.zeros(m, dtype=float)

        # Softmax weights.
        z = _softmax(self._xi)

        # Weighted gradient sum: direction = sum_i z_i * g_i.
        dim = next(iter(grads.values())).shape[0]
        direction = np.zeros(dim, dtype=float)
        for i, n in enumerate(names):
            direction += z[i] * grads[n].ravel()

        # Cache losses (ordered by sorted name).
        loss_arr = np.array(
            [max(losses.get(n, 0.0), self._min_loss) for n in names]
        )
        self._prev_losses = loss_arr

        # --- Diagnostics ---
        self._last_diag = self._compute_common_diagnostics(grads, direction)
        for i, n in enumerate(names):
            self._last_diag[f"mo_famo_z_{n}"] = float(z[i])
            self._last_diag[f"mo_famo_xi_{n}"] = float(self._xi[i])
        self._last_diag["mo_famo_z_entropy"] = float(
            -np.sum(z * np.log(z + 1e-12))
        )
        self._last_diag["mo_famo_xi_norm"] = float(np.linalg.norm(self._xi))

        return direction

    # ------------------------------------------------------------------
    # Weight update (called AFTER the parameter step)
    # ------------------------------------------------------------------

    def update_weights(self, new_losses: Dict[str, float]) -> None:
        """Update logits xi using per-objective log-loss improvements.

        Should be called after the model parameters have been updated and the
        losses have been re-evaluated on the same mini-batch (or a fresh one).

        The update rule (adapted from FAMO):

            delta_i = log(old_loss_i) - log(new_loss_i)
            grad_xi = -J_softmax^T @ delta + gamma * xi
            xi <- xi - w_lr * grad_xi

        where J_softmax is the Jacobian of softmax evaluated at the current xi.
        Positive delta means loss decreased; the update encourages the logits
        to up-weight objectives that improved more.
        """
        if self._prev_losses is None:
            return  # First call — nothing to compare against.

        names = sorted(new_losses.keys())
        m = len(names)

        # Ensure dimensionality matches.
        if m != self._n_tasks:
            return

        new_arr = np.array(
            [max(new_losses.get(n, 0.0), self._min_loss) for n in names]
        )
        old_arr = self._prev_losses

        # Per-objective log-loss improvement (positive means loss decreased).
        delta = np.log(old_arr) - np.log(new_arr)

        # Current softmax weights & Jacobian.
        z = _softmax(self._xi)
        J = _softmax_jacobian(z)  # (m, m)

        # Gradient w.r.t. logits: we want to *increase* logits for objectives
        # that improved more, so the update direction for ascent on the
        # improvement metric is J^T @ delta.  We also add weight-decay on xi.
        # Combined descent direction: grad_xi = -J^T delta + gamma * xi
        grad_xi = -J.T @ delta + self._gamma * self._xi
        self._xi -= self._w_lr * grad_xi

        # Record update diagnostics.
        for i, n in enumerate(names):
            self._last_diag[f"mo_famo_delta_{n}"] = float(delta[i])
            self._last_diag[f"mo_famo_new_loss_{n}"] = float(new_arr[i])
        self._last_diag["mo_famo_delta_norm"] = float(np.linalg.norm(delta))
        self._last_diag["mo_famo_grad_xi_norm"] = float(
            np.linalg.norm(grad_xi)
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def extra_logs(self) -> Dict[str, float]:
        return dict(self._last_diag)


# ======================================================================
# 7. NestedPLGFairPrimaryHandler  (PLG-FP)
# ======================================================================

class NestedPLGFairPrimaryHandler(MultiObjectiveGradientHandler):
    """Nested PLG with fairness-primary first step (PLG-FP).

    Two-step prediction-loss-guided direction:

    Step 1 (pred-space): d_pred_space = g_fair + kappa1 * orth(g_pred, g_fair)
        Fairness is primary; prediction guides via its orthogonal component.

    Step 2 (final):      d_final = g_dec + kappa2 * orth(d_pred_space, g_dec)
        Decision regret is primary; the pred-space direction guides.

    Both kappa values decay over training:
        kappa_i(t) = kappa_i_0 / (1 + kappa_decay * t)
    """

    def __init__(
        self,
        kappa1_0: float = 1.0,
        kappa2_0: float = 1.0,
        kappa_decay: float = 0.01,
    ) -> None:
        self._kappa1_0 = kappa1_0
        self._kappa2_0 = kappa2_0
        self._kappa_decay = kappa_decay
        self._last_diag: Dict[str, float] = {}

    def compute_direction(
        self,
        grads: Dict[str, np.ndarray],
        losses: Dict[str, float],
        step: int,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        g_dec = grads["decision_regret"].ravel()
        g_pred = grads["pred_loss"].ravel()
        g_fair = grads["pred_fairness"].ravel()

        kappa1_t = self._kappa1_0 / (1.0 + self._kappa_decay * step)
        kappa2_t = self._kappa2_0 / (1.0 + self._kappa_decay * step)

        # Step 1: fairness primary, prediction guides
        orth_step1 = project_orthogonal(g_pred, g_fair)
        d_pred_space = g_fair + kappa1_t * orth_step1

        # Step 2: decision primary, pred-space guides
        orth_step2 = project_orthogonal(d_pred_space, g_dec)
        d_final = g_dec + kappa2_t * orth_step2

        # Diagnostics
        self._last_diag = self._compute_common_diagnostics(grads, d_final)
        self._last_diag["kappa1_t"] = kappa1_t
        self._last_diag["kappa2_t"] = kappa2_t
        self._last_diag["d_pred_space_norm"] = l2_norm(d_pred_space)
        self._last_diag["d_final_norm"] = l2_norm(d_final)
        self._last_diag["orth_component_step1_norm"] = l2_norm(orth_step1)
        self._last_diag["orth_component_step2_norm"] = l2_norm(orth_step2)

        return d_final

    def extra_logs(self) -> Dict[str, float]:
        return dict(self._last_diag)


# ======================================================================
# 8. NestedPLGPredPrimaryHandler  (PLG-PP)
# ======================================================================

class NestedPLGPredPrimaryHandler(MultiObjectiveGradientHandler):
    """Nested PLG with prediction-primary first step (PLG-PP).

    Two-step prediction-loss-guided direction:

    Step 1 (pred-space): d_pred_space = g_pred + kappa1 * orth(g_fair, g_pred)
        Prediction is primary; fairness guides via its orthogonal component.

    Step 2 (final):      d_final = g_dec + kappa2 * orth(d_pred_space, g_dec)
        Decision regret is primary; the pred-space direction guides.

    Both kappa values decay over training:
        kappa_i(t) = kappa_i_0 / (1 + kappa_decay * t)
    """

    def __init__(
        self,
        kappa1_0: float = 1.0,
        kappa2_0: float = 1.0,
        kappa_decay: float = 0.01,
    ) -> None:
        self._kappa1_0 = kappa1_0
        self._kappa2_0 = kappa2_0
        self._kappa_decay = kappa_decay
        self._last_diag: Dict[str, float] = {}

    def compute_direction(
        self,
        grads: Dict[str, np.ndarray],
        losses: Dict[str, float],
        step: int,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        g_dec = grads["decision_regret"].ravel()
        g_pred = grads["pred_loss"].ravel()
        g_fair = grads["pred_fairness"].ravel()

        kappa1_t = self._kappa1_0 / (1.0 + self._kappa_decay * step)
        kappa2_t = self._kappa2_0 / (1.0 + self._kappa_decay * step)

        # Step 1: prediction primary, fairness guides
        orth_step1 = project_orthogonal(g_fair, g_pred)
        d_pred_space = g_pred + kappa1_t * orth_step1

        # Step 2: decision primary, pred-space guides
        orth_step2 = project_orthogonal(d_pred_space, g_dec)
        d_final = g_dec + kappa2_t * orth_step2

        # Diagnostics
        self._last_diag = self._compute_common_diagnostics(grads, d_final)
        self._last_diag["kappa1_t"] = kappa1_t
        self._last_diag["kappa2_t"] = kappa2_t
        self._last_diag["d_pred_space_norm"] = l2_norm(d_pred_space)
        self._last_diag["d_final_norm"] = l2_norm(d_final)
        self._last_diag["orth_component_step1_norm"] = l2_norm(orth_step1)
        self._last_diag["orth_component_step2_norm"] = l2_norm(orth_step2)

        return d_final

    def extra_logs(self) -> Dict[str, float]:
        return dict(self._last_diag)
