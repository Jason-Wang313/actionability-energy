from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class ResidualResult:
    """Solution of the one-step actionability projection problem."""

    residual: float
    u: np.ndarray
    projected_delta: np.ndarray
    unconstrained: bool
    success: bool = True


def _weight_matrix(dim: int, W: Optional[np.ndarray]) -> np.ndarray:
    if W is None:
        return np.eye(dim)
    W = np.asarray(W, dtype=float)
    if W.ndim == 1:
        return np.diag(W)
    return W


def weighted_norm_sq(x: np.ndarray, W: Optional[np.ndarray] = None) -> float:
    x = np.asarray(x, dtype=float)
    Wm = _weight_matrix(x.shape[0], W)
    return float(x.T @ Wm @ x)


def ridge_projection(
    delta: np.ndarray,
    J: np.ndarray,
    W: Optional[np.ndarray] = None,
    lam: float = 1e-4,
) -> ResidualResult:
    """Closed-form projection of delta onto Im(J) with ridge action penalty."""

    delta = np.asarray(delta, dtype=float).reshape(-1)
    J = np.asarray(J, dtype=float)
    Wm = _weight_matrix(delta.shape[0], W)
    action_dim = J.shape[1]
    A = J.T @ Wm @ J + lam * np.eye(action_dim)
    b = J.T @ Wm @ delta
    try:
        u = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        u = np.linalg.lstsq(A, b, rcond=None)[0]
    projected = J @ u
    residual = weighted_norm_sq(delta - projected, Wm) + lam * float(u.T @ u)
    return ResidualResult(float(residual), u, projected, unconstrained=True)


def box_projection(
    delta: np.ndarray,
    J: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    W: Optional[np.ndarray] = None,
    lam: float = 1e-4,
) -> ResidualResult:
    """Constrained actionability projection for box-bounded controls."""

    delta = np.asarray(delta, dtype=float).reshape(-1)
    J = np.asarray(J, dtype=float)
    low, high = (np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float))
    Wm = _weight_matrix(delta.shape[0], W)
    ridge = ridge_projection(delta, J, Wm, lam)
    if np.all(ridge.u >= low - 1e-10) and np.all(ridge.u <= high + 1e-10):
        return ResidualResult(ridge.residual, ridge.u, ridge.projected_delta, unconstrained=False)

    def objective(u: np.ndarray) -> float:
        err = delta - J @ u
        return weighted_norm_sq(err, Wm) + lam * float(u.T @ u)

    # Exact active-set enumeration is much faster than a generic optimizer for
    # the one- and two-dimensional action spaces used in the benchmark.
    action_dim = J.shape[1]
    if action_dim <= 4:
        A = J.T @ Wm @ J + lam * np.eye(action_dim)
        b = J.T @ Wm @ delta
        best_u = np.clip(ridge.u, low, high)
        best_val = objective(best_u)
        states = [-1, 0, 1]  # low, free, high

        def recurse(i: int, assignment: list[int]) -> None:
            nonlocal best_u, best_val
            if i == action_dim:
                assignment_arr = np.asarray(assignment)
                fixed = assignment_arr != 0
                free = ~fixed
                u = np.zeros(action_dim)
                u[assignment_arr == -1] = low[assignment_arr == -1]
                u[assignment_arr == 1] = high[assignment_arr == 1]
                if np.any(free):
                    Aff = A[np.ix_(free, free)]
                    rhs = b[free] - A[np.ix_(free, fixed)] @ u[fixed]
                    try:
                        u[free] = np.linalg.solve(Aff, rhs)
                    except np.linalg.LinAlgError:
                        u[free] = np.linalg.lstsq(Aff, rhs, rcond=None)[0]
                if np.all(u >= low - 1e-9) and np.all(u <= high + 1e-9):
                    val = objective(u)
                    if val < best_val:
                        best_val = val
                        best_u = u.copy()
                return
            for s in states:
                assignment.append(s)
                recurse(i + 1, assignment)
                assignment.pop()

        recurse(0, [])
        projected = J @ best_u
        return ResidualResult(float(best_val), best_u, projected, unconstrained=False)

    init = np.clip(ridge.u, low, high)
    res = minimize(
        objective,
        init,
        method="L-BFGS-B",
        bounds=list(zip(low, high)),
        options={"maxiter": 100, "ftol": 1e-10},
    )
    u = np.asarray(res.x, dtype=float)
    projected = J @ u
    return ResidualResult(float(objective(u)), u, projected, unconstrained=False, success=bool(res.success))


def solve_actionability(
    delta: np.ndarray,
    J: np.ndarray,
    W: Optional[np.ndarray] = None,
    lam: float = 1e-4,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> ResidualResult:
    """Solve the local actionability residual problem."""

    if bounds is None:
        return ridge_projection(delta, J, W, lam)
    return box_projection(delta, J, bounds, W, lam)


def trajectory_residuals(
    trajectory: np.ndarray,
    jacobian_fn,
    W: Optional[np.ndarray] = None,
    lam: float = 1e-4,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return residuals, recovered controls, and projected deltas for a trajectory."""

    traj = np.asarray(trajectory, dtype=float)
    residuals = []
    controls = []
    projected = []
    for t in range(len(traj) - 1):
        J = jacobian_fn(traj[t])
        out = solve_actionability(traj[t + 1] - traj[t], J, W=W, lam=lam, bounds=bounds)
        residuals.append(out.residual)
        controls.append(out.u)
        projected.append(out.projected_delta)
    return np.asarray(residuals), np.asarray(controls), np.asarray(projected)
