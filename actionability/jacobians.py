from __future__ import annotations

import numpy as np


def finite_difference_action_jacobian(step_fn, z: np.ndarray, action_dim: int, eps: float = 1e-4) -> np.ndarray:
    """Finite-difference d step(z, u) / d u around u=0."""

    z = np.asarray(z, dtype=float)
    base = step_fn(z, np.zeros(action_dim))
    cols = []
    for i in range(action_dim):
        u = np.zeros(action_dim)
        u[i] = eps
        cols.append((step_fn(z, u) - base) / eps)
    return np.stack(cols, axis=1)


class LocalLinearJacobian:
    """Small CPU local-regression model for action-conditioned deltas."""

    def __init__(self, ridge: float = 1e-4, k: int = 48, length_scale: float = 0.7):
        self.ridge = ridge
        self.k = k
        self.length_scale = length_scale
        self.z = None
        self.u = None
        self.delta = None

    def fit(self, z: np.ndarray, u: np.ndarray, delta: np.ndarray) -> "LocalLinearJacobian":
        self.z = np.asarray(z, dtype=float)
        self.u = np.asarray(u, dtype=float)
        self.delta = np.asarray(delta, dtype=float)
        return self

    def predict(self, z_query: np.ndarray) -> np.ndarray:
        if self.z is None:
            raise RuntimeError("LocalLinearJacobian.fit must be called before predict.")
        z_query = np.asarray(z_query, dtype=float)
        d = np.linalg.norm(self.z - z_query[None, :], axis=1)
        idx = np.argsort(d)[: min(self.k, len(d))]
        w = np.exp(-0.5 * (d[idx] / max(self.length_scale, 1e-8)) ** 2)
        U = self.u[idx]
        D = self.delta[idx]
        sw = np.sqrt(w + 1e-12)[:, None]
        Uw = U * sw
        Dw = D * sw
        A = Uw.T @ Uw + self.ridge * np.eye(U.shape[1])
        B = Uw.T @ Dw
        coef = np.linalg.solve(A, B)
        return coef.T


class GlobalLinearJacobian:
    """Single ridge linear action map delta ~= J u.

    This intentionally ignores state and is used as a weak learned-J baseline.
    """

    def __init__(self, ridge: float = 1e-4):
        self.ridge = ridge
        self.J = None

    def fit(self, z: np.ndarray, u: np.ndarray, delta: np.ndarray) -> "GlobalLinearJacobian":
        del z
        u = np.asarray(u, dtype=float)
        delta = np.asarray(delta, dtype=float)
        A = u.T @ u + self.ridge * np.eye(u.shape[1])
        B = u.T @ delta
        self.J = np.linalg.solve(A, B).T
        return self

    def predict(self, z_query: np.ndarray) -> np.ndarray:
        del z_query
        if self.J is None:
            raise RuntimeError("GlobalLinearJacobian.fit must be called before predict.")
        return self.J


class InverseDynamicsKNN:
    """KNN inverse dynamics baseline mapping (z, delta) to action."""

    def __init__(self, k: int = 32, z_scale: float = 1.0, delta_scale: float = 0.4):
        self.k = k
        self.z_scale = z_scale
        self.delta_scale = delta_scale
        self.features = None
        self.u = None

    def fit(self, z: np.ndarray, u: np.ndarray, delta: np.ndarray) -> "InverseDynamicsKNN":
        z = np.asarray(z, dtype=float) / self.z_scale
        delta = np.asarray(delta, dtype=float) / self.delta_scale
        self.features = np.hstack([z, delta])
        self.u = np.asarray(u, dtype=float)
        return self

    def predict(self, z_query: np.ndarray, delta_query: np.ndarray) -> np.ndarray:
        if self.features is None:
            raise RuntimeError("InverseDynamicsKNN.fit must be called before predict.")
        feat = np.r_[np.asarray(z_query, dtype=float) / self.z_scale, np.asarray(delta_query, dtype=float) / self.delta_scale]
        d = np.linalg.norm(self.features - feat[None, :], axis=1)
        idx = np.argsort(d)[: min(self.k, len(d))]
        w = 1.0 / (d[idx] + 1e-6)
        return np.average(self.u[idx], axis=0, weights=w)


def noisy_jacobian(J: np.ndarray, noise_scale: float, rng: np.random.Generator) -> np.ndarray:
    J = np.asarray(J, dtype=float)
    scale = noise_scale * (np.linalg.norm(J) / np.sqrt(J.size) + 1e-8)
    return J + rng.normal(0.0, scale, size=J.shape)


def corrupted_jacobian(env, z: np.ndarray) -> np.ndarray:
    """Same-shaped wrong action map used for robustness stress tests."""

    J = np.asarray(env.jacobian(z), dtype=float).copy()
    name = getattr(env, "name", "")
    if name == "point_mass":
        return 0.6 * np.array([[0.0, 1.0], [1.0, 0.0]])
    if name == "nonholonomic_car":
        z_bad = np.asarray(z, dtype=float).copy()
        z_bad[2] = z_bad[2] + np.pi / 2.0
        return env.jacobian(z_bad)
    if name == "two_link_arm":
        z_bad = np.asarray(z, dtype=float) + np.array([0.55, -0.35])
        return env.jacobian(z_bad)
    if name == "planar_pusher":
        J[2:4, :] = 0.0
        return J
    return J[:, ::-1]


def collect_rollout_jacobian_data(env, n: int, seed: int = 0, action_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z_rows, u_rows, d_rows = [], [], []
    low, high = env.action_bounds
    for _ in range(n):
        z = env.sample_state(rng)
        u = rng.uniform(low * action_scale, high * action_scale)
        z_next = env.step(z, u)
        z_rows.append(z)
        u_rows.append(u)
        d_rows.append(z_next - z)
    return np.asarray(z_rows), np.asarray(u_rows), np.asarray(d_rows)


def recommended_local_j_params(env, n: int) -> tuple[int, float]:
    name = getattr(env, "name", "")
    if name == "two_link_arm":
        return min(16, n), 0.15
    if name == "planar_pusher":
        return min(32, n), 0.25
    if name == "nonholonomic_car":
        return min(32, n), 0.45
    return min(32, n), 0.55
