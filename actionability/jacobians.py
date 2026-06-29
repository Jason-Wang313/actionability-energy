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


def collect_rollout_jacobian_data(env, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z_rows, u_rows, d_rows = [], [], []
    low, high = env.action_bounds
    for _ in range(n):
        z = env.sample_state(rng)
        u = rng.uniform(low, high)
        z_next = env.step(z, u)
        z_rows.append(z)
        u_rows.append(u)
        d_rows.append(z_next - z)
    return np.asarray(z_rows), np.asarray(u_rows), np.asarray(d_rows)
