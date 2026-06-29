from __future__ import annotations

import numpy as np


class PlanarPusherEnv:
    """A mode-switching pusher/object toy with a contact-dependent Jacobian."""

    name = "planar_pusher"
    dim = 4
    action_dim = 2
    success_threshold = 0.35

    def __init__(self, dt: float = 0.18, max_speed: float = 1.0, contact_radius: float = 0.32):
        self.dt = dt
        self.contact_radius = contact_radius
        self.object_gain = 0.85
        self.action_bounds = (
            -max_speed * np.ones(self.action_dim),
            max_speed * np.ones(self.action_dim),
        )

    def _contact_normal(self, z: np.ndarray) -> tuple[bool, np.ndarray]:
        z = np.asarray(z, dtype=float)
        pusher = z[:2]
        obj = z[2:4]
        vec = obj - pusher
        dist = float(np.linalg.norm(vec))
        if dist < 1e-8:
            return True, np.array([1.0, 0.0])
        return dist <= self.contact_radius, vec / dist

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        contact, n = self._contact_normal(z)
        J = np.zeros((4, 2), dtype=float)
        J[:2, :] = np.eye(2)
        if contact:
            J[2:4, :] = self.object_gain * np.outer(n, n)
        return self.dt * J

    def step(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        u = np.clip(np.asarray(u, dtype=float), *self.action_bounds)
        out = z.copy()
        contact, n = self._contact_normal(z)
        out[:2] = z[:2] + self.dt * u
        if contact and float(u @ n) > 0:
            out[2:4] = z[2:4] + self.dt * self.object_gain * (u @ n) * n
        return out

    def sample_state(self, rng: np.random.Generator) -> np.ndarray:
        obj = rng.uniform(-0.7, 0.7, size=2)
        if rng.random() < 0.5:
            angle = rng.uniform(-np.pi, np.pi)
            pusher = obj - 0.25 * np.array([np.cos(angle), np.sin(angle)])
        else:
            pusher = obj + rng.uniform(0.7, 1.1) * np.array([1.0, 0.0])
        return np.r_[pusher, obj]

    def default_start(self) -> np.ndarray:
        return np.array([-0.7, 0.0, 0.0, 0.0])

    def plot_xy(self, traj: np.ndarray) -> np.ndarray:
        return np.asarray(traj)[:, 2:4]
