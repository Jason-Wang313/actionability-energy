from __future__ import annotations

import numpy as np


class PointMassEnv:
    name = "point_mass"
    dim = 2
    action_dim = 2
    success_threshold = 0.25

    def __init__(self, dt: float = 0.25, max_speed: float = 1.0):
        self.dt = dt
        self.max_speed = max_speed
        self.action_bounds = (
            -max_speed * np.ones(self.action_dim),
            max_speed * np.ones(self.action_dim),
        )

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        return self.dt * np.eye(2)

    def step(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float), *self.action_bounds)
        return np.asarray(z, dtype=float) + self.dt * u

    def sample_state(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(-1.2, 1.2, size=2)

    def default_start(self) -> np.ndarray:
        return np.array([0.0, 0.0])

    def plot_xy(self, traj: np.ndarray) -> np.ndarray:
        return np.asarray(traj)[:, :2]
