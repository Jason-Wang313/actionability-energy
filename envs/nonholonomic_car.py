from __future__ import annotations

import numpy as np


def wrap_angle(theta: float) -> float:
    return float((theta + np.pi) % (2 * np.pi) - np.pi)


class NonholonomicCarEnv:
    name = "nonholonomic_car"
    dim = 3
    action_dim = 2
    success_threshold = 0.32

    def __init__(self, dt: float = 0.18, max_v: float = 1.0, max_omega: float = 1.3):
        self.dt = dt
        self.action_bounds = (
            np.array([-max_v, -max_omega], dtype=float),
            np.array([max_v, max_omega], dtype=float),
        )

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        theta = float(np.asarray(z)[2])
        return self.dt * np.array(
            [
                [np.cos(theta), 0.0],
                [np.sin(theta), 0.0],
                [0.0, 1.0],
            ]
        )

    def step(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        v, omega = np.clip(np.asarray(u, dtype=float), *self.action_bounds)
        out = z.copy()
        out[0] += self.dt * v * np.cos(z[2])
        out[1] += self.dt * v * np.sin(z[2])
        out[2] = wrap_angle(z[2] + self.dt * omega)
        return out

    def sample_state(self, rng: np.random.Generator) -> np.ndarray:
        return np.array(
            [
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-np.pi, np.pi),
            ]
        )

    def default_start(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0])

    def plot_xy(self, traj: np.ndarray) -> np.ndarray:
        return np.asarray(traj)[:, :2]
