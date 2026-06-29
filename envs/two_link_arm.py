from __future__ import annotations

import numpy as np


class TwoLinkArmEnv:
    """End-effector representation controlled by two joint velocities.

    The benchmark state z is the end-effector keypoint. The local action map is
    the analytic manipulator Jacobian evaluated at an inverse-kinematics branch.
    """

    name = "two_link_arm"
    dim = 2
    action_dim = 2
    success_threshold = 0.28

    def __init__(self, l1: float = 1.0, l2: float = 0.8, dt: float = 0.22, max_qdot: float = 0.8):
        self.l1 = l1
        self.l2 = l2
        self.dt = dt
        self.action_bounds = (
            -max_qdot * np.ones(self.action_dim),
            max_qdot * np.ones(self.action_dim),
        )

    def fk(self, q: np.ndarray) -> np.ndarray:
        q1, q2 = np.asarray(q, dtype=float)
        return np.array(
            [
                self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2),
                self.l1 * np.sin(q1) + self.l2 * np.sin(q1 + q2),
            ]
        )

    def ik(self, z: np.ndarray) -> np.ndarray:
        x, y = np.asarray(z, dtype=float)
        r = np.hypot(x, y)
        max_r = self.l1 + self.l2 - 1e-4
        min_r = abs(self.l1 - self.l2) + 1e-4
        if r < 1e-8:
            x, y = min_r, 0.0
            r = min_r
        if r > max_r:
            scale = max_r / r
            x, y = x * scale, y * scale
            r = max_r
        if r < min_r:
            scale = min_r / r
            x, y = x * scale, y * scale
            r = min_r
        c2 = np.clip((r * r - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2), -1.0, 1.0)
        q2 = np.arccos(c2)
        q1 = np.arctan2(y, x) - np.arctan2(self.l2 * np.sin(q2), self.l1 + self.l2 * np.cos(q2))
        return np.array([q1, q2])

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        q1, q2 = self.ik(z)
        J = np.array(
            [
                [-self.l1 * np.sin(q1) - self.l2 * np.sin(q1 + q2), -self.l2 * np.sin(q1 + q2)],
                [self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2), self.l2 * np.cos(q1 + q2)],
            ]
        )
        return self.dt * J

    def step(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        q = self.ik(z)
        u = np.clip(np.asarray(u, dtype=float), *self.action_bounds)
        return self.fk(q + self.dt * u)

    def sample_state(self, rng: np.random.Generator) -> np.ndarray:
        q = rng.uniform(np.array([-1.8, 0.35]), np.array([1.8, 2.2]))
        return self.fk(q)

    def default_start(self) -> np.ndarray:
        return self.fk(np.array([0.15, 1.2]))

    def plot_xy(self, traj: np.ndarray) -> np.ndarray:
        return np.asarray(traj)[:, :2]
