from __future__ import annotations

import numpy as np
import pandas as pd

from actionability.residuals import solve_actionability
from envs import NonholonomicCarEnv, PlanarPusherEnv, PointMassEnv, TwoLinkArmEnv


def _state_for_direction(env, dx: float, dy: float):
    if isinstance(env, PointMassEnv):
        start = np.array([0.0, 0.0])
        delta = np.array([dx, dy])
    elif isinstance(env, NonholonomicCarEnv):
        start = np.array([0.0, 0.0, 0.0])
        delta = np.array([dx, dy, 0.0])
    elif isinstance(env, TwoLinkArmEnv):
        start = env.default_start()
        delta = np.array([dx, dy])
    elif isinstance(env, PlanarPusherEnv):
        start = np.array([-0.7, 0.0, 0.0, 0.0])
        delta = np.array([0.0, 0.0, dx, dy])
    else:
        raise TypeError(type(env))
    return start, delta


def run_embodiment_swap(grid: int = 31) -> pd.DataFrame:
    envs = [PointMassEnv(), NonholonomicCarEnv(), TwoLinkArmEnv(), PlanarPusherEnv()]
    values = np.linspace(-0.22, 0.22, grid)
    rows = []
    for env in envs:
        for dx in values:
            for dy in values:
                start, delta = _state_for_direction(env, float(dx), float(dy))
                out = solve_actionability(delta, env.jacobian(start), bounds=env.action_bounds)
                rows.append({"env": env.name, "dx": float(dx), "dy": float(dy), "residual": out.residual})
    return pd.DataFrame(rows)
