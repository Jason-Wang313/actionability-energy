import numpy as np

from actionability.repair import RepairConfig, actionability_energy, repair_trajectory
from envs import NonholonomicCarEnv
from experiments.common import interpolate


def test_repair_reduces_sideways_car_actionability_energy():
    env = NonholonomicCarEnv()
    candidate = interpolate(env.default_start(), np.array([0.0, 1.0, 0.0]), 10)
    before = actionability_energy(env, candidate)
    repaired, info = repair_trajectory(
        env,
        candidate,
        candidate[-1],
        RepairConfig(alpha_goal=60.0, beta_actionability=40.0, gamma_smoothness=0.4, world_weight=0.5, maxiter=80),
    )
    after = actionability_energy(env, repaired)
    assert repaired.shape == candidate.shape
    assert info["iterations"] > 0
    assert after < before
