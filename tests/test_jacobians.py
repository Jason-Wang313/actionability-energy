import numpy as np

from actionability.jacobians import LocalLinearJacobian, collect_rollout_jacobian_data
from envs import PointMassEnv


def test_local_linear_jacobian_learns_point_mass_map():
    env = PointMassEnv()
    z, u, d = collect_rollout_jacobian_data(env, 80, seed=0)
    model = LocalLinearJacobian(k=40).fit(z, u, d)
    J_hat = model.predict(np.array([0.2, -0.1]))
    assert J_hat.shape == (2, 2)
    assert np.allclose(J_hat, env.jacobian(np.zeros(2)), atol=2e-2)
