import numpy as np

from envs import make_envs


def test_environment_shapes_and_bounds():
    rng = np.random.default_rng(4)
    for env in make_envs():
        z = env.sample_state(rng)
        J = env.jacobian(z)
        low, high = env.action_bounds
        u = rng.uniform(low, high)
        z_next = env.step(z, u)
        assert z.shape == (env.dim,)
        assert J.shape == (env.dim, env.action_dim)
        assert z_next.shape == (env.dim,)


def test_car_sideways_motion_has_large_local_residual():
    from actionability.residuals import solve_actionability
    from envs import NonholonomicCarEnv

    env = NonholonomicCarEnv()
    z = env.default_start()
    sideways = np.array([0.0, 0.2, 0.0])
    forward = np.array([0.2, 0.0, 0.0])
    side = solve_actionability(sideways, env.jacobian(z), bounds=env.action_bounds)
    fwd = solve_actionability(forward, env.jacobian(z), bounds=env.action_bounds)
    assert side.residual > 20 * fwd.residual
