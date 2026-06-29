import numpy as np

from actionability.residuals import solve_actionability


def test_unconstrained_projection_zero_residual_for_reachable_delta():
    J = np.eye(2)
    delta = np.array([0.2, -0.3])
    out = solve_actionability(delta, J, lam=1e-8)
    assert out.residual < 1e-6
    assert np.allclose(out.projected_delta, delta, atol=1e-5)


def test_box_projection_penalizes_action_limit_violation():
    J = np.eye(1)
    delta = np.array([2.0])
    out = solve_actionability(delta, J, bounds=(np.array([-0.5]), np.array([0.5])), lam=0.0)
    assert out.u[0] <= 0.5 + 1e-8
    assert out.residual > 2.0
