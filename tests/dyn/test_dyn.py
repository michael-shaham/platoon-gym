import numpy as np

from platoon_gym.dyn.utils import (
    DISCRETIZATION_METHODS,
    pw_const_input_discretization,
    forward_euler_discretization,
)

from platoon_gym.dyn.linear_accel import LinearAccel
from platoon_gym.dyn.linear_vel import LinearVel


def test_discretization_methods():
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])

    dts = np.linspace(0.05, 1.0, 20)

    for dt in dts:
        # piecewise constant input discretization from Stanford LDS course
        Ad_true = np.array([[1, dt], [0, 1]])
        Bd_true = np.array([[0.5 * dt**2], [dt]])
        Ad, Bd = pw_const_input_discretization(A, B, dt)
        assert np.allclose(Ad_true, Ad)
        assert np.allclose(Bd_true, Bd)

        # forward euler discretization
        Ad_true = np.array([[1, dt], [0, 1]])
        Bd_true = np.array([[0.0], [dt]])
        Ad, Bd = forward_euler_discretization(A, B, dt)
        assert np.allclose(Ad_true, Ad)
        assert np.allclose(Bd_true, Bd)


def test_linear_accel():
    dt = 0.1
    x_lims = np.array([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])
    u_lims = np.array([[-np.inf, np.inf]])
    tau = 0.5
    method = "forward euler"
    assert method in DISCRETIZATION_METHODS
    dyn = LinearAccel(dt, x_lims, u_lims, tau, discretization_method=method)

    H = 10
    x_traj = np.zeros((dyn.n, H + 1))
    x_traj[:, 0] = np.array([0.0, 20.0, 0.0])
    y_traj = np.zeros((dyn.p, H + 1))
    y_traj[:, 0] = np.array([0.0, 20.0])

    x_traj_true = np.zeros_like(x_traj)
    x_traj_true[:, 0] = x_traj[:, 0]
    y_traj_true = np.zeros_like(y_traj)
    y_traj_true[:, 0] = dyn.sense(x_traj_true[:, 0])

    u_traj = np.zeros((dyn.m, H))

    for k in range(H):
        x_traj_true[0, k + 1] = x_traj_true[0, k] + dt * x_traj_true[1, k]
        x_traj_true[1, k + 1] = x_traj_true[1, k] + dt * x_traj_true[2, k]
        x_traj_true[2, k + 1] = (1 - dt / tau) * x_traj_true[2, k] + dt / tau * u_traj[
            0, k
        ]

        y_traj_true[:, k + 1] = x_traj_true[:2, k + 1]

        x_traj[:, k + 1] = dyn.forward(x_traj[:, k], u_traj[:, k])
        y_traj[:, k + 1] = dyn.sense(x_traj[:, k + 1])

    assert np.allclose(x_traj_true, x_traj)
    assert np.allclose(y_traj_true, y_traj)


def test_linear_vel():
    dt = 0.1
    x_lims = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
    u_lims = np.array([[-np.inf, np.inf]])
    tau = 0.5
    method = "forward euler"
    assert method in DISCRETIZATION_METHODS
    dyn = LinearVel(dt, x_lims, u_lims, tau, discretization_method=method)

    H = 10
    x_traj = np.zeros((dyn.n, H + 1))
    x_traj[:, 0] = np.array([0.0, 20.0])
    y_traj = np.zeros((dyn.p, H + 1))
    y_traj[:, 0] = np.array([0.0, 20.0])

    x_traj_true = np.zeros_like(x_traj)
    x_traj_true[:, 0] = x_traj[:, 0]
    y_traj_true = np.zeros_like(y_traj)
    y_traj_true[:, 0] = dyn.sense(x_traj_true[:, 0])

    u_traj = np.zeros((dyn.m, H))

    for k in range(H):
        x_traj_true[0, k + 1] = x_traj_true[0, k] + dt * x_traj_true[1, k]
        x_traj_true[1, k + 1] = (1 - dt / tau) * x_traj_true[1, k] + dt / tau * u_traj[
            0, k
        ]

        y_traj_true[:, k + 1] = x_traj_true[:2, k + 1]

        x_traj[:, k + 1] = dyn.forward(x_traj[:, k], u_traj[:, k])
        y_traj[:, k + 1] = dyn.sense(x_traj[:, k + 1])

    assert np.allclose(x_traj_true, x_traj)
    assert np.allclose(y_traj_true, y_traj)
