import numpy as np

from platoon_gym.ctrl.mpc import LinearMPC


def test_mpc():
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0], [1]])
    C = np.eye(2)
    Q = np.eye(2)
    R = np.eye(1)
    Qf = np.eye(2)
    Cx = np.block([[np.eye(2)], [-np.eye(2)]])
    Cu = np.block([[np.eye(1)], [-np.eye(1)]])
    dx = np.array([np.inf, 5, np.inf, 5])
    du = np.array([1, 1])
    H = 25
    initial_state = np.array([100.0, 0.0])

    mpc = LinearMPC(A, B, C, Q, R, Qf, Cx, Cu, dx, du, H)
    action, info = mpc.control(initial_state)
    assert info["status"] == "optimal", f"MPC returned {info['status']}"
    assert action.item() < 0

    mpc = LinearMPC(A, B, C, Q, R, Qf, Cx, Cu, dx, du, H)
    action, info = mpc.control(initial_state)
    assert info["status"] == "optimal", f"MPC returned {info['status']}"
    assert action.item() < 0
