import numpy as np

from platoon_gym.ctrl.sparse_dmpc import QuadPFMPC, L1PFMPC


def test_quad_pfmpc():
    # case 1
    H = 10
    d_des = 5.0
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0], [1]])
    C = np.eye(2)
    x_min = np.array([-np.inf, -10.0])
    x_max = np.array([np.inf, 10.0])
    u_min = -5.0
    u_max = 10.0
    Q = np.eye(2)
    Qp = np.eye(2)
    R = np.eye(1)
    Cslew = np.array([[0, 1]])
    dslew = np.array([1.0])
    Qf = None
    Qpf = None
    Afx = np.eye(2)
    Afu = None

    pfmpc = QuadPFMPC(
        H,
        d_des,
        A,
        B,
        C,
        x_min,
        x_max,
        u_min,
        u_max,
        Q,
        Qp,
        R,
        Cslew,
        dslew,
        Qf,
        Qpf,
        Afx,
        Afu,
    )

    x0 = np.array([-d_des, 1.0])
    yp = np.zeros((2, H + 1))
    yp[:, 0] = np.array([0.0, 1.0])
    for i in range(H):
        yp[:, i + 1] = C @ A @ yp[:, i]
    ur = None
    bfx = yp[:, -1] - np.array([d_des, 0.0])
    bfu = None

    action, info = pfmpc.control(x0, yp, ur, bfx, bfu)
    assert info["status"] == "optimal"
    assert np.allclose(action, 0.0)
    assert np.allclose(info["planned states"][1, :], x0[1])
    assert np.allclose(info["planned states"][0, :], np.arange(x0[0], x0[0] + H + 1, 1))
    assert np.allclose(info["planned inputs"], 0.0)

    # case 2
    H = 10
    d_des = 5.0
    A = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    B = np.array([[0], [0], [1]])
    C = np.block([np.eye(2), np.zeros((2, 1))])
    x_min = np.array([-np.inf, -10.0, -5.0])
    x_max = np.array([np.inf, 10.0, 5.0])
    u_min = -5.0
    u_max = 10.0
    Q = np.eye(2)
    Qp = np.eye(2)
    R = np.eye(1)
    Cslew = np.array([[0, 0, 1]])
    dslew = np.array([1.0])
    Qf = None
    Qpf = None
    Afx = np.eye(3)
    Afu = None

    pfmpc = QuadPFMPC(
        H,
        d_des,
        A,
        B,
        C,
        x_min,
        x_max,
        u_min,
        u_max,
        Q,
        Qp,
        R,
        Cslew,
        dslew,
        Qf,
        Qpf,
        Afx,
        Afu,
    )

    x0 = np.array([-d_des, 1.0, 0.0])
    xp = np.zeros((3, H + 1))
    xp[:, 0] = np.array([0.0, 1.0, 0.0])
    yp = np.zeros((2, H + 1))
    yp[:, 0] = C @ xp[:, 0]
    for i in range(H):
        xp[:, i + 1] = A @ xp[:, i]
        yp[:, i + 1] = C @ xp[:, i + 1]
    ur = None
    bfx = np.concatenate((yp[:, -1] - np.array([d_des, 0.0]), np.array([0.0])))
    bfu = None

    action, info = pfmpc.control(x0, yp, ur, bfx, bfu)
    assert info["status"] == "optimal"
    assert np.allclose(action, 0.0)
    assert np.allclose(info["planned states"][1, :], x0[1])
    assert np.allclose(info["planned states"][0, :], np.arange(x0[0], x0[0] + H + 1, 1))
    assert np.allclose(info["planned inputs"], 0.0)

    # case 3
    H = 50
    d_des = 5.0
    dt = 0.1
    tau = 0.5
    A = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1 - dt / tau]])
    B = np.array([[0], [0], [dt / tau]])
    C = np.block([np.eye(2), np.zeros((2, 1))])
    x_min = np.array([-np.inf, -10.0, -5.0])
    x_max = np.array([np.inf, 10.0, 5.0])
    u_min = -5.0
    u_max = 10.0
    Q = np.eye(2)
    Qp = np.eye(2)
    R = np.eye(1)
    Cslew = np.array([[0, 0, 1]])
    dslew = np.array([0.5])
    Qf = None
    Qpf = None
    Afx = np.eye(3)
    Afu = None

    x0 = np.array([-d_des - 1, 2.0, 0.0])
    xp = np.zeros((3, H + 1))
    xp[:, 0] = np.array([0.0, 2.0, 0.0])
    yp = np.zeros((2, H + 1))
    yp[:, 0] = C @ xp[:, 0]
    for i in range(H):
        xp[:, i + 1] = A @ xp[:, i]
        yp[:, i + 1] = C @ xp[:, i + 1]
    ur = None
    bfx = np.concatenate((yp[:, -1] - np.array([d_des, 0.0]), np.array([0.0])))
    bfu = None

    pfmpc = QuadPFMPC(
        H,
        d_des,
        A,
        B,
        C,
        x_min,
        x_max,
        u_min,
        u_max,
        Q,
        Qp,
        R,
        Cslew,
        dslew,
        Qf,
        Qpf,
        Afx,
        Afu,
    )

    action, info = pfmpc.control(x0, yp, ur, bfx, bfu)
    assert info["status"] == "optimal"
    assert action > 0.0
    assert np.allclose(
        info["planned states"][:2, -1], yp[:, -1] - np.array([d_des, 0.0])
    )
    assert np.isclose(info["planned states"][2, -1], 0.0)


def test_l1_pfmpc():
    # case 1
    H = 10
    d_des = 5.0
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0], [1]])
    C = np.eye(2)
    x_min = np.array([-np.inf, -10.0])
    x_max = np.array([np.inf, 10.0])
    u_min = -5.0
    u_max = 10.0
    q = 1.0
    qp = 1.0
    r = 1.0
    W = np.eye(2)
    Cslew = np.array([[0, 1]])
    dslew = np.array([1.0])
    qf = None
    qpf = None
    Afx = np.eye(2)
    Afu = None

    pfmpc = L1PFMPC(
        H,
        d_des,
        A,
        B,
        C,
        x_min,
        x_max,
        u_min,
        u_max,
        q,
        qp,
        r,
        W,
        Cslew,
        dslew,
        qf,
        qpf,
        Afx,
        Afu,
    )

    x0 = np.array([-d_des, 1.0])
    yp = np.zeros((2, H + 1))
    yp[:, 0] = np.array([0.0, 1.0])
    for i in range(H):
        yp[:, i + 1] = C @ A @ yp[:, i]
    ur = None
    bfx = yp[:, -1] - np.array([d_des, 0.0])
    bfu = None

    action, info = pfmpc.control(x0, yp, ur, bfx, bfu)
    assert info["status"] == "optimal"
    assert np.allclose(action, 0.0)
    assert np.allclose(info["planned states"][1, :], x0[1])
    assert np.allclose(info["planned states"][0, :], np.arange(x0[0], x0[0] + H + 1, 1))
    assert np.allclose(info["planned inputs"], 0.0)

    # case 2
    H = 10
    d_des = 5.0
    A = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    B = np.array([[0], [0], [1]])
    C = np.block([np.eye(2), np.zeros((2, 1))])
    x_min = np.array([-np.inf, -10.0, -5.0])
    x_max = np.array([np.inf, 10.0, 5.0])
    u_min = -5.0
    u_max = 10.0
    q = 1.0
    qp = 1.0
    r = 1.0
    W = np.eye(2)
    Cslew = np.array([[0, 0, 1]])
    dslew = np.array([1.0])
    qf = None
    qpf = None
    Afx = np.eye(3)
    Afu = None

    pfmpc = L1PFMPC(
        H,
        d_des,
        A,
        B,
        C,
        x_min,
        x_max,
        u_min,
        u_max,
        q,
        qp,
        r,
        W,
        Cslew,
        dslew,
        qf,
        qpf,
        Afx,
        Afu,
    )

    x0 = np.array([-d_des, 1.0, 0.0])
    xp = np.zeros((3, H + 1))
    xp[:, 0] = np.array([0.0, 1.0, 0.0])
    yp = np.zeros((2, H + 1))
    yp[:, 0] = C @ xp[:, 0]
    for i in range(H):
        xp[:, i + 1] = A @ xp[:, i]
        yp[:, i + 1] = C @ xp[:, i + 1]
    ur = None
    bfx = np.concatenate((yp[:, -1] - np.array([d_des, 0.0]), np.array([0.0])))
    bfu = None

    action, info = pfmpc.control(x0, yp, ur, bfx, bfu)
    assert info["status"] == "optimal"
    assert np.allclose(action, 0.0)
    assert np.allclose(info["planned states"][1, :], x0[1])
    assert np.allclose(info["planned states"][0, :], np.arange(x0[0], x0[0] + H + 1, 1))
    assert np.allclose(info["planned inputs"], 0.0)

    # case 3
    H = 50
    d_des = 5.0
    dt = 0.1
    tau = 0.5
    A = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1 - dt / tau]])
    B = np.array([[0], [0], [dt / tau]])
    C = np.block([np.eye(2), np.zeros((2, 1))])
    x_min = np.array([-np.inf, -10.0, -5.0])
    x_max = np.array([np.inf, 10.0, 5.0])
    u_min = -5.0
    u_max = 10.0
    q = 1.0
    qp = 1.0
    r = 1.0
    W = np.eye(2)
    Cslew = np.array([[0, 0, 1]])
    dslew = np.array([0.5])
    qf = None
    qpf = None
    Afx = np.eye(3)
    Afu = None

    pfmpc = L1PFMPC(
        H,
        d_des,
        A,
        B,
        C,
        x_min,
        x_max,
        u_min,
        u_max,
        q,
        qp,
        r,
        W,
        Cslew,
        dslew,
        qf,
        qpf,
        Afx,
        Afu,
    )

    x0 = np.array([-d_des - 1, 1.0, 0.0])
    xp = np.zeros((3, H + 1))
    xp[:, 0] = np.array([0.0, 1.0, 0.0])
    yp = np.zeros((2, H + 1))
    yp[:, 0] = C @ xp[:, 0]
    for i in range(H):
        xp[:, i + 1] = A @ xp[:, i]
        yp[:, i + 1] = C @ xp[:, i + 1]
    ur = None
    bfx = np.concatenate((yp[:, -1] - np.array([d_des, 0.0]), np.array([0.0])))
    bfu = None

    action, info = pfmpc.control(x0, yp, ur, bfx, bfu)
    assert info["status"] == "optimal"
    assert action > 0.0
    assert np.allclose(
        info["planned states"][:2, -1], yp[:, -1] - np.array([d_des, 0.0])
    )
    assert np.isclose(info["planned states"][2, -1], 0.0)


if __name__ == "__main__":
    test_quad_pfmpc()
    test_l1_pfmpc()
