"""
Test the platoon environment with sparse distributed MPC controllers.
"""

import copy
import gymnasium as gym
import numpy as np
import sys

from platoon_gym.ctrl.sparse_dmpc import QuadPFMPC, L1PFMPC
from platoon_gym.dyn.linear_vel import LinearVel
from platoon_gym.veh.vehicle import Vehicle
from platoon_gym.veh.virtual_leader import VirtualLeader
from platoon_gym.veh.utils import VL_TRAJECTORY_TYPES


def test_platoon_env_vel_dyn_quad_pfmpc_ctrl():
    # set up dynamics
    tau = 0.5
    dt = 0.1
    x_lims = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
    u_lims = np.array([[-np.inf, np.inf]])
    dyn = LinearVel(dt, x_lims, u_lims, tau)
    d_des = 5.0

    # set up controller
    H = 50
    A = dyn.Ad
    B = dyn.Bd
    C = dyn.C
    n, m, p = dyn.n, dyn.m, dyn.p
    x_min = np.array([-np.inf, 0.0])
    x_max = np.array([np.inf, 30.0])
    u_min = u_lims[0, 0]
    u_max = u_lims[0, 1]
    Q = np.eye(p)
    Qp = np.eye(p)
    R = np.eye(m)
    Cslew = np.array([[0.0, 1.0]])
    dslew = np.array([1.0])
    # Cslew, dslew = None, None
    Qf = None
    Qpf = None
    Afx = np.eye(n)
    Afu = None
    head_args = [
        H,
        0.0,
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
    ]
    trail_args = [
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
    ]

    # set up virtual leader
    vl_vel = 22.0
    vl_traj_type = "constant velocity"
    assert vl_traj_type in VL_TRAJECTORY_TYPES
    vl_traj_args = {"horizon": H, "dt": dt}
    vl = VirtualLeader("constant velocity", vl_traj_args, velocity=vl_vel)

    # set up platoon env
    plot_size = (6, 4)
    if sys.platform.startswith("linux"):
        dpi = 100
    elif sys.platform == "darwin":
        dpi = 50
    else:
        exit("Unsupported OS found: {}".format(sys.platform))
    env_args = {
        "headway": "CDH",
        "desired distance": d_des,
        "topology": "PF",
        "dt": dt,
        "horizon": H,
        "plot history length": 100,
        "plot size": plot_size,
        "render dpi": dpi,
        "reset time": 10.0,
    }
    n_vehicles = 10
    platoon_vel = 20.0
    dyns = [dyn for _ in range(n_vehicles)]
    ctrls = [QuadPFMPC(*head_args)] + [
        QuadPFMPC(*trail_args) for _ in range(1, n_vehicles)
    ]
    vehs = [Vehicle(dyns[0], position=0, velocity=20.0)]
    vehs += [
        Vehicle(dyns[i], position=-i * d_des, velocity=platoon_vel)
        for i in range(1, n_vehicles)
    ]
    render_mode = "plot"
    env = gym.make(
        "platoon_gym-v0",
        vehicles=vehs,
        virtual_leader=vl,
        env_args=env_args,
        render_mode=render_mode,
    )
    obs, env_info = env.reset()
    veh_states = env_info["vehicle states"]

    prev_assumed_states = []
    for i in range(n_vehicles):
        uref = veh_states[i][1] * np.ones((m, H))
        xa = ctrls[i].init_assumed_trajectory(veh_states[i], uref)
        prev_assumed_states.append(xa)

    while True:
        try:
            env.render()
            vl_plan = copy.deepcopy(env_info["virtual leader plan"][:2, : H + 1])
            actions = []
            assumed_states = []
            for i in range(len(obs)):
                if i == 0:
                    end_state = copy.deepcopy(vl_plan[:, -1])
                    uref = end_state[1] * np.ones((m, H))
                    action, ctrl_info = ctrls[i].control(
                        veh_states[i],
                        vl_plan,
                        uref,
                        bfx=end_state,
                        ua_end=end_state[1],
                    )
                    assumed_states.append(ctrl_info["assumed outputs"])
                else:
                    end_state = copy.deepcopy(prev_assumed_states[i - 1][:, -1])
                    end_state[0] -= d_des
                    uref = end_state[1] * np.ones((m, H))
                    action, ctrl_info = ctrls[i].control(
                        veh_states[i],
                        prev_assumed_states[i - 1],
                        uref,
                        bfx=end_state,
                        ua_end=end_state[1],
                    )
                    assumed_states.append(ctrl_info["assumed outputs"])
                actions.append(action)

            # step environment
            obs, _, _, trunc, env_info = env.step(action=actions)
            if trunc:
                break
            veh_states = copy.deepcopy(env_info["vehicle states"])
            prev_assumed_states = copy.deepcopy(assumed_states)
        except KeyboardInterrupt:
            env.close()
            break


def test_platoon_env_vel_dyn_l1_pfmpc_ctrl():
    # set up dynamics
    tau = 0.5
    dt = 0.1
    x_lims = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
    u_lims = np.array([[-np.inf, np.inf]])
    dyn = LinearVel(dt, x_lims, u_lims, tau)
    d_des = 5.0
    # set up controller
    H = 50
    A = dyn.Ad
    B = dyn.Bd
    C = dyn.C
    n, m, p = dyn.n, dyn.m, dyn.p
    x_min = np.array([-np.inf, 0.0])
    x_max = np.array([np.inf, 30.0])
    u_min = u_lims[0, 0]
    u_max = u_lims[0, 1]
    q = 1.0
    qp = 1.0
    r = 1.0
    W = np.eye(p)
    Cslew = np.array([[0, 1]])
    dslew = np.array([0.5])
    qf = None
    qpf = None
    Afx = np.eye(n)
    Afu = None
    head_args = [
        H,
        0.0,
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
    ]
    trail_args = [
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
    ]

    # set up virtual leader
    vl_vel = 22.0
    vl_traj_type = "constant velocity"
    assert vl_traj_type in VL_TRAJECTORY_TYPES
    vl_traj_args = {"horizon": H, "dt": dt}
    vl = VirtualLeader("constant velocity", vl_traj_args, velocity=vl_vel)

    # set up platoon env
    plot_size = (6, 4)
    if sys.platform.startswith("linux"):
        dpi = 100
    elif sys.platform == "darwin":
        dpi = 50
    else:
        exit("Unsupported OS found: {}".format(sys.platform))
    env_args = {
        "headway": "CDH",
        "desired distance": d_des,
        "topology": "PF",
        "dt": dt,
        "horizon": H,
        "plot history length": 100,
        "plot size": plot_size,
        "render dpi": dpi,
        "reset time": 10.0,
    }
    n_vehicles = 10
    platoon_vel = 20.0

    dyns = [dyn for _ in range(n_vehicles)]
    ctrls = [L1PFMPC(*head_args)] + [L1PFMPC(*trail_args) for _ in range(1, n_vehicles)]
    vehs = [Vehicle(dyns[0], position=0, velocity=20.0)]
    vehs += [
        Vehicle(dyns[i], position=-i * d_des, velocity=platoon_vel)
        for i in range(1, n_vehicles)
    ]
    render_mode = "plot"

    env = gym.make(
        "platoon_gym-v0",
        vehicles=vehs,
        virtual_leader=vl,
        env_args=env_args,
        render_mode=render_mode,
    )
    obs, env_info = env.reset()
    veh_states = env_info["vehicle states"]

    prev_assumed_states = []
    for i in range(n_vehicles):
        uref = veh_states[i][1] * np.ones((m, H))
        xa = ctrls[i].init_assumed_trajectory(veh_states[i], uref)
        prev_assumed_states.append(xa)

    while True:
        try:
            env.render()
            vl_plan = copy.deepcopy(env_info["virtual leader plan"][:2, : H + 1])
            actions = []
            assumed_states = []
            for i in range(len(obs)):
                if i == 0:
                    end_state = copy.deepcopy(vl_plan[:, -1])
                    uref = end_state[1] * np.ones((m, H))
                    action, ctrl_info = ctrls[i].control(
                        veh_states[i],
                        vl_plan,
                        uref,
                        bfx=end_state,
                        ua_end=end_state[1],
                    )
                    assumed_states.append(ctrl_info["assumed outputs"])
                else:
                    end_state = copy.deepcopy(prev_assumed_states[i - 1][:, -1])
                    end_state[0] -= d_des
                    uref = end_state[1] * np.ones((m, H))
                    action, ctrl_info = ctrls[i].control(
                        veh_states[i],
                        prev_assumed_states[i - 1],
                        uref,
                        bfx=end_state,
                        ua_end=end_state[1],
                    )
                    assumed_states.append(ctrl_info["assumed outputs"])
                actions.append(action)

            # step environment
            obs, _, _, trunc, env_info = env.step(action=actions)
            if trunc:
                break
            veh_states = copy.deepcopy(env_info["vehicle states"])
            prev_assumed_states = copy.deepcopy(assumed_states)
        except KeyboardInterrupt:
            env.close()
            break


if __name__ == "__main__":
    test_platoon_env_vel_dyn_quad_pfmpc_ctrl()
    test_platoon_env_vel_dyn_l1_pfmpc_ctrl()
