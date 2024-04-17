"""
Test the platoon environment with linear MPC controller.
"""

import gymnasium as gym
import numpy as np
import sys

from platoon_gym.ctrl.mpc import LinearMPC
from platoon_gym.dyn.linear_vel import LinearVel
from platoon_gym.dyn.linear_accel import LinearAccel
from platoon_gym.veh.vehicle import Vehicle
from platoon_gym.veh.virtual_leader import VirtualLeader
from platoon_gym.veh.utils import VL_TRAJECTORY_TYPES


def test_platoon_env_vel_dyn_mpc_ctrl():
    # set up dynamics
    tau = 0.5
    dt = 0.1
    x_lims = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
    u_lims = np.array([[-np.inf, np.inf]])
    dyn = LinearVel(dt, x_lims, u_lims, tau)

    # set up controller
    A = dyn.Ad
    B = dyn.Bd
    C = dyn.C
    Q = np.eye(dyn.n)
    R = np.eye(dyn.m)
    Qf = np.eye(dyn.n)
    Cx = np.r_[-np.eye(dyn.n), np.eye(dyn.n)]
    Cu = np.r_[-np.eye(dyn.m), np.eye(dyn.m)]
    dx = np.r_[-dyn.x_lims[:, 0], dyn.x_lims[:, 1]]
    du = np.r_[-dyn.u_lims[:, 0], dyn.u_lims[:, 1]]
    H = 50
    mpc = LinearMPC(A, B, C, Q, R, Qf, Cx, Cu, dx, du, H)

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
    d_des = 5.0
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
    ctrls = [mpc for _ in range(n_vehicles)]
    vehs = [Vehicle(dyns[0], position=0, velocity=20.0)]
    vehs += [
        Vehicle(dyns[i], position=-i * d_des - i, velocity=platoon_vel)
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

    veh_state_plans = []
    veh_control_plans = []
    for v in vehs:
        n, m = v.dyn.n, v.dyn.m
        state_plan = np.zeros((n, H + 1))
        state_plan[:, 0] = v.state[:n]
        control_plan = v.state[1] * np.ones((m, H))
        for k in range(H):
            state_plan[:, k + 1] = v.dyn.forward(state_plan[:, k], control_plan[:, k])
        veh_state_plans.append(state_plan)
        veh_control_plans.append(control_plan)
    veh_states = env_info["vehicle states"]
    vl_plan = env_info["virtual leader plan"][:2, : H + 1]
    actions = []
    for i, o in enumerate(obs):
        if i == 0:
            action, ctrl_info = ctrls[i].control(
                veh_states[i], vl_plan, veh_control_plans[i]
            )
        else:
            pred_plan = veh_state_plans[i - 1]
            pred_plan[0, :] -= pred_plan[0, 0] - veh_states[i][0] - o[0] + d_des
            action, ctrl_info = ctrls[i].control(
                veh_states[i], pred_plan, veh_control_plans[i]
            )
        if ctrl_info["status"] != "optimal":
            assert False, f"MPC returned {ctrl_info['status']}"
        veh_state_plans[i] = ctrl_info["planned states"]
        veh_control_plans[i] = ctrl_info["planned inputs"]
        actions.append(action)

    while True:
        try:
            obs, _, _, trunc, env_info = env.step(action=actions)
            if trunc:
                break
            veh_states = env_info["vehicle states"]
            vl_plan = env_info["virtual leader plan"][:2, : H + 1]
            actions = []
            for i, o in enumerate(obs):
                if i == 0:
                    action, ctrl_info = ctrls[i].control(
                        veh_states[i], vl_plan, veh_control_plans[i]
                    )
                else:
                    pred_plan = veh_state_plans[i - 1]
                    pred_plan[0, :] -= pred_plan[0, 0] - veh_states[i][0] - o[0] + d_des
                    action, ctrl_info = ctrls[i].control(
                        veh_states[i], pred_plan, veh_control_plans[i]
                    )
                if ctrl_info["status"] != "optimal":
                    assert False, f"MPC returned {ctrl_info['status']}"
                veh_state_plans[i] = ctrl_info["planned states"]
                veh_control_plans[i] = ctrl_info["planned inputs"]
                actions.append(action)
            env.render()
        except KeyboardInterrupt:
            env.close()
            break


def test_platoon_env_accel_dyn_mpc_ctrl():
    # set up dynamics
    tau = 0.5
    dt = 0.1
    x_lims = np.array([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])
    u_lims = np.array([[-5.0, 5.0]])
    dyn = LinearAccel(dt, x_lims, u_lims, tau, full_obs=True)

    # set up controller
    A = dyn.Ad
    B = dyn.Bd
    C = dyn.C
    Q = np.eye(dyn.n)
    R = np.eye(dyn.m)
    Qf = np.eye(dyn.n)
    Cx = np.r_[-np.eye(dyn.n), np.eye(dyn.n)]
    Cu = np.r_[-np.eye(dyn.m), np.eye(dyn.m)]
    dx = np.r_[-dyn.x_lims[:, 0], dyn.x_lims[:, 1]]
    du = np.r_[-dyn.u_lims[:, 0], dyn.u_lims[:, 1]]
    H = 50
    mpc = LinearMPC(A, B, C, Q, R, Qf, Cx, Cu, dx, du, H)

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
    d_des = 5.0
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
    ctrls = [mpc for _ in range(n_vehicles)]
    vehs = [Vehicle(dyns[0], position=0, velocity=20.0, acceleration=0)]
    vehs += [
        Vehicle(dyns[i], position=-i * d_des - i, velocity=platoon_vel, acceleration=0)
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

    veh_plans = []
    for v in vehs:
        n, m = v.dyn.n, v.dyn.m
        state_plan = np.zeros((n, H + 1))
        state_plan[:, 0] = v.state
        control_plan = np.zeros((m, H))
        for k in range(H):
            state_plan[:, k + 1] = v.dyn.forward(state_plan[:, k], control_plan[:, k])
        veh_plans.append(state_plan)
    veh_states = env_info["vehicle states"]
    vl_plan = env_info["virtual leader plan"][:, : H + 1]
    actions = []
    for i, o in enumerate(obs):
        if i == 0:
            action, ctrl_info = ctrls[i].control(veh_states[i], vl_plan)
        else:
            pred_plan = veh_plans[i - 1]
            pred_plan[0, :] -= pred_plan[0, 0] - veh_states[i][0] - o[0] + d_des
            action, ctrl_info = ctrls[i].control(veh_states[i], pred_plan)
        if ctrl_info["status"] != "optimal":
            assert False, f"MPC returned {ctrl_info['status']}"
        veh_plans[i] = ctrl_info["planned states"]
        actions.append(action)

    while True:
        try:
            obs, _, _, trunc, env_info = env.step(action=actions)
            if trunc:
                break
            veh_states = env_info["vehicle states"]
            vl_plan = env_info["virtual leader plan"][:, : H + 1]
            actions = []
            for i, o in enumerate(obs):
                if i == 0:
                    action, ctrl_info = ctrls[i].control(veh_states[i], vl_plan)
                else:
                    pred_plan = veh_plans[i - 1]
                    pred_plan[0, :] -= pred_plan[0, 0] - veh_states[i][0] - o[0] + d_des
                    action, ctrl_info = ctrls[i].control(veh_states[i], pred_plan)
                if ctrl_info["status"] != "optimal":
                    assert False, f"MPC returned {ctrl_info['status']}"
                veh_plans[i] = ctrl_info["planned states"]
                actions.append(action)
            env.render()
        except KeyboardInterrupt:
            env.close()
            break


if __name__ == "__main__":
    test_platoon_env_vel_dyn_mpc_ctrl()
    test_platoon_env_accel_dyn_mpc_ctrl()
