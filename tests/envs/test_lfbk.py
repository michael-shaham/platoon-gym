"""
Test the platoon environment with linear feedback controller.
"""

import gymnasium as gym
import numpy as np
import sys

from platoon_gym.ctrl.linear_feedback import LinearFeedback
from platoon_gym.dyn.linear_vel import LinearVel
from platoon_gym.dyn.linear_accel import LinearAccel
from platoon_gym.veh.vehicle import Vehicle
from platoon_gym.veh.virtual_leader import VirtualLeader
from platoon_gym.veh.utils import VL_TRAJECTORY_TYPES


def test_platoon_env_vel_dyn_lfbk_ctrl():
    # set up dynamics
    tau = 0.5
    dt = 0.1
    x_lims = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
    u_lims = np.array([[-np.inf, np.inf]])
    dyn = LinearVel(dt, x_lims, u_lims, tau)

    # set up controller
    if dyn.p == 2:
        k = np.array([[1, 2]])
    elif dyn.p == 3:
        k = np.array([[1, 2, 1]])
    else:
        exit("Unsupported output dimension: {}".format(dyn.p))
    ctrl = LinearFeedback(k)

    horizon = 100

    # set up virtual leader
    vl_vel = 22.0
    vl_traj_type = "constant velocity"
    assert vl_traj_type in VL_TRAJECTORY_TYPES
    vl_traj_args = {"horizon": horizon, "dt": dt}
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
        "horizon": None,  # not using MPC method
        "plot history length": 100,
        "plot size": plot_size,
        "render dpi": dpi,
        "reset time": 10.0,
    }
    n_vehicles = 10
    dyns = [dyn for _ in range(n_vehicles)]
    ctrls = [ctrl for _ in range(n_vehicles)]
    vehs = [Vehicle(dyns[0], position=0, velocity=20.0)]
    vehs += [
        Vehicle(dyns[i], position=-i * d_des - i, velocity=20.0)
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
    obs, _ = env.reset()
    actions = []
    for i, o in enumerate(obs):
        d = d_des if i > 0 else 0
        error = np.array([o[0] - d, o[1]])
        action, _ = ctrls[i].control(error)
        actions.append(action + vehs[i].state[1])

    while True:
        try:
            obs, _, _, trunc, _ = env.step(action=actions)
            if trunc:
                break
            actions = []
            for i, o in enumerate(obs):
                d = d_des if i > 0 else 0
                error = np.array([o[0] - d, o[1]])
                action, _ = ctrls[i].control(error)
                actions.append(action + vehs[i].state[1])
            env.render()
        except KeyboardInterrupt:
            env.close()
            break


def test_platoon_env_accel_dyn_lfbk_ctrl():
    # set up dynamics
    tau = 0.5
    dt = 0.1
    x_lims = np.array([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])
    u_lims = np.array([[-5.0, 5.0]])
    dyn = LinearAccel(dt, x_lims, u_lims, tau)

    # set up controller
    if dyn.p == 2:
        k = np.array([[1, 2]])
    elif dyn.p == 3:
        k = np.array([[1, 2, 1]])
    else:
        exit("Unsupported output dimension: {}".format(dyn.p))
    ctrl = LinearFeedback(k)

    horizon = 100

    # set up virtual leader
    vl_vel = 22.0
    vl_traj_type = "constant velocity"
    assert vl_traj_type in VL_TRAJECTORY_TYPES
    vl_traj_args = {"horizon": horizon, "dt": dt}
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
        "horizon": None,  # not using MPC method
        "plot history length": 100,
        "plot size": plot_size,
        "render dpi": dpi,
        "reset time": 10.0,
    }
    n_vehicles = 10
    platoon_vel = 20.0
    dyns = [dyn for _ in range(n_vehicles)]
    ctrls = [ctrl for _ in range(n_vehicles)]
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
    obs, _ = env.reset()
    actions = []
    for i, o in enumerate(obs):
        d = d_des if i > 0 else 0
        error = np.array([o[0] - d, o[1]])
        action, _ = ctrls[i].control(error)
        actions.append(action)

    while True:
        try:
            obs, _, _, trunc, _ = env.step(action=actions)
            if trunc:
                break
            actions = []
            for i, o in enumerate(obs):
                d = d_des if i > 0 else 0
                error = np.array([o[0] - d, o[1]])
                action, _ = ctrls[i].control(error)
                actions.append(action)
            env.render()
        except KeyboardInterrupt:
            env.close()
            break


if __name__ == "__main__":
    test_platoon_env_vel_dyn_lfbk_ctrl()
    test_platoon_env_accel_dyn_lfbk_ctrl()
