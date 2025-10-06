"""
Test the platoon environment with linear feedback controller.
"""

import gymnasium as gym
import numpy as np

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
    k = np.array([[1, 2]])
    ctrl = LinearFeedback(k)

    # set up virtual leader
    vl_vel = 22.0
    vl_traj_type = "constant_velocity"
    assert vl_traj_type in VL_TRAJECTORY_TYPES
    vl_traj_args = {"horizon": None, "dt": dt}
    vl = VirtualLeader("constant_velocity", vl_traj_args, velocity=vl_vel)

    # set up platoon env
    n_vehicles = 10
    dyns = [dyn for _ in range(n_vehicles)]
    ctrls = [ctrl for _ in range(n_vehicles)]
    vehs = [Vehicle(dyns[0], position=0, velocity=20.0)]
    vehs += [
        Vehicle(dyns[i], position=-i * 5.0, velocity=20.0) for i in range(1, n_vehicles)
    ]
    render_mode = "human"
    d_des = 5.0
    config = {
        "distance_headway": d_des,
        "time_headway": 0.0,
        "reset_time": 10.0,
        "vehicles": vehs,
        "virtual_leader": vl,
    }
    env = gym.make(
        "platoon_env-v0",
        config=config,
        render_mode=render_mode,
    )
    obs, _ = env.reset()
    actions = []
    for i, o in enumerate(obs):
        d = d_des if i > 0 else 0
        error = np.array([o[0] - d, o[1]])
        action, _ = ctrls[i](error)
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
                action, _ = ctrls[i](error)
                actions.append(action + vehs[i].state[1])
            env.render()
        except KeyboardInterrupt:
            break
    env.close()


def test_platoon_env_accel_dyn_lfbk_ctrl():
    # set up dynamics
    tau = 0.5
    dt = 0.1
    x_lims = np.array([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])
    u_lims = np.array([[-np.inf, np.inf]])
    dyn = LinearAccel(dt, tau, x_lims, u_lims, full_obs=True)

    # set up controller
    k = np.array([[1, 4, 1]])
    ctrl = LinearFeedback(k)

    # set up virtual leader
    vl_vel = 22.0
    vl_traj_type = "constant_velocity"
    assert vl_traj_type in VL_TRAJECTORY_TYPES
    vl_traj_args = {"horizon": None, "dt": dt}
    vl = VirtualLeader("constant_velocity", vl_traj_args, velocity=vl_vel)

    # set up platoon env
    d_des = 5.0
    n_vehicles = 10
    platoon_vel = 20.0
    dyns = [dyn for _ in range(n_vehicles)]
    ctrls = [ctrl for _ in range(n_vehicles)]
    vehs = [Vehicle(dyns[0], position=0, velocity=20.0, acceleration=0)]
    vehs += [
        Vehicle(dyns[i], position=-i * d_des, velocity=platoon_vel, acceleration=0)
        for i in range(1, n_vehicles)
    ]
    render_mode = "human"
    config = {
        "distance_headway": d_des,
        "time_headway": 0.0,
        "reset_time": 10.0,
        "vehicles": vehs,
        "virtual_leader": vl,
    }
    env = gym.make(
        "platoon_env-v0",
        config=config,
        render_mode=render_mode,
    )
    obs, _ = env.reset()
    actions = []
    for i, o in enumerate(obs):
        d = d_des if i > 0 else 0
        error = np.array([o[0] - d, o[1], o[2]])
        action, _ = ctrls[i](error)
        actions.append(action)

    while True:
        try:
            obs, _, _, trunc, _ = env.step(action=actions)
            if trunc:
                break
            actions = []
            for i, o in enumerate(obs):
                d = d_des if i > 0 else 0
                error = np.array([o[0] - d, o[1], o[2]])
                action, _ = ctrls[i](error)
                actions.append(action)
            env.render()
        except KeyboardInterrupt:
            break
    env.close()


if __name__ == "__main__":
    test_platoon_env_vel_dyn_lfbk_ctrl()
    test_platoon_env_accel_dyn_lfbk_ctrl()
