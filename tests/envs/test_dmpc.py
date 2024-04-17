"""
Test the platoon environment with distributed MPC controllers.
"""

import copy
import gymnasium as gym
import numpy as np
import sys

from platoon_gym.ctrl.dmpc import DMPC
from platoon_gym.dyn.linear_vel import LinearVel
from platoon_gym.veh.vehicle import Vehicle
from platoon_gym.veh.virtual_leader import VirtualLeader
from platoon_gym.veh.utils import VL_TRAJECTORY_TYPES


def test_platoon_env_vel_dyn_pf_dmpc():
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
    Q = np.eye(p)
    Q_neighbors = [np.eye(p)]
    R = np.eye(m)
    u_slew_rate = np.array([0.5])
    terminal_constraint = True
    Qf = None
    Qf_neighbors = None
    d_des_head = [0.0]
    d_des_trail = [d_des]
    time_headway = None
    output_norm = "quadratic"
    input_norm = "quadratic"
    head_args = [
        H,
        Q,
        Q_neighbors,
        R,
        A,
        B,
        C,
        x_lims,
        u_lims,
        u_slew_rate,
        terminal_constraint,
        Qf,
        Qf_neighbors,
        d_des_head,
        time_headway,
        output_norm,
        input_norm,
    ]
    trail_args = copy.deepcopy(head_args)
    trail_args[-4] = d_des_trail

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
    ctrls = [DMPC(*head_args)] + [DMPC(*trail_args) for _ in range(1, n_vehicles)]
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
        uref = veh_states[i][1] * np.ones((H, m))
        xa = ctrls[i].initialize_assumed_trajectory(veh_states[i], uref)
        prev_assumed_states.append(xa)

    while True:
        try:
            env.render()
            vl_plan = copy.deepcopy(env_info["virtual leader plan"][:2, : H + 1]).T
            actions = []
            assumed_states = []
            for i in range(len(obs)):
                if i == 0:
                    end_state = copy.deepcopy(vl_plan[-1])
                    y_neighbors = [vl_plan]
                else:
                    end_state = copy.deepcopy(prev_assumed_states[i - 1][-1])
                    end_state[0] -= d_des
                    y_neighbors = [prev_assumed_states[i - 1]]
                uref = end_state[1] * np.ones((H, m))
                action, ctrl_info = ctrls[i].control(
                    x0=veh_states[i],
                    y_neighbors=y_neighbors,
                    xa=prev_assumed_states[i],
                    ua=uref,
                    xf=end_state,
                )
                xa, ua = ctrl_info["x"], ctrl_info["u"]
                ua[:-1] = ua[1:]
                ua[-1] = xa[-1][1]
                xa[:-1] = xa[1:]
                xa[-1] = dyns[i].forward(xa[-2], ua[-1])
                assumed_states.append(xa)
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
    test_platoon_env_vel_dyn_pf_dmpc()
