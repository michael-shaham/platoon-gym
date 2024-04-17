"""
Test the virtual leader functions.
"""

import matplotlib.pyplot as plt
import numpy as np

from platoon_gym.veh.virtual_leader import VirtualLeader
from platoon_gym.veh.utils import VL_TRAJECTORY_TYPES


def plot_vl_plan(vl: VirtualLeader, title: str = "Virtual Leader"):
    fig, ax = plt.subplots(3, 1, figsize=(4, 6), sharex=True)
    fig.suptitle(title)
    fig.subplots_adjust(0.25, 0.1, 0.9, 0.9, 0.2, 0.2)
    ax[0].plot(vl.time_forecast, vl.plan[0, :])
    ax[0].set_ylabel("position")
    ax[1].plot(vl.time_forecast, vl.plan[1, :])
    ax[1].set_ylabel("velocity")
    ax[2].plot(vl.time_forecast, vl.plan[2, :])
    ax[2].set_ylabel("acceleration")
    ax[2].set_xlabel("time")
    plt.show()


def test_constant_velocity():
    vl_vel = 10.0
    dt = 0.1
    H = 10
    vl_traj_type = "constant velocity"
    assert vl_traj_type in VL_TRAJECTORY_TYPES
    vl_traj_args = {"horizon": H, "dt": dt}
    vl = VirtualLeader(vl_traj_type, vl_traj_args, velocity=vl_vel)
    vl.reset()
    assert (vl.plan[1, :] == vl_vel).all()
    assert (vl.plan[2, :] == 0).all()
    assert np.isclose(vl.plan[0, :], dt * vl_vel * np.arange(0, H + 1)).all()
    plot_vl_plan(vl, "constant velocity")


def test_step_velocity():
    vl_vel = 10.0
    dt = 0.1
    H = 50
    vl_traj_type = "velocity step"
    step_time = 1.0
    step_vel = 20.0
    assert vl_traj_type in VL_TRAJECTORY_TYPES
    vl_traj_args = {
        "horizon": H,
        "dt": dt,
        "step time": step_time,
        "step velocity": step_vel,
    }
    vl = VirtualLeader(vl_traj_type, vl_traj_args, velocity=vl_vel)
    vl.reset()
    assert vl.plan[1, np.where(vl.time_forecast == step_time)[0]] == step_vel
    assert (vl.plan[1, : int(step_time / dt)] == vl_vel).all()
    assert (vl.plan[1, int(step_time / dt) :] == step_vel).all()
    plot_vl_plan(vl, "step velocity")

    rng = np.random.default_rng(4)
    vl_vel = 10.0
    dt = 0.1
    H = 100
    vl_traj_type = "velocity step"
    step_time = rng.uniform(2.0, 4.0)
    step_vel = rng.uniform(11.0, 15.0)
    step_accel = rng.uniform(0.5, 2.0)
    assert vl_traj_type in VL_TRAJECTORY_TYPES
    vl_traj_args = {
        "horizon": H,
        "dt": dt,
        "step time": step_time,
        "step velocity": step_vel,
        "step acceleration": step_accel,
    }
    vl = VirtualLeader(vl_traj_type, vl_traj_args, velocity=vl_vel)
    vl.reset()
    assert (vl.plan[1, np.where(vl.time_forecast <= step_time)[0]] == vl_vel).all()
    plot_vl_plan(vl, "step velocity with accel")
