import matplotlib.pyplot as plt
import numpy as np
import sys
from typing import List, Optional, Tuple

from platoon_gym.ctrl.dmpc import DMPC
from platoon_gym.dyn.linear_accel import LinearAccel
from platoon_gym.envs.platoon_env import PlatoonEnv
from platoon_gym.veh.vehicle import Vehicle
from platoon_gym.veh.virtual_leader import VirtualLeader


def create_env(
    dt: float,
    vehs: List[Vehicle],
    vl: VirtualLeader,
    reset_time: float,
    spacing_policy: str,
    distance_headways: List[List[float]],
    time_headways: Optional[List[List[float]]] = None,
    render_mode: Optional[str] = None,
) -> PlatoonEnv:
    plot_size = (6, 4)
    if sys.platform.startswith("linux"):
        dpi = 100
    elif sys.platform == "darwin":
        dpi = 50
    else:
        exit("Unsupported OS found: {}".format(sys.platform))
    if spacing_policy.lower() == "cdh":
        desired_distances = [d[0] for d in distance_headways]
        time_headways_pred = [0.0 for _ in desired_distances]
    elif spacing_policy.lower() == "cth":
        time_headways_pred = [t[0] for t in time_headways]
        desired_distances = [d[0] for d in distance_headways]
    env_args = {
        "headway": spacing_policy,
        "desired distance": desired_distances,
        "time headway": time_headways_pred,
        "dt": dt,
        "plot history length": 200,
        "plot size": plot_size,
        "render dpi": dpi,
        "reset time": reset_time,
    }
    env = PlatoonEnv(vehs, vl, env_args, render_mode=render_mode)
    return env


def create_dynamics(
    rng: Optional[np.random.Generator],
    N: int,
    dt: float,
    tau_lims: Tuple[float, float],
    x_lims: Optional[np.ndarray] = None,
    u_lims: Optional[np.ndarray] = None,
) -> List[LinearAccel]:
    taus = rng.uniform(tau_lims[0], tau_lims[1], N)
    return [LinearAccel(dt, x_lims, u_lims, tau, full_obs=False) for tau in taus]


def create_controller(
    H: int,
    dyn: LinearAccel,
    qself: float,
    qneighbors: List[float],
    r: float,
    distance_headways: Optional[List[float]] = None,
    time_headways: Optional[List[float]] = None,
    output_norm: str = "quadratic",
    input_norm: str = "quadratic",
) -> DMPC:
    assert distance_headways is not None or time_headways is not None
    return DMPC(
        H=H,
        Q=qself * np.eye(dyn.p),
        Q_neighbors=[q * np.eye(dyn.p) for q in qneighbors],
        R=r * np.eye(dyn.m),
        A=dyn.Ad,
        B=dyn.Bd,
        C=dyn.C,
        x_lims=dyn.x_lims,
        u_lims=dyn.u_lims,
        u_slew_rate=np.array([float("inf")]),
        distance_headways=distance_headways,
        time_headways=time_headways,
        terminal_constraint=True,
        output_norm=output_norm,
        input_norm=input_norm,
    )


def create_headways(
    N: int,
    spacing_policy: str,
    topology: str,
    desired_distance: float,
    time_headway: float,
) -> Tuple[List[float], List[float]]:
    distance_headways = []
    time_headways = []
    for i in range(N):
        if spacing_policy.lower() == "cdh":
            if topology.lower() == "pf":
                time_headways.append([0.0])
                if i == 0:
                    distance_headways.append([0.0])
                else:
                    distance_headways.append([desired_distance])
            elif topology.lower() == "bd":
                if i == 0:
                    distance_headways.append([0.0, -desired_distance])
                    time_headways.append([0.0, 0.0])
                elif i < N - 1:
                    distance_headways.append([desired_distance, -desired_distance])
                    time_headways.append([0.0, 0.0])
                else:
                    distance_headways.append([desired_distance])
                    time_headways.append([0.0])
        elif spacing_policy.lower() == "cth":
            if topology.lower() == "pf":
                if i == 0:
                    distance_headways.append([0.0])
                    time_headways.append([0.0])
                else:
                    distance_headways.append([desired_distance])
                    time_headways.append([time_headway])
            elif topology.lower() == "bd":
                if i == 0:
                    distance_headways.append([0.0, -desired_distance])
                    time_headways.append([0.0, -time_headway])
                elif i < N - 1:
                    distance_headways.append([desired_distance, -desired_distance])
                    time_headways.append([time_headway, -time_headway])
                else:
                    distance_headways.append([desired_distance])
                    time_headways.append([time_headway])
    return distance_headways, time_headways


def plot_data(
    spacing_policy: str,
    topology: str,
    output_norm: str,
    time: np.ndarray,
    vl_data: np.ndarray,
    veh_state_data: np.ndarray,
    input_data: np.ndarray,
    leader_error_data: np.ndarray,
    predecessor_error_data: np.ndarray,
):
    N = veh_state_data.shape[1]
    plot_inds = np.linspace(0, N - 1, num=min(N, 10)).round().astype(int)

    fig, ax = plt.subplots(3, 4, sharex=True, figsize=(12, 8))
    fig.subplots_adjust(0.06, 0.07, 0.85, 0.91, 0.33, 0.2)
    fig.suptitle(
        rf"Platoon trajectory and error: {spacing_policy}, {topology}, ${output_norm}$"
    )
    ax[0, 0].set_title("position [m]")
    ax[0, 1].set_title("velocity [m/s]")
    ax[0, 2].set_title(r"acceleration [m/s$^2$]")
    ax[0, 3].set_title(r"input [m/s$^2$]")
    ax[1, 0].set_title("leader pos err [m]")
    ax[1, 1].set_title("leader vel err [m/s]")
    ax[1, 2].set_title(r"leader acc err [m/s$^2$]")
    ax[1, 3].axis("off")
    ax[2, 0].set_title("predecessor pos err [m]")
    ax[2, 1].set_title("predecessor vel err [m/s]")
    ax[2, 2].set_title(r"predecessor acc err [m/s$^2$]")
    ax[2, 3].axis("off")

    # input data
    ax[0, 3].plot(time[:-1], vl_data[:-1, 2], label="virtual leader", c="k")
    for i in plot_inds:
        ax[0, 3].plot(time[:-1], input_data[:, i], label=f"vehicle {i+1}")

    # state and error data
    for n in range(vl_data.shape[1]):
        ax[0, n].plot(time, vl_data[:, n], label="virtual leader", c="k")
        for i in plot_inds:
            ax[0, n].plot(time, veh_state_data[:, i, n], label=f"vehicle {i+1}")
            ax[1, n].plot(time, leader_error_data[:, i, n], label=f"vehicle {i+1}")
            ax[2, n].plot(time, predecessor_error_data[:, i, n], label=f"vehicle {i+1}")

    ax[0, 3].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[1, 2].legend(loc="center left", bbox_to_anchor=(1.1, 0.5))
    ax[2, 2].legend(loc="center left", bbox_to_anchor=(1.1, 0.5))

    ax[2, 0].set_xlabel("time [s]")
    ax[2, 1].set_xlabel("time [s]")
    ax[2, 2].set_xlabel("time [s]")
    ax[0, 3].set_xlabel("time [s]")

    for a in ax.flatten():
        a.grid()

    return fig, ax
