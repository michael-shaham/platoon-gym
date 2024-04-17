import argparse
import copy
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from tqdm import tqdm
from typing import List, Optional, Tuple

from platoon_gym.utils.utils import get_project_root
from platoon_gym.veh.vehicle import Vehicle
from platoon_gym.veh.virtual_leader import VirtualLeader

from utils import (
    create_dynamics,
    create_controller,
    create_headways,
    create_env,
    plot_data,
)


def evaluate_dmpc(
    N: int,
    output_norm: str,
    input_norm: str,
    topology: str,
    spacing_policy: str,
    qself: float = 1.0,
    r: float = 1.0,
    desired_distance: float = 5.0,
    time_headway: float = 0.25,
    render_mode: Optional[str] = None,
) -> Tuple[
    List[float],
    List[np.ndarray],
    List[List[np.ndarray]],
    List[List[np.ndarray]],
    List[List[np.ndarray]],
    List[List[np.ndarray]],
    List[List[float]],
]:
    rng = np.random.default_rng(0)

    # set up vehicle dynamics
    dt = 0.1
    tau_lims = (0.25, 0.9)
    x_lims = np.array([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])
    u_lims = np.array([[-3.0, 3.0]])
    dyns = create_dynamics(rng, N, dt, tau_lims, x_lims, u_lims)

    # set up DMPC controllers
    H = 60
    assert spacing_policy.lower() in ["cdh", "cth"]
    assert topology.lower() in ["pf", "bd"]
    distance_headways, time_headways = create_headways(
        N, spacing_policy, topology, desired_distance, time_headway
    )
    qself = [qself for _ in range(N)]
    if topology.lower() == "pf":
        qneighbors = [[qself[i]] for i in range(N)]
    elif topology.lower() == "bd":
        qneighbors = [[qself[i] / 2.0, qself[i] / 2.0] for i in range(N - 1)] + [[qself[-1]]]
    ctrls = [
        create_controller(
            H,
            dyns[i],
            qself[i],
            qneighbors[i],
            r,
            distance_headways[i],
            time_headways[i],
            output_norm,
            input_norm,
        )
        for i in range(N)
    ]

    # set up virtual leader
    v0 = 20.0  # m/s
    vf = 22.0  # m/s
    vl_acc = 1.0  # m/s^2
    step_time = 0.0  # sec
    total_time = 14.0  # sec
    vl_traj_type = "velocity step"
    vl_traj_args = {
        "horizon": round(total_time / dt),
        "dt": dt,
        "step time": step_time,
        "step velocity": vf,
        "step acceleration": vl_acc,
    }
    vl = VirtualLeader(vl_traj_type, vl_traj_args, velocity=v0)
    vl.reset()

    # set up vehicles
    vehs = [Vehicle(dyns[0], 0.0, v0, 0.0)]
    vehs += [
        Vehicle(
            dyns[i], -i * (distance_headways[i][0] + time_headways[i][0] * v0), v0, 0.0
        )
        for i in range(1, N)
    ]

    # create env
    env = create_env(
        dt,
        vehs,
        vl,
        total_time,
        spacing_policy,
        distance_headways,
        time_headways,
        render_mode,
    )

    # simulate env
    obs, env_info = env.reset()
    veh_states = env_info["vehicle states"]

    # log data
    timesteps = [env.time]
    if time_headways is None:
        time_headways = [[0.0] for _ in range(N)]
    if distance_headways is None:
        distance_headways = [[0.0] for _ in range(N)]
    vl_data = [env.vl.state]
    veh_state_data = [veh_states]
    input_data = []
    leader_error_data = []
    predecessor_error_data = []
    solve_times = []

    def append_error_data(vl_state: np.ndarray, veh_states: List[np.ndarray]):
        leader_errors = []
        predecessor_errors = []
        for i in range(N):
            # predecessor error
            d_des = time_headways[i][0] * veh_states[i][1] + distance_headways[i][0]
            if i == 0:
                predecessor_error = vl_state - veh_states[i]
            else:
                predecessor_error = veh_states[i - 1] - veh_states[i]
            predecessor_error[0] -= d_des
            # leader error
            leader_error = vl_state - veh_states[i]
            for j in range(i + 1):
                leader_error[0] -= (
                    time_headways[j][0] * veh_states[j][1] + distance_headways[j][0]
                )
            leader_errors.append(leader_error)
            predecessor_errors.append(predecessor_error)
        leader_error_data.append(leader_errors)
        predecessor_error_data.append(predecessor_errors)

    append_error_data(env.vl.state, veh_states)

    # initialize assumed states
    prev_assumed_states = []
    for i in range(N):
        uref = np.zeros((H, dyns[i].m))
        xa = ctrls[i].initialize_assumed_trajectory(veh_states[i], uref)
        prev_assumed_states.append(xa)

    # simulate platoon
    for _ in tqdm(range(round(total_time / dt))):
        solve_times.append([])
        env.render()
        vl_plan = env_info["virtual leader plan"][:, : H + 1].T
        xopts = []
        actions = []
        assumed_states = []
        for i in range(len(obs)):
            start_time = time.time()
            if i == 0:
                end_state = vl_plan[-1]
                y_neighbors = [vl_plan[:, : dyns[i].p].copy()]
                if topology.lower() == "bd" and i < N - 1:
                    y_neighbors.append(
                        prev_assumed_states[i + 1][:, : dyns[i].p].copy()
                    )
            else:
                end_state = np.concatenate(
                    (prev_assumed_states[i - 1][-1][: dyns[i].p].copy(), [0.0])
                )
                end_state[0] -= distance_headways[i][0]
                end_state[0] -= time_headways[i][0] * prev_assumed_states[i - 1][-1, 1]
                y_neighbors = [prev_assumed_states[i - 1][:, : dyns[i].p].copy()]
                if topology.lower() == "bd" and i < N - 1:
                    y_neighbors.append(
                        prev_assumed_states[i + 1][:, : dyns[i].p].copy()
                    )
            end_state[-1] = 0.0
            action, ctrl_info = ctrls[i].control(
                x0=veh_states[i],
                y_neighbors=y_neighbors,
                xa=prev_assumed_states[i],
                ua=np.zeros((H, dyns[i].m)),
                xf=end_state.copy(),
            )

            xopt = ctrl_info["x"]
            xopts.append(xopt)
            xa = np.zeros_like(xopt)
            xa[:-1] = xopt[1:]
            xa[-1] = dyns[i].forward(xopt[-1], np.zeros(dyns[i].m))
            assumed_states.append(xa)
            actions.append(action)
            end_time = time.time()
            solve_time = end_time - start_time
            solve_times[-1].append(solve_time)

        # step environment
        obs, _, _, trunc, env_info = env.step(action=actions)
        veh_states = env_info["vehicle states"]
        prev_assumed_states = copy.deepcopy(assumed_states)

        # log data
        timesteps.append(env.time)
        vl_data.append(env.vl.state)
        veh_state_data.append(veh_states)
        input_data.append(actions)
        append_error_data(env.vl.state, veh_states)

        if trunc:
            break

    env.close()
    return (
        timesteps,
        vl_data,
        veh_state_data,
        input_data,
        leader_error_data,
        predecessor_error_data,
        solve_times,
    )


if __name__ == "__main__":
    project_dir = os.path.join(get_project_root(), "scripts", "itsc_2024")
    data_dir = os.path.join(project_dir, "data")
    figure_dir = os.path.join(project_dir, "figures")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("-nv", "--num-vehicles", type=int, default=10)
    parser.add_argument("-dd", "--desired-distance", type=float, default=5.0)
    parser.add_argument("-sd", "--safety-distance", type=float, default=1.0)
    parser.add_argument("-th", "--time-headway", type=float, default=0.2)
    parser.add_argument("-rm", "--render-mode", type=int, default=1)
    args = parser.parse_args()
    N = args.num_vehicles
    desired_distance = args.desired_distance
    safety_distance = args.safety_distance
    time_headway = args.time_headway
    render_mode = bool(args.render_mode)
    render_mode = "plot" if render_mode else None
    input_norm = "quadratic"
    spacing_policies = ["CTH", "CDH"]
    topologies = ["PF", "BD"]
    norms = ["quadratic", "l2", "l1"]
    for spacing_policy in spacing_policies:
        for topology in topologies:
            for output_norm in norms:
                print(f"Evaluating {spacing_policy}, {topology}, {output_norm}")
                d_des = desired_distance if spacing_policy == "CDH" else safety_distance
                data = evaluate_dmpc(
                    N,
                    output_norm,
                    input_norm,
                    topology,
                    spacing_policy,
                    desired_distance=d_des,
                    time_headway=time_headway,
                    render_mode=render_mode,
                )

                (
                    timesteps,
                    vl_data,
                    veh_state_data,
                    input_data,
                    leader_error_data,
                    predecessor_error_data,
                    solve_times,
                ) = data

                timesteps = np.array(timesteps)
                vl_data = np.array(vl_data)
                veh_state_data = np.array(veh_state_data)
                input_data = np.array(input_data).squeeze(-1)
                leader_error_data = np.array(leader_error_data)
                predecessor_error_data = np.array(predecessor_error_data)
                solve_times = np.array(solve_times)
                fig, ax = plot_data(
                    spacing_policy,
                    topology,
                    output_norm,
                    timesteps,
                    vl_data,
                    veh_state_data,
                    input_data,
                    leader_error_data,
                    predecessor_error_data,
                )

                if spacing_policy.lower() == "cdh":
                    trial_name = (
                        f"dmpc_{N}_vehs_{spacing_policy}_{desired_distance}"
                        + f"_{topology}_{output_norm}"
                    )
                elif spacing_policy.lower() == "cth":
                    trial_name = (
                        f"dmpc_{N}_vehs_{spacing_policy}_{time_headway}"
                        + f"_{topology}_{output_norm}"
                    )
                fig_file = os.path.join(figure_dir, trial_name) + ".pdf"
                plt.savefig(fig_file, bbox_inches="tight")

                data_file = os.path.join(data_dir, trial_name)
                np.save(data_file + "_time.npy", timesteps)
                np.save(data_file + "_vl_data.npy", vl_data)
                np.save(data_file + "_veh_state_data.npy", veh_state_data)
                np.save(data_file + "_input_data.npy", input_data)
                np.save(data_file + "_leader_error_data.npy", leader_error_data)
                np.save(data_file + "_predecessor_error_data.npy", predecessor_error_data)
                np.save(data_file + "_solve_times.npy", solve_times)

                # calculate max predecessor spacing error per vehicle
                # max_pos_errs = np.max(np.abs(predecessor_error_data[:, :, 0]), axis=0)
                # plt.figure()
                # plt.plot(range(1, N + 1), max_pos_errs)
                # plt.show()
