import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

from platoon_gym.utils.general_utils import get_project_dir

fig_dir = get_project_dir() / "figures"
fig_dir.mkdir(exist_ok=True)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze platoon attack simulation logs.")

    # Environment arguments
    parser.add_argument("--max_trajectories", type=int, default=10, help="Maximum simulation steps.")
    parser.add_argument("--num_vehicles", type=int, default=20, help="Number of vehicles in the platoon.")
    parser.add_argument("--distance_headway", type=float, default=4.0, help="Desired distance headway between vehicles.")
    parser.add_argument("--time_headway", type=float, default=0.5, help="Desired time headway between vehicles.")
    parser.add_argument("--use_truncated_gaussian", action="store_true", help="Use truncated Gaussian distribution.")
    parser.add_argument("--use_quantile", action="store_true", help="Use quantile regression.")
    parser.add_argument("--qms_ccmpc", type=float, default=10.0, help="Move suppression weight for CCMPC.")
    parser.add_argument("--attack_frequency", type=float, default=0.0, help="Frequency of DoS attacks.")
    parser.add_argument("--attack_duration", type=float, default=0.0, help="Duration of DoS attacks.")
    parser.add_argument("--plot_trajectories", action="store_true", help="Plot vehicle trajectories.")

    return parser.parse_args()


def analyze_platoon_metrics(
    num_vehicles: int, 
    max_trajectories: int, 
    distance_headway: float, 
    time_headway: float,
    use_truncated_gaussian: bool,
    use_quantile: bool,
    attack_frequency: float,
    attack_duration: float,
    qms_ccmpc: float,
    plot_trajectories: bool,
    figure_title: str = ""
):
    data_dir = get_project_dir() / "data"

    model_type = "truncated_gaussian" if use_truncated_gaussian else "gaussian"
    model_type = "quantile" if use_quantile else model_type  # overrides

    log_filename = "attack_scenario"
    log_filename += f"_{model_type}_{max_trajectories}_trajectories"
    log_filename += f"_{num_vehicles}_vehicles"
    log_filename += f"_{time_headway:.2f}s_{distance_headway:.2f}m_headways"
    log_filename += f"_{qms_ccmpc:.2f}_qms"
    log_filename += f"_{attack_frequency:.2f}_af_{attack_duration:.1f}s_ad.hdf5"
    fig_name = log_filename[:-5]
    log_filename = data_dir / log_filename

    if not log_filename.exists():
        print(f"Log file {log_filename} not found.")
        exit()

    with h5py.File(log_filename, "r") as log_file:
        N = log_file.attrs["num_vehicles"]
        dh = log_file.attrs["distance_headway"]
        th = log_file.attrs["time_headway"]
        
        position_errors = []
        velocity_errors = []
        acceleration_data = []
        n_collisions = 0

        for traj_key in log_file.keys():

            traj_group = log_file[traj_key]
            veh_states = np.array(traj_group["veh_states"])  # (N, T, 3)
            veh_controls = np.array(traj_group["veh_controls"])  # (N, T, 1)
            attack_status = np.array(traj_group["veh_attacked"])  # (N, T)
            n_collisions += int(traj_group.attrs["collided"])

            # prepend vl data to veh data
            vl_states = np.expand_dims(np.array(traj_group["vl_states"]), 0)  # (1, T, 3)
            vl_states[:, 1:, 2] = (vl_states[:, 1:, 1] - vl_states[:, :-1, 1]) / 0.1  # compute vl acceleration

            veh_attacked = np.sum(attack_status, axis=1) > 0
            attacked_inds = np.where(veh_attacked)[0]
            post_attacked_inds = attacked_inds[:] + 1
            plot_inds = sorted([0] + attacked_inds.tolist() + post_attacked_inds.tolist())
            plot_inds = [0] + [1 + i for i in plot_inds]  # to incorporate vl
            plot_inds = plot_inds[:10]

            plot_inds = np.linspace(0, N, 10, endpoint=True).astype(int)
            
            states = np.concatenate([vl_states, veh_states], axis=0)  # (N + 1, T, 3)
            t_plot = np.arange(states.shape[1]) * 0.1
            T = len(t_plot)

            vl_position_error = np.zeros((1, T))
            vl_velocity_error = np.zeros((1, T))
            vl_controls = np.zeros((1, T - 1))

            positions = states[:, :, 0]
            velocities = states[:, :, 1]
            accelerations = states[:, :, 2]
            controls = veh_controls[:, :, 0]

            desired_gaps = dh + th * velocities[1:]
            actual_gaps = positions[:-1] - positions[1:]  # pred position - actual position
            position_error = actual_gaps - desired_gaps
            velocity_error = velocities[:-1] - velocities[1:]  # pred velocity - actual velocity

            # prepend vl data to errors and controls
            controls = np.concatenate([vl_controls, controls], axis=0)
            position_error = np.concatenate([vl_position_error, position_error], axis=0)
            velocity_error = np.concatenate([vl_velocity_error, velocity_error], axis=0)

            # plot the traj
            if plot_trajectories:
                fig, ax = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
                
                if attacked_inds.size > 0:
                    fig.suptitle(f"DoS Attack on {attacked_inds.tolist()}")
                else:
                    fig.suptitle("No DoS Attack")

                ax[0, 0].plot(t_plot, positions[plot_inds].T)
                ax[0, 1].plot(t_plot, position_error[plot_inds].T)
                ax[1, 0].plot(t_plot, velocities[plot_inds].T)
                ax[1, 1].plot(t_plot, velocity_error[plot_inds].T)
                ax[2, 0].plot(t_plot, accelerations[plot_inds].T)
                ax[2, 1].plot(t_plot[:-1], controls[plot_inds].T)

                ax[0, 0].set_title("Position (m)")
                ax[0, 1].set_title("Position Error (m)")
                ax[1, 0].set_title("Velocity (m/s)")
                ax[1, 1].set_title("Velocity Error (m/s)")
                ax[2, 0].set_title("Acceleration (m/s²)")
                ax[2, 1].set_title("Control input (m/s²)")
                ax[2, 0].set_xlabel("Time (s)")
                ax[2, 1].set_xlabel("Time (s)")

                # shade region where attack is active
                attack_region = np.where(np.sum(attack_status, axis=0) > 0)[0]
                min_pos, max_pos = ax[0, 0].get_ylim()
                min_vel, max_vel = ax[1, 0].get_ylim()
                min_acc, max_acc = ax[2, 0].get_ylim()
                min_perr, max_perr = ax[0, 1].get_ylim()
                min_verr, max_verr = ax[1, 1].get_ylim()
                min_ctrl, max_ctrl = ax[2, 1].get_ylim()
                ax[0, 0].fill_between(t_plot[attack_region], min_pos, max_pos, color="gray", alpha=0.2)
                ax[1, 0].fill_between(t_plot[attack_region], min_vel, max_vel, color="gray", alpha=0.2)
                ax[2, 0].fill_between(t_plot[attack_region], min_acc, max_acc, color="gray", alpha=0.2)
                ax[0, 1].fill_between(t_plot[attack_region], min_perr, max_perr, color="gray", alpha=0.2)
                ax[1, 1].fill_between(t_plot[attack_region], min_verr, max_verr, color="gray", alpha=0.2)
                ax[2, 1].fill_between(t_plot[attack_region], min_ctrl, max_ctrl, color="gray", alpha=0.2)

                ax[1, 1].legend(plot_inds, bbox_to_anchor=(1.05, 0.5), loc="center left")
                for a in ax.flat:
                    a.grid()
                plt.tight_layout()
                plt.savefig(fig_dir / (fig_name + f"_traj_{traj_key}.png"))
                plt.show()

            position_errors.append(position_error)
            velocity_errors.append(velocity_error)
            acceleration_data.append(accelerations)  # ignore vl
        
        position_errors = np.concatenate(position_errors, axis=1)
        velocity_errors = np.concatenate(velocity_errors, axis=1)
        acceleration_data = np.concatenate(acceleration_data, axis=1)

    fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    if attacked_inds.size > 0:
        fig.suptitle(f"DoS Attack on {attacked_inds.tolist()}")
    else:
        fig.suptitle("No DoS Attack")
    ax[0].set_ylabel("Position Error (m)")
    ax[1].set_ylabel("Velocity Error (m/s)")
    ax[2].set_ylabel("Acceleration (m/s²)")
    ax[2].set_xlabel("Vehicle Index")

    labels = np.arange(N + 1)
    ax[0].boxplot(position_errors.T, tick_labels=labels, flierprops={'markersize': 1})
    ax[1].boxplot(velocity_errors.T, tick_labels=labels, flierprops={'markersize': 1})
    ax[2].boxplot(acceleration_data.T, tick_labels=labels, flierprops={'markersize': 1})

    # Change the color of attacked tick labels
    for a in ax:
        for tick, label in zip(a.get_xticks(), a.get_xticklabels()):
            if int(label.get_text()) in attacked_inds:
                label.set_color("red")

    for a in ax.flat:
        a.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / (fig_name + "_boxplot.png"))
    plt.show()


if __name__ == "__main__":
    args = parse_arguments()
    analyze_platoon_metrics(
        args.num_vehicles, 
        args.max_trajectories, 
        args.distance_headway, 
        args.time_headway,
        args.use_truncated_gaussian,
        args.use_quantile,
        args.attack_frequency,
        args.attack_duration,
        args.qms_ccmpc,
        args.plot_trajectories
    )
