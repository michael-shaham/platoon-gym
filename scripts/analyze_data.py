import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from platoon_gym.utils.general_utils import get_project_dir


def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze platoon simulation logs.")

    # Environment arguments
    parser.add_argument("--controller_types", type=str, nargs="+", default=["DMPC", "CCMPC", "LFBK"], help="Controller types.")
    parser.add_argument("--max_trajectories", type=int, default=100, help="Maximum simulation steps.")
    parser.add_argument("--num_vehicles", type=int, default=50, help="Number of vehicles in the platoon.")
    parser.add_argument("--distance_headway", type=float, default=4.0, help="Desired distance headway between vehicles.")
    parser.add_argument("--time_headway", type=float, default=1.0, help="Desired time headway between vehicles.")
    parser.add_argument("--use_truncated_gaussian", action="store_true", help="Use truncated Gaussian distribution.")
    parser.add_argument("--use_quantile", action="store_true", help="Use quantile regression.")
    parser.add_argument("--qms", type=float, default=0.0, help="Move suppression weight for CCMPC.")

    return parser.parse_args()


def analyze_platoon_metrics(
    ctrl_types: list[str], 
    num_vehicles: int, 
    max_trajectories: int, 
    distance_headway: float, 
    time_headway: float,
    use_truncated_gaussian: bool,
    use_quantile: bool,
    qms: float
):
    data_dir = get_project_dir() / "data"

    model_type = "truncated_gaussian" if use_truncated_gaussian else "gaussian"
    model_type = "quantile" if use_quantile else model_type  # overrides

    fig, ax = plt.subplots(
        3, len(ctrl_types), figsize=(8, 4 * len(ctrl_types)), sharex=True, sharey='row'
    )
    ax[0, 0].set_ylabel("Position RMSE (m)")
    ax[1, 0].set_ylabel("Velocity RMSE (m/s)")
    ax[2, 0].set_ylabel("Acceleration (m/s²)")
    for i in range(len(ctrl_types)):
        ax[2, i].set_xlabel("Vehicle Index")

    for i, ctrl in enumerate(ctrl_types):
        log_filename = ctrl.lower()
        log_filename += f"_{model_type}_{max_trajectories}_trajectories"
        log_filename += f"_{num_vehicles}_vehicles"
        log_filename += f"_{time_headway:.2f}s_{distance_headway:.2f}m_headways"
        log_filename += f"_{qms:.2f}_qms.hdf5"
        log_filename = data_dir / log_filename
        if not log_filename.exists():
            raise FileNotFoundError(f"Log file {log_filename} not found.")

        with h5py.File(log_filename, "r") as log_file:
            N = log_file.attrs["num_vehicles"]
            dh = log_file.attrs["distance_headway"]
            th = log_file.attrs["time_headway"]
            
            position_errors = []
            velocity_errors = []
            acceleration_stats = []
            n_collisions = 0

            for traj_key in log_file.keys():
                traj_group = log_file[traj_key]
                veh_states = np.array(traj_group["veh_states"])  # (N, T, 3)
                vl_states = np.expand_dims(np.array(traj_group["vl_states"]), 0)  # (1, T, 3)
                n_collisions += int(traj_group.attrs["collided"])
                
                states = np.concatenate([vl_states, veh_states], axis=0)  # (N + 1, T, 3)

                positions = states[:, :, 0]
                velocities = states[:, :, 1]
                accelerations = states[:, :, 2]

                desired_gaps = dh + th * velocities
                actual_gaps = positions[:-1] - positions[1:]
                position_error = actual_gaps - desired_gaps[1:]  # Ignore leader
                velocity_error = np.diff(velocities, axis=0)  # Compute velocity differences
                
                # Compute errors for first vehicle relative to virtual leader
                position_rmse = np.sqrt(np.mean(position_error, axis=1)**2)
                velocity_rmse = np.sqrt(np.mean(velocity_error, axis=1)**2)

                acceleration_means = np.mean(accelerations[1:], axis=1)  # Ignore leader
                acceleration_stds = np.std(accelerations[1:], axis=1)  # Ignore leader

                position_errors.append(position_rmse)
                velocity_errors.append(velocity_rmse)
                acceleration_stats.append((acceleration_means, acceleration_stds))

        results_df = pd.DataFrame({
            "Vehicle": np.arange(1, N + 1),
            "Position RMSE": np.mean(position_errors, axis=0),
            "Velocity RMSE": np.mean(velocity_errors, axis=0),
            "Acceleration Mean": np.mean([a[0] for a in acceleration_stats], axis=0),
            "Acceleration Std": np.mean([a[1] for a in acceleration_stats], axis=0)
        })
        
        print(results_df)

        ax[0, i].set_title(f"{ctrl.upper()}, {n_collisions} collisions")
        ax[0, i].plot(results_df["Vehicle"], results_df["Position RMSE"])
        ax[1, i].plot(results_df["Vehicle"], results_df["Velocity RMSE"])
        ax[2, i].errorbar(
            results_df["Vehicle"], 
            results_df["Acceleration Mean"], 
            yerr=results_df["Acceleration Std"], 
            fmt='o', 
            capsize=5
        )
    for a in ax.flat:
        a.grid()
    plt.show()


if __name__ == "__main__":
    args = parse_arguments()
    ctrl_types = [ctrl.lower() for ctrl in args.controller_types]
    analyze_platoon_metrics(
        ctrl_types, 
        args.num_vehicles, 
        args.max_trajectories, 
        args.distance_headway, 
        args.time_headway,
        args.use_truncated_gaussian,
        args.use_quantile,
        args.qms
    )
