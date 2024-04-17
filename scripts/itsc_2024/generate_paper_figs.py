import argparse
import matplotlib.pyplot as plt
import os
import numpy as np

from platoon_gym.utils.utils import get_project_root

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['pdf.fonttype'] = 42

title_size = 9
subtitle_size = 8
axes_size = 8
legend_title_size = 8
legend_font_size = 7
tick_label_size = 7
linewidth = 0.8
linestyle = "solid"
scatter_size = 2

if __name__ == "__main__":
    project_dir = os.path.join(get_project_root(), "scripts", "itsc_2024")
    data_dir = os.path.join(project_dir, "data")
    figure_dir = os.path.join(project_dir, "figures")
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("-nv", "--num-vehicles", type=int, default=50)
    parser.add_argument("-dd", "--desired_distance", type=float, default=5.0)
    parser.add_argument("-th", "--time_headway", type=float, default=0.2)
    args = parser.parse_args()
    N = args.num_vehicles
    desired_distance = args.desired_distance
    time_headway = args.time_headway

    # create velocity trajectory plot comparing CTH and CDH
    topology = "PF"
    norm = "l1"
    cth_trial_name = f"dmpc_{N}_vehs_CTH_{time_headway}" + f"_{topology}_{norm}"
    cdh_trial_name = f"dmpc_{N}_vehs_CDH_{desired_distance}" + f"_{topology}_{norm}"
    cth_data_file = os.path.join(data_dir, cth_trial_name)
    cdh_data_file = os.path.join(data_dir, cdh_trial_name)

    cth_time_data = np.load(cth_data_file + "_time.npy")
    cth_vl_data = np.load(cth_data_file + "_vl_data.npy")
    cth_veh_state_data = np.load(cth_data_file + "_veh_state_data.npy")

    cdh_time_data = np.load(cdh_data_file + "_time.npy")
    cdh_vl_data = np.load(cdh_data_file + "_vl_data.npy")
    cdh_veh_state_data = np.load(cdh_data_file + "_veh_state_data.npy")

    # plot the velocity trajectory
    # state data dims are (time, vehicle, state)
    plot_inds = [1, 10, 20, 30, 40, 50]
    fig, ax = plt.subplots(2, 2, figsize=(3.3, 2.0), sharex=True, sharey='row')
    fig.subplots_adjust(0.12, 0.19, 0.84, 0.91, 0.1, 0.12)
    ax[0, 0].set_title("CTH", fontsize=subtitle_size)
    ax[0, 1].set_title("CDH", fontsize=subtitle_size)
    ax[0, 0].plot(
        cth_time_data,
        cth_vl_data[:, 1],
        linewidth=linewidth,
        linestyle=linestyle,
        label="vl",
        c="k",
    )
    ax[0, 1].plot(
        cdh_time_data,
        cdh_vl_data[:, 1],
        linewidth=linewidth,
        linestyle=linestyle,
        label="vl",
        c="k",
    )
    for k, i in enumerate(plot_inds):
        ax[0, 0].plot(
            cth_time_data,
            cth_veh_state_data[:, i - 1, 1],
            linewidth=linewidth,
            linestyle=linestyle,
            label=f"{i}",
            color=f"C{k}",
        )
        ax[0, 1].plot(
            cdh_time_data,
            cdh_veh_state_data[:, i - 1, 1],
            linewidth=linewidth,
            linestyle=linestyle,
            label=f"{i}",
            color=f"C{k}",
        )
    for a in ax[0]:
        a.grid()
        a.tick_params(axis="both", labelsize=tick_label_size)
        a.set_xticks([0, 4, 8, 12])
        a.set_yticks([20, 22, 24])
    ax[0, 0].set_ylabel("velocity [m/s]", fontsize=axes_size)
    ax[0, 1].legend(
        bbox_to_anchor=(1.0, 0.0),
        loc="center left",
        title="vehicle",
        title_fontsize=legend_title_size,
        fontsize=legend_font_size,
    )
    # plot errors
    cdh_pred_error_data = np.load(cdh_data_file + "_predecessor_error_data.npy")
    cth_pred_error_data = np.load(cth_data_file + "_predecessor_error_data.npy")
    for k, i in enumerate(plot_inds):
        ax[1, 0].plot(
            cth_time_data,
            cth_pred_error_data[:, i - 1, 0],
            linewidth=linewidth,
            linestyle=linestyle,
            label=f"{i}",
            color=f"C{k}",
        )
        ax[1, 1].plot(
            cdh_time_data,
            cdh_pred_error_data[:, i - 1, 1],
            linewidth=linewidth,
            linestyle=linestyle,
            label=f"{i}",
            color=f"C{k}",
        )
    for a in ax[1]:
        a.set_xlabel("time [s]", fontsize=axes_size)
        a.grid()
        a.tick_params(axis="both", labelsize=tick_label_size)
        a.set_xticks([0, 4, 8, 12])
    ax[1, 0].set_ylabel("spacing error [m]", fontsize=axes_size)
    fig_file = os.path.join(figure_dir, "itsc_platoon_velocity_spacing_errs.pdf")
    plt.savefig(fig_file, bbox_inches="tight", pad_inches=0)

    # plot max errors
    fig, ax = plt.subplots(1, 2, figsize=(3.3, 1.8), sharex=True)
    fig.subplots_adjust(0.095, 0.42, 0.985, 0.9, 0.2, 0.2)
    ax[0].set_title("Max Spacing Error [m]", fontsize=subtitle_size)
    ax[1].set_title("Max Velocity Error [m/s]", fontsize=subtitle_size)
    spacing_policies = ["CTH", "CDH"]
    topologies = ["PF", "BD"]
    norms = ["l1", "l2", "quadratic"]
    labels = []
    max_pos_errs = []
    max_vel_errs = []
    for sp in spacing_policies:
        for top in topologies:
            for norm in norms:
                if sp == "CDH":
                    trial_name = (
                        f"dmpc_{N}_vehs_{sp}_{desired_distance}" + f"_{top}_{norm}"
                    )
                elif sp == "CTH":
                    trial_name = f"dmpc_{N}_vehs_{sp}_{time_headway}" + f"_{top}_{norm}"
                if norm == "l1":
                    norm_str = r"$\ell_1$"
                elif norm == "l2":
                    norm_str = r"$\ell_2$"
                elif norm == "quadratic":
                    norm_str = "quad"
                data_file = os.path.join(data_dir, trial_name)
                predecessor_error_data = np.load(
                    data_file + "_predecessor_error_data.npy"
                )
                labels.append(f"{sp}, {top}, {norm_str}")
                max_pos_err = np.max(np.abs(predecessor_error_data[:, 1:, 0]), axis=0)
                max_vel_err = np.max(np.abs(predecessor_error_data[:, 1:, 1]), axis=0)
                max_pos_errs.append(max_pos_err)
                max_vel_errs.append(max_vel_err)

    ax[0].boxplot(
        max_pos_errs,
        vert=True,
        labels=labels,
        patch_artist=True,
        showfliers=False,
    )
    ax[1].boxplot(
        max_vel_errs,
        vert=True,
        labels=labels,
        patch_artist=True,
        showfliers=False,
    )
    for a in ax:
        a.tick_params(axis="x", labelrotation=90)
        a.tick_params(axis="both", labelsize=tick_label_size)
    fig_file = os.path.join(figure_dir, "itsc_platoon_max_spacing_error_boxplot.pdf")
    plt.savefig(fig_file, bbox_inches="tight", pad_inches=0)

    # plot the max position errors seen by vehicle
    fig, ax = plt.subplots(1, 2, figsize=(3.3, 1.5), sharex=True, sharey=True)
    fig.subplots_adjust(0.12, 0.26, 0.76, 0.87, 0.1, 0.2)
    ax[0].set_title("CTH", fontsize=subtitle_size)
    ax[1].set_title("CDH", fontsize=subtitle_size)
    spacing_policies = ["CTH", "CDH"]
    topologies = ["PF", "BD"]
    norms = ["l1", "l2", "quadratic"]
    cdh_labels = []
    cth_labels = []
    cdh_max_pos_errs = []
    cth_max_pos_errs = []
    for sp in spacing_policies:
        for top in topologies:
            for norm in norms:
                if sp == "CDH":
                    trial_name = (
                        f"dmpc_{N}_vehs_{sp}_{desired_distance}" + f"_{top}_{norm}"
                    )
                elif sp == "CTH":
                    trial_name = f"dmpc_{N}_vehs_{sp}_{time_headway}" + f"_{top}_{norm}"
                if norm == "l1":
                    norm_str = r"$\ell_1$"
                elif norm == "l2":
                    norm_str = r"$\ell_2$"
                elif norm == "quadratic":
                    norm_str = "quad"
                data_file = os.path.join(data_dir, trial_name)
                predecessor_error_data = np.load(
                    data_file + "_predecessor_error_data.npy"
                )
                max_pos_err = np.max(np.abs(predecessor_error_data[:, :, 0]), axis=0)
                if sp == "CDH":
                    cdh_labels.append(f"{top}, {norm_str}")
                    cdh_max_pos_errs.append(max_pos_err)
                elif sp == "CTH":
                    cth_labels.append(f"{top}, {norm_str}")
                    cth_max_pos_errs.append(max_pos_err)

    for i, label in enumerate(cth_labels):
        ax[0].plot(range(1, N + 1), cth_max_pos_errs[i], label=label)
    for i, label in enumerate(cdh_labels):
        ax[1].plot(range(1, N + 1), cdh_max_pos_errs[i], label=label)
    ax[0].set_ylabel("max spacing error [m]", fontsize=axes_size)
    for a in ax:
        a.set_xlabel("vehicle", fontsize=axes_size)
        a.tick_params(axis="both", labelsize=tick_label_size)
        a.set_xticks([1, 10, 20, 30, 40, 50])
    ax[1].legend(
        bbox_to_anchor=(1.0, 0.4), loc="center left", fontsize=legend_font_size
    )

    plt.show()