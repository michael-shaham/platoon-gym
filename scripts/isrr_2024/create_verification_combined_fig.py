import matplotlib.pyplot as plt
import numpy as np
import os

from platoon_gym.utils.utils import get_project_root

title_size = 9
subtitle_size = 8
axes_size = 8
legend_title_size = 8
legend_font_size = 7
tick_label_size = 7
linewidth = 0.8
linestyle = "solid"
alphas = [0.7, 0.8, 0.8, 0.8]
alpha = 0.8
scatter_size = 6
plt.rcParams["font.family"] = "Times New Roman"

if __name__ == "__main__":
    save_dir = os.path.join(get_project_root(), "scripts", "iros_2024")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_vehs = [2, 3]
    fig, ax = plt.subplots(1, 1, figsize=(2.95, 1.75))
    fig.suptitle("Maximum Lyapunov condition violation", fontsize=title_size)
    fig.subplots_adjust(0.175, 0.21, 0.98, 0.88, 0.2, 0.2)
    for k, n in enumerate(n_vehs):
        data_file = os.path.join(save_dir, f"verification_losses_{n}.csv")
        data = np.loadtxt(data_file)
        ax.plot(
            data[:, 1],
            label=f"positivity, N={n}",
            color=f"C{k}",
            linestyle="dotted",
            linewidth=linewidth,
        )
        ax.plot(
            data[:, 2],
            label=f"decreasing, N={n}",
            color=f"C{k}",
            linewidth=linewidth,
            linestyle="dashed",
        )
    ax.tick_params(labelsize=tick_label_size)
    ax.set_xlabel("episode", fontsize=axes_size)
    ax.set_ylabel("violation", fontsize=axes_size)
    ax.legend(fontsize=legend_font_size)
    save_file = os.path.join(save_dir, "figures" "verification_combined.pdf")
    plt.savefig(save_file, bbox_inches="tight")
    plt.show()
