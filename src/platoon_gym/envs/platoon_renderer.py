from datetime import datetime
import os
import sys

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import numpy as np
import pygame

from platoon_gym.veh.vehicle import Vehicle
from platoon_gym.veh.virtual_leader import VirtualLeader

os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (0, 0)


class PlatoonRenderer:
    """
    Renderer for the platoon environment.
    """

    def __init__(
        self,
        state_dim: int,
        N: int,
        render_mode: str | None = None,
        render_fps: int = 10,
        render_history_length: int = 100,
        plot_size: tuple[int, int] | None = None,
        record: bool = False,
        record_directory: str | None = None,
    ):
        """
        Initialize the renderer.

        Args:
            state_dim: int, vehicle state dimension
            N: int, number of vehicles in the platoon
            render_mode: str, mode for rendering the environment
            render_fps: int, frames per second for rendering
            render_history_length: int, number of past frames to render
            plot_size: tuple[int, int], size of the matplotlib plot
            record: bool, whether to record the rendering
            record_directory: str, directory to record the rendering to
        """
        self.state_dim = state_dim
        self.N = N
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.render_history_length = render_history_length
        default_size = (6, 4) if self.state_dim == 2 else (8, 4)
        self.plot_size = plot_size if plot_size else default_size
        self.record = record
        self.record_directory = record_directory
        if self.record:
            assert self.record_directory is not None
            self.video_dir = os.path.join(
                self.record_directory,
                f'platoon_env-{datetime.now().isoformat(timespec="seconds")}',
            )
            if not os.path.exists(self.video_dir):
                os.makedirs(self.video_dir)

        self.window = None

        # data for plotting
        if self.render_mode == "human":
            self.plot_inds = np.linspace(0, N - 1, min(10, N)).round().astype(int)
            self.position_err_lines = [[]] + [[] for _ in self.plot_inds]
            self.velocity_err_lines = [[]] + [[] for _ in self.plot_inds]
            self.position_lines = [[]] + [[] for _ in self.plot_inds]
            self.velocity_lines = [[]] + [[] for _ in self.plot_inds]
            if self.state_dim == 3:
                self.action_lines = [[] for _ in self.plot_inds]
                self.accel_lines = [[]] + [[] for _ in self.plot_inds]
            self._init_history()

    def render(self) -> None:
        if not self.render_mode:
            return

        self._render_frame()
        if self.record:
            self._record_frame()
        self.timestep += 1

    def reset(self) -> None:
        if not self.render_mode:
            return

        self._init_history()
        self.timestep = 0

        if self.window is None:
            self._init_render()
        # else:
        #     self._update_render_lines()
        #     self._render_frame()

    def close(self) -> None:
        if not self.render_mode:
            return
        plt.close(self.fig)
        pygame.display.quit()
        pygame.quit()

    def _init_history(self) -> None:
        """
        Resets/initializes the history for plotting to empty.
        """
        self.err_history = [np.array([]) for _ in range(self.N)]
        self.obs_history = [np.array([]) for _ in range(self.N)]
        self.state_history = [np.array([]) for _ in range(self.N)]
        self.act_history = [np.array([]) for _ in range(self.N)]
        self.vl_state_history = np.array([])
        self.time_history = np.array([])

    def _init_render(self) -> None:
        if not self.render_mode:
            return

        dpi = 100 if sys.platform.startswith("linux") else 50
        if self.state_dim == 2:
            self.fig, self.ax = plt.subplots(
                nrows=2, ncols=2, sharex=True, figsize=self.plot_size, dpi=dpi
            )
        else:
            self.fig, self.ax = plt.subplots(
                nrows=2, ncols=3, sharex=True, figsize=self.plot_size, dpi=dpi
            )
        if self.state_dim == 2:
            subplots_adjust_settings = [0.08, 0.13, 0.85, 0.85, 0.25, 0.3]
        elif self.state_dim == 3:
            subplots_adjust_settings = [0.07, 0.13, 0.90, 0.85, 0.25, 0.3]
        self.fig.subplots_adjust(*subplots_adjust_settings)
        self.fig.suptitle("Platoon dynamics")
        self.ax[0, 0].set_title("spacing error [m]")
        self.ax[0, 1].set_title("velocity error [m/s]")
        self.ax[1, 0].set_title("position [m]")
        self.ax[1, 1].set_title("velocity [m/s]")
        if self.state_dim == 3:
            self.ax[0, 2].set_title(r"control input [m/s$^2$]")
            self.ax[1, 2].set_title(r"acceleration [m/s$^2$]")
        self.ax[1, 0].set_xlabel("time [s]")
        self.ax[1, 1].set_xlabel("time [s]")
        for a in self.ax.flatten():
            a.grid()
        vl_line = [mlines.Line2D([], [], color="k", label="vl")]
        veh_lines = [
            mlines.Line2D([], [], color=f"C{k}", label=f"{i + 1}")
            for k, i in enumerate(self.plot_inds)
        ]
        self.fig.legend(handles=vl_line + veh_lines, loc="center right")

        if self.window is None:
            self.canvas = agg.FigureCanvasAgg(self.fig)
            self.canvas.draw()
            self.renderer = self.canvas.get_renderer()
            self.raw_data = self.renderer.buffer_rgba()

            pygame.init()
            self.window = pygame.display.set_mode(self.raw_data.shape[:2][::-1])
            self.screen = pygame.display.get_surface()
            self.canvas_size = self.canvas.get_width_height()
            self.surf = pygame.image.frombuffer(self.raw_data, self.canvas_size, "RGBA")
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
            self.clock = pygame.time.Clock()

    def _render_frame(self):
        self._update_render_lines()
        self._set_ax_lims()
        self.canvas.draw()
        self.renderer = self.canvas.get_renderer()
        self.raw_data = self.renderer.buffer_rgba()
        self.surf = pygame.image.frombuffer(self.raw_data, self.canvas_size, "RGBA")
        self.screen.blit(self.surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.render_fps)

    def _update_render_lines(self):
        if self.position_err_lines[0]:
            self.position_err_lines[0] = self.position_err_lines[0].pop(0)
            self.position_err_lines[0].remove()
        self.position_err_lines[0] = self.ax[0, 0].plot(
            self.time_history,
            np.zeros_like(self.time_history),
            color="k",
            label="vl",
        )
        if self.velocity_err_lines[0]:
            self.velocity_err_lines[0] = self.velocity_err_lines[0].pop(0)
            self.velocity_err_lines[0].remove()
        self.velocity_err_lines[0] = self.ax[0, 1].plot(
            self.time_history,
            np.zeros_like(self.time_history),
            color="k",
        )
        if self.position_lines[0]:
            self.position_lines[0] = self.position_lines[0].pop(0)
            self.position_lines[0].remove()
        self.position_lines[0] = self.ax[1, 0].plot(
            self.time_history, self.vl_state_history[0, :], color="k"
        )
        if self.velocity_lines[0]:
            self.velocity_lines[0] = self.velocity_lines[0].pop(0)
            self.velocity_lines[0].remove()
        self.velocity_lines[0] = self.ax[1, 1].plot(
            self.time_history, self.vl_state_history[1, :], color="k"
        )
        if self.state_dim == 3:
            if self.accel_lines[0]:
                self.accel_lines[0] = self.accel_lines[0].pop(0)
                self.accel_lines[0].remove()
            self.accel_lines[0] = self.ax[1, 2].plot(
                self.time_history, self.vl_state_history[2, :], color="k"
            )
        for k, i in enumerate(self.plot_inds):
            if self.position_err_lines[k + 1]:
                self.position_err_lines[k + 1] = self.position_err_lines[k + 1].pop(0)
                self.position_err_lines[k + 1].remove()
            self.position_err_lines[k + 1] = self.ax[0, 0].plot(
                self.time_history,
                self.err_history[i][0, :],
                color=f"C{k}",
                label=f"{i + 1}",
            )
            if self.velocity_err_lines[k + 1]:
                self.velocity_err_lines[k + 1] = self.velocity_err_lines[k + 1].pop(0)
                self.velocity_err_lines[k + 1].remove()
            self.velocity_err_lines[k + 1] = self.ax[0, 1].plot(
                self.time_history, self.err_history[i][1, :], color=f"C{k}"
            )
            if self.position_lines[k + 1]:
                self.position_lines[k + 1] = self.position_lines[k + 1].pop(0)
                self.position_lines[k + 1].remove()
            self.position_lines[k + 1] = self.ax[1, 0].plot(
                self.time_history, self.state_history[i][0, :], color=f"C{k}"
            )
            if self.velocity_lines[k + 1]:
                self.velocity_lines[k + 1] = self.velocity_lines[k + 1].pop(0)
                self.velocity_lines[k + 1].remove()
            self.velocity_lines[k + 1] = self.ax[1, 1].plot(
                self.time_history, self.state_history[i][1, :], color=f"C{k}"
            )
            if self.state_dim == 3:
                if self.action_lines[k]:
                    self.action_lines[k] = self.action_lines[k].pop(0)
                    self.action_lines[k].remove()
                self.action_lines[k] = self.ax[0, 2].plot(
                    self.time_history, self.act_history[i][0, :], color=f"C{k}"
                )
                if self.accel_lines[k + 1]:
                    self.accel_lines[k + 1] = self.accel_lines[k + 1].pop(0)
                    self.accel_lines[k + 1].remove()
                self.accel_lines[k + 1] = self.ax[1, 2].plot(
                    self.time_history, self.state_history[i][2, :], color=f"C{k}"
                )

    def _set_ax_lims(self):
        for a in self.ax.flatten():
            a.set_xlim([self.time_history[0], self.time_history[-1] + 1])

        pos_err_lims = (
            min([self.err_history[i][0, :].min() for i in range(self.N)]) - 0.1,
            max([self.err_history[i][0, :].max() for i in range(self.N)]) + 0.1,
        )
        vel_err_lims = (
            min([self.err_history[i][1, :].min() for i in range(self.N)]) - 0.1,
            max([self.err_history[i][1, :].max() for i in range(self.N)]) + 0.1,
        )
        pos_lims = [
            min([self.state_history[i][0, :].min() for i in range(self.N)]) - 1,
            max([self.state_history[i][0, :].max() for i in range(self.N)]) + 1,
        ]
        pos_lims[0] = min(pos_lims[0], self.vl_state_history[0, :].min() - 1)
        pos_lims[1] = max(pos_lims[1], self.vl_state_history[0, :].max() + 1)
        vel_lims = [
            min([self.state_history[i][1, :].min() for i in range(self.N)]) - 0.1,
            max([self.state_history[i][1, :].max() for i in range(self.N)]) + 0.1,
        ]
        vel_lims[0] = min(vel_lims[0], self.vl_state_history[1, :].min() - 0.1)
        vel_lims[1] = max(vel_lims[1], self.vl_state_history[1, :].max() + 0.1)
        self.ax[0, 0].set_ylim(pos_err_lims)
        self.ax[0, 1].set_ylim(vel_err_lims)
        self.ax[1, 0].set_ylim(pos_lims)
        self.ax[1, 1].set_ylim(vel_lims)
        if self.state_dim == 3:
            action_lims = [
                min([self.act_history[i][0, :].min() for i in range(self.N)]) - 0.1,
                max([self.act_history[i][0, :].max() for i in range(self.N)]) + 0.1,
            ]
            accel_lims = [
                min([self.state_history[i][2, :].min() for i in range(self.N)]) - 0.1,
                max([self.state_history[i][2, :].max() for i in range(self.N)]) + 0.1,
            ]
            accel_lims[0] = min(accel_lims[0], self.vl_state_history[2, :].min() - 0.1)
            accel_lims[1] = max(accel_lims[1], self.vl_state_history[2, :].max() + 0.1)
            self.ax[0, 2].set_ylim(action_lims)
            self.ax[1, 2].set_ylim(accel_lims)

    def update_history(
        self,
        time: float,
        vehicles: list[Vehicle],
        virtual_leader: VirtualLeader,
        observations: list[np.ndarray],
        errors: list[np.ndarray],
        actions: list[np.ndarray] | None = None,
    ) -> None:
        if not self.render_mode:
            return

        if actions is None:
            actions = [np.zeros((v.dyn.m, 1)) for v in vehicles]
        for i, (obs, err, actions) in enumerate(zip(observations, errors, actions)):
            self.obs_history[i] = (
                np.c_[self.obs_history[i], obs]
                if self.obs_history[i].size
                else obs.reshape(-1, 1).copy()
            )
            self.err_history[i] = (
                np.c_[self.err_history[i], err]
                if self.err_history[i].size
                else err.reshape(-1, 1).copy()
            )
            self.state_history[i] = (
                np.c_[self.state_history[i], vehicles[i].state]
                if self.state_history[i].size
                else vehicles[i].state.reshape(-1, 1).copy()
            )
            self.act_history[i] = (
                np.c_[self.act_history[i], actions]
                if self.act_history[i].size
                else actions.reshape(-1, 1).copy()
            )
            if self.obs_history[i].shape[1] > self.render_history_length:
                self.obs_history[i] = self.obs_history[i][:, 1:]
            if self.err_history[i].shape[1] > self.render_history_length:
                self.err_history[i] = self.err_history[i][:, 1:]
            if self.state_history[i].shape[1] > self.render_history_length:
                self.state_history[i] = self.state_history[i][:, 1:]
            if self.act_history[i].shape[1] > self.render_history_length:
                self.act_history[i] = self.act_history[i][:, 1:]
        self.time_history = (
            np.concatenate((self.time_history, np.array([time])))
            if self.time_history.size
            else np.array([time])
        )
        if len(self.time_history) > self.render_history_length:
            self.time_history = self.time_history[1:]
        # update virtual leader history for plotting
        self.vl_state_history = (
            np.c_[self.vl_state_history, virtual_leader.state]
            if self.vl_state_history.size
            else virtual_leader.state.reshape(-1, 1).copy()
        )
        if self.vl_state_history.shape[1] > self.render_history_length:
            self.vl_state_history = self.vl_state_history[:, 1:]

    def _record_frame(self):
        pygame.image.save(
            self.window,
            os.path.join(
                self.video_dir, f"timestep-{str(len(self.time_history)).zfill(5)}.png"
            ),
        )
