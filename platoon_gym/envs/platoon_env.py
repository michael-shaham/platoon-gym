import os

os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (0, 0)
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from platoon_gym.envs.utils import HEADWAY_OPTIONS
from platoon_gym.veh.vehicle import Vehicle
from platoon_gym.veh.virtual_leader import VirtualLeader


class PlatoonEnv(gym.Env):
    """Platoon environment.

    The main platooning gym environment. Visualization is done using pygame.
    Users pass in a list of vehicles and environment arguments. The environment
    arguments contain information about plotting and platoon attributes.

    Attributes:
        vehicles: list[Vehicle], the list of vehicles in the platoon
        virtual_leader: VirtualLeader, the virtual leader
    """

    metadata = {"render_modes": ["plot"], "render_fps": 10}

    def __init__(
        self,
        vehicles: List[Vehicle],
        virtual_leader: VirtualLeader,
        env_args: dict,
        seed: int = 4,
        render_mode: Optional[str] = None,
    ):
        """Initializes the platoon environment.

        Args:
            vehicles: list[Vehicle], the list of vehicles in the platoon
            env_args: dict, environment arguments
            seed: int, random seed
            render_mode: str, the rendering mode
        """
        super().__init__()
        assert env_args["headway"] in HEADWAY_OPTIONS
        if "reset time" not in env_args:
            self.reset_time = np.inf
        else:
            self.reset_time = env_args["reset time"]
        if "reset threshold" not in env_args:
            self.reset_thresh = None
        else:
            self.reset_thresh = env_args["reset threshold"]
        self.env_args = env_args
        self.time = 0.0
        self.dt = env_args["dt"]
        self.n_veh = len(vehicles)
        self.seed = seed
        self.headway = env_args["headway"]

        self.vehs = vehicles
        self.vl = virtual_leader
        self.vl.reset()

        self.observation_space = spaces.Tuple(
            [
                spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(2,),
                    dtype=np.float64,
                )
                for _ in vehicles
            ]
        )
        self.action_space = spaces.Tuple(
            [
                spaces.Box(
                    low=v.dyn.u_lims[:, 0],
                    high=v.dyn.u_lims[:, 1],
                    shape=(v.dyn.m,),
                    dtype=np.float64,
                )
                for v in vehicles
            ]
        )

        self.n_plot = (
            env_args["plot history length"]
            if "plot history length" in env_args
            else 100
        )

        if self.headway.lower() == "cdh":
            self._init_desired_distance(self.env_args["desired distance"])
        elif self.headway.lower() == "cth":
            self._init_desired_time_headway(
                self.env_args["time headway"], self.env_args["desired distance"]
            )
        else:
            raise NotImplementedError

        # rendering stuff
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.plot_inds = (
            np.linspace(0, self.n_veh - 1, min(10, self.n_veh)).round().astype(int)
        )
        self.render_mode = render_mode
        self.window = None
        self.position_err_lines = []
        self.velocity_err_lines = []
        self.position_lines = []
        self.velocity_lines = []

    def _get_obs(self) -> Tuple[np.ndarray, ...]:
        """
        Returns:
            tuple: observations of the vehicles in the platoon
        """
        observations = []
        errors = []
        for i in range(self.n_veh):
            # first vehicle
            d_des = self.time_headway[i] * self.vehs[i].output[1] + self.d_des[i]
            if i == 0:
                distance = self.vl.state[0] - self.vehs[i].output[0]
                position_error = distance - d_des
                velocity_error = self.vl.state[1] - self.vehs[i].output[1]
            # other vehicles
            else:
                distance = self.vehs[i - 1].output[0] - self.vehs[i].output[0]
                position_error = distance - d_des
                velocity_error = self.vehs[i - 1].output[1] - self.vehs[i].output[1]
            obs = np.array([distance, velocity_error])
            err = np.array([position_error, velocity_error])
            observations.append(obs)
            errors.append(err)

        # update history for plotting
        self._update_history(observations, errors)
        return tuple(observations)

    def _get_info(self):
        return {
            "virtual leader plan": self.vl.plan,
            "vehicle states": [v.state for v in self.vehs],
        }

    def reset(
        self, seed: Optional[int] = None, options: dict = {}
    ) -> Tuple[Tuple[np.ndarray, ...], dict]:
        """
        Resets the environment.

        Args:
            seed: int, random seed
            options: dict, provides optional arguments for resetting

        Returns:
            tuple(np.ndarray, ...): observation of each vehicle
            dict: environment information
        """
        super().reset(seed=seed)
        self.seed = seed
        if "vehicles" in options:
            self.vehs = options["vehicles"]
            self.n_veh = len(self.vehs)
            if self.headway.lower() == "cdh":
                if "desired distance" in options:
                    self._init_desired_distance(options["desired distance"])
                else:
                    self._init_desired_distance(self.d_des[-1])
        else:
            for v in self.vehs:
                v.reset()
        if "virtual leader" in options:
            self.vl = options["virtual leader"]
            self.vl.reset()
        else:
            if self.vl.state[0] > 0.0:
                self.vl.reset()

        self.err_history = [np.array([]) for _ in range(self.n_veh)]
        self.obs_history = [np.array([]) for _ in range(self.n_veh)]
        self.state_history = [np.array([]) for _ in range(self.n_veh)]
        self.vl_state_history = np.array([])
        self.time_history = np.array([])

        self.time = 0.0
        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "plot" and self.window is None:
            self._init_render()
        elif self.render_mode == "plot":
            self._remove_render_lines()
            self._render_frame()
        return obs, info

    def step(
        self, action: List[np.ndarray]
    ) -> Tuple[Tuple[np.ndarray, ...], float, bool, bool, dict]:
        """
        Steps the environment forward.

        Args:
            action: list[np.ndarray], the action of each vehicle

        Returns:
            tuple(np.ndarray, ...): observation of each vehicle
            float: reward
            bool: whether the episode is terminated
            bool: whether the episode is truncated
            dict: environment information
        """
        for i, a in enumerate(action):
            assert a.shape == (self.vehs[i].m,)
        for i, v in enumerate(self.vehs):
            v.step(action[i])
        self.vl.step()
        self.time += self.dt
        obs = self._get_obs()
        reward = 0.0
        terminated = self._check_collision(obs)
        truncated = False
        if self.time >= self.reset_time:
            # print("time limit reached, truncating")
            truncated = True
        if self.reset_thresh is not None and all_close:
            all_close = self._check_all_close()
            if all_close:
                # print("all close, truncating")
                truncated = True
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "plot":
            self._render_frame()
        else:
            pass

    def close(self):
        if self.render_mode == "plot":
            plt.close(self.fig)
            pygame.display.quit()
            pygame.quit()

    def _init_render(self):
        self.plot_size = self.env_args["plot size"]

        self.fig, self.ax = plt.subplots(
            nrows=2,
            ncols=2,
            sharex=True,
            figsize=self.plot_size,
            dpi=self.env_args["render dpi"],
        )
        if "subplots adjust" not in self.env_args:
            subplots_adjust_settings = [0.08, 0.13, 0.85, 0.85, 0.25, 0.3]
        else:
            subplots_adjust_settings = self.env_args["subplots adjust"]
        self.fig.subplots_adjust(*subplots_adjust_settings)
        self.fig.suptitle("Platoon dynamics")
        self.ax[0, 0].set_title("spacing error [m]")
        self.ax[0, 1].set_title("velocity error [m/s]")
        self.ax[1, 0].set_title("position [m]")
        self.ax[1, 1].set_title("velocity [m/s]")
        self.ax[1, 0].set_xlabel("time [s]")
        self.ax[1, 1].set_xlabel("time [s]")

        self.position_err_lines.append(
            self.ax[0, 0].plot(
                self.time_history,
                np.zeros_like(self.time_history),
                color="k",
                label="vl",
            )
        )
        self.velocity_err_lines.append(
            self.ax[0, 1].plot(
                self.time_history,
                np.zeros_like(self.time_history),
                color="k",
            )
        )
        self.position_lines.append(
            self.ax[1, 0].plot(
                self.time_history, self.vl_state_history[0, :], color="k"
            )
        )
        self.velocity_lines.append(
            self.ax[1, 1].plot(
                self.time_history, self.vl_state_history[1, :], color="k"
            )
        )
        for k, i in enumerate(self.plot_inds):
            self.position_err_lines.append(
                self.ax[0, 0].plot(
                    self.time_history,
                    self.err_history[i][0, :],
                    color=f"C{k}",
                    label=f"{i + 1}",
                )
            )
            self.velocity_err_lines.append(
                self.ax[0, 1].plot(
                    self.time_history, self.err_history[i][1, :], color=f"C{k}"
                )
            )
            self.position_lines.append(
                self.ax[1, 0].plot(
                    self.time_history, self.state_history[i][0, :], color=f"C{k}"
                )
            )
            self.velocity_lines.append(
                self.ax[1, 1].plot(
                    self.time_history, self.state_history[i][1, :], color=f"C{k}"
                )
            )
        self.fig.legend(loc="center right")
        for a in self.ax.flatten():
            a.grid()
        self._set_ax_lims()

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
        self._remove_render_lines()
        self._set_ax_lims()
        self.canvas.draw()
        self.renderer = self.canvas.get_renderer()
        self.raw_data = self.renderer.buffer_rgba()
        self.surf = pygame.image.frombuffer(self.raw_data, self.canvas_size, "RGBA")
        self.screen.blit(self.surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _remove_render_lines(self):
        self.position_err_lines[0] = self.position_err_lines[0].pop(0)
        self.position_err_lines[0].remove()
        self.position_err_lines[0] = self.ax[0, 0].plot(
            self.time_history,
            np.zeros_like(self.time_history),
            color="k",
            label="vl",
        )
        self.velocity_err_lines[0] = self.velocity_err_lines[0].pop(0)
        self.velocity_err_lines[0].remove()
        self.velocity_err_lines[0] = self.ax[0, 1].plot(
            self.time_history,
            np.zeros_like(self.time_history),
            color="k",
        )
        self.position_lines[0] = self.position_lines[0].pop(0)
        self.position_lines[0].remove()
        self.position_lines[0] = self.ax[1, 0].plot(
            self.time_history, self.vl_state_history[0, :], color="k"
        )
        self.velocity_lines[0] = self.velocity_lines[0].pop(0)
        self.velocity_lines[0].remove()
        self.velocity_lines[0] = self.ax[1, 1].plot(
            self.time_history, self.vl_state_history[1, :], color="k"
        )
        for k, i in enumerate(self.plot_inds):
            self.position_err_lines[k + 1] = self.position_err_lines[k + 1].pop(0)
            self.position_err_lines[k + 1].remove()
            self.position_err_lines[k + 1] = self.ax[0, 0].plot(
                self.time_history,
                self.err_history[i][0, :],
                color=f"C{k}",
                label=f"{i + 1}",
            )
            self.velocity_err_lines[k + 1] = self.velocity_err_lines[k + 1].pop(0)
            self.velocity_err_lines[k + 1].remove()
            self.velocity_err_lines[k + 1] = self.ax[0, 1].plot(
                self.time_history, self.err_history[i][1, :], color=f"C{k}"
            )
            self.position_lines[k + 1] = self.position_lines[k + 1].pop(0)
            self.position_lines[k + 1].remove()
            self.position_lines[k + 1] = self.ax[1, 0].plot(
                self.time_history, self.state_history[i][0, :], color=f"C{k}"
            )
            self.velocity_lines[k + 1] = self.velocity_lines[k + 1].pop(0)
            self.velocity_lines[k + 1].remove()
            self.velocity_lines[k + 1] = self.ax[1, 1].plot(
                self.time_history, self.state_history[i][1, :], color=f"C{k}"
            )

    def _set_ax_lims(self):
        for a in self.ax.flatten():
            a.set_xlim([self.time_history[0], self.time_history[-1] + 1])

        pos_err_lims = (
            min([self.err_history[i][0, :].min() for i in range(self.n_veh)]) - 0.1,
            max([self.err_history[i][0, :].max() for i in range(self.n_veh)]) + 0.1,
        )
        vel_err_lims = (
            min([self.err_history[i][1, :].min() for i in range(self.n_veh)]) - 0.1,
            max([self.err_history[i][1, :].max() for i in range(self.n_veh)]) + 0.1,
        )
        pos_lims = (
            min([self.state_history[i][0, :].min() for i in range(self.n_veh)]) - 1,
            max([self.state_history[i][0, :].max() for i in range(self.n_veh)]) + 1,
        )
        vel_lims = (
            min([self.state_history[i][1, :].min() for i in range(self.n_veh)]) - 0.1,
            max([self.state_history[i][1, :].max() for i in range(self.n_veh)]) + 0.1,
        )
        self.ax[0, 0].set_ylim(pos_err_lims)
        self.ax[0, 1].set_ylim(vel_err_lims)
        self.ax[1, 0].set_ylim(pos_lims)
        self.ax[1, 1].set_ylim(vel_lims)

    def _update_history(self, observations, errors):
        for i, (obs, err) in enumerate(zip(observations, errors)):
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
                np.c_[self.state_history[i], self.vehs[i].state]
                if self.state_history[i].size
                else self.vehs[i].state.reshape(-1, 1).copy()
            )
            if self.obs_history[i].shape[1] > self.n_plot:
                self.obs_history[i] = self.obs_history[i][:, 1:]
            if self.err_history[i].shape[1] > self.n_plot:
                self.err_history[i] = self.err_history[i][:, 1:]
            if self.state_history[i].shape[1] > self.n_plot:
                self.state_history[i] = self.state_history[i][:, 1:]
        self.time_history = (
            np.concatenate((self.time_history, np.array([self.time])))
            if self.time_history.size
            else np.array([self.time])
        )
        if len(self.time_history) > self.n_plot:
            self.time_history = self.time_history[1:]
        # update virtual leader history for plotting
        self.vl_state_history = (
            np.c_[self.vl_state_history, self.vl.state]
            if self.vl_state_history.size
            else self.vl.state.reshape(-1, 1).copy()
        )
        if self.vl_state_history.shape[1] > self.n_plot:
            self.vl_state_history = self.vl_state_history[:, 1:]

    def _check_collision(self, obs):
        for i, o in enumerate(obs):
            if i == 0:
                continue
            else:
                if o[0] <= 0.0:
                    return True
        return False

    def _init_desired_distance(self, d_des):
        self.time_headway = [0.0 for _ in range(self.n_veh)]
        assert type(d_des) in [float, list]
        if type(d_des) == float:
            self.d_des = [0.0] + [d_des] * (self.n_veh - 1)
        elif type(d_des) == list:
            self.d_des = d_des

    def _init_desired_time_headway(self, time_headway, d_des):
        assert type(time_headway) in [float, list]
        if d_des is not None:
            if type(d_des) == float:
                self.d_des = [0.0] + [d_des] * (self.n_veh - 1)
            elif type(d_des) == list:
                self.d_des = d_des
        else:
            self.d_des = [0.0] * (self.n_veh)
        if type(time_headway) == float:
            self.time_headway = [0.0] + [time_headway] * (self.n_veh - 1)
        elif type(time_headway) == list:
            self.time_headway = time_headway

    def _check_all_close(self):
        for i in range(self.n_veh):
            if i == 0:
                distance = self.vl.state[0] - self.vehs[i].output[0]
                velocity_error = self.vl.state[1] - self.vehs[i].output[1]
            else:
                distance = self.vehs[i - 1].output[0] - self.vehs[i].output[0]
                velocity_error = self.vehs[i - 1].output[1] - self.vehs[i].output[1]
            if self.headway.lower() == "cdh":
                position_error = distance - self.d_des[i]
            elif self.headway.lower() == "cth":
                position_error = (
                    distance - self.time_headway[i] * self.vehs[i].output[1]
                )
            error = np.array([position_error, velocity_error])
            close = (np.abs(error) < self.reset_thresh).all()
            if not close:
                return False
        return True
