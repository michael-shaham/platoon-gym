from datetime import datetime
import gymnasium as gym
import numpy as np
import os
import pygame

from platoon_gym.veh.vehicle import Vehicle
from platoon_gym.utils.utils import get_project_root


class RingEnv(gym.Env):
    """
    Ring road environment. Vehicles drive in a circle. The first vehicle's
    predecessor is the last vehicle in the string. Allows simulating shockwaves
    and dissipation of shockwaves.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10, "record": False}

    def __init__(
        self,
        vehicles: list[Vehicle],
        env_args: dict,
        seed: int = 0,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.rng = np.random.default_rng(seed)

        self.seed = seed
        self.time = 0.0
        self.timestep = 0
        self.vehs = vehicles
        self.n_vehs = len(vehicles)

        # get environment params
        self.env_args = env_args
        self.dt = env_args["dt"]
        self.circumference = env_args["ring circumference"]
        self.radius = self.circumference / (2 * np.pi)

        self.observation_space = gym.spaces.Tuple(
            [
                gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(2,),
                    dtype=np.float64,
                )
                for _ in vehicles
            ]
        )
        self.action_space = gym.spaces.Tuple(
            [
                gym.spaces.Box(
                    low=v.dyn.u_lims[:, 0],
                    high=v.dyn.u_lims[:, 1],
                    shape=(v.dyn.m,),
                    dtype=np.float64,
                )
                for v in vehicles
            ]
        )

        # rendering stuff
        if "render_fps" in env_args:
            self.metadata["render_fps"] = env_args["render_fps"]
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        if render_mode is not None and "record" in env_args:
            self.metadata["record"] = env_args["record"]
        self.render_mode = render_mode
        self.window = None

    def reset(
        self, seed: int | None = None, options: dict = {}
    ) -> tuple[tuple[np.ndarray, ...], dict]:
        """
        Resets the environment.

        Args:
            seed: random seed
            options: dict, provides additional options for the reset

        Returns:
            tuple(np.ndarray, ...): state of the environment
            dict: additional info
        """
        super().reset(seed=seed)
        self.time = 0.0
        self.timestep = 0
        self.seed = seed
        if "vehicles" in options:
            self.vehs = options["vehicles"]
            self.n_vehs = len(self.vehs)
        else:
            for veh in self.vehs:
                veh.reset()

        self._init_render()

        return self._get_obs(), self._get_info()

    def step(
        self, action: tuple[np.ndarray]
    ) -> tuple[tuple[np.ndarray, ...], float, bool, bool, dict]:
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
        self.time += self.dt
        self.timestep += 1
        for i, v in enumerate(self.vehs):
            v.step(action[i])
        obs = self._get_obs()
        reward = 0.0
        terminated = self._check_collision(obs)
        truncated = False
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_frame()
        if self.metadata["record"]:
            pygame.image.save(
                self.window,
                os.path.join(
                    self.video_dir, f"timestep-{str(self.timestep).zfill(5)}.png"
                ),
            )

    def close(self):
        if self.render_mode == "human":
            pygame.display.quit()
            pygame.quit()

    def _get_obs(self) -> tuple[np.ndarray, ...]:
        """
        Observations for this environment are distance to the preceding vehicle
        and velocity difference relative to the preceding vehicle.

        Returns:
            tuple(np.ndarray, ...): observations of the environment
        """
        observations = []
        for i in range(self.n_vehs):
            if i == 0:
                # project the last vehicle's position to the circle
                project_pos = (
                    np.abs(self.vehs[i].state[0]) % self.circumference
                ) * np.sign(self.vehs[i].state[0])
                angle = project_pos / self.radius
                project_pred_pos = (
                    np.abs(self.vehs[-1].state[0]) % self.circumference
                ) * np.sign(self.vehs[-1].state[0])
                angle_pred = project_pred_pos / self.radius
                diff = angle_pred - angle
                while diff <= 0:
                    diff += 2 * np.pi
                while diff > 2 * np.pi:
                    diff -= 2 * np.pi
                distance = diff * self.radius
                velocity_error = self.vehs[-1].state[1] - self.vehs[i].state[1]
            else:
                distance = self.vehs[i - 1].state[0] - self.vehs[i].state[0]
                velocity_error = self.vehs[i - 1].state[1] - self.vehs[i].state[1]
            distance += self.rng.normal(0, 0.1)
            velocity_error += self.rng.normal(0, 0.1)
            observations.append(np.array([distance, velocity_error]))
        return tuple(observations)

    def _get_info(self) -> dict:
        """
        Returns a dictionary of extra information. In this case, the dictionary
        contains each vehicle's current state.
        """
        return {
            "vehicle states": [v.state for v in self.vehs],
        }

    def _check_collision(self, obs):
        for i, o in enumerate(obs):
            if i == 0:
                continue
            else:
                if o[0] <= 0.0:
                    return True
        return False

    def _init_render(self):
        if self.render_mode != "human":
            return
        if self.metadata["record"]:
            self.video_dir = os.path.join(
                get_project_root(),
                "videos",
                f'ring_env-{datetime.now().isoformat(timespec="seconds")}',
            )
            if not os.path.exists(self.video_dir):
                os.makedirs(self.video_dir)
        self.bg_color = (
            (50, 50, 50)
            if "background color" not in self.env_args
            else self.env_args["background color"]
        )
        self.vehicle_colors = (
            [(255, 0, 0)]
            + [(0, 255, 0) for _ in range(1, self.n_vehs - 1)]
            + [(0, 0, 255)]
            if "vehicle colors" not in self.env_args
            else self.env_args["vehicle colors"]
        )
        self.size = (
            600 if "plot size" not in self.env_args else self.env_args["plot size"]
        )
        self.draw_radius = 0.8 * self.size / 2
        self.center = self.size // 2
        pygame.init()
        self.font_size = 16
        self.font = pygame.font.SysFont("Arial", self.font_size)
        self.clock = pygame.time.Clock()
        self.window = pygame.display.set_mode((self.size, self.size))

    def _render_frame(self):
        mean_speed = np.mean([v.state[1] for v in self.vehs])
        std_speed = np.std([v.state[1] for v in self.vehs])
        pygame.display.set_caption(
            f"time: {self.time:.1f}, speed mean: {mean_speed:.2f} m/s, speed std: {std_speed:.2f} m/s"
        )
        display_info = [
            f"time: {self.time:.1f}",
            f"n vehicles: {self.n_vehs}",
            f"circumference: {self.circumference:.1f} m",
            f"speed mean: {mean_speed:.2f} m/s",
            f"speed std: {std_speed:.2f} m/s",
        ]
        text_surfaces = [
            self.font.render(info, True, (255, 255, 255), (50, 50, 50))
            for info in display_info
        ]
        self.window.fill(self.bg_color)
        pixel_sep = self.font_size + 5
        for i, tsurf in enumerate(text_surfaces):
            self.window.blit(tsurf, (5, pixel_sep * i + 5))
        pygame.draw.circle(
            self.window,
            (255, 255, 255),
            (self.center, self.center),
            self.draw_radius,
            width=2,
        )
        self._render_vehicles()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _render_vehicles(self):
        for i, v in enumerate(self.vehs):
            veh_pos = v.state[0]
            circle_pos = -(np.abs(veh_pos) % self.circumference) * np.sign(veh_pos)
            circle_angle = circle_pos / self.circumference * 2 * np.pi
            draw_x = self.center + self.draw_radius * np.cos(circle_angle)
            draw_y = self.center + self.draw_radius * np.sin(circle_angle)
            pygame.draw.circle(
                self.window,
                self.vehicle_colors[i],
                (round(draw_x), round(draw_y)),
                8,
            )
