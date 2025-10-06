import numpy as np

import gymnasium as gym

from platoon_gym.envs.platoon_renderer import PlatoonRenderer
from platoon_gym.veh.vehicle import Vehicle
from platoon_gym.veh.virtual_leader import VirtualLeader

TOPOLOGY_OPTIONS = ["PF"]  # TODO: implement BD


def default_vehicles(
    num_vehicles: int,
    distance_headway: float | list[float],
    time_headway: float | list[float],
    initial_velocity: float = 20.0,
    dt: float = 0.1,
) -> list[Vehicle]:
    """
    Default vehicles for the platoon environment.

    Returns:
        list[Vehicle]: N vehicles
    """
    # vehicle dynamics
    from platoon_gym.dyn.linear_accel import LinearAccel

    x_lims = np.array([[-1e6, 1e6], [-100, 100], [-100, 100]])
    u_lims = np.array([[-3.0, 3.0]])
    dyn = LinearAccel(dt=dt, tau=0.5, x_lims=x_lims, u_lims=u_lims)

    # vehicles
    if type(distance_headway) == list and type(time_headway) == list:
        assert len(distance_headway) == len(time_headway)
        N = len(distance_headway)
    elif type(distance_headway) == list and type(time_headway) == float:
        N = len(distance_headway)
        time_headway = [time_headway] * N
    elif type(distance_headway) == float and type(time_headway) == list:
        N = len(time_headway)
        distance_headway = [distance_headway] * N
    else:
        N = num_vehicles
        distance_headway = [distance_headway] * N
        time_headway = [time_headway] * N
    vehicles = []
    for i in range(N):
        d_des = distance_headway[i] + time_headway[i] * initial_velocity
        vehicles.append(
            Vehicle(dyn, position=-i * d_des, velocity=initial_velocity, acceleration=0)
        )

    return vehicles


def default_virtual_leader(dt: float) -> VirtualLeader:
    """
    Default virtual leader for the platoon environment.

    Args:
        dt: float, time step

    Retruns:
        VirtualLeader: virtual leader
    """
    # virtual leader
    from platoon_gym.veh.virtual_leader import VirtualLeader

    vl_traj_type = "random_step"
    vl_traj_args = {
        "horizon": 1000,
        "dt": dt,
        "step_time_min": 1.0,
        "step_time_max": 3.0,
        "step_vel_min": 15.0,
        "step_vel_max": 25.0,
        "step_accel_min": 0.5,
        "step_accel_max": 1.5,
    }
    vl = VirtualLeader(vl_traj_type, vl_traj_args, velocity=20.0)
    return vl


class PlatoonEnv(gym.Env):
    """
    Multi-agent platooning gymnasium (gym) environment.
    """

    metadata = {
        "name": "platoon_env-v0",
        "render_modes": ["human"],
        "render_fps": 10,
        "render_history_length": 100,
        "record": False,
        "record_directory": None,
    }

    def __init__(
        self,
        config: dict = {},
        render_mode: str | None = None,
    ):
        """
                Initializes the platoon environment. 'config' is a dictionary that
                contains arguments for the environment and environment metadata.

                Args:
                    config: dict, environment configuration
                    render_mode: str, the rendering mode
        """
        super().__init__()
        self.env_args = self.default_config()
        self.configure(config)
        self.dt = self.env_args["dt"]
        self.vehs: list[Vehicle] = self.env_args["vehicles"]
        self.vl: VirtualLeader = self.env_args["virtual_leader"]
        self.N = len(self.vehs)
        self.agents = list(range(self.N))
        self.reset_thresh = self.env_args["reset_threshold"]
        self.reset_time = self.env_args["reset_time"]

        self._init_headway()

        low_obs = (
            np.array([-100.0, -10.0, -10.0])
            if self.vehs[0].p == 3
            else np.array([-100.0, -10.0])
        )
        high_obs = (
            np.array([100.0, 10.0, 10.0])
            if self.vehs[0].p == 3
            else np.array([100.0, 10.0])
        )
        self.observation_space = gym.spaces.Tuple(
            [
                gym.spaces.Box(low=low_obs, high=high_obs, dtype=np.float64)
                for _ in self.vehs
            ]
        )
        self.action_space = gym.spaces.Tuple(
            [
                gym.spaces.Box(
                    low=v.dyn.u_lims[:, 0], high=v.dyn.u_lims[:, 1], dtype=np.float64
                )
                for v in self.vehs
            ]
        )

        # rendering stuff
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.renderer = PlatoonRenderer(
            state_dim=self.vehs[0].dyn.n,
            N=self.N,
            render_mode=render_mode,
            render_fps=self.metadata["render_fps"],
            render_history_length=self.metadata["render_history_length"],
            record=self.metadata["record"],
            record_directory=self.metadata["record_directory"],
        )

    def configure(self, config: dict = {}):
        if config:
            for k, v in config.items():
                if k in self.metadata:
                    self.metadata[k] = v
                else:
                    self.env_args[k] = v
        if "vehicles" not in config:
            self.env_args["vehicles"] = default_vehicles(
                self.env_args["num_vehicles"],
                self.env_args["distance_headway"], 
                self.env_args["time_headway"], 
                dt=self.env_args["dt"]
            )
        if "virtual_leader" not in config:
            self.env_args["virtual_leader"] = default_virtual_leader(
                self.env_args["dt"]
            )

    @staticmethod
    def default_config() -> dict:
        env_args = {
            "num_vehicles": 10,
            "topology": "PF",
            "dt": 0.1,
            "distance_headway": 2.0,
            "time_headway": .5,
            "reset_threshold": None,
            "reset_time": float("inf"),
            "position_cost": 1.0,
            "velocity_cost": 10.0,
            "acceleration_cost": 0.0,
            "input_cost": 1.0,
            "vehicles": None,
            "virtual_leader": None,
        }
        return env_args

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[tuple[np.ndarray, ...], dict]:
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
        if self.vl.traj_type == "i24_trajectory":
            self.vl.reset(seed=seed)
            v0 = self.vl.state[1]
            p0s = [-self.distance_headway[0] - v0 * self.time_headway[0]]
            for i in range(1, self.N):
                p0s.append(p0s[-1] - self.distance_headway[i] - v0 * self.time_headway[i])
            for i, v in enumerate(self.vehs):
                v.reset(position=p0s[i], velocity=v0, seed=seed)
        else:
            self.vl.reset(seed=seed)
            for v in self.vehs:
                v.reset(seed=seed)

        self.renderer.reset()

        self.time = 0.0
        self.timestep = 0
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(
        self, actions: list[np.ndarray]
    ) -> tuple[tuple[np.ndarray, ...], float, bool, bool, dict]:
        """
        Steps the environment forward.

        Args:
            actions: list[np.ndarray], the action of each vehicle

        Returns:
            tuple[np.ndarray, ...]: observation of each vehicle
            float: reward
            bool: whether the episode is terminated
            bool: whether the episode is truncated
            dict: environment information
        """
        for i, a in enumerate(actions):
            assert a.shape == (self.vehs[i].m,)
        for i, v in enumerate(self.vehs):
            v.step(actions[i])
        vl_done = self.vl.step()
        self.time += self.dt
        self.timestep += 1
        obs = self._get_obs(actions)
        reward = self._get_reward(actions, obs)
        terminated = self._check_collision(obs)
        truncated = False
        if self.time > self.reset_time or vl_done:
            # print("time limit reached, truncating")
            truncated = True
        if self.reset_thresh is not None:
            all_close = self._check_all_close()
            if all_close:
                # print("all close, truncating")
                truncated = True
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def render(self):
        self.renderer.render()

    def close(self):
        self.renderer.close()

    def _get_obs(
        self, actions: list[np.ndarray] | None = None
    ) -> tuple[np.ndarray, ...]:
        """
        Args:
            actions: list[np.ndarray], the action of each vehicle

        Returns:
            tuple: observations of the vehicles in the platoon
        """
        observations = []
        for i in self.agents:
            # first vehicle
            if i == 0:
                distance = self.vl.state[0] - self.vehs[i].output[0]
                velocity_error = self.vl.state[1] - self.vehs[i].output[1]
                if self.vehs[0].dyn.p == 3:
                    accel_error = self.vl.state[2] - self.vehs[0].output[2]
            # other vehicles
            else:
                distance = self.vehs[i - 1].output[0] - self.vehs[i].output[0]
                velocity_error = self.vehs[i - 1].output[1] - self.vehs[i].output[1]
                if self.vehs[0].dyn.p == 3:
                    accel_error = self.vehs[i - 1].output[2] - self.vehs[i].output[2]
            obs = np.array([distance, velocity_error])
            if self.vehs[i].p == 3:
                obs = np.r_[obs, accel_error]
            observations.append(obs)

        # update history for plotting
        errors = self._get_error_from_obs(observations)
        self.renderer.update_history(
            self.time, self.vehs, self.vl, observations, errors, actions
        )
        return tuple(observations)

    def _get_info(self):
        return {
            "virtual_leader_plan": self.vl.plan,
            "vehicle_states": [v.state for v in self.vehs],
        }

    def _get_reward(
        self, actions: list[np.ndarray], obs: tuple[np.ndarray, ...]
    ) -> float:
        reward = 0.0
        for a in self.agents:
            reward += (
                self.env_args["input_cost"] * sum(actions[a] ** 2)
                + self.env_args["position_cost"] * obs[a][0] ** 2
                + self.env_args["velocity_cost"] * obs[a][1] ** 2
            )
            if self.vehs[a].n == 3:
                reward += (
                    self.env_args["acceleration_cost"] * self.vehs[a].state[2] ** 2
                )
        return -reward

    def _get_error_from_obs(
        self, obs: tuple[np.ndarray, ...]
    ) -> tuple[np.ndarray, ...]:
        errors = []
        for i, o in enumerate(obs):
            d_des = (
                self.time_headway[i] * self.vehs[i].output[1] + self.distance_headway[i]
            )
            error = np.zeros(self.vehs[i].p)
            error[0] = o[0] - d_des
            error[1] = o[1]
            if self.vehs[i].p == 3:
                error[2] = o[2]
            errors.append(error)
        return tuple(errors)

    def _check_collision(self, obs):
        for i, o in enumerate(obs):
            if i == 0:
                continue
            else:
                if o[0] <= 0.0:
                    # print("collision!!!")
                    return True
        return False

    def _init_headway(self):
        assert type(self.env_args["distance_headway"]) in [float, list]
        assert type(self.env_args["time_headway"]) in [float, list]
        distance_headway = self.env_args["distance_headway"]
        time_headway = self.env_args["time_headway"]

        if type(distance_headway) == float:
            self.distance_headway = [0.0] + [distance_headway] * (self.N - 1)
        elif type(distance_headway) == list:
            self.distance_headway = distance_headway
        else:
            self.distance_headway = [0.0] * (self.N)

        if type(time_headway) == float:
            self.time_headway = [0.0] + [time_headway] * (self.N - 1)
        elif type(time_headway) == list:
            self.time_headway = time_headway
        else:
            self.time_headway = [0.0] * (self.N)

    def _check_all_close(self):
        for i in self.agents:
            if i == 0:
                distance = self.vl.state[0] - self.vehs[i].output[0]
                velocity_error = self.vl.state[1] - self.vehs[i].output[1]
            else:
                distance = self.vehs[i - 1].output[0] - self.vehs[i].output[0]
                velocity_error = self.vehs[i - 1].output[1] - self.vehs[i].output[1]
            position_error = (
                distance
                - self.distance_headway[i]
                - self.time_headway[i] * self.vehs[i].output[1]
            )
            error = np.array([position_error, velocity_error])
            close = (np.abs(error) < self.reset_thresh).all()
            if not close:
                return False
        return True


class PlatoonEnvActionNormalized(gym.ActionWrapper):

    def __init__(self, env: PlatoonEnv):
        super().__init__(env)
        self.action_space = gym.spaces.Tuple(
            [
                gym.spaces.Box(low=-1.0, high=1.0, dtype=np.float64)
                for v in self.env.unwrapped.vehs
            ]
        )

    def action(self, action: list[np.ndarray]) -> list[np.ndarray]:
        return self._denormalize_action(action)

    def _denormalize_action(self, actions: np.ndarray) -> np.ndarray:
        denormalized_actions = []
        for i, a in enumerate(actions):
            a_min = self.env.unwrapped.vehs[i].dyn.u_lims[:, 0]
            a_max = self.env.unwrapped.vehs[i].dyn.u_lims[:, 1]
            if a < 0.0:
                denormalized_action = -a * a_min
            else:
                denormalized_action = a * a_max
            denormalized_actions.append(denormalized_action)
        return denormalized_actions


class PlatoonEnvErrorObs(gym.ObservationWrapper):

    def __init__(self, env: PlatoonEnv):
        super().__init__(env)
        lows = []
        highs = []
        for v in self.env.unwrapped.vehs:
            if v.n == 3:
                lows.append(np.array([-100.0, -10.0, -10.0]))
                highs.append(np.array([100.0, 10.0, 10.0]))
            elif v.n == 2:
                lows.append(np.array([-100.0, -10.0]))
                highs.append(np.array([100.0, 10.0]))
            else:
                raise ValueError(f"Invalid state dimension {v.n}")

        self.observation_space = gym.spaces.Tuple(
            [
                gym.spaces.Box(low=lows[i], high=highs[i], dtype=np.float64)
                for i in range(len(self.env.unwrapped.vehs))
            ]
        )

    def observation(
        self, observation: tuple[np.ndarray, ...]
    ) -> tuple[np.ndarray, ...]:
        return self.env.unwrapped._get_error_from_obs(observation)

class PlatoonEnvErrorObsActionNormalized(gym.Wrapper):

    def __init__(self, env: PlatoonEnv):
        """
        Wrapper for use with PlatoonEnvPZ. Normalizes action space and uses 
        error observations. Does not use virtual leader plan in info.
        """
        super().__init__(env)
        
        # observation space
        lows = []
        highs = []
        for v in self.env.unwrapped.vehs:
            if v.p == 3:
                lows.append(np.array([-1e3, -1e3, -1e3]))
                highs.append(np.array([1e3, 1e3, 1e3]))
            elif v.p == 2:
                lows.append(np.array([-1e3, -1e3]))
                highs.append(np.array([1e3, 1e3]))
            else:
                raise ValueError(f"Invalid output dimension {v.p}")

        self.observation_space = gym.spaces.Tuple(
            [
                gym.spaces.Box(low=lows[i], high=highs[i], dtype=np.float64)
                for i in range(len(self.env.unwrapped.vehs))
            ]
        )

        # action space
        self.action_space = gym.spaces.Tuple(
            [
                gym.spaces.Box(low=-1.0, high=1.0, dtype=np.float64)
                for v in self.env.unwrapped.vehs
            ]
        )

    def _get_error_from_obs(
        self, observation: tuple[np.ndarray, ...]
    ) -> tuple[np.ndarray, ...]:
        return self.env.unwrapped._get_error_from_obs(observation)

    def _denormalize_action(self, actions: np.ndarray) -> np.ndarray:
        denormalized_actions = []
        for i, a in enumerate(actions):
            a_min = self.env.unwrapped.vehs[i].dyn.u_lims[:, 0]
            a_max = self.env.unwrapped.vehs[i].dyn.u_lims[:, 1]
            if a < 0.0:
                denormalized_action = -a * a_min
            else:
                denormalized_action = a * a_max
            denormalized_actions.append(denormalized_action)
        return denormalized_actions
    
    def step(
        self, actions: list[np.ndarray]
    ) -> tuple[tuple[np.ndarray, ...], float, bool, bool, dict]:
        actions = self._denormalize_action(actions)
        obs, reward, terminated, truncated, info = self.env.step(actions)
        obs = self._get_error_from_obs(obs)
        info.pop("virtual_leader_plan")
        return obs, reward, terminated, truncated, info
    
    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[tuple[np.ndarray, ...], dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._get_error_from_obs(obs)
        info.pop("virtual_leader_plan")
        return obs, info