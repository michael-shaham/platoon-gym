from gymnasium import spaces
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from platoon_gym.envs.platoon_env import PlatoonEnv, PlatoonEnvErrorObsActionNormalized
from platoon_gym.veh.vehicle import Vehicle


class PlatoonEnvPZ(AECEnv):
    """
    Wrapper for the platoon environment that implements the AECEnv interface
    from PettingZoo. Will be used for training with torchrl, and thus does not
    use the virtual leader plan in the infos. Furthermore, the environment uses
    the error state, which is the relative position, velocity, and optionally 
    acceleration error, as the observation.

    Arguments are the same as in PlatoonEnv.
    """

    metadata = {
        "name": "platoon_env-v0",
        "is_parallelizable": True,
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
        self.env = PlatoonEnvErrorObsActionNormalized(
            PlatoonEnv(config=config, render_mode=render_mode)
        )
        self.env_args = self.env.env.unwrapped.env_args
        self.vehs : list[Vehicle] = self.env.env.unwrapped.vehs
        self.possible_agents = ["vehicle_" + str(a) for a in self.env.env.unwrapped.agents]
        self.stoi = {agent: i for i, agent in enumerate(self.possible_agents)}
        self.itos = {i: agent for i, agent in enumerate(self.possible_agents)}
        self.render_mode = render_mode
        self.action_spaces = {
            a: self.env.action_space[self.stoi[a]] for a in self.possible_agents
        }
        self.observation_spaces = {
            a: self.env.observation_space[self.stoi[a]] for a in self.possible_agents
        }

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        """
        Resets the environment.

        Args:
            seed: int, the random seed
            options: dict, additional options
        """
        env_obs, env_info = self.env.reset(seed=seed, options=options)
        self.agents: list[str] = self.possible_agents[:]
        self.observations = self._get_obs_from_env(env_obs)
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = self._get_info_from_env(env_info)
        self.states = {agent: {} for agent in self.agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action: np.ndarray) -> None:
        """
        Steps the environment.

        Args:
            action: np.ndarray, agent action
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        agent = self.agent_selection
        self.states[agent]["action"] = action
        self.states[agent]["state"] = self.vehs[self.stoi[agent]].state.copy()

        if self._agent_selector.is_last():
            env_obs, _, env_term, env_trunc, env_info = self.env.step(
                [self.states[agent]["action"] for agent in self.agents]
            )
            self.observations = self._get_obs_from_env(env_obs)
            self.rewards = self._get_reward()
            self.terminations = {agent: env_term for agent in self.agents}
            self.truncations = {agent: env_trunc for agent in self.agents}
            self.infos = self._get_info_from_env(env_info)
            self.states = {agent: {} for agent in self.agents}
            self.render()
        else:
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()

    def render(self) -> None:
        """
        Renders the environment.
        """
        self.env.render()

    def observe(self, agent: str) -> np.ndarray:
        return self.observations[agent]

    def close(self) -> None:
        self.env.close()

    def observation_space(self, agent: str) -> spaces.Box:
        return self.env.observation_space[self.stoi[agent]]

    def action_space(self, agent: str) -> spaces.Box:
        return self.env.action_space[self.stoi[agent]]

    def _get_obs_from_env(self, env_obs: list[np.ndarray]) -> dict:
        return {self.itos[i]: obs for i, obs in enumerate(env_obs)}

    def _get_info_from_env(self, env_info: dict) -> dict:
        if (
            sum([self.terminations[a] for a in self.agents]) > 0
            or sum([self.truncations[a] for a in self.agents]) > 0
        ):
            return {a: {} for a in self.agents}
        info = {
            self.itos[i]: {"vehicle_state": s}
            for i, s in enumerate(env_info["vehicle_states"])
        }
        return info

    def _get_reward(self) -> dict:
        actions = [self.states[agent]["action"] for agent in self.agents]
        rewards = {}
        for a in self.agents:
            i = self.stoi[a]
            rewards[a] = -(
                self.env_args["input_cost"] * (actions[i] @ actions[i]).item()
                + self.env_args["position_cost"] * self.observations[a][0] ** 2
                + self.env_args["velocity_cost"] * self.observations[a][1] ** 2
            )
            if self.vehs[i].n == 3:
                rewards[a] -= self.env_args["acceleration_cost"] * self.vehs[i].state[2] ** 2
        return rewards


def pz_env(**kwargs) -> PlatoonEnvPZ:
    env = PlatoonEnvPZ(**kwargs)
    env = wrappers.ClipOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_pz_env = parallel_wrapper_fn(pz_env)
