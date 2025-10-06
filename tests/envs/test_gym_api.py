import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from platoon_gym.envs.platoon_env import (
    PlatoonEnvActionNormalized, PlatoonEnvErrorObs, PlatoonEnvErrorObsActionNormalized
)


def test_gym_env():
    config = {}
    render_mode = None
    env = gym.make("platoon_env-v0", config=config, render_mode=render_mode)
    check_env(env.unwrapped)
    env.close()


def test_normalized_action_gym_env():
    config = {}
    render_mode = None
    env = gym.make("platoon_env-v0", config=config, render_mode=render_mode)
    env = PlatoonEnvActionNormalized(env)
    check_env(env.unwrapped)
    env.close()


def test_error_obs_gym_env():
    config = {}
    render_mode = None
    env = gym.make("platoon_env-v0", config=config, render_mode=render_mode)
    env = PlatoonEnvErrorObs(env)
    check_env(env.unwrapped)
    env.close()


def test_error_obs_normalized_action_gym_env():
    config = {}
    render_mode = None
    env = gym.make("platoon_env-v0", config=config, render_mode=render_mode)
    env = PlatoonEnvErrorObsActionNormalized(env)
    check_env(env.unwrapped)
    env.close()


if __name__ == "__main__":
    print("\nTesting gym environment\n")
    test_gym_env()
    print("\nTesting normalized action gym environment\n")
    test_normalized_action_gym_env()
    print("\nTesting error observation gym environment\n")
    test_error_obs_gym_env()
    print("\nTesting error observation and normalized action gym environment\n")
    test_error_obs_normalized_action_gym_env()
