# gym registration

from gymnasium.envs.registration import register

register(
    id="platoon_env-v0",
    entry_point="platoon_gym.envs.platoon_env:PlatoonEnv",
)

# environment setup

from platoon_gym.envs.platoon_env_pz import pz_env, parallel_pz_env
from platoon_gym.envs.platoon_env import PlatoonEnv

__all__ = ["PlatoonEnv", "pz_env", "parallel_pz_env"]
