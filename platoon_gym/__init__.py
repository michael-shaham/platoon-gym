from gymnasium.envs.registration import register

register(id="platoon_env-v0", entry_point="platoon_gym.envs.platoon_env:PlatoonEnv")
register(id="ring_env-v0", entry_point="platoon_gym.envs.ring_env:RingEnv")
