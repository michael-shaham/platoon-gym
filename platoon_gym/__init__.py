from gymnasium.envs.registration import register

register(id="platoon_gym-v0", entry_point="platoon_gym.envs.platoon_env:PlatoonEnv")
