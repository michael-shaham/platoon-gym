import numpy as np

import platoon_gym
from platoon_gym.veh.virtual_leader import VirtualLeader


def test_pz_aec():
    vl = VirtualLeader("constant_velocity", {"horizon": 10, "dt": 0.1}, velocity=20.0)
    config = {"reset_time": 10, "virtual_leader": vl}
    render_mode = "human"
    env_args = {"config": config, "render_mode": render_mode}
    env = platoon_gym.pz_env(**env_args)
    env.reset()
    while True:
        try:
            for _ in env.agents:
                env.step(np.array([0.0]))
            done = any(env.terminations.values()) or any(env.truncations.values())
            if done:
                break
        except KeyboardInterrupt:
            break
    env.close()


def test_pz_parallel():
    vl = VirtualLeader("constant_velocity", {"horizon": 10, "dt": 0.1}, velocity=20.0)
    config = {"reset_time": 10, "virtual_leader": vl}
    render_mode = "human"
    env_args = {"config": config, "render_mode": render_mode}
    env = platoon_gym.parallel_pz_env(**env_args)
    env.reset()
    while True:
        try:
            actions = {a: np.array([0.0]) for a in env.agents}
            _, _, term, trunc, _ = env.step(actions)
            done = any(term.values()) or any(trunc.values())
            if done:
                break
        except KeyboardInterrupt:
            break
    env.close()


if __name__ == "__main__":
    test_pz_aec()
    test_pz_parallel()
