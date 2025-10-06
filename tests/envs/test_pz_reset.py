import numpy as np

import platoon_gym
from platoon_gym.ctrl.linear_feedback import LinearFeedback


def test_pz_reset():
    reset_time = 30

    config = {"reset_time": reset_time}
    render_mode = "human"
    env_args = {"config": config, "render_mode": render_mode}
    env = platoon_gym.parallel_pz_env(**env_args)
    obs, _ = env.reset()

    count = 0
    total_count = 2
    ctrl = LinearFeedback(np.array([[2.0, 4.0]]))
    while True:
        try:
            actions = {}
            for a in env.agents:
                o = obs[a]
                action = ctrl(o)[0] / 3
                action = np.clip(action, -1, 1)
                actions[a] = action

            obs, _, term, trunc, _ = env.step(actions)
            done = any(term.values()) or any(trunc.values())
            if done:
                obs, _ = env.reset()
                count += 1
            if count == total_count:
                break
        except KeyboardInterrupt:
            break
    env.close()


if __name__ == "__main__":
    test_pz_reset()
