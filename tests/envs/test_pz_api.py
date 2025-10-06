from pettingzoo.test import api_test, parallel_api_test

import platoon_gym


def test_aec_env():
    env = platoon_gym.pz_env()
    api_test(env)
    env.close()


def test_parallel_env():
    env = platoon_gym.parallel_pz_env()
    parallel_api_test(env)
    env.close()


if __name__ == "__main__":
    test_aec_env()
    test_parallel_env()
