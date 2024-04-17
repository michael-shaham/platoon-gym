import numpy as np

from platoon_gym.ctrl.linear_feedback import LinearFeedback


def test_lfbk():
    k = np.ones((1, 3))
    lfbk = LinearFeedback(k)
    e = np.ones(3)
    assert np.allclose(k @ e, lfbk.control(e)[0])

    k = np.ones((1, 3))
    lfbk = LinearFeedback(k)
    e = np.ones((3, 1))
    assert np.allclose(k @ e, lfbk.control(e)[0])

    k = np.ones((2, 3))
    lfbk = LinearFeedback(k)
    e = np.ones(3)
    assert np.allclose(k @ e, lfbk.control(e)[0])

    k = np.ones((2, 3))
    lfbk = LinearFeedback(k)
    e = np.ones((3, 1))
    assert np.allclose(k @ e, lfbk.control(e)[0])
