import numpy as np
from typing import Tuple

from platoon_gym.ctrl.controller_base import ControllerBase


class LinearFeedback(ControllerBase):
    """
    Linear feedback controller. Uses negative sum of positive gains multiplied
    by error signals to determine control action.

    Attributes:
        k: shape (m, n), n is state (or error vector) size, m is input size
    """

    def __init__(self, k: np.ndarray):
        assert k.ndim == 2
        self.k = k

    def control(self, e: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Returns a control input based on a linear feedback control policy.

        Args:
            e: shape (n,), error or state (if driving state to zero) vector

        Returns:
            np.ndarray, shape (m,): control input
            dict: empty
        """
        return self.k @ e, {}
