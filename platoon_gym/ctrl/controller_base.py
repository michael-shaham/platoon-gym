from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class ControllerBase(ABC):
    """
    Base vehicle controller class.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def control(self) -> Tuple[np.ndarray, dict]:
        """
        Compute the control action for the vehicle. This method should return
        the selected action as well as a dict containing any additional info.

        Returns:
            np.ndarray: shape (m,), the selected action
            dict: additional info (e.g., MPC plan)
        """
        pass
