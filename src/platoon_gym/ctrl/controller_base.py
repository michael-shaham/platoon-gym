from abc import ABC, abstractmethod

import numpy as np


class ControllerBase(ABC):
    """
    Base vehicle controller class.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def control(self, *args, **kwargs) -> tuple[np.ndarray, dict]:
        """
        Compute the control action for the vehicle. This method should return
        the selected action as well as a dict containing any additional info.

        Args:
            *args: args to pass to control method
            **kwargs: kwargs to pass to control method

        Returns:
            np.ndarray: shape (m,), the selected action
            dict: additional info (e.g., MPC plan)
        """
        pass

    def __call__(self, *args, **kwargs) -> tuple[np.ndarray, dict]:
        """
        Convenience method to call the control method.

        Args:
            *args: args to pass to control method
            **kwargs: kwargs to pass to control method

        Returns:
            np.ndarray: shape (m,), the selected action
            dict: additional info (e.g., MPC plan)
        """
        return self.control(*args, **kwargs)
    
    def reset(self, veh_state: np.ndarray, obs: np.ndarray, **kwargs):
        """
        Reset the controller state if necessary (occurs after environment 
        reset).

        Args:
            veh_state: vehicle state provided by environment
            obs: observation from the environment
            **kwargs: additional kwargs
        """
        pass

    def step(self, veh_state: np.ndarray, obs: np.ndarray, **kwargs):
        """
        Update the controller state if necessary (occurs after all vehicles 
        in platoon have calculated their action for the current timestep).

        Args:
            veh_state: vehicle state provided by environment
            obs: observation from the environment
            **kwargs: additional kwargs
        """
        pass