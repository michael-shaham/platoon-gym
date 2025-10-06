import numpy as np

from platoon_gym.ctrl.controller_base import ControllerBase


class LinearFeedback(ControllerBase):
    """
    Linear feedback controller. Uses negative sum of positive gains multiplied
    by error signals to determine control action.
    """

    def __init__(
        self, 
        k: np.ndarray, 
        distance_headway: float,
        time_headway: float,
        saturate: bool = False, 
        control_min: float = -float('inf'),
        control_max: float = float('inf')
    ) -> None:
        """
        Attributes:
            k: shape (m, n), n is state (or error vector) size, m is input size
            distance_headway: distance headway
            time_headway: time headway
            saturate: whether to saturate the control input
            control_min: minimum control input
            control_max: maximum control input
        """
        if k.ndim == 1:
            k = k.reshape(1, -1)
        assert k.ndim == 2
        self.k = k
        self.distance_headway = distance_headway
        self.time_headway = time_headway
        self.saturate = saturate
        self.u_min = control_min
        self.u_max = control_max

        # controller state
        self.e = None

    def control(self, **kwargs) -> tuple[np.ndarray, dict]:
        """
        Returns a control input based on a linear feedback control policy.

        Args:
            **kwargs: unused
        
        Returns:
            tuple: control input and an empty dictionary
        """
        return np.clip(self.k @ self.e, self.u_min, self.u_max), {}

    def reset(self, veh_state: np.ndarray, obs: np.ndarray, **kwargs) -> None:
        """
        Reset the controller state if necessary (occurs after environment 
        reset).

        Args:
            veh_state: shape (n,), vehicle state provided by environment
            obs: shape (p,), observation from the environment
            **kwargs: additional kwargs
        """
        p = len(obs)
        assert p == self.k.shape[1]
        d_des = self.time_headway * veh_state[1] + self.distance_headway
        self.e = np.zeros(p)
        self.e[0] = obs[0] - d_des
        self.e[1] = obs[1]
        if p > 2:
            self.e[2:] = obs[2:]
    
    def step(self, veh_state: np.ndarray, obs: np.ndarray, **kwargs) -> None:
        """
        Update the controller state if necessary (occurs after all vehicles 
        in platoon have calculated their action for the current timestep, and 
        the simulation Gym environment has been stepped).

        Args:
            veh_state: shape (n,), vehicle state provided by environment
            obs: shape (p,), observation from the environment
            **kwargs: additional kwargs
        """
        p = len(obs)
        assert p == self.k.shape[1]
        d_des = self.time_headway * veh_state[1] + self.distance_headway
        self.e = np.zeros(p)
        self.e[0] = obs[0] - d_des
        self.e[1] = obs[1]
        if p > 2:
            self.e[2:] = obs[2:]