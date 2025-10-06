"""Vehicle class for platoon_gym environment.

This class contains all vehicle information needed from the platooning 
environment."""

from typing import Optional
import numpy as np

from platoon_gym.dyn.dynamics_base import DynamicsBase


class Vehicle:
    """
    Vehicle class. Contains a dynamics model and the vehicle's state/output.

    Attributes:
        dyn: Derived[DynamicsBase] dynamics class derived from DynamicsBase
        position: float, longitudinal position on road [m]
        velocity: float, longitudinal velocity (signed) [m/s]
        acceleration: float, longitudinal acceleration (signed) [m/s^2],
            optional
    """

    def __init__(
        self,
        dyn: DynamicsBase,
        position: float = 0.0,
        velocity: float = 0.0,
        acceleration: Optional[float] = None,
    ):
        """
        Initialize the vehicle with its dynamics model and starting state.
        """
        self.dyn = dyn
        self.dt = self.dyn.dt
        self.n, self.m, self.p = dyn.n, dyn.m, dyn.p
        if acceleration is not None:
            self.init_state = np.array([position, velocity, acceleration])
            self.state = self.init_state.copy()
        else:
            self.init_state = np.array([position, velocity])
            self.state = self.init_state.copy()
        self.output = np.array([position, velocity])

    def reset(
        self,
        position: Optional[float] = None,
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
    ):
        """
        Reset the vehicle to new given state if given or to initial state.

        Args:
            position: float, longitudinal position on road [m]
            velocity: float, longitudinal velocity (signed) [m/s]
            acceleration: float, longitudinal acceleration (signed) [m/s^2],
                optional
        """
        if position is not None:
            self.init_state[0] = position
        if velocity is not None:
            self.init_state[1] = velocity
        if acceleration is not None and self.n == 3:
            self.init_state[2] = acceleration
        self.state = self.init_state.copy()
        self.output = self.init_state.copy()

    def step(self, control: np.ndarray):
        """
        Step the vehicle dynamics forward one time step based on the
        control input given by control.

        Args:
            control: np.ndarray, control input to the vehicle dynamics model
        """
        self.state = self.dyn.forward(self.state, control)
        self.output = self.dyn.sense(self.state)
