from abc import ABC, abstractmethod
import numpy as np


class DynamicsBase(ABC):
    """
    Base class for any dynamics model. All dynamics models should inherit from
    this class.

    Attributes:
        dt: float, discrete timestep
        x_lims: np.ndarray, shape (n, 2), state limits
        u_lims: np.ndarray, shape (m, 2), control limits
        n: int, state dimension
        m: int, input dimension
        p: int, sensor dimension
    """

    n: int  # state dimension
    m: int  # input dimension
    p: int  # output dimension
    tau: int  # longitudinal delay

    def __init__(self, dt: float, x_lims: np.ndarray, u_lims: np.ndarray):
        """
        Base class initialization. All derived classes will have these
        parameters.

        Parameters:
            dt: float, discrete timestep
            x_lims: np.ndarray, shape (n, 2), state limits
            u_lims: np.ndarray, shape (m, 2), control limits
        """
        self.dt = dt
        self.x_lims = x_lims
        self.u_lims = u_lims

    @abstractmethod
    def forward(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Forward dynamics functon. Returns the state at the next timestep when
        starting at a current state and applying some input.

        Args:
            x: np.ndarray, shape (n,), current state
            u: np.ndarray, shape (m,), input

        Returns:
            np.ndarray, shape (n,): state of the vehicle at the next timestep
        """
        pass

    @abstractmethod
    def sense(self, x: np.ndarray) -> np.ndarray:
        """
        Sensing function. Returns some function of the state that one may have
        access to due to some sensor.

        Args:
            x: shape (n,), current state

        Returns:
            np.ndarray, shape (p,): sensor observation
        """
        pass

    def clip_input(self, u: np.ndarray) -> np.ndarray:
        """
        Clips inputs to lie in u_lims.

        Args:
            u: shape (m,), desired input

        Returns:
            np.ndarray, shape (m,): clipped input
        """
        return np.clip(u, self.u_lims[:, 0], self.u_lims[:, 1])
