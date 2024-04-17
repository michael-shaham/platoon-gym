import numpy as np

from platoon_gym.dyn.dynamics_base import DynamicsBase
from platoon_gym.dyn.utils import forward_euler_discretization


class DoubleIntegrator(DynamicsBase):
    """
    Double integrator dynamics model for longitudinal vehicle dynamics. In this
    model, the state and output are given by (position, velocity) and the input
    is the acceleration.
    """

    def __init__(self, dt: float, x_lims: np.ndarray, u_lims: np.ndarray):
        super().__init__(dt, x_lims, u_lims)

        # continuous-time dynamics
        Ac = np.array([[0, 1], [0, 0]])
        Bc = np.array([[0], [1]])

        self.n = Ac.shape[0]
        self.m = Bc.shape[1]

        # discretization
        self.Ad, self.Bd = forward_euler_discretization(Ac, Bc, dt)

        self.C = np.eye(self.n)
        self.p = self.C.shape[0]

        assert x_lims.shape == (self.n, 2)
        assert u_lims.shape == (self.m, 2)

    def forward(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.Ad @ x + self.Bd @ u

    def sense(self, x: np.ndarray) -> np.ndarray:
        return self.C @ x
