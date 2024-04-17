import numpy as np
import torch
from typing import Optional, Union

from platoon_gym.dyn.double_integrator import DoubleIntegrator


class PlatoonDoubleIntegrator:
    """
    Dynamics for a platoon of N vehicles modeled as double integrators where
    all N vehicles have the same control and dynamic limits.

    Attributes:
        N: int, number of vehicles
        dt: float, discrete timestep
        x_lims: np.ndarray, shape (n, 2), state limits
        u_lims: np.ndarray, shape (m, 2), control limits
    """

    def __init__(
        self,
        N: int,
        dt: float,
        x_lims: Optional[np.ndarray] = None,
        u_lims: Optional[np.ndarray] = None,
        use_torch: Optional[bool] = None,
        device: Optional[str] = None,
    ):
        self.N = N
        if x_lims is None:
            x_lims = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
        if u_lims is None:
            u_lims = np.array([[-np.inf, np.inf]])
        self.x_lims, self.u_lims = x_lims, u_lims
        double_int = DoubleIntegrator(dt, x_lims, u_lims)

        self.A_forw = np.kron(np.eye(N), double_int.Ad)
        self.B_forw = np.kron(np.eye(N), double_int.Bd)

        n, m = double_int.n, double_int.m
        self.A_err = np.kron(np.eye(N), double_int.Ad)
        self.B_err = -np.kron(np.eye(N), double_int.Bd)
        self.B_err[n:, :-m] += np.kron(np.eye(N - 1), double_int.Bd)

        if use_torch:
            assert device is not None
            self.A_forw = torch.from_numpy(self.A_forw).float().to(device)
            self.B_forw = torch.from_numpy(self.B_forw).float().to(device)
            self.A_err = torch.from_numpy(self.A_err).float().to(device)
            self.B_err = torch.from_numpy(self.B_err).float().to(device)

    def forward(
        self, x: Union[np.ndarray, torch.Tensor], u: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        return self.A_forw @ x + self.B_forw @ u

    def forward_error(
        self, x: Union[np.ndarray, torch.Tensor], u: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        return self.A_err @ x + self.B_err @ u
