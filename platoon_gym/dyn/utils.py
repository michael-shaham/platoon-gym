import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad_vec
from typing import Tuple


DISCRETIZATION_METHODS = ["forward euler", "piecewise constant input"]

DYNAMICS_OPTIONS = ["linear acceleration", "linear velocity"]


def pw_const_input_discretization(
    A: np.ndarray, B: np.ndarray, dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretization of linear continuous-time dynamics using piecewise constant
    inputs. See Stanford EE263, lecture on LDS with iputs & outputs for details.

    :param A: shape (n, n), continuous-time dynamics matrix
    :param B: shape (n, m), continuous-time input matrix
    :param dt: discretization timestep
    :return: shape (n, n), shape (n, m), discrete-time system matrices
    """

    def integrand(t, A, B):
        return expm(A * t) @ B

    return expm(A * dt), quad_vec(integrand, 0, dt, args=(A, B))[0]


def forward_euler_discretization(
    A: np.ndarray, B: np.ndarray, dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretization of linear continuous-time dynamics using the simpler (but
    less accurate) forward Euler method.

    :param A: shape (n, n), continuous-time dynamics matrix
    :param B: shape (n, m), continuous-time input matrix
    :param dt: discretization timestep
    :return: shape (n, n), shape (n, m), discrete-time system matrices
    """
    return np.eye(A.shape[0]) + dt * A, dt * B
