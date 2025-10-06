import numpy as np

from platoon_gym.ctrl.controller_base import ControllerBase


class IntelligentDriverModel(ControllerBase):
    """
    Intelligent Driver Model (IDM) controller. Uses a set of parameters to
    select a desired acceleration based on distance to preceding vehicle,
    current speed, and relative speed with respect to preceding vehicle.
    """

    def __init__(
        self,
        s0: float = 1.0,
        v0: float = 35.0,
        T: float = 1.0,
        delta: float = 4.0,
        a: float = 1.5,
        b: float = 3.0,
    ):
        """
        Initializes the controller with the given parameters. Default parameters
        are set to values that will cause shockwaves (or stop-and-go waves)
        reasonably quickly.

        Args:
            s0: minimum bumper-to-bumper distance [m]
            v0: desired speed [m/s]
            T: desired time headway [s]
            delta: acceleration exponent
            a: acceleration exponent
            b: acceleration exponent
        """
        self.s0 = s0
        self.v0 = v0
        self.T = T
        self.delta = delta
        self.a = a
        self.b = b

    def control(self, delta_s: float, v: float, delta_v: float) -> float:
        """
        Returns a control input based on the IDM controller.

        Args:
            delta_s: distance to preceding vehicle [m]
            v: current speed [m/s]
            delta_v: relative speed with respect to preceding vehicle [m/s]

        Returns:
            float: desired acceleration [m/s^2]
        """
        s_star = self.s0 + max(
            0, v * self.T - v * delta_v / (2 * np.sqrt(self.a * self.b))
        )
        return np.array(
            [
                self.a
                * (
                    1
                    - (v / self.v0) ** self.delta
                    - (s_star / (max(1e-3, delta_s))) ** 2
                )
            ]
        )
