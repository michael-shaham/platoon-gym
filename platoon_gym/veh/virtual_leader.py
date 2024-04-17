"""
Virtual leader class that provides a reference trajectory for platoon.
"""

import numpy as np
from typing import Optional

from platoon_gym.veh.utils import VL_TRAJECTORY_TYPES
from platoon_gym.ctrl.vl_reference import velocity_step


class VirtualLeader:
    """
    Virtual leader that (at least) the platoon leader has access to.

    Attributes:
       state: np.ndarray, state (p, v, a) of the virtual leader
       H: int, prediction horizon
       dt: float, time step
       plan: np.ndarray, plan of the virtual leader
       time: float, current time
       time_forecast: np.ndarray, time prediction horizon which is planned over
    """

    def __init__(
        self,
        trajectory_type: str,
        trajectory_args: dict,
        position: float = 0.0,
        velocity: float = 0.0,
        acceleration: float = 0.0,
    ):
        """
        Initialize the virtual leader with its trajectory type and initial state.

        Args:
            trajectory_type: str, type of trajectory to follow
            trajectory_args: dict, arguments for the trajectory
            position: float, initial position of the virtual leader
            velocity: float, initial velocity of the virtual leader
            acceleration: float, initial acceleration of the virtual leader
        """
        self.traj_type = trajectory_type
        self.traj_args = trajectory_args
        self.init_state = np.array([position, velocity, acceleration])
        self.rng = (
            np.random.default_rng(self.traj_args["seed"])
            if "seed" in self.traj_args
            else np.random.default_rng()
        )

    def reset(
        self,
        trajectory_type: Optional[str] = None,
        trajectory_args: Optional[dict] = None,
        position: Optional[float] = None,
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
    ):
        if trajectory_type is None:
            trajectory_type = self.traj_type
            assert trajectory_type in VL_TRAJECTORY_TYPES
        if trajectory_args is None:
            trajectory_args = self.traj_args
        if position is None:
            position = self.init_state[0]
        if velocity is None:
            velocity = self.init_state[1]
        if acceleration is None:
            acceleration = self.init_state[2]
        self.traj_type = trajectory_type
        self.init_state = np.array([position, velocity, acceleration])
        self.state = self.init_state.copy()
        self.traj_args = trajectory_args
        self.H = trajectory_args["horizon"]
        self.dt = trajectory_args["dt"]
        self.plan = np.zeros((len(self.state), self.H + 1))
        self.plan[:, 0] = self.state
        self.time = 0.0
        self.time_forecast = np.arange(self.H + 1) * self.dt
        self.init_traj()

    def init_traj(self):
        """Initialize the trajectory."""
        if self.traj_type == "constant velocity":
            self.plan[0, :] = self.state[0] + self.state[1] * self.time_forecast
            self.plan[1, :] = self.state[1]

        elif self.traj_type == "velocity step":
            assert "step time" in self.traj_args, "velocity step time not specified"
            assert "step velocity" in self.traj_args, "final velocity not specified"
            step_time = self.traj_args["step time"]
            step_velocity = self.traj_args["step velocity"]
            if "step acceleration" in self.traj_args:
                step_accel = self.traj_args["step acceleration"]
                assert np.sign(step_accel) == np.sign(step_velocity - self.state[1])
            else:
                step_accel = None

            if step_accel is None:
                # move with constant velocity (but velocity changes once)
                step_index = int(step_time / self.dt)
                self.plan[1, :step_index] = self.state[1]
                self.plan[1, step_index:] = step_velocity
                for k in range(1, self.H + 1):
                    self.plan[0, k] = (
                        self.plan[0, k - 1] + self.plan[1, k - 1] * self.dt
                    )
            else:
                accel_time = abs((step_velocity - self.state[1]) / step_accel)
                self.plan, _ = velocity_step(
                    v_init=self.state[1],
                    v_des=step_velocity,
                    accel_time=accel_time,
                    accel_start_time=step_time,
                    total_time=self.H * self.dt,
                    dt=self.dt,
                )

        elif self.traj_type == "random step":
            assert "step time min" in self.traj_args
            assert "step time max" in self.traj_args
            step_time_min = self.traj_args["step time min"]
            step_time_max = self.traj_args["step time max"]
            assert step_time_max >= step_time_min
            assert "step velocity min" in self.traj_args
            assert "step velocity max" in self.traj_args
            step_velocity_min = self.traj_args["step velocity min"]
            step_velocity_max = self.traj_args["step velocity max"]
            assert step_velocity_max >= step_velocity_min
            assert (
                "step acceleration magnitude min" in self.traj_args
                and "step acceleration magnitude max" in self.traj_args
            ) or (
                "step acceleration magnitude min" not in self.traj_args
                and "step acceleration magnitude max" not in self.traj_args
            )

            step_time = self.rng.uniform(step_time_min, step_time_max)
            step_velocity = self.rng.uniform(step_velocity_min, step_velocity_max)
            if (
                "step acceleration magnitude min" in self.traj_args
                and "step acceleration magnitude max" in self.traj_args
            ):
                step_acc_min = self.traj_args["step acceleration magnitude min"]
                step_acc_max = self.traj_args["step acceleration magnitude max"]
                assert step_acc_max >= 0 and step_acc_min >= 0
                assert step_acc_max >= step_acc_min
                step_accel = self.rng.uniform(step_acc_min, step_acc_max)
            else:
                step_accel = None

            if step_accel is None:
                # move with constant velocity (but velocity changes once)
                step_index = int(step_time / self.dt)
                self.plan[1, :step_index] = self.state[1]
                self.plan[1, step_index:] = step_velocity
                for k in range(1, self.H + 1):
                    self.plan[0, k] = (
                        self.plan[0, k - 1] + self.plan[1, k - 1] * self.dt
                    )
            else:
                accel_time = abs(step_velocity - self.state[1]) / step_accel
                self.plan, _ = velocity_step(
                    v_init=self.state[1],
                    v_des=step_velocity,
                    accel_time=accel_time,
                    accel_start_time=step_time,
                    total_time=self.H * self.dt,
                    dt=self.dt,
                )

        else:
            raise NotImplementedError

    def step(self):
        """Step the virtual leader by one time step."""
        if self.traj_type == "constant velocity":
            self.time += self.dt
            self.time_forecast += self.dt
            self.plan[:, :-1] = self.plan[:, 1:].copy()
            self.plan[0, -1] = self.plan[0, -2] + self.dt * self.plan[1, -2]
            self.state[0] += self.state[1] * self.dt
        elif self.traj_type == "velocity step":
            self.time += self.dt
            self.time_forecast += self.dt
            self.plan[:, :-1] = self.plan[:, 1:].copy()
            self.plan[0, -1] = self.plan[0, -2] + self.dt * self.plan[1, -2]
            self.plan[1, -1] = self.plan[1, -2]
            self.state = self.plan[:, 0].copy()
        elif self.traj_type == "random step":
            self.time += self.dt
            self.time_forecast += self.dt
            self.plan[:, :-1] = self.plan[:, 1:].copy()
            self.plan[0, -1] = self.plan[0, -2] + self.dt * self.plan[1, -2]
            self.state = self.plan[:, 0].copy()
        else:
            raise NotImplementedError
