"""
Virtual leader class that provides a reference trajectory for platoon.
"""

from functools import partial
import h5py
import numpy as np
from scipy.interpolate import interp1d

from platoon_gym.veh.utils import VL_TRAJECTORY_TYPES
from platoon_gym.ctrl.vl_reference import velocity_step

from i24_forecasting.utils.general_utils import get_learning_data_dir


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
        if self.traj_type == "i24_trajectory":
            data_file = get_learning_data_dir() / "learning_test.hdf5"
            self.i24_file = h5py.File(data_file, "r")
            self.interp_func = partial(interp1d, kind="linear", fill_value="extrapolate")

    def reset(
        self,
        trajectory_type: str | None = None,
        trajectory_args: dict | None = None,
        position: float | None = None,
        velocity: float | None = None,
        acceleration: float | None = None,
        seed: int | None = None,
    ):
        if seed:
            self.rng = np.random.default_rng(seed)
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
        self.time = 0.0
        self.timestep = 0
        self.dt = trajectory_args["dt"]

        if self.traj_type == "i24_trajectory":
            self.traj_args = trajectory_args
            self.H = 0
            self.init_traj()
        else:
            self.init_state = np.array([position, velocity, acceleration])
            self.state = self.init_state.copy()
            self.traj_args = trajectory_args
            self.H = trajectory_args["horizon"] or 100
            self.plan = np.zeros((len(self.state), self.H + 1))
            self.plan[:, 0] = self.state
            self.time_forecast = np.arange(self.H + 1) * self.dt
            self.init_traj()

    def init_traj(self):
        """Initialize the trajectory."""
        if self.traj_type == "constant_velocity":
            self.plan[0, :] = self.state[0] + self.state[1] * self.time_forecast
            self.plan[1, :] = self.state[1]

        elif self.traj_type == "velocity_step":
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

        elif self.traj_type == "random_step":
            assert "step_time_min" in self.traj_args
            assert "step_time_max" in self.traj_args
            step_time_min = self.traj_args["step_time_min"]
            step_time_max = self.traj_args["step_time_max"]
            assert step_time_max >= step_time_min
            assert "step_vel_min" in self.traj_args
            assert "step_vel_max" in self.traj_args
            step_velocity_min = self.traj_args["step_vel_min"]
            step_velocity_max = self.traj_args["step_vel_max"]
            assert step_velocity_max >= step_velocity_min
            assert (
                "step_accel_min" in self.traj_args
                and "step_accel_max" in self.traj_args
            ) or (
                "step_accel_min" not in self.traj_args
                and "step_accel_max" not in self.traj_args
            )

            step_time = self.rng.uniform(step_time_min, step_time_max)
            step_velocity = self.rng.uniform(step_velocity_min, step_velocity_max)
            if (
                "step_accel_min" in self.traj_args
                and "step_accel_max" in self.traj_args
            ):
                step_acc_min = self.traj_args["step_accel_min"]
                step_acc_max = self.traj_args["step_accel_max"]
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

        elif self.traj_type == "velocity_trajectory":
            assert "velocity_trajectory" in self.traj_args
            self.velocity_trajectory = self.traj_args["velocity_trajectory"]
            self.state[0] = 0.0
            self.state[1] = self.velocity_trajectory[0]
            self.state[2] = 0.0
            self.init_state = self.state.copy()
            self.plan[1, :] = self.velocity_trajectory[: self.H + 1]
            self.plan[0, 0] = self.state[0].copy()
            self.plan[0, 1:] = (
                self.plan[0, 0] + np.cumsum(self.plan[1, : self.H]) * self.dt
            )
        
        elif self.traj_type == "i24_trajectory":
            self.plan = None  # not used
            while True:
                key = self.rng.choice(list(self.i24_file.keys()))
                velocity_trajectory = self.i24_file[key]["longitudinal_velocity"][...]
                if self.dt != 0.04:
                    i24_t = np.arange(0, len(velocity_trajectory), 1) * 0.04
                    # if i24_t[-1] < 30.0:  # ignore short trajectories
                    #     continue
                    t = np.arange(0, i24_t[-1] + self.dt, self.dt)
                    velocity_trajectory = self.interp_func(i24_t, velocity_trajectory)(t)
                    break
            self.coarse_vehicle_class = self.i24_file[key].attrs["coarse_vehicle_class"]
            self.length = self.i24_file[key].attrs["length"]
            self.width = self.i24_file[key].attrs["width"]
            self.height = self.i24_file[key].attrs["height"]
            self.i24_trajectory = np.zeros((3, len(velocity_trajectory)))
            self.i24_trajectory[0, 1:] = np.cumsum(velocity_trajectory[:-1]) * self.dt
            self.i24_trajectory[1] = velocity_trajectory
            self.state = self.i24_trajectory[:, 0].copy()

        else:
            raise NotImplementedError

    def step(self) -> bool:
        """
        Step the virtual leader by one time step.

        Returns:
            bool: indicates if virtual leader reached end of trajectory
        """
        self.time = round(self.time + self.dt, 2)
        self.timestep += 1
        if self.traj_type == "constant_velocity":
            self.plan[:, :-1] = self.plan[:, 1:].copy()
            self.plan[0, -1] = self.plan[0, -2] + self.dt * self.plan[1, -2]
            self.state[0] += self.state[1] * self.dt
        elif self.traj_type == "velocity_step":
            self.plan[:, :-1] = self.plan[:, 1:].copy()
            self.plan[0, -1] = self.plan[0, -2] + self.dt * self.plan[1, -2]
            self.plan[1, -1] = self.plan[1, -2]
            self.state = self.plan[:, 0].copy()
        elif self.traj_type == "random_step":
            self.plan[:, :-1] = self.plan[:, 1:].copy()
            self.plan[0, -1] = self.plan[0, -2] + self.dt * self.plan[1, -2]
            self.state = self.plan[:, 0].copy()
        elif self.traj_type == "velocity_trajectory":
            self.plan[:, :-1] = self.plan[:, 1:].copy()
            self.plan[0, -1] = self.plan[0, -2] + self.dt * self.plan[1, -2]
            self.plan[1, -1] = self.velocity_trajectory[self.timestep + self.H]
            self.state = self.plan[:, 0].copy()
            if self.timestep + self.H >= len(self.velocity_trajectory) - 1:
                return True
        elif self.traj_type == "i24_trajectory":
            self.state = self.i24_trajectory[:, self.timestep].copy()
            if self.timestep >= len(self.i24_trajectory[0]) - 1:
                return True
        else:
            raise NotImplementedError
        return False
