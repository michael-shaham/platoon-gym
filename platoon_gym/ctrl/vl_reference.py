"""
Various modules for the virtual leader trajectory.
"""

import numpy as np
from typing import Tuple


def velocity_step(
    v_init: float,
    v_des: float,
    accel_time: float,
    accel_start_time: float,
    total_time: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a reference trajectory that accelerates from v_init to v_des in
    accel_time seconds, starting at accel_start_time.

    Args:
        v_init: float, initial velocity
        v_des: float, desired velocity
        accel_time: float, time to reach v_des from v_init
        accel_start_time: float, time to start accelerating
        total_time: float, total time for trajectory
        dt: float, timestep

    Returns:
        np.ndarray: shape (n, T), reference states at timesteps
        np.ndarray: shape (T,), corresponding timesteps
    """
    assert accel_start_time < total_time
    accel_time = round(accel_time / dt) * dt
    accel_start_time = round(accel_start_time / dt) * dt
    t_range = np.linspace(
        start=0, stop=total_time, num=round(total_time / dt) + 1, endpoint=True
    )
    T = len(t_range)

    accel_end_time = accel_start_time + accel_time
    accel = (v_des - v_init) / accel_time

    mask1 = t_range <= accel_start_time + dt / 2
    mask2 = np.logical_and(
        t_range > accel_start_time + dt / 2, t_range <= accel_end_time + dt / 2
    )
    mask3 = t_range > accel_end_time + dt / 2

    v_ref = np.zeros(T)
    v_ref[mask1] = v_init
    v_ref[mask2] = v_init + accel * (t_range[mask2] - accel_start_time)
    v_ref[mask3] = v_des

    a_ref = np.zeros(T)
    a_ref[mask2] = accel

    s_ref = np.zeros(T)
    s_ref[mask1] = v_init * t_range[mask1]
    s_ref[mask2] = (
        0.5 * accel * (t_range[mask2] - accel_start_time) ** 2
        + v_init * (t_range[mask2] - accel_start_time)
        + s_ref[mask1][-1]
    )
    s_ref[mask3] = v_des * (t_range[mask3] - accel_end_time) + s_ref[mask2][-1]

    x_ref = np.r_[s_ref, v_ref, a_ref].reshape((3, -1))
    return x_ref, t_range


def sinusoidal_ref(
    T: float, v_min: float, v_max, dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a reference trajectory that follows a sinusoidal curve with a
    period of T seconds that alternates between v_min and v_max.

    Args:
        T: period (s)
        v_min: min speed (m/s)
        v_max: max speed (m/s)

    Returns:
        np.ndarray: shape (n, T), reference states at timesteps
        np.ndarray: shape (T,), corresponding timesteps
    """
    A = (v_max - v_min) / 2
    v_0 = (v_max + v_min) / 2
    t_range = np.arange(0, T + dt / 2, dt)

    s_ref = v_0 * t_range - A * T / (2 * np.pi) * np.sin(2 * np.pi / T * t_range)
    v_ref = -A * np.cos(2 * np.pi / T * t_range) + v_0
    a_ref = 2 * A * np.pi / T * np.sin(2 * np.pi / T * t_range)

    x_ref = np.r_[s_ref, v_ref, a_ref].reshape((3, -1))
    return x_ref, t_range


def accel_decel_step(
    total_time: float,
    dt: float,
    accel_start: float,
    accel_end: float,
    decel_start: float,
    decel_end: float,
    v_low: float,
    v_high: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a reference trajectory that accelerates from v_low to v_high
    then decelerates back to v_low.

    Args:
        total_time: float, total time for the reference trajectory (s)
        dt: float, discrete timestep (s)
        accel_start: float,time when acceleration starts (s)
        accel_end: float, time when acceleration ends (s)
        decel_start: float, time when deceleration starts (s)
        decel_end: float, time when deceleration ends (s)
        v_low: float, starting velocity for the vehicle (m/s)
        v_high: float, velocity the vehicle keeps before decelerating (m/s)

    Returns:
        np.ndarray: shape (n, T), reference states at timesteps
        np.ndarray: shape (T,), corresponding timesteps
    """
    assert accel_start < accel_end, "start must be before end"
    assert accel_end < decel_start, "must accelerate before slowing down"
    assert decel_start < decel_end, "start must be before end"
    assert decel_end < total_time, "end time must be less than total time"

    t_range = np.arange(start=0.0, stop=total_time + dt, step=dt)
    T = len(t_range)

    mask1 = np.logical_and(t_range >= accel_start, t_range <= accel_end)
    mask2 = np.logical_and(t_range >= accel_end, t_range <= decel_start)
    mask3 = np.logical_and(t_range >= decel_start, t_range <= decel_end)

    accel = (v_high - v_low) / (accel_end - accel_start)
    decel = (v_low - v_high) / (decel_end - decel_start)

    v_ref = np.zeros(T)
    v_ref[t_range <= accel_start] = v_low
    v_ref[mask1] = v_low + accel * (t_range[mask1] - t_range[mask1][0])
    v_ref[mask2] = v_high
    v_ref[mask3] = v_high + decel * (t_range[mask3] - t_range[mask3][0])
    v_ref[t_range > decel_end] = v_low

    a_ref = np.zeros(T)
    a_ref[mask1] = accel
    a_ref[mask3] = decel

    s_ref = np.zeros(T)
    s_ref[t_range <= accel_start] = v_low * t_range[t_range <= accel_start]
    s_ref[mask1] = (
        0.5 * accel * (t_range[mask1] - accel_start) ** 2
        + v_low * (t_range[mask1] - accel_start)
        + s_ref[t_range <= accel_start][-1]
    )
    offset = s_ref[mask1][-1] - v_high * t_range[mask2][0]
    s_ref[mask2] = v_high * t_range[mask2] + offset
    s_ref[mask3] = (
        0.5 * decel * (t_range[mask3] - decel_start) ** 2
        + v_high * (t_range[mask3] - decel_start)
        + s_ref[t_range <= decel_start][-1]
    )
    s_ref[t_range > decel_end] = (
        v_low * (t_range[t_range > decel_end] - decel_end) + s_ref[mask3][-1]
    )

    x_ref = np.r_[s_ref, v_ref, a_ref].reshape((3, -1))
    return x_ref, t_range


def accel_step(
    v_des: float, accel_time: float, dt: float, total_time: float, v_init: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a reference trajectory that accelerates from zero initial state
    (zero pos, vel, accel) to the desired velocity in des_time time.

    Args:
        v_des: float, desired velocity to reach from zero velocity
        accel_time: float, time to reach v_des from v = 0
        dt: float, timestep
        total_time: float, total time for trajectory
        v_init: starting speed if not zero

    Returns:
        np.ndarray: shape (n, T), reference states at timesteps
        np.ndarray: shape (T,), corresponding timesteps
    """
    t_range = np.arange(start=0.0, stop=total_time + dt / 2, step=dt)
    if accel_time == 0.0:
        v_ref = np.ones_like(t_range) * v_init
        a_ref = np.zeros_like(t_range)
        s_ref = t_range * v_ref
        x_ref = np.r_[s_ref, v_ref, a_ref].reshape((3, -1))
        return x_ref, t_range

    T = len(t_range)
    a_des = (v_des - v_init) / accel_time
    mask1 = t_range <= accel_time
    mask2 = t_range > accel_time

    v_ref = np.zeros(T)
    v_ref[mask1] = a_des * (t_range[mask1]) + v_init
    v_ref[mask2] = v_des

    a_ref = np.zeros(T)
    a_ref[mask1] = a_des

    s_ref = np.zeros(T)
    s_ref[mask1] = 0.5 * a_des * t_range[mask1] ** 2 + v_init * t_range[mask1]
    s_ref[mask2] = v_des * (t_range[mask2] - t_range[mask1][-1]) + s_ref[mask1][-1]
    x_ref = np.r_[s_ref, v_ref, a_ref].reshape((3, -1))

    return x_ref, t_range


def brake_step(
    decel: float, dt: float, total_time: float, v_init: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a reference trajectory that decelerates from init vel/accel
    to the desired velocity in des_time time.

    Args:
        decel: float < 0, rate at which to decelerate
        dt: float, timestep
        total_time: float, total time for trajectory
        v_init: float, starting speed
        a_init: float, starting accel

    Returns:
        np.ndarray: shape (n, T), reference states at timesteps
        np.ndarray: shape (T,), corresponding timesteps
    """
    assert decel < 0, "decel must be negative"
    assert v_init > 0, "init vel must be positive"
    t_range = np.arange(start=0.0, stop=total_time + dt / 2, step=dt)
    T = len(t_range)
    decel_time = -v_init / decel
    mask1 = t_range <= decel_time
    mask2 = t_range > decel_time

    v_ref = np.zeros(T)
    v_ref[mask1] = decel * t_range[mask1] + v_init
    # v_ref[mask2] = 0

    a_ref = np.zeros(T)
    a_ref[mask1] = decel
    # a_ref[mask2] = 0

    s_ref = np.zeros(T)
    s_ref[mask1] = 0.5 * decel * t_range[mask1] ** 2 + v_init * t_range[mask1]
    if len(s_ref[mask1]) > 0:
        s_ref[mask2] = s_ref[mask1][-1]
    x_ref = np.r_[s_ref, v_ref, a_ref].reshape((3, -1))

    return x_ref, t_range
