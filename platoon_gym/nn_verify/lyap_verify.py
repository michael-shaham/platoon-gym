"""
Uses LP and MILP methods to check bounds on Lyapunov neural network.
"""

import cvxpy as cp
import gurobipy as gp
import numpy as np
from typing import List, Optional, Tuple

from platoon_gym.nn_models.feedforward import FullyConnected
from platoon_gym.nn_verify.lp_milp_methods import nn_as_lp_milp, l1_norm_constraint
from platoon_gym.nn_verify.utils import model_to_weights_biases_acts


def verify_lyapunov_positivity(
    lyap_model: FullyConnected,
    init_state_bounds: np.ndarray,
    norm_weight: float,
    method: str = "milp",
    R: Optional[np.ndarray] = None,
    time_limit: int = 120,
) -> Tuple[float, np.ndarray]:
    """
    Verify that the neural network Lyapunov function is positive for all
    possible inputs in init_state_bounds. Checks this by maximizing
        norm_weight * ||Rx||_1 - lyap_model(x)
    subject to x in init_state_bounds and the neural network constraints.

    Args:
        lyap_model: neural network Lyapunov function
        init_state_bounds: bounds on input to neural network
        R: full rank matrix to calculate 1-norm of state
        norm_weight: weight on ||x||_1 term in objective function
        solve_milp: whether to use MILP or LP to solve the problem
        time_limit: time limit for optimization

    Returns:
        float: optimal value
        np.ndarray: optimal state
    """
    assert method in ["milp", "lp"]
    x, z, _, constraints = nn_as_lp_milp(lyap_model, init_state_bounds, method)
    if init_state_bounds.ndim == 2:
        init_state_bounds = np.expand_dims(init_state_bounds, axis=-1)

    # 1-norm maximization variables
    if R is None:
        R = np.eye(init_state_bounds.shape[0])
    Rx = cp.Variable(R.shape[0])
    Rx_bounds = np.hstack([R @ init_state_bounds[:, 0], R @ init_state_bounds[:, 1]])
    constraints += [Rx == (R @ x[0]).reshape(-1)]
    t, _, constraints_l1 = l1_norm_constraint(Rx, Rx_bounds, method)
    constraints += constraints_l1

    # objective
    obj = norm_weight * t.sum() - z[-1][0][0]

    # set time limit for optimization using gurobi
    env = gp.Env()
    env.setParam("TimeLimit", time_limit)
    env.setParam("Threads", 8)
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.GUROBI, env=env)

    if prob.status != cp.OPTIMAL:
        print(f"Optimization finished with status {prob.status}")
    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.USER_LIMIT]:
        assert type(prob.value) == np.float64
        return float(prob.value), x[0].value
    else:
        return np.nan, np.empty(0)


def verify_lyapunov_decreasing(
    lyap_model: FullyConnected,
    ctrl_model: FullyConnected,
    init_state_bounds: np.ndarray,
    norm_weight: float,
    dt: float,
    state_size: int,
    method: str = "milp",
    ctrl_lims: Optional[List[np.ndarray]] = None,
    time_limit: int = 120,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Verify that the neural network Lyapunov function is decreasing for all
    possible inputs (aka states) in init_state_bounds. Checks this by maximizing
        lyap_model(x_next) - lyap_model(x) + norm_weight * lyap_model(x)
    subject to x in init_state_bounds and the neural network constraints. Note
    that the x_next is calculated as following:
        action = ctrl_model(x)
        x_next = forward_method(x, action)

    Args:
        lyap_model: neural network Lyapunov function
        ctrl_model: neural network controller if that's what we're verifying
        init_state_bounds: shape (n_vehs, 2), bounds on input to neural network
        norm_weight: weight to ensure that the Lyapunov function is decreasing
        dt: float, time step
        state_size: int, dimension of state space
        method: "milp" or "lp" for mixed-integer or linear programming
        ctrl_lims: list bounds on control actions for each agent, each array is
            shape (m, 2) (if None, calculate bounds based on init_state_bounds
            and ctrl_model)
        time_limit: time limit for optimization

    Returns:
        float: optimal value
        np.ndarray: optimal initial state
        np.ndarray: optimal next state
        np.ndarray: optimal action
    """
    n = state_size
    assert init_state_bounds.shape[0] % n == 0
    n_vehs = init_state_bounds.shape[0] // n

    ctrl_bounds = []
    if ctrl_lims is None:
        raise NotImplementedError
    ctrl_bounds = np.vstack(ctrl_lims)

    assert method in ["milp", "lp"]

    # first pass through neural Lyapunov model
    x, z, _, constraints = nn_as_lp_milp(lyap_model, init_state_bounds, method)
    constraints += [
        x[0][:, 0] >= init_state_bounds[:, 0],
        x[0][:, 0] <= init_state_bounds[:, 1],
    ]

    # now calculate bounds on next state based on ctrl_bounds
    if n == 2:
        # first forward propagate the states
        Adt = np.array([[1, dt], [0, 1]])
        Aerr = np.kron(np.eye(n_vehs), Adt)
        Bdt = np.array([[0], [dt]])
        Berr = -np.kron(np.eye(n_vehs), Bdt)
        n = Adt.shape[0]
        m = Bdt.shape[1]
        Berr[n:, :-m] += np.kron(np.eye(n_vehs - 1), Bdt)

        # IBP on x
        mu_x = (init_state_bounds[:, 1] + init_state_bounds[:, 0]) / 2
        r_x = (init_state_bounds[:, 1] - init_state_bounds[:, 0]) / 2
        mu_x = Aerr @ mu_x
        r_x = np.abs(Aerr) @ r_x
        next_state_lbounds = mu_x - r_x
        next_state_ubounds = mu_x + r_x

        # IBP on u
        mu_u = (ctrl_bounds[:, 1] + ctrl_bounds[:, 0]) / 2
        r_u = (ctrl_bounds[:, 1] - ctrl_bounds[:, 0]) / 2
        mu_u = Berr @ mu_u
        r_u = np.abs(Berr) @ r_u
        next_ctrl_lbounds = mu_u - r_u
        next_ctrl_ubounds = mu_u + r_u

        next_state_lbounds += next_ctrl_lbounds
        next_state_ubounds += next_ctrl_ubounds

        next_state_bounds = np.hstack(
            [next_state_lbounds.reshape(-1, 1), next_state_ubounds.reshape(-1, 1)]
        )
    else:
        raise NotImplementedError

    # next state variables
    xnext, znext, _, constraints_next = nn_as_lp_milp(
        lyap_model, next_state_bounds, method
    )
    constraints += constraints_next

    # now for bounds on controller
    # bounds on controller network input
    ctrl_net_input_bounds = []
    Ws, _, _, _ = model_to_weights_biases_acts(ctrl_model)
    for i in range(0, n * n_vehs, n):
        ctrl_net_input_bounds.append(
            init_state_bounds[i : i + n, :].reshape((1, n, 2)),
        )
    ctrl_net_input_bounds = np.vstack(ctrl_net_input_bounds)
    xc, zc, _, constraints_ctrl = nn_as_lp_milp(
        ctrl_model, ctrl_net_input_bounds, method, n_inputs=n_vehs
    )
    constraints += constraints_ctrl
    # input to controller network is same as "current state"
    constraints += [xc[0] == x[0].reshape((-1, n_vehs), order="F")]
    # next state is calculated based on current state and controller output
    constraints += [xnext[0] == Aerr @ x[0] + Berr @ zc[-1][0].reshape((-1, 1))]

    # objective
    obj = znext[-1][0][0] - z[-1][0][0] + norm_weight * z[-1][0][0]

    # set time limit for optimization using gurobi
    env = gp.Env()
    env.setParam("TimeLimit", time_limit)
    env.setParam("Threads", 8)
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.GUROBI, env=env)

    if prob.status != cp.OPTIMAL:
        print(f"Optimization finished with status {prob.status}")
    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.USER_LIMIT]:
        xvars = [xvar.value for xvar in x]
        xvars_next = [xvar.value for xvar in xnext]
        assert type(prob.value) == np.float64
        return (float(prob.value), xvars[0], xvars_next[0], zc[-1].value[0, :])
    else:
        return np.nan, np.empty(0), np.empty(0), np.empty(0)
