"""
Linear programming and mixed integer linear programming methods for neural 
network verification.
"""

import cvxpy as cp
import numpy as np
from typing import Any, List, Tuple

from platoon_gym.nn_models.feedforward import FullyConnected
from platoon_gym.nn_verify.utils import model_to_weights_biases_acts
from platoon_gym.nn_verify.activation_bounds import ibp_preactivation_bounds


def nn_as_lp_milp(
    model: FullyConnected, input_bounds: np.ndarray, method: str, n_inputs: int = 1
) -> Tuple[
    List[cp.Variable], List[cp.Variable], List[cp.Variable], List[cp.Constraint]
]:
    """
    Initialize pre-activation, post-activation, and activation function
    decision variables for the feedforward neural network model. Only supports
    ReLU and LeakyReLU activation functions.

    Args:
        model: FullyConnected model
        input_bounds: known bounds on neural network input (box constraint)
        method: 'milp', 'lp'
        n_inputs: number of inputs we're evaluating

    Returns:
        x: input variable and post-activation variables
        z: pre-activation variables and output variable
        a: binary variable for each activation function
        constraints: list of constraints
    """
    assert method in ["milp", "lp"]
    constraints = []
    assert n_inputs >= 1 and isinstance(n_inputs, int)

    if input_bounds.ndim == 2:
        input_bounds = np.expand_dims(input_bounds, axis=0)

    # get neural network weights, biases, and activation functions
    Ws, bs, relus, lrelus = model_to_weights_biases_acts(model)

    x = [cp.Variable((W.shape[1], n_inputs)) for W in Ws]
    for i in range(n_inputs):
        constraints += [
            x[0][:, i] <= input_bounds[i, :, 1],
            x[0][:, i] >= input_bounds[i, :, 0],
        ]
    z = [cp.Variable((W.shape[0], n_inputs)) for W in Ws]
    a = []
    if method == "milp":
        for i, (relu, lrelu) in enumerate(zip(relus, lrelus)):
            if relu or lrelu:
                a.append(cp.Variable((Ws[i].shape[0], n_inputs), boolean=True))
            else:
                a.append(None)
    else:
        for i, (relu, lrelu) in enumerate(zip(relus, lrelus)):
            if relu or lrelu:
                a.append(cp.Variable((Ws[i].shape[0], n_inputs)))
                constraints += [a[-1] >= 0, a[-1] <= 1]
            else:
                a.append(None)

    # get pre-activation bounds
    ls = [np.zeros(z.shape) for z in z]
    us = [np.zeros(z.shape) for z in z]
    for i in range(n_inputs):
        ls_i, us_i = ibp_preactivation_bounds(model, input_bounds[i])
        for j, (l, u) in enumerate(zip(ls_i, us_i)):
            ls[j][:, i] = l
            us[j][:, i] = u

    L = len(Ws)  # (L-1)-layer neural network
    # iterate through each layer, add constraints
    for i in range(L):
        b = np.tile(bs[i].reshape(-1, 1), (1, n_inputs))
        constraints += [z[i] == Ws[i] @ x[i] + b]
        if relus[i]:
            constraints += relu_constraints(x[i + 1], z[i], a[i], ls[i], us[i])
        elif lrelus[i]:
            constraints += lrelu_constraints(
                x[i + 1], z[i], a[i], ls[i], us[i], lrelus[i]
            )
        elif i < L - 1:  # two consecutive linear layers
            constraints += [x[i + 1] == z[i]]

    return x, z, a, constraints


def relu_constraints(
    y: cp.Variable, x: cp.Variable, a: cp.Variable, l: np.ndarray, u: np.ndarray
) -> List[Any]:
    """
    Get constraints for ReLU activation function y = max(0, x) where
    l <= x <= u and a is the decision variable for which part of the piecewise
    linear function we are in.

    Args:
        y: post-activation variable
        x: pre-activation variable
        a: binary variable
        l: lower bounds
        u: upper bounds

    Returns:
        constraints: list of cvxpy constraints
    """
    assert x.shape == y.shape
    assert x.shape == a.shape
    assert x.shape == l.shape
    assert x.shape == u.shape
    constraints = [
        y >= x,
        y >= 0,
        y <= cp.multiply(u, a),
        y <= x - cp.multiply(l, 1 - a),
    ]
    if (l > 0).any():
        constraints += [
            a[l > 0] == 1,
            y[l > 0] == x[l > 0],
        ]
    if (u < 0).any():
        constraints += [a[u < 0] == 0, y[u < 0] == 0]
    return constraints


def lrelu_constraints(
    y: cp.Variable,
    x: cp.Variable,
    a: cp.Variable,
    l: np.ndarray,
    u: np.ndarray,
    slope: float,
) -> List[Any]:
    """
    Get constraints for leaky ReLU activation function y = max(slope * x, x)
    where l <= x <= u and a is the decision variable for which part of the
    piecewise linear function we are in.

    Args:
        y: post-activation variable
        x: pre-activation variable
        a: binary variable
        l: lower bounds
        u: upper bounds
        slope: slope of the leaky ReLU

    Returns:
        constraints: list of cvxpy constraints
    """
    assert x.shape == y.shape
    assert x.shape == a.shape
    assert x.shape == l.shape
    assert x.shape == u.shape
    assert slope >= 0 and slope < 1
    constraints = [
        y >= x,
        y >= slope * x,
        y <= slope * x - (slope - 1) * cp.multiply(u, a),
        y <= x + (slope - 1) * cp.multiply(l, 1 - a),
    ]
    if (l > 0).any():
        constraints += [
            a[l > 0] == 1,
            y[l > 0] == x[l > 0],
        ]
    if (u < 0).any():
        constraints += [a[u < 0] == 0, y[u < 0] == slope * x[u < 0]]
    return constraints


def nn_verify_lp_milp(
    model: FullyConnected, input_range: np.ndarray, C: np.ndarray, method: str
) -> Tuple[
    List[float], List[List[np.ndarray]], List[List[np.ndarray]], List[List[np.ndarray]]
]:
    """
    Calculates output bounds of fully connected ReLU network using (mixed
    integer) linear programming. Assumes ReLU or leaky ReLU activation
    functions.

    Args:
        model: FullyConnectedReLU model
        input_range: input range of the network
        c: cost matrix, each column is a cost vector to evaluate
        method: 'milp', 'lp'

    Returns:
        output_range: list of output bounds for each cost vector
        xopts: list of optimal input and post-activation variable values for
            each cost vector
        zopts: list of optimal pre-activation and output variable values for
            each cost vector
        aopts: list of optimal activation function decision variable values for
            each cost vector

    """
    assert C.ndim == 2
    assert method in ["milp", "lp"]
    x, z, a, constraints = nn_as_lp_milp(model, input_range, method)

    cs = [C[:, i] for i in range(C.shape[1])]
    out_rng = []
    xopts = []
    zopts = []
    aopts = []
    for c in cs:
        cost = c @ z[-1]
        prob = cp.Problem(cp.Minimize(cost), constraints)
        if method == "milp":
            prob.solve()
        else:
            prob.solve(solver=cp.CLARABEL)
        if prob.status != "optimal":
            print(f"Solver failed for {method} with c = {c}, status: {prob.status}")
        if (c < 0).any():
            out_rng.append(-prob.value)
        else:
            out_rng.append(prob.value)
        xopts.append([x[i].value for i in range(len(x))])
        zopts.append([z[i].value for i in range(len(z))])
        aopts.append([a[i].value for i in range(len(a))])

    return out_rng, xopts, zopts, aopts


def l1_norm_constraint(
    x: cp.Variable, x_bounds: np.ndarray, method: str
) -> Tuple[cp.Variable, cp.Variable, List[cp.Constraint]]:
    """
    Create L1 norm equality constraint for variable x. Returns the variable
    representing the absolute value of each element of x, the binary variable,
    and the constraints. Assume x is a vector.

    Args:
        x: cp.Variable
        x_bounds: np.ndarray, box constraints on x
        method: 'milp', 'lp'

    Returns:
        z: cp.Variable s.t. z_i = |x_i|, ||x||_1 = sum z_i
        a: cp.Variable, encoding binary variable
        constraints: cvxpy constraints encoding the L1 norm equality
    """
    assert x.ndim == 1
    assert x_bounds.ndim == 2
    assert method in ["milp", "lp"]
    t = cp.Variable(x.shape)
    if method == "milp":
        a = cp.Variable(x.shape, boolean=True)
        constraints = []
    else:
        a = cp.Variable(x.shape)
        constraints = [a >= 0, a <= 1]
    # 1-norm maximization constraints
    constraints += [
        t >= x,
        t >= -x,
        t <= x + 2 * cp.multiply(x_bounds[:, 0], a - 1),
        t <= 2 * cp.multiply(x_bounds[:, 1], a) - x,
    ]
    return t, a, constraints
