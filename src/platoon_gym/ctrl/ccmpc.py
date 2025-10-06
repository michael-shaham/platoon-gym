from collections import deque
import copy
import cvxpy as cp
from functools import partial
import json
import numpy as np
from pathlib import Path
from scipy.stats import norm
import torch
import torch.nn.functional as F
from typing import Any

from i24_forecasting.data.transforms import denormalize, normalize
from i24_forecasting.train.train_utils import load_model
from i24_forecasting.utils.general_utils import get_learning_data_dir
from platoon_gym.ctrl.controller_base import ControllerBase

vmean, vstd = np.load(get_learning_data_dir() / "train_velocity_mean_std.npy")
normalize_vel = partial(normalize, mean=vmean, std=vstd)
denormalize_vel = partial(denormalize, mean=vmean, std=vstd)


class CCMPC(ControllerBase):
    """
    Chance-Constrained Model Predictive Control (CCMPC) for vehicle platooning.

    This class formulates an MPC problem that accounts for uncertainty in the 
    lead vehicle's velocity via chance constraints. The problem minimizes a 
    cost composed of position error, velocity error, and control effort while 
    ensuring that the gap between vehicles is maintained above a safe limit 
    with high probability. Also can optionally add a move suppression term 
    where we penalize our deviation from the optimal state trajectory we 
    computed at the previous timestep, as in the distributed MPC case.

    The desired gap used in the cost function is defined as:
        d_des(v) = t_h * v + d_h,
    and the safe gap used in the chance constraints is defined as:
        d_safe(v) = t_h_safe * v + d_h_safe.
    """

    def __init__(
        self, 
        model_dir: Path,
        model_name: str,
        device: str,
        A: np.ndarray, 
        B: np.ndarray, 
        qp: float, 
        qv: float, 
        r: float, 
        dt: float, 
        model_type: str, 
        t_h: float, 
        d_h: float, 
        t_h_safe: float, 
        d_h_safe: float, 
        v_min: float, 
        v_max: float, 
        a_min: float, 
        a_max: float, 
        u_min: float, 
        u_max: float,
        qms: float = 0.0,
    ) -> None:
        """
        Initializes the chance-constrained MPC optimization problem.

        Args:
            model_dir: Directory containing the forecasting model.
            model_name: Name of the forecasting model.
            device: Device to use for the forecasting model.
            A: Dynamics matrix of shape (n, n) for the follower.
            B: Control matrix of shape (n, m) for the follower.
            qp: Weight for the position error in the cost function.
            qv: Weight for the velocity error in the cost function.
            r: Weight for the control input magnitude in the cost function.
            dt: Discrete time step.
            model_type: Type of probabilistic model. 
                Options: 'gaussian', 'quantile', or 'truncated_gaussian'.
            t_h: Time headway used in the desired gap function for the cost: 
                d_des(v) = t_h * v + d_h.
            d_h: Distance headway offset used in the desired gap function.
            t_h_safe: Time headway used in the safe gap function for chance 
                constraints: d_safe(v) = t_h_safe * v + d_h_safe.
            d_h_safe: Distance headway offset used in the safe gap function.
            v_min: Minimum allowable follower velocity.
            v_max: Maximum allowable follower velocity.
            a_min: Minimum allowable follower acceleration.
            a_max: Maximum allowable follower acceleration.
            u_min: Minimum allowable control input.
            u_max: Maximum allowable control input.
            qms: Weight for the move suppression term in the cost function.
        """
        # Set up forecasting model
        self.device = device
        self.model = load_model(model_dir, model_name, device)
        self.model.eval()
        # Get max context length for model (always assume using transformer)
        config_path = model_dir / f"{model_name}_config.json"
        with open(config_path, "r") as f:
            model_config = json.load(f)
            self.max_context_window = model_config["max_context_len"]
        self.predecessor_velocity_history = None  # initialize in reset

        if model_type == 'quantile':
            self.model_quantiles = self.model.quantile_values.cpu().numpy()
            self.median_index = np.where(self.model_quantiles == 0.5)[0][0]
        
        # assumed states
        self.prev_assumed_state = None  # initialized in reset
        self.assumed_state = None  # updated after each control step

        # DMPC stuff below
        self.A = A
        self.B = B
        self.qp = qp
        self.qv = qv
        self.r = r
        self.N = self.model.prediction_window
        self.dt = dt
        self.model_type = model_type.lower()

        # Desired gap parameters for the cost function.
        self.t_h = t_h
        self.d_h = d_h

        # Safe gap parameters for the chance constraints.
        self.t_h_safe = t_h_safe
        self.d_h_safe = d_h_safe

        # Bounds for state and control:
        self.v_min = v_min
        self.v_max = v_max
        self.a_min = a_min
        self.a_max = a_max
        self.u_min = u_min
        self.u_max = u_max

        # Dimensions: state dimension (n) and control dimension (m)
        self.n = A.shape[0]
        self.m = B.shape[1]

        # -------------------------------------------------------------------- #
        # Define CVXPY decision variables.
        # x ∈ ℝ^((N+1) x n): state (position, velocity, acceleration)
        # u ∈ ℝ^(N x m): control inputs (desired acceleration)
        self.x = cp.Variable((self.N + 1, self.n), name="x")
        self.u = cp.Variable((self.N, self.m), name="u")

        # -------------------------------------------------------------------- #
        # Define CVXPY parameters for data provided at each timestep.
        # These parameters are updated externally prior to each MPC solve.
        #   - x0: current follower state.
        #   - p_l0: lead vehicle's current position.
        #   - v_l0: lead vehicle's current velocity.
        #   - d0: current gap (p_l0 - p_f0).
        self.x0 = cp.Parameter(self.n, name="x0")
        self.p_l0 = cp.Parameter(name="p_l0")
        self.v_l0 = cp.Parameter(name="v_l0")
        self.d0 = cp.Parameter(name="d0")

        # Prediction of the mean (or median in quantile case) of the lead 
        # vehicle's position and velocity (position and velocity of the leader 
        # are known for timestep 0, use the forecasting model for the rest).
        self.p_leader_pred = cp.Parameter(self.N + 1, name="p_leader_pred")
        self.v_leader_pred = cp.Parameter(self.N + 1, name="v_leader_pred")
        
        # -------------------------------------------------------------------- #
        # Build the optimization problem.
        self.constraints: list[Any] = []
        self.cost: cp.Expression = 0

        # -------------------------------------------------------------------- #
        # The cost function includes:
        #   (a) Position error cost, k = 0, ..., N:
        #         qp * precision[k] * (pl[k] - pf[k] - th*vf[k] - dh)^2.
        #   (b) Velocity error cost, k = 0, ..., N: 
        #         qv * precision[k] * (vl[k] - vf[k])^2.
        #   (c) Control effort cost, k = 0, ..., N - 1: 
        #         r * precision[k] * (u)^2.
        
        # Position and velocity error first
        self.leader_pos_over_std = cp.Parameter(self.N + 1, name="leader_pos_over_std")
        self.leader_vel_over_std = cp.Parameter(self.N + 1, name="leader_vel_over_std")
        self.inv_pos_std = cp.Parameter(self.N + 1, name="inv_pos_std")
        self.inv_vel_std = cp.Parameter(self.N + 1, name="inv_vel_std")
        for k in range(self.N + 1):
            # Desired gap is affine function of follower's velocity at time k+1:
            d_des_expr = self.t_h * self.x[k, 1] + self.d_h
            d_des_expr *= self.inv_pos_std[k]

            # Position error: 
            pos_error = self.leader_pos_over_std[k] - self.inv_pos_std[k] * self.x[k, 0] - d_des_expr
            self.cost += self.qp * cp.square(pos_error)

            # Velocity error: (predicted lead velocity - follower velocity)
            vel_error = self.leader_vel_over_std[k] - self.inv_vel_std[k] * self.x[k, 1]
            self.cost += self.qv * cp.square(vel_error)

        # Control effort cost
        self.control_weight = self.r * np.ones(self.N)
        for k in range(self.N):
            self.cost += self.control_weight[k] * cp.huber(self.u[k], 0.5)
        
        # Optional move suppression term in cost (similar to DMPC)
        self.qms = qms
        if qms > 0:
            self.xa = cp.Parameter((self.N + 1, self.n), name="xa")
            self.assumed_weight = self.qms * 1 / np.linspace(0.05, 5, self.N + 1, endpoint=True)
            for k in range(self.N + 1):
                self.cost += self.assumed_weight[k] * cp.sum_squares(self.x[k] - self.xa[k])
        
        # Optional penalty on smoothness of the control input
        self.prev_action = cp.Parameter(self.m, name="prev_action")
        self.cost += 10.0 * cp.huber(self.u[0] - self.prev_action, 0.1)
        
        # -------------------------------------------------------------------- #
        # Constraints setup: initial conditions, state and input bounds, 
        # dynamics, and chance constraints.

        # Initial condition for the follower state.
        self.constraints.append(self.x[0] == self.x0)

        # Follower dynamics constraints for k = 0, ..., N-1.
        for k in range(self.N):
            self.constraints.append(
                self.x[k + 1] == self.A @ self.x[k] + self.B @ self.u[k]
            )

        # Enforce velocity and acceleration limits at all time steps.
        for k in range(self.N + 1):
            self.constraints.append(self.x[k, 1] >= self.v_min)
            self.constraints.append(self.x[k, 1] <= self.v_max)
            self.constraints.append(self.x[k, 2] >= self.a_min)
            self.constraints.append(self.x[k, 2] <= self.a_max)

        # Add control input bounds for k = 0, ..., N-1.
        for k in range(self.N):
            self.constraints.append(self.u[k] >= self.u_min)
            self.constraints.append(self.u[k] <= self.u_max)
        
        # Chance (safety) constraints on distance to predecessor
        self.cc_rhs = cp.Parameter(self.N, name="cc_rhs")
        self.cc_lhs = self.get_chance_constraint_lhs_mat(self.N)
        self.constraints.append(self.cc_lhs @ self.x[:, 1] <= self.cc_rhs)
        
        # -------------------------------------------------------------------- #
        # Form the complete CVXPY problem.
        self.prob = cp.Problem(cp.Minimize(self.cost), self.constraints)
        assert self.prob.is_dpp(), "Problem not DPP, check constraints and cost."

    def control(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Solves the MPC optimization problem given current measurements and 
        forecasts, and returns the first control input along with diagnostic 
        information. Assumes `step` is called prior to this method to set 
        the current state and observations.

        The leader's current position (p_l0) is computed from the follower's position (x0[0])
        and the gap (d0): p_l0 = x0[0] + d0.

        Args:
            **kwargs: Additional arguments

        Returns:
            A tuple containing:
              - The optimal control input (first element of u).
              - A dictionary with the following keys:
                  "optimal_state": optimal state trajectory.
                  "optimal_control": optimal control trajectory.
                  "cost": The optimal cost.
                  "solve_time": The solve time.

        Raises:
            RuntimeError: If the solver does not return an optimal solution.
        """
        pred_vels = torch.tensor(
            self.predecessor_velocity_history, dtype=torch.float, device=self.device
        ).reshape(1, -1, 1)
        pred_vels += torch.randn_like(pred_vels) * 0.1  # Add noise to the input as in training
        pred_vels = normalize_vel(F.relu(pred_vels))
        with torch.no_grad():
            (mu_forecast, logvar_forecast), quantile_forecasts = self.model(pred_vels)
            # denormalize predictions
            if 'gaussian' in self.model_type:
                mu_forecast = denormalize_vel(mu_forecast.squeeze().cpu().numpy())
                sigma_forecast = vstd * torch.exp(0.5 * logvar_forecast).squeeze().cpu().numpy()
            else:
                quantile_forecasts = denormalize_vel(quantile_forecasts).squeeze().cpu().numpy()

        # Update the expressions needed for cost and constraints
        self.set_problem_parameters(
            mu_sigma=(mu_forecast, sigma_forecast), 
            quantile_forecasts=quantile_forecasts,
            confidence_level=kwargs['confidence_level']
        )
        
        if self.qms > 0:
            self.xa.value = self.prev_assumed_state
        
        # Solve the optimization problem.
        try:
            self.prob.solve(eps_abs=1e-4, eps_rel=1e-4, max_iter=100000)
        except Exception as e:
            print(f"CCMPC failed with exception: {e}")
            raise e

        # Check if the problem was solved to optimality.
        if "optimal" not in self.prob.status:
            raise RuntimeError(f"MPC solver return with status {self.prob.status}")
        
        self.assumed_state, _ = self.update_assumed_states_controls(
            self.x.value, self.u.value, np.zeros(self.m)
        )

        # Retrieve the optimal control input (first time step) and diagnostic information.
        optimal_control = np.clip(self.u.value[0], self.u_min, self.u_max)
        self.prev_action.value = optimal_control
        results = {
            "optimal_state": self.x.value,
            "optimal_control": self.u.value,
            "cost": self.prob.value,
            "solve_time": self.prob.solver_stats.solve_time
        }
        return optimal_control, results
    
    def set_problem_parameters(
        self, 
        mu_sigma: tuple[np.ndarray, np.ndarray] | None = None,
        quantile_forecasts: np.ndarray | None = None,
        confidence_level: float | None = None
    ):
        """
        Set the CVXPY problem parameters for the current timestep, i.e., set 
        `p_leader_pred`, `v_leader_pred`, and `cc_rhs` based on the model 
        forecasts. Also set `leader_pos_over_std`, `leader_vel_over_std`,
        `inv_pos_std`, and `inv_vel_std` for the cost function.

        Args:
            mu_sigma: Tuple of mean and standard deviation for the leader's 
                position and velocity predictions (assumed denormalized).
            quantile_forecasts: Quantile forecasts for the leader's position 
                and velocity predictions (assumed denormalized).
        """
        p_leader_pred = np.zeros(self.N + 1)
        p_leader_pred[0] = self.p_l0.value
        v_leader_pred = np.zeros(self.N + 1)
        v_leader_pred[0] = self.v_l0.value

        # Set `p_leader_pred` and `v_leader_pred` based on the model type
        if self.model_type == "gaussian":
            mu_forecast, sigma_forecast = mu_sigma
            v_leader_pred[1:] = mu_forecast
            p_leader_pred[1:] = p_leader_pred[0] + self.dt * np.cumsum(v_leader_pred[:-1])
            var_forecast = sigma_forecast ** 2
        elif self.model_type == "truncated_gaussian":
            mu, sigma = mu_sigma
            lower_bound = -0.0 * np.ones(self.N)  # minimum velocity assumed 0
            alphas = (lower_bound - mu) / sigma
            lambdas = norm.pdf(alphas) / (1 - norm.cdf(alphas))
            mu_forecast = mu + sigma * lambdas  # adjusted mean for truncated gaussian
            var_forecast = sigma ** 2 * (1 + alphas * lambdas - lambdas ** 2)
            sigma_forecast = var_forecast ** 0.5
            v_leader_pred[1:] = mu_forecast
            p_leader_pred[1:] = p_leader_pred[0] + self.dt * np.cumsum(v_leader_pred[:-1])
        elif self.model_type == "quantile":
            # Use the median quantile forecast for the leader's velocity
            v_leader_pred[1:] = quantile_forecasts[:, self.median_index]
            p_leader_pred[1:] = p_leader_pred[0] + self.dt * np.cumsum(v_leader_pred[:-1])
        
        # Set position and velocity inverse std and other vars for cost function
        if 'gaussian' in self.model_type:
            self.inv_pos_std.value = np.zeros(self.N + 1)
            self.inv_vel_std.value = np.zeros(self.N + 1)
            self.leader_pos_over_std.value = np.zeros(self.N + 1)
            self.leader_vel_over_std.value = np.zeros(self.N + 1)

            vel_std = np.zeros(self.N + 1)
            vel_std[1:] = sigma_forecast
            vel_std[0] = sigma_forecast[0]
            self.inv_vel_std.value = 1 / vel_std

            self.inv_pos_std.value = np.zeros(self.N + 1)
            pos_std = self.dt * np.sqrt(np.cumsum(var_forecast))
            self.inv_pos_std.value[1:] = 1 / pos_std
            self.inv_pos_std.value[0] = 1 / pos_std[0]
            pos_std = 1 / self.inv_pos_std.value

            self.leader_pos_over_std.value = p_leader_pred / pos_std
            self.leader_vel_over_std.value = v_leader_pred / vel_std

        # Set `cc_rhs` based on the model type
        dt_mat = np.tril(np.ones((self.N, self.N))) * self.dt
        if 'gaussian' in self.model_type:
            cc_rhs = np.zeros(self.N)
            var_vec = np.zeros(self.N)
            var_vec[1:] = np.sqrt(np.cumsum(var_forecast[:-1]))
            cc_rhs = (
                self.d0.value - self.d_h_safe + dt_mat @ v_leader_pred[:-1]
                - self.dt * norm.ppf(confidence_level) * var_vec
            )
        else:
            lower_index = np.where(self.model_quantiles == 1 - confidence_level)[0][0]
            v_leader_lower = np.zeros(self.N + 1)  # lower quantile estimate
            v_leader_lower[0] = self.v_l0.value
            v_leader_lower[1:] = quantile_forecasts[:, lower_index]
            cc_rhs = self.d0.value - self.d_h_safe + dt_mat @ v_leader_lower[:-1]
        
        self.p_leader_pred.value = p_leader_pred
        self.v_leader_pred.value = v_leader_pred
        self.cc_rhs.value = cc_rhs

    def initialize_assumed_trajectory(self) -> np.ndarray:
        """
        Initializes the assumed trajectory.

        Returns:
            np.ndarray: shape (H + 1, n) assumed trajectory
        """
        xa = np.zeros((self.N + 1, self.n))
        xa[0] = self.x0.value
        for k in range(self.N):
            xa[k + 1] = self.A @ xa[k] + self.B @ self.u_ref[k]
        return xa
    
    def update_assumed_states_controls(
        self, xopt: np.ndarray, uopt: np.ndarray, uzero: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Gets the assumed states and controls based on the optimal solution.

        Args:
            xopt: optimal state trajectory, shape (H + 1, n)
            uopt: optimal control input, shape (H, m)
            uzero: zero control input, shape (m,)
        
        Returns:
            np.ndarray: shape (H + 1, n) assumed state trajectory
            np.ndarray: shape (H, m) assumed control input
        """
        n, m = self.A.shape[0], self.B.shape[1]
        xa = np.zeros((self.N + 1, n))
        ua = np.zeros((self.N, m))
        xa[:-1] = xopt[1:]
        ua[:-1] = uopt[1:]
        ua[-1] = uzero
        xa[-1] = self.A @ xa[-2] + self.B @ ua[-1]
        return xa, ua

    def reset(self, veh_state: np.ndarray, obs: np.ndarray, **kwargs):
        """
        Reset the controller state if necessary (occurs after environment 
        reset).

        Args:
            veh_state: vehicle state provided by environment
            obs: observation from the environment
            **kwargs: additional kwargs
        """
        self.prev_action.value = np.zeros(self.m)
        pred_vel = veh_state[1] + obs[1]

        # set cvxpy parameters
        self.x0.value = veh_state
        self.d0.value = obs[0]
        self.p_l0.value = veh_state[0] + obs[0]
        self.v_l0.value = pred_vel

        self.predecessor_velocity_history = deque(maxlen=self.max_context_window)
        self.predecessor_velocity_history.append(pred_vel)

        self.u_ref = kwargs.get("u_ref", np.zeros((self.N, self.m)))
        self.prev_assumed_state = self.initialize_assumed_trajectory()
        self.assumed_state = None
    
    def step(self, veh_state: np.ndarray, obs: np.ndarray, **kwargs):
        """
        Update the controller state if necessary (occurs after all vehicles 
        in platoon have calculated their action for the current timestep).

        Args:
            veh_state: vehicle state provided by environment
            obs: observation from the environment
            **kwargs: additional kwargs
        """
        pred_vel = veh_state[1] + obs[1]

        # set cvxpy parameters
        self.x0.value = veh_state
        self.d0.value = obs[0]
        self.p_l0.value = veh_state[0] + obs[0]
        self.v_l0.value = pred_vel

        self.predecessor_velocity_history.append(pred_vel)

        self.prev_assumed_state = copy.deepcopy(self.assumed_state)
    
    def get_chance_constraint_lhs_mat(self, N: int) -> np.ndarray:
        """
        Returns the left-hand side matrix for the chance constraint which 
        left-multiplies the velocity state variables (N+1-dimensional).

        Args:
            N: Prediction horizon length.
        
        Returns:
            np.ndarray: Left-hand side matrix for the chance constraint.
        """
        cc_lhs = np.zeros((N, N + 1))
        for k in range(N):
            cc_lhs[k, :k + 1] = self.dt
            cc_lhs[k, k + 1] = self.t_h_safe
        return cc_lhs