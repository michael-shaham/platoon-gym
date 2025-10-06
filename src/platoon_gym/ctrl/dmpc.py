import copy
import cvxpy as cp
import numpy as np
from typing import Any

from platoon_gym.ctrl.controller_base import ControllerBase

NORM_OPTIONS = {"l1", "l2", "quadratic"}


class DMPC(ControllerBase):
    """
    Generic class for distributed model predictive control for platooning,
    assuming the vehicles can share their planned trajectores with one another.
    """

    def __init__(
        self,
        H: int,
        Q: float | np.ndarray,
        Q_neighbors: list[float | np.ndarray],
        R: float | np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        x_lims: np.ndarray,
        u_lims: np.ndarray,
        u_slew_rate: np.ndarray,
        distance_headways: list[float],
        time_headways: list[float],
        terminal_constraint: bool = True,
        Qf: float | np.ndarray | None = None,
        Qf_neighbors: list[float] | list[np.ndarray] | None = None,
        output_norm: str = "quadratic",
        input_norm: str = "quadratic",
    ):
        """
        Args:
            H: int, horizon
            Q: np.ndarray, state cost matrix
            Q_neighbors: List[np.ndarray], state cost matrices for neighbors
            R: np.ndarray, input cost matrix
            A: np.ndarray, dynamics matrix
            B: np.ndarray, input matrix
            C: np.ndarray, observation matrix
            x_lims: np.ndarray, state limits
            u_lims: np.ndarray, input limits
            u_slew_rate: np.ndarray, input slew rate
            distance_headways: List[float], desired distance between vehicles
            time_headways: List[float], desired time headway between vehicles
            terminal_constraint: bool, use terminal equality constraint if true
            Qf: np.ndarray, terminal state cost matrix
            Qf_neighbors: List[np.ndarray], neighbor terminal state cost matrices
            output_norm: str, norm to use for output cost
            input_norm: str, norm to use for input cost
        """
        self.H = H
        n, m, p = A.shape[0], B.shape[1], C.shape[0]
        n_neighbors = len(Q_neighbors)
        self.terminal_constraint = terminal_constraint

        # assumed states
        self.prev_assumed_state = None  # initialized in reset
        self.assumed_state = None  # updated after each control step

        # check given data
        assert A.shape == (n, n)
        assert B.shape == (n, m)
        assert C.shape == (p, n)
        self.A, self.B, self.C = A, B, C
        n, m, p = A.shape[0], B.shape[1], C.shape[0]
        if not terminal_constraint:
            assert Qf is not None and Qf_neighbors is not None
        assert x_lims.shape == (n, 2)
        assert u_lims.shape == (m, 2)
        self.u_min, self.u_max = u_lims[:, 0], u_lims[:, 1]
        assert u_slew_rate.shape == (m,)
        for i in range(m):
            assert u_slew_rate[i] >= 0
        if isinstance(Q, float):
            Q = Q * np.eye(p)
        assert Q.shape == (p, p)
        assert isinstance(Q_neighbors, list)
        for i in range(len(Q_neighbors)):
            if isinstance(Q_neighbors[i], float):
                Q_neighbors[i] = Q_neighbors[i] * np.eye(p)
            assert Q_neighbors[i].shape == (p, p)
        if isinstance(R, float):
            R = R * np.eye(m)
        assert R.shape == (m, m)
        if Qf is not None:
            if isinstance(Qf, float):
                Qf = Qf * np.eye(p)
            assert Qf.shape == (p, p)
        if Qf_neighbors is not None:
            assert len(Qf_neighbors) == n_neighbors
            for i in range(len(Qf_neighbors)):
                if isinstance(Qf_neighbors[i], float):
                    Qf_neighbors[i] = Qf_neighbors[i] * np.eye(p)
                assert Qf_neighbors[i].shape == (p, p)

        assert isinstance(distance_headways, list)
        assert len(distance_headways) == n_neighbors
        assert isinstance(time_headways, list)
        assert len(time_headways) == n_neighbors
        self.distance_headways = distance_headways
        self.time_headways = time_headways

        assert output_norm in NORM_OPTIONS
        assert input_norm in NORM_OPTIONS
        self.output_norm, self.input_norm = output_norm, input_norm

        # set up cost function depending on norm used
        if output_norm == "quadratic":
            self.out_cost = lambda x, X: cp.quad_form(x, X)
        elif output_norm == "l1":
            self.out_cost = lambda x, X: cp.norm(X @ x, 1)
        elif output_norm == "l2":
            self.out_cost = lambda x, X: cp.norm(X @ x, 2)
        if input_norm == "quadratic":
            self.in_cost = lambda u, U: cp.quad_form(u, U)
        elif input_norm == "l1":
            self.in_cost = lambda u, U: cp.norm(U @ u, 1)
        elif input_norm == "l2":
            self.in_cost = lambda u, U: cp.norm(U @ u, 2)

        # construct cvxpy problem
        self.constraints = []
        self.x = cp.Variable((H + 1, n))
        self.u = cp.Variable((H, m))
        self.x0 = cp.Parameter(n)
        self.xf = cp.Parameter(n)
        self.x_a = cp.Parameter((H + 1, n))
        self.u_ref = cp.Parameter((H, m))
        self.u_ref.value = np.zeros((H, m))
        self.y_neighbors = [cp.Parameter((H + 1, p)) for _ in range(n_neighbors)]

        # constraints at time k = 0
        self.constraints += [
            self.x[0] == self.x0,
            self.u[0] <= u_lims[:, 1],
            self.u[0] >= u_lims[:, 0],
        ]
        # cost at time k = 0
        move_supp_err = C @ (self.x[0] - self.x_a[0])
        self.cost = self.out_cost(move_supp_err, Q)
        self.cost += self.in_cost(self.u[0] - self.u_ref[0], R)
        for j in range(len(Q_neighbors)):
            neighbor_pos_err = (
                self.y_neighbors[j][0, 0]
                - self.x[0, 0]
                - distance_headways[j]
                - time_headways[j] * self.x[0, 1]
            )
            neighbor_vel_err = self.y_neighbors[j][0, 1] - self.x[0, 1]
            neighbor_err = cp.vstack([neighbor_pos_err, neighbor_vel_err])
            self.cost += self.out_cost(neighbor_err, Q_neighbors[j])

        # constraints and cost for k = 1, ..., H - 1
        for k in range(1, H):
            self.constraints += [
                self.x[k] == A @ self.x[k - 1] + B @ self.u[k - 1],
                self.x[k] <= x_lims[:, 1],
                self.x[k] >= x_lims[:, 0],
                self.u[k] <= u_lims[:, 1],
                self.u[k] >= u_lims[:, 0],
                self.u[k] - self.u[k - 1] <= u_slew_rate,
                self.u[k] - self.u[k - 1] >= -u_slew_rate,
            ]
            move_supp_err = C @ (self.x[k] - self.x_a[k])
            self.cost += self.out_cost(move_supp_err, Q)
            self.cost += self.in_cost(self.u[k] - self.u_ref[k], R)
            for j in range(len(Q_neighbors)):
                neighbor_pos_err = (
                    self.y_neighbors[j][k, 0]
                    - self.x[k, 0]
                    - distance_headways[j]
                    - time_headways[j] * self.x[k, 1]
                )
                neighbor_vel_err = self.y_neighbors[j][k, 1] - self.x[k, 1]
                neighbor_err = cp.vstack([neighbor_pos_err, neighbor_vel_err])
                self.cost += self.out_cost(neighbor_err, Q_neighbors[j])

        # terminal constraints and cost
        self.constraints += [
            self.x[H] == A @ self.x[H - 1] + B @ self.u[H - 1],
            self.x[H] <= x_lims[:, 1],
            self.x[H] >= x_lims[:, 0],
        ]
        if terminal_constraint:
            self.constraints += [self.x[H] == self.xf]
        else:
            move_supp_err = C @ (self.x[H] - self.x_a[H])
            self.cost += self.out_cost(move_supp_err, Qf)
            for j in range(len(Q_neighbors)):
                neighbor_pos_err = (
                    self.y_neighbors[j][H, 0]
                    - self.x[H, 0]
                    - distance_headways[j]
                    - time_headways[j] * self.x[H, 1]
                )
                neighbor_vel_err = self.y_neighbors[j][H, 1] - self.x[H, 1]
                neighbor_err = cp.vstack([neighbor_pos_err, neighbor_vel_err])
                self.cost += 1 / len(Q_neighbors) * self.out_cost(neighbor_err, Q_neighbors[j])

        self.prob = cp.Problem(cp.Minimize(self.cost), self.constraints)

    def control(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Solve the optimization problem and return the control input.

        Args:
            **kwargs: not used

        Returns:
            np.ndarray, control input
            dict: solution info
        """
        self.x_a.value = copy.deepcopy(self.prev_assumed_state)
        self.xf.value = self.calculate_terminal_constraint()
        try:
            self.prob.solve(eps_abs=1e-4, eps_rel=1e-4)
        except Exception as e:
            print("DMPC failed with exception:", e)
            raise e

        if "optimal" not in self.prob.status:
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(2, 1)
            # fig.suptitle(f'x0={np.round(self.x0.value, 2)}, xf={np.round(self.xf.value, 2)}')
            # for i in range(len(self.y_neighbors)):
            #     ax[0].plot(self.y_neighbors[i].value[:, 0], label=f"neighbor {i}")
            #     ax[1].plot(self.y_neighbors[i].value[:, 1], label=f"neighbor {i}")
            # ax[0].plot(self.x_a.value[:, 0], label="assumed")
            # ax[1].plot(self.x_a.value[:, 1], label="assumed")
            # for a in ax:
            #     a.legend()
            #     a.grid()
            # plt.show()
            raise Exception(f"Solver failed with status {self.prob.status}")
        
        self.assumed_state, _ = self.update_assumed_states_controls(
            self.x.value, self.u.value, self.u_ref.value[0]
        )

        optimal_control = np.clip(self.u.value[0], self.u_min, self.u_max)

        return (
            optimal_control,
            {
                "x": self.x.value,
                "u": self.u.value,
                "cost": self.prob.value,
                "solve time": self.prob.solver_stats.solve_time,
            },
        )
    
    def calculate_terminal_constraint(self) -> np.ndarray | None:
        """
        Calculate the terminal constraint based on the neighbors' planned 
        trajectories.

        Returns:
            np.ndarray, terminal constraint (or None)
        """
        if not self.terminal_constraint:
            return None
        n = self.A.shape[0]
        xf = np.zeros(n)
        yf_neighbors = []
        p = self.C.shape[0]
        for i in range(len(self.y_neighbors)):
            end_out = self.y_neighbors[i].value[-1]
            end_pos, end_vel = end_out[0], end_out[1]
            d_des = self.time_headways[i] * end_vel + self.distance_headways[i]
            yf_neighbors.append(np.array([end_pos - d_des, end_vel]))
        yf = sum(yf_neighbors) / len(yf_neighbors)
        xf[:p] = yf
        return xf

    def initialize_assumed_trajectory(self) -> np.ndarray:
        """
        Initializes the assumed trajectory.

        Returns:
            np.ndarray: shape (H + 1, n) assumed trajectory
        """
        self.x_a.value = np.zeros(self.x_a.shape)
        self.x_a.value[0] = self.x0.value[:]
        for k in range(self.H):
            self.x_a.value[k + 1] = self.A @ self.x_a.value[k] + self.B @ self.u_ref.value[k]
        return self.x_a.value
    
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
        xa = np.zeros((self.H + 1, n))
        ua = np.zeros((self.H, m))
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
        self.x0.value = veh_state
        self.u_ref.value = kwargs.get("u_ref", np.zeros(self.u_ref.shape))
        self.prev_assumed_state = self.initialize_assumed_trajectory()
        self.assumed_state = None
        y_neighbors = kwargs["y_neighbors"]
        if y_neighbors is None:
            return
        for i, yn in enumerate(y_neighbors):
            self.y_neighbors[i].value = yn[:, :self.C.shape[0]]

    def step(self, veh_state: np.ndarray, obs: np.ndarray, **kwargs):
        """
        Update the controller state if necessary (occurs after all vehicles 
        in platoon have calculated their action for the current timestep).

        Args:
            veh_state: vehicle state provided by environment
            obs: observation from the environment
            **kwargs: additional kwargs
        """
        self.x0.value = veh_state
        self.u_ref.value = kwargs.get("u_ref", np.zeros(self.u_ref.shape))
        self.prev_assumed_state = copy.deepcopy(self.assumed_state)
        y_neighbors = kwargs["y_neighbors"]
        for i, yn in enumerate(y_neighbors):
            self.y_neighbors[i].value = yn[:, :self.C.shape[0]]