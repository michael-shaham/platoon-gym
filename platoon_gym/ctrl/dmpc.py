import cvxpy as cp
import numpy as np
from typing import List, Optional, Union

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
        Q: Union[float, np.ndarray],
        Q_neighbors: List[Union[float, np.ndarray]],
        R: Union[float, np.ndarray],
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        x_lims: np.ndarray,
        u_lims: np.ndarray,
        u_slew_rate: np.ndarray,
        distance_headways: Optional[List[float]],
        time_headways: Optional[List[float]],
        terminal_constraint: bool = True,
        Qf: Optional[np.ndarray] = None,
        Qf_neighbors: Optional[List[np.ndarray]] = None,
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
            terminal_constraint: bool, use terminal equality constraint if true
            Qf: np.ndarray, terminal state cost matrix
            Qf_neighbors: List[np.ndarray], neighbor terminal state cost matrices
            distance_headways: List[float], desired distance between vehicles
            time_headway: List[float], desired time headway between vehicles
            output_norm: str, norm to use for output cost
            input_norm: str, norm to use for input cost
        """
        self.H = H
        n, m, p = A.shape[0], B.shape[1], C.shape[0]
        n_neighbors = len(Q_neighbors)
        self.terminal_constraint = terminal_constraint

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
        self.u_a = cp.Parameter((H, m))
        self.u_a.value = np.zeros((H, m))
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
        self.cost += self.in_cost(self.u[0] - self.u_a[0], R)
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
            self.cost += self.in_cost(self.u[k] - self.u_a[k], R)
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
            self.cost += self.out_cost(move_supp_err, Q)
            for j in range(len(Q_neighbors)):
                neighbor_pos_err = (
                    self.y_neighbors[j][H, 0]
                    - self.x[H, 0]
                    - distance_headways[j]
                    - time_headways[j] * self.x[H, 1]
                )
                neighbor_vel_err = self.y_neighbors[j][H, 1] - self.x[H, 1]
                neighbor_err = cp.vstack([neighbor_pos_err, neighbor_vel_err])
                self.cost += self.out_cost(neighbor_err, Q_neighbors[j])

        self.prob = cp.Problem(cp.Minimize(self.cost), self.constraints)

    def control(
        self,
        x0: np.ndarray,
        y_neighbors: List[np.ndarray],
        xa: Optional[np.ndarray] = None,
        ua: Optional[np.ndarray] = None,
        xf: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Args:
            x0: np.ndarray, initial state
            y_neighbors: List[np.ndarray], neighbors planned trajectories
            xa: np.ndarray, assumed trajectory
            ua: np.ndarray, assumed control input
            xf: np.ndarray, terminal constraint

        Returns:
            np.ndarray, control input
            dict: solution info
        """
        assert x0.shape == self.x0.shape
        self.x0.value = x0
        assert len(y_neighbors) == len(self.y_neighbors)
        for i in range(len(y_neighbors)):
            assert y_neighbors[i].shape == self.y_neighbors[i].shape
            self.y_neighbors[i].value = y_neighbors[i]
        if xa is not None:
            assert xa.shape == self.x_a.shape
            self.x_a.value = xa
        if ua is not None:
            assert ua.shape == self.u_a.shape
            self.u_a.value = ua
        if self.terminal_constraint:
            assert xf.shape == self.xf.shape
            self.xf.value = xf

        self.prob.solve(solver=cp.CLARABEL)

        if "optimal" not in self.prob.status:
            raise Exception(f"Solver failed with status {self.prob.status}")

        return (
            self.u.value[0],
            {
                "x": self.x.value,
                "u": self.u.value,
                "cost": self.prob.value,
                "solve time": self.prob.solver_stats.solve_time,
            },
        )

    def initialize_assumed_trajectory(
        self, x0: np.ndarray, uref: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Initializes the assumed trajectory.

        Args:
            x0: shape (n,), current state
            uref: shape (m, H), optional reference control input

        Returns:
            np.ndarray: shape (H + 1, n) assumed trajectory
        """
        if uref is not None:
            assert uref.shape == self.u_a.shape
            self.u_a.value = uref
        self.x_a.value = np.zeros(self.x_a.shape)
        self.x_a.value[0] = x0
        for k in range(self.H):
            if uref is None:
                self.x_a.value[k + 1] = self.A @ self.x_a.value[k]
            else:
                self.x_a.value[k + 1] = self.A @ self.x_a.value[k] + self.B @ uref[k]
        return self.x_a.value
