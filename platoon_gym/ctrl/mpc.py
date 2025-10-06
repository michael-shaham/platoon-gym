from abc import abstractmethod
import cvxpy as cp
import numpy as np
import scipy as sp
from scipy.linalg import block_diag
from typing import Optional, Tuple

from platoon_gym.ctrl.controller_base import ControllerBase


class MPC(ControllerBase):
    """
    Generic MPC base class.
    """

    n: int  # state dimension
    m: int  # input dimension

    def __init__(self, H: int):
        """
        Args:
            H: int, prediction horizon
        """
        super().__init__()
        self.H = H

    @abstractmethod
    def _mpc_problem(self):
        pass


class LinearMPC(MPC):
    """
    Classic linear MPC implementation using CVXPY and OSQP.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
        Cx: np.ndarray,
        Cu: np.ndarray,
        dx: np.ndarray,
        du: np.ndarray,
        H: int,
    ):
        """
        Args:
            A: shape (n, n), state transition matrix
            B: shape (n, m), control matrix
            C: shape (p, n), output matrix
            Q: shape (p, p), state cost matrix
            R: shape (m, m), input cost matrix
            Qf: shape (n, n), terminal state cost matrix
            Cx: shape (n, n), state constraint matrix
            Cu: shape (m, m), input constraint matrix
            dx: shape (n,), state constraint vector
            du: shape (m,), input constraint vector
            H: int, prediction horizon
        """
        super().__init__(H=H)
        assert H > 1
        n, m, p = A.shape[0], B.shape[1], C.shape[0]
        self.n, self.m, self.p = n, m, p
        self.A, self.C = A, C
        (
            self.Q,
            self.R,
            self.Qf,
        ) = (
            Q,
            R,
            Qf,
        )
        self.opt_var_dim = H * (n + m)

        assert Q.shape == (p, p) and R.shape == (m, m) and Qf.shape == (n, n)

        P = block_diag(*([R, C.T @ Q @ C] * (H - 1) + [R, Qf]))
        self.P = sp.sparse.csr_matrix(P)
        assert self.P.shape == (H * (n + m), H * (n + m))
        self.q = np.zeros(self.opt_var_dim)

        A_bar = np.zeros((H * n, self.opt_var_dim))
        A_bar[:, :-n] += block_diag(*([-B] + [np.block([-A, -B])] * (H - 1)))
        A_bar[:-n, m:-n] += block_diag(
            *([np.block([np.eye(n), np.zeros((n, m))])] * (H - 1))
        )
        A_bar[-n:, -n:] += np.eye(n)
        self.A_bar = sp.sparse.csr_matrix(A_bar)
        self.b_bar = np.zeros(self.A_bar.shape[0])

        C_bar = block_diag(*([Cu, Cx] * H))
        self.C_bar = sp.sparse.csr_matrix(C_bar)
        self.d_bar = np.tile(np.concatenate((du, dx)), H)
        assert self.C_bar.shape[0] == self.d_bar.shape[0]

    def control(
        self,
        x0: np.ndarray,
        y_ref: Optional[np.ndarray] = None,
        u_ref: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Returns a control input based on an MPC policy. Also returns a dict
        with the planned trajectory and planned inputs.

        Args:
            x0: shape (n,), current state
            y_ref: shape (p, H+1), optional reference trajectory
            u_ref: shape (m, H), optional reference control input

        Returns:
            np.ndarray, shape (m,): control input
            dict: extra information related to the problem
        """
        n, m, p, H = self.n, self.m, self.p, self.H
        assert x0.ndim == 1 and x0.shape[0] == n
        if y_ref is not None:
            assert y_ref.ndim == 2 and y_ref.shape == (p, self.H + 1)
        else:
            y_ref = np.zeros((p, self.H + 1))
        if u_ref is not None:
            assert u_ref.ndim == 2 and u_ref.shape == (m, self.H)
        else:
            u_ref = np.zeros((m, self.H))

        prob, y = self._mpc_problem(x0, y_ref, u_ref)
        try:
            prob.solve(solver=cp.OSQP)
        except cp.error.SolverError as e:
            prob.solve(solver=cp.OSQP, verbose=True)
            print(f"Solver failed with error: {e}")
            print(f"Problem data:")
            print(f"x0:\n{x0}")
            print(f"y_ref:\n{y_ref}")
            print(f"u_ref:\n{u_ref}")
            print(f"P:\n{self.P.toarray()}")
            print(f"q:\n{self.q}")
            print(f"A_bar:\n{self.A_bar.toarray()}")
            print(f"b_bar:\n{self.b_bar}")
            print(f"C_bar:\n{self.C_bar.toarray()}")
            print(f"d_bar:\n{self.d_bar}")
            raise e
        u_opt = np.empty((m, H))
        x_opt = np.empty((n, H + 1))
        if prob.status == cp.OPTIMAL:
            u_opt[:, 0] = y.value[:m]
            x_opt[:, 0] = x0
            for k, i in enumerate(range(m, m + (H - 1) * (m + n), m + n)):
                x_opt[:, k + 1] = y.value[i : i + n]
                u_opt[:, k + 1] = y.value[i + n : i + n + m]
            x_opt[:, -1] = y.value[-n:]
        info = {"status": prob.status, "planned states": x_opt, "planned inputs": u_opt}
        return np.atleast_1d(u_opt[:, 0]), info

    def _mpc_problem(self, x0, y_ref, u_ref) -> Tuple[cp.Problem, cp.Variable]:
        """
        Creates the CVXPY problem and optimization variable for the MPC problem.
        """
        n, m, H = self.n, self.m, self.H
        y = cp.Variable(self.opt_var_dim)

        self.b_bar[: self.n] = self.A @ x0

        self.q[:m] = self.R @ u_ref[:, 0]
        for k, i in enumerate(range(m, m + (H - 1) * (m + n), m + n)):
            self.q[i : i + n] = self.C.T @ self.Q @ y_ref[:, k + 1]
            self.q[i + n : i + n + m] = self.R @ u_ref[:, k + 1]
        zf = y_ref[:, -1] if self.p == self.n else np.array([*y_ref[:, -1], 0.0])
        self.q[-n:] = self.Qf @ zf
        self.q *= -2

        cost = cp.quad_form(y, self.P) + self.q @ y
        constraints = [self.A_bar @ y == self.b_bar, self.C_bar @ y <= self.d_bar]
        return cp.Problem(cp.Minimize(cost), constraints), y
