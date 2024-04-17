from abc import abstractmethod
import copy
import cvxpy as cp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
from typing import Optional, Tuple, Union

from platoon_gym.ctrl.controller_base import ControllerBase


class SparseDMPC(ControllerBase):
    """
    Generic class sparse distributed model predictive control algorithms
    for platooning (assuming linear dynamics).
    """

    n: int  # state dimension
    m: int  # input dimension
    p: int  # output dimension
    A: np.ndarray  # dynamics matrix
    B: np.ndarray  # input matrix
    C: np.ndarray  # sensor matrix

    def __init__(self, H: int, d_des: float):
        """
        Args:
            H: int, prediction horizon
            d_des: float, desired distance between vehicles
        """
        super().__init__()
        self.H = H
        self.d_des = d_des

    @abstractmethod
    def _mpc_problem(self) -> Tuple[cp.Problem, cp.Variable]:
        pass

    def init_assumed_trajectory(
        self, x0: np.ndarray, uref: Optional[np.ndarray] = None
    ):
        """
        Initializes the assumed trajectory.

        Args:
            x0: shape (n,), current state
            uref: shape (m, H), optional reference control input
        """
        self.xa = np.zeros((self.n, self.H + 1))
        self.xa[:, 0] = x0
        for k in range(self.H):
            if uref is None:
                self.xa[:, k + 1] = self.A @ self.xa[:, k]
            else:
                self.xa[:, k + 1] = self.A @ self.xa[:, k] + self.B @ uref[:, k]
        return self.xa

    def set_desired_distance(self, d_des: float):
        """
        Sets the desired distance between vehicles.

        Args:
            d_des: float, desired distance between vehicles
        """
        self.d_des = d_des

    @abstractmethod
    def set_terminal_constraint(
        self, Afx: Optional[np.ndarray] = None, Afu: Optional[np.ndarray] = None
    ):
        """
        Sets the terminal equality constraint. If none, removes the
        corresponding rows in the equality constraint matrix.

        Args:
            Afx: shape (n, n), state terminal constraint matrix
            Afu: shape (m, m), input terminal constraint matrix
        """
        pass

    @abstractmethod
    def get_optimal_trajectory(self, x0: np.ndarray, z: np.ndarray):
        """
        Returns the optimal trajectory and, optionally, other variables.

        Args:
            x0: np.ndarray, shape (n,), current state
            z: np.ndarray, shape (self.z_dim,), optimization problem solution

        Returns:
            x_opt: np.ndarray, shape (n, H+1), optimal state trajectory
            y_opt: np.ndarray, shape (p, H+1), optimal output trajectory
            u_opt: np.ndarray, shape (m, H), optimal input trajectory
        """
        pass

    def update_assumed_trajectory(
        self,
        x_opt: np.ndarray,
        y_opt: np.ndarray,
        u_opt: np.ndarray,
        ua_end: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Updates the assumed trajectory for the next timestep based on the
        optimal trajectory.

        Args:
            x_opt: shape (n, H+1), optimal state trajectory
            y_opt: shape (p, H+1), optimal output trajectory
            u_opt: shape (m, H), optimal input trajectory
            ua_end: shape (m,), terminal assumed input

        Returns:
            np.ndarray, shape (n, H+1): assumed state trajectory
            np.ndarray, shape (p, H+1): updated assumed output trajectory
            np.ndarray, shape (m, H): updated assumed input trajectory
        """
        ua = np.zeros_like(u_opt)
        xa = np.zeros_like(x_opt)
        ya = np.zeros_like(y_opt)
        ua[:, :-1] = u_opt[:, 1:]
        ua[:, -1] = ua_end
        xa[:, :-1] = x_opt[:, 1:]
        xa[:, -1] = self.A @ x_opt[:, -1] + self.B @ ua_end
        ya[:, :-1] = y_opt[:, 1:]
        ya[:, -1] = self.C @ xa[:, -1]
        self.xa = xa
        return xa, ya, ua


class QuadPFMPC(SparseDMPC):
    """
    Distributed predecessor follower MPC controller for platoon of vehicles
    using quadratic cost (squared l2 norm).
    """

    def __init__(
        self,
        H: int,
        d_des: float,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        x_min: np.ndarray,
        x_max: np.ndarray,
        u_min: Union[np.ndarray, float],
        u_max: Union[np.ndarray, float],
        Q: np.ndarray,
        Qp: np.ndarray,
        R: np.ndarray,
        Cslew: Optional[np.ndarray] = None,
        dslew: Optional[np.ndarray] = None,
        Qf: Optional[np.ndarray] = None,
        Qpf: Optional[np.ndarray] = None,
        Afx: Optional[np.ndarray] = None,
        Afu: Optional[np.ndarray] = None,
    ):
        """
        Args:
            H: int, prediction horizon
            d_des: desired distance between vehicles
            A: shape (n, n), state transition matrix
            B: shape (n, m), control matrix
            C: shape (p, n), output matrix
            u_min: shape (m,), minimum control input
            u_max: shape (m,), maximum control input
            Q: shape (p, p), assumed state cost matrix
            Qp: shape (p, p), predecessor state cost matrix
            R: shape (m, m), input cost matrix
            Qf: shape (n, n), terminal state cost matrix
            Qpf: shape (p, p), terminal predecessor state cost matrix
            Afx: shape (n, n), state terminal constraint matrix
            Afu: shape (m, m), input terminal constraint matrix
        """
        super().__init__(H=H, d_des=d_des)
        assert H > 1, "Prediction horizon must be greater than 1."
        n, m, p = A.shape[0], B.shape[1], C.shape[0]
        self.n, self.m, self.p = n, m, p
        self.A, self.B, self.C = A, B, C
        self.z_dim = H * (m + n)
        self.Q, self.Qp, self.R = Q, Qp, R

        if Qf is None:
            Qf = np.zeros((n, n))
        self.Qf = Qf
        if Qpf is None:
            Qpf = np.zeros_like(Qp)
        self.Qpf = Qpf
        assert Q.shape == (p, p) and Qp.shape == (p, p)
        assert Qf.shape == (n, n) and Qpf.shape == (p, p)

        # cost: z'Pz + q'z + r (but ignore r since it doesn't affect solution)
        P = block_diag(
            *(
                [R]
                + [np.kron(np.eye(H - 1), block_diag(C.T @ (Q + Qp) @ C, R))]
                + [Qf + C.T @ Qpf @ C]
            )
        )
        self.P = csr_matrix(P)
        # q will be set later once we receive the assumed trajectories

        # equality constraint: Az = b
        Aeq = np.zeros((n * H, self.z_dim))
        Aeq[:, :-n] += block_diag(-B, np.kron(np.eye(H - 1), np.c_[-A, -B]))
        Aeq[:-n, m:-n] += np.kron(np.eye(H - 1), np.c_[np.eye(n), np.zeros((n, m))])
        Aeq[-n:, -n:] += np.eye(n)
        self.Aeq = Aeq
        self.set_terminal_constraint(Afx, Afu)
        # beq will be set later once we receive the assumed trajectories

        # inequality constraint: Cz <= d
        Cblock = np.block(
            [
                [np.eye(m), np.zeros((m, n))],
                [-np.eye(m), np.zeros((m, n))],
                [np.zeros((n, m)), np.eye(n)],
                [np.zeros((n, m)), -np.eye(n)],
            ]
        )
        self.Cslew = Cslew
        self.dslew = dslew
        if Cslew is not None:
            assert dslew is not None
            assert Cslew.shape[1] == n
            assert dslew.shape == (Cslew.shape[0],)
            Cblock = np.block(
                [
                    [Cblock],
                    [np.zeros((Cslew.shape[0], m)), Cslew],
                    [np.zeros((Cslew.shape[0], m)), -Cslew],
                ]
            )
        Cineq = np.kron(np.eye(H), Cblock)
        if Cslew is not None:
            Cineq[2 * m + 2 * n + 2 * Cslew.shape[0] :, : -(m + n)] += np.kron(
                np.eye(H - 1),
                np.block(
                    [
                        [np.zeros((2 * (m + n), m + n))],
                        [np.zeros((Cslew.shape[0], m)), -Cslew],
                        [np.zeros((Cslew.shape[0], m)), Cslew],
                    ]
                ),
            )
        self.Cineq = csr_matrix(Cineq)

        u_min, u_max = np.atleast_1d(u_min), np.atleast_1d(u_max)
        assert u_min.shape == (m,) and u_max.shape == (m,)
        assert x_min.shape == (n,) and x_max.shape == (n,)
        dblock = np.concatenate([u_max, -u_min, x_max, -x_min])
        if dslew is not None:
            assert Cslew is not None
            dblock = np.concatenate([dblock, dslew, dslew])
        self.dineq = np.kron(np.ones((H, 1)), dblock).flatten()

        # set some controller parameters
        self.xa = np.zeros((n, H + 1))

    def control(
        self,
        x0: np.ndarray,
        yp: np.ndarray,
        ur: Optional[np.ndarray] = None,
        bfx: Optional[np.ndarray] = None,
        bfu: Optional[np.ndarray] = None,
        ua_end: Optional[Union[np.ndarray, float]] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Returns a control input based on a distributed MPC policy. Also returns
        a dict with the planned trajectory and planned inputs.

        Args:
            x0: shape (n,), current state
            yp: shape (n, H+1), neighbors state trajectories
            ur: shape (m, H), optional reference control input
            bfx: shape (n,), optional terminal state constraint
            bfu: shape (m,), optional terminal input constraint
            ua_end: shape (m,), optional terminal assumed input

        Returns:
            np.ndarray, shape (m,): control input
            dict: extra information related to the problem
        """
        n, m, p, H = self.n, self.m, self.p, self.H
        assert x0.shape == (n,)
        assert yp.shape == (p, H + 1)
        if ur is not None:
            assert ur.shape == (m, H)
        if bfx is not None:
            assert bfx.shape == (n,)
        if bfu is not None:
            assert bfu.shape == (m,)
        if ua_end is not None:
            ua_end = np.atleast_1d(ua_end)
            assert ua_end.shape == (m,)
        else:
            ua_end = np.zeros(m)

        # initialize assumed trajectory if it's not already set
        if np.allclose(self.xa, 0.0):
            self.init_assumed_trajectory(x0, ur)

        prob, z = self._mpc_problem(x0, yp, ur, bfx, bfu)
        prob.solve(solver=cp.OSQP)
        if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
            x_opt, y_opt, u_opt = self.get_optimal_trajectory(x0, z.value)
            if prob.status == cp.OPTIMAL_INACCURATE:
                print(f"Solver returned with status {prob.status}")
        else:
            raise RuntimeError(f"Solver returned with status {prob.status}")

        # update assumed trajectory
        self.xa, ya, ua = self.update_assumed_trajectory(x_opt, y_opt, u_opt, ua_end)

        info = {
            "status": prob.status,
            "planned states": x_opt,
            "planned inputs": u_opt,
            "planned outputs": y_opt,
            "assumed states": self.xa,
            "assumed inputs": ua,
            "assumed outputs": ya,
        }

        return np.atleast_1d(u_opt[:, 0]), info

    def _mpc_problem(
        self,
        x0: np.ndarray,
        yp: np.ndarray,
        ur: Optional[np.ndarray] = None,
        bfx: Optional[np.ndarray] = None,
        bfu: Optional[np.ndarray] = None,
    ) -> Tuple[cp.Problem, cp.Variable]:
        """
        Creates the CVXPY problem and optimization variable for the MPC problem.

        Args:
            x0: shape (n,), current state
            yp: shape (n, H+1), neighbors state trajectories
            ur: shape (m, H), optional reference control input

        Returns:
            cp.Problem: CVXPY problem
            cp.Variable: optimization variable
        """
        # initialize assumed trajectory if it's not already set
        if np.allclose(self.xa, 0.0):
            self.init_assumed_trajectory(x0, ur)

        n, m, p, H = self.n, self.m, self.p, self.H
        z = cp.Variable(self.z_dim)

        if ur is None:
            ur = np.zeros((m, H))

        # cost: z'Pz + q'z + r (can ignore r since it doesn't affect solution)
        A, C, Q, Qp = self.A, self.C, self.Q, self.Qp
        Qf, Qpf, R = self.Qf, self.Qpf, self.R
        q = np.zeros(self.z_dim)
        q[:m] = R @ ur[:, 0]
        d = np.zeros(p)
        d[0] = self.d_des
        for k, i in enumerate(range(m, m + (H - 1) * (m + n), m + n)):
            q[i : i + n] = C.T @ Q @ C @ self.xa[:, k + 1] + C.T @ Qp @ (
                yp[:, k + 1] - d
            )
            q[i + n : i + n + m] = R @ ur[:, k + 1]
        q[-n:] = Qf @ self.xa[:, -1] + C.T @ Qpf @ (yp[:, -1] - d)
        q *= -2

        r = 0.0
        for k in range(1, H):
            r += cp.quad_form(C @ self.xa[:, k], Q)
            r += cp.quad_form(yp[:, k] - d, Qp)
        for k in range(H):
            r += cp.quad_form(ur[:, k], R)
        r += cp.quad_form(self.xa[:, -1], Qf)
        r += cp.quad_form(yp[:, -1] - d, Qpf)

        # equality constraint: Az = b  (RHS)
        beq = np.zeros(self.Aeq.shape[0])
        beq[:n] = A @ x0
        bend = []
        if self.Afu is not None:
            assert bfu is not None
            assert bfu.shape == (m,)
            bend.append(bfu)
        if self.Afx is not None:
            assert bfx is not None
            assert bfx.shape == (n,)
            bend.append(bfx)
        if len(bend) > 0:
            bend = np.concatenate(bend)
            beq[-len(bend) :] = bend

        # inequality constraint: Cz <= d  - update RHS if dslew
        dineq = copy.deepcopy(self.dineq)
        if self.dslew is not None and self.Cslew is not None:
            dineq[
                2 * (m + n) : 2 * (m + n) + 2 * self.dslew.shape[0]
            ] += np.concatenate([self.Cslew @ x0, -self.Cslew @ x0])

        cost = cp.quad_form(z, self.P) + q @ z + r
        constraints = [self.Aeq @ z == beq, self.Cineq @ z <= dineq]
        return cp.Problem(cp.Minimize(cost), constraints), z

    def set_terminal_constraint(
        self, Afx: Optional[np.ndarray] = None, Afu: Optional[np.ndarray] = None
    ):
        """
        Sets the terminal equality constraint. If none, removes the
        corresponding rows in the equality constraint matrix.

        Args:
            Afx: shape (n, n), state terminal constraint matrix
            Afu: shape (m, m), input terminal constraint matrix
        """
        if type(self.Aeq) == csr_matrix:
            self.Aeq = self.Aeq.toarray()
        n, m = self.n, self.m
        self.Aeq = self.Aeq[: self.H * n, :]

        self.Afu = Afu
        if Afu is not None:
            assert Afu.shape == (m, m)
            self.Aeq = np.r_[self.Aeq, np.zeros((m, self.z_dim))]
            self.Aeq[-m:, -(n + m) : -n] += Afu
        self.Afx = Afx
        if Afx is not None:
            assert Afx.shape == (n, n)
            self.Aeq = np.r_[self.Aeq, np.zeros((n, self.z_dim))]
            self.Aeq[-n:, -n:] += Afx

        self.Aeq = csr_matrix(self.Aeq)

    def get_optimal_trajectory(
        self, x0: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m, n, p, H = self.m, self.n, self.p, self.H
        u_opt = np.zeros((m, H))
        x_opt = np.zeros((n, H + 1))
        y_opt = np.zeros((p, H + 1))
        u_opt[:, 0] = z[:m]
        x_opt[:, 0] = x0
        y_opt[:, 0] = self.C @ x0
        for k, i in enumerate(range(m, m + (H - 1) * (m + n), m + n)):
            x_opt[:, k + 1] = z[i : i + n]
            y_opt[:, k + 1] = self.C @ x_opt[:, k + 1]
            u_opt[:, k + 1] = z[i + n : i + n + m]
        x_opt[:, -1] = z[-n:]
        y_opt[:, -1] = self.C @ x_opt[:, -1]
        return x_opt, y_opt, u_opt


class L1PFMPC(SparseDMPC):
    """
    Distributed predecessor follower MPC controller for platoon of vehicles
    using l1 norm cost.
    """

    def __init__(
        self,
        H: int,
        d_des: float,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        x_min: np.ndarray,
        x_max: np.ndarray,
        u_min: Union[np.ndarray, float],
        u_max: Union[np.ndarray, float],
        q: float,
        qp: float,
        r: float,
        W: Optional[np.ndarray] = None,
        Cslew: Optional[np.ndarray] = None,
        dslew: Optional[np.ndarray] = None,
        qf: Optional[float] = None,
        qpf: Optional[float] = None,
        Afx: Optional[np.ndarray] = None,
        Afu: Optional[np.ndarray] = None,
    ):
        """
        Args:
            H: int, prediction horizon
            d_des: desired distance between vehicles
            A: shape (n, n), state transition matrix
            B: shape (n, m), control matrix
            C: shape (p, n), output matrix
            u_min: shape (m,), minimum control input
            u_max: shape (m,), maximum control input
            q: float, assumed state weight
            qp: float, predecessor state weight
            r: float, input weight
            W: shape (p, p), diagonal weight matrix specifying how much we care
                about penalizing each dimension of the output error
            qf: float, terminal state weight
            qpf: float, terminal predecessor state weight
            Afx: shape (n, n), state terminal constraint matrix
            Afu: shape (m, m), input terminal constraint matrix
        """
        super().__init__(H=H, d_des=d_des)
        assert H > 1, "Prediction horizon must be greater than 1."
        n, m, p = A.shape[0], B.shape[1], C.shape[0]
        self.n, self.m, self.p = n, m, p
        self.A, self.B, self.C = A, B, C
        self.u_min, self.u_max = np.atleast_1d(u_min), np.atleast_1d(u_max)
        assert self.u_min.shape == (m,) and self.u_max.shape == (m,)
        assert x_min.shape == (n,) and x_max.shape == (n,)
        self.x_min, self.x_max = x_min, x_max
        self.q, self.qp, self.r = q, qp, r
        self.z_dim = H * (n + 2 * m + 2 * p)

        if W is None:
            W = np.eye(p)
        assert W.shape == (p, p)
        self.W = W
        self.qf = qf if qf is not None else 0.0
        self.qpf = qpf if qpf is not None else 0.0

        # cost: sum of the introduced slack variables
        self.c = np.zeros(self.z_dim)
        self.c[m : 2 * m] = np.ones(m)
        for i in range(2 * m, 2 * m + (H - 1) * (n + 2 * m + 2 * p), n + 2 * m + 2 * p):
            self.c[i + n : i + n + 2 * p] = np.ones(2 * p)
            self.c[i + n + 2 * p + m : i + n + 2 * p + 2 * m] = np.ones(m)
        if self.qf:
            self.c[-2 * p : -p] = np.ones(p)
        if self.qpf:
            self.c[-p:] = np.ones(p)

        # equality constraint: Az = b
        Aeq = np.zeros((H * n, self.z_dim))
        A1 = np.c_[-B, np.zeros((n, m))]
        A2 = np.c_[-A, np.zeros((n, 2 * p)), -B, np.zeros((n, m))]
        I1 = np.c_[np.eye(n), np.zeros((n, 2 * (p + m)))]
        I2 = np.c_[np.eye(n), np.zeros((n, 2 * p))]
        Aeq[:, : -(n + 2 * p)] += block_diag(*([A1] + [A2] * (H - 1)))
        Aeq[:-n, 2 * m : -(n + 2 * p)] += block_diag(*([I1] * (H - 1)))
        Aeq[-n:, -(n + 2 * p) :] += I2
        self.Aeq = Aeq
        self.set_terminal_constraint(Afx, Afu)
        # beq will be set later once we receive the assumed trajectories

        # inequality constraint: Cz <= d
        self.Cslew, self.dslew = Cslew, dslew
        self.Cineq = self._construct_inequality_constraint_matrix()
        # dineq will need to be set once we receive the assumed trajectories

        # set some controller parameters
        self.xa = np.zeros((n, H + 1))

    def control(
        self,
        x0: np.ndarray,
        yp: np.ndarray,
        ur: Optional[np.ndarray] = None,
        bfx: Optional[np.ndarray] = None,
        bfu: Optional[np.ndarray] = None,
        ua_end: Optional[Union[np.ndarray, float]] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Returns a control input based on a distributed MPC policy. Also returns
        a dict with the planned trajectory and planned inputs.

        Args:
            x0: shape (n,), current state
            yp: shape (n, H+1), neighbors state trajectories
            ur: shape (m, H), optional reference control input
            bfx: shape (n,), optional terminal state constraint
            bfu: shape (m,), optional terminal input constraint
            ua_end: shape (m,), optional terminal assumed input

        Returns:
            np.ndarray, shape (m,): control input
            dict: extra information related to the problem
        """
        n, m, p, H = self.n, self.m, self.p, self.H
        assert x0.shape == (n,)
        assert yp.shape == (p, H + 1)
        if ur is not None:
            assert ur.shape == (m, H)
        if bfx is not None:
            assert bfx.shape == (n,)
        if bfu is not None:
            assert bfu.shape == (m,)
        if ua_end is not None:
            ua_end = np.atleast_1d(ua_end)
            assert ua_end.shape == (m,)
        else:
            ua_end = np.zeros(m)

        # initialize assumed trajectory if it's not already set
        if np.allclose(self.xa, 0.0):
            self.init_assumed_trajectory(x0, ur)

        prob, z = self._mpc_problem(x0, yp, ur, bfx, bfu)
        prob.solve(solver=cp.CLARABEL)
        if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
            soln = self.get_optimal_trajectory(x0, z.value)
            x_opt, y_opt, u_opt, cself, cpred, cu = soln
            if prob.status == cp.OPTIMAL_INACCURATE:
                print(f"Solver returned with status {prob.status}")
        else:
            raise RuntimeError(f"Solver returned with status {prob.status}")

        # update assumed trajectory
        self.xa, ya, ua = self.update_assumed_trajectory(x_opt, y_opt, u_opt, ua_end)

        info = {
            "status": prob.status,
            "planned states": x_opt,
            "planned inputs": u_opt,
            "planned outputs": y_opt,
            "assumed states": self.xa,
            "assumed inputs": ua,
            "assumed outputs": ya,
            "input error": cu,
            "assumed error": cself,
            "predecessor error": cpred,
        }

        return np.atleast_1d(u_opt[:, 0]), info

    def _mpc_problem(
        self,
        x0: np.ndarray,
        yp: np.ndarray,
        ur: Optional[np.ndarray] = None,
        bfx: Optional[np.ndarray] = None,
        bfu: Optional[np.ndarray] = None,
    ) -> Tuple[cp.Problem, cp.Variable]:
        """
        Creates the CVXPY problem and optimization variable for the MPC problem.

        Args:
            x0: shape (n,), current state
            yp: shape (n, H+1), neighbors state trajectories
            ur: shape (m, H), optional reference control input

        Returns:
            cp.Problem: CVXPY problem
            cp.Variable: optimization variable
        """
        n, m, p, H = self.n, self.m, self.p, self.H
        z = cp.Variable(self.z_dim)

        if ur is None:
            ur = np.zeros((m, H))

        # equality constraint: Az = b (RHS)
        beq = np.zeros(self.Aeq.shape[0])
        beq[:n] = self.A @ x0
        bend = []
        if self.Afu is not None:
            assert bfu is not None
            assert bfu.shape == (m,)
            bend.append(bfu)
        if self.Afx is not None:
            assert bfx is not None
            assert bfx.shape == (n,)
            bend.append(bfx)
        if len(bend) > 0:
            bend = np.concatenate(bend)
            beq[-len(bend) :] = bend

        # inequality constraint: Cz <= d (RHS)
        q, qp, r = self.q, self.qp, self.r
        qf, qpf = self.qf, self.qpf
        C, W = self.C, self.W
        Cslew, dslew = self.Cslew, self.dslew
        assert (Cslew is not None and dslew is not None) or (
            Cslew is None and dslew is None
        )
        d_tilde = np.zeros(p)
        d_tilde[0] = self.d_des
        d1 = np.concatenate(
            [
                self.u_max,
                -self.u_min,
                r * ur[:, 0],
                -r * ur[:, 0],
                self.x_max,
                -self.x_min,
                q * W @ C @ self.xa[:, 1],
                -q * W @ C @ self.xa[:, 1],
                qp * W @ (yp[:, 1] - d_tilde),
                -qp * W @ (yp[:, 1] - d_tilde),
            ]
        )
        if Cslew is not None and dslew is not None:
            d1 = np.concatenate([d1, dslew + Cslew @ x0, dslew - Cslew @ x0])
        dineq = [d1]
        for k in range(2, H):
            dk = np.concatenate(
                [
                    self.u_max,
                    -self.u_min,
                    r * ur[:, k - 1],
                    -r * ur[:, k - 1],
                    self.x_max,
                    -self.x_min,
                    q * W @ C @ self.xa[:, k],
                    -q * W @ C @ self.xa[:, k],
                    qp * W @ (yp[:, k] - d_tilde),
                    -qp * W @ (yp[:, k] - d_tilde),
                ]
            )
            if Cslew is not None and dslew is not None:
                dk = np.concatenate([dk, dslew, dslew])
            dineq.append(dk)
        dH = np.concatenate(
            [
                self.u_max,
                -self.u_min,
                r * ur[:, -1],
                -r * ur[:, -1],
                self.x_max,
                -self.x_min,
            ]
        )
        if qf:
            dH = np.concatenate(
                [dH, qf * W @ C @ self.xa[:, -1], -qf * W @ C @ self.xa[:, -1]]
            )
        if qpf:
            dH = np.concatenate(
                [
                    dH,
                    qpf * W @ (yp[:, -1] - d_tilde),
                    -qpf * W @ (yp[:, -1] - d_tilde),
                ]
            )
        if Cslew is not None and dslew is not None:
            dH = np.concatenate([dH, dslew, dslew])
        dineq.append(dH)
        dineq = np.concatenate(dineq)

        cost = self.c @ z
        constraints = [self.Aeq @ z == beq, self.Cineq @ z <= dineq]

        return cp.Problem(cp.Minimize(cost), constraints), z

    def get_optimal_trajectory(
        self, x0: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        m, n, p, H = self.m, self.n, self.p, self.H
        u_opt = np.zeros((m, H))
        x_opt = np.zeros((n, H + 1))
        y_opt = np.zeros((p, H + 1))
        cu = np.zeros_like(u_opt)
        cself = np.zeros_like(y_opt)
        cpred = np.zeros_like(y_opt)
        u_opt[:, 0] = z[:m]
        cu[:, 0] = z[m : 2 * m]
        x_opt[:, 0] = x0
        y_opt[:, 0] = self.C @ x0
        for k, i in enumerate(
            range(2 * m, 2 * m + (H - 1) * (n + 2 * (p + m)), n + 2 * (m + p))
        ):
            x_opt[:, k + 1] = z[i : i + n]
            y_opt[:, k + 1] = self.C @ x_opt[:, k + 1]
            cself[:, k + 1] = z[i + n : i + n + p]
            cpred[:, k + 1] = z[i + n + p : i + n + 2 * p]
            u_opt[:, k + 1] = z[i + n + 2 * p : i + n + 2 * p + m]
            cu[:, k + 1] = z[i + n + 2 * p + m : i + n + 2 * p + 2 * m]
        x_opt[:, -1] = z[-(2 * p + n) : -2 * p]
        y_opt[:, -1] = self.C @ x_opt[:, -1]
        cself[:, -1] = z[-2 * p : -p]
        cpred[:, -1] = z[-p:]
        return x_opt, y_opt, u_opt, cself, cpred, cu

    def set_terminal_constraint(
        self, Afx: Optional[np.ndarray] = None, Afu: Optional[np.ndarray] = None
    ):
        """
        Sets the terminal equality constraint. If none, removes the
        corresponding rows in the equality constraint matrix.

        Args:
            Afx: shape (n, n), state terminal constraint matrix
            Afu: shape (m, m), input terminal constraint matrix
        """
        if type(self.Aeq) == csr_matrix:
            self.Aeq = self.Aeq.toarray()
        n, m, p = self.n, self.m, self.p
        self.Aeq = self.Aeq[: self.H * n, :]

        self.Afu = Afu
        if Afu is not None:
            assert Afu.shape == (m, m)
            self.Aeq = np.r_[self.Aeq, np.zeros((m, self.z_dim))]
            self.Aeq[-m:, -(n + 2 * (p + m)) : -(n + 2 * p + m)] = Afu
        self.Afx = Afx
        if Afx is not None:
            assert Afx.shape == (n, n)
            self.Aeq = np.r_[self.Aeq, np.zeros((n, self.z_dim))]
            self.Aeq[-n:, -(n + 2 * p) : -2 * p] = Afx

        self.Aeq = csr_matrix(self.Aeq)

    def _construct_inequality_constraint_matrix(
        self,
    ) -> np.ndarray:
        """
        Constructs the Cd, Cl, CdH, and ClH matrices as described in Michael
        Shaham's notes on fast DMPC, and uses these matrices to construct
        the inequality constraint matrix Cineq.

        Returns:
            np.ndarray: LHS matrix of inequality constraint Cz <= d
        """
        n, m, p, H = self.n, self.m, self.p, self.H
        r, q, qp = self.r, self.q, self.qp
        qf, qpf = self.qf, self.qpf
        W, C = self.W, self.C
        Cslew = self.Cslew
        Cu = np.block(
            [
                [np.eye(m), np.zeros((m, m))],
                [-np.eye(m), np.zeros((m, m))],
                [r * np.eye(m), -np.eye(m)],
                [-r * np.eye(m), -np.eye(m)],
            ]
        )
        Cx = np.block(
            [
                [np.eye(n), np.zeros((n, 2 * p))],
                [-np.eye(n), np.zeros((n, 2 * p))],
                [q * W @ C, -np.eye(p), np.zeros((p, p))],
                [-q * W @ C, -np.eye(p), np.zeros((p, p))],
                [qp * W @ C, np.zeros((p, p)), -np.eye(p)],
                [-qp * W @ C, np.zeros((p, p)), -np.eye(p)],
            ]
        )
        if Cslew is not None:
            Cx = np.r_[
                Cx,
                np.block(
                    [
                        [Cslew, np.zeros((Cslew.shape[0], 2 * p))],
                        [-Cslew, np.zeros((Cslew.shape[0], 2 * p))],
                    ]
                ),
            ]
        Cd = block_diag(Cu, Cx)

        CxH = np.r_[
            np.c_[np.eye(n), np.zeros((n, 2 * p))],
            np.c_[-np.eye(n), np.zeros((n, 2 * p))],
        ]
        if qf:
            CxH = np.r_[
                CxH,
                np.c_[qf * W @ C, -np.eye(p), np.zeros((p, p))],
                np.c_[-qf * W @ C, -np.eye(p), np.zeros((p, p))],
            ]
        if qpf:
            CxH = np.r_[
                CxH,
                np.c_[qpf * W @ C, np.zeros((p, p)), -np.eye(p)],
                np.c_[-qpf * W @ C, np.zeros((p, p)), -np.eye(p)],
            ]
        if Cslew is not None:
            CxH = np.r_[
                CxH,
                np.c_[Cslew, np.zeros((Cslew.shape[0], 2 * p))],
                np.c_[-Cslew, np.zeros((Cslew.shape[0], 2 * p))],
            ]
        CdH = block_diag(Cu, CxH)

        Cineq = block_diag(*([Cd] * (H - 1)), CdH)
        if Cslew is not None:
            Cl = np.zeros_like(Cd)
            Cl[-2 * Cslew.shape[0] :, 2 * m : -2 * p] = np.r_[-Cslew, Cslew]
            ClH = np.zeros_like(CdH)
            ClH[-2 * Cslew.shape[0] :, 2 * m : -2 * p] = np.r_[-Cslew, Cslew]
            Cineq[Cd.shape[0] :, : -CdH.shape[1]] += block_diag(*([Cl] * (H - 2)), ClH)

        return Cineq

    def set_terminal_cost(
        self, qf: Optional[float] = None, qpf: Optional[float] = None
    ):
        """
        Sets (or removes if None) the terminal inequality state constraints
        that correspond to terminal costs (for use when not using terminal
        constraints, for instance if the problem returns infeasible when
        using an equality terminal constraint).

        Args:
            qf: float, terminal assumed deviation weight
            qpf: float, terminal predecessor deviation weight
        """
        self.qf, self.qpf = qf, qpf
        self.Cineq = self._construct_inequality_constraint_matrix()
