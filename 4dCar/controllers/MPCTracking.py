import numpy as np
import torch
from scipy.optimize import minimize

try:
    import cvxpy as cp
except ModuleNotFoundError:
    cp = None


class LinearizedErrorMPCController:
    """
    One-shot LTV-MPC for the 4D vehicle world-frame error dynamics.

    At each solve call, the controller:
        1. rolls out a nominal error trajectory using the previous control plan,
        2. linearizes the error dynamics along that nominal trajectory,
        3. solves a QP for the tracking inputs [omega, acceleration],
        4. returns the first control input.

    The 2D single-integrator planner input sequence v_seq is treated as given.
    """

    def __init__(
        self,
        error_model,
        dt=0.04,
        horizon=20,
        Q=None,
        R=None,
        Qf=None,
        u_min=None,
        u_max=None,
        u_ref=None,
        solver="OSQP",
        sqp_iterations=3,
        sqp_tolerance=1e-3,
        nominal_position_gain=1.5,
        nominal_heading_gain=3.0,
        nominal_speed_gain=3.0,
    ):
        self.error_model = error_model
        self.planning_model = error_model.planning_model
        self.dt = float(dt)
        self.N = int(horizon)
        self.solver = solver
        self.sqp_iterations = int(sqp_iterations)
        self.sqp_tolerance = float(sqp_tolerance)
        self.nominal_position_gain = float(nominal_position_gain)
        self.nominal_heading_gain = float(nominal_heading_gain)
        self.nominal_speed_gain = float(nominal_speed_gain)
        if self.dt <= 0.0:
            raise ValueError("Expected dt to be positive")
        if self.N <= 0:
            raise ValueError("Expected horizon to be positive")
        if self.sqp_iterations <= 0:
            raise ValueError("Expected sqp_iterations to be positive")
        if self.sqp_tolerance < 0.0:
            raise ValueError("Expected sqp_tolerance to be nonnegative")
        if min(
            self.nominal_position_gain,
            self.nominal_heading_gain,
            self.nominal_speed_gain,
        ) <= 0.0:
            raise ValueError("Expected nominal feedback gains to be positive")

        self.ne = error_model.error_dim
        self.nu = error_model.tracking_model.input_dim
        self.nz = error_model.planning_model.state_dim
        self.nv = error_model.planning_model.input_dim

        self.Q = self._check_square_matrix(Q, self.ne, "Q") if Q is not None else np.eye(self.ne)
        self.R = self._check_square_matrix(R, self.nu, "R") if R is not None else 1e-2 * np.eye(self.nu)
        self.Qf = self._check_square_matrix(Qf, self.ne, "Qf") if Qf is not None else self.Q
        self._check_cost_matrix(self.Q, "Q")
        self._check_cost_matrix(self.R, "R")
        self._check_cost_matrix(self.Qf, "Qf")

        self.u_min = self._check_vector(u_min, self.nu, "u_min") if u_min is not None else -np.inf * np.ones(self.nu)
        self.u_max = self._check_vector(u_max, self.nu, "u_max") if u_max is not None else np.inf * np.ones(self.nu)
        self.u_ref = self._default_u_ref() if u_ref is None else self._check_vector(u_ref, self.nu, "u_ref")
        if np.any(self.u_min >= self.u_max):
            raise ValueError("Expected every element of u_min to be smaller than u_max")
        self.u_ref = np.clip(self.u_ref, self.u_min, self.u_max)

        self.last_solution = np.tile(self.u_ref, (self.N, 1))

    @staticmethod
    def _check_square_matrix(value, dim, name):
        matrix = np.asarray(value, dtype=float)
        if matrix.shape != (dim, dim):
            raise ValueError(f"Expected {name} shape ({dim}, {dim}), got {matrix.shape}")
        return matrix

    @staticmethod
    def _check_vector(value, dim, name):
        vector = np.asarray(value, dtype=float).reshape(-1)
        if vector.shape != (dim,):
            raise ValueError(f"Expected {name} shape ({dim},), got {vector.shape}")
        return vector

    @staticmethod
    def _check_cost_matrix(matrix, name):
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"Expected {name} to contain only finite values")
        if not np.allclose(matrix, matrix.T):
            raise ValueError(f"Expected {name} to be symmetric")
        if np.min(np.linalg.eigvalsh(matrix)) < -1e-10:
            raise ValueError(f"Expected {name} to be positive semidefinite")

    def _default_u_ref(self):
        """The vehicle's nominal input is zero angular rate and acceleration."""
        return np.zeros(self.nu)

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=float)

    @staticmethod
    def _to_tensor(x):
        return torch.tensor(x, dtype=torch.float32)

    @staticmethod
    def _copy_error_numpy(e):
        """
        Copy an error state without changing its coordinates.

        The current ErrorDynamics definition has e_theta = theta, so wrapping
        the third component here would make the MPC state inconsistent with
        the model passed to the controller.
        """
        return np.asarray(e, dtype=float).copy()

    def _planning_step_numpy(self, z, v):
        z_t = self._to_tensor(z)
        v_t = self._to_tensor(v)
        with torch.no_grad():
            z_dot = self.planning_model.dynamics(z_t, v_t)
        return z + self.dt * self._to_numpy(z_dot)

    def _error_step_numpy(self, e, z, v, u):
        e_t = self._to_tensor(e)
        z_t = self._to_tensor(z)
        v_t = self._to_tensor(v)
        u_t = self._to_tensor(u)
        with torch.no_grad():
            e_dot = self.error_model.dynamics_from_error(e_t, z_t, v_t, u_t)
        e_next = e + self.dt * self._to_numpy(e_dot)
        return self._copy_error_numpy(e_next)

    def rollout_planner(self, z0, v_seq):
        z_seq = np.zeros((self.N + 1, self.nz))
        z_seq[0] = np.asarray(z0, dtype=float).reshape(-1)

        for k in range(self.N):
            z_seq[k + 1] = self._planning_step_numpy(z_seq[k], v_seq[k])

        return z_seq

    def rollout_nominal_error(self, e0, z_seq, v_seq, u_seq):
        e_seq = np.zeros((self.N + 1, self.ne))
        e_seq[0] = self._copy_error_numpy(np.asarray(e0, dtype=float).reshape(-1))

        for k in range(self.N):
            e_seq[k + 1] = self._error_step_numpy(
                e_seq[k],
                z_seq[k],
                v_seq[k],
                u_seq[k],
            )

        return e_seq

    @staticmethod
    def _wrap_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def feedback_nominal(self, e0, z_seq, v_seq):
        """
        Build a dynamically feasible seed that can expose lateral controllability.

        A zero-input nominal at zero speed makes lateral position error invisible
        to a first-order linearization. This seed first points the vehicle toward
        a velocity that combines planner feedforward and position-error feedback.
        """
        e_seq = np.zeros((self.N + 1, self.ne))
        u_seq = np.zeros((self.N, self.nu))
        e_seq[0] = self._copy_error_numpy(e0)

        for k in range(self.N):
            position_error = e_seq[k, :2]
            desired_velocity = v_seq[k] - self.nominal_position_gain * position_error
            desired_speed = np.linalg.norm(desired_velocity)
            if desired_speed > 1e-8:
                desired_heading = np.arctan2(desired_velocity[1], desired_velocity[0])
            else:
                desired_heading = e_seq[k, 2]

            heading_error = self._wrap_angle(desired_heading - e_seq[k, 2])
            u_seq[k, 0] = self.nominal_heading_gain * heading_error
            u_seq[k, 1] = self.nominal_speed_gain * (desired_speed - e_seq[k, 3])
            u_seq[k] = np.clip(u_seq[k], self.u_min, self.u_max)
            e_seq[k + 1] = self._error_step_numpy(
                e_seq[k],
                z_seq[k],
                v_seq[k],
                u_seq[k],
            )

        return e_seq, u_seq

    def linearize_along_trajectory(self, e_bar, z_seq, v_seq, u_bar):
        Ad_seq = []
        Bd_seq = []
        cd_seq = []

        for k in range(self.N):
            e_t = self._to_tensor(e_bar[k])
            z_t = self._to_tensor(z_seq[k])
            v_t = self._to_tensor(v_seq[k])
            u_t = self._to_tensor(u_bar[k])

            Ad, Bd, cd = self.error_model.linearize_discrete(
                e_t,
                z_t,
                v_t,
                u_t,
                self.dt,
            )

            Ad_seq.append(self._to_numpy(Ad))
            Bd_seq.append(self._to_numpy(Bd))
            cd_seq.append(self._to_numpy(cd))

        return Ad_seq, Bd_seq, cd_seq

    def solve(self, e0, z0, v_seq):
        e0 = self._copy_error_numpy(np.asarray(e0, dtype=float).reshape(-1))
        z0 = np.asarray(z0, dtype=float).reshape(-1)
        v_seq = np.asarray(v_seq, dtype=float)

        if e0.shape != (self.ne,):
            raise ValueError(f"Expected e0 shape ({self.ne},), got {e0.shape}")
        if z0.shape != (self.nz,):
            raise ValueError(f"Expected z0 shape ({self.nz},), got {z0.shape}")
        if v_seq.shape != (self.N, self.nv):
            raise ValueError(
                f"Expected v_seq shape ({self.N}, {self.nv}), got {v_seq.shape}"
            )
        if not np.all(np.isfinite(e0)):
            raise ValueError("Expected e0 to contain only finite values")
        if not np.all(np.isfinite(z0)):
            raise ValueError("Expected z0 to contain only finite values")
        if not np.all(np.isfinite(v_seq)):
            raise ValueError("Expected v_seq to contain only finite values")

        z_seq = self.rollout_planner(z0, v_seq)
        e_bar, u_bar = self.feedback_nominal(e0, z_seq, v_seq)

        result = None
        iterations = 0
        for iterations in range(1, self.sqp_iterations + 1):
            Ad_seq, Bd_seq, cd_seq = self.linearize_along_trajectory(
                e_bar,
                z_seq,
                v_seq,
                u_bar,
            )
            result = self._solve_linearized_qp(e0, Ad_seq, Bd_seq, cd_seq, u_bar)
            if not result["success"]:
                return self._fallback_solution(e0, z_seq, e_bar, u_bar, result["status"])

            U_opt = result["U_pred"]
            control_change = np.max(np.abs(U_opt - u_bar))
            u_bar = U_opt
            e_bar = self.rollout_nominal_error(e0, z_seq, v_seq, u_bar)
            if control_change <= self.sqp_tolerance:
                break

        self.last_solution = np.vstack([u_bar[1:], u_bar[-1:]])
        info = {
            "success": True,
            "status": result["status"],
            "cost": self._nonlinear_cost(e_bar, u_bar),
            "E_pred": e_bar,
            "Z_pred": z_seq,
            "U_pred": u_bar,
            "E_nominal": e_bar,
            "U_nominal": u_bar,
            "u_ref": self.u_ref,
            "sqp_iterations": iterations,
        }
        return u_bar[0], info

    def _nonlinear_cost(self, E, U):
        cost = 0.0
        for k in range(self.N):
            cost += E[k].T @ self.Q @ E[k]
            u_error = U[k] - self.u_ref
            cost += u_error.T @ self.R @ u_error
        cost += E[self.N].T @ self.Qf @ E[self.N]
        return float(cost)

    def _solve_linearized_qp(self, e0, Ad_seq, Bd_seq, cd_seq, u_bar):
        if cp is None:
            return self._solve_linearized_with_scipy(e0, Ad_seq, Bd_seq, cd_seq, u_bar)

        E = cp.Variable((self.ne, self.N + 1))
        U = cp.Variable((self.nu, self.N))

        cost = 0
        constraints = [E[:, 0] == e0]
        finite_lower = np.flatnonzero(np.isfinite(self.u_min))
        finite_upper = np.flatnonzero(np.isfinite(self.u_max))

        for k in range(self.N):
            cost += cp.quad_form(E[:, k], self.Q)
            cost += cp.quad_form(U[:, k] - self.u_ref, self.R)

            constraints.append(
                E[:, k + 1]
                == Ad_seq[k] @ E[:, k] + Bd_seq[k] @ U[:, k] + cd_seq[k]
            )
            if finite_lower.size:
                constraints.append(U[finite_lower, k] >= self.u_min[finite_lower])
            if finite_upper.size:
                constraints.append(U[finite_upper, k] <= self.u_max[finite_upper])

        cost += cp.quad_form(E[:, self.N], self.Qf)

        problem = cp.Problem(cp.Minimize(cost), constraints)

        try:
            problem.solve(
                solver=self.solver,
                warm_start=True,
                verbose=False,
            )
        except cp.error.SolverError as exc:
            return {"success": False, "status": str(exc)}

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return {"success": False, "status": problem.status}

        U_opt = np.asarray(U.value.T, dtype=float)
        E_pred = np.asarray(E.value.T, dtype=float)
        return {
            "success": True,
            "status": problem.status,
            "cost": problem.value,
            "E_pred": E_pred,
            "U_pred": U_opt,
        }

    def _rollout_linear(self, e0, U, Ad_seq, Bd_seq, cd_seq):
        E = np.zeros((self.N + 1, self.ne))
        E[0] = e0

        for k in range(self.N):
            E[k + 1] = Ad_seq[k] @ E[k] + Bd_seq[k] @ U[k] + cd_seq[k]

        return E

    def _linear_cost(self, U_flat, e0, Ad_seq, Bd_seq, cd_seq):
        U = U_flat.reshape(self.N, self.nu)
        E = self._rollout_linear(e0, U, Ad_seq, Bd_seq, cd_seq)

        cost = 0.0
        for k in range(self.N):
            cost += E[k].T @ self.Q @ E[k]
            u_error = U[k] - self.u_ref
            cost += u_error.T @ self.R @ u_error
        cost += E[self.N].T @ self.Qf @ E[self.N]

        if not np.isfinite(cost):
            return 1e30

        return float(cost)

    def _solve_linearized_with_scipy(self, e0, Ad_seq, Bd_seq, cd_seq, u_bar):
        bounds = []
        for _ in range(self.N):
            for j in range(self.nu):
                bounds.append((self.u_min[j], self.u_max[j]))

        result = minimize(
            fun=self._linear_cost,
            x0=u_bar.reshape(-1),
            args=(e0, Ad_seq, Bd_seq, cd_seq),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 100,
                "ftol": 1e-9,
            },
        )

        if not result.success:
            return {"success": False, "status": result.message}

        U_opt = result.x.reshape(self.N, self.nu)
        E_pred = self._rollout_linear(e0, U_opt, Ad_seq, Bd_seq, cd_seq)

        return {
            "success": True,
            "status": result.message,
            "cost": result.fun,
            "E_pred": E_pred,
            "U_pred": U_opt,
        }

    def _fallback_solution(self, e0, z_seq, e_bar, u_bar, status):
        u0 = np.clip(u_bar[0], self.u_min, self.u_max)

        info = {
            "success": False,
            "status": status,
            "cost": np.inf,
            "E_pred": e_bar,
            "Z_pred": z_seq,
            "U_pred": u_bar,
            "E_nominal": e_bar,
            "U_nominal": u_bar,
            "u_ref": self.u_ref,
        }

        return u0, info
