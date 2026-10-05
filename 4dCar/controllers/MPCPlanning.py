import numpy as np
from scipy.optimize import minimize

try:
    import cvxpy as cp
except ModuleNotFoundError:
    cp = None


class LinearizedPlanningMPCController:
    """
    MPC controller for the 2D single-integrator planning model.

    The continuous and discrete planning dynamics are

        z_dot = v,
        z_next = z + dt * v.

    The controller solves a constrained finite-horizon problem that drives the
    planar planning state z = [x_hat, y_hat] toward z_goal.
    """

    def __init__(
        self,
        planning_model,
        dt=0.04,
        horizon=20,
        Q=None,
        R=None,
        Qf=None,
        z_min=None,
        z_max=None,
        z_constraint_matrix=None,
        z_constraint_lower=None,
        z_constraint_upper=None,
        v_min=None,
        v_max=None,
        v_ref=None,
        solver="OSQP",
    ):
        self.planning_model = planning_model
        self.dt = float(dt)
        self.N = int(horizon)
        self.solver = solver

        if self.dt <= 0.0:
            raise ValueError("Expected dt to be positive")
        if self.N <= 0:
            raise ValueError("Expected horizon to be positive")

        self.nz = planning_model.state_dim
        self.nv = planning_model.input_dim
        if self.nz != 2 or self.nv != 2:
            raise ValueError(
                "Expected a 2D single-integrator planning model with "
                f"state_dim=2 and input_dim=2, got ({self.nz}, {self.nv})"
            )

        self.Q = self._check_square_matrix(Q, self.nz, "Q") if Q is not None else np.eye(self.nz)
        self.R = self._check_square_matrix(R, self.nv, "R") if R is not None else 1e-2 * np.eye(self.nv)
        self.Qf = self._check_square_matrix(Qf, self.nz, "Qf") if Qf is not None else 10.0 * self.Q

        self.z_min = self._check_vector(z_min, self.nz, "z_min") if z_min is not None else -np.inf * np.ones(self.nz)
        self.z_max = self._check_vector(z_max, self.nz, "z_max") if z_max is not None else np.inf * np.ones(self.nz)
        (
            self.z_constraint_matrix,
            self.z_constraint_lower,
            self.z_constraint_upper,
        ) = self._check_linear_state_constraints(
            z_constraint_matrix,
            z_constraint_lower,
            z_constraint_upper,
        )
        self.v_min = self._check_vector(v_min, self.nv, "v_min") if v_min is not None else -0.5 * np.ones(self.nv)
        self.v_max = self._check_vector(v_max, self.nv, "v_max") if v_max is not None else 0.5 * np.ones(self.nv)
        self.v_ref = self._check_vector(v_ref, self.nv, "v_ref") if v_ref is not None else np.zeros(self.nv)

        if np.any(self.z_min > self.z_max):
            raise ValueError("Expected z_min <= z_max")
        if np.any(self.z_constraint_lower > self.z_constraint_upper):
            raise ValueError("Expected z_constraint_lower <= z_constraint_upper")
        if np.any(self.v_min > self.v_max):
            raise ValueError("Expected v_min <= v_max")

        self.v_ref = np.clip(self.v_ref, self.v_min, self.v_max)
        self.last_solution = np.tile(self.v_ref, (self.N, 1))

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

    def _check_linear_state_constraints(self, matrix, lower, upper):
        if matrix is None:
            if lower is not None or upper is not None:
                raise ValueError(
                    "Expected z_constraint_matrix when linear state bounds are provided"
                )
            return np.zeros((0, self.nz)), np.zeros(0), np.zeros(0)

        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.nz:
            raise ValueError(
                f"Expected z_constraint_matrix shape (m, {self.nz}), got {matrix.shape}"
            )

        count = matrix.shape[0]
        lower = -np.inf * np.ones(count) if lower is None else self._check_vector(lower, count, "z_constraint_lower")
        upper = np.inf * np.ones(count) if upper is None else self._check_vector(upper, count, "z_constraint_upper")
        return matrix, lower, upper

    def _planning_step_numpy(self, z, v):
        """Apply the exact discrete-time single-integrator dynamics."""
        return np.asarray(z, dtype=float) + self.dt * np.asarray(v, dtype=float)

    def rollout_nominal(self, z0, v_seq):
        z_seq = np.zeros((self.N + 1, self.nz))
        z_seq[0] = np.asarray(z0, dtype=float).reshape(-1)
        for k in range(self.N):
            z_seq[k + 1] = self._planning_step_numpy(z_seq[k], v_seq[k])
        return z_seq

    def linearize_discrete(self, z, v):
        """
        Return the exact discrete-time affine model.

        The arguments are accepted for interface compatibility; the
        single-integrator matrices are constant everywhere.
        """
        z = np.asarray(z, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float).reshape(-1)
        if z.shape != (self.nz,):
            raise ValueError(f"Expected z shape ({self.nz},), got {z.shape}")
        if v.shape != (self.nv,):
            raise ValueError(f"Expected v shape ({self.nv},), got {v.shape}")

        Ad = np.eye(self.nz)
        Bd = self.dt * np.eye(self.nz, self.nv)
        cd = np.zeros(self.nz)
        return Ad, Bd, cd

    def linearize_along_trajectory(self, z_bar, v_bar):
        Ad_seq = []
        Bd_seq = []
        cd_seq = []
        for k in range(self.N):
            Ad, Bd, cd = self.linearize_discrete(z_bar[k], v_bar[k])
            Ad_seq.append(Ad)
            Bd_seq.append(Bd)
            cd_seq.append(cd)
        return Ad_seq, Bd_seq, cd_seq

    @staticmethod
    def _append_finite_bounds(constraints, variable, lower, upper):
        finite_lower = np.isfinite(lower)
        finite_upper = np.isfinite(upper)
        if np.any(finite_lower):
            constraints.append(variable[finite_lower] >= lower[finite_lower])
        if np.any(finite_upper):
            constraints.append(variable[finite_upper] <= upper[finite_upper])

    def solve(self, z0, z_goal):
        z0 = np.asarray(z0, dtype=float).reshape(-1)
        z_goal = np.asarray(z_goal, dtype=float).reshape(-1)
        if z0.shape != (self.nz,):
            raise ValueError(f"Expected z0 shape ({self.nz},), got {z0.shape}")
        if z_goal.shape != (self.nz,):
            raise ValueError(f"Expected z_goal shape ({self.nz},), got {z_goal.shape}")

        v_bar = np.clip(self.last_solution, self.v_min, self.v_max)
        z_bar = self.rollout_nominal(z0, v_bar)
        Ad_seq, Bd_seq, cd_seq = self.linearize_along_trajectory(z_bar, v_bar)

        if cp is None:
            return self._solve_with_scipy(z0, z_goal, v_bar, Ad_seq, Bd_seq, cd_seq)

        Z = cp.Variable((self.nz, self.N + 1))
        V = cp.Variable((self.nv, self.N))
        constraints = [Z[:, 0] == z0]
        cost = 0

        for k in range(self.N):
            cost += cp.quad_form(Z[:, k] - z_goal, self.Q)
            cost += cp.quad_form(V[:, k] - self.v_ref, self.R)
            constraints.append(
                Z[:, k + 1]
                == Ad_seq[k] @ Z[:, k] + Bd_seq[k] @ V[:, k] + cd_seq[k]
            )
            self._append_finite_bounds(constraints, Z[:, k + 1], self.z_min, self.z_max)
            self._append_finite_bounds(
                constraints,
                self.z_constraint_matrix @ Z[:, k + 1],
                self.z_constraint_lower,
                self.z_constraint_upper,
            )
            self._append_finite_bounds(constraints, V[:, k], self.v_min, self.v_max)

        cost += cp.quad_form(Z[:, self.N] - z_goal, self.Qf)
        problem = cp.Problem(cp.Minimize(cost), constraints)

        try:
            solve_options = {}
            if str(self.solver).upper() == "OSQP":
                solve_options = {"eps_abs": 1e-7, "eps_rel": 1e-7, "polishing": True}
            problem.solve(
                solver=self.solver,
                warm_start=True,
                verbose=False,
                **solve_options,
            )
        except cp.error.SolverError as exc:
            return self._fallback_solution(z0, z_goal, v_bar, str(exc))

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return self._fallback_solution(z0, z_goal, v_bar, problem.status)

        z_pred = np.asarray(Z.value.T, dtype=float)
        v_pred = np.asarray(V.value.T, dtype=float)
        self.last_solution = np.vstack([v_pred[1:], v_pred[-1:]])
        return v_pred[0], self._info(True, problem.status, problem.value, z_goal, z_pred, v_pred)

    def _rollout_linear(self, z0, v_seq, Ad_seq, Bd_seq, cd_seq):
        z_seq = np.zeros((self.N + 1, self.nz))
        z_seq[0] = z0
        for k in range(self.N):
            z_seq[k + 1] = Ad_seq[k] @ z_seq[k] + Bd_seq[k] @ v_seq[k] + cd_seq[k]
        return z_seq

    def _linear_cost(self, v_flat, z0, z_goal, Ad_seq, Bd_seq, cd_seq):
        v_seq = v_flat.reshape(self.N, self.nv)
        z_seq = self._rollout_linear(z0, v_seq, Ad_seq, Bd_seq, cd_seq)
        cost = 0.0
        for k in range(self.N):
            z_error = z_seq[k] - z_goal
            v_error = v_seq[k] - self.v_ref
            cost += z_error.T @ self.Q @ z_error + v_error.T @ self.R @ v_error
        terminal_error = z_seq[-1] - z_goal
        return float(cost + terminal_error.T @ self.Qf @ terminal_error)

    def _state_constraint_values(self, v_flat, z0, Ad_seq, Bd_seq, cd_seq):
        v_seq = v_flat.reshape(self.N, self.nv)
        z_seq = self._rollout_linear(z0, v_seq, Ad_seq, Bd_seq, cd_seq)[1:]
        values = []
        finite_lower = np.isfinite(self.z_min)
        finite_upper = np.isfinite(self.z_max)
        if np.any(finite_lower):
            values.append((z_seq[:, finite_lower] - self.z_min[finite_lower]).reshape(-1))
        if np.any(finite_upper):
            values.append((self.z_max[finite_upper] - z_seq[:, finite_upper]).reshape(-1))
        linear_values = z_seq @ self.z_constraint_matrix.T
        finite_linear_lower = np.isfinite(self.z_constraint_lower)
        finite_linear_upper = np.isfinite(self.z_constraint_upper)
        if np.any(finite_linear_lower):
            values.append(
                (
                    linear_values[:, finite_linear_lower]
                    - self.z_constraint_lower[finite_linear_lower]
                ).reshape(-1)
            )
        if np.any(finite_linear_upper):
            values.append(
                (
                    self.z_constraint_upper[finite_linear_upper]
                    - linear_values[:, finite_linear_upper]
                ).reshape(-1)
            )
        return np.concatenate(values) if values else np.array([1.0])

    def _solve_with_scipy(self, z0, z_goal, v_bar, Ad_seq, Bd_seq, cd_seq):
        bounds = [(self.v_min[j], self.v_max[j]) for _ in range(self.N) for j in range(self.nv)]
        result = minimize(
            fun=self._linear_cost,
            x0=v_bar.reshape(-1),
            args=(z0, z_goal, Ad_seq, Bd_seq, cd_seq),
            method="SLSQP",
            bounds=bounds,
            constraints={
                "type": "ineq",
                "fun": self._state_constraint_values,
                "args": (z0, Ad_seq, Bd_seq, cd_seq),
            },
            options={"maxiter": 200, "ftol": 1e-9},
        )
        if not result.success:
            return self._fallback_solution(z0, z_goal, v_bar, result.message)

        v_pred = result.x.reshape(self.N, self.nv)
        z_pred = self._rollout_linear(z0, v_pred, Ad_seq, Bd_seq, cd_seq)
        self.last_solution = np.vstack([v_pred[1:], v_pred[-1:]])
        return v_pred[0], self._info(True, result.message, result.fun, z_goal, z_pred, v_pred)

    def _fallback_solution(self, z0, z_goal, v_bar, status):
        v_pred = np.clip(v_bar, self.v_min, self.v_max)
        z_pred = self.rollout_nominal(z0, v_pred)
        return v_pred[0], self._info(False, status, np.inf, z_goal, z_pred, v_pred)

    @staticmethod
    def _info(success, status, cost, z_goal, z_pred, v_pred):
        return {
            "success": bool(success),
            "status": str(status),
            "cost": float(cost),
            "z_goal": np.asarray(z_goal, dtype=float),
            "Z_pred": np.asarray(z_pred, dtype=float),
            "V_pred": np.asarray(v_pred, dtype=float),
        }
