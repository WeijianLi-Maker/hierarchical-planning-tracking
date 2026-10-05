import numpy as np
import torch
from scipy.optimize import minimize

try:
    import cvxpy as cp
except ModuleNotFoundError:
    cp = None


class LinearizedErrorMPCController:
    """
    One-shot LTV-MPC for the dynamic bicycle tracking-error dynamics.

    At each solve call, the controller:
        1. rolls out a nominal error trajectory using the previous control plan,
        2. linearizes the error dynamics along that nominal trajectory,
        3. solves a QP for the tracking input sequence,
        4. returns the first control input.

    The planner input sequence v_seq is treated as given.
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
    ):
        self.error_model = error_model
        self.planning_model = error_model.planning_model
        self.dt = float(dt)
        self.N = int(horizon)
        self.solver = solver

        self.ne = error_model.error_dim
        self.nu = error_model.tracking_model.input_dim
        self.nz = error_model.planning_model.state_dim
        self.nv = error_model.planning_model.input_dim
        if (self.ne, self.nu, self.nz, self.nv) != (6, 2, 3, 2):
            raise ValueError(
                "Expected Bicycle dimensions error=6, tracking_input=2, "
                "planning_state=3, planning_input=2"
            )
        if self.dt <= 0.0:
            raise ValueError("Expected dt to be positive")
        if self.N <= 0:
            raise ValueError("Expected horizon to be positive")

        default_Q = np.diag([10.0, 10.0, 8.0, 1.0, 1.0, 1.0])
        self.Q = self._check_square_matrix(Q, self.ne, "Q") if Q is not None else default_Q
        self.R = self._check_square_matrix(R, self.nu, "R") if R is not None else np.diag([1.0, 0.1])
        self.Qf = self._check_square_matrix(Qf, self.ne, "Qf") if Qf is not None else 10.0 * self.Q
        default_u_min = np.array([-np.deg2rad(10.0), -3.0])
        default_u_max = np.array([np.deg2rad(10.0), 3.0])
        self.u_min = self._check_vector(u_min, self.nu, "u_min") if u_min is not None else default_u_min
        self.u_max = self._check_vector(u_max, self.nu, "u_max") if u_max is not None else default_u_max
        if np.any(self.u_min > self.u_max):
            raise ValueError("Expected u_min <= u_max")
        self.u_ref = self._default_u_ref() if u_ref is None else self._check_vector(u_ref, self.nu, "u_ref")
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

    def _default_u_ref(self):
        return np.zeros(self.nu)

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=float)

    @staticmethod
    def _to_tensor(x):
        return torch.tensor(x, dtype=torch.float32)

    def _wrap_error_numpy(self, e):
        wrapped = np.asarray(e, dtype=float).copy()
        wrapped[..., 2] = (wrapped[..., 2] + np.pi) % (2.0 * np.pi) - np.pi
        return wrapped

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
        if not torch.all(torch.isfinite(e_dot)):
            raise ValueError(
                "Non-finite error dynamics in nominal rollout. The dynamic bicycle "
                "model requires longitudinal speed v_x to stay nonzero."
            )
        e_next = e + self.dt * self._to_numpy(e_dot)
        return self._wrap_error_numpy(e_next)

    def rollout_planner(self, z0, v_seq):
        z_seq = np.zeros((self.N + 1, self.nz))
        z_seq[0] = np.asarray(z0, dtype=float).reshape(-1)

        for k in range(self.N):
            z_seq[k + 1] = self._planning_step_numpy(z_seq[k], v_seq[k])

        return z_seq

    def rollout_nominal_error(self, e0, z_seq, v_seq, u_seq):
        e_seq = np.zeros((self.N + 1, self.ne))
        e_seq[0] = self._wrap_error_numpy(np.asarray(e0, dtype=float).reshape(-1))

        for k in range(self.N):
            e_seq[k + 1] = self._error_step_numpy(
                e_seq[k],
                z_seq[k],
                v_seq[k],
                u_seq[k],
            )

        return e_seq

    def error_reference(self, v_seq):
        """
        Return the moving error reference induced by the Dubins planner.

        Since pi(z) embeds zero tracking velocity, a vehicle perfectly following
        a moving plan has e_vx = speed_hat and e_omega = omega_hat.
        """
        v_seq = np.asarray(v_seq, dtype=float)
        e_ref = np.zeros((len(v_seq), self.ne))
        e_ref[:, 3] = v_seq[:, 0]
        e_ref[:, 5] = v_seq[:, 1]
        return e_ref

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
        e0 = self._wrap_error_numpy(np.asarray(e0, dtype=float).reshape(-1))
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

        u_bar = np.clip(self.last_solution, self.u_min, self.u_max)
        z_seq = self.rollout_planner(z0, v_seq)
        e_ref = self.error_reference(v_seq)
        e_ref_terminal = e_ref[-1]
        e_bar = self.rollout_nominal_error(e0, z_seq, v_seq, u_bar)
        Ad_seq, Bd_seq, cd_seq = self.linearize_along_trajectory(
            e_bar,
            z_seq,
            v_seq,
            u_bar,
        )

        if cp is None:
            return self._solve_with_scipy(
                e0, z_seq, e_ref, e_ref_terminal, e_bar, u_bar, Ad_seq, Bd_seq, cd_seq
            )

        E = cp.Variable((self.ne, self.N + 1))
        U = cp.Variable((self.nu, self.N))

        cost = 0
        constraints = [E[:, 0] == e0]

        for k in range(self.N):
            cost += cp.quad_form(E[:, k] - e_ref[k], self.Q)
            cost += cp.quad_form(U[:, k] - self.u_ref, self.R)
            constraints.append(
                E[:, k + 1]
                == Ad_seq[k] @ E[:, k] + Bd_seq[k] @ U[:, k] + cd_seq[k]
            )
            constraints.append(U[:, k] >= self.u_min)
            constraints.append(U[:, k] <= self.u_max)

        cost += cp.quad_form(E[:, self.N] - e_ref_terminal, self.Qf)

        problem = cp.Problem(cp.Minimize(cost), constraints)

        status = self._solve_cvxpy_problem(problem)
        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return self._fallback_solution(e0, z_seq, e_bar, u_bar, status)

        U_opt = np.asarray(U.value.T, dtype=float)
        E_pred = np.asarray(E.value.T, dtype=float)

        self.last_solution = np.vstack([U_opt[1:], U_opt[-1:]])

        info = {
            "success": True,
            "status": problem.status,
            "cost": problem.value,
            "E_pred": E_pred,
            "Z_pred": z_seq,
            "U_pred": U_opt,
            "E_nominal": e_bar,
            "U_nominal": u_bar,
            "u_ref": self.u_ref,
            "E_ref": np.vstack([e_ref, e_ref_terminal]),
        }

        return U_opt[0], info

    def _solve_cvxpy_problem(self, problem):
        """Solve the tracking QP and retry with CLARABEL after OSQP failure."""
        primary_solver = str(self.solver).upper()
        attempts = [(self.solver, self._solver_options(primary_solver))]
        if primary_solver != "CLARABEL" and "CLARABEL" in cp.installed_solvers():
            attempts.append(("CLARABEL", {}))

        failures = []
        for solver, options in attempts:
            try:
                problem.solve(
                    solver=solver,
                    warm_start=True,
                    verbose=False,
                    **options,
                )
            except cp.error.SolverError as exc:
                failures.append(f"{solver}: {exc}")
                continue

            if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return problem.status
            failures.append(f"{solver}: {problem.status}")

        return "; ".join(failures)

    @staticmethod
    def _solver_options(solver):
        if solver == "OSQP":
            return {
                "eps_abs": 1e-5,
                "eps_rel": 1e-5,
                "max_iter": 100000,
                "polishing": True,
            }
        return {}

    def _rollout_linear(self, e0, U, Ad_seq, Bd_seq, cd_seq):
        E = np.zeros((self.N + 1, self.ne))
        E[0] = e0

        for k in range(self.N):
            E[k + 1] = Ad_seq[k] @ E[k] + Bd_seq[k] @ U[k] + cd_seq[k]
            E[k + 1] = self._wrap_error_numpy(E[k + 1])

        return E

    def _linear_cost(
        self,
        U_flat,
        e0,
        e_ref,
        e_ref_terminal,
        Ad_seq,
        Bd_seq,
        cd_seq,
    ):
        U = U_flat.reshape(self.N, self.nu)
        E = self._rollout_linear(e0, U, Ad_seq, Bd_seq, cd_seq)

        cost = 0.0
        for k in range(self.N):
            e_error = E[k] - e_ref[k]
            cost += e_error.T @ self.Q @ e_error
            u_error = U[k] - self.u_ref
            cost += u_error.T @ self.R @ u_error
        terminal_error = E[self.N] - e_ref_terminal
        cost += terminal_error.T @ self.Qf @ terminal_error

        if not np.isfinite(cost):
            return 1e30

        return float(cost)

    def _solve_with_scipy(
        self,
        e0,
        z_seq,
        e_ref,
        e_ref_terminal,
        e_bar,
        u_bar,
        Ad_seq,
        Bd_seq,
        cd_seq,
    ):
        bounds = []
        for _ in range(self.N):
            for j in range(self.nu):
                bounds.append((self.u_min[j], self.u_max[j]))

        result = minimize(
            fun=self._linear_cost,
            x0=u_bar.reshape(-1),
            args=(e0, e_ref, e_ref_terminal, Ad_seq, Bd_seq, cd_seq),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 100,
                "ftol": 1e-9,
            },
        )

        if not result.success:
            return self._fallback_solution(e0, z_seq, e_bar, u_bar, result.message)

        U_opt = result.x.reshape(self.N, self.nu)
        E_pred = self._rollout_linear(e0, U_opt, Ad_seq, Bd_seq, cd_seq)

        self.last_solution = np.vstack([U_opt[1:], U_opt[-1:]])

        info = {
            "success": True,
            "status": result.message,
            "cost": result.fun,
            "E_pred": E_pred,
            "Z_pred": z_seq,
            "U_pred": U_opt,
            "E_nominal": e_bar,
            "U_nominal": u_bar,
            "u_ref": self.u_ref,
            "E_ref": np.vstack([e_ref, e_ref_terminal]),
        }

        return U_opt[0], info

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
