# quadrotor/error_dynamics.py

import torch

from models.tracking import QuadratorModel
from models.planning import VehicleModel


class ErrorDynamics:
    def __init__(self):
        self.tracking_model = QuadratorModel()
        self.planning_model = VehicleModel()

        self.error_dim = 10
        self.projection_matrix = torch.zeros(
            self.tracking_model.state_dim,
            self.planning_model.state_dim,
        )
        self.projection_matrix[0, 0] = 1.0
        self.projection_matrix[4, 1] = 1.0
        self.projection_matrix[8, 2] = 1.0

    def _projection_matrix_like(self, z: torch.Tensor) -> torch.Tensor:
        return self.projection_matrix.to(device=z.device, dtype=z.dtype)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        """
        pi(z) = P z = [z_x, 0, 0, 0, z_y, 0, 0, 0, z_z, 0].
        """
        P = self._projection_matrix_like(z)
        return z @ P.T

    def error(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        e = x - pi(z).

        The three position components compare the quadrotor position with the
        vehicle position. All other quadrotor states are measured relative to 0.
        """
        return x - self.projection(z)

    def projection_dot(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        d/dt pi(z) = P f_hat(z, v) = [b_x, 0, 0, 0, b_y, 0, 0, 0, b_z, 0].
        """
        z_dot = self.planning_model.dynamics(z, v)
        return self.projection(z_dot)

    def dynamics(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        v: torch.Tensor,
        u: torch.Tensor,
        d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        e_dot = x_dot - pi_dot(z), where x_dot = F(x, u, d).
        """
        x_dot = self.tracking_model.dynamics(x, u, d)
        pi_dot = self.projection_dot(z, v)

        return x_dot - pi_dot
    
    def dynamics_from_error(
        self,
        e: torch.Tensor,
        z: torch.Tensor,
        v: torch.Tensor,
        u: torch.Tensor,
        d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute e_dot from error coordinates.

        This uses x = e + pi(z).
        """
        x = e + self.projection(z)
        return self.dynamics(x, z, v, u, d)

    def linearize(
        self,
        e: torch.Tensor,
        z: torch.Tensor,
        v: torch.Tensor,
        u: torch.Tensor,
        d: torch.Tensor | None = None,
    ):
        """
        Linearize continuous-time error dynamics around (e, z, v, u).

        The planner state z, planner input v, and disturbance d are treated as
        fixed external signals.
        The returned affine model is

            e_dot ~= A e + B u + c.

        Returns:
            A: Tensor with shape (10, 10)
            B: Tensor with shape (10, 3)
            c: Tensor with shape (10,)
        """
        e0 = e.detach().clone().reshape(-1).requires_grad_(True)
        z0 = z.detach().clone().reshape(-1)
        v0 = v.detach().clone().reshape(-1)
        u0 = u.detach().clone().reshape(-1).requires_grad_(True)
        d0 = None if d is None else d.detach().clone().reshape(-1)

        def dynamics_e(e_var):
            return self.dynamics_from_error(e_var, z0, v0, u0, d0)

        def dynamics_u(u_var):
            return self.dynamics_from_error(e0, z0, v0, u_var, d0)

        f0 = self.dynamics_from_error(e0, z0, v0, u0, d0)
        A = torch.autograd.functional.jacobian(dynamics_e, e0)
        B = torch.autograd.functional.jacobian(dynamics_u, u0)
        c = f0.detach() - A.detach() @ e0.detach() - B.detach() @ u0.detach()

        return A.detach(), B.detach(), c

    def linearize_discrete(
        self,
        e: torch.Tensor,
        z: torch.Tensor,
        v: torch.Tensor,
        u: torch.Tensor,
        dt: float,
        d: torch.Tensor | None = None,
    ):
        """
        Euler-discretize the local affine error model.

        Continuous model:
            e_dot ~= A e + B u + c

        Discrete model:
            e_next ~= Ad e + Bd u + cd
        """
        A, B, c = self.linearize(e, z, v, u, d)
        I = torch.eye(self.error_dim, device=A.device, dtype=A.dtype)

        Ad = I + dt * A
        Bd = dt * B
        cd = dt * c

        return Ad, Bd, cd


if __name__ == "__main__":
    error_model = ErrorDynamics()

    x = torch.zeros(1, 10)
    z = torch.tensor([[0.0, 0.0, 0.0]])
    v = torch.tensor([[0.1, 0.2, 0.3]])
    u = torch.tensor([[0.0, 0.0, error_model.tracking_model.g / error_model.tracking_model.kT]])

    e = error_model.error(x, z)
    e_dot = error_model.dynamics(x, z, v, u)
    e_dot_from_error = error_model.dynamics_from_error(e, z, v, u)

    print("e =", e)
    print("e_dot =", e_dot)
    print("e_dot_from_error =", e_dot_from_error)
