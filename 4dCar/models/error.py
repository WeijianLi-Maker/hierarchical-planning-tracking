import torch

from models.planning import PlanningModel
from models.tracking import TrackingModel


class ErrorDynamics:
    """
    Error dynamics between the 4D tracking model and 2D planning model.

    The planning state is embedded into the tracking state as

        pi(z) = [z_x, z_y, 0, 0],

    and the error is expressed in the world frame with phi equal to identity:

        e = phi(x, z) (x - pi(z)).
        phi(x, z) = I.

    Since pi(z) = [z_x, z_y, 0, 0], the four-dimensional error is

        e = [e_x, e_y, e_theta, e_v],

    where e_theta = theta and e_v = speed.
    """

    def __init__(self):
        self.tracking_model = TrackingModel()
        self.planning_model = PlanningModel()
        self.error_dim = 4
        self.error_coordinates = "world"
        self.error_dynamics_version = 1

    def model_signature(self) -> dict[str, int | str]:
        """Return metadata that identifies the current model and error semantics."""
        return {
            "error_dim": self.error_dim,
            "planning_state_dim": self.planning_model.state_dim,
            "planning_input_dim": self.planning_model.input_dim,
            "tracking_state_dim": self.tracking_model.state_dim,
            "tracking_input_dim": self.tracking_model.input_dim,
            "error_coordinates": self.error_coordinates,
            "error_dynamics_version": self.error_dynamics_version,
        }

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        """Return pi(z) = [z_x, z_y, 0, 0]."""
        zeros = torch.zeros(*z.shape[:-1], 2, device=z.device, dtype=z.dtype)
        return torch.cat((z, zeros), dim=-1)

    def projection_dot(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Return d/dt pi(z) = [u_x, u_y, 0, 0]."""
        return self.projection(self.planning_model.dynamics(z, v))

    def phi(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """
        Return the identity error-coordinate transformation phi(x, z) = I.
        """
        return torch.eye(
            self.error_dim,
            device=x.device,
            dtype=x.dtype,
        ).expand(*x.shape[:-1], self.error_dim, self.error_dim)

    def error(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return e = phi(x, z) (x - pi(z))."""
        state_difference = x - self.projection(z)
        return torch.matmul(
            self.phi(x, z),
            state_difference.unsqueeze(-1),
        ).squeeze(-1)

    def state_from_error(self, e: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Recover the tracking state x from error coordinates and z."""
        return e + self.projection(z)

    def dynamics(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        v: torch.Tensor,
        u: torch.Tensor,
        d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the world-frame error dynamics from tracking state x."""
        if d is not None:
            raise ValueError("The current tracking model does not define a disturbance.")
        return self.dynamics_from_error(self.error(x, z), z, v, u)

    def dynamics_from_error(
        self,
        e: torch.Tensor,
        z: torch.Tensor,
        v: torch.Tensor,
        u: torch.Tensor,
        d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the analytical error dynamics.

        With e = [e_x, e_y, e_theta, e_v], v = [v_hat_x, v_hat_y], and
        u = [omega, acceleration],

            e_x_dot = e_v cos(e_theta) - v_hat_x
            e_y_dot = e_v sin(e_theta) - v_hat_y
            e_theta_dot = omega
            e_v_dot = acceleration
        """
        if d is not None:
            raise ValueError("The current tracking model does not define a disturbance.")

        e_theta = e[..., 2]
        e_v = e[..., 3]
        planner_v_x = v[..., 0]
        planner_v_y = v[..., 1]
        omega = u[..., 0]
        acceleration = u[..., 1]

        cos_theta = torch.cos(e_theta)
        sin_theta = torch.sin(e_theta)

        e_dot = torch.zeros_like(e)
        e_dot[..., 0] = e_v * cos_theta - planner_v_x
        e_dot[..., 1] = e_v * sin_theta - planner_v_y
        e_dot[..., 2] = omega
        e_dot[..., 3] = acceleration
        return e_dot

    def linearize(
        self,
        e: torch.Tensor,
        z: torch.Tensor,
        v: torch.Tensor,
        u: torch.Tensor,
        d: torch.Tensor | None = None,
    ):
        """
        Linearize the continuous-time error dynamics around (e, z, v, u).

        The returned affine model is e_dot ~= A e + B u + c.
        """
        e0 = e.detach().clone().reshape(-1).requires_grad_(True)
        z0 = z.detach().clone().reshape(-1)
        v0 = v.detach().clone().reshape(-1)
        u0 = u.detach().clone().reshape(-1).requires_grad_(True)

        def dynamics_e(e_var):
            return self.dynamics_from_error(e_var, z0, v0, u0, d)

        def dynamics_u(u_var):
            return self.dynamics_from_error(e0, z0, v0, u_var, d)

        f0 = self.dynamics_from_error(e0, z0, v0, u0, d)
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
        """Euler-discretize the local affine error model."""
        A, B, c = self.linearize(e, z, v, u, d)
        identity = torch.eye(self.error_dim, device=A.device, dtype=A.dtype)
        return identity + dt * A, dt * B, dt * c


if __name__ == "__main__":
    error_model = ErrorDynamics()

    x = torch.tensor([[2.0, 3.0, 0.5, 1.0]])
    z = torch.tensor([[1.0, 1.0]])
    v = torch.tensor([[0.2, 0.3]])
    u = torch.tensor([[0.1, -0.2]])

    e = error_model.error(x, z)
    recovered_x = error_model.state_from_error(e, z)
    e_dot = error_model.dynamics(x, z, v, u)
    e_dot_from_error = error_model.dynamics_from_error(e, z, v, u)

    print("e =", e)
    print("recovered_x =", recovered_x)
    print("e_dot =", e_dot)
    print("e_dot_from_error =", e_dot_from_error)
