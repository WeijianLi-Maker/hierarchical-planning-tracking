import torch

from models.planning import DubinsModel
from models.tracking import TrackingModel


class ErrorDynamics:
    """
    Tracking error dynamics in the planning vehicle's body frame.

    The error is

        e = phi(z) (x - pi(z)),

    where

        pi(z) = [X_hat, Y_hat, psi_hat, 0, 0, 0],
        phi(z) = diag(R(psi_hat)^-1, I_4).
    """

    def __init__(self):
        self.tracking_model = TrackingModel()
        self.planning_model = DubinsModel()
        self.error_dim = self.tracking_model.state_dim

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        """Return ``pi(z) = [X_hat, Y_hat, psi_hat, 0, 0, 0]``."""
        self._check_last_dimension(z, self.planning_model.state_dim, "z")

        pi_z = torch.zeros(
            *z.shape[:-1],
            self.tracking_model.state_dim,
            device=z.device,
            dtype=z.dtype,
        )
        pi_z[..., :3] = z
        return pi_z

    def projection_dot(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Return the time derivative of ``pi(z)``."""
        z_dot = self.planning_model.dynamics(z, v)
        return self.projection(z_dot)

    def mapping_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """Return ``phi(z) = diag(R(psi_hat)^-1, I_4)``."""
        self._check_last_dimension(z, self.planning_model.state_dim, "z")

        psi_hat = z[..., 2]
        cos_psi = torch.cos(psi_hat)
        sin_psi = torch.sin(psi_hat)

        phi = torch.zeros(
            *z.shape[:-1],
            self.error_dim,
            self.error_dim,
            device=z.device,
            dtype=z.dtype,
        )
        phi[..., 0, 0] = cos_psi
        phi[..., 0, 1] = sin_psi
        phi[..., 1, 0] = -sin_psi
        phi[..., 1, 1] = cos_psi
        diagonal = torch.arange(2, self.error_dim, device=z.device)
        phi[..., diagonal, diagonal] = 1.0
        return phi

    def transform(self, value: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Apply ``phi(z)`` to a tracking-state-shaped vector."""
        self._check_last_dimension(value, self.error_dim, "value")
        phi = self.mapping_matrix(z)
        return torch.matmul(phi, value.unsqueeze(-1)).squeeze(-1)

    def error(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute ``e = phi(z) (x - pi(z))``."""
        self._check_last_dimension(x, self.tracking_model.state_dim, "x")
        return self.transform(x - self.projection(z), z)

    def state_from_error(self, e: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Recover ``x = pi(z) + phi(z)^-1 e``."""
        self._check_last_dimension(e, self.error_dim, "e")
        self._check_last_dimension(z, self.planning_model.state_dim, "z")

        psi_hat = z[..., 2]
        cos_psi = torch.cos(psi_hat)
        sin_psi = torch.sin(psi_hat)

        relative_state = e.clone()
        relative_state[..., 0] = cos_psi * e[..., 0] - sin_psi * e[..., 1]
        relative_state[..., 1] = sin_psi * e[..., 0] + cos_psi * e[..., 1]
        return self.projection(z) + relative_state

    def dynamics(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        v: torch.Tensor,
        u: torch.Tensor,
        d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the continuous-time error dynamics.

        The position-error derivative includes the ``phi_dot`` term caused by
        the rotating planning frame.
        """
        if d is not None:
            raise ValueError("The bicycle tracking model does not define a disturbance input")

        e = self.error(x, z)
        relative_dot = self.tracking_model.dynamics(x, u) - self.projection_dot(z, v)
        e_dot = self.transform(relative_dot, z)

        omega_hat = v[..., 1]
        rotating_frame_term_0 = omega_hat * e[..., 1]
        rotating_frame_term_1 = -omega_hat * e[..., 0]
        e_dot[..., 0] = e_dot[..., 0] + rotating_frame_term_0
        e_dot[..., 1] = e_dot[..., 1] + rotating_frame_term_1
        return e_dot

    def dynamics_from_error(
        self,
        e: torch.Tensor,
        z: torch.Tensor,
        v: torch.Tensor,
        u: torch.Tensor,
        d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ``e_dot`` after recovering the tracking state from ``(e, z)``."""
        x = self.state_from_error(e, z)
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
        Return the local affine model ``e_dot ~= A e + B u + c``.
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
        """Euler-discretize the local affine error model."""
        A, B, c = self.linearize(e, z, v, u, d)
        identity = torch.eye(self.error_dim, device=A.device, dtype=A.dtype)
        return identity + dt * A, dt * B, dt * c

    @staticmethod
    def _check_last_dimension(value: torch.Tensor, expected: int, name: str) -> None:
        if value.shape[-1] != expected:
            raise ValueError(
                f"Expected {name} last dimension {expected}, got {value.shape[-1]}"
            )


if __name__ == "__main__":
    error_model = ErrorDynamics()

    x = torch.tensor([[1.0, 0.5, 0.1, 10.0, 0.0, 0.0]])
    z = torch.tensor([[0.0, 0.0, 0.0]])
    v = torch.tensor([[10.0, 0.0]])
    u = torch.tensor([[0.0, 0.0]])

    e = error_model.error(x, z)
    e_dot = error_model.dynamics(x, z, v, u)
    e_dot_from_error = error_model.dynamics_from_error(e, z, v, u)

    print("e =", e)
    print("e_dot =", e_dot)
    print("e_dot_from_error =", e_dot_from_error)
