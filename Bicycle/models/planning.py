import torch


class DubinsModel:
    """
    Dubins vehicle planning model.

    State:
        z = [X_hat, Y_hat, psi_hat]

        X_hat, Y_hat : planar position
        psi_hat      : heading angle

    Input:
        v = [speed_hat, omega_hat]

        speed_hat : forward speed
        omega_hat : angular velocity
    """

    def __init__(self):
        self.state_dim = 3
        self.input_dim = 2

    def dynamics(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Continuous-time dynamics ``z_dot = f_hat(z, v)``.

        Args:
            z: Tensor with shape (..., 3).
            v: Tensor with shape (..., 2).

        Returns:
            z_dot: Tensor with shape (..., 3).
        """
        self._check_dimensions(z, v)

        psi_hat = z[..., 2]
        speed_hat = v[..., 0]
        omega_hat = v[..., 1]

        z_dot = torch.zeros_like(z)
        z_dot[..., 0] = speed_hat * torch.cos(psi_hat)
        z_dot[..., 1] = speed_hat * torch.sin(psi_hat)
        z_dot[..., 2] = omega_hat
        return z_dot

    def step(self, z: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance the state by one Euler integration step."""
        return z + dt * self.dynamics(z, v)

    def _check_dimensions(self, z: torch.Tensor, v: torch.Tensor) -> None:
        if z.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected z last dimension {self.state_dim}, got {z.shape[-1]}"
            )
        if v.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected v last dimension {self.input_dim}, got {v.shape[-1]}"
            )


if __name__ == "__main__":
    model = DubinsModel()

    z = torch.tensor([[0.0, 0.0, torch.pi / 4.0]])
    v = torch.tensor([[2.0, 0.3]])

    z_dot = model.dynamics(z, v)
    z_next = model.step(z, v, dt=0.01)

    print("z_dot =", z_dot)
    print("z_next =", z_next)
