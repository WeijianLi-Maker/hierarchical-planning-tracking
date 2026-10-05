import torch


class TrackingModel:
    """
    Four-dimensional high-fidelity vehicle tracking model.

    State:
        x = [x, y, theta, v]

        x, y  : planar position
        theta : heading angle
        v     : speed

    Input:
        u = [omega, a]

        omega : angular velocity
        a     : acceleration
    """

    def __init__(self):
        self.state_dim = 4
        self.input_dim = 2

    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Continuous-time dynamics x_dot = F(x, u).

        Args:
            x: Tensor with shape (..., 4).
            u: Tensor with shape (..., 2).

        Returns:
            x_dot: Tensor with shape (..., 4).
        """
        theta = x[..., 2]
        speed = x[..., 3]
        omega = u[..., 0]
        acceleration = u[..., 1]

        x_dot = torch.zeros_like(x)
        x_dot[..., 0] = speed * torch.cos(theta)
        x_dot[..., 1] = speed * torch.sin(theta)
        x_dot[..., 2] = omega
        x_dot[..., 3] = acceleration
        return x_dot

    def step(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance the state by one Euler integration step."""
        return x + dt * self.dynamics(x, u)


# Backward-compatible alias for modules that still use the old class name.
QuadratorModel = TrackingModel


if __name__ == "__main__":
    model = TrackingModel()

    x = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    u = torch.tensor([[0.5, 0.2]])

    x_dot = model.dynamics(x, u)
    x_next = model.step(x, u, dt=0.01)

    print("x_dot =", x_dot)
    print("x_next =", x_next)
