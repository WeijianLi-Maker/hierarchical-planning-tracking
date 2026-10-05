import torch


class PlanningModel:
    """
    Two-dimensional single-integrator planning model.

    State:
        z = [x_hat, y_hat]

    Input:
        v = [u_x, u_y]
    """

    def __init__(self):
        self.state_dim = 2
        self.input_dim = 2

    def dynamics(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Continuous-time dynamics z_dot = f_hat(z, v).

        Args:
            z: Tensor with shape (..., 2).
            v: Tensor with shape (..., 2).

        Returns:
            z_dot: Tensor with shape (..., 2).
        """
        return v

    def step(self, z: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance the planning state by one Euler integration step."""
        return z + dt * self.dynamics(z, v)


# Backward-compatible alias for modules that still use the old class name.
VehicleModel = PlanningModel


if __name__ == "__main__":
    model = PlanningModel()

    z = torch.tensor([[0.0, 0.0]])
    v = torch.tensor([[0.1, 0.2]])

    z_dot = model.dynamics(z, v)
    z_next = model.step(z, v, dt=0.01)

    print("z_dot =", z_dot)
    print("z_next =", z_next)
