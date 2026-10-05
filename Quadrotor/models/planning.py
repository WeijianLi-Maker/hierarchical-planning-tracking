# quadrotor/planning.py

import torch


class VehicleModel:
    """
    3D vehicle planning model.

    State:
        z = [x, y, z]

        x, y, z : position

    Input:
        v = [b_x, b_y, b_z]

        b_x, b_y, b_z : velocity command in each position dimension
    """

    def __init__(self):
        self.state_dim = 3
        self.input_dim = 3

    def dynamics(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Continuous-time dynamics z_dot = f_hat(z, v).

        Args:
            z: Tensor with shape (..., 3)
            v: Tensor with shape (..., 3)

        Returns:
            z_dot: Tensor with shape (..., 3)
        """
        z_dot = torch.zeros_like(z)

        z_dot[..., 0] = v[..., 0]
        z_dot[..., 1] = v[..., 1]
        z_dot[..., 2] = v[..., 2]

        return z_dot

    def step(self, z: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
        """
        One-step Euler integration.

        z_{k+1} = z_k + dt * f_hat(z_k, v_k)
        """
        return z + dt * self.dynamics(z, v)


if __name__ == "__main__":
    model = VehicleModel()

    z = torch.tensor([[0.0, 0.0, 0.0]])
    v = torch.tensor([[0.1, 0.2, 0.3]])

    z_dot = model.dynamics(z, v)
    z_next = model.step(z, v, dt=0.01)

    print("z_dot =", z_dot)
    print("z_next =", z_next)
