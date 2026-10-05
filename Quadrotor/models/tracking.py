# quadrotor/tracking.py

import torch

class QuadratorModel:
    """
    10D quadrotor tracking model.

    State:
        x = [x, vx, theta_x, omega_x, y, vy, theta_y, omega_y, z, vz]

        x, y, z                  : position
        vx, vy, vz               : velocity
        theta_x, theta_y         : pitch and roll
        omega_x, omega_y         : pitch and roll rates

    Input:
        u = [a_x, a_y, a_z]

        a_x, a_y : desired pitch and roll angles
        a_z      : vertical thrust

    Disturbance:
        d = [d_x, d_y, d_z]

        d_x, d_y, d_z : bounded additive disturbances on position rates
        |d_x|, |d_y|, |d_z| <= disturbance_bound
    """

    def __init__(
        self,
        d0: float = 10.0,
        d1: float = 8.0,
        n0: float = 10.0,
        kT: float = 0.91,
        g: float = 9.81,
        disturbance_bound: float = 0.1,
    ):
        self.d0 = d0
        self.d1 = d1
        self.n0 = n0
        self.kT = kT
        self.g = g
        self.disturbance_bound = float(disturbance_bound)

        if self.disturbance_bound < 0.0:
            raise ValueError("disturbance_bound must be nonnegative")

        self.state_dim = 10
        self.input_dim = 3
        self.disturbance_dim = 3

    def dynamics(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Continuous-time dynamics x_dot = F(x, u, d).

        Args:
            x: Tensor with shape (..., 10)
            u: Tensor with shape (..., 3)
            d: Optional tensor with shape (..., 3). Defaults to zero disturbance.

        Returns:
            x_dot: Tensor with shape (..., 10)
        """
        d = self._prepare_disturbance(x, d)

        vx = x[..., 1]
        theta_x = x[..., 2]
        omega_x = x[..., 3]
        vy = x[..., 5]
        theta_y = x[..., 6]
        omega_y = x[..., 7]
        vz = x[..., 9]

        ax = u[..., 0]
        ay = u[..., 1]
        az = u[..., 2]
        dx = d[..., 0]
        dy = d[..., 1]
        dz = d[..., 2]

        x_dot = torch.zeros_like(x)

        x_dot[..., 0] = vx + dx
        x_dot[..., 1] = self.g * torch.tan(theta_x)
        x_dot[..., 2] = -self.d1 * theta_x + omega_x
        x_dot[..., 3] = -self.d0 * theta_x + self.n0 * ax
        x_dot[..., 4] = vy + dy
        x_dot[..., 5] = self.g * torch.tan(theta_y)
        x_dot[..., 6] = -self.d1 * theta_y + omega_y
        x_dot[..., 7] = -self.d0 * theta_y + self.n0 * ay
        x_dot[..., 8] = vz + dz
        x_dot[..., 9] = self.kT * az - self.g

        return x_dot

    def step(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        dt: float,
        d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        One-step Euler integration.

        x_{k+1} = x_k + dt * F(x_k, u_k, d_k)
        """
        return x + dt * self.dynamics(x, u, d)

    def _prepare_disturbance(
        self,
        x: torch.Tensor,
        d: torch.Tensor | None,
    ) -> torch.Tensor:
        if d is None:
            return torch.zeros(
                *x.shape[:-1],
                self.disturbance_dim,
                device=x.device,
                dtype=x.dtype,
            )

        d = torch.as_tensor(d, device=x.device, dtype=x.dtype)
        if d.shape[-1] != self.disturbance_dim:
            raise ValueError(
                f"Expected d last dimension {self.disturbance_dim}, got {d.shape[-1]}"
            )
        if torch.any(torch.abs(d) > self.disturbance_bound):
            raise ValueError(
                f"Expected |d_i| <= {self.disturbance_bound}, got maximum "
                f"{torch.max(torch.abs(d)).detach().cpu().item()}"
            )
        return d


if __name__ == "__main__":
    model = QuadratorModel()

    x = torch.zeros(1, model.state_dim)
    u = torch.tensor([[0.0, 0.0, model.g / model.kT]])

    d = torch.tensor([[0.1, -0.1, 0.05]])
    x_dot = model.dynamics(x, u, d)
    x_next = model.step(x, u, dt=0.01, d=d)

    print("x_dot =", x_dot)
    print("x_next =", x_next)
