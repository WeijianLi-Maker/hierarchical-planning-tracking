import torch


class TrackingModel:
    """
    Dynamic bicycle tracking model.

    State:
        x = [X, Y, psi, v_x, v_y, omega]

        X, Y  : inertial-frame position
        psi   : inertial heading
        v_x   : longitudinal speed
        v_y   : lateral speed
        omega : yaw rate

    Input:
        u = [delta_f, a_x]

        delta_f : front-wheel steering angle
        a_x     : longitudinal acceleration

    The tire model is undefined at v_x = 0. Callers must keep the
    longitudinal speed nonzero.
    """

    def __init__(
        self,
        m: float = 1.67e3,
        I_z: float = 2.1e3,
        l_f: float = 0.99,
        l_r: float = 1.7,
        C_alpha_f: float = 6.1595e4,
        C_alpha_r: float = 5.2095e4,
    ):
        self.m = float(m)
        self.I_z = float(I_z)
        self.l_f = float(l_f)
        self.l_r = float(l_r)
        self.C_alpha_f = float(C_alpha_f)
        self.C_alpha_r = float(C_alpha_r)

        if self.m <= 0.0:
            raise ValueError("m must be positive")
        if self.I_z <= 0.0:
            raise ValueError("I_z must be positive")
        if self.l_f <= 0.0 or self.l_r <= 0.0:
            raise ValueError("l_f and l_r must be positive")
        if self.C_alpha_f <= 0.0 or self.C_alpha_r <= 0.0:
            raise ValueError("Cornering stiffnesses must be positive")

        self.state_dim = 6
        self.input_dim = 2

    def tire_forces(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return front and rear lateral tire forces ``(F_c_f, F_c_r)``."""
        self._check_dimensions(x, u)

        v_x = x[..., 3]
        v_y = x[..., 4]
        omega = x[..., 5]
        delta_f = u[..., 0]

        alpha_f = (v_y + self.l_f * omega) / v_x - delta_f
        alpha_r = (v_y - self.l_r * omega) / v_x

        F_c_f = -self.C_alpha_f * alpha_f
        F_c_r = -self.C_alpha_r * alpha_r
        return F_c_f, F_c_r

    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Continuous-time dynamics ``x_dot = F(x, u)``.

        Args:
            x: Tensor with shape (..., 6).
            u: Tensor with shape (..., 2).

        Returns:
            x_dot: Tensor with shape (..., 6).
        """
        self._check_dimensions(x, u)

        psi = x[..., 2]
        v_x = x[..., 3]
        v_y = x[..., 4]
        omega = x[..., 5]

        delta_f = u[..., 0]
        a_x = u[..., 1]
        F_c_f, F_c_r = self.tire_forces(x, u)

        x_dot = torch.zeros_like(x)
        x_dot[..., 0] = v_x * torch.cos(psi) - v_y * torch.sin(psi)
        x_dot[..., 1] = v_x * torch.sin(psi) + v_y * torch.cos(psi)
        x_dot[..., 2] = omega
        x_dot[..., 3] = omega * v_y + a_x
        x_dot[..., 4] = -omega * v_x + 2.0 / self.m * (
            F_c_f * torch.cos(delta_f) + F_c_r
        )
        x_dot[..., 5] = 2.0 / self.I_z * (
            self.l_f * F_c_f - self.l_r * F_c_r
        )
        return x_dot

    def step(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance the state by one Euler integration step."""
        return x + dt * self.dynamics(x, u)

    def _check_dimensions(self, x: torch.Tensor, u: torch.Tensor) -> None:
        if x.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected x last dimension {self.state_dim}, got {x.shape[-1]}"
            )
        if u.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected u last dimension {self.input_dim}, got {u.shape[-1]}"
            )


if __name__ == "__main__":
    model = TrackingModel()

    x = torch.tensor([[0.0, 0.0, 0.0, 10.0, 0.0, 0.0]])
    u = torch.tensor([[0.0, 0.0]])

    x_dot = model.dynamics(x, u)
    x_next = model.step(x, u, dt=0.01)

    print("x_dot =", x_dot)
    print("x_next =", x_next)
