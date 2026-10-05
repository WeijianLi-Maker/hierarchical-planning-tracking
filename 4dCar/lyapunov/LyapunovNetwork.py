import torch
from torch import nn


class NeuralLyapunovFunction(nn.Module):
    """
    Neural Lyapunov candidate for the 4D vehicle tracking error

        e = [e_x, e_y, theta, speed],

    with the parameterization

        V(e) = C(e; theta)^T C(e; theta) + delta * e^T e.

    The squared network output makes V nonnegative, while the delta term makes
    it positive definite for any nonzero error when delta > 0.
    """

    def __init__(
        self,
        error_dim: int = 4,
        hidden_sizes=(32, 32),
        feature_dim: int = 8,
        activation=nn.Tanh,
        delta: float = 1e-1,
    ):
        super().__init__()

        self.error_dim = int(error_dim)
        self.feature_dim = int(feature_dim)
        self.delta = float(delta)

        if self.error_dim <= 0:
            raise ValueError("error_dim must be positive")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.delta <= 0.0:
            raise ValueError("delta must be positive")

        layers = []
        in_dim = self.error_dim
        for hidden_dim in hidden_sizes:
            hidden_dim = int(hidden_dim)
            if hidden_dim <= 0:
                raise ValueError("Expected every hidden layer size to be positive")
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.feature_dim))

        self.C = nn.Sequential(*layers)
        self.reset_output_layer()

    def reset_output_layer(self):
        """
        Start from V(e) = delta * ||e||^2.

        This gives a simple positive definite quadratic Lyapunov candidate
        before learning adjusts the neural feature map C(e; theta).
        """
        final_layer = self.C[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    def _check_error(self, e: torch.Tensor):
        if e.shape[-1] != self.error_dim:
            raise ValueError(f"Expected e last dimension {self.error_dim}, got {e.shape[-1]}")

    def features(self, e: torch.Tensor) -> torch.Tensor:
        """
        Compute C(e; theta).
        """
        self._check_error(e)
        return self.C(e)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """
        Evaluate V(e).

        Args:
            e: Tensor with shape (..., error_dim)

        Returns:
            Tensor with shape (...), one scalar Lyapunov value per error.
        """
        self._check_error(e)

        C_e = self.C(e)
        neural_term = torch.sum(C_e.square(), dim=-1)
        quadratic_term = self.delta * torch.sum(e.square(), dim=-1)

        return neural_term + quadratic_term

    def gradient(self, e: torch.Tensor) -> torch.Tensor:
        """
        Compute dV/de for each error sample.
        """
        self._check_error(e)

        e_req = e.requires_grad_(True)
        V = self.forward(e_req)
        grad = torch.autograd.grad(
            V.sum(),
            e_req,
            create_graph=True,
            retain_graph=True,
        )[0]

        return grad

    def lie_derivative(self, e: torch.Tensor, e_dot: torch.Tensor) -> torch.Tensor:
        """
        Compute V_dot = dV/de * e_dot.
        """
        self._check_error(e)
        if e_dot.shape[-1] != self.error_dim:
            raise ValueError(
                f"Expected e_dot last dimension {self.error_dim}, got {e_dot.shape[-1]}"
            )

        grad = self.gradient(e)
        return torch.sum(grad * e_dot, dim=-1)
