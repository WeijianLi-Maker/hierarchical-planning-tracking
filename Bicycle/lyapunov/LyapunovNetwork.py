import torch
from torch import nn


class NeuralLyapunovFunction(nn.Module):
    """
    Neural Lyapunov candidate with the parameterization

        V(e) = C(e; theta)^T C(e; theta) + delta * e^T e.

    The squared network output makes V nonnegative, while the delta term makes
    it positive definite for any nonzero error when delta > 0.
    """

    def __init__(
        self,
        error_dim: int = 6,
        hidden_sizes=(64, 64),
        feature_dim: int = 16,
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
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one hidden layer")
        if any(int(hidden_dim) <= 0 for hidden_dim in hidden_sizes):
            raise ValueError("Every hidden layer size must be positive")

        layers = []
        in_dim = self.error_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(activation())
            in_dim = int(hidden_dim)
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
        if not torch.is_tensor(e):
            raise TypeError("Expected e to be a torch.Tensor")
        if not torch.is_floating_point(e):
            raise TypeError("Expected e to use a floating-point dtype")
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
        if not torch.is_tensor(e_dot):
            raise TypeError("Expected e_dot to be a torch.Tensor")
        if e_dot.shape != e.shape:
            raise ValueError(
                f"Expected e_dot shape {tuple(e.shape)}, got {tuple(e_dot.shape)}"
            )
        if not torch.all(torch.isfinite(e_dot)):
            raise ValueError("Expected finite e_dot values")

        grad = self.gradient(e)
        return torch.sum(grad * e_dot, dim=-1)
