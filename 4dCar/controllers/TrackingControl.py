import torch
from torch import nn


class NeuralTrackingController(nn.Module):
    """
    Differentiable tracking controller kappa(e, v) for the 4D vehicle model.

    Inputs:
        e: world-frame tracking error with shape (..., 4)
        v: single-integrator planning input with shape (..., 2)

    Output:
        u: tracking input with shape (..., 2), ordered as [omega, acceleration].

    The network predicts a bounded residual around u_ref and enforces
    kappa(0, 0) = u_ref. By default, the controller starts at [0, 0].
    """

    def __init__(
        self,
        error_dim: int = 4,
        planning_input_dim: int = 2,
        control_dim: int = 2,
        hidden_sizes=(32, 32),
        activation=nn.Tanh,
        u_min=None,
        u_max=None,
        u_ref=None,
    ):
        super().__init__()

        self.error_dim = int(error_dim)
        self.planning_input_dim = int(planning_input_dim)
        self.control_dim = int(control_dim)
        self.input_dim = self.error_dim + self.planning_input_dim
        self.output_dim = self.control_dim

        if self.error_dim <= 0 or self.planning_input_dim <= 0 or self.control_dim <= 0:
            raise ValueError("Expected all controller dimensions to be positive")

        if u_min is None:
            u_min = -torch.ones(self.control_dim)
        if u_max is None:
            u_max = torch.ones(self.control_dim)
        if u_ref is None:
            u_ref = torch.zeros(self.control_dim)

        self.register_buffer("u_min", self._as_control_tensor(u_min, "u_min"))
        self.register_buffer("u_max", self._as_control_tensor(u_max, "u_max"))
        self.register_buffer("u_ref", self._as_control_tensor(u_ref, "u_ref"))

        if not torch.all(torch.isfinite(self.u_min)):
            raise ValueError("Expected u_min to contain only finite values")
        if not torch.all(torch.isfinite(self.u_max)):
            raise ValueError("Expected u_max to contain only finite values")
        if not torch.all(torch.isfinite(self.u_ref)):
            raise ValueError("Expected u_ref to contain only finite values")
        if torch.any(self.u_min >= self.u_max):
            raise ValueError("Expected every element of u_min to be smaller than u_max")

        self.u_ref.data.copy_(torch.clamp(self.u_ref, self.u_min, self.u_max))

        layers = []
        in_dim = self.input_dim
        for hidden_dim in hidden_sizes:
            hidden_dim = int(hidden_dim)
            if hidden_dim <= 0:
                raise ValueError("Expected every hidden layer size to be positive")
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.output_dim))

        self.network = nn.Sequential(*layers)
        self.reset_output_layer()

    def _as_control_tensor(self, value, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if tensor.shape != (self.control_dim,):
            raise ValueError(
                f"Expected {name} shape ({self.control_dim},), got {tuple(tensor.shape)}"
            )
        return tensor

    def reset_output_layer(self):
        """
        Initialize the final layer to zero so kappa starts at u_ref.
        """
        final_layer = self.network[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    def _check_inputs(self, e: torch.Tensor, v: torch.Tensor):
        if e.ndim == 0 or v.ndim == 0:
            raise ValueError("Expected e and v to have at least one dimension")
        if e.shape[-1] != self.error_dim:
            raise ValueError(f"Expected e last dimension {self.error_dim}, got {e.shape[-1]}")
        if v.shape[-1] != self.planning_input_dim:
            raise ValueError(
                f"Expected v last dimension {self.planning_input_dim}, got {v.shape[-1]}"
            )
        if e.shape[:-1] != v.shape[:-1]:
            raise ValueError(
                f"Expected matching leading dimensions for e and v, got "
                f"{tuple(e.shape[:-1])} and {tuple(v.shape[:-1])}"
            )

    def forward(self, e: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        self._check_inputs(e, v)

        features = torch.cat([e, v], dim=-1)
        zero_features = torch.zeros_like(features)
        raw = self.network(features) - self.network(zero_features)

        lower_span = self.u_ref - self.u_min
        upper_span = self.u_max - self.u_ref
        residual = torch.where(
            raw >= 0.0,
            upper_span * torch.tanh(raw),
            lower_span * torch.tanh(raw),
        )

        return self.u_ref + residual
