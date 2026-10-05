import torch
from torch import nn


class NeuralTrackingController(nn.Module):
    """
    Differentiable tracking controller kappa(e, v) for the bicycle model.

    Inputs:
        e: tracking error with shape (..., 6)
        v: Dubins planning input with shape (..., 2)

    Output:
        u: tracking input with shape (..., 2), ordered as [delta_f, a_x].

    The network predicts a bounded residual around u_ref. By default u_ref is
    [0, 0], corresponding to zero steering and zero acceleration.
    """

    def __init__(
        self,
        error_dim: int = 6,
        planning_input_dim: int = 2,
        hidden_sizes=(64, 64),
        activation=nn.Tanh,
        u_min=None,
        u_max=None,
        u_ref=None,
    ):
        super().__init__()

        self.error_dim = int(error_dim)
        self.planning_input_dim = int(planning_input_dim)
        self.input_dim = self.error_dim + self.planning_input_dim
        self.output_dim = 2

        if u_min is None:
            u_min = [-torch.pi / 18.0, -3.0]
        if u_max is None:
            u_max = [torch.pi / 18.0, 3.0]
        if u_ref is None:
            u_ref = [0.0, 0.0]

        self.register_buffer("u_min", self._as_control_tensor(u_min, "u_min"))
        self.register_buffer("u_max", self._as_control_tensor(u_max, "u_max"))
        self.register_buffer("u_ref", self._as_control_tensor(u_ref, "u_ref"))
        self.register_buffer("feature_center", torch.zeros(self.input_dim))
        self.register_buffer("feature_scale", torch.ones(self.input_dim))

        if torch.any(self.u_min >= self.u_max):
            raise ValueError("Expected every element of u_min to be smaller than u_max")

        self.u_ref.data.copy_(torch.clamp(self.u_ref, self.u_min, self.u_max))

        layers = []
        in_dim = self.input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(activation())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, self.output_dim))

        self.network = nn.Sequential(*layers)
        self.reset_output_layer()

    def _as_control_tensor(self, value, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if tensor.shape != (self.output_dim,):
            raise ValueError(
                f"Expected {name} shape ({self.output_dim},), got {tuple(tensor.shape)}"
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

    def set_feature_normalization(
        self,
        center: torch.Tensor,
        scale: torch.Tensor,
        min_scale: float = 1e-3,
    ) -> None:
        center = torch.as_tensor(center, dtype=self.feature_center.dtype).reshape(-1)
        scale = torch.as_tensor(scale, dtype=self.feature_scale.dtype).reshape(-1)
        if center.shape != (self.input_dim,) or scale.shape != (self.input_dim,):
            raise ValueError(f"Expected feature normalization shape ({self.input_dim},)")
        self.feature_center.copy_(center)
        self.feature_scale.copy_(torch.clamp(scale, min=float(min_scale)))

    def _check_inputs(self, e: torch.Tensor, v: torch.Tensor):
        if e.shape[-1] != self.error_dim:
            raise ValueError(f"Expected e last dimension {self.error_dim}, got {e.shape[-1]}")
        if v.shape[-1] != self.planning_input_dim:
            raise ValueError(
                f"Expected v last dimension {self.planning_input_dim}, got {v.shape[-1]}"
            )

    def forward(self, e: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        self._check_inputs(e, v)

        features = torch.cat([e, v], dim=-1)
        features = (features - self.feature_center) / self.feature_scale
        raw = self.network(features)

        lower_span = self.u_ref - self.u_min
        upper_span = self.u_max - self.u_ref
        residual = torch.where(
            raw >= 0.0,
            upper_span * torch.tanh(raw),
            lower_span * torch.tanh(raw),
        )

        return self.u_ref + residual
