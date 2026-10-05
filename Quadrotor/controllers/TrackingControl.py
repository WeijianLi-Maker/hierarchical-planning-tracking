import torch
from torch import nn


class NeuralTrackingController(nn.Module):
    """
    Differentiable tracking controller kappa(e, v) for the quadrotor model.

    Inputs:
        e: tracking error with shape (..., 10)
        v: planning input with shape (..., 3)

    Output:
        u: tracking input with shape (..., 3), ordered as [a_x, a_y, a_z].

    The network predicts a bounded residual around u_ref. By default u_ref is
    the hover thrust [0, 0, g / kT], which gives a sensible initial controller
    before learning.
    """

    def __init__(
        self,
        error_dim: int = 10,
        planning_input_dim: int = 3,
        hidden_sizes=(128, 128),
        activation=nn.Tanh,
        u_min=None,
        u_max=None,
        u_ref=None,
        error_scale=None,
        planning_input_scale=None,
        g: float = 9.81,
        kT: float = 0.91,
    ):
        super().__init__()

        self.error_dim = int(error_dim)
        self.planning_input_dim = int(planning_input_dim)
        self.input_dim = self.error_dim + self.planning_input_dim
        self.output_dim = 3

        if u_min is None:
            u_min = [-torch.pi / 18.0, -torch.pi / 18.0, 0.0]
        if u_max is None:
            u_max = [torch.pi / 18.0, torch.pi / 18.0, 1.5 * g]
        if u_ref is None:
            u_ref = [0.0, 0.0, g / kT]

        self.register_buffer("u_min", self._as_control_tensor(u_min, "u_min"))
        self.register_buffer("u_max", self._as_control_tensor(u_max, "u_max"))
        self.register_buffer("u_ref", self._as_control_tensor(u_ref, "u_ref"))
        if error_scale is None:
            error_scale = [2.5, 1.5, 0.4, 0.8, 2.5, 1.5, 0.4, 0.8, 4.0, 1.2]
        if planning_input_scale is None:
            planning_input_scale = [0.5, 0.5, 0.5]
        self.register_buffer(
            "error_scale",
            self._as_positive_tensor(error_scale, self.error_dim, "error_scale"),
            persistent=False,
        )
        self.register_buffer(
            "planning_input_scale",
            self._as_positive_tensor(
                planning_input_scale,
                self.planning_input_dim,
                "planning_input_scale",
            ),
            persistent=False,
        )

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

    @staticmethod
    def _as_control_tensor(value, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if tensor.shape != (3,):
            raise ValueError(f"Expected {name} shape (3,), got {tuple(tensor.shape)}")
        return tensor

    @staticmethod
    def _as_positive_tensor(value, dim: int, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if tensor.shape != (dim,):
            raise ValueError(f"Expected {name} shape ({dim},), got {tuple(tensor.shape)}")
        if torch.any(tensor <= 0.0):
            raise ValueError(f"Expected every element of {name} to be positive")
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
        if e.shape[-1] != self.error_dim:
            raise ValueError(f"Expected e last dimension {self.error_dim}, got {e.shape[-1]}")
        if v.shape[-1] != self.planning_input_dim:
            raise ValueError(
                f"Expected v last dimension {self.planning_input_dim}, got {v.shape[-1]}"
            )

    def forward(self, e: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        self._check_inputs(e, v)

        features = torch.cat(
            [e / self.error_scale, v / self.planning_input_scale],
            dim=-1,
        )
        raw = self.network(features)

        lower_span = self.u_ref - self.u_min
        upper_span = self.u_max - self.u_ref
        residual = torch.where(
            raw >= 0.0,
            upper_span * torch.tanh(raw),
            lower_span * torch.tanh(raw),
        )

        return self.u_ref + residual
