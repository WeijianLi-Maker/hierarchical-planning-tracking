from dataclasses import dataclass
import math

import torch
from torch import nn

from learning.LossFunction import LossWeights, certified_learning_loss, quadratic_form


@dataclass
class TrainingConfig:
    """
    Basic optimizer and loop settings for offline learning.
    """

    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = 10.0
    level: float = 1.0
    decrease_margin: float = 0.01
    max_steps: int | None = None
    device: str = "cpu"


class LearnableEllipsoid(nn.Module):
    """
    Positive definite parameterization of the tracking error bound matrix E.

    Instead of optimizing E directly, this module learns a lower triangular
    Cholesky-like factor L and returns

        E = L L^T.

    The diagonal of L is constrained to be positive with softplus, so E remains
    positive definite throughout training.
    """

    def __init__(self, dim: int = 6, init_scale: float = 1.0, min_diag: float = 1e-2):
        super().__init__()

        self.dim = int(dim)
        self.min_diag = float(min_diag)

        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive")
        if self.min_diag <= 0.0:
            raise ValueError("min_diag must be positive")

        init_diag = math.sqrt(init_scale)
        raw_diag_value = self._inverse_softplus(init_diag - self.min_diag)

        self.raw_lower = nn.Parameter(torch.zeros(self.dim, self.dim))
        self.raw_diag = nn.Parameter(torch.full((self.dim,), raw_diag_value))

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        if value <= 0.0:
            raise ValueError("value must be positive")
        return math.log(math.expm1(value))

    def factor(self) -> torch.Tensor:
        """
        Return the lower triangular factor L.
        """
        lower = torch.tril(self.raw_lower, diagonal=-1)
        diag = torch.nn.functional.softplus(self.raw_diag) + self.min_diag
        return lower + torch.diag(diag)

    def matrix(self) -> torch.Tensor:
        """
        Return E = L L^T.
        """
        L = self.factor()
        return L @ L.T

    def forward(self) -> torch.Tensor:
        return self.matrix()

    def value(self, e: torch.Tensor) -> torch.Tensor:
        """
        Compute e^T E e.
        """
        return quadratic_form(e, self.matrix())

    def contains(self, e: torch.Tensor, level: float = 1.0) -> torch.Tensor:
        """
        Check whether e lies inside {e: e^T E e <= level}.
        """
        return self.value(e) <= level


def make_optimizer(
    lyapunov: nn.Module,
    controller: nn.Module,
    ellipsoid: nn.Module,
    config: TrainingConfig | None = None,
) -> torch.optim.Optimizer:
    """
    Create one optimizer over V, kappa, and E parameters.
    """
    if config is None:
        config = TrainingConfig()

    parameters = [parameter for parameter in lyapunov.parameters() if parameter.requires_grad]
    parameters += [
        parameter for parameter in controller.parameters() if parameter.requires_grad
    ]
    parameters += [
        parameter for parameter in ellipsoid.parameters() if parameter.requires_grad
    ]

    return torch.optim.Adam(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _require_batch_key(batch: dict, key: str) -> torch.Tensor:
    if key not in batch:
        raise KeyError(f"Expected batch to contain key '{key}'")
    value = batch[key]
    if not torch.is_tensor(value):
        raise TypeError(f"Expected batch['{key}'] to be a torch.Tensor")
    return value


def _check_last_dimension(value: torch.Tensor, expected: int, name: str) -> None:
    if value.shape[-1] != expected:
        raise ValueError(
            f"Expected batch['{name}'] last dimension {expected}, got {value.shape[-1]}"
        )


def _validate_bicycle_batch(batch, error_model, controller, ellipsoid) -> None:
    e = _require_batch_key(batch, "e")
    z = _require_batch_key(batch, "z")
    v = _require_batch_key(batch, "v")

    _check_last_dimension(e, error_model.error_dim, "e")
    _check_last_dimension(z, error_model.planning_model.state_dim, "z")
    _check_last_dimension(v, error_model.planning_model.input_dim, "v")

    if controller.error_dim != error_model.error_dim:
        raise ValueError("Controller error_dim does not match ErrorDynamics")
    if controller.planning_input_dim != error_model.planning_model.input_dim:
        raise ValueError("Controller planning_input_dim does not match planning model")
    if ellipsoid.dim != error_model.error_dim:
        raise ValueError("Ellipsoid dimension does not match ErrorDynamics")

    u_expert = batch.get("u_expert")
    if u_expert is not None:
        if not torch.is_tensor(u_expert):
            raise TypeError("Expected batch['u_expert'] to be a torch.Tensor")
        _check_last_dimension(u_expert, error_model.tracking_model.input_dim, "u_expert")

    if "d" in batch:
        raise ValueError("The Bicycle model does not define a disturbance input 'd'")


def train_step(
    lyapunov: nn.Module,
    controller: nn.Module,
    ellipsoid: LearnableEllipsoid,
    error_model,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    weights: LossWeights | None = None,
    config: TrainingConfig | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Run one gradient update for V, kappa, and E.

    Expected batch keys:
        e: Tensor with shape (batch, error_dim)
        z: Tensor with shape (batch, planning_dim), used by error dynamics
        v: Tensor with shape (batch, planning_input_dim), used by the controller
        u_expert: optional Tensor with shape (batch, control_dim)
    """
    if config is None:
        config = TrainingConfig()

    device = torch.device(config.device)
    lyapunov.train()
    controller.train()
    ellipsoid.train()

    lyapunov.to(device)
    controller.to(device)
    ellipsoid.to(device)

    batch = _to_device(batch, device)
    _validate_bicycle_batch(batch, error_model, controller, ellipsoid)
    e = _require_batch_key(batch, "e")
    z = _require_batch_key(batch, "z")
    v = _require_batch_key(batch, "v")
    u_expert = batch.get("u_expert")

    optimizer.zero_grad(set_to_none=True)

    E = ellipsoid.matrix()
    loss, info = certified_learning_loss(
        lyapunov=lyapunov,
        controller=controller,
        error_model=error_model,
        e=e,
        z=z,
        v=v,
        E=E,
        u_expert=u_expert,
        weights=weights,
        level=config.level,
        decrease_margin=config.decrease_margin,
        use_lyapunov_loss=batch.get("use_lyapunov_loss"),
        use_error_bound_loss=batch.get("use_error_bound_loss"),
        use_behavior_cloning_loss=batch.get("use_behavior_cloning_loss"),
    )
    loss.backward()

    parameters = list(lyapunov.parameters())
    parameters += list(controller.parameters())
    parameters += list(ellipsoid.parameters())
    if config.gradient_clip_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm)
    else:
        grad_norm = _compute_grad_norm(parameters)

    optimizer.step()

    with torch.no_grad():
        E_after = ellipsoid.matrix()
        eigvals = torch.linalg.eigvalsh(E_after)

    info["grad_norm"] = grad_norm.detach()
    info["E_min_eig"] = eigvals.min().detach()
    info["E_max_eig"] = eigvals.max().detach()

    return loss.detach(), info


def _compute_grad_norm(parameters) -> torch.Tensor:
    squared_norm = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        term = parameter.grad.detach().norm().square()
        squared_norm = term if squared_norm is None else squared_norm + term

    if squared_norm is None:
        return torch.zeros(())
    return torch.sqrt(squared_norm)


def train(
    lyapunov: nn.Module,
    controller: nn.Module,
    ellipsoid: LearnableEllipsoid,
    error_model,
    data_loader,
    optimizer: torch.optim.Optimizer,
    weights: LossWeights | None = None,
    config: TrainingConfig | None = None,
) -> list[dict]:
    """
    Train V, kappa, and E over batches from data_loader.

    The data_loader can be any iterable that yields dictionaries with keys
    required by train_step.
    """
    if config is None:
        config = TrainingConfig()

    history = []
    for step, batch in enumerate(data_loader):
        if config.max_steps is not None and step >= config.max_steps:
            break

        _, info = train_step(
            lyapunov=lyapunov,
            controller=controller,
            ellipsoid=ellipsoid,
            error_model=error_model,
            optimizer=optimizer,
            batch=batch,
            weights=weights,
            config=config,
        )
        info = {key: _detach_to_float(value) for key, value in info.items()}
        info["step"] = step
        history.append(info)

    return history


def _detach_to_float(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu()
    return value
