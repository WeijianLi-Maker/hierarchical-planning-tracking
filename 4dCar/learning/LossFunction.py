from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    """
    Weights for the offline learning objective.
    """

    lyapunov: float = 3.0
    error_bound: float = 3.0
    behavior_cloning: float = 1.0
    origin: float = 1.0
    logdet: float = 0.01


def _check_batch_tensor(value: torch.Tensor, dim: int, name: str) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"Expected {name} to be a torch.Tensor")
    if value.ndim != 2 or value.shape[-1] != dim:
        raise ValueError(f"Expected {name} shape (batch, {dim}), got {tuple(value.shape)}")
    if not torch.all(torch.isfinite(value)):
        raise ValueError(f"Expected {name} to contain only finite values")


def _check_common_inputs(lyapunov, controller, error_model, e, z, v) -> None:
    error_dim = getattr(error_model, "error_dim", e.shape[-1])
    planning_model = getattr(error_model, "planning_model", None)
    planning_state_dim = getattr(planning_model, "state_dim", z.shape[-1])
    planning_input_dim = getattr(planning_model, "input_dim", v.shape[-1])

    _check_batch_tensor(e, error_dim, "e")
    _check_batch_tensor(z, planning_state_dim, "z")
    _check_batch_tensor(v, planning_input_dim, "v")
    if e.shape[0] != z.shape[0] or e.shape[0] != v.shape[0]:
        raise ValueError(
            f"Expected matching batch sizes for e, z, and v, got "
            f"{e.shape[0]}, {z.shape[0]}, and {v.shape[0]}"
        )
    if hasattr(lyapunov, "error_dim") and lyapunov.error_dim != error_dim:
        raise ValueError(
            f"Expected lyapunov.error_dim={error_dim}, got {lyapunov.error_dim}"
        )
    if hasattr(controller, "error_dim") and controller.error_dim != error_dim:
        raise ValueError(
            f"Expected controller.error_dim={error_dim}, got {controller.error_dim}"
        )
    if (
        hasattr(controller, "planning_input_dim")
        and controller.planning_input_dim != planning_input_dim
    ):
        raise ValueError("Controller planning input dimension does not match Planning Model")
    tracking_model = getattr(error_model, "tracking_model", None)
    tracking_input_dim = getattr(tracking_model, "input_dim", None)
    if (
        tracking_input_dim is not None
        and hasattr(controller, "output_dim")
        and controller.output_dim != tracking_input_dim
    ):
        raise ValueError("Controller output dimension does not match Tracking Model input")


def quadratic_form(e: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """
    Compute e^T E e for each error sample.

    Args:
        e: Tensor with shape (..., n)
        E: Tensor with shape (n, n)

    Returns:
        Tensor with shape (...).
    """
    if E.ndim != 2 or E.shape[0] != E.shape[1]:
        raise ValueError(f"Expected E shape (n, n), got {tuple(E.shape)}")
    if e.shape[-1] != E.shape[0]:
        raise ValueError(f"Expected e last dimension {E.shape[0]}, got {e.shape[-1]}")

    return torch.sum((e @ E) * e, dim=-1)


def logdet_spd(E: torch.Tensor) -> torch.Tensor:
    """
    Compute log det(E) for a positive definite matrix.
    """
    sign, logabsdet = torch.linalg.slogdet(E)
    if sign.detach().item() <= 0.0:
        raise ValueError("Expected E to be positive definite with positive determinant")
    return logabsdet


def masked_mean(value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return value.mean()
    if not torch.is_tensor(mask):
        raise TypeError("Expected mask to be a torch.Tensor")
    if mask.shape != value.shape:
        raise ValueError(
            f"Expected mask shape {tuple(value.shape)}, got {tuple(mask.shape)}"
        )
    mask = mask.to(device=value.device, dtype=value.dtype)
    return torch.sum(value * mask) / torch.clamp(mask.sum(), min=1.0)


def lyapunov_decrease_loss(
    lyapunov,
    controller,
    error_model,
    e: torch.Tensor,
    z: torch.Tensor,
    v: torch.Tensor,
    level: float = 1.0,
    origin_weight: float = 1.0,
    decrease_margin: float = 0.01,
    sample_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Lyapunov decrease loss from the paper:

        L_V = E[1_{V(e) >= 1} max(0, V_dot + epsilon)] + alpha V(0).

    The closed-loop error dynamics use u = kappa(e, v).
    """
    _check_common_inputs(lyapunov, controller, error_model, e, z, v)
    V = lyapunov(e)
    u = controller(e, v)
    e_dot = error_model.dynamics_from_error(e, z, v, u)
    V_dot = lyapunov.lie_derivative(e, e_dot)

    outside_level_set = (V >= level).to(dtype=V.dtype)
    decrease_violation = outside_level_set * F.relu(V_dot + decrease_margin)

    e_zero = torch.zeros(1, e.shape[-1], device=e.device, dtype=e.dtype)
    V_zero = lyapunov(e_zero).mean()

    loss = masked_mean(decrease_violation, sample_mask) + origin_weight * V_zero
    info = {
        "lyapunov_loss": loss.detach(),
        "V_mean": masked_mean(V.detach(), sample_mask),
        "V_dot_mean": masked_mean(V_dot.detach(), sample_mask),
        "decrease_violation_mean": masked_mean(decrease_violation.detach(), sample_mask),
        "outside_level_fraction": masked_mean(outside_level_set.detach(), sample_mask),
        "V_zero": V_zero.detach(),
    }

    return loss, info


def error_bound_loss(
    lyapunov,
    e: torch.Tensor,
    E: torch.Tensor,
    level: float = 1.0,
    logdet_weight: float = 0.01,
    sample_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Tracking error bound loss:

        L_E = E[1_{V(e) < 1} max(0, e^T E e - 1)] - beta log det(E).

    This encourages the Lyapunov level set {V(e) <= 1} to lie inside the
    ellipsoid {e: e^T E e <= 1}. Maximizing log det(E) minimizes the volume
    of this ellipsoid, so its contribution to a minimized loss has a minus sign.
    """
    _check_batch_tensor(e, lyapunov.error_dim, "e")
    V = lyapunov(e)
    ellipsoid_value = quadratic_form(e, E)

    inside_level_set = (V < level).to(dtype=V.dtype)
    bound_violation = inside_level_set * F.relu(ellipsoid_value - 1.0)
    logdet = logdet_spd(E)

    loss = masked_mean(bound_violation, sample_mask) - logdet_weight * logdet
    info = {
        "error_bound_loss": loss.detach(),
        "ellipsoid_value_mean": masked_mean(ellipsoid_value.detach(), sample_mask),
        "bound_violation_mean": masked_mean(bound_violation.detach(), sample_mask),
        "inside_level_fraction": masked_mean(inside_level_set.detach(), sample_mask),
        "logdet_E": logdet.detach(),
    }

    return loss, info


def behavior_cloning_loss(
    controller,
    e: torch.Tensor,
    v: torch.Tensor,
    u_expert: torch.Tensor,
    sample_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Behavior cloning loss:

        L_BC = E[||u - u_expert||^2],

    where u = kappa(e, v).
    """
    _check_batch_tensor(e, controller.error_dim, "e")
    _check_batch_tensor(v, controller.planning_input_dim, "v")
    _check_batch_tensor(u_expert, controller.output_dim, "u_expert")
    if e.shape[0] != v.shape[0] or e.shape[0] != u_expert.shape[0]:
        raise ValueError("Expected matching batch sizes for e, v, and u_expert")
    u_pred = controller(e, v)
    squared_error = torch.sum((u_pred - u_expert).square(), dim=-1)
    error_norm = torch.linalg.norm(u_pred.detach() - u_expert.detach(), dim=-1)
    loss = masked_mean(squared_error, sample_mask)
    info = {
        "behavior_cloning_loss": loss.detach(),
        "u_error_norm_mean": masked_mean(error_norm, sample_mask),
    }

    return loss, info


def certified_learning_loss(
    lyapunov,
    controller,
    error_model,
    e: torch.Tensor,
    z: torch.Tensor,
    v: torch.Tensor,
    E: torch.Tensor,
    u_expert: torch.Tensor | None = None,
    weights: LossWeights | None = None,
    level: float = 1.0,
    decrease_margin: float = 0.01,
    use_lyapunov_loss: torch.Tensor | None = None,
    use_error_bound_loss: torch.Tensor | None = None,
    use_behavior_cloning_loss: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Combined offline learning loss:

        L = w_V L_V + w_E L_E + w_BC L_BC.

    The behavior cloning term is included only when u_expert is provided.
    """
    if weights is None:
        weights = LossWeights()
    for name, weight in vars(weights).items():
        if weight < 0.0:
            raise ValueError(f"Expected nonnegative loss weight '{name}', got {weight}")
    _check_common_inputs(lyapunov, controller, error_model, e, z, v)

    L_V, info_V = lyapunov_decrease_loss(
        lyapunov=lyapunov,
        controller=controller,
        error_model=error_model,
        e=e,
        z=z,
        v=v,
        level=level,
        origin_weight=weights.origin,
        decrease_margin=decrease_margin,
        sample_mask=use_lyapunov_loss,
    )
    L_E, info_E = error_bound_loss(
        lyapunov=lyapunov,
        e=e,
        E=E,
        level=level,
        logdet_weight=weights.logdet,
        sample_mask=use_error_bound_loss,
    )

    total = weights.lyapunov * L_V + weights.error_bound * L_E
    info = {}
    info.update(info_V)
    info.update(info_E)

    if u_expert is not None:
        L_BC, info_BC = behavior_cloning_loss(
            controller,
            e,
            v,
            u_expert,
            sample_mask=use_behavior_cloning_loss,
        )
        total = total + weights.behavior_cloning * L_BC
        info.update(info_BC)
    else:
        info["behavior_cloning_loss"] = torch.zeros((), device=e.device, dtype=e.dtype)

    info["total_loss"] = total.detach()

    return total, info
