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
    logdet: float = 1.0


def moving_reference_error(e: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Return the deviation from the moving Dubins tracking reference.

    The absolute Bicycle error stores tracking longitudinal speed and yaw rate
    in e[3] and e[5]. The controlled equilibrium is therefore
    [0, 0, 0, speed_hat, 0, omega_hat], not the origin.
    """
    if e.shape[-1] != 6:
        raise ValueError(f"Expected e last dimension 6, got {e.shape[-1]}")
    if v.shape[-1] != 2:
        raise ValueError(f"Expected v last dimension 2, got {v.shape[-1]}")

    reference = torch.zeros_like(e)
    reference[..., 3] = v[..., 0]
    reference[..., 5] = v[..., 1]
    return e - reference


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

        L_V = E[1_{0 < V(r) <= 1} max(0, V_dot + lambda V)] + alpha V(0),

    where r is the deviation from the moving tracking reference. Planner inputs
    are treated as piecewise constant over the continuous-time derivative.

    The closed-loop error dynamics use u = kappa(e, v).
    """
    r = moving_reference_error(e, v)
    V = lyapunov(r)
    u = controller(e, v)
    e_dot = error_model.dynamics_from_error(e, z, v, u)
    if not torch.all(torch.isfinite(e_dot)):
        raise ValueError(
            "Non-finite Bicycle error dynamics. Training samples must keep "
            "longitudinal speed v_x nonzero."
        )
    V_dot = lyapunov.lie_derivative(r, e_dot)

    inside_level_set = (V <= level).to(dtype=V.dtype)
    decrease_violation = inside_level_set * F.relu(V_dot + decrease_margin * V)

    e_zero = torch.zeros(1, e.shape[-1], device=e.device, dtype=e.dtype)
    V_zero = lyapunov(e_zero).mean()

    loss = masked_mean(decrease_violation, sample_mask) + origin_weight * V_zero
    info = {
        "lyapunov_loss": loss.detach(),
        "V_mean": masked_mean(V.detach(), sample_mask),
        "V_dot_mean": masked_mean(V_dot.detach(), sample_mask),
        "decrease_violation_mean": masked_mean(decrease_violation.detach(), sample_mask),
        "inside_decrease_region_fraction": masked_mean(
            inside_level_set.detach(), sample_mask
        ),
        # Kept for checkpoint/log compatibility.
        "outside_level_fraction": masked_mean((1.0 - inside_level_set).detach(), sample_mask),
        "V_zero": V_zero.detach(),
    }

    return loss, info


def error_bound_loss(
    lyapunov,
    e: torch.Tensor,
    v: torch.Tensor,
    E: torch.Tensor,
    level: float = 1.0,
    logdet_weight: float = 1e-3,
    sample_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Tracking error bound loss:

        L_E = E[1_{V(r) < 1} max(0, r^T E r - 1)] - beta log det(E),

    where r is the deviation from the moving reference. This encourages the
    Lyapunov level set {V(r) <= 1} to lie inside the ellipsoid
    {r: r^T E r <= 1}. Maximizing log det(E) minimizes the volume
    of this ellipsoid, so its contribution to a minimized loss has a minus sign.
    """
    r = moving_reference_error(e, v)
    V = lyapunov(r)
    ellipsoid_value = quadratic_form(r, E)

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
    u_pred = controller(e, v)
    if u_expert.shape != u_pred.shape:
        raise ValueError(
            f"Expected u_expert shape {tuple(u_pred.shape)}, got {tuple(u_expert.shape)}"
        )
    control_scale = 0.5 * (controller.u_max - controller.u_min)
    normalized_error = (u_pred - u_expert) / control_scale.to(u_pred)
    squared_error = torch.sum(normalized_error.square(), dim=-1)
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

    info = {}
    zero = torch.zeros((), device=e.device, dtype=e.dtype)
    total = zero

    if weights.lyapunov != 0.0:
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
        total = total + weights.lyapunov * L_V
        info.update(info_V)
    else:
        info.update(
            {
                "lyapunov_loss": zero,
                "V_mean": zero,
                "V_dot_mean": zero,
                "decrease_violation_mean": zero,
                "inside_decrease_region_fraction": zero,
                "outside_level_fraction": zero,
                "V_zero": zero,
            }
        )

    if weights.error_bound != 0.0:
        L_E, info_E = error_bound_loss(
            lyapunov=lyapunov,
            e=e,
            v=v,
            E=E,
            level=level,
            logdet_weight=weights.logdet,
            sample_mask=use_error_bound_loss,
        )
        total = total + weights.error_bound * L_E
        info.update(info_E)
    else:
        info.update(
            {
                "error_bound_loss": zero,
                "ellipsoid_value_mean": zero,
                "bound_violation_mean": zero,
                "inside_level_fraction": zero,
                "logdet_E": zero,
            }
        )

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
