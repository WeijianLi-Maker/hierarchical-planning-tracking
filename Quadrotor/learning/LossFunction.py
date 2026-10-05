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
    boundary: float = 1.0
    interior: float = 0.0
    logdet: float = 1.0
    error_bound_level: float = 1.0
    lyapunov_utilization: float = 0.0
    ellipsoid_utilization: float = 0.0
    lyapunov_ellipsoid_alignment: float = 0.0
    lyapunov_ellipsoid_alignment_margin: float = 0.0
    horizontal_position_bound: float = 0.0


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


def projected_axis_bound_squared(
    E: torch.Tensor,
    indices: tuple[int, ...] = (0, 4),
) -> torch.Tensor:
    """
    Sum squared projected bounds for selected error coordinates.

    ErrorBound.py computes projected bounds from

        S = P E^{-1} P^T,  bound_i = sqrt(S_ii).

    Minimizing trace(S) directly reduces the squared projected bounds for the
    selected coordinates.  By default this targets horizontal position errors
    e_x and e_y.
    """
    if E.ndim != 2 or E.shape[0] != E.shape[1]:
        raise ValueError(f"Expected E shape (n, n), got {tuple(E.shape)}")
    if any(index < 0 or index >= E.shape[0] for index in indices):
        raise ValueError(f"Projected bound indices {indices} are invalid for E shape {tuple(E.shape)}")

    selector = torch.zeros(len(indices), E.shape[0], dtype=E.dtype, device=E.device)
    for row, index in enumerate(indices):
        selector[row, index] = 1.0
    projected_shape = selector @ torch.linalg.solve(E, selector.T)
    return torch.trace(projected_shape)


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
    boundary_weight: float = 1.0,
    interior_weight: float = 0.0,
    utilization_weight: float = 0.0,
    utilization_target: float = 0.9,
    utilization_fraction: float = 0.1,
    decrease_margin: float = 0.02,
    d: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
    boundary_mask: torch.Tensor | None = None,
    interior_mask: torch.Tensor | None = None,
    utilization_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Lyapunov decrease loss:

        L_V = E[1_{V(e) >= 1} max(0, V_dot + epsilon)]
              + E_boundary[max(0, 1 - V(e))] + alpha V(0).

    This enforces V_dot + epsilon <= 0 outside the certified level set.
    The boundary term prevents the level set from escaping the verified domain.
    """
    V = lyapunov(e)
    u = controller(e, v)
    # Keep the tracking controller anchored by behavior cloning.  Letting the
    # certificate loss update the controller directly can improve V_dot while
    # destroying rollout tracking performance.
    e_dot = error_model.dynamics_from_error(e, z, v, u.detach(), d)
    V_dot = lyapunov.lie_derivative(e, e_dot)

    outside_level_set = (V >= level).to(dtype=V.dtype)
    decrease_violation = outside_level_set * F.relu(V_dot + decrease_margin)
    boundary_violation = F.relu(level - V)
    boundary_loss = (
        torch.zeros((), device=e.device, dtype=e.dtype)
        if boundary_mask is None
        else masked_mean(boundary_violation, boundary_mask)
    )
    interior_violation = F.relu(V - level)
    interior_loss = (
        torch.zeros((), device=e.device, dtype=e.dtype)
        if interior_mask is None
        else masked_mean(interior_violation, interior_mask)
    )

    e_zero = torch.zeros(1, e.shape[-1], device=e.device, dtype=e.dtype)
    V_zero = lyapunov(e_zero).mean()

    utilization_loss = torch.zeros((), device=e.device, dtype=e.dtype)
    if utilization_mask is not None:
        eligible_indices = torch.where(utilization_mask > 0)[0]
        if utilization_weight > 0.0 and eligible_indices.numel() > 0:
            count = max(1, int(round(utilization_fraction * eligible_indices.numel())))
            eligible_values = V[eligible_indices]
            top_values = torch.topk(eligible_values.detach(), count).indices
            selected_values = eligible_values[top_values]
            utilization_loss = F.relu(utilization_target - selected_values).square().mean()

    loss = (
        masked_mean(decrease_violation, sample_mask)
        + boundary_weight * boundary_loss
        + interior_weight * interior_loss
        + origin_weight * V_zero
        + utilization_weight * utilization_loss
    )
    info = {
        "lyapunov_loss": loss.detach(),
        "V_mean": masked_mean(V.detach(), sample_mask),
        "V_dot_mean": masked_mean(V_dot.detach(), sample_mask),
        "decrease_violation_mean": masked_mean(decrease_violation.detach(), sample_mask),
        "boundary_violation_mean": boundary_loss.detach(),
        "interior_violation_mean": interior_loss.detach(),
        "outside_level_fraction": masked_mean(outside_level_set.detach(), sample_mask),
        "V_zero": V_zero.detach(),
        "lyapunov_utilization_loss": utilization_loss.detach(),
    }

    return loss, info


def error_bound_loss(
    lyapunov,
    e: torch.Tensor,
    E: torch.Tensor,
    level: float = 1.0,
    bound_level: float = 1.0,
    logdet_weight: float = 1e-3,
    utilization_weight: float = 0.0,
    utilization_target: float = 0.9,
    utilization_fraction: float = 0.1,
    sample_mask: torch.Tensor | None = None,
    utilization_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Tracking error bound loss:

        L_E = E[1_{V(e) < 1} max(0, e^T E e - rho)] - beta log det(E).

    This encourages the Lyapunov level set {V(e) <= 1} to lie inside the
    ellipsoid {e: e^T E e <= rho}.  Using rho < 1 leaves a safety margin for
    rollout points that are near but not identical to training samples.
    Maximizing log det(E) minimizes the volume of this ellipsoid, so its
    contribution to a minimized loss has a minus sign.
    """
    if bound_level <= 0.0:
        raise ValueError("bound_level must be positive")

    V = lyapunov(e)
    ellipsoid_value = quadratic_form(e, E)

    inside_level_set = (V < level).to(dtype=V.dtype)
    bound_violation = F.relu(ellipsoid_value - bound_level)
    logdet = logdet_spd(E)

    inside_mask = inside_level_set
    if sample_mask is not None:
        inside_mask = inside_mask * sample_mask.to(
            device=inside_mask.device,
            dtype=inside_mask.dtype,
        )
    utilization_loss = torch.zeros((), device=e.device, dtype=e.dtype)
    eligible = inside_mask
    if utilization_mask is not None:
        eligible = eligible * utilization_mask.to(
            device=eligible.device,
            dtype=eligible.dtype,
        )
    eligible_indices = torch.where(eligible > 0.0)[0]
    if utilization_weight > 0.0 and eligible_indices.numel() > 0:
        count = max(1, int(round(utilization_fraction * eligible_indices.numel())))
        eligible_values = ellipsoid_value[eligible_indices]
        top_values = torch.topk(eligible_values.detach(), count).indices
        selected_values = eligible_values[top_values]
        utilization_loss = F.relu(utilization_target - selected_values).square().mean()

    loss = (
        masked_mean(bound_violation, inside_mask)
        - logdet_weight * logdet
        + utilization_weight * utilization_loss
    )
    info = {
        "error_bound_loss": loss.detach(),
        "ellipsoid_value_mean": masked_mean(ellipsoid_value.detach(), sample_mask),
        "bound_violation_mean": masked_mean(bound_violation.detach(), inside_mask),
        "inside_level_fraction": masked_mean(inside_level_set.detach(), sample_mask),
        "logdet_E": logdet.detach(),
        "ellipsoid_utilization_loss": utilization_loss.detach(),
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

        L_BC = E[||(u - u_expert) / (u_max - u_min)||^2],

    where u = kappa(e, v). Normalizing by each control dimension's range
    prevents the vertical thrust scale from dominating the angle commands.
    """
    u_pred = controller(e, v)
    control_scale = controller.u_max - controller.u_min
    normalized_error = (u_pred - u_expert) / control_scale
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
    decrease_margin: float = 0.02,
    d: torch.Tensor | None = None,
    use_lyapunov_loss: torch.Tensor | None = None,
    use_error_bound_loss: torch.Tensor | None = None,
    use_behavior_cloning_loss: torch.Tensor | None = None,
    use_boundary_loss: torch.Tensor | None = None,
    use_interior_loss: torch.Tensor | None = None,
    use_utilization_loss: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Combined offline learning loss:

        L = w_V L_V + w_E L_E + w_BC L_BC.

    The behavior cloning term is included only when u_expert is provided.
    """
    if weights is None:
        weights = LossWeights()

    L_V, info_V = lyapunov_decrease_loss(
        lyapunov=lyapunov,
        controller=controller,
        error_model=error_model,
        e=e,
        z=z,
        v=v,
        d=d,
        level=level,
        origin_weight=weights.origin,
        boundary_weight=weights.boundary,
        interior_weight=weights.interior,
        utilization_weight=weights.lyapunov_utilization,
        decrease_margin=decrease_margin,
        sample_mask=use_lyapunov_loss,
        boundary_mask=use_boundary_loss,
        interior_mask=use_interior_loss,
        utilization_mask=use_utilization_loss,
    )
    L_E, info_E = error_bound_loss(
        lyapunov=lyapunov,
        e=e,
        E=E,
        level=level,
        bound_level=weights.error_bound_level,
        logdet_weight=weights.logdet,
        utilization_weight=weights.ellipsoid_utilization,
        sample_mask=use_error_bound_loss,
        utilization_mask=use_utilization_loss,
    )

    alignment_loss = torch.zeros((), device=e.device, dtype=e.dtype)
    if weights.lyapunov_ellipsoid_alignment > 0.0:
        if use_error_bound_loss is not None:
            alignment_mask = use_error_bound_loss.to(device=e.device, dtype=e.dtype)
        elif use_utilization_loss is not None:
            alignment_mask = use_utilization_loss.to(device=e.device, dtype=e.dtype)
        else:
            alignment_mask = None
        V = lyapunov(e)
        ellipsoid_value = quadratic_form(e, E)
        # Encourage the learned Lyapunov value to stay below the quadratic
        # ellipsoid value on samples used to shape the certified error bound.
        alignment_loss = masked_mean(
            F.relu(
                V
                - ellipsoid_value
                + weights.lyapunov_ellipsoid_alignment_margin
            ).square(),
            alignment_mask,
        )

    horizontal_bound_loss = torch.zeros((), device=e.device, dtype=e.dtype)
    if weights.horizontal_position_bound > 0.0:
        horizontal_bound_loss = projected_axis_bound_squared(E, indices=(0, 4))

    total = (
        weights.lyapunov * L_V
        + weights.error_bound * L_E
        + weights.lyapunov_ellipsoid_alignment * alignment_loss
        + weights.horizontal_position_bound * horizontal_bound_loss
    )
    info = {}
    info.update(info_V)
    info.update(info_E)
    info["lyapunov_ellipsoid_alignment_loss"] = alignment_loss.detach()
    info["horizontal_position_bound_loss"] = horizontal_bound_loss.detach()

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
