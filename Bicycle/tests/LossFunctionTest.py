# tests/LossFunctionTest.py

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.TrackingControl import NeuralTrackingController
from learning.LossFunction import (
    LossWeights,
    behavior_cloning_loss,
    certified_learning_loss,
    error_bound_loss,
    lyapunov_decrease_loss,
    moving_reference_error,
    quadratic_form,
)
from lyapunov.LyapunovNetwork import NeuralLyapunovFunction
from models.error import ErrorDynamics


def assert_close(actual, expected, atol=1e-6):
    if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"\nactual:   {actual}\nexpected: {expected}")


def make_models():
    error_model = ErrorDynamics()
    controller = NeuralTrackingController(
        hidden_sizes=(32, 32),
    )
    lyapunov = NeuralLyapunovFunction(
        error_dim=error_model.error_dim,
        hidden_sizes=(32, 32),
        feature_dim=8,
        delta=1e-3,
    )
    return error_model, controller, lyapunov


def make_bicycle_batch(error_model, batch_size, requires_grad=False):
    z = torch.randn(batch_size, error_model.planning_model.state_dim)
    speed_hat = 5.0 + 5.0 * torch.rand(batch_size)
    omega_hat = 0.2 * torch.randn(batch_size)
    v = torch.stack([speed_hat, omega_hat], dim=-1)

    e = 0.1 * torch.randn(batch_size, error_model.error_dim)
    e[:, 3] = speed_hat + 0.5 * torch.randn(batch_size)
    e[:, 5] = omega_hat + 0.05 * torch.randn(batch_size)
    e.requires_grad_(requires_grad)
    return e, z, v


def test_quadratic_form():
    e = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    E = torch.diag(torch.tensor([2.0, 0.5]))

    value = quadratic_form(e, E)
    expected = torch.tensor([4.0, 26.0])

    assert_close(value, expected)
    print("quadratic form: ok")


def test_individual_losses_are_finite_and_differentiable():
    error_model, controller, lyapunov = make_models()

    batch_size = 16
    e, z, v = make_bicycle_batch(error_model, batch_size, requires_grad=True)
    E = torch.eye(error_model.error_dim, requires_grad=True)
    u_expert = torch.zeros(batch_size, error_model.tracking_model.input_dim)

    L_V, info_V = lyapunov_decrease_loss(lyapunov, controller, error_model, e, z, v)
    L_E, info_E = error_bound_loss(lyapunov, e, v, E)
    L_BC, info_BC = behavior_cloning_loss(controller, e, v, u_expert)

    total = L_V + L_E + L_BC
    total.backward()

    if not torch.isfinite(total):
        raise AssertionError(f"Expected finite total loss, got {total}")
    if e.grad is None or not torch.all(torch.isfinite(e.grad)):
        raise AssertionError("Expected finite gradients through learning losses")
    if E.grad is None or not torch.all(torch.isfinite(E.grad)):
        raise AssertionError("Expected finite gradient for E")

    for info in (info_V, info_E, info_BC):
        for key, value in info.items():
            if not torch.isfinite(value):
                raise AssertionError(f"Expected finite info[{key}], got {value}")

    print("individual losses finite and differentiable: ok")


def test_logdet_term_prevents_ellipsoid_collapse():
    _, _, lyapunov = make_models()

    e = torch.zeros(1, lyapunov.error_dim)
    v = torch.zeros(1, 2)
    E = torch.eye(lyapunov.error_dim, requires_grad=True)
    loss, _ = error_bound_loss(lyapunov, e, v, E, logdet_weight=1.0)
    loss.backward()

    expected_gradient = -torch.eye(lyapunov.error_dim)
    assert_close(E.grad, expected_gradient)
    print("log-det term prevents ellipsoid collapse: ok")


def test_moving_reference_error_centers_certificate():
    v = torch.tensor([[8.0, 0.2]])
    e = torch.tensor([[0.0, 0.0, 0.0, 8.0, 0.0, 0.2]])
    assert_close(moving_reference_error(e, v), torch.zeros_like(e))
    print("moving reference centers certificate: ok")


def test_lyapunov_decrease_margin():
    class FixedLyapunov:
        def __call__(self, e):
            return torch.ones(e.shape[0], device=e.device, dtype=e.dtype)

        def lie_derivative(self, e, e_dot):
            return torch.full(
                (e.shape[0],),
                -0.005,
                device=e.device,
                dtype=e.dtype,
            )

    class ZeroController:
        def __call__(self, e, v):
            return torch.zeros(e.shape[0], 2, device=e.device, dtype=e.dtype)

    class ZeroErrorModel:
        def dynamics_from_error(self, e, z, v, u):
            return torch.zeros_like(e)

    e = torch.ones(1, 6)
    z = torch.zeros(1, 3)
    v = torch.zeros(1, 2)
    loss, info = lyapunov_decrease_loss(
        FixedLyapunov(),
        ZeroController(),
        ZeroErrorModel(),
        e,
        z,
        v,
        origin_weight=0.0,
        decrease_margin=0.01,
    )

    assert_close(loss, torch.tensor(0.005))
    assert_close(info["decrease_violation_mean"], torch.tensor(0.005))
    print("lyapunov decrease margin: ok")


def test_lyapunov_decrease_ignores_outside_candidate_level_set():
    class FixedLyapunov:
        def __call__(self, e):
            return torch.full((e.shape[0],), 2.0, device=e.device, dtype=e.dtype)

        def lie_derivative(self, e, e_dot):
            return torch.ones(e.shape[0], device=e.device, dtype=e.dtype)

    class ZeroController:
        def __call__(self, e, v):
            return torch.zeros(e.shape[0], 2, device=e.device, dtype=e.dtype)

    class ZeroErrorModel:
        def dynamics_from_error(self, e, z, v, u):
            return torch.zeros_like(e)

    loss, _ = lyapunov_decrease_loss(
        FixedLyapunov(),
        ZeroController(),
        ZeroErrorModel(),
        torch.ones(1, 6),
        torch.zeros(1, 3),
        torch.zeros(1, 2),
        origin_weight=0.0,
    )
    assert_close(loss, torch.tensor(0.0))
    print("lyapunov decrease is local to candidate level set: ok")


def test_behavior_cloning_rejects_wrong_control_dimension():
    error_model, controller, _ = make_models()
    e, _, v = make_bicycle_batch(error_model, batch_size=4)
    wrong_u_expert = torch.zeros(4, 3)

    try:
        behavior_cloning_loss(controller, e, v, wrong_u_expert)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected three-dimensional expert controls to be rejected")

    print("behavior cloning rejects wrong control dimension: ok")


def test_lyapunov_loss_rejects_bicycle_singularity():
    error_model, controller, lyapunov = make_models()
    e, z, v = make_bicycle_batch(error_model, batch_size=4)
    e[:, 3] = 0.0

    try:
        lyapunov_decrease_loss(lyapunov, controller, error_model, e, z, v)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected samples at v_x=0 to be rejected")

    print("lyapunov loss rejects bicycle singularity: ok")


def test_combined_loss_without_behavior_cloning():
    error_model, controller, lyapunov = make_models()

    batch_size = 8
    e, z, v = make_bicycle_batch(error_model, batch_size, requires_grad=True)
    E = torch.eye(error_model.error_dim, requires_grad=True)

    loss, info = certified_learning_loss(
        lyapunov=lyapunov,
        controller=controller,
        error_model=error_model,
        e=e,
        z=z,
        v=v,
        E=E,
        weights=LossWeights(behavior_cloning=0.0),
    )
    loss.backward()

    if not torch.isfinite(loss):
        raise AssertionError(f"Expected finite combined loss, got {loss}")
    if "total_loss" not in info or "behavior_cloning_loss" not in info:
        raise AssertionError(f"Combined loss info missing expected keys: {info.keys()}")

    print("combined loss without behavior cloning: ok")


def test_behavior_cloning_only_skips_certificate_computation():
    error_model, controller, lyapunov = make_models()
    e = torch.zeros(4, 6)
    z = torch.zeros(4, 3)
    v = torch.zeros(4, 2)
    u_expert = torch.zeros(4, 2)
    E = torch.eye(6, requires_grad=True)

    loss, info = certified_learning_loss(
        lyapunov=lyapunov,
        controller=controller,
        error_model=error_model,
        e=e,
        z=z,
        v=v,
        E=E,
        u_expert=u_expert,
        weights=LossWeights(
            lyapunov=0.0,
            error_bound=0.0,
            behavior_cloning=1.0,
            origin=0.0,
            logdet=0.0,
        ),
    )
    loss.backward()

    assert_close(info["lyapunov_loss"], torch.tensor(0.0))
    assert_close(info["error_bound_loss"], torch.tensor(0.0))
    if E.grad is not None:
        raise AssertionError("Pure behavior cloning should not compute ellipsoid gradients")
    print("behavior cloning only skips certificate computation: ok")


def main():
    test_quadratic_form()
    test_individual_losses_are_finite_and_differentiable()
    test_logdet_term_prevents_ellipsoid_collapse()
    test_moving_reference_error_centers_certificate()
    test_lyapunov_decrease_margin()
    test_lyapunov_decrease_ignores_outside_candidate_level_set()
    test_behavior_cloning_rejects_wrong_control_dimension()
    test_lyapunov_loss_rejects_bicycle_singularity()
    test_combined_loss_without_behavior_cloning()
    test_behavior_cloning_only_skips_certificate_computation()
    print("all loss function tests passed")


if __name__ == "__main__":
    main()
