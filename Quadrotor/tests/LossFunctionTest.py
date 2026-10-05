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
    quadratic_form,
)
from lyapunov.LyapunovNetwork import NeuralLyapunovFunction
from models.error import ErrorDynamics


def assert_close(actual, expected, atol=1e-6):
    if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"\nactual:   {actual}\nexpected: {expected}")


def make_models():
    error_model = ErrorDynamics()
    tracking_model = error_model.tracking_model
    controller = NeuralTrackingController(
        hidden_sizes=(32, 32),
        g=tracking_model.g,
        kT=tracking_model.kT,
    )
    lyapunov = NeuralLyapunovFunction(
        error_dim=error_model.error_dim,
        hidden_sizes=(32, 32),
        feature_dim=8,
        delta=1e-3,
    )
    return error_model, controller, lyapunov


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
    e = torch.randn(batch_size, error_model.error_dim, requires_grad=True)
    z = torch.randn(batch_size, error_model.planning_model.state_dim)
    v = torch.randn(batch_size, error_model.planning_model.input_dim)
    E = torch.eye(error_model.error_dim, requires_grad=True)
    u_expert = torch.zeros(batch_size, error_model.tracking_model.input_dim)
    u_expert[:, 2] = error_model.tracking_model.g / error_model.tracking_model.kT

    L_V, info_V = lyapunov_decrease_loss(lyapunov, controller, error_model, e, z, v)
    L_E, info_E = error_bound_loss(lyapunov, e, E)
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
    E = torch.eye(lyapunov.error_dim, requires_grad=True)
    loss, _ = error_bound_loss(lyapunov, e, E, logdet_weight=1.0)
    loss.backward()

    expected_gradient = -torch.eye(lyapunov.error_dim)
    assert_close(E.grad, expected_gradient)
    print("log-det term prevents ellipsoid collapse: ok")


def test_utilization_term_tightens_ellipsoid_around_large_samples():
    _, _, lyapunov = make_models()

    e = torch.zeros(4, lyapunov.error_dim)
    e[:, 0] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    E = torch.eye(lyapunov.error_dim, requires_grad=True)
    loss, info = error_bound_loss(
        lyapunov,
        e,
        E,
        logdet_weight=0.0,
        utilization_weight=1.0,
        utilization_target=0.8,
        utilization_fraction=0.25,
    )
    loss.backward()

    if info["ellipsoid_utilization_loss"] <= 0.0:
        raise AssertionError("Expected positive utilization loss")
    if E.grad[0, 0] >= 0.0:
        raise AssertionError("Expected utilization term to increase E along sampled direction")
    print("ellipsoid utilization term tightens large-sample bound: ok")


def test_utilization_term_scales_large_trajectory_lyapunov_values():
    class ScaledLyapunov:
        def __init__(self):
            self.scale = torch.tensor(1.0, requires_grad=True)

        def __call__(self, e):
            return self.scale * torch.sum(e.square(), dim=-1)

        def lie_derivative(self, e, e_dot):
            return torch.zeros(e.shape[0], device=e.device, dtype=e.dtype)

    class ZeroController:
        def __call__(self, e, v):
            return torch.zeros(e.shape[0], 3, device=e.device, dtype=e.dtype)

    class ZeroErrorModel:
        def dynamics_from_error(self, e, z, v, u, d=None):
            return torch.zeros_like(e)

    lyapunov = ScaledLyapunov()
    e = torch.zeros(4, 10)
    e[:, 0] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    loss, info = lyapunov_decrease_loss(
        lyapunov,
        ZeroController(),
        ZeroErrorModel(),
        e,
        torch.zeros(4, 3),
        torch.zeros(4, 3),
        origin_weight=0.0,
        utilization_weight=1.0,
        utilization_mask=torch.ones(4, dtype=torch.bool),
    )
    loss.backward()

    if info["lyapunov_utilization_loss"] <= 0.0:
        raise AssertionError("Expected positive Lyapunov utilization loss")
    if lyapunov.scale.grad >= 0.0:
        raise AssertionError("Expected utilization term to increase large-sample V")
    print("Lyapunov utilization term scales large trajectory values: ok")


def test_alignment_term_raises_lyapunov_without_expanding_ellipsoid():
    class ScaledLyapunov:
        def __init__(self):
            self.scale = torch.tensor(0.1, requires_grad=True)

        def __call__(self, e):
            return self.scale * torch.sum(e.square(), dim=-1)

        def lie_derivative(self, e, e_dot):
            return torch.zeros(e.shape[0], device=e.device, dtype=e.dtype)

    class ZeroController:
        def __call__(self, e, v):
            return torch.zeros(e.shape[0], 3, device=e.device, dtype=e.dtype)

    class ZeroErrorModel:
        def dynamics_from_error(self, e, z, v, u, d=None):
            return torch.zeros_like(e)

    lyapunov = ScaledLyapunov()
    e = torch.zeros(2, 10)
    e[:, 0] = 1.0
    E = torch.eye(10, requires_grad=True)
    loss, info = certified_learning_loss(
        lyapunov=lyapunov,
        controller=ZeroController(),
        error_model=ZeroErrorModel(),
        e=e,
        z=torch.zeros(2, 3),
        v=torch.zeros(2, 3),
        E=E,
        weights=LossWeights(
            lyapunov=0.0,
            error_bound=0.0,
            behavior_cloning=0.0,
            origin=0.0,
            boundary=0.0,
            logdet=0.0,
            lyapunov_ellipsoid_alignment=1.0,
        ),
        use_utilization_loss=torch.ones(2, dtype=torch.bool),
    )
    loss.backward()

    if info["lyapunov_ellipsoid_alignment_loss"] <= 0.0:
        raise AssertionError("Expected positive Lyapunov-ellipsoid alignment loss")
    if lyapunov.scale.grad >= 0.0:
        raise AssertionError("Expected alignment term to increase V")
    if E.grad is not None and not torch.allclose(E.grad, torch.zeros_like(E.grad)):
        raise AssertionError("Expected alignment term not to expand or shrink E")
    print("Lyapunov-ellipsoid alignment raises V without changing E: ok")


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
            return torch.zeros(e.shape[0], 3, device=e.device, dtype=e.dtype)

    class ZeroErrorModel:
        def dynamics_from_error(self, e, z, v, u, d=None):
            return torch.zeros_like(e)

    e = torch.ones(1, 10)
    z = torch.zeros(1, 3)
    v = torch.zeros(1, 3)
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


def test_combined_loss_without_behavior_cloning():
    error_model, controller, lyapunov = make_models()

    batch_size = 8
    e = torch.randn(batch_size, error_model.error_dim, requires_grad=True)
    z = torch.randn(batch_size, error_model.planning_model.state_dim)
    v = torch.randn(batch_size, error_model.planning_model.input_dim)
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


def main():
    test_quadratic_form()
    test_individual_losses_are_finite_and_differentiable()
    test_logdet_term_prevents_ellipsoid_collapse()
    test_utilization_term_tightens_ellipsoid_around_large_samples()
    test_utilization_term_scales_large_trajectory_lyapunov_values()
    test_alignment_term_raises_lyapunov_without_expanding_ellipsoid()
    test_lyapunov_decrease_margin()
    test_combined_loss_without_behavior_cloning()
    print("all loss function tests passed")


if __name__ == "__main__":
    main()
