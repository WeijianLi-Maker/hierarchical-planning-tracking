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
    controller = NeuralTrackingController(
        error_dim=error_model.error_dim,
        planning_input_dim=error_model.planning_model.input_dim,
        control_dim=error_model.tracking_model.input_dim,
        hidden_sizes=(32, 32),
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


def test_lyapunov_decrease_margin():
    class FixedLyapunov:
        error_dim = 4

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
        error_dim = 4
        planning_input_dim = 2
        output_dim = 2

        def __call__(self, e, v):
            return torch.zeros(e.shape[0], 2, device=e.device, dtype=e.dtype)

    class ZeroErrorModel:
        error_dim = 4

        class PlanningModel:
            state_dim = 2
            input_dim = 2

        class TrackingModel:
            input_dim = 2

        planning_model = PlanningModel()
        tracking_model = TrackingModel()

        def dynamics_from_error(self, e, z, v, u):
            return torch.zeros_like(e)

    e = torch.ones(1, 4)
    z = torch.zeros(1, 2)
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


def test_current_model_loss_dimensions():
    error_model, controller, lyapunov = make_models()
    batch_size = 5
    e = torch.randn(batch_size, 4)
    z = torch.randn(batch_size, 2)
    v = torch.randn(batch_size, 2)
    u_expert = torch.randn(batch_size, 2)
    E = torch.eye(4)

    total, info = certified_learning_loss(
        lyapunov=lyapunov,
        controller=controller,
        error_model=error_model,
        e=e,
        z=z,
        v=v,
        E=E,
        u_expert=u_expert,
    )
    if total.ndim != 0:
        raise AssertionError(f"Expected scalar total loss, got shape {tuple(total.shape)}")
    for key in ("lyapunov_loss", "error_bound_loss", "behavior_cloning_loss", "total_loss"):
        if key not in info or info[key].ndim != 0:
            raise AssertionError(f"Expected scalar info['{key}'], got {info.get(key)}")

    print("current model loss dimensions: ok")


def test_loss_input_validation():
    error_model, controller, lyapunov = make_models()
    e = torch.zeros(4, 4)
    z = torch.zeros(4, 2)
    v = torch.zeros(4, 2)
    u_expert = torch.zeros(4, 2)
    E = torch.eye(4)

    invalid_calls = (
        lambda: lyapunov_decrease_loss(lyapunov, controller, error_model, e[:, :3], z, v),
        lambda: lyapunov_decrease_loss(lyapunov, controller, error_model, e, z[:, :1], v),
        lambda: lyapunov_decrease_loss(lyapunov, controller, error_model, e, z, v[:, :1]),
        lambda: behavior_cloning_loss(controller, e, v, u_expert[:, :1]),
        lambda: error_bound_loss(lyapunov, e, torch.eye(3)),
        lambda: behavior_cloning_loss(
            controller,
            e,
            v,
            u_expert,
            sample_mask=torch.ones(4, 1),
        ),
    )
    for call in invalid_calls:
        try:
            call()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("Expected invalid loss input to be rejected")

    try:
        certified_learning_loss(
            lyapunov,
            controller,
            error_model,
            e,
            z,
            v,
            E,
            weights=LossWeights(lyapunov=-1.0),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected negative loss weight to be rejected")

    print("loss input validation: ok")


def main():
    test_quadratic_form()
    test_individual_losses_are_finite_and_differentiable()
    test_logdet_term_prevents_ellipsoid_collapse()
    test_lyapunov_decrease_margin()
    test_combined_loss_without_behavior_cloning()
    test_current_model_loss_dimensions()
    test_loss_input_validation()
    print("all loss function tests passed")


if __name__ == "__main__":
    main()
