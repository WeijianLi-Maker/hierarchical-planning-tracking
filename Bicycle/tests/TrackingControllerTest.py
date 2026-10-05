# tests/tracking_controller_test.py

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.TrackingControl import NeuralTrackingController
from models.error import ErrorDynamics


def assert_close(actual, expected, atol=1e-6):
    if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"\nactual:   {actual}\nexpected: {expected}")


def test_neural_controller_zero_control_initialization():
    error_model = ErrorDynamics()
    controller = NeuralTrackingController()

    e = torch.zeros(4, error_model.error_dim)
    v = torch.zeros(4, error_model.planning_model.input_dim)

    u = controller(e, v)
    expected = torch.zeros(4, 2)

    assert_close(u, expected)
    print("neural tracking controller zero-control initialization: ok")


def test_neural_controller_default_structure_and_bounds():
    controller = NeuralTrackingController()
    linear_layers = [
        layer for layer in controller.network if isinstance(layer, torch.nn.Linear)
    ]
    expected_shapes = ((8, 64), (64, 64), (64, 2))
    actual_shapes = tuple(
        (layer.in_features, layer.out_features) for layer in linear_layers
    )

    if actual_shapes != expected_shapes:
        raise AssertionError(
            f"Expected default network shapes {expected_shapes}, got {actual_shapes}"
        )
    assert_close(controller.u_min, torch.tensor([-torch.pi / 18.0, -3.0]))
    assert_close(controller.u_max, torch.tensor([torch.pi / 18.0, 3.0]))
    assert_close(controller.u_ref, torch.zeros(2))

    print("neural tracking controller default structure and bounds: ok")


def test_neural_controller_bounded_output_mapping():
    controller = NeuralTrackingController(
        hidden_sizes=(8,),
        u_min=[-0.2, -2.0],
        u_max=[0.1, 4.0],
        u_ref=[-0.05, 1.0],
    )
    final_layer = controller.network[-1]
    with torch.no_grad():
        final_layer.bias.copy_(torch.tensor([100.0, -100.0]))

    e = torch.zeros(3, 6)
    v = torch.zeros(3, 2)
    u = controller(e, v)

    tolerance = 1e-6
    if torch.any(u < controller.u_min - tolerance) or torch.any(
        u > controller.u_max + tolerance
    ):
        raise AssertionError(f"Bounded output mapping violated constraints: {u}")
    assert_close(u[:, 0], controller.u_max[0].expand(3), atol=1e-5)
    assert_close(u[:, 1], controller.u_min[1].expand(3), atol=1e-5)

    print("neural tracking controller bounded output mapping: ok")


def test_neural_controller_rejects_invalid_inputs_and_controls():
    controller = NeuralTrackingController()

    invalid_inputs = (
        (torch.zeros(2, 10), torch.zeros(2, 2)),
        (torch.zeros(2, 6), torch.zeros(2, 3)),
    )
    for e, v in invalid_inputs:
        try:
            controller(e, v)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Expected invalid controller inputs to be rejected: {e.shape}, {v.shape}"
            )

    invalid_control_configs = (
        {"u_min": [-1.0, -1.0, -1.0]},
        {"u_min": [0.0, 0.0], "u_max": [0.0, 1.0]},
    )
    for kwargs in invalid_control_configs:
        try:
            NeuralTrackingController(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid control configuration rejection: {kwargs}")

    print("neural tracking controller input and control validation: ok")


def test_neural_controller_bounds_and_gradients():
    error_model = ErrorDynamics()
    tracking_model = error_model.tracking_model

    controller = NeuralTrackingController(hidden_sizes=(32, 32))

    speed_hat = 5.0 + 5.0 * torch.rand(8)
    omega_hat = 0.2 * torch.randn(8)
    v = torch.stack([speed_hat, omega_hat], dim=-1)
    e = 0.1 * torch.randn(8, error_model.error_dim)
    e[:, 3] = speed_hat + 0.5 * torch.randn(8)
    e[:, 5] = omega_hat + 0.05 * torch.randn(8)
    e.requires_grad_(True)
    z = torch.randn(8, error_model.planning_model.state_dim)

    u = controller(e, v)

    if u.shape != (8, tracking_model.input_dim):
        raise AssertionError(f"Expected u shape (8, 2), got {u.shape}")
    if torch.any(u < controller.u_min) or torch.any(u > controller.u_max):
        raise AssertionError(f"Controller output violates bounds: {u}")

    e_dot = error_model.dynamics_from_error(e, z, v, u)
    loss = e_dot.square().mean()
    loss.backward()

    if e.grad is None or not torch.all(torch.isfinite(e.grad)):
        raise AssertionError("Expected finite gradient through closed-loop dynamics")
    parameter_gradients = [
        parameter.grad
        for parameter in controller.parameters()
        if parameter.grad is not None
    ]
    if not parameter_gradients or not all(
        torch.all(torch.isfinite(gradient)) for gradient in parameter_gradients
    ):
        raise AssertionError("Expected finite controller parameter gradients")

    print("neural tracking controller bounds and gradients: ok")


def main():
    test_neural_controller_zero_control_initialization()
    test_neural_controller_default_structure_and_bounds()
    test_neural_controller_bounded_output_mapping()
    test_neural_controller_rejects_invalid_inputs_and_controls()
    test_neural_controller_bounds_and_gradients()
    print("all tracking controller tests passed")


if __name__ == "__main__":
    main()
