import os
import sys

import torch
from torch import nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.TrackingControl import NeuralTrackingController
from models.error import ErrorDynamics


def assert_close(actual, expected, atol=1e-6):
    if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"\nactual:   {actual}\nexpected: {expected}")


def make_controller(error_model, **kwargs):
    return NeuralTrackingController(
        error_dim=error_model.error_dim,
        planning_input_dim=error_model.planning_model.input_dim,
        control_dim=error_model.tracking_model.input_dim,
        **kwargs,
    )


def test_neural_controller_structure_and_initialization():
    error_model = ErrorDynamics()
    controller = make_controller(error_model)

    linear_shapes = [
        (layer.in_features, layer.out_features)
        for layer in controller.network
        if isinstance(layer, nn.Linear)
    ]
    if linear_shapes != [(6, 32), (32, 32), (32, 2)]:
        raise AssertionError(f"Unexpected controller structure: {linear_shapes}")

    e = torch.randn(4, error_model.error_dim)
    v = torch.randn(4, error_model.planning_model.input_dim)
    assert_close(controller(e, v), torch.zeros(4, 2))
    print("neural tracking controller structure and initialization: ok")


def test_neural_controller_bounds_and_gradients():
    error_model = ErrorDynamics()
    controller = make_controller(
        error_model,
        u_min=[-0.5, -2.0],
        u_max=[0.5, 2.0],
        u_ref=[0.1, -0.2],
    )

    e = torch.randn(8, error_model.error_dim, requires_grad=True)
    z = torch.randn(8, error_model.planning_model.state_dim)
    v = torch.randn(8, error_model.planning_model.input_dim)

    assert_close(controller(e, v), torch.tensor([[0.1, -0.2]]).repeat(8, 1))

    with torch.no_grad():
        controller.network[-1].weight.normal_()
        controller.network[-1].bias.copy_(torch.tensor([10.0, -10.0]))
    u = controller(e, v)

    if u.shape != (8, error_model.tracking_model.input_dim):
        raise AssertionError(f"Expected u shape (8, 2), got {u.shape}")
    if torch.any(u < controller.u_min) or torch.any(u > controller.u_max):
        raise AssertionError(f"Controller output violates bounds: {u}")

    e_dot = error_model.dynamics_from_error(e, z, v, u)
    e_dot.square().mean().backward()
    if e.grad is None or not torch.all(torch.isfinite(e.grad)):
        raise AssertionError("Expected finite gradient through closed-loop dynamics")

    print("neural tracking controller bounds and gradients: ok")


def test_neural_controller_zero_error_equilibrium():
    error_model = ErrorDynamics()
    controller = make_controller(error_model)
    with torch.no_grad():
        for parameter in controller.parameters():
            parameter.normal_()

    e = torch.zeros(5, error_model.error_dim)
    v = torch.zeros(5, error_model.planning_model.input_dim)
    assert_close(controller(e, v), torch.zeros(5, error_model.tracking_model.input_dim))
    print("neural tracking controller zero-error equilibrium: ok")


def test_neural_controller_input_validation():
    controller = NeuralTrackingController()

    try:
        controller(torch.zeros(2, 4), torch.zeros(3, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected mismatched batch dimensions to be rejected")

    for kwargs in (
        {"hidden_sizes": (0,)},
        {"u_min": [-float("inf"), -1.0]},
        {"u_max": [1.0, float("nan")]},
    ):
        try:
            NeuralTrackingController(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid controller arguments to be rejected: {kwargs}")

    print("neural tracking controller input validation: ok")


def main():
    test_neural_controller_structure_and_initialization()
    test_neural_controller_bounds_and_gradients()
    test_neural_controller_zero_error_equilibrium()
    test_neural_controller_input_validation()
    print("all tracking controller tests passed")


if __name__ == "__main__":
    main()
