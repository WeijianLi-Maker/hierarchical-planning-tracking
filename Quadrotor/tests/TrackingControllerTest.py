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


def test_neural_controller_hover_initialization():
    error_model = ErrorDynamics()
    tracking_model = error_model.tracking_model

    controller = NeuralTrackingController(
        g=tracking_model.g,
        kT=tracking_model.kT,
    )

    e = torch.zeros(4, error_model.error_dim)
    v = torch.zeros(4, error_model.planning_model.input_dim)

    u = controller(e, v)
    expected = torch.tensor([0.0, 0.0, tracking_model.g / tracking_model.kT]).repeat(4, 1)

    assert_close(u, expected)
    print("neural tracking controller hover initialization: ok")


def test_neural_controller_bounds_and_gradients():
    error_model = ErrorDynamics()
    tracking_model = error_model.tracking_model

    controller = NeuralTrackingController(
        hidden_sizes=(32, 32),
        g=tracking_model.g,
        kT=tracking_model.kT,
    )

    e = torch.randn(8, error_model.error_dim, requires_grad=True)
    z = torch.randn(8, error_model.planning_model.state_dim)
    v = torch.randn(8, error_model.planning_model.input_dim)

    u = controller(e, v)

    if u.shape != (8, tracking_model.input_dim):
        raise AssertionError(f"Expected u shape (8, 3), got {u.shape}")
    if torch.any(u < controller.u_min) or torch.any(u > controller.u_max):
        raise AssertionError(f"Controller output violates bounds: {u}")

    e_dot = error_model.dynamics_from_error(e, z, v, u)
    loss = e_dot.square().mean()
    loss.backward()

    if e.grad is None or not torch.all(torch.isfinite(e.grad)):
        raise AssertionError("Expected finite gradient through closed-loop dynamics")

    print("neural tracking controller bounds and gradients: ok")


def main():
    test_neural_controller_hover_initialization()
    test_neural_controller_bounds_and_gradients()
    print("all tracking controller tests passed")


if __name__ == "__main__":
    main()
