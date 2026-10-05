import os
import sys

import torch
from torch import nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.TrackingControl import NeuralTrackingController
from lyapunov.LyapunovNetwork import NeuralLyapunovFunction
from models.error import ErrorDynamics


def assert_close(actual, expected, atol=1e-6):
    if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"\nactual:   {actual}\nexpected: {expected}")


def test_default_network_structure():
    error_model = ErrorDynamics()
    lyapunov = NeuralLyapunovFunction()

    if lyapunov.error_dim != error_model.error_dim:
        raise AssertionError(
            f"Expected Lyapunov error dimension {error_model.error_dim}, "
            f"got {lyapunov.error_dim}"
        )

    linear_shapes = [
        (layer.in_features, layer.out_features)
        for layer in lyapunov.C
        if isinstance(layer, nn.Linear)
    ]
    if linear_shapes != [(4, 32), (32, 32), (32, 8)]:
        raise AssertionError(f"Unexpected Lyapunov network structure: {linear_shapes}")

    features = lyapunov.features(torch.zeros(3, error_model.error_dim))
    if features.shape != (3, 8):
        raise AssertionError(f"Expected feature shape (3, 8), got {features.shape}")

    print("lyapunov default 4-32-32-8 structure: ok")


def test_quadratic_initialization():
    error_model = ErrorDynamics()
    delta = 1e-3
    lyapunov = NeuralLyapunovFunction(error_dim=error_model.error_dim, delta=delta)

    e = torch.randn(5, error_model.error_dim)
    expected = delta * torch.sum(e.square(), dim=-1)
    assert_close(lyapunov(e), expected)
    print("lyapunov quadratic initialization: ok")


def test_positive_definite_values():
    error_model = ErrorDynamics()
    lyapunov = NeuralLyapunovFunction(error_dim=error_model.error_dim)

    e_zero = torch.zeros(1, error_model.error_dim)
    e_nonzero = torch.tensor([[0.5, -0.2, 0.1, 0.3]])

    assert_close(lyapunov(e_zero), torch.zeros(1))
    if not torch.all(lyapunov(e_nonzero) > 0.0):
        raise AssertionError("Expected V(e)>0 for every nonzero error")

    print("lyapunov positive definite values: ok")


def test_gradient_and_lie_derivative():
    error_model = ErrorDynamics()
    delta = 2e-3
    lyapunov = NeuralLyapunovFunction(error_dim=error_model.error_dim, delta=delta)

    e = torch.randn(7, error_model.error_dim, requires_grad=True)
    e_dot = torch.randn(7, error_model.error_dim)

    grad = lyapunov.gradient(e)
    expected_grad = 2.0 * delta * e
    assert_close(grad, expected_grad)

    V_dot = lyapunov.lie_derivative(e, e_dot)
    assert_close(V_dot, torch.sum(expected_grad * e_dot, dim=-1))

    V_dot.square().mean().backward()
    if e.grad is None or not torch.all(torch.isfinite(e.grad)):
        raise AssertionError("Expected finite gradient through Lyapunov derivative")

    print("lyapunov gradient and lie derivative: ok")


def test_closed_loop_error_dynamics_gradient():
    error_model = ErrorDynamics()
    lyapunov = NeuralLyapunovFunction()
    controller = NeuralTrackingController()

    e = torch.randn(8, 4, requires_grad=True)
    z = torch.randn(8, 2)
    v = torch.randn(8, 2)
    u = controller(e, v)
    e_dot = error_model.dynamics_from_error(e, z, v, u)
    V_dot = lyapunov.lie_derivative(e, e_dot)

    if V_dot.shape != (8,) or not torch.all(torch.isfinite(V_dot)):
        raise AssertionError(f"Unexpected closed-loop V_dot: {V_dot}")

    V_dot.square().mean().backward()
    if e.grad is None or not torch.all(torch.isfinite(e.grad)):
        raise AssertionError("Expected finite closed-loop Lyapunov gradient")

    print("lyapunov closed-loop Error Dynamics gradient: ok")


def test_input_validation():
    lyapunov = NeuralLyapunovFunction()

    for invalid_error in (torch.zeros(2, 3), torch.zeros(2, 5)):
        try:
            lyapunov(invalid_error)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected invalid error dimension to be rejected")

    for kwargs in (
        {"error_dim": 0},
        {"hidden_sizes": (0,)},
        {"feature_dim": 0},
        {"delta": 0.0},
    ):
        try:
            NeuralLyapunovFunction(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid Lyapunov arguments to be rejected: {kwargs}")

    print("lyapunov input validation: ok")


def main():
    test_default_network_structure()
    test_quadratic_initialization()
    test_positive_definite_values()
    test_gradient_and_lie_derivative()
    test_closed_loop_error_dynamics_gradient()
    test_input_validation()
    print("all lyapunov network tests passed")


if __name__ == "__main__":
    main()
