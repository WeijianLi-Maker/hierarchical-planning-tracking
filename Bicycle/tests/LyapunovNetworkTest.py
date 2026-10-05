# tests/LyapunovNetworkTest.py

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyapunov.LyapunovNetwork import NeuralLyapunovFunction
from models.error import ErrorDynamics


def assert_close(actual, expected, atol=1e-6):
    if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"\nactual:   {actual}\nexpected: {expected}")


def test_quadratic_initialization():
    delta = 1e-3
    lyapunov = NeuralLyapunovFunction(
        error_dim=6,
        hidden_sizes=(32, 32),
        feature_dim=8,
        delta=delta,
    )

    e = torch.randn(5, 6)
    V = lyapunov(e)
    expected = delta * torch.sum(e.square(), dim=-1)

    assert_close(V, expected)
    print("lyapunov quadratic initialization: ok")


def test_positive_definite_values():
    lyapunov = NeuralLyapunovFunction()

    e_zero = torch.zeros(1, 6)
    e_nonzero = torch.zeros(1, 6)
    e_nonzero[0, 0] = 0.5

    V_zero = lyapunov(e_zero)
    V_nonzero = lyapunov(e_nonzero)

    assert_close(V_zero, torch.zeros(1))
    if not torch.all(V_nonzero > 0.0):
        raise AssertionError(f"Expected V(e)>0 for nonzero e, got {V_nonzero}")

    print("lyapunov positive definite values: ok")


def test_gradient_and_lie_derivative():
    delta = 2e-3
    lyapunov = NeuralLyapunovFunction(
        error_dim=6,
        hidden_sizes=(16,),
        feature_dim=4,
        delta=delta,
    )

    e = torch.randn(7, 6, requires_grad=True)
    e_dot = torch.randn(7, 6)

    grad = lyapunov.gradient(e)
    expected_grad = 2.0 * delta * e
    assert_close(grad, expected_grad)

    V_dot = lyapunov.lie_derivative(e, e_dot)
    expected_V_dot = torch.sum(expected_grad * e_dot, dim=-1)
    assert_close(V_dot, expected_V_dot)

    loss = V_dot.square().mean()
    loss.backward()

    if e.grad is None or not torch.all(torch.isfinite(e.grad)):
        raise AssertionError("Expected finite gradient through Lyapunov derivative")

    print("lyapunov gradient and lie derivative: ok")


def test_bicycle_error_dynamics_integration():
    error_model = ErrorDynamics()
    lyapunov = NeuralLyapunovFunction()

    e = torch.tensor(
        [[0.2, -0.1, 0.03, 10.0, 0.1, 0.02]],
        requires_grad=True,
    )
    z = torch.tensor([[0.0, 0.0, 0.1]])
    v = torch.tensor([[10.0, 0.05]])
    u = torch.tensor([[0.01, 0.2]])

    e_dot = error_model.dynamics_from_error(e, z, v, u)
    V_dot = lyapunov.lie_derivative(e, e_dot)

    if e_dot.shape != e.shape:
        raise AssertionError(f"Expected e_dot shape {e.shape}, got {e_dot.shape}")
    if V_dot.shape != (1,):
        raise AssertionError(f"Expected V_dot shape (1,), got {V_dot.shape}")
    if not torch.all(torch.isfinite(V_dot)):
        raise AssertionError(f"Expected finite V_dot, got {V_dot}")

    V_dot.sum().backward()
    if e.grad is None or not torch.all(torch.isfinite(e.grad)):
        raise AssertionError("Expected finite gradient through Bicycle error dynamics")

    print("lyapunov Bicycle error dynamics integration: ok")


def test_input_validation():
    lyapunov = NeuralLyapunovFunction()
    e = torch.zeros(2, 6)

    try:
        lyapunov.lie_derivative(e, torch.zeros(1, 6))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected mismatched e_dot batch shape to be rejected")

    try:
        lyapunov(torch.zeros(2, 6, dtype=torch.int64))
    except TypeError:
        pass
    else:
        raise AssertionError("Expected integer-valued errors to be rejected")

    try:
        NeuralLyapunovFunction(hidden_sizes=())
    except ValueError:
        pass
    else:
        raise AssertionError("Expected an empty hidden layer configuration to be rejected")

    print("lyapunov input validation: ok")


def main():
    test_quadratic_initialization()
    test_positive_definite_values()
    test_gradient_and_lie_derivative()
    test_bicycle_error_dynamics_integration()
    test_input_validation()
    print("all lyapunov network tests passed")


if __name__ == "__main__":
    main()
