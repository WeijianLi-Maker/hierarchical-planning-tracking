# tests/LyapunovNetworkTest.py

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyapunov.LyapunovNetwork import NeuralLyapunovFunction


def assert_close(actual, expected, atol=1e-6):
    if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"\nactual:   {actual}\nexpected: {expected}")


def test_quadratic_initialization():
    delta = 1e-3
    lyapunov = NeuralLyapunovFunction(
        error_dim=10,
        hidden_sizes=(32, 32),
        feature_dim=8,
        delta=delta,
    )

    e = torch.randn(5, 10)
    V = lyapunov(e)
    expected = delta * torch.sum(e.square(), dim=-1)

    assert_close(V, expected)
    print("lyapunov quadratic initialization: ok")


def test_positive_definite_values():
    lyapunov = NeuralLyapunovFunction(error_dim=10)

    e_zero = torch.zeros(1, 10)
    e_nonzero = torch.zeros(1, 10)
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
        error_dim=10,
        hidden_sizes=(16,),
        feature_dim=4,
        delta=delta,
    )

    e = torch.randn(7, 10, requires_grad=True)
    e_dot = torch.randn(7, 10)

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


def main():
    test_quadratic_initialization()
    test_positive_definite_values()
    test_gradient_and_lie_derivative()
    print("all lyapunov network tests passed")


if __name__ == "__main__":
    main()
