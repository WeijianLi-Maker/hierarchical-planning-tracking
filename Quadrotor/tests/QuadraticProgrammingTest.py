import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.QuadraticProgramming import solve_quadratic_program


def test_unconstrained_qp():
    # min 0.5 ||x - [1, 2]||^2  =>  x* = [1, 2]
    P = np.eye(2)
    q = np.array([-1.0, -2.0])
    result = solve_quadratic_program(P, q)

    if not result.success:
        raise AssertionError(f"Expected successful solve, got {result.status}")
    if not np.allclose(result.x, np.array([1.0, 2.0]), atol=1e-5):
        raise AssertionError(f"Unexpected solution: {result.x}")

    print("unconstrained qp: ok")


def test_constrained_qp():
    # min 0.5(x1^2 + x2^2)  s.t. x1 + x2 >= 1, x1 >= 0, x2 >= 0
    # Equivalent to min 0.5 x^T x with -x1 - x2 <= -1
    P = np.eye(2)
    q = np.zeros(2)
    G = np.array([[-1.0, -1.0]])
    h = np.array([-1.0])
    lb = np.zeros(2)

    result = solve_quadratic_program(P, q, G=G, h=h, lb=lb)
    expected = np.array([0.5, 0.5])

    if not result.success:
        raise AssertionError(f"Expected successful solve, got {result.status}")
    if not np.allclose(result.x, expected, atol=1e-5):
        raise AssertionError(f"Expected {expected}, got {result.x}")

    print("constrained qp: ok")


def test_equality_constrained_qp():
    # min 0.5(x1^2 + x2^2) + x1  s.t. x1 + x2 = 1
    P = np.eye(2)
    q = np.array([1.0, 0.0])
    A_eq = np.array([[1.0, 1.0]])
    b_eq = np.array([1.0])

    result = solve_quadratic_program(P, q, A_eq=A_eq, b_eq=b_eq)
    expected = np.array([0.0, 1.0])

    if not result.success:
        raise AssertionError(f"Expected successful solve, got {result.status}")
    if not np.allclose(result.x, expected, atol=1e-5):
        raise AssertionError(f"Expected {expected}, got {result.x}")

    print("equality constrained qp: ok")


if __name__ == "__main__":
    test_unconstrained_qp()
    test_constrained_qp()
    test_equality_constrained_qp()
    print("all quadratic programming tests passed")
