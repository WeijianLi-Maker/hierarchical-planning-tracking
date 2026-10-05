import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.error import ErrorDynamics
from models.planning import PlanningModel
from models.tracking import TrackingModel


def assert_close(actual, expected, atol=1e-6):
    if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"\nactual:   {actual}\nexpected: {expected}")


def test_planning_model():
    model = PlanningModel()

    if model.state_dim != 2 or model.input_dim != 2:
        raise AssertionError(
            f"Expected PlanningModel dimensions (2, 2), got "
            f"({model.state_dim}, {model.input_dim})"
        )

    z = torch.tensor([[1.0, -2.0]])
    v = torch.tensor([[0.2, 0.5]])

    assert_close(model.dynamics(z, v), v)
    assert_close(model.step(z, v, dt=0.1), torch.tensor([[1.02, -1.95]]))
    print("planning model: ok")


def test_tracking_model():
    model = TrackingModel()

    if model.state_dim != 4 or model.input_dim != 2:
        raise AssertionError(
            f"Expected TrackingModel dimensions (4, 2), got "
            f"({model.state_dim}, {model.input_dim})"
        )

    x = torch.tensor(
        [
            [1.0, 2.0, 0.0, 3.0],
            [-1.0, 0.5, torch.pi / 2.0, 2.0],
        ]
    )
    u = torch.tensor([[0.4, -0.2], [-0.3, 0.5]])

    expected_x_dot = torch.tensor(
        [
            [3.0, 0.0, 0.4, -0.2],
            [0.0, 2.0, -0.3, 0.5],
        ]
    )
    assert_close(model.dynamics(x, u), expected_x_dot)
    assert_close(model.step(x, u, dt=0.1), x + 0.1 * expected_x_dot)
    print("tracking model: ok")


def test_error_coordinates():
    model = ErrorDynamics()

    if model.error_dim != 4:
        raise AssertionError(f"Expected error_dim 4, got {model.error_dim}")
    if model.error_coordinates != "world":
        raise AssertionError(
            f"Expected world-frame error coordinates, got {model.error_coordinates}"
        )
    expected_signature = {
        "error_dim": 4,
        "planning_state_dim": 2,
        "planning_input_dim": 2,
        "tracking_state_dim": 4,
        "tracking_input_dim": 2,
        "error_coordinates": "world",
        "error_dynamics_version": 1,
    }
    if model.model_signature() != expected_signature:
        raise AssertionError(f"Unexpected model signature: {model.model_signature()}")

    x = torch.tensor([[2.0, 3.0, torch.pi / 2.0, 1.5]])
    z = torch.tensor([[1.0, 1.0]])

    expected_projection = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    expected_projection_dot = torch.tensor([[0.2, -0.3, 0.0, 0.0]])
    expected_phi = torch.eye(4).unsqueeze(0)
    expected_error = torch.tensor([[1.0, 2.0, torch.pi / 2.0, 1.5]])

    assert_close(model.projection(z), expected_projection)
    assert_close(model.projection_dot(z, torch.tensor([[0.2, -0.3]])), expected_projection_dot)
    assert_close(model.phi(x, z), expected_phi)
    assert_close(model.error(x, z), expected_error)
    assert_close(model.state_from_error(expected_error, z), x)
    print("error coordinates: ok")


def test_error_dynamics():
    model = ErrorDynamics()

    x = torch.tensor([2.0, 3.0, 0.5, 1.2], requires_grad=True)
    z = torch.tensor([1.0, 1.5], requires_grad=True)
    v = torch.tensor([0.2, -0.3])
    u = torch.tensor([0.4, -0.1])

    e = model.error(x, z)
    theta = e[2]
    expected_e_dot = torch.stack(
        (
            e[3] * torch.cos(theta) - v[0],
            e[3] * torch.sin(theta) - v[1],
            u[0],
            u[1],
        )
    )

    assert_close(model.dynamics(x, z, v, u), expected_e_dot)
    assert_close(model.dynamics_from_error(e, z, v, u), expected_e_dot)

    x_dot = model.tracking_model.dynamics(x, u)
    z_dot = model.planning_model.dynamics(z, v)
    _, automatic_e_dot = torch.autograd.functional.jvp(
        lambda x_var, z_var: model.error(x_var, z_var),
        (x, z),
        (x_dot, z_dot),
    )
    assert_close(expected_e_dot, automatic_e_dot)

    A, B, c = model.linearize(e.detach(), z.detach(), v, u)
    if A.shape != (4, 4) or B.shape != (4, 2) or c.shape != (4,):
        raise AssertionError(
            f"Unexpected linearization shapes: A={A.shape}, B={B.shape}, c={c.shape}"
        )
    assert_close(A @ e.detach() + B @ u + c, expected_e_dot.detach())

    Ad, Bd, cd = model.linearize_discrete(e.detach(), z.detach(), v, u, dt=0.1)
    assert_close(Ad, torch.eye(4) + 0.1 * A)
    assert_close(Bd, 0.1 * B)
    assert_close(cd, 0.1 * c)
    print("error dynamics: ok")


def plot_planning_trajectory():
    model = PlanningModel()
    dt = 0.02
    steps = int(20.0 / dt)

    z = torch.zeros(1, model.state_dim)
    history = [z.squeeze(0).clone()]
    for k in range(steps):
        t = k * dt
        v = torch.tensor(
            [[0.3 * np.cos(0.4 * t), 0.3 * np.sin(0.4 * t)]],
            dtype=torch.float32,
        )
        z = model.step(z, v, dt)
        history.append(z.squeeze(0).clone())

    history = torch.stack(history).numpy()
    plt.plot(history[:, 0], history[:, 1])
    plt.title("2D Single-Integrator Planning Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def plot_tracking_trajectory():
    model = TrackingModel()
    dt = 0.01
    steps = int(12.0 / dt)

    x = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    history = [x.squeeze(0).clone()]
    for _ in range(steps):
        u = torch.tensor([[0.3, 0.0]])
        x = model.step(x, u, dt)
        history.append(x.squeeze(0).clone())

    history = torch.stack(history).numpy()
    plt.plot(history[:, 0], history[:, 1])
    plt.title("4D Vehicle Tracking Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def main():
    test_planning_model()
    test_tracking_model()
    test_error_coordinates()
    test_error_dynamics()
    print("all model tests passed")


if __name__ == "__main__":
    main()
