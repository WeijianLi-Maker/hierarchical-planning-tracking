# tests/model_test.py

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.error import ErrorDynamics
from models.planning import VehicleModel
from models.tracking import QuadratorModel


def assert_close(actual, expected, atol=1e-6):
    if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"\nactual:   {actual}\nexpected: {expected}")


def test_planning_model():
    model = VehicleModel()

    if model.state_dim != 3 or model.input_dim != 3:
        raise AssertionError(
            f"Expected VehicleModel dimensions (3, 3), got "
            f"({model.state_dim}, {model.input_dim})"
        )

    z = torch.tensor([[0.0, 0.0, 0.0]])
    v = torch.tensor([[0.2, 0.5, -0.3]])  # [b_x, b_y, b_z]

    z_dot = model.dynamics(z, v)
    expected_z_dot = torch.tensor([[0.2, 0.5, -0.3]])
    assert_close(z_dot, expected_z_dot)

    z_next = model.step(z, v, dt=0.1)
    expected_z_next = torch.tensor([[0.02, 0.05, -0.03]])
    assert_close(z_next, expected_z_next)

    print("planning model: ok")


def test_tracking_model():
    model = QuadratorModel()

    if model.state_dim != 10 or model.input_dim != 3:
        raise AssertionError(
            f"Expected QuadratorModel dimensions (10, 3), got "
            f"({model.state_dim}, {model.input_dim})"
        )

    x = torch.zeros(1, model.state_dim)
    u = torch.tensor([[0.0, 0.0, model.g / model.kT]])

    x_dot = model.dynamics(x, u)
    expected_x_dot = torch.zeros_like(x)
    assert_close(x_dot, expected_x_dot)

    x = torch.tensor(
        [[
            1.0,
            0.2,
            0.1,
            -0.3,
            2.0,
            -0.4,
            -0.2,
            0.5,
            3.0,
            0.6,
        ]]
    )
    u = torch.tensor([[0.05, -0.04, model.g / model.kT + 0.2]])

    x_dot = model.dynamics(x, u)
    expected_x_dot = torch.zeros_like(x)
    expected_x_dot[..., 0] = x[..., 1]
    expected_x_dot[..., 1] = model.g * torch.tan(x[..., 2])
    expected_x_dot[..., 2] = -model.d1 * x[..., 2] + x[..., 3]
    expected_x_dot[..., 3] = -model.d0 * x[..., 2] + model.n0 * u[..., 0]
    expected_x_dot[..., 4] = x[..., 5]
    expected_x_dot[..., 5] = model.g * torch.tan(x[..., 6])
    expected_x_dot[..., 6] = -model.d1 * x[..., 6] + x[..., 7]
    expected_x_dot[..., 7] = -model.d0 * x[..., 6] + model.n0 * u[..., 1]
    expected_x_dot[..., 8] = x[..., 9]
    expected_x_dot[..., 9] = model.kT * u[..., 2] - model.g
    assert_close(x_dot, expected_x_dot)

    d = torch.tensor([[0.1, -0.05, 0.08]])
    disturbed_x_dot = model.dynamics(x, u, d)
    expected_disturbed_x_dot = expected_x_dot.clone()
    expected_disturbed_x_dot[..., 0] += d[..., 0]
    expected_disturbed_x_dot[..., 4] += d[..., 1]
    expected_disturbed_x_dot[..., 8] += d[..., 2]
    assert_close(disturbed_x_dot, expected_disturbed_x_dot)

    try:
        model.dynamics(x, u, torch.tensor([[0.11, 0.0, 0.0]]))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected disturbance outside [-0.1, 0.1] to be rejected")

    print("tracking model: ok")


def test_error_dynamics():
    model = ErrorDynamics()

    if model.error_dim != 10:
        raise AssertionError(f"Expected error_dim 10, got {model.error_dim}")

    x = torch.tensor([[1.0, 0.2, 0.1, -0.1, 2.0, -0.3, 0.2, 0.4, 3.0, 0.5]])
    z = torch.tensor([[0.5, 1.5, 2.5]])

    pi_z = model.projection(z)
    expected_pi_z = torch.tensor([[0.5, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 2.5, 0.0]])
    assert_close(pi_z, expected_pi_z)

    e = model.error(x, z)
    expected_e = x - expected_pi_z
    assert_close(e, expected_e)

    v = torch.tensor([[0.2, -0.1, 0.3]])
    u = torch.tensor([[0.0, 0.0, model.tracking_model.g / model.tracking_model.kT]])

    pi_dot = model.projection_dot(z, v)
    expected_pi_dot = torch.tensor([[0.2, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.3, 0.0]])
    assert_close(pi_dot, expected_pi_dot)

    e_dot_from_x = model.dynamics(x, z, v, u)
    expected_e_dot = model.tracking_model.dynamics(x, u) - expected_pi_dot
    assert_close(e_dot_from_x, expected_e_dot)

    e_dot_from_e = model.dynamics_from_error(e, z, v, u)
    assert_close(e_dot_from_e, e_dot_from_x, atol=1e-4)

    d = torch.tensor([[0.1, -0.1, 0.05]])
    disturbed_e_dot = model.dynamics(x, z, v, u, d)
    expected_disturbed_e_dot = model.tracking_model.dynamics(x, u, d) - expected_pi_dot
    assert_close(disturbed_e_dot, expected_disturbed_e_dot)

    A, B, c = model.linearize(
        e.squeeze(0),
        z.squeeze(0),
        v.squeeze(0),
        u.squeeze(0),
        d.squeeze(0),
    )
    if A.shape != (10, 10) or B.shape != (10, 3) or c.shape != (10,):
        raise AssertionError(
            f"Unexpected linearization shapes: A={A.shape}, B={B.shape}, c={c.shape}"
        )

    print("error dynamics: ok")


def plot_planning_harmonic_trajectory():
    model = VehicleModel()

    dt = 0.02
    T = 20.0
    steps = int(T / dt)
    time = np.arange(steps + 1) * dt

    z = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    z_hist = [z.squeeze(0).clone()]
    v_hist = []

    for k in range(steps):
        t = k * dt

        bx = 0.3 * np.sin(2.0 * np.pi * 0.1 * t)
        by = 0.3 * np.cos(2.0 * np.pi * 0.1 * t)
        bz = 0.2 * np.sin(2.0 * np.pi * 0.2 * t)
        v = torch.tensor([[bx, by, bz]], dtype=torch.float32)

        z = model.step(z, v, dt)

        z_hist.append(z.squeeze(0).clone())
        v_hist.append([bx, by, bz])

    z_hist = torch.stack(z_hist).detach().cpu().numpy()
    v_hist = np.array(v_hist)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(z_hist[:, 0], z_hist[:, 1], linewidth=2)
    axes[0].set_xlabel("px")
    axes[0].set_ylabel("py")
    axes[0].set_title("Planning Trajectory")
    axes[0].axis("equal")
    axes[0].grid(True)

    axes[1].plot(time, z_hist[:, 0], label="px")
    axes[1].plot(time, z_hist[:, 1], label="py")
    axes[1].plot(time, z_hist[:, 2], label="z")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("z")
    axes[1].set_title("Planning State")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(time[:-1], v_hist[:, 0], label="b_x")
    axes[2].plot(time[:-1], v_hist[:, 1], label="b_y")
    axes[2].plot(time[:-1], v_hist[:, 2], label="b_z")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("v")
    axes[2].set_title("Harmonic Input")
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle("Vehicle Planning Model with Harmonic Input")
    fig.tight_layout()

    plt.show()


def plot_tracking_harmonic_trajectory():
    model = QuadratorModel()

    dt = 0.01
    T = 8.0
    steps = int(T / dt)
    time = np.arange(steps + 1) * dt

    x = torch.zeros(1, model.state_dim, dtype=torch.float32)
    x_hist = [x.squeeze(0).clone()]
    u_hist = []

    for k in range(steps):
        t = k * dt

        ax = 0.08 * np.sin(2.0 * np.pi * 0.25 * t)
        ay = 0.08 * np.cos(2.0 * np.pi * 0.25 * t)
        az = model.g / model.kT + 0.5 * np.sin(2.0 * np.pi * 0.15 * t)
        u = torch.tensor([[ax, ay, az]], dtype=torch.float32)

        x = model.step(x, u, dt)

        x_hist.append(x.squeeze(0).clone())
        u_hist.append([ax, ay, az])

    x_hist = torch.stack(x_hist).detach().cpu().numpy()
    u_hist = np.array(u_hist)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].plot(x_hist[:, 0], x_hist[:, 1], linewidth=2)
    axes[0].set_xlabel("x position")
    axes[0].set_ylabel("y position")
    axes[0].set_title("Tracking Trajectory")
    axes[0].axis("equal")
    axes[0].grid(True)

    state_labels = [
        "x",
        "vx",
        "theta_x",
        "omega_x",
        "y",
        "vy",
        "theta_y",
        "omega_y",
        "z",
        "vz",
    ]
    for i, label in enumerate(state_labels):
        axes[1].plot(time, x_hist[:, i], label=label)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("x")
    axes[1].set_title("Tracking State")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(time[:-1], u_hist[:, 0], label="a_x")
    axes[2].plot(time[:-1], u_hist[:, 1], label="a_y")
    axes[2].plot(time[:-1], u_hist[:, 2], label="a_z")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("u")
    axes[2].set_title("Harmonic Input")
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle("Quadrotor Tracking Model with Harmonic Input")
    fig.tight_layout()

    plt.show()


def main():
    test_planning_model()
    test_tracking_model()
    test_error_dynamics()
    print("all model tests passed")
    plot_planning_harmonic_trajectory()
    plot_tracking_harmonic_trajectory()


if __name__ == "__main__":
    main()
