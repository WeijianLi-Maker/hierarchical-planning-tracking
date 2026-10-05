# tests/model_test.py

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.error import ErrorDynamics
from models.planning import DubinsModel
from models.tracking import TrackingModel


def assert_close(actual, expected, atol=1e-6):
    if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"\nactual:   {actual}\nexpected: {expected}")


def test_planning_model():
    model = DubinsModel()

    if model.state_dim != 3 or model.input_dim != 2:
        raise AssertionError(
            f"Expected DubinsModel dimensions (3, 2), got "
            f"({model.state_dim}, {model.input_dim})"
        )

    z = torch.tensor([[1.0, -2.0, torch.pi / 6.0]])
    v = torch.tensor([[2.0, -0.3]])  # [speed_hat, omega_hat]

    z_dot = model.dynamics(z, v)
    expected_z_dot = torch.tensor([[3.0**0.5, 1.0, -0.3]])
    assert_close(z_dot, expected_z_dot)

    z_next = model.step(z, v, dt=0.1)
    expected_z_next = z + 0.1 * expected_z_dot
    assert_close(z_next, expected_z_next)

    print("planning model: ok")


def test_tracking_model():
    model = TrackingModel()

    if model.state_dim != 6 or model.input_dim != 2:
        raise AssertionError(
            f"Expected TrackingModel dimensions (6, 2), got "
            f"({model.state_dim}, {model.input_dim})"
        )

    x = torch.tensor(
        [[1.0, 2.0, 0.3, 12.0, -0.4, 0.2]],
        dtype=torch.float64,
    )
    u = torch.tensor([[0.05, 0.7]], dtype=torch.float64)

    x_dot = model.dynamics(x, u)
    alpha_f = (x[..., 4] + model.l_f * x[..., 5]) / x[..., 3] - u[..., 0]
    alpha_r = (x[..., 4] - model.l_r * x[..., 5]) / x[..., 3]
    F_c_f = -model.C_alpha_f * alpha_f
    F_c_r = -model.C_alpha_r * alpha_r

    expected_x_dot = torch.zeros_like(x)
    expected_x_dot[..., 0] = x[..., 3] * torch.cos(x[..., 2]) - x[..., 4] * torch.sin(x[..., 2])
    expected_x_dot[..., 1] = x[..., 3] * torch.sin(x[..., 2]) + x[..., 4] * torch.cos(x[..., 2])
    expected_x_dot[..., 2] = x[..., 5]
    expected_x_dot[..., 3] = x[..., 5] * x[..., 4] + u[..., 1]
    expected_x_dot[..., 4] = -x[..., 5] * x[..., 3] + 2.0 / model.m * (
        F_c_f * torch.cos(u[..., 0]) + F_c_r
    )
    expected_x_dot[..., 5] = 2.0 / model.I_z * (
        model.l_f * F_c_f - model.l_r * F_c_r
    )
    assert_close(x_dot, expected_x_dot)

    straight_x = torch.tensor([[0.0, 0.0, 0.0, 10.0, 0.0, 0.0]])
    straight_u = torch.zeros(1, 2)
    expected_straight_dot = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert_close(model.dynamics(straight_x, straight_u), expected_straight_dot)

    print("tracking model: ok")


def test_error_dynamics():
    model = ErrorDynamics()

    if model.error_dim != 6:
        raise AssertionError(f"Expected error_dim 6, got {model.error_dim}")

    x = torch.tensor([[2.0, 3.0, 0.7, 10.0, -0.2, 0.1]], dtype=torch.float64)
    z = torch.tensor([[1.0, 1.0, 0.4]], dtype=torch.float64)

    pi_z = model.projection(z)
    expected_pi_z = torch.tensor([[1.0, 1.0, 0.4, 0.0, 0.0, 0.0]], dtype=torch.float64)
    assert_close(pi_z, expected_pi_z)

    e = model.error(x, z)
    relative = x - expected_pi_z
    c = torch.cos(z[..., 2])
    s = torch.sin(z[..., 2])
    expected_e = relative.clone()
    expected_e[..., 0] = c * relative[..., 0] + s * relative[..., 1]
    expected_e[..., 1] = -s * relative[..., 0] + c * relative[..., 1]
    assert_close(e, expected_e)
    assert_close(model.state_from_error(e, z), x)

    v = torch.tensor([[8.0, 0.15]], dtype=torch.float64)
    u = torch.tensor([[0.03, 0.4]], dtype=torch.float64)

    pi_dot = model.projection_dot(z, v)
    expected_pi_dot = torch.tensor(
        [[8.0 * c.item(), 8.0 * s.item(), 0.15, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    assert_close(pi_dot, expected_pi_dot)

    e_dot_from_x = model.dynamics(x, z, v, u)
    e_dot_from_e = model.dynamics_from_error(e, z, v, u)
    assert_close(e_dot_from_e, e_dot_from_x)

    dt = 1e-6
    x_next = x + dt * model.tracking_model.dynamics(x, u)
    z_next = z + dt * model.planning_model.dynamics(z, v)
    finite_difference_e_dot = (model.error(x_next, z_next) - e) / dt
    assert_close(e_dot_from_x, finite_difference_e_dot, atol=1e-4)

    A, B, c = model.linearize(
        e.squeeze(0),
        z.squeeze(0),
        v.squeeze(0),
        u.squeeze(0),
    )
    if A.shape != (6, 6) or B.shape != (6, 2) or c.shape != (6,):
        raise AssertionError(
            f"Unexpected linearization shapes: A={A.shape}, B={B.shape}, c={c.shape}"
        )

    print("error dynamics: ok")


def plot_planning_harmonic_trajectory():
    model = DubinsModel()

    dt = 0.02
    T = 20.0
    steps = int(T / dt)
    time = np.arange(steps + 1) * dt

    z = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    z_hist = [z.squeeze(0).clone()]
    v_hist = []

    for k in range(steps):
        t = k * dt

        speed_hat = 0.5 + 0.2 * np.sin(2.0 * np.pi * 0.1 * t)
        omega_hat = 0.3 * np.cos(2.0 * np.pi * 0.1 * t)
        v = torch.tensor([[speed_hat, omega_hat]], dtype=torch.float32)

        z = model.step(z, v, dt)

        z_hist.append(z.squeeze(0).clone())
        v_hist.append([speed_hat, omega_hat])

    z_hist = torch.stack(z_hist).detach().cpu().numpy()
    v_hist = np.array(v_hist)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(z_hist[:, 0], z_hist[:, 1], linewidth=2)
    axes[0].set_xlabel("px")
    axes[0].set_ylabel("py")
    axes[0].set_title("Planning Trajectory")
    axes[0].axis("equal")
    axes[0].grid(True)

    axes[1].plot(time, z_hist[:, 0], label="X_hat")
    axes[1].plot(time, z_hist[:, 1], label="Y_hat")
    axes[1].plot(time, z_hist[:, 2], label="psi_hat")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("z")
    axes[1].set_title("Planning State")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(time[:-1], v_hist[:, 0], label="speed_hat")
    axes[2].plot(time[:-1], v_hist[:, 1], label="omega_hat")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("v")
    axes[2].set_title("Harmonic Input")
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle("Vehicle Planning Model with Harmonic Input")
    fig.tight_layout()

    plt.show()


def plot_tracking_harmonic_trajectory():
    model = TrackingModel()

    dt = 0.01
    T = 8.0
    steps = int(T / dt)
    time = np.arange(steps + 1) * dt

    x = torch.tensor([[0.0, 0.0, 0.0, 8.0, 0.0, 0.0]], dtype=torch.float32)
    x_hist = [x.squeeze(0).clone()]
    u_hist = []

    for k in range(steps):
        t = k * dt

        delta_f = 0.02 * np.sin(2.0 * np.pi * 0.25 * t)
        a_x = 0.2 * np.sin(2.0 * np.pi * 0.15 * t)
        u = torch.tensor([[delta_f, a_x]], dtype=torch.float32)

        x = model.step(x, u, dt)

        x_hist.append(x.squeeze(0).clone())
        u_hist.append([delta_f, a_x])

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
        "X",
        "Y",
        "psi",
        "v_x",
        "v_y",
        "omega",
    ]
    for i, label in enumerate(state_labels):
        axes[1].plot(time, x_hist[:, i], label=label)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("x")
    axes[1].set_title("Tracking State")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(time[:-1], u_hist[:, 0], label="delta_f")
    axes[2].plot(time[:-1], u_hist[:, 1], label="a_x")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("u")
    axes[2].set_title("Harmonic Input")
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle("Dynamic Bicycle Tracking Model with Harmonic Input")
    fig.tight_layout()

    plt.show()


def main():
    test_planning_model()
    test_tracking_model()
    test_error_dynamics()
    print("all model tests passed")


if __name__ == "__main__":
    main()
