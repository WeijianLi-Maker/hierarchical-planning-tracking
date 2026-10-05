import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.MPCTracking import LinearizedErrorMPCController
from models.error import ErrorDynamics


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def dynamics_numpy(model, state, control):
    with torch.no_grad():
        return to_numpy(
            model.dynamics(
                torch.tensor(state, dtype=torch.float32),
                torch.tensor(control, dtype=torch.float32),
            )
        )


def simulate_closed_loop():
    dt = 0.02
    horizon = 20
    simulation_seconds = 3.0
    steps = int(simulation_seconds / dt)

    error_model = ErrorDynamics()
    planning_model = error_model.planning_model
    tracking_model = error_model.tracking_model
    controller = LinearizedErrorMPCController(error_model, dt=dt, horizon=horizon)

    z = np.array([0.0, 0.0, 0.0])
    x = np.array([0.4, -0.2, 0.04, 9.5, 0.15, 0.03])
    planner_input = np.array([10.0, 0.0])
    v_seq = np.tile(planner_input, (horizon, 1))

    z_hist = [z.copy()]
    x_hist = [x.copy()]
    e_hist = [to_numpy(error_model.error(torch.tensor(x), torch.tensor(z)))]
    u_hist = []
    success_hist = []

    for _ in range(steps):
        e = to_numpy(error_model.error(torch.tensor(x), torch.tensor(z)))
        u, info = controller.solve(e, z, v_seq)

        z = z + dt * dynamics_numpy(planning_model, z, planner_input)
        x = x + dt * dynamics_numpy(tracking_model, x, u)

        z_hist.append(z.copy())
        x_hist.append(x.copy())
        e_hist.append(to_numpy(error_model.error(torch.tensor(x), torch.tensor(z))))
        u_hist.append(u.copy())
        success_hist.append(info["success"])

    return {
        "z_hist": np.asarray(z_hist),
        "x_hist": np.asarray(x_hist),
        "e_hist": np.asarray(e_hist),
        "u_hist": np.asarray(u_hist),
        "success_hist": np.asarray(success_hist, dtype=bool),
        "controller": controller,
        "dt": dt,
    }


def test_tracking_mpc_straight_path():
    result = simulate_closed_loop()
    initial_error = result["e_hist"][0]
    final_error = result["e_hist"][-1]
    controller = result["controller"]
    reference_error = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0])

    if not np.all(result["success_hist"]):
        raise AssertionError("Expected every tracking MPC solve to succeed")
    if np.any(result["u_hist"] < controller.u_min - 1e-5):
        raise AssertionError("Tracking input violates lower bounds")
    if np.any(result["u_hist"] > controller.u_max + 1e-5):
        raise AssertionError("Tracking input violates upper bounds")
    if np.linalg.norm(final_error - reference_error) >= np.linalg.norm(
        initial_error - reference_error
    ):
        raise AssertionError(
            f"Tracking error did not approach its moving reference: "
            f"initial={initial_error}, final={final_error}"
        )
    if not np.all(np.isfinite(result["x_hist"])):
        raise AssertionError("Closed-loop tracking state contains non-finite values")

    print(f"initial error norm: {np.linalg.norm(initial_error):.6f}")
    print(f"final error norm:   {np.linalg.norm(final_error):.6f}")
    print(f"final pose error:   {np.linalg.norm(final_error[:3]):.6f}")
    print(f"MPC success rate:   {100.0 * np.mean(result['success_hist']):.1f}%")


def test_tracking_mpc_moving_reference_and_prediction_shapes():
    error_model = ErrorDynamics()
    controller = LinearizedErrorMPCController(error_model, dt=0.02, horizon=8)
    planner_input = np.array([8.0, 0.15])
    v_seq = np.tile(planner_input, (controller.N, 1))
    e_ref = controller.error_reference(v_seq)

    expected_reference = np.array([0.0, 0.0, 0.0, 8.0, 0.0, 0.15])
    if not np.allclose(e_ref, expected_reference):
        raise AssertionError(f"Unexpected moving error reference: {e_ref}")

    e0 = np.array([0.1, -0.1, 0.02, 8.0, 0.05, 0.1])
    z0 = np.array([0.0, 0.0, 0.1])
    u, info = controller.solve(e0, z0, v_seq)

    if not info["success"]:
        raise AssertionError(f"Tracking MPC failed for turning input: {info['status']}")
    if u.shape != (2,):
        raise AssertionError(f"Expected tracking input shape (2,), got {u.shape}")
    if info["E_pred"].shape != (controller.N + 1, 6):
        raise AssertionError(f"Unexpected E_pred shape: {info['E_pred'].shape}")
    if info["Z_pred"].shape != (controller.N + 1, 3):
        raise AssertionError(f"Unexpected Z_pred shape: {info['Z_pred'].shape}")
    if info["U_pred"].shape != (controller.N, 2):
        raise AssertionError(f"Unexpected U_pred shape: {info['U_pred'].shape}")
    if info["E_ref"].shape != (controller.N + 1, 6):
        raise AssertionError(f"Unexpected E_ref shape: {info['E_ref'].shape}")

    print("tracking MPC moving reference and prediction shapes: ok")


def test_tracking_mpc_wraps_heading_and_rejects_wrong_planner_shape():
    controller = LinearizedErrorMPCController(ErrorDynamics(), dt=0.02, horizon=5)
    wrapped = controller._wrap_error_numpy(
        np.array([0.0, 0.0, 3.0 * np.pi, 8.0, 0.0, 0.0])
    )
    if not np.isclose(wrapped[2], -np.pi):
        raise AssertionError(f"Expected wrapped heading error -pi, got {wrapped[2]}")

    try:
        controller.solve(
            np.array([0.0, 0.0, 0.0, 8.0, 0.0, 0.0]),
            np.zeros(3),
            np.zeros((controller.N, 3)),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected three-dimensional planning inputs to be rejected")

    print("tracking MPC heading wrapping and input validation: ok")


def plot_results(result):
    time_state = np.arange(len(result["e_hist"])) * result["dt"]
    time_input = np.arange(len(result["u_hist"])) * result["dt"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].plot(result["z_hist"][:, 0], result["z_hist"][:, 1], "--", label="planning")
    axes[0].plot(result["x_hist"][:, 0], result["x_hist"][:, 1], label="tracking")
    axes[0].axis("equal")
    axes[0].grid(True)
    axes[0].legend()

    for index, label in enumerate(("e_X", "e_Y", "e_psi", "e_vx", "e_vy", "e_omega")):
        axes[1].plot(time_state, result["e_hist"][:, index], label=label)
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(time_input, result["u_hist"][:, 0], label="delta_f")
    axes[2].plot(time_input, result["u_hist"][:, 1], label="a_x")
    axes[2].grid(True)
    axes[2].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_tracking_mpc_moving_reference_and_prediction_shapes()
    test_tracking_mpc_wraps_heading_and_rejects_wrong_planner_shape()
    test_tracking_mpc_straight_path()
