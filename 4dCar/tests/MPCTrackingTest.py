import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.MPCTracking import LinearizedErrorMPCController
from controllers.MPCPlanning import LinearizedPlanningMPCController
from models.error import ErrorDynamics


def to_tensor(value):
    return torch.tensor(value, dtype=torch.float32)


def error_step(error_model, e, z, v, u, dt):
    with torch.no_grad():
        e_dot = error_model.dynamics_from_error(
            to_tensor(e),
            to_tensor(z),
            to_tensor(v),
            to_tensor(u),
        )
    return np.asarray(e, dtype=float) + dt * e_dot.numpy()


def make_controller(error_model, dt=0.05, horizon=20):
    return LinearizedErrorMPCController(
        error_model=error_model,
        dt=dt,
        horizon=horizon,
        Q=np.diag([10.0, 20.0, 3.0, 2.0]),
        R=np.diag([0.02, 0.02]),
        Qf=30.0 * np.eye(4),
        u_min=np.array([-2.0, -3.0]),
        u_max=np.array([2.0, 3.0]),
    )


def test_tracking_mpc_prediction_and_bounds():
    error_model = ErrorDynamics()
    controller = make_controller(error_model, horizon=8)

    e0 = np.array([0.2, -0.1, 3.2, 0.3])
    z0 = np.array([0.0, 0.0])
    v_seq = np.tile(np.array([0.4, 0.1]), (controller.N, 1))

    u, info = controller.solve(e0, z0, v_seq)

    if not info["success"]:
        raise AssertionError(f"Tracking MPC solve failed: {info['status']}")
    if u.shape != (2,):
        raise AssertionError(f"Expected control shape (2,), got {u.shape}")
    if info["E_pred"].shape != (controller.N + 1, 4):
        raise AssertionError(f"Unexpected E_pred shape: {info['E_pred'].shape}")
    if info["Z_pred"].shape != (controller.N + 1, 2):
        raise AssertionError(f"Unexpected Z_pred shape: {info['Z_pred'].shape}")
    if info["U_pred"].shape != (controller.N, 2):
        raise AssertionError(f"Unexpected U_pred shape: {info['U_pred'].shape}")
    if np.any(info["U_pred"] < controller.u_min - 1e-6) or np.any(
        info["U_pred"] > controller.u_max + 1e-6
    ):
        raise AssertionError("Predicted tracking controls violate bounds")

    # The Error Model defines e_theta = theta, so MPC must not wrap it.
    if not np.isclose(info["E_pred"][0, 2], e0[2], atol=1e-7) or not np.isclose(
        info["E_nominal"][0, 2], e0[2], atol=1e-7
    ):
        raise AssertionError("Tracking MPC changed the input error coordinates")

    print("tracking MPC prediction dimensions and bounds: ok")


def test_tracking_mpc_linearization():
    error_model = ErrorDynamics()
    controller = make_controller(error_model, dt=0.05, horizon=5)

    e = np.array([0.3, -0.2, 0.4, 0.5])
    z = np.array([1.0, -1.0])
    v = np.array([0.2, -0.1])
    u = np.array([0.3, -0.4])

    Ad, Bd, cd = error_model.linearize_discrete(
        to_tensor(e),
        to_tensor(z),
        to_tensor(v),
        to_tensor(u),
        controller.dt,
    )
    predicted_next = Ad.numpy() @ e + Bd.numpy() @ u + cd.numpy()
    nonlinear_next = error_step(error_model, e, z, v, u, controller.dt)

    if not np.allclose(predicted_next, nonlinear_next, atol=1e-6):
        raise AssertionError(
            f"Local affine model does not reproduce its linearization point:\n"
            f"predicted={predicted_next}\nnonlinear={nonlinear_next}"
        )

    print("tracking MPC local affine model: ok")


def simulate_closed_loop():
    error_model = ErrorDynamics()
    controller = make_controller(error_model)

    e = np.array([0.5, -0.2, 0.4, 0.2])
    z = np.zeros(2)
    v_seq = np.zeros((controller.N, 2))

    e_hist = [e.copy()]
    u_hist = []
    success_hist = []

    for step in range(100):
        u, info = controller.solve(e, z, v_seq)
        if not info["success"]:
            raise AssertionError(f"Tracking MPC failed at step {step}: {info['status']}")
        if np.any(u < controller.u_min - 1e-6) or np.any(u > controller.u_max + 1e-6):
            raise AssertionError(f"Tracking input violates bounds at step {step}: {u}")

        e = error_step(error_model, e, z, v_seq[0], u, controller.dt)
        if not np.all(np.isfinite(e)):
            raise AssertionError(f"Non-finite error state at step {step}: {e}")

        e_hist.append(e.copy())
        u_hist.append(u.copy())
        success_hist.append(info["success"])

    return {
        "e_hist": np.asarray(e_hist),
        "u_hist": np.asarray(u_hist),
        "success_hist": np.asarray(success_hist, dtype=bool),
        "controller": controller,
    }


def test_tracking_mpc_closed_loop():
    result = simulate_closed_loop()
    initial_error = np.linalg.norm(result["e_hist"][0])
    final_error = np.linalg.norm(result["e_hist"][-1])

    if final_error > 0.15:
        raise AssertionError(f"Expected final error below 0.15, got {final_error}")
    if final_error >= 0.2 * initial_error:
        raise AssertionError(
            f"Expected substantial error reduction, initial={initial_error}, final={final_error}"
        )
    if not np.all(result["success_hist"]):
        raise AssertionError("Expected every tracking MPC solve to succeed")

    print(f"tracking MPC closed-loop error: {initial_error:.6f} -> {final_error:.6f}")
    print("tracking MPC constrained closed loop: ok")
    return result


def test_tracking_mpc_recovers_from_pure_lateral_error():
    error_model = ErrorDynamics()
    controller = make_controller(error_model, dt=0.04, horizon=20)
    e = np.array([0.0, -0.9, 0.0, 0.0])
    z = np.zeros(2)
    v_seq = np.zeros((controller.N, 2))
    initial_position_error = np.linalg.norm(e[:2])

    first_u, first_info = controller.solve(e, z, v_seq)
    if not first_info["success"]:
        raise AssertionError(f"Tracking MPC failed: {first_info['status']}")
    if np.linalg.norm(first_u) < 1e-3:
        raise AssertionError("Tracking MPC stalled at a pure lateral error")

    for _ in range(60):
        u, info = controller.solve(e, z, v_seq)
        if not info["success"]:
            raise AssertionError(f"Tracking MPC failed: {info['status']}")
        e = error_step(error_model, e, z, v_seq[0], u, controller.dt)

    final_position_error = np.linalg.norm(e[:2])
    if final_position_error >= 0.75 * initial_position_error:
        raise AssertionError(
            "Expected lateral-error recovery, "
            f"initial={initial_position_error}, final={final_position_error}"
        )

    print(
        "tracking MPC pure lateral error: "
        f"{initial_position_error:.6f} -> {final_position_error:.6f}"
    )


def test_tracking_mpc_input_validation():
    error_model = ErrorDynamics()
    controller = make_controller(error_model, horizon=5)

    invalid_inputs = (
        (np.zeros(3), np.zeros(2), np.zeros((5, 2))),
        (np.zeros(4), np.zeros(3), np.zeros((5, 2))),
        (np.zeros(4), np.zeros(2), np.zeros((5, 3))),
        (np.array([0.0, np.nan, 0.0, 0.0]), np.zeros(2), np.zeros((5, 2))),
    )
    for e0, z0, v_seq in invalid_inputs:
        try:
            controller.solve(e0, z0, v_seq)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected invalid Tracking MPC input to be rejected")

    print("tracking MPC input validation: ok")


def test_and_plot_planning_tracking_trajectory(show=True):
    """
    Generate a constrained Planning MPC trajectory and track it with Tracking MPC.

    The first subplot compares the planning and tracking x-y paths. The second
    subplot shows all four error variables over time.
    """
    error_model = ErrorDynamics()
    dt = 0.1
    steps = 100
    z_goal = np.array([4.0, 3.0])

    planning_controller = LinearizedPlanningMPCController(
        planning_model=error_model.planning_model,
        dt=dt,
        horizon=30,
        Q=np.diag([5.0, 5.0]),
        R=0.05 * np.eye(2),
        Qf=20.0 * np.eye(2),
        z_min=np.array([-0.5, -0.5]),
        z_max=np.array([4.5, 3.5]),
        z_constraint_matrix=np.array([[-0.75, 1.0]]),
        z_constraint_lower=np.array([-0.3]),
        z_constraint_upper=np.array([0.3]),
        v_min=np.array([-0.5, -0.5]),
        v_max=np.array([0.5, 0.5]),
    )

    z = np.zeros(2)
    z_hist = [z.copy()]
    v_hist = []
    for step in range(steps):
        v, info = planning_controller.solve(z, z_goal)
        if not info["success"]:
            raise AssertionError(f"Planning MPC failed at step {step}: {info['status']}")
        v_hist.append(v.copy())
        z = z + dt * v
        z_hist.append(z.copy())

    z_hist = np.asarray(z_hist)
    v_hist = np.asarray(v_hist)

    tracking_controller = LinearizedErrorMPCController(
        error_model=error_model,
        dt=dt,
        horizon=20,
        Q=np.diag([100.0, 100.0, 0.1, 0.1]),
        R=0.02 * np.eye(2),
        Qf=20.0 * np.diag([100.0, 100.0, 0.1, 0.1]),
        u_min=np.array([-2.0, -3.0]),
        u_max=np.array([2.0, 3.0]),
    )

    x = np.array([0.15, -0.1, 0.0, 0.0])
    x_hist = [x.copy()]
    e_hist = []

    for step in range(steps):
        e = error_model.error(to_tensor(x), to_tensor(z_hist[step])).numpy()
        e_hist.append(e.copy())

        v_seq = v_hist[step : step + tracking_controller.N]
        if len(v_seq) < tracking_controller.N:
            padding = np.tile(v_seq[-1], (tracking_controller.N - len(v_seq), 1))
            v_seq = np.vstack((v_seq, padding))

        u, info = tracking_controller.solve(e, z_hist[step], v_seq)
        if not info["success"]:
            raise AssertionError(f"Tracking MPC failed at step {step}: {info['status']}")

        with torch.no_grad():
            x_dot = error_model.tracking_model.dynamics(to_tensor(x), to_tensor(u)).numpy()
        x = x + dt * x_dot
        x_hist.append(x.copy())

    final_error = error_model.error(to_tensor(x), to_tensor(z_hist[-1])).numpy()
    e_hist.append(final_error)
    x_hist = np.asarray(x_hist)
    e_hist = np.asarray(e_hist)

    position_error = np.linalg.norm(x_hist[:, :2] - z_hist, axis=1)
    if position_error[-1] > 1e-2:
        raise AssertionError(f"Expected final position error below 1e-2, got {position_error[-1]}")

    time = np.arange(steps + 1) * dt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(
        z_hist[:, 0],
        z_hist[:, 1],
        color="tab:blue",
        linestyle="--",
        linewidth=2,
        label="Planning Model",
    )
    axes[0].plot(
        x_hist[:, 0],
        x_hist[:, 1],
        color="tab:blue",
        linestyle="-",
        linewidth=2,
        label="Tracking Model",
    )
    axes[0].scatter(*z_hist[0], color="tab:green", s=60, label="start")
    axes[0].scatter(*z_goal, color="tab:red", marker="*", s=140, label="goal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Tracking and Planning x-y Trajectories")
    axes[0].axis("equal")
    axes[0].grid(True)
    axes[0].legend()

    for index, label in enumerate(("e_x", "e_y", "e_theta", "e_v")):
        axes[1].plot(time, e_hist[:, index], linewidth=1.8, label=label)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("error")
    axes[1].set_title("Error Variables vs Time")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        "MPCTracking_planning_tracking.png",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"planning/tracking figure saved to: {output_path}")
    print(f"final tracking position error: {position_error[-1]:.6e}")
    return output_path


def plot_results(result):
    e_hist = result["e_hist"]
    u_hist = result["u_hist"]
    controller = result["controller"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for index, label in enumerate(("e_x", "e_y", "e_theta", "e_v")):
        axes[0].plot(e_hist[:, index], label=label)
    axes[0].set_title("Tracking Error")
    axes[0].grid(True)
    axes[0].legend()

    for index, label in enumerate(("omega", "a")):
        axes[1].plot(u_hist[:, index], label=label)
        axes[1].axhline(controller.u_min[index], color="black", linestyle=":")
        axes[1].axhline(controller.u_max[index], color="black", linestyle=":")
    axes[1].set_title("Tracking Control")
    axes[1].grid(True)
    axes[1].legend()
    fig.tight_layout()
    plt.show()


def main():
    test_tracking_mpc_prediction_and_bounds()
    test_tracking_mpc_linearization()
    test_tracking_mpc_closed_loop()
    test_tracking_mpc_recovers_from_pure_lateral_error()
    test_tracking_mpc_input_validation()
    test_and_plot_planning_tracking_trajectory(show=True)
    print("all tracking MPC tests passed")


if __name__ == "__main__":
    main()
