import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.MPCPlanning import LinearizedPlanningMPCController
from models.planning import PlanningModel


def make_controller(dt=0.1, horizon=30, corridor_bound=0.4):
    model = PlanningModel()

    z_min = np.array([-0.5, -0.5])
    z_max = np.array([4.5, 3.5])
    corridor_matrix = np.array([[-0.75, 1.0]])
    corridor_lower = np.array([-corridor_bound])
    corridor_upper = np.array([corridor_bound])
    v_min = np.array([-0.5, -0.5])
    v_max = np.array([0.5, 0.5])

    controller = LinearizedPlanningMPCController(
        planning_model=model,
        dt=dt,
        horizon=horizon,
        Q=np.diag([5.0, 5.0]),
        R=0.05 * np.eye(2),
        Qf=20.0 * np.eye(2),
        z_min=z_min,
        z_max=z_max,
        z_constraint_matrix=corridor_matrix,
        z_constraint_lower=corridor_lower,
        z_constraint_upper=corridor_upper,
        v_min=v_min,
        v_max=v_max,
    )
    return controller


def test_exact_single_integrator_model():
    controller = make_controller(dt=0.1, horizon=5)

    z = np.array([1.0, -2.0])
    v = np.array([0.3, -0.4])
    expected_next = z + controller.dt * v
    if not np.allclose(controller._planning_step_numpy(z, v), expected_next):
        raise AssertionError("Planning rollout does not match z_next = z + dt * v")

    Ad, Bd, cd = controller.linearize_discrete(z, v)
    if not np.allclose(Ad, np.eye(2)):
        raise AssertionError(f"Expected Ad=I, got {Ad}")
    if not np.allclose(Bd, controller.dt * np.eye(2)):
        raise AssertionError(f"Expected Bd=dt*I, got {Bd}")
    if not np.allclose(cd, np.zeros(2)):
        raise AssertionError(f"Expected cd=0, got {cd}")

    print("planning MPC exact single-integrator model: ok")


def simulate_planning_mpc():
    dt = 0.1
    steps = 120
    controller = make_controller(dt=dt, horizon=30)

    z = np.array([0.0, 0.0])
    z_goal = np.array([4.0, 3.0])

    z_hist = [z.copy()]
    v_hist = []
    success_hist = []

    for step in range(steps):
        v, info = controller.solve(z, z_goal)

        if not info["success"]:
            raise AssertionError(f"Planning MPC failed at step {step}: {info['status']}")
        if info["Z_pred"].shape != (controller.N + 1, 2):
            raise AssertionError(f"Unexpected Z_pred shape: {info['Z_pred'].shape}")
        if info["V_pred"].shape != (controller.N, 2):
            raise AssertionError(f"Unexpected V_pred shape: {info['V_pred'].shape}")
        if np.any(v < controller.v_min - 1e-5) or np.any(v > controller.v_max + 1e-5):
            raise AssertionError(f"Planning input violates bounds at step {step}: {v}")
        if np.any(info["Z_pred"][1:] < controller.z_min - 1e-5) or np.any(
            info["Z_pred"][1:] > controller.z_max + 1e-5
        ):
            raise AssertionError(f"Predicted planning state violates bounds at step {step}")

        corridor_values = info["Z_pred"][1:] @ controller.z_constraint_matrix.T
        if np.any(corridor_values < controller.z_constraint_lower - 1e-5) or np.any(
            corridor_values > controller.z_constraint_upper + 1e-5
        ):
            raise AssertionError(f"Predicted planning state violates corridor at step {step}")

        z = z + dt * v
        z_hist.append(z.copy())
        v_hist.append(v.copy())
        success_hist.append(info["success"])

    return {
        "z_hist": np.asarray(z_hist),
        "v_hist": np.asarray(v_hist),
        "success_hist": np.asarray(success_hist, dtype=bool),
        "z_goal": z_goal,
        "controller": controller,
        "dt": dt,
    }


def test_planning_mpc_origin_to_goal():
    result = simulate_planning_mpc()
    z_hist = result["z_hist"]
    z_goal = result["z_goal"]
    controller = result["controller"]

    final_error = np.linalg.norm(z_hist[-1] - z_goal)
    corridor_values = z_hist @ controller.z_constraint_matrix.T
    maximum_corridor_value = np.max(np.abs(corridor_values))

    if final_error > 2e-2:
        raise AssertionError(
            f"Planning state did not converge to goal: final={z_hist[-1]}, goal={z_goal}"
        )
    if maximum_corridor_value > 0.4 + 1e-5:
        raise AssertionError(
            f"Expected |y - 0.75x| <= 0.4, maximum value={maximum_corridor_value}"
        )
    if not np.all(result["success_hist"]):
        raise AssertionError("Expected every planning MPC solve to succeed")

    print(f"planning MPC origin-to-goal final error: {final_error:.6e}")
    print("planning MPC constrained closed loop: ok")
    return result


def test_planning_mpc_input_validation():
    controller = make_controller()

    for z0, z_goal in (
        (np.zeros(3), np.zeros(2)),
        (np.zeros(2), np.zeros(3)),
    ):
        try:
            controller.solve(z0, z_goal)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected invalid planning state dimensions to be rejected")

    print("planning MPC input validation: ok")


def test_and_plot_planning_mpc_corridor():
    """
    Drive z from (0, 0) to (4, 3) subject to |y - 0.75x| <= 0.3.

    The generated figure contains the x-y path and the x/y state trajectories
    over time.
    """
    dt = 0.1
    steps = 120
    corridor_bound = 0.3
    controller = make_controller(
        dt=dt,
        horizon=30,
        corridor_bound=corridor_bound,
    )

    z = np.array([0.0, 0.0])
    z_goal = np.array([4.0, 3.0])
    z_hist = [z.copy()]

    for step in range(steps):
        v, info = controller.solve(z, z_goal)
        if not info["success"]:
            raise AssertionError(f"Planning MPC failed at step {step}: {info['status']}")

        z = z + dt * v
        z_hist.append(z.copy())

    z_hist = np.asarray(z_hist)
    corridor_value = z_hist[:, 1] - 0.75 * z_hist[:, 0]
    if np.max(np.abs(corridor_value)) > corridor_bound + 1e-5:
        raise AssertionError(
            f"Expected |y - 0.75x| <= {corridor_bound}, "
            f"got maximum {np.max(np.abs(corridor_value))}"
        )

    time = np.arange(len(z_hist)) * dt
    corridor_x = np.linspace(controller.z_min[0], controller.z_max[0], 300)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].fill_between(
        corridor_x,
        0.75 * corridor_x - corridor_bound,
        0.75 * corridor_x + corridor_bound,
        color="tab:green",
        alpha=0.18,
        label=r"$|y-0.75x|\leq0.3$",
    )
    axes[0].plot(z_hist[:, 0], z_hist[:, 1], linewidth=2, label="MPC trajectory")
    axes[0].scatter(*z_hist[0], color="tab:green", s=70, label="start")
    axes[0].scatter(*z_goal, color="tab:red", marker="*", s=150, label="goal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Planning MPC: x-y Trajectory")
    axes[0].axis("equal")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(time, z_hist[:, 0], linewidth=2, label="x")
    axes[1].plot(time, z_hist[:, 1], linewidth=2, label="y")
    axes[1].axhline(z_goal[0], color="tab:blue", linestyle=":", label="x goal")
    axes[1].axhline(z_goal[1], color="tab:orange", linestyle=":", label="y goal")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("planning state")
    axes[1].set_title("Planning State vs Time")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        "MPCPlanning_corridor_0p3.png",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    final_error = np.linalg.norm(z_hist[-1] - z_goal)
    if final_error > 2e-2:
        raise AssertionError(f"Expected convergence to {z_goal}, final state={z_hist[-1]}")

    print(f"planning MPC corridor figure saved to: {output_path}")
    print(f"planning MPC |y-0.75x| <= 0.3 final error: {final_error:.6e}")
    return output_path


def plot_results(result):
    z_hist = result["z_hist"]
    v_hist = result["v_hist"]
    z_goal = result["z_goal"]
    controller = result["controller"]
    dt = result["dt"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(z_hist[:, 0], z_hist[:, 1], label="planned path")
    axes[0].scatter(*z_hist[0], label="start")
    axes[0].scatter(*z_goal, marker="*", s=120, label="goal")
    axes[0].axis("equal")
    axes[0].grid(True)
    axes[0].legend()

    time = np.arange(len(v_hist)) * dt
    for index, label in enumerate(("v_x", "v_y")):
        axes[1].plot(time, v_hist[:, index], label=label)
        axes[1].axhline(controller.v_min[index], color="black", linestyle=":")
        axes[1].axhline(controller.v_max[index], color="black", linestyle=":")
    axes[1].grid(True)
    axes[1].legend()
    fig.tight_layout()
    plt.show()


def main():
    test_exact_single_integrator_model()
    test_planning_mpc_origin_to_goal()
    test_planning_mpc_input_validation()
    test_and_plot_planning_mpc_corridor()
    print("all planning MPC tests passed")


if __name__ == "__main__":
    main()
