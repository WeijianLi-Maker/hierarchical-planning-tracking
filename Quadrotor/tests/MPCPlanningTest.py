import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.MPCPlanning import LinearizedPlanningMPCController
from models.planning import VehicleModel


def simulate_planning_mpc():
    dt = 0.1
    horizon = 30
    simulation_seconds = 12.0
    steps = int(simulation_seconds / dt)

    model = VehicleModel()

    # Keep z fixed at zero while allowing motion from (0, 0) to (4, 3).
    z_min = np.array([-0.5, -0.5, 0.0])
    z_max = np.array([4.5, 3.5, 0.0])
    z_constraint_matrix = np.array([[-0.75, 1.0, 0.0]])
    z_constraint_lower = np.array([-0.4])
    z_constraint_upper = np.array([0.4])
    v_min = np.array([-0.5, -0.5, -0.5])
    v_max = np.array([0.5, 0.5, 0.5])

    controller = LinearizedPlanningMPCController(
        planning_model=model,
        dt=dt,
        horizon=horizon,
        Q=np.diag([5.0, 5.0, 10.0]),
        R=0.05 * np.eye(3),
        Qf=20.0 * np.eye(3),
        z_min=z_min,
        z_max=z_max,
        z_constraint_matrix=z_constraint_matrix,
        z_constraint_lower=z_constraint_lower,
        z_constraint_upper=z_constraint_upper,
        v_min=v_min,
        v_max=v_max,
    )

    z = np.array([0.0, 0.0, 0.0])
    z_goal = np.array([4.0, 3.0, 0.0])

    z_hist = [z.copy()]
    v_hist = []
    cost_hist = []
    success_hist = []

    for step in range(steps):
        v, info = controller.solve(z, z_goal)

        if not info["success"]:
            raise AssertionError(f"Planning MPC failed at step {step}: {info['status']}")
        if np.any(v < v_min - 1e-5) or np.any(v > v_max + 1e-5):
            raise AssertionError(f"Planning input violates bounds at step {step}: {v}")
        if np.any(info["Z_pred"][1:] < z_min - 1e-5) or np.any(info["Z_pred"][1:] > z_max + 1e-5):
            raise AssertionError(f"Predicted planning state violates bounds at step {step}")
        predicted_corridor_value = info["Z_pred"][1:] @ z_constraint_matrix.T
        if np.any(predicted_corridor_value < z_constraint_lower - 1e-5) or np.any(
            predicted_corridor_value > z_constraint_upper + 1e-5
        ):
            raise AssertionError(f"Predicted planning state violates corridor at step {step}")

        with torch.no_grad():
            z_dot = model.dynamics(
                torch.tensor(z, dtype=torch.float32),
                torch.tensor(v, dtype=torch.float32),
            ).numpy()
        z = z + dt * z_dot

        z_hist.append(z.copy())
        v_hist.append(v.copy())
        cost_hist.append(info["cost"])
        success_hist.append(info["success"])

    result = {
        "time_state": np.arange(steps + 1) * dt,
        "time_input": np.arange(steps) * dt,
        "z_hist": np.asarray(z_hist),
        "v_hist": np.asarray(v_hist),
        "cost_hist": np.asarray(cost_hist),
        "success_hist": np.asarray(success_hist, dtype=bool),
        "z_goal": z_goal,
        "z_min": z_min,
        "z_max": z_max,
        "v_min": v_min,
        "v_max": v_max,
        "z_constraint_matrix": z_constraint_matrix,
        "z_constraint_lower": z_constraint_lower,
        "z_constraint_upper": z_constraint_upper,
    }
    return result


def test_planning_mpc_origin_to_goal():
    result = simulate_planning_mpc()
    z_hist = result["z_hist"]
    z_goal = result["z_goal"]

    final_error = np.linalg.norm(z_hist[-1] - z_goal)
    maximum_z_deviation = np.max(np.abs(z_hist[:, 2]))
    corridor_value = z_hist[:, 1] - 0.75 * z_hist[:, 0]
    maximum_corridor_value = np.max(np.abs(corridor_value))

    if final_error > 2e-2:
        raise AssertionError(
            f"Planning state did not converge to goal: final={z_hist[-1]}, goal={z_goal}"
        )
    if maximum_z_deviation > 1e-8:
        raise AssertionError(f"Expected fixed z coordinate, maximum deviation={maximum_z_deviation}")
    if maximum_corridor_value > 0.4 + 1e-5:
        raise AssertionError(
            f"Expected |y - 0.75x| <= 0.4, maximum value={maximum_corridor_value}"
        )
    if not np.all(result["success_hist"]):
        raise AssertionError("Expected every planning MPC solve to succeed")

    print(f"initial position: {z_hist[0]} m")
    print(f"goal position:    {z_goal} m")
    print(f"final position:   {z_hist[-1]} m")
    print(f"final error:      {final_error:.6f} m")
    print(f"maximum z drift:  {maximum_z_deviation:.6f} m")
    print(f"max |y-0.75x|:   {maximum_corridor_value:.6f} m")
    print(f"MPC success rate: {100.0 * np.mean(result['success_hist']):.1f}%")
    return result


def plot_results(result):
    z_hist = result["z_hist"]
    v_hist = result["v_hist"]
    z_goal = result["z_goal"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].plot(z_hist[:, 0], z_hist[:, 1], color="tab:blue", label="planned path")
    axes[0].scatter(z_hist[0, 0], z_hist[0, 1], color="tab:green", s=80, label="start")
    axes[0].scatter(z_goal[0], z_goal[1], color="tab:red", marker="*", s=160, label="goal")
    corridor_x = np.linspace(result["z_min"][0], result["z_max"][0], 200)
    axes[0].fill_between(
        corridor_x,
        0.75 * corridor_x - 0.4,
        0.75 * corridor_x + 0.4,
        color="tab:green",
        alpha=0.15,
        label=r"$|y-0.75x|\leq0.4$",
    )
    axes[0].plot(corridor_x, 0.75 * corridor_x - 0.4, color="tab:green", linestyle="--")
    axes[0].plot(corridor_x, 0.75 * corridor_x + 0.4, color="tab:green", linestyle="--")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title("Planning Trajectory in XY Plane")
    axes[0].axis("equal")
    axes[0].grid(True)
    axes[0].legend()

    for index, label in enumerate(("x", "y", "z")):
        axes[1].plot(result["time_state"], z_hist[:, index], label=f"{label} position")
        axes[1].axhline(z_goal[index], linestyle="--", linewidth=1)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("position [m]")
    axes[1].set_title("Planning State")
    axes[1].grid(True)
    axes[1].legend()

    for index, label in enumerate(("v_x", "v_y", "v_z")):
        axes[2].plot(result["time_input"], v_hist[:, index], label=label)
        axes[2].axhline(result["v_min"][index], color="black", linestyle=":", linewidth=0.8)
        axes[2].axhline(result["v_max"][index], color="black", linestyle=":", linewidth=0.8)
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("planning input [m/s]")
    axes[2].set_title("Planning Input and Constraints")
    axes[2].grid(True)
    axes[2].legend()

    fig.suptitle("MPC Planning: Origin to (4 m, 3 m), Fixed z")
    fig.tight_layout()
    plt.show()


def main():
    result = test_planning_mpc_origin_to_goal()
    plot_results(result)
    print("planning MPC origin-to-goal test passed")


if __name__ == "__main__":
    main()
