import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.MPCPlanning import LinearizedPlanningMPCController
from models.planning import DubinsModel


def simulate_planning_mpc():
    dt = 0.1
    horizon = 30
    simulation_seconds = 12.0
    steps = int(simulation_seconds / dt)

    model = DubinsModel()

    z_min = np.array([-0.5, -0.5, -np.pi])
    z_max = np.array([4.5, 3.5, np.pi])
    z_constraint_matrix = np.zeros((0, 3))
    z_constraint_lower = np.zeros(0)
    z_constraint_upper = np.zeros(0)
    v_min = np.array([0.0, -1.0])
    v_max = np.array([1.0, 1.0])

    controller = LinearizedPlanningMPCController(
        planning_model=model,
        dt=dt,
        horizon=horizon,
        Q=np.diag([5.0, 5.0, 10.0]),
        R=0.05 * np.eye(2),
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
    z_goal = np.array([4.0, 3.0, np.arctan2(3.0, 4.0)])

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

    if final_error > 1e-1:
        raise AssertionError(
            f"Planning state did not converge to goal: final={z_hist[-1]}, goal={z_goal}"
        )
    if not np.all(result["success_hist"]):
        raise AssertionError("Expected every planning MPC solve to succeed")

    print(f"initial position: {z_hist[0]} m")
    print(f"goal position:    {z_goal} m")
    print(f"final position:   {z_hist[-1]} m")
    print(f"final error:      {final_error:.6f} m")
    print(f"MPC success rate: {100.0 * np.mean(result['success_hist']):.1f}%")


def test_planning_mpc_linearization_and_prediction_shapes():
    model = DubinsModel()
    controller = LinearizedPlanningMPCController(model, dt=0.1, horizon=8)
    z = np.array([1.0, -2.0, 0.4])
    v = np.array([0.8, -0.2])

    Ad, Bd, cd = controller.linearize_discrete(z, v)
    actual_next = controller._planning_step_numpy(z, v)
    affine_next = Ad @ z + Bd @ v + cd

    if Ad.shape != (3, 3) or Bd.shape != (3, 2) or cd.shape != (3,):
        raise AssertionError(f"Unexpected linearization shapes: {Ad.shape}, {Bd.shape}, {cd.shape}")
    if not np.allclose(actual_next, affine_next, atol=1e-6):
        raise AssertionError(
            f"Affine model does not match its linearization point: "
            f"actual={actual_next}, affine={affine_next}"
        )

    u, info = controller.solve(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.2, 0.1]),
    )
    if u.shape != (2,):
        raise AssertionError(f"Expected planning input shape (2,), got {u.shape}")
    if info["Z_pred"].shape != (controller.N + 1, 3):
        raise AssertionError(f"Unexpected Z_pred shape: {info['Z_pred'].shape}")
    if info["V_pred"].shape != (controller.N, 2):
        raise AssertionError(f"Unexpected V_pred shape: {info['V_pred'].shape}")

    print("planning MPC linearization and prediction shapes: ok")


def test_planning_mpc_wraps_equivalent_goal_heading():
    controller = LinearizedPlanningMPCController(DubinsModel(), dt=0.1, horizon=5)
    z0 = np.array([0.0, 0.0, np.pi - 0.05])
    z_goal = np.array([0.0, 0.0, -np.pi + 0.05])

    equivalent_goal = controller._equivalent_goal(z0, z_goal)
    heading_difference = equivalent_goal[2] - z0[2]
    if not np.isclose(heading_difference, 0.1, atol=1e-8):
        raise AssertionError(
            f"Expected shortest heading difference 0.1 rad, got {heading_difference}"
        )

    print("planning MPC equivalent goal heading: ok")


def plot_results(result):
    z_hist = result["z_hist"]
    v_hist = result["v_hist"]
    z_goal = result["z_goal"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].plot(z_hist[:, 0], z_hist[:, 1], color="tab:blue", label="planned path")
    axes[0].scatter(z_hist[0, 0], z_hist[0, 1], color="tab:green", s=80, label="start")
    axes[0].scatter(z_goal[0], z_goal[1], color="tab:red", marker="*", s=160, label="goal")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title("Planning Trajectory in XY Plane")
    axes[0].axis("equal")
    axes[0].grid(True)
    axes[0].legend()

    for index, label in enumerate(("X_hat", "Y_hat", "psi_hat")):
        axes[1].plot(result["time_state"], z_hist[:, index], label=label)
        axes[1].axhline(z_goal[index], linestyle="--", linewidth=1)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("planning state")
    axes[1].set_title("Planning State")
    axes[1].grid(True)
    axes[1].legend()

    for index, label in enumerate(("speed_hat", "omega_hat")):
        axes[2].plot(result["time_input"], v_hist[:, index], label=label)
        axes[2].axhline(result["v_min"][index], color="black", linestyle=":", linewidth=0.8)
        axes[2].axhline(result["v_max"][index], color="black", linestyle=":", linewidth=0.8)
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("planning input [m/s]")
    axes[2].set_title("Planning Input and Constraints")
    axes[2].grid(True)
    axes[2].legend()

    fig.suptitle("Dubins MPC Planning: Origin to (4 m, 3 m)")
    fig.tight_layout()
    plt.show()


def main():
    test_planning_mpc_linearization_and_prediction_shapes()
    test_planning_mpc_wraps_equivalent_goal_heading()
    test_planning_mpc_origin_to_goal()
    print("planning MPC origin-to-goal test passed")


if __name__ == "__main__":
    main()
