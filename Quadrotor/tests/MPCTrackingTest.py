# tests/MPC_test.py

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.MPCTracking import LinearizedErrorMPCController
from models.error import ErrorDynamics


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


def dynamics_numpy(model, state, control):
    state_t = to_tensor(state)
    control_t = to_tensor(control)

    with torch.no_grad():
        state_dot = model.dynamics(state_t, control_t)

    return to_numpy(state_dot)


def planner_signal(z, z_goal, gain=0.4, v_max=0.5):
    """
    Planner input v = [b_x, b_y, b_z] that drives z to a constant goal.
    """
    v = -gain * (np.asarray(z, dtype=float) - np.asarray(z_goal, dtype=float))
    return np.clip(v, -v_max, v_max)


def make_v_sequence(z0, z_goal, dt, horizon, planning_model):
    v_seq = []
    z = np.asarray(z0, dtype=float).copy()

    for _ in range(horizon):
        v = planner_signal(z, z_goal)
        v_seq.append(v)
        z = z + dt * dynamics_numpy(planning_model, z, v)

    return np.array(v_seq)


def make_controller(error_model, dt, horizon):
    Q = np.diag([5.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1, 1, 5.0, 1.0])
    R = np.diag([0.01, 0.01, 0.05])
    Qf = 10.0 * Q

    u_min = np.array([-np.deg2rad(10.0), -np.deg2rad(10.0), 0.0])
    u_max = np.array([np.deg2rad(10.0), np.deg2rad(10.0), 1.5 * error_model.tracking_model.g])

    return LinearizedErrorMPCController(
        error_model=error_model,
        dt=dt,
        horizon=horizon,
        Q=Q,
        R=R,
        Qf=Qf,
        u_min=u_min,
        u_max=u_max,
    )


def simulate_closed_loop():
    dt = 0.05
    horizon = 20
    T_sim = 15.0
    steps = int(T_sim / dt)

    error_model = ErrorDynamics()
    planning_model = error_model.planning_model
    tracking_model = error_model.tracking_model
    mpc = make_controller(error_model, dt, horizon)

    z = np.array([0.0, 0.0, 0.0], dtype=float)
    z_goal = np.array([0.8, 1.0, 0.5], dtype=float)

    # x = [x, vx, theta_x, omega_x, y, vy, theta_y, omega_y, z, vz]
    x = np.array([0.1, 0.0, 0.02, 0.0, -0.1, 0.0, -0.02, 0.0, 0.2, 0.0], dtype=float)

    time = np.arange(steps + 1) * dt

    x_hist = [x.copy()]
    z_hist = [z.copy()]
    e_hist = [to_numpy(error_model.error(to_tensor(x), to_tensor(z)))]
    u_hist = []
    v_hist = []
    cost_hist = []
    success_hist = []
    status_hist = []

    for i in range(steps):
        v_seq = make_v_sequence(z, z_goal, dt, horizon, planning_model)

        e = to_numpy(error_model.error(to_tensor(x), to_tensor(z)))
        u, info = mpc.solve(e, z, v_seq)

        v = v_seq[0]
        z = z + dt * dynamics_numpy(planning_model, z, v)
        x = x + dt * dynamics_numpy(tracking_model, x, u)

        e_next = to_numpy(error_model.error(to_tensor(x), to_tensor(z)))

        x_hist.append(x.copy())
        z_hist.append(z.copy())
        e_hist.append(e_next.copy())
        u_hist.append(u.copy())
        v_hist.append(v.copy())
        cost_hist.append(info["cost"])
        success_hist.append(info["success"])
        status_hist.append(info["status"])

        if i % 10 == 0:
            print(
                f"step={i:04d}, "
                f"||e||={np.linalg.norm(e_next):.4f}, "
                f"||e_pos||={np.linalg.norm(e_next[[0, 4, 8]]):.4f}, "
                f"u={u}, "
                f"status={info['status']}"
            )

        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(e_next)):
            print(f"Stopping early at step {i}: non-finite state or error.")
            break

    result = {
        "time_state": time[: len(x_hist)],
        "time_input": time[: len(u_hist)],
        "x_hist": np.array(x_hist),
        "z_hist": np.array(z_hist),
        "e_hist": np.array(e_hist),
        "u_hist": np.array(u_hist),
        "v_hist": np.array(v_hist),
        "cost_hist": np.array(cost_hist),
        "success_hist": np.array(success_hist, dtype=bool),
        "status_hist": status_hist,
        "dt": dt,
        "horizon": horizon,
        "z_goal": z_goal,
    }

    return result


def plot_results(result):
    time_state = result["time_state"]
    x_hist = result["x_hist"]
    z_hist = result["z_hist"]
    e_hist = result["e_hist"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    coordinate_styles = [
        ("x", x_hist[:, 0], z_hist[:, 0], "tab:blue"),
        ("y", x_hist[:, 4], z_hist[:, 1], "tab:orange"),
        ("z", x_hist[:, 8], z_hist[:, 2], "tab:green"),
    ]
    for coordinate, tracking_value, planning_value, color in coordinate_styles:
        axes[0].plot(
            time_state,
            tracking_value,
            color=color,
            linestyle="-",
            label=f"tracking {coordinate}",
        )
        axes[0].plot(
            time_state,
            planning_value,
            color=color,
            linestyle="--",
            label=f"planning {coordinate}",
        )
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("position")
    axes[0].set_title("Tracking vs Planning Position")
    axes[0].legend(ncol=2)
    axes[0].grid(True)

    labels = [
        "e_x",
        "e_vx",
        "e_theta_x",
        "e_omega_x",
        "e_y",
        "e_vy",
        "e_theta_y",
        "e_omega_y",
        "e_z",
        "e_vz",
    ]
    for j, label in enumerate(labels):
        axes[1].plot(time_state, e_hist[:, j], label=label)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("error")
    axes[1].set_title("Error State")
    axes[1].legend(ncol=2)
    axes[1].grid(True)

    fig.suptitle("Linearized Error MPC")
    fig.tight_layout()

    plt.show()


def main():
    result = simulate_closed_loop()
    success_hist = result["success_hist"]

    if len(success_hist) > 0:
        print(f"MPC success rate: {100.0 * np.mean(success_hist):.1f}%")
        print(f"Final error norm: {np.linalg.norm(result['e_hist'][-1]):.4f}")
        print(
            "Final position error norm: "
            f"{np.linalg.norm(result['e_hist'][-1, [0, 4, 8]]):.4f}"
        )

    plot_results(result)


if __name__ == "__main__":
    main()
