# tests/TrainingTest.py

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.MPCPlanning import LinearizedPlanningMPCController
from controllers.TrackingControl import NeuralTrackingController
from learning.Training import LearnableEllipsoid
from lyapunov.LyapunovNetwork import NeuralLyapunovFunction
from models.error import ErrorDynamics


CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "JointTraining",
    "checkpoint.pt",
)
PLOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "plot",
)

ERROR_LABELS = (
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
)
POSITION_INDICES = (0, 4, 8)
POSITION_LABELS = ("e_x", "e_y", "e_z")


@dataclass
class RolloutTestConfig:
    dt: float = 0.04
    rollout_steps: int = 3000
    planner_mpc_horizon: int = 20
    planner_z_min: tuple[float, float, float] = (-21, -17, 0)
    planner_z_max: tuple[float, float, float] = (32, 27, 22)
    planner_v_min: tuple[float, float, float] = (-0.5, -0.5, -0.5)
    planner_v_max: tuple[float, float, float] = (0.5, 0.5, 0.5)
    planner_v_rate_weight: float = 4.0
    planner_reference_duration: float = 95.0
    planner_reference_time_lookahead: float = 2.0
    z_goal: tuple[float, float, float] = (30, 25, 20)
    max_final_error_norm: float = 10.0
    max_mean_error_norm: float = 10.0
    progress_every: int = 100
    show_plot: bool = True
    save_plot_data: bool = True
    plot_dir: str = PLOT_DIR


def load_models_from_checkpoint(path=CHECKPOINT_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run experiments\\JointTraining.py first: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    error_model = ErrorDynamics()
    tracking_model = error_model.tracking_model

    lyapunov = NeuralLyapunovFunction(
        error_dim=error_model.error_dim,
        hidden_sizes=(64, 64),
        feature_dim=16,
        delta=0.02,
    )
    controller = NeuralTrackingController(
        hidden_sizes=(64, 64),
        g=tracking_model.g,
        kT=tracking_model.kT,
    )
    ellipsoid = LearnableEllipsoid(dim=error_model.error_dim, init_scale=1.0)

    lyapunov.load_state_dict(checkpoint["lyapunov_state_dict"])
    controller.load_state_dict(checkpoint["controller_state_dict"])
    ellipsoid.load_state_dict(checkpoint["ellipsoid_state_dict"])

    lyapunov.eval()
    controller.eval()
    ellipsoid.eval()

    return checkpoint, error_model, lyapunov, controller, ellipsoid


def check_joint_training_checkpoint(checkpoint):
    required_keys = (
        "lyapunov_state_dict",
        "controller_state_dict",
        "ellipsoid_state_dict",
        "E",
        "joint_training_config",
        "training_config",
        "loss_weights",
        "history",
    )
    missing = [key for key in required_keys if key not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint is missing keys: {missing}")


def projected_position_bounds(E: torch.Tensor) -> torch.Tensor:
    selector = torch.zeros(3, E.shape[0], dtype=E.dtype, device=E.device)
    for row, index in enumerate(POSITION_INDICES):
        selector[row, index] = 1.0
    position_shape = selector @ torch.linalg.solve(E, selector.T)
    return torch.sqrt(torch.diagonal(position_shape))


def smoothstep_reference_target(
    time: float,
    z_start: np.ndarray,
    z_goal: np.ndarray,
    duration: float,
) -> np.ndarray:
    if duration <= 1e-8:
        return z_goal.copy()

    tau = float(np.clip(time / duration, 0.0, 1.0))
    alpha = tau * tau * (3.0 - 2.0 * tau)
    return z_start + alpha * (z_goal - z_start)


def print_rollout_diagnostics(traj: dict[str, torch.Tensor], ellipsoid: LearnableEllipsoid) -> None:
    E = ellipsoid.matrix().detach()
    E_inv = torch.linalg.inv(E)
    axis_bounds = torch.sqrt(torch.diagonal(E_inv))
    position_bounds = projected_position_bounds(E)
    eigvals = torch.linalg.eigvalsh(E)
    e_abs_max = traj["e"].detach().abs().max(dim=0).values
    e_abs_p90 = torch.quantile(traj["e"].detach().abs(), 0.9, dim=0)

    print(
        f"E spectrum: min={eigvals.min().item():.6f}, "
        f"max={eigvals.max().item():.6f}, "
        f"condition={eigvals.max().item() / eigvals.min().item():.3f}"
    )
    print("projected position bounds from e^T E e <= 1:")
    for label, bound in zip(POSITION_LABELS, position_bounds):
        print(f"  |{label}| <= {bound.item():.6f}")

    print("rollout error coverage relative to full-state axis bounds:")
    for label, p90_i, max_i, bound_i in zip(
        ERROR_LABELS,
        e_abs_p90,
        e_abs_max,
        axis_bounds,
    ):
        print(
            f"  {label:12s} p90={p90_i.item():.6f}, "
            f"max={max_i.item():.6f}, "
            f"axis_bound={bound_i.item():.6f}, "
            f"max/bound={(max_i / bound_i).item():.3f}"
        )


def rollout(
    error_model,
    lyapunov,
    controller,
    ellipsoid,
    rollout_config: RolloutTestConfig,
    x0,
    z0,
):
    x = torch.tensor(x0, dtype=torch.float32)
    z = torch.tensor(z0, dtype=torch.float32)
    z_start = np.asarray(z0, dtype=float)
    z_goal = np.asarray(rollout_config.z_goal, dtype=float)
    dt = float(rollout_config.dt)
    steps = int(rollout_config.rollout_steps)
    planning_mpc = LinearizedPlanningMPCController(
        planning_model=error_model.planning_model,
        dt=dt,
        horizon=rollout_config.planner_mpc_horizon,
        Rd=rollout_config.planner_v_rate_weight * np.eye(3),
        z_min=rollout_config.planner_z_min,
        z_max=rollout_config.planner_z_max,
        v_min=rollout_config.planner_v_min,
        v_max=rollout_config.planner_v_max,
    )

    x_traj = [x.clone()]
    z_traj = [z.clone()]
    e_traj = []
    u_traj = []
    v_traj = []
    V_traj = []
    V_dot_traj = []
    E_value_traj = []
    start_time = time.perf_counter()

    for step in range(steps):
        z_numpy = z.detach().cpu().numpy()
        z_target = smoothstep_reference_target(
            time=(step * dt) + rollout_config.planner_reference_time_lookahead,
            z_start=z_start,
            z_goal=z_goal,
            duration=rollout_config.planner_reference_duration,
        )

        v_numpy, planning_info = planning_mpc.solve(z_numpy, z_target)
        if not planning_info["success"]:
            raise AssertionError(
                f"Planning MPC failed during rollout at step {step}: "
                f"{planning_info['status']}"
            )
        v = torch.as_tensor(v_numpy, dtype=z.dtype, device=z.device)

        with torch.no_grad():
            e = error_model.error(x, z)
            u = controller(e, v)
            e_dot = error_model.dynamics(x, z, v, u)

            if torch.any(u < controller.u_min) or torch.any(u > controller.u_max):
                raise AssertionError(f"Controller output violates bounds during rollout: {u}")

            V = lyapunov(e)
            E_value = ellipsoid.value(e)
            x = x + dt * error_model.tracking_model.dynamics(x, u)
            z = z + dt * error_model.planning_model.dynamics(z, v)

        with torch.enable_grad():
            e_for_gradient = e.detach().clone().requires_grad_(True)
            V_dot = lyapunov.lie_derivative(e_for_gradient, e_dot.detach()).detach()

        x_traj.append(x.clone())
        z_traj.append(z.clone())
        e_traj.append(e.clone())
        u_traj.append(u.clone())
        v_traj.append(v.clone())
        V_traj.append(V.clone())
        V_dot_traj.append(V_dot.clone())
        E_value_traj.append(E_value.clone())

        completed = step + 1
        if rollout_config.progress_every and (
            completed % rollout_config.progress_every == 0 or completed == steps
        ):
            elapsed = time.perf_counter() - start_time
            steps_per_second = completed / elapsed
            eta_seconds = (steps - completed) / steps_per_second
            print(
                f"rollout {completed}/{steps}, "
                f"time={completed * dt:.1f}s, "
                f"elapsed={elapsed:.1f}s, ETA={eta_seconds:.1f}s",
                flush=True,
            )

    return {
        "x": torch.stack(x_traj),
        "z": torch.stack(z_traj),
        "e": torch.stack(e_traj),
        "u": torch.stack(u_traj),
        "v": torch.stack(v_traj),
        "V": torch.stack(V_traj),
        "V_dot": torch.stack(V_dot_traj),
        "E_value": torch.stack(E_value_traj),
        "position_bounds": projected_position_bounds(ellipsoid.matrix()).detach(),
    }


def main():
    checkpoint, error_model, lyapunov, controller, ellipsoid = load_models_from_checkpoint()
    check_joint_training_checkpoint(checkpoint)
    rollout_config = RolloutTestConfig()

    # x0 = [0.2, 0.0, 0.03, 0.0, -0.2, 0.0, -0.03, 0.0, 0.1, 0.0]
    x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    z0 = [0.0, 0.0, 0.0]
    traj = rollout(
        error_model=error_model,
        lyapunov=lyapunov,
        controller=controller,
        ellipsoid=ellipsoid,
        rollout_config=rollout_config,
        x0=x0,
        z0=z0,
    )

    for name, value in traj.items():
        if not torch.all(torch.isfinite(value)):
            raise AssertionError(f"{name} trajectory contains non-finite values")

    e_norm = torch.linalg.norm(traj["e"], dim=-1)
    pos_e_norm = torch.linalg.norm(traj["e"][:, [0, 4, 8]], dim=-1)
    E_eigvals = torch.linalg.eigvalsh(ellipsoid.matrix())

    if not torch.all(E_eigvals > 0.0):
        raise AssertionError(f"Loaded E is not positive definite: {E_eigvals}")
    if e_norm[-1].item() > rollout_config.max_final_error_norm:
        raise AssertionError(f"Final error is too large: {e_norm[-1].item():.4f}")
    if e_norm.mean().item() > rollout_config.max_mean_error_norm:
        raise AssertionError(f"Mean error is too large: {e_norm.mean().item():.4f}")

    history = checkpoint["history"]
    print(f"x trajectory shape: {tuple(traj['x'].shape)}")
    print(f"e trajectory shape: {tuple(traj['e'].shape)}")
    print(f"joint training recorded steps: {len(history)}")
    if history:
        print(f"last training loss: {history[-1]['total_loss']:.6f}")
        print(f"last behavior cloning loss: {history[-1]['behavior_cloning_loss']:.6f}")
    print(f"initial ||e||: {e_norm[0].item():.4f}")
    print(f"final ||e||:   {e_norm[-1].item():.4f}")
    print(f"mean ||e||:    {e_norm.mean().item():.4f}")
    print(f"final position ||e||: {pos_e_norm[-1].item():.4f}")
    print(f"final V(e): {traj['V'][-1].item():.4f}")
    print(f"90th percentile V(e): {torch.quantile(traj['V'], 0.9).item():.4f}")
    print(f"maximum V(e): {traj['V'].max().item():.4f}")
    print(f"final V_dot: {traj['V_dot'][-1].item():.4f}")
    print(f"final e^T E e: {traj['E_value'][-1].item():.4f}")
    print(
        f"90th percentile e^T E e: "
        f"{torch.quantile(traj['E_value'], 0.9).item():.4f}"
    )
    print(f"maximum e^T E e: {traj['E_value'].max().item():.4f}")
    alignment_gap = traj["V"] - traj["E_value"]
    alignment_violation = alignment_gap > 0.0
    print(f"maximum V(e) - e^T E e: {alignment_gap.max().item():.4f}")
    print(
        "fraction with V(e) > e^T E e: "
        f"{alignment_violation.to(dtype=torch.float32).mean().item():.3f}"
    )
    nonzero_V = traj["V"] > 1e-6
    if torch.any(nonzero_V):
        bound_to_V = traj["E_value"][nonzero_V] / traj["V"][nonzero_V]
        print(f"median (e^T E e) / V(e): {torch.median(bound_to_V).item():.4f}")
        print(
            f"90th percentile (e^T E e) / V(e): "
            f"{torch.quantile(bound_to_V, 0.9).item():.4f}"
        )
    print(f"E min eigenvalue: {E_eigvals.min().item():.6f}")
    print_rollout_diagnostics(traj, ellipsoid)
    print(
        "planning MPC: "
        f"horizon={rollout_config.planner_mpc_horizon}, "
        f"z_goal={rollout_config.z_goal}, "
        f"v_min={rollout_config.planner_v_min}, "
        f"v_max={rollout_config.planner_v_max}"
    )
    print("loaded V, kappa, and E rollout test passed")

    if rollout_config.save_plot_data:
        save_plot_data(traj, rollout_config)

    if rollout_config.show_plot:
        plot_trajectories(traj, rollout_config)


def save_plot_data(traj, rollout_config: RolloutTestConfig):
    os.makedirs(rollout_config.plot_dir, exist_ok=True)

    dt = float(rollout_config.dt)
    x = traj["x"].detach().cpu().numpy()
    z = traj["z"].detach().cpu().numpy()
    e = traj["e"].detach().cpu().numpy()
    u = traj["u"].detach().cpu().numpy()
    V = traj["V"].detach().cpu().numpy()
    E_value = traj["E_value"].detach().cpu().numpy()
    position_bounds = traj["position_bounds"].detach().cpu().numpy()

    time_state = np.arange(x.shape[0]) * dt
    time_sample = np.arange(e.shape[0]) * dt
    tracker_xyz = x[:, [0, 4, 8]]
    planner_xyz = z
    position_error = e[:, [0, 4, 8]]

    np.savez_compressed(
        os.path.join(rollout_config.plot_dir, "training_plot_data.npz"),
        time_state=time_state,
        time_sample=time_sample,
        tracker_xyz=tracker_xyz,
        planner_xyz=planner_xyz,
        position_error=position_error,
        position_error_bounds=position_bounds,
        V=V,
        E_value=E_value,
        tracker_control_inputs=u,
    )

    np.savetxt(
        os.path.join(rollout_config.plot_dir, "training_position_trajectories.csv"),
        np.column_stack([time_state, tracker_xyz, planner_xyz]),
        delimiter=",",
        header="time,tracker_x,tracker_y,tracker_z,planner_x,planner_y,planner_z",
        comments="",
    )
    np.savetxt(
        os.path.join(rollout_config.plot_dir, "training_position_errors.csv"),
        np.column_stack(
            [
                time_sample,
                position_error,
                np.tile(position_bounds, (time_sample.shape[0], 1)),
                -np.tile(position_bounds, (time_sample.shape[0], 1)),
            ]
        ),
        delimiter=",",
        header=(
            "time,e_x,e_y,e_z,"
            "bound_x,bound_y,bound_z,"
            "neg_bound_x,neg_bound_y,neg_bound_z"
        ),
        comments="",
    )
    np.savetxt(
        os.path.join(rollout_config.plot_dir, "training_ellipsoid_lyapunov.csv"),
        np.column_stack([time_sample, V, E_value]),
        delimiter=",",
        header="time,V,E_value",
        comments="",
    )
    np.savetxt(
        os.path.join(rollout_config.plot_dir, "training_tracker_control_inputs.csv"),
        np.column_stack([time_sample, u]),
        delimiter=",",
        header="time,u_x,u_y,u_z",
        comments="",
    )

    print(f"saved TrainingTest plot data to: {rollout_config.plot_dir}")


def plot_trajectories(traj, rollout_config: RolloutTestConfig):
    dt = float(rollout_config.dt)
    time_xz = np.arange(traj["x"].shape[0]) * dt
    time_e = np.arange(traj["e"].shape[0]) * dt

    x = traj["x"].detach().cpu().numpy()
    z = traj["z"].detach().cpu().numpy()
    e = traj["e"].detach().cpu().numpy()
    E_value = traj["E_value"].detach().cpu().numpy()
    V = traj["V"].detach().cpu().numpy()
    u = traj["u"].detach().cpu().numpy()

    fig_position, ax_position = plt.subplots(1, 1, figsize=(9, 5))

    coordinate_specs = [
        ("x", 0, 0, "tab:blue"),
        ("y", 4, 1, "tab:orange"),
        ("z", 8, 2, "tab:green"),
    ]
    for label, x_index, z_index, color in coordinate_specs:
        ax_position.plot(time_xz, x[:, x_index], color=color, linestyle="-", label=f"x {label}")
        ax_position.plot(time_xz, z[:, z_index], color=color, linestyle="--", label=f"z {label}")

    ax_position.set_xlabel("time (s)")
    ax_position.set_ylabel("position")
    ax_position.set_title("Position Trajectories")
    ax_position.legend(ncol=2)
    ax_position.grid(True)
    fig_position.tight_layout()

    fig_error, ax_error = plt.subplots(1, 1, figsize=(9, 5))
    position_error_specs = [
        ("e_x", 0, "tab:blue"),
        ("e_y", 4, "tab:orange"),
        ("e_z", 8, "tab:green"),
    ]
    position_bounds = traj["position_bounds"].detach().cpu().numpy()
    for bound_index, (label, error_index, color) in enumerate(position_error_specs):
        ax_error.plot(time_e, e[:, error_index], color=color, linewidth=1.8, label=label)
        bound = float(position_bounds[bound_index])
        ax_error.axhline(bound, color=color, linestyle="--", linewidth=1.0, alpha=0.75)
        ax_error.axhline(-bound, color=color, linestyle="--", linewidth=1.0, alpha=0.75)

    ax_error.set_xlabel("time (s)")
    ax_error.set_ylabel("position error")
    ax_error.set_title("Position Error Trajectories with Error Bounds")
    ax_error.legend(ncol=3)
    ax_error.grid(True)
    fig_error.tight_layout()

    fig2, ax2 = plt.subplots(1, 1, figsize=(9, 5))

    V_display = np.minimum(V, E_value)
    ax2.plot(time_e, E_value, color="tab:blue", label=r"$e^T E e$")
    ax2.plot(time_e, V_display, color="tab:orange", label=r"$V(e)$")
    ax2.axhline(1.0, color="tab:red", linestyle="--", label="level 1")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("value")
    ax2.set_title("Ellipsoid and Lyapunov Values")
    ax2.legend()
    ax2.grid(True)

    fig2.tight_layout()

    fig3, axes3 = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    control_labels = [r"$u_x$", r"$u_y$", r"$u_z$"]
    for index, label in enumerate(control_labels):
        axes3[index].plot(time_e, u[:, index], label=label)
        axes3[index].set_ylabel(label)
        axes3[index].legend()
        axes3[index].grid(True)

    axes3[-1].set_xlabel("time (s)")
    fig3.suptitle("Control Input Trajectories")

    fig3.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
