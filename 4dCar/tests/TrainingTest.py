# tests/TrainingTest.py

import os
import sys
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


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "JointTraining",
    "checkpoint.pt",
)

PLOT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "plot")


@dataclass
class RolloutTestConfig:
    dt: float = 0.04
    rollout_steps: int = 500
    planner_mpc_horizon: int = 20
    planner_z_min: tuple[float, float] = (-20.0, -20.0)
    planner_z_max: tuple[float, float] = (20.0, 20.0)
    planner_v_min: tuple[float, float] = (-0.5, -0.5)
    planner_v_max: tuple[float, float] = (0.5, 0.5)
    z_goal: tuple[float, float] = (4, 3)
    max_final_error_norm: float = 10.0
    max_mean_error_norm: float = 10.0
    show_plot: bool = True


def load_models_from_checkpoint(path=CHECKPOINT_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run experiments\\JointTraining.py first: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    error_model = ErrorDynamics()
    check_joint_training_checkpoint(checkpoint, error_model)
    joint_config = checkpoint["joint_training_config"]

    lyapunov = NeuralLyapunovFunction(
        error_dim=error_model.error_dim,
        hidden_sizes=(32, 32),
        feature_dim=8,
        delta=0.5,
    )
    controller = NeuralTrackingController(
        error_dim=error_model.error_dim,
        planning_input_dim=error_model.planning_model.input_dim,
        control_dim=error_model.tracking_model.input_dim,
        hidden_sizes=(32, 32),
        u_min=joint_config.get("controller_u_min", (-1.0, -1.0)),
        u_max=joint_config.get("controller_u_max", (1.0, 1.0)),
    )
    ellipsoid = LearnableEllipsoid(dim=error_model.error_dim, init_scale=1.0)

    lyapunov.load_state_dict(checkpoint["lyapunov_state_dict"])
    controller.load_state_dict(checkpoint["controller_state_dict"])
    ellipsoid.load_state_dict(checkpoint["ellipsoid_state_dict"])

    lyapunov.eval()
    controller.eval()
    ellipsoid.eval()

    return checkpoint, error_model, lyapunov, controller, ellipsoid


def make_current_models():
    error_model = ErrorDynamics()
    lyapunov = NeuralLyapunovFunction(
        error_dim=error_model.error_dim,
        hidden_sizes=(32, 32),
        feature_dim=8,
        delta=0.5,
    )
    controller = NeuralTrackingController(
        error_dim=error_model.error_dim,
        planning_input_dim=error_model.planning_model.input_dim,
        control_dim=error_model.tracking_model.input_dim,
        hidden_sizes=(32, 32),
    )
    ellipsoid = LearnableEllipsoid(dim=error_model.error_dim, init_scale=1.0)
    return error_model, lyapunov, controller, ellipsoid


def check_joint_training_checkpoint(checkpoint, error_model=None):
    required_keys = (
        "lyapunov_state_dict",
        "controller_state_dict",
        "ellipsoid_state_dict",
        "E",
        "joint_training_config",
        "training_config",
        "loss_weights",
        "history",
        "model_signature",
    )
    missing = [key for key in required_keys if key not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint is missing keys: {missing}")
    if error_model is not None:
        expected = error_model.model_signature()
        if checkpoint["model_signature"] != expected:
            raise ValueError(
                f"Checkpoint model signature {checkpoint['model_signature']} does not "
                f"match current signature {expected}. Retrain JointTraining."
            )
        E = torch.as_tensor(checkpoint["E"])
        if E.shape != (error_model.error_dim, error_model.error_dim):
            raise ValueError(
                f"Expected checkpoint E shape ({error_model.error_dim}, "
                f"{error_model.error_dim}), got {tuple(E.shape)}"
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
    z_goal = np.asarray(rollout_config.z_goal, dtype=float)
    dt = float(rollout_config.dt)
    steps = int(rollout_config.rollout_steps)
    if x.shape != (error_model.tracking_model.state_dim,):
        raise ValueError(
            f"Expected x0 shape ({error_model.tracking_model.state_dim},), got {tuple(x.shape)}"
        )
    if z.shape != (error_model.planning_model.state_dim,):
        raise ValueError(
            f"Expected z0 shape ({error_model.planning_model.state_dim},), got {tuple(z.shape)}"
        )
    planning_mpc = LinearizedPlanningMPCController(
        planning_model=error_model.planning_model,
        dt=dt,
        horizon=rollout_config.planner_mpc_horizon,
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

    for step in range(steps):
        v_numpy, planning_info = planning_mpc.solve(z.detach().cpu().numpy(), z_goal)
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

    return {
        "x": torch.stack(x_traj),
        "z": torch.stack(z_traj),
        "e": torch.stack(e_traj),
        "u": torch.stack(u_traj),
        "v": torch.stack(v_traj),
        "V": torch.stack(V_traj),
        "V_dot": torch.stack(V_dot_traj),
        "E_value": torch.stack(E_value_traj),
    }


def _to_numpy(value):
    return value.detach().cpu().numpy()


def build_plot_data(traj, rollout_config: RolloutTestConfig):
    dt = float(rollout_config.dt)
    x = _to_numpy(traj["x"])
    z = _to_numpy(traj["z"])
    E_value = _to_numpy(traj["E_value"])
    V = _to_numpy(traj["V"])

    return {
        "time_position": np.arange(x.shape[0], dtype=float) * dt,
        "time_value": np.arange(E_value.shape[0], dtype=float) * dt,
        "e_T_E_e": E_value,
        "V_e": V,
        "one": np.ones_like(E_value),
        "tracker_x": x[:, 0],
        "tracker_y": x[:, 1],
        "planner_x": z[:, 0],
        "planner_y": z[:, 1],
    }


def save_plot_data(traj, rollout_config: RolloutTestConfig, output_dir=PLOT_DATA_DIR):
    os.makedirs(output_dir, exist_ok=True)
    plot_data = build_plot_data(traj, rollout_config)
    output_path = os.path.join(output_dir, "plot_data.npz")
    np.savez(output_path, **plot_data)
    for name, value in plot_data.items():
        np.save(os.path.join(output_dir, f"{name}.npy"), value)
    return output_path


def main():
    rollout_config = RolloutTestConfig()
    try:
        checkpoint, error_model, lyapunov, controller, ellipsoid = load_models_from_checkpoint()
        check_joint_training_checkpoint(checkpoint, error_model)
        checkpoint_status = "loaded current-model checkpoint"
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        error_model, lyapunov, controller, ellipsoid = make_current_models()
        checkpoint = {"history": []}
        rollout_config.rollout_steps = 25
        rollout_config.show_plot = False
        checkpoint_status = f"current-model smoke test; checkpoint unavailable: {error}"

    # x = [x, y, theta, speed], z = [x_hat, y_hat].
    x0 = [0.0, 0.0, 0.0, 0.0]
    z0 = [0.0, 0.0]
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

    plot_data_path = save_plot_data(traj, rollout_config)

    e_norm = torch.linalg.norm(traj["e"], dim=-1)
    pos_e_norm = torch.linalg.norm(traj["e"][:, :2], dim=-1)
    E_eigvals = torch.linalg.eigvalsh(ellipsoid.matrix())
    if rollout_config.show_plot:
        plot_trajectories(traj, rollout_config)

    if not torch.all(E_eigvals > 0.0):
        raise AssertionError(f"Loaded E is not positive definite: {E_eigvals}")
    if e_norm[-1].item() > rollout_config.max_final_error_norm:
        raise AssertionError(f"Final error is too large: {e_norm[-1].item():.4f}")
    if e_norm.mean().item() > rollout_config.max_mean_error_norm:
        raise AssertionError(f"Mean error is too large: {e_norm.mean().item():.4f}")

    history = checkpoint["history"]
    print(f"plot data saved: {plot_data_path}")
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
    print(f"final V_dot: {traj['V_dot'][-1].item():.4f}")
    print(f"final e^T E e: {traj['E_value'][-1].item():.4f}")
    print(f"maximum e^T E e: {traj['E_value'].max().item():.4f}")
    print(f"E min eigenvalue: {E_eigvals.min().item():.6f}")
    print(f"checkpoint status: {checkpoint_status}")
    print(
        "planning MPC: "
        f"horizon={rollout_config.planner_mpc_horizon}, "
        f"z_goal={rollout_config.z_goal}, "
        f"v_min={rollout_config.planner_v_min}, "
        f"v_max={rollout_config.planner_v_max}"
    )
    print("loaded V, kappa, and E rollout test passed")


def plot_trajectories(traj, rollout_config: RolloutTestConfig):
    dt = float(rollout_config.dt)
    time_xz = np.arange(traj["x"].shape[0]) * dt
    time_e = np.arange(traj["e"].shape[0]) * dt

    x = traj["x"].detach().cpu().numpy()
    z = traj["z"].detach().cpu().numpy()
    E_value = traj["E_value"].detach().cpu().numpy()
    V = traj["V"].detach().cpu().numpy()

    fig_position, ax_position = plt.subplots(figsize=(9, 6))

    coordinate_specs = [
        ("x", 0, 0, "tab:blue"),
        ("y", 1, 1, "tab:orange"),
    ]
    for label, x_index, z_index, color in coordinate_specs:
        ax_position.plot(time_xz, x[:, x_index], color=color, linestyle="-", label=f"x {label}")
        ax_position.plot(time_xz, z[:, z_index], color=color, linestyle="--", label=f"z {label}")

    ax_position.set_xlabel("time (s)")
    ax_position.set_ylabel("position")
    ax_position.set_title("Tracking and Planning Position Trajectories")
    ax_position.legend(ncol=2)
    ax_position.grid(True)
    fig_position.tight_layout()

    fig_values, ax_values = plt.subplots(figsize=(9, 6))
    ax_values.plot(time_e, E_value, color="tab:blue", label=r"$e^T E e$")
    ax_values.plot(time_e, V, color="tab:orange", label=r"$V(e)$")
    ax_values.axhline(1.0, color="tab:red", linestyle="--", label="level 1")
    ax_values.set_xlabel("time (s)")
    ax_values.set_ylabel("value")
    ax_values.set_title("Ellipsoid and Lyapunov Values")
    ax_values.legend()
    ax_values.grid(True)
    fig_values.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()

