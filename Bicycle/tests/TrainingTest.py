# tests/TrainingTest.py

import os
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.TrackingControl import NeuralTrackingController
from learning.Training import LearnableEllipsoid, TrainingConfig, make_optimizer, train_step
from lyapunov.LyapunovNetwork import NeuralLyapunovFunction
from models.error import ErrorDynamics
from learning.LossFunction import moving_reference_error


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
    min_planning_speed_for_tracking: float = 4.0
    planning_input: tuple[float, float] = (8.0, 0.15)
    max_final_reference_error_norm: float = 10.0
    max_mean_reference_error_norm: float = 10.0
    show_plot: bool = False
    stop_on_nonpositive_speed: bool = True


def load_models_from_checkpoint(path=CHECKPOINT_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run experiments\\JointTraining.py first: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    check_joint_training_checkpoint(checkpoint)
    E = torch.as_tensor(checkpoint["E"])
    if E.shape != (6, 6):
        raise ValueError(
            f"Expected Bicycle checkpoint E shape (6, 6), got {tuple(E.shape)}. "
            "Run the aligned JointTraining.py first."
        )
    error_model = ErrorDynamics()
    lyapunov = NeuralLyapunovFunction(
        error_dim=error_model.error_dim,
        hidden_sizes=(64, 64),
        feature_dim=16,
        delta=0.5,
    )
    controller = NeuralTrackingController(
        hidden_sizes=(64, 64),
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
    common_required_keys = (
        "lyapunov_state_dict",
        "controller_state_dict",
        "ellipsoid_state_dict",
        "E",
        "joint_training_config",
        "training_config",
        "loss_weights",
        "history",
    )
    missing = [key for key in common_required_keys if key not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint is missing keys: {missing}")

    E = torch.as_tensor(checkpoint["E"])
    if E.shape != (6, 6):
        raise ValueError(f"Expected Bicycle checkpoint E shape (6, 6), got {tuple(E.shape)}")

    training_mode = checkpoint.get("training_mode", "certified_joint_training")
    if training_mode == "certified_joint_training":
        if checkpoint.get("certificate_coordinates") != "moving_reference_local_exponential_v2":
            raise ValueError("Checkpoint uses incompatible certificate coordinates")
        if not checkpoint.get("final_verification", {}).get("success", False):
            final_verification = checkpoint.get("final_verification", {})
            counterexample_count = len(final_verification.get("counterexamples", []))
            warnings.warn(
                "Checkpoint failed its final counterexample search "
                f"({counterexample_count} counterexamples). Loading it for diagnostic "
                "rollout and plotting only; it must not be treated as verified.",
                UserWarning,
            )
    elif training_mode == "behavior_cloning_only":
        if checkpoint.get("best_val_behavior_cloning_step") is None:
            raise ValueError(
                "Behavior-cloning checkpoint predates best-validation model recovery. "
                "Rerun experiments/JointTraining.py before evaluating it."
            )
    else:
        raise ValueError(f"Unknown checkpoint training mode: {training_mode}")


def test_bicycle_train_step():
    error_model = ErrorDynamics()
    lyapunov = NeuralLyapunovFunction()
    controller = NeuralTrackingController()
    ellipsoid = LearnableEllipsoid()
    optimizer = make_optimizer(lyapunov, controller, ellipsoid)

    batch_size = 16
    speed_hat = 5.0 + 5.0 * torch.rand(batch_size)
    omega_hat = 0.2 * torch.randn(batch_size)
    e = 0.1 * torch.randn(batch_size, error_model.error_dim)
    e[:, 3] = speed_hat + 0.5 * torch.randn(batch_size)
    e[:, 5] = omega_hat + 0.05 * torch.randn(batch_size)
    batch = {
        "e": e,
        "z": torch.randn(batch_size, error_model.planning_model.state_dim),
        "v": torch.stack([speed_hat, omega_hat], dim=-1),
        "u_expert": torch.zeros(batch_size, error_model.tracking_model.input_dim),
    }

    loss, info = train_step(
        lyapunov,
        controller,
        ellipsoid,
        error_model,
        optimizer,
        batch,
        config=TrainingConfig(),
    )

    if not torch.isfinite(loss):
        raise AssertionError(f"Expected finite training loss, got {loss}")
    if not torch.isfinite(info["grad_norm"]):
        raise AssertionError(f"Expected finite gradient norm, got {info['grad_norm']}")
    if info["E_min_eig"] <= 0.0:
        raise AssertionError(f"Expected positive-definite ellipsoid, got {info['E_min_eig']}")

    print("Bicycle train step: ok")


def test_train_step_rejects_legacy_batch_dimensions():
    error_model = ErrorDynamics()
    lyapunov = NeuralLyapunovFunction()
    controller = NeuralTrackingController()
    ellipsoid = LearnableEllipsoid()
    optimizer = make_optimizer(lyapunov, controller, ellipsoid)
    batch = {
        "e": torch.zeros(4, 10),
        "z": torch.zeros(4, 3),
        "v": torch.zeros(4, 3),
        "u_expert": torch.zeros(4, 3),
    }

    try:
        train_step(lyapunov, controller, ellipsoid, error_model, optimizer, batch)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected legacy Quadrotor batch dimensions to be rejected")

    print("legacy training batch dimensions rejected: ok")


def test_checkpoint_validation_rejects_legacy_ellipsoid():
    checkpoint = {
        "lyapunov_state_dict": {},
        "controller_state_dict": {},
        "ellipsoid_state_dict": {},
        "E": torch.eye(10),
        "joint_training_config": {},
        "training_config": {},
        "loss_weights": {},
        "history": [],
        "certificate_coordinates": "moving_reference_local_exponential_v2",
        "final_verification": {"success": False},
    }

    try:
        check_joint_training_checkpoint(checkpoint)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected legacy 10x10 checkpoint ellipsoid to be rejected")

    print("legacy checkpoint ellipsoid rejected: ok")


def generate_planning_reference(error_model, rollout_config, z0):
    z = np.asarray(z0, dtype=float)
    v = np.asarray(rollout_config.planning_input, dtype=float)
    dt = float(rollout_config.dt)

    if v.shape != (error_model.planning_model.input_dim,):
        raise ValueError(
            f"Expected planning_input shape ({error_model.planning_model.input_dim},), "
            f"got {v.shape}"
        )

    z_traj = [z.copy()]
    v_traj = []
    for _ in range(rollout_config.rollout_steps):
        v_traj.append(v.copy())
        z_t = torch.as_tensor(z, dtype=torch.float32)
        v_t = torch.as_tensor(v, dtype=torch.float32)
        z = (
            z_t + dt * error_model.planning_model.dynamics(z_t, v_t)
        ).detach().cpu().numpy()
        z_traj.append(z.copy())

    return np.asarray(z_traj), np.asarray(v_traj)


def rollout(
    error_model,
    lyapunov,
    controller,
    ellipsoid,
    rollout_config: RolloutTestConfig,
    x0,
    z0,
):
    dt = float(rollout_config.dt)
    z_reference, v_reference = generate_planning_reference(error_model, rollout_config, z0)
    z = torch.as_tensor(z_reference[0], dtype=torch.float32)
    if x0 is None:
        e0 = torch.zeros(error_model.error_dim, dtype=z.dtype)
        e0[3] = float(v_reference[0, 0])
        e0[5] = float(v_reference[0, 1])
        x = error_model.state_from_error(e0, z)
    else:
        x = torch.tensor(x0, dtype=torch.float32)

    x_traj = [x.clone()]
    z_traj = [z.clone()]
    e_traj = []
    u_traj = []
    v_traj = []
    V_traj = []
    V_dot_traj = []
    E_value_traj = []
    e_ref_traj = []

    for step, v_numpy in enumerate(v_reference):
        v = torch.as_tensor(v_numpy, dtype=z.dtype, device=z.device)
        terminate_after_step = False
        with torch.no_grad():
            e = error_model.error(x, z)
            u = controller(e, v)
            e_dot = error_model.dynamics(x, z, v, u)
            e_ref = torch.zeros_like(e)
            e_ref[3] = v[0]
            e_ref[5] = v[1]

            if torch.any(u < controller.u_min) or torch.any(u > controller.u_max):
                raise AssertionError(f"Controller output violates bounds during rollout: {u}")

            reference_error = moving_reference_error(e, v)
            V = lyapunov(reference_error)
            E_value = ellipsoid.value(reference_error)
            x = x + dt * error_model.tracking_model.dynamics(x, u)
            z = torch.as_tensor(z_reference[step + 1], dtype=z.dtype, device=z.device)
            if x[3] <= 0.0:
                if rollout_config.stop_on_nonpositive_speed:
                    raise AssertionError(f"Tracking longitudinal speed became nonpositive: {x[3]}")
                print(
                    f"rollout stopped at step {step}: tracking longitudinal speed "
                    f"became nonpositive ({x[3].item():.4f})"
                )
                terminate_after_step = True

        with torch.enable_grad():
            r_for_gradient = reference_error.detach().clone().requires_grad_(True)
            V_dot = lyapunov.lie_derivative(r_for_gradient, e_dot.detach()).detach()

        x_traj.append(x.clone())
        z_traj.append(z.clone())
        e_traj.append(e.clone())
        u_traj.append(u.clone())
        v_traj.append(v.clone())
        V_traj.append(V.clone())
        V_dot_traj.append(V_dot.clone())
        E_value_traj.append(E_value.clone())
        e_ref_traj.append(e_ref.clone())
        if terminate_after_step:
            break

    return {
        "x": torch.stack(x_traj),
        "z": torch.stack(z_traj),
        "e": torch.stack(e_traj),
        "u": torch.stack(u_traj),
        "v": torch.stack(v_traj),
        "V": torch.stack(V_traj),
        "V_dot": torch.stack(V_dot_traj),
        "E_value": torch.stack(E_value_traj),
        "E": ellipsoid.matrix().detach().clone(),
        "e_ref": torch.stack(e_ref_traj),
    }


def test_short_bicycle_rollout_is_finite():
    error_model = ErrorDynamics()
    lyapunov = NeuralLyapunovFunction()
    controller = NeuralTrackingController()
    ellipsoid = LearnableEllipsoid()
    config = RolloutTestConfig(rollout_steps=5, show_plot=False)

    traj = rollout(
        error_model,
        lyapunov,
        controller,
        ellipsoid,
        config,
        x0=[0.0, 0.0, 0.0, 8.0, 0.0, 0.0],
        z0=[0.0, 0.0, 0.0],
    )

    expected_shapes = {
        "x": (config.rollout_steps + 1, 6),
        "z": (config.rollout_steps + 1, 3),
        "e": (config.rollout_steps, 6),
        "e_ref": (config.rollout_steps, 6),
        "u": (config.rollout_steps, 2),
        "v": (config.rollout_steps, 2),
    }
    for key, shape in expected_shapes.items():
        if traj[key].shape != shape:
            raise AssertionError(f"Expected traj['{key}'] shape {shape}, got {traj[key].shape}")
        if not torch.all(torch.isfinite(traj[key])):
            raise AssertionError(f"traj['{key}'] contains non-finite values")

    print("short Bicycle rollout: ok")


def test_checkpoint_rollout(path=CHECKPOINT_PATH, show_plot=False):
    checkpoint, error_model, lyapunov, controller, ellipsoid = load_models_from_checkpoint(path)
    if checkpoint.get("training_mode") == "behavior_cloning_only":
        evaluate_behavior_cloning_checkpoint_rollout(
            checkpoint,
            error_model,
            controller,
            show_plot=show_plot,
        )
        return

    rollout_config = RolloutTestConfig(
        show_plot=show_plot,
        stop_on_nonpositive_speed=False,
    )

    x0 = [0.0, 0.0, 0.0, 8.0, 0.0, 0.0]
    z0 = [4.0, 5.0, 0.0]
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

    reference_error_norm = torch.linalg.norm(traj["e"] - traj["e_ref"], dim=-1)
    pos_e_norm = torch.linalg.norm(traj["e"][:, :2], dim=-1)
    E_eigvals = torch.linalg.eigvalsh(ellipsoid.matrix())
    ellipsoid_error_bound = 1.0 / torch.sqrt(E_eigvals.min())
    certificate_verified = checkpoint.get("final_verification", {}).get("success", False)
    save_plot_data(traj, rollout_config)

    if rollout_config.show_plot:
        plot_trajectories(traj, rollout_config)
        plot_control_inputs(traj["u"], rollout_config.dt)
        plt.show()

    if not torch.all(E_eigvals > 0.0):
        raise AssertionError(f"Loaded E is not positive definite: {E_eigvals}")
    if (
        certificate_verified
        and reference_error_norm[-1].item() > rollout_config.max_final_reference_error_norm
    ):
        raise AssertionError(
            f"Final moving-reference error is too large: {reference_error_norm[-1].item():.4f}"
        )
    if (
        certificate_verified
        and reference_error_norm.mean().item() > rollout_config.max_mean_reference_error_norm
    ):
        raise AssertionError(
            f"Mean moving-reference error is too large: {reference_error_norm.mean().item():.4f}"
        )

    history = checkpoint["history"]
    print(f"x trajectory shape: {tuple(traj['x'].shape)}")
    print(f"e trajectory shape: {tuple(traj['e'].shape)}")
    print(f"joint training recorded steps: {len(history)}")
    if history:
        print(f"last training loss: {history[-1]['total_loss']:.6f}")
        print(f"last behavior cloning loss: {history[-1]['behavior_cloning_loss']:.6f}")
    print(f"initial ||e-e_ref||: {reference_error_norm[0].item():.4f}")
    print(f"final ||e-e_ref||:   {reference_error_norm[-1].item():.4f}")
    print(f"mean ||e-e_ref||:    {reference_error_norm.mean().item():.4f}")
    print(f"final position ||e||: {pos_e_norm[-1].item():.4f}")
    print(f"final V(r): {traj['V'][-1].item():.4f}")
    print(f"maximum V(r): {traj['V'].max().item():.4f}")
    print(f"final V_dot: {traj['V_dot'][-1].item():.4f}")
    print(f"E min eigenvalue: {E_eigvals.min().item():.6f}")
    print(f"ellipsoid worst-case ||r|| bound: {ellipsoid_error_bound.item():.4f}")
    if certificate_verified:
        print(
            f"loaded {checkpoint.get('training_mode', 'certified_joint_training')} "
            "rollout test passed"
        )
    else:
        print(
            "diagnostic rollout completed; checkpoint is not verified and the "
            "rollout metrics must not be interpreted as a certificate"
        )


def evaluate_behavior_cloning_checkpoint_rollout(
    checkpoint,
    error_model,
    controller,
    show_plot=False,
):
    dt = 0.04
    steps = 250
    v = torch.tensor([8.0, 0.15])
    e_ref = torch.tensor([0.0, 0.0, 0.0, 8.0, 0.0, 0.15])
    x = torch.tensor([0.25, -0.1, 0.02, 7.8, 0.05, 0.16])
    z = torch.zeros(3)
    norms = []
    error_hist = []
    x_hist = []
    z_hist = []
    u_hist = []
    valid_domain = True
    failed_step = None

    with torch.no_grad():
        for step in range(steps):
            e = error_model.error(x, z)
            reference_error = e - e_ref
            norms.append(torch.linalg.norm(reference_error))
            error_hist.append(reference_error.clone())
            u = controller(e, v)
            x_hist.append(x.clone())
            z_hist.append(z.clone())
            u_hist.append(u.clone())
            x = x + dt * error_model.tracking_model.dynamics(x, u)
            z = z + dt * error_model.planning_model.dynamics(z, v)
            if not torch.all(torch.isfinite(x)) or x[3] <= 0.0:
                valid_domain = False
                failed_step = step
                break

    norms = torch.stack(norms)
    result = {
        "norms": norms,
        "error": torch.stack(error_hist),
        "x": torch.stack(x_hist),
        "z": torch.stack(z_hist),
        "u": torch.stack(u_hist),
        "dt": dt,
        "valid_domain": valid_domain,
        "failed_step": failed_step,
    }
    save_plot_data(result, dt)
    if show_plot:
        plot_behavior_cloning_rollout(result)
        plot_control_inputs(result["u"], result["dt"])
        plt.show()

    print(f"BC best validation loss: {checkpoint['best_val_behavior_cloning_loss']:.6f}")
    print(f"BC reference control error: {checkpoint['reference_control_error']:.6f}")
    print(f"BC in-domain initial error: {norms[0].item():.4f}")
    print(f"BC in-domain final error:   {norms[-1].item():.4f}")
    print(f"BC in-domain mean error:    {norms.mean().item():.4f}")
    print(f"BC in-domain error ratio:   {(norms[-1] / norms[0]).item():.4f}")
    print(f"BC remained in valid domain: {valid_domain}")
    if failed_step is not None:
        print(f"BC left valid Bicycle domain at step {failed_step} ({failed_step * dt:.2f} s)")
    print("loaded behavior_cloning_only in-domain rollout evaluation completed")
    return result


def save_plot_data(result, rollout_config_or_dt):
    dt = float(rollout_config_or_dt.dt if hasattr(rollout_config_or_dt, "dt") else rollout_config_or_dt)
    x = torch.as_tensor(result["x"]).detach().cpu().numpy()
    z = torch.as_tensor(result["z"]).detach().cpu().numpy()
    u = torch.as_tensor(result["u"]).detach().cpu().numpy()

    os.makedirs(PLOT_DATA_DIR, exist_ok=True)
    input_time = np.arange(u.shape[0]) * dt
    xy_time = np.arange(x.shape[0]) * dt
    nn_input = np.column_stack((input_time, u[:, 0], u[:, 1]))
    planner_tracker_xy = np.column_stack((xy_time, z[:, 0], z[:, 1], x[:, 0], x[:, 1]))

    np.savetxt(
        os.path.join(PLOT_DATA_DIR, "neural_network_input.csv"),
        nn_input,
        delimiter=",",
        header="time,delta_f,a_x",
        comments="",
    )
    np.savetxt(
        os.path.join(PLOT_DATA_DIR, "planner_tracker_xy.csv"),
        planner_tracker_xy,
        delimiter=",",
        header="time,planner_x,planner_y,tracker_x,tracker_y",
        comments="",
    )


def plot_behavior_cloning_rollout(result):
    x = result["x"].detach().cpu().numpy()
    z = result["z"].detach().cpu().numpy()
    error = result["error"].detach().cpu().numpy()
    time = np.arange(len(error)) * result["dt"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(time, x[:, 0], color="tab:blue", label="x X")
    axes[0].plot(time, z[:, 0], "--", color="tab:blue", label="z X")
    axes[0].plot(time, x[:, 1], color="tab:orange", label="x Y")
    axes[0].plot(time, z[:, 1], "--", color="tab:orange", label="z Y")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("position")
    axes[0].set_title("x and z Position Trajectories")
    axes[0].legend(ncol=2)
    axes[0].grid(True)

    error_labels = (
        "e_X",
        "e_Y",
        "e_psi",
        "e_vx",
        "e_vy",
        "e_omega",
    )
    for index, label in enumerate(error_labels):
        axes[1].plot(time, error[:, index], label=label)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("error")
    axes[1].set_title("Moving-reference Error Trajectories")
    axes[1].legend(ncol=2)
    axes[1].grid(True)

    fig.tight_layout()

def plot_trajectories(traj, rollout_config: RolloutTestConfig):
    dt = float(rollout_config.dt)
    time_xz = np.arange(traj["x"].shape[0]) * dt
    time_e = np.arange(traj["e"].shape[0]) * dt

    x = traj["x"].detach().cpu().numpy()
    z = traj["z"].detach().cpu().numpy()
    reference_error = (traj["e"] - traj["e_ref"]).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    coordinate_specs = [
        ("X", 0, 0, "tab:blue"),
        ("Y", 1, 1, "tab:orange"),
    ]
    for label, x_index, z_index, color in coordinate_specs:
        axes[0].plot(time_xz, x[:, x_index], color=color, linestyle="-", label=f"x {label}")
        axes[0].plot(time_xz, z[:, z_index], color=color, linestyle="--", label=f"z {label}")

    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("position")
    axes[0].set_title("Tracking and Planning Position Trajectories")
    axes[0].legend(ncol=2)
    axes[0].grid(True)

    error_labels = [
        "r_X",
        "r_Y",
        "r_psi",
        "r_vx = e_vx - v_hat",
        "r_vy",
        "r_omega = e_omega - omega_hat",
    ]
    for index, label in enumerate(error_labels):
        axes[1].plot(time_e, reference_error[:, index], label=label)

    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("error")
    axes[1].set_title("Moving-reference Error Trajectories (r = e - e_ref)")
    axes[1].legend(ncol=2)
    axes[1].grid(True)

    fig.tight_layout()


def plot_control_inputs(u_nn, dt):
    u_nn = torch.as_tensor(u_nn).detach().cpu().numpy()
    time = np.arange(len(u_nn)) * float(dt)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    control_specs = (
        ("Steering Input", r"$\delta_f$", 0, "tab:blue"),
        ("Longitudinal Acceleration Input", r"$a_x$", 1, "tab:orange"),
    )
    for axis, (title, label, index, color) in zip(axes, control_specs):
        axis.plot(time, u_nn[:, index], color=color, linestyle="-", label=f"NN {label}")
        axis.set_xlabel("time (s)")
        axis.set_ylabel(label)
        axis.set_title(title)
        axis.legend()
        axis.grid(True)

    fig.suptitle("Neural Network Tracking Inputs")
    fig.tight_layout()


if __name__ == "__main__":
    main()


