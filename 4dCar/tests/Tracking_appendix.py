"""Compare ROM planning trajectories with NN-controlled FOM trajectories.

Three start/goal pairs are sampled from the intervals stored in the expert
dataset.  The reduced-order model (ROM) is driven by the planning MPC, while
the full-order model (FOM) is driven by the trained neural tracking controller.
"""

import json
import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from controllers.MPCPlanning import LinearizedPlanningMPCController
from controllers.TrackingControl import NeuralTrackingController
from models.error import ErrorDynamics


CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, "results", "JointTraining", "checkpoint.pt"
)
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "expert_dataset.npz")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "Tracking_appendix.pdf")


@dataclass
class AppendixConfig:
    seed: int = 7
    num_trials: int = 3
    dt: float = 0.04
    rollout_seconds: float = 14.0
    planner_horizon: int = 20
    minimum_goal_distance: float = 1.5
    initial_position_error_limit: float = 0.4
    minimum_initial_position_error: float = 0.05
    show_plot: bool = True


def load_dataset_config(path: str = DATASET_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expert dataset not found: {path}")

    with np.load(path, allow_pickle=False) as dataset:
        if "config_json" not in dataset:
            raise KeyError("Expert dataset does not contain 'config_json'")
        return json.loads(str(dataset["config_json"].reshape(())))


def load_trained_controller(
    error_model: ErrorDynamics,
    path: str = CHECKPOINT_PATH,
) -> NeuralTrackingController:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run experiments\\JointTraining.py first: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected_signature = error_model.model_signature()
    if checkpoint.get("model_signature") != expected_signature:
        raise ValueError(
            "Checkpoint model signature does not match the current models. "
            "Regenerate the dataset and retrain the controller."
        )

    joint_config = checkpoint["joint_training_config"]
    controller = NeuralTrackingController(
        error_dim=error_model.error_dim,
        planning_input_dim=error_model.planning_model.input_dim,
        control_dim=error_model.tracking_model.input_dim,
        hidden_sizes=(32, 32),
        u_min=joint_config.get("controller_u_min", (-1.0, -1.0)),
        u_max=joint_config.get("controller_u_max", (1.0, 1.0)),
    )
    controller.load_state_dict(checkpoint["controller_state_dict"])
    controller.eval()
    return controller


def sample_start_goal_pairs(
    rng: np.random.Generator,
    dataset_config: dict,
    config: AppendixConfig,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    z0_low = np.asarray(dataset_config["z0_low"], dtype=float)
    z0_high = np.asarray(dataset_config["z0_high"], dtype=float)
    goal_low = np.asarray(dataset_config["z_goal_low"], dtype=float)
    goal_high = np.asarray(dataset_config["z_goal_high"], dtype=float)

    pairs = []
    while len(pairs) < config.num_trials:
        z0 = rng.uniform(z0_low, z0_high)
        goal = rng.uniform(goal_low, goal_high)
        position_error = rng.uniform(
            -config.initial_position_error_limit,
            config.initial_position_error_limit,
            size=2,
        )
        sufficiently_separated = np.all(
            np.abs(position_error) >= config.minimum_initial_position_error
        )
        if (
            np.linalg.norm(goal - z0) >= config.minimum_goal_distance
            and sufficiently_separated
        ):
            pairs.append((z0, goal, position_error))
    return pairs


def rollout_trial(
    error_model: ErrorDynamics,
    controller: NeuralTrackingController,
    z0: np.ndarray,
    goal: np.ndarray,
    initial_position_error: np.ndarray,
    dataset_config: dict,
    config: AppendixConfig,
) -> dict[str, np.ndarray]:
    planning_mpc = LinearizedPlanningMPCController(
        planning_model=error_model.planning_model,
        dt=config.dt,
        horizon=config.planner_horizon,
        z_min=dataset_config["planner_z_min"],
        z_max=dataset_config["planner_z_max"],
        v_min=dataset_config["planner_v_min"],
        v_max=dataset_config["planner_v_max"],
    )

    z = torch.as_tensor(z0, dtype=torch.float32)
    # Begin with a bounded position offset and a stationary FOM.
    x = error_model.projection(z).clone()
    x[:2] += torch.as_tensor(initial_position_error, dtype=x.dtype)
    steps = int(round(config.rollout_seconds / config.dt))
    z_history = [z.detach().cpu().numpy().copy()]
    x_history = [x.detach().cpu().numpy().copy()]

    for step in range(steps):
        v_numpy, info = planning_mpc.solve(z.detach().cpu().numpy(), goal)
        if not info["success"]:
            raise RuntimeError(
                f"Planning MPC failed at step {step}: {info['status']}"
            )

        v = torch.as_tensor(v_numpy, dtype=torch.float32)
        with torch.no_grad():
            e = error_model.error(x, z)
            u = controller(e, v)
            x = error_model.tracking_model.step(x, u, config.dt)
            z = error_model.planning_model.step(z, v, config.dt)

        x_history.append(x.detach().cpu().numpy().copy())
        z_history.append(z.detach().cpu().numpy().copy())

    return {
        "x": np.asarray(x_history),
        "z": np.asarray(z_history),
        "goal": np.asarray(goal),
    }


def plot_tracking_results(
    trajectories: list[dict[str, np.ndarray]],
    config: AppendixConfig,
    output_path: str = OUTPUT_PATH,
):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.titlesize": 18,
            "axes.labelsize": 19,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 11,
        }
    )
    time = np.arange(trajectories[0]["x"].shape[0]) * config.dt
    colors = plt.get_cmap("tab10").colors
    fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(16, 5), sharex=True)

    for index, trajectory in enumerate(trajectories):
        color = colors[index]
        case = index + 1
        ax_x.plot(
            time,
            trajectory["x"][:, 0],
            color=color,
            linewidth=2.4,
            label=rf"$p_x$, case {case}",
        )
        ax_x.plot(
            time,
            trajectory["z"][:, 0],
            color=color,
            linestyle="--",
            linewidth=2.4,
            label=rf"$\hat{{p}}_x$, case {case}",
        )
        ax_y.plot(
            time,
            trajectory["x"][:, 1],
            color=color,
            linewidth=2.4,
            label=rf"$p_y$, case {case}",
        )
        ax_y.plot(
            time,
            trajectory["z"][:, 1],
            color=color,
            linestyle="--",
            linewidth=2.4,
            label=rf"$\hat{{p}}_y$, case {case}",
        )

    ax_x.set_title(r"Tracking in the $x$ direction")
    ax_y.set_title(r"Tracking in the $y$ direction")
    for axis in (ax_x, ax_y):
        axis.set_xlim(0.0, config.rollout_seconds)
        axis.margins(x=0.0)
        axis.set_xlabel("Time (sec)")
        axis.set_ylabel("Position")
        axis.xaxis.set_major_locator(MultipleLocator(2.0))
        axis.xaxis.set_minor_locator(AutoMinorLocator(2))
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))
        axis.grid(which="major", color="#9a9a9a", linewidth=0.8, alpha=0.55)
        axis.grid(
            which="minor",
            color="#bdbdbd",
            linewidth=0.55,
            linestyle=":",
            alpha=0.45,
        )
        axis.tick_params(which="major", length=5, width=0.9)
        axis.tick_params(which="minor", length=3, width=0.7)
        axis.legend(ncol=2, frameon=True)

    ax_x.set_ylim(-3.0, 3.0)
    ax_y.set_ylim(-3.0, 2.0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if config.show_plot:
        plt.show()
    else:
        plt.close(fig)
    return fig


def main() -> None:
    config = AppendixConfig()
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    dataset_config = load_dataset_config()
    error_model = ErrorDynamics()
    controller = load_trained_controller(error_model)
    pairs = sample_start_goal_pairs(rng, dataset_config, config)

    trajectories = []
    for case, (z0, goal, initial_position_error) in enumerate(pairs, start=1):
        trajectory = rollout_trial(
            error_model,
            controller,
            z0,
            goal,
            initial_position_error,
            dataset_config,
            config,
        )
        trajectories.append(trajectory)
        final_position_error = np.linalg.norm(
            trajectory["x"][-1, :2] - trajectory["z"][-1, :2]
        )
        print(
            f"case {case}: start={z0}, goal={goal}, "
            f"initial position error={initial_position_error}, "
            f"final position tracking error={final_position_error:.6f}"
        )

    plot_tracking_results(trajectories, config)
    print(f"saved tracking figure: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
