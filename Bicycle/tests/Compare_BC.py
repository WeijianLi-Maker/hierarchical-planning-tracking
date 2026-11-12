"""A controlled ablation study for the behavior-cloning loss.

The two variants have identical initialization, data batches, certificate
losses, and optimizer settings.  Their only difference is the coefficient of
L_BC.  The figure compares a common validation BC loss and closed-loop planar
position tracking error.
"""

import copy
import os
import sys
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from experiments.JointTraining import (
    JointTrainingConfig,
    add_equilibrium_anchors,
    load_expert_dataset,
    make_models,
    split_dataset,
)
from learning.LossFunction import LossWeights
from learning.Training import TrainingConfig, make_optimizer, train_step
from models.error import ErrorDynamics


OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "Compare_BC.pdf")
CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, "results", "CompareBC", "checkpoint.pt"
)


@dataclass
class CompareBCConfig:
    seed: int = 11
    steps: int = 1500
    batch_size: int = 200
    evaluation_every: int = 25
    learning_rate: float = 1e-3
    lyapunov_weight: float = 4.0
    error_bound_weight: float = 2.0
    logdet_weight: float = 1e-3
    equilibrium_anchor_fraction: float = 0.25
    dt: float = 0.04
    rollout_seconds: float = 12.0
    planning_input: tuple[float, float] = (8.0, 0.15)
    initial_reference_error: tuple[float, ...] = (
        0.30,
        -0.20,
        0.02,
        0.20,
        0.05,
        -0.02,
    )
    minimum_tracking_speed: float = 1.0
    device: str = "cpu"
    show_plot: bool = True


def validation_bc_mse(
    controller,
    dataset: dict[str, torch.Tensor],
    device: str,
) -> float:
    controller.eval()
    with torch.no_grad():
        e = dataset["e"].to(device)
        v = dataset["v"].to(device)
        u_expert = dataset["u_expert"].to(device)
        u_pred = controller(e, v)
        return float(torch.mean(torch.sum((u_pred - u_expert).square(), dim=-1)).cpu())


def make_identical_model_pairs(
    error_model: ErrorDynamics,
    train_data: dict[str, torch.Tensor],
    config: CompareBCConfig,
):
    torch.manual_seed(config.seed)
    base_models = make_models(error_model, config.device)
    base_lyapunov, base_controller, base_ellipsoid = base_models

    features = torch.cat([train_data["e"], train_data["v"]], dim=-1)
    base_controller.set_feature_normalization(
        features.mean(dim=0),
        features.std(dim=0),
        min_scale=1e-2,
    )

    with_bc = (
        copy.deepcopy(base_lyapunov),
        copy.deepcopy(base_controller),
        copy.deepcopy(base_ellipsoid),
    )
    without_bc = (
        copy.deepcopy(base_lyapunov),
        copy.deepcopy(base_controller),
        copy.deepcopy(base_ellipsoid),
    )
    return with_bc, without_bc


def train_ablation(config: CompareBCConfig):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    data_config = JointTrainingConfig(
        data_path="data/expert_dataset_bc.npz",
        behavior_cloning_only=False,
        require_closed_loop_dataset=True,
    )
    dataset = load_expert_dataset(data_config)
    train_data, val_data = split_dataset(dataset, data_config.train_fraction, config.seed)
    error_model = ErrorDynamics()
    with_bc, without_bc = make_identical_model_pairs(error_model, train_data, config)

    training_config = TrainingConfig(
        learning_rate=config.learning_rate,
        gradient_clip_norm=10.0,
        level=1.0,
        decrease_margin=0.01,
        device=config.device,
    )
    optimizer_with = make_optimizer(*with_bc, training_config)
    optimizer_without = make_optimizer(*without_bc, training_config)
    shared_weights = dict(
        lyapunov=config.lyapunov_weight,
        error_bound=config.error_bound_weight,
        origin=2.0,
        logdet=config.logdet_weight,
    )
    weights_with = LossWeights(**shared_weights, behavior_cloning=1.0)
    weights_without = LossWeights(**shared_weights, behavior_cloning=0.0)

    batch_generator = torch.Generator().manual_seed(config.seed + 1)
    history = {
        "step": [],
        "with_bc_validation_loss": [],
        "without_bc_validation_loss": [],
    }
    num_train = train_data["e"].shape[0]

    for step in range(config.steps):
        indices = torch.randint(
            num_train,
            (config.batch_size,),
            generator=batch_generator,
        )
        shared_batch = {
            key: value[indices].to(config.device)
            for key, value in train_data.items()
        }
        shared_batch = add_equilibrium_anchors(
            shared_batch,
            config.equilibrium_anchor_fraction,
        )

        train_step(
            *with_bc,
            error_model,
            optimizer_with,
            {key: value.clone() for key, value in shared_batch.items()},
            weights=weights_with,
            config=training_config,
        )
        train_step(
            *without_bc,
            error_model,
            optimizer_without,
            {key: value.clone() for key, value in shared_batch.items()},
            weights=weights_without,
            config=training_config,
        )

        if step % config.evaluation_every == 0 or step == config.steps - 1:
            with_loss = validation_bc_mse(with_bc[1], val_data, config.device)
            without_loss = validation_bc_mse(without_bc[1], val_data, config.device)
            history["step"].append(step)
            history["with_bc_validation_loss"].append(with_loss)
            history["without_bc_validation_loss"].append(without_loss)
            print(
                f"step={step:04d}, val BC loss: "
                f"with L_BC={with_loss:.6f}, without L_BC={without_loss:.6f}",
                flush=True,
            )

    return error_model, with_bc, without_bc, history


def rollout_tracking_error(
    error_model: ErrorDynamics,
    controller,
    config: CompareBCConfig,
) -> tuple[np.ndarray, np.ndarray]:
    controller.eval()
    dt = float(config.dt)
    steps = int(round(config.rollout_seconds / dt))
    v = torch.tensor(config.planning_input, dtype=torch.float32)
    z = torch.zeros(3, dtype=torch.float32)

    e_reference = torch.zeros(6, dtype=torch.float32)
    e_reference[3] = v[0]
    e_reference[5] = v[1]
    initial_deviation = torch.tensor(
        config.initial_reference_error,
        dtype=torch.float32,
    )
    e = e_reference + initial_deviation
    x = error_model.state_from_error(e, z)

    time = [0.0]
    position_error = [
        float(torch.linalg.norm(error_model.error(x, z)[:2]).detach())
    ]
    for step in range(steps):
        with torch.no_grad():
            e = error_model.error(x, z)
            u = controller(e, v)
            x = error_model.tracking_model.step(x, u, dt)
            z = error_model.planning_model.step(z, v, dt)

        if not torch.all(torch.isfinite(x)):
            print(f"rollout stopped at step {step}: non-finite FOM state")
            break
        if x[3] <= config.minimum_tracking_speed:
            print(
                f"rollout stopped at step {step}: "
                f"longitudinal speed={x[3].item():.4f}"
            )
            break

        time.append((step + 1) * dt)
        position_error.append(
            float(torch.linalg.norm(error_model.error(x, z)[:2]).detach())
        )

    return np.asarray(time), np.asarray(position_error)


def save_checkpoint(
    config: CompareBCConfig,
    with_bc,
    without_bc,
    history: dict,
) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "history": history,
            "with_bc": {
                "lyapunov_state_dict": with_bc[0].state_dict(),
                "controller_state_dict": with_bc[1].state_dict(),
                "ellipsoid_state_dict": with_bc[2].state_dict(),
            },
            "without_bc": {
                "lyapunov_state_dict": without_bc[0].state_dict(),
                "controller_state_dict": without_bc[1].state_dict(),
                "ellipsoid_state_dict": without_bc[2].state_dict(),
            },
        },
        CHECKPOINT_PATH,
    )


def plot_comparison(
    history: dict,
    with_rollout: tuple[np.ndarray, np.ndarray],
    without_rollout: tuple[np.ndarray, np.ndarray],
    config: CompareBCConfig,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 17,
            "axes.labelsize": 22,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 17,
        }
    )
    fig, (ax_loss, ax_tracking) = plt.subplots(1, 2, figsize=(16, 4.7))

    ax_loss.plot(
        history["step"],
        history["with_bc_validation_loss"],
        linewidth=2.4,
        label=r"With $L_{BC}$",
    )
    ax_loss.plot(
        history["step"],
        history["without_bc_validation_loss"],
        linewidth=2.4,
        linestyle="--",
        label=r"Without $L_{BC}$",
    )
    ax_loss.set_xlabel("Training step")
    ax_loss.set_ylabel("Expert Imitation Error")
    ax_loss.set_xlim(0.0, 1500.0)
    ax_loss.set_xticks([0, 300, 600, 900, 1200, 1500])
    ax_loss.set_ylim(0.0, 5.0)
    ax_loss.set_yticks([0, 1, 2, 3, 4, 5])

    ax_tracking.plot(
        with_rollout[0],
        with_rollout[1],
        linewidth=2.4,
        label=r"With $L_{BC}$",
    )
    ax_tracking.plot(
        without_rollout[0],
        without_rollout[1],
        linewidth=2.4,
        linestyle="--",
        label=r"Without $L_{BC}$",
    )
    ax_tracking.set_xlabel("Time (sec)")
    ax_tracking.set_ylabel("Position Tracking Error")
    ax_tracking.set_xlim(0.0, config.rollout_seconds)
    ax_tracking.set_ylim(0.0, 200.0)
    ax_tracking.set_yticks([0, 40, 80, 120, 160, 200])
    ax_tracking.margins(x=0.0)

    for axis in (ax_loss, ax_tracking):
        axis.minorticks_off()
        axis.grid(which="major", color="#999999", linewidth=0.8, alpha=0.55)
        axis.legend()

    ax_loss.text(
        0.5,
        -0.20,
        "(a)",
        transform=ax_loss.transAxes,
        ha="center",
        va="top",
        fontsize=22,
    )
    ax_tracking.text(
        0.5,
        -0.20,
        "(b)",
        transform=ax_tracking.transAxes,
        ha="center",
        va="top",
        fontsize=22,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.21)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
    if config.show_plot:
        plt.show()
    else:
        plt.close(fig)


def run_comparison(config: CompareBCConfig | None = None) -> str:
    config = config or CompareBCConfig()
    error_model, with_bc, without_bc, history = train_ablation(config)
    with_rollout = rollout_tracking_error(error_model, with_bc[1], config)
    without_rollout = rollout_tracking_error(error_model, without_bc[1], config)
    save_checkpoint(config, with_bc, without_bc, history)
    plot_comparison(history, with_rollout, without_rollout, config)
    print(f"saved comparison checkpoint: {CHECKPOINT_PATH}")
    print(f"saved comparison figure: {OUTPUT_PATH}")
    return OUTPUT_PATH


def main() -> None:
    run_comparison()


if __name__ == "__main__":
    main()
