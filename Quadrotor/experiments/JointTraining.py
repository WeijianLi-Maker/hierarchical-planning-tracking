import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.TrackingControl import NeuralTrackingController
from learning.LossFunction import LossWeights
from learning.Training import LearnableEllipsoid, TrainingConfig, make_optimizer, train_step
from lyapunov.LyapunovNetwork import NeuralLyapunovFunction
from models.error import ErrorDynamics
from SMT.CounterexampleBuffer import CounterexampleBuffer, CounterexampleBufferConfig
from SMT.SMTSolver import SMTConfig, SMTSolver

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class JointTrainingConfig:
    seed: int = 1
    data_path: str = "data/expert_dataset.npz"
    results_dir: str = "results/JointTraining"
    steps: int = 9000
    batch_size: int = 200
    bc_warmup_steps: int = 1000
    joint_ramp_steps: int = 1500
    smt_steps: tuple[int, ...] = (2000, 3000, 4000, 5000, 6000, 7000, 8000, 8800)
    certificate_fraction: float = 0.25
    counterexample_fraction: float = 0.35
    counterexample_neighborhood_samples: int = 64
    counterexample_neighborhood_radius: float = 0.1
    log_every: int = 40
    validation_size: int = 1024
    device: str = "cpu"
    use_success_only: bool = True
    train_fraction: float = 0.95


def resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def load_expert_dataset(config: JointTrainingConfig) -> dict[str, torch.Tensor]:
    data_path = resolve_project_path(config.data_path)
    data = np.load(data_path, allow_pickle=False)

    required_keys = ("e", "z", "v", "u_expert")
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise KeyError(f"Missing required dataset keys: {missing}")

    mask = np.ones(data["e"].shape[0], dtype=bool)
    if config.use_success_only:
        if "success" not in data:
            raise KeyError("Dataset does not contain 'success', but use_success_only=True")
        mask = data["success"].astype(bool)

    if not np.any(mask):
        raise ValueError("No samples available after filtering")

    dataset = {
        "e": torch.as_tensor(data["e"][mask], dtype=torch.float32),
        "z": torch.as_tensor(data["z"][mask], dtype=torch.float32),
        "v": torch.as_tensor(data["v"][mask], dtype=torch.float32),
        "u_expert": torch.as_tensor(data["u_expert"][mask], dtype=torch.float32),
    }
    if "traj_id" in data:
        dataset["traj_id"] = torch.as_tensor(data["traj_id"][mask], dtype=torch.long)
    if "is_closed_loop" not in data:
        raise KeyError(
            "Dataset does not contain 'is_closed_loop'. Regenerate it with "
            "experiments\\DataGeneration.py before training."
        )
    dataset["use_utilization_loss"] = torch.as_tensor(
        data["is_closed_loop"][mask],
        dtype=torch.bool,
    )
    dataset["use_interior_loss"] = torch.ones(
        dataset["e"].shape[0],
        dtype=torch.bool,
    )
    if not torch.any(dataset["use_utilization_loss"]):
        raise ValueError(
            "Dataset contains no closed-loop utilization samples. Regenerate it "
            "with closed_loop_trajectory_fraction > 0."
        )
    return dataset


def split_dataset(
    dataset: dict[str, torch.Tensor],
    train_fraction: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    generator = torch.Generator().manual_seed(seed)
    if "traj_id" in dataset:
        trajectory_ids = torch.unique(dataset["traj_id"])
        trajectory_ids = trajectory_ids[
            torch.randperm(trajectory_ids.numel(), generator=generator)
        ]
        num_train_trajectories = int(round(train_fraction * trajectory_ids.numel()))
        train_trajectory_ids = trajectory_ids[:num_train_trajectories]
        train_mask = torch.isin(dataset["traj_id"], train_trajectory_ids)
        train_indices = torch.where(train_mask)[0]
        val_indices = torch.where(~train_mask)[0]
    else:
        num_samples = dataset["e"].shape[0]
        indices = torch.randperm(num_samples, generator=generator)
        num_train = int(round(train_fraction * num_samples))
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]

    train_data = {key: value[train_indices] for key, value in dataset.items()}
    val_data = {key: value[val_indices] for key, value in dataset.items()}
    return train_data, val_data


def sample_batch(
    dataset: dict[str, torch.Tensor],
    batch_size: int,
    device: str,
) -> dict[str, torch.Tensor]:
    num_samples = dataset["e"].shape[0]
    indices = torch.randint(num_samples, (batch_size,))
    return {key: value[indices].to(device) for key, value in dataset.items()}


def mix_certificate_samples(
    batch: dict[str, torch.Tensor],
    fraction: float,
    smt_config: SMTConfig,
) -> dict[str, torch.Tensor]:
    if fraction <= 0.0:
        return batch

    total = batch["e"].shape[0]
    count = max(1, min(total, int(round(total * fraction))))
    target = torch.randperm(total, device=batch["e"].device)[:count]
    mixed = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}

    for key, low, high in (
        ("e", smt_config.e_low, smt_config.e_high),
        ("z", smt_config.z_low, smt_config.z_high),
        ("v", smt_config.v_low, smt_config.v_high),
    ):
        low_t = torch.tensor(low, device=batch[key].device, dtype=batch[key].dtype)
        high_t = torch.tensor(high, device=batch[key].device, dtype=batch[key].dtype)
        samples = low_t + torch.rand(count, low_t.numel(), device=low_t.device) * (high_t - low_t)
        mixed[key][target] = samples

    mixed["use_lyapunov_loss"] = torch.ones(total, dtype=torch.bool, device=batch["e"].device)
    mixed["use_error_bound_loss"] = torch.ones(total, dtype=torch.bool, device=batch["e"].device)
    mixed["use_behavior_cloning_loss"] = torch.ones(total, dtype=torch.bool, device=batch["e"].device)
    mixed["use_boundary_loss"] = torch.zeros(total, dtype=torch.bool, device=batch["e"].device)
    if "use_interior_loss" not in mixed:
        mixed["use_interior_loss"] = torch.ones(
            total,
            dtype=torch.bool,
            device=batch["e"].device,
        )
    if "use_utilization_loss" not in mixed:
        mixed["use_utilization_loss"] = torch.ones(
            total,
            dtype=torch.bool,
            device=batch["e"].device,
        )
    mixed["use_behavior_cloning_loss"][target] = False
    mixed["use_interior_loss"][target] = False
    mixed["use_utilization_loss"][target] = False
    boundary_count = count
    if boundary_count > 0:
        boundary_target = target[:boundary_count]
        dimensions = torch.randint(
            batch["e"].shape[-1],
            (boundary_count,),
            device=batch["e"].device,
        )
        use_high = torch.rand(boundary_count, device=batch["e"].device) >= 0.5
        e_low = torch.tensor(smt_config.e_low, device=batch["e"].device, dtype=batch["e"].dtype)
        e_high = torch.tensor(smt_config.e_high, device=batch["e"].device, dtype=batch["e"].dtype)
        rows = torch.arange(boundary_count, device=batch["e"].device)
        mixed["e"][boundary_target[rows], dimensions] = torch.where(
            use_high,
            e_high[dimensions],
            e_low[dimensions],
        )
        mixed["use_boundary_loss"][boundary_target] = True
    return mixed


def validate_expert_controls(
    dataset: dict[str, torch.Tensor],
    controller: NeuralTrackingController,
    tolerance: float = 1e-4,
) -> None:
    u_expert = dataset["u_expert"]
    u_min = controller.u_min.cpu()
    u_max = controller.u_max.cpu()
    below = u_expert < u_min - tolerance
    above = u_expert > u_max + tolerance
    outside = below | above
    outside_fraction = outside.any(dim=-1).float().mean().item()
    if outside_fraction > 0.0:
        per_dimension = outside.float().mean(dim=0).tolist()
        raise ValueError(
            f"{outside_fraction:.1%} of expert controls are outside the neural "
            f"controller bounds (per dimension: {per_dimension}). Regenerate "
            "the expert dataset with experiments/DataGeneration.py."
        )
    dataset["u_expert"] = torch.clamp(u_expert, min=u_min, max=u_max)


def make_models(error_model: ErrorDynamics, device: str):
    tracking_model = error_model.tracking_model

    lyapunov = NeuralLyapunovFunction(
        error_dim=error_model.error_dim,
        hidden_sizes=(64, 64),
        feature_dim=16,
        delta=0.02,
    ).to(device)
    controller = NeuralTrackingController(
        hidden_sizes=(64, 64),
        g=tracking_model.g,
        kT=tracking_model.kT,
    ).to(device)
    ellipsoid = LearnableEllipsoid(
        dim=error_model.error_dim,
        init_scale=1.0,
    ).to(device)

    return lyapunov, controller, ellipsoid


@torch.no_grad()
def behavior_cloning_mse(controller, dataset: dict[str, torch.Tensor], config: JointTrainingConfig) -> float:
    if dataset["e"].shape[0] == 0:
        return float("nan")

    batch_size = min(config.validation_size, dataset["e"].shape[0])
    if batch_size == dataset["e"].shape[0]:
        batch = {key: value.to(config.device) for key, value in dataset.items()}
    else:
        batch = sample_batch(dataset, batch_size, config.device)
    controller.eval()
    u_pred = controller(batch["e"], batch["v"])
    return float(torch.mean(torch.sum((u_pred - batch["u_expert"]).square(), dim=-1)).cpu())


def main() -> None:
    config = JointTrainingConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    dataset = load_expert_dataset(config)
    error_model = ErrorDynamics()
    lyapunov, controller, ellipsoid = make_models(error_model, config.device)
    validate_expert_controls(dataset, controller)
    train_data, val_data = split_dataset(dataset, config.train_fraction, config.seed)
    print(
        f"loaded {dataset['e'].shape[0]} expert samples "
        f"({train_data['e'].shape[0]} train, {val_data['e'].shape[0]} val), "
        f"{int(dataset['use_utilization_loss'].sum())} closed-loop utilization samples"
    )

    training_config = TrainingConfig(
        learning_rate=1e-3,
        gradient_clip_norm=10.0,
        level=1.0,
        decrease_margin=0.02,
        device=config.device,
    )
    target_weights = LossWeights(
        lyapunov=2,
        error_bound=3,
        behavior_cloning=8.0,
        origin=2.0,
        boundary=3.0,
        interior=3.0,
        logdet=0.001,
        error_bound_level=0.85,
        lyapunov_utilization=1.0,
        ellipsoid_utilization=0.25,
        lyapunov_ellipsoid_alignment=15.0,
        lyapunov_ellipsoid_alignment_margin=0.02,
        horizontal_position_bound=0.005,
    )
    optimizer = make_optimizer(lyapunov, controller, ellipsoid, training_config)
    smt_config = SMTConfig(
        decrease_margin=training_config.decrease_margin,
        device=config.device,
        e_low=(-2.5, -1.5, -0.4, -0.8, -2.5, -1.5, -0.4, -0.8, -4.0, -1.2),
        e_high=(2.5, 1.5, 0.4, 0.8, 2.5, 1.5, 0.4, 0.8, 4.0, 1.2),
        z_low=(-21.0, -17.0, 0.0),
        z_high=(21.0, 17.0, 11.0),
    )
    smt_solver = SMTSolver(smt_config)
    counterexample_buffer = CounterexampleBuffer(
        CounterexampleBufferConfig(
            neighborhood_samples=config.counterexample_neighborhood_samples,
            e_radius=config.counterexample_neighborhood_radius,
            z_radius=config.counterexample_neighborhood_radius,
            v_radius=config.counterexample_neighborhood_radius,
            device=config.device,
            e_low=smt_config.e_low,
            e_high=smt_config.e_high,
            z_low=smt_config.z_low,
            z_high=smt_config.z_high,
            v_low=smt_config.v_low,
            v_high=smt_config.v_high,
        )
    )

    history = []
    smt_history = []
    best_val_bc = float("inf")
    best_step = -1
    best_controller_state_dict = None
    for step in range(config.steps):
        if step < config.bc_warmup_steps:
            joint_scale = 0.0
        else:
            joint_scale = min(
                1.0,
                (step - config.bc_warmup_steps + 1) / config.joint_ramp_steps,
            )
        weights = LossWeights(
            lyapunov=target_weights.lyapunov * joint_scale,
            error_bound=target_weights.error_bound * joint_scale,
            behavior_cloning=target_weights.behavior_cloning,
            origin=target_weights.origin,
            boundary=target_weights.boundary,
            interior=target_weights.interior,
            logdet=target_weights.logdet,
            error_bound_level=target_weights.error_bound_level,
            lyapunov_utilization=target_weights.lyapunov_utilization,
            ellipsoid_utilization=target_weights.ellipsoid_utilization,
            lyapunov_ellipsoid_alignment=target_weights.lyapunov_ellipsoid_alignment,
            lyapunov_ellipsoid_alignment_margin=(
                target_weights.lyapunov_ellipsoid_alignment_margin
            ),
            horizontal_position_bound=target_weights.horizontal_position_bound,
        )
        batch = sample_batch(train_data, config.batch_size, config.device)
        batch = mix_certificate_samples(batch, config.certificate_fraction, smt_config)
        batch = counterexample_buffer.mix_batch(batch, config.counterexample_fraction)
        _, info = train_step(
            lyapunov=lyapunov,
            controller=controller,
            ellipsoid=ellipsoid,
            error_model=error_model,
            optimizer=optimizer,
            batch=batch,
            weights=weights,
            config=training_config,
        )

        record = {
            "step": step,
            "joint_loss_scale": joint_scale,
            "lyapunov_weight": weights.lyapunov,
            "error_bound_weight": weights.error_bound,
            "total_loss": float(info["total_loss"].detach().cpu()),
            "lyapunov_loss": float(info["lyapunov_loss"].detach().cpu()),
            "lyapunov_utilization_loss": float(
                info["lyapunov_utilization_loss"].detach().cpu()
            ),
            "error_bound_loss": float(info["error_bound_loss"].detach().cpu()),
            "ellipsoid_utilization_loss": float(
                info["ellipsoid_utilization_loss"].detach().cpu()
            ),
            "lyapunov_ellipsoid_alignment_loss": float(
                info["lyapunov_ellipsoid_alignment_loss"].detach().cpu()
            ),
            "horizontal_position_bound_loss": float(
                info["horizontal_position_bound_loss"].detach().cpu()
            ),
            "behavior_cloning_loss": float(info["behavior_cloning_loss"].detach().cpu()),
            "u_error_norm_mean": float(info["u_error_norm_mean"].detach().cpu()),
            "V_dot_mean": float(info["V_dot_mean"].detach().cpu()),
            "decrease_violation_mean": float(info["decrease_violation_mean"].detach().cpu()),
            "boundary_violation_mean": float(info["boundary_violation_mean"].detach().cpu()),
            "interior_violation_mean": float(info["interior_violation_mean"].detach().cpu()),
            "outside_level_fraction": float(info["outside_level_fraction"].detach().cpu()),
            "E_min_eig": float(info["E_min_eig"].detach().cpu()),
            "E_max_eig": float(info["E_max_eig"].detach().cpu()),
        }

        if step % config.log_every == 0 or step == config.steps - 1:
            val_bc = behavior_cloning_mse(controller, val_data, config)
            record["val_behavior_cloning_loss"] = val_bc
            if val_bc < best_val_bc:
                best_val_bc = val_bc
                best_step = step
                best_controller_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in controller.state_dict().items()
                }
            print(
                f"step={step:04d}, "
                f"loss={record['total_loss']:.5f}, "
                f"L_V={record['lyapunov_loss']:.5f}, "
                f"L_E={record['error_bound_loss']:.5f}, "
                f"L_BC={record['behavior_cloning_loss']:.5f}, "
                f"L_xy={record['horizontal_position_bound_loss']:.5f}, "
                f"val_BC={val_bc:.5f}, "
                f"u_err={record['u_error_norm_mean']:.5f}, "
                f"Vdot={record['V_dot_mean']:.5f}, "
                f"viol={record['decrease_violation_mean']:.5f}, "
                f"outside={record['outside_level_fraction']:.3f}, "
                f"E_min={record['E_min_eig']:.5f}, "
                f"E_max={record['E_max_eig']:.5f}"
            )

        history.append(record)

        completed_steps = step + 1
        if completed_steps in config.smt_steps:
            print(f"running SMT counterexample search after {completed_steps} training steps")
            result = smt_solver.verify_all(
                lyapunov=lyapunov,
                controller=controller,
                error_model=error_model,
                E=ellipsoid.matrix(),
            )
            counterexample_buffer.add_many(result.counterexamples)
            smt_record = {
                "after_training_steps": completed_steps,
                "status": result.status,
                "counterexamples": [
                    {
                        "kind": counterexample.kind,
                        "violation": counterexample.violation,
                        "e": counterexample.e.tolist(),
                        "z": None
                        if counterexample.z is None
                        else counterexample.z.tolist(),
                        "v": None
                        if counterexample.v is None
                        else counterexample.v.tolist(),
                        "values": counterexample.values,
                    }
                    for counterexample in result.counterexamples
                ],
                "buffer_size": len(counterexample_buffer),
            }
            smt_history.append(smt_record)
            print(
                f"SMT status={result.status}, "
                f"counterexamples={len(result.counterexamples)}, "
                f"buffer_size={len(counterexample_buffer)}"
            )

    results_dir = resolve_project_path(config.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_path = os.path.join(results_dir, "checkpoint.pt")
    if best_controller_state_dict is None:
        best_controller_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in controller.state_dict().items()
        }
    checkpoint = {
        "lyapunov_state_dict": lyapunov.state_dict(),
        "controller_state_dict": controller.state_dict(),
        "best_behavior_cloning_controller_state_dict": best_controller_state_dict,
        "best_validation_behavior_cloning_loss": best_val_bc,
        "best_validation_step": best_step,
        "ellipsoid_state_dict": ellipsoid.state_dict(),
        "E": ellipsoid.matrix().detach().cpu(),
        "joint_training_config": asdict(config),
        "training_config": asdict(training_config),
        "loss_weights": asdict(target_weights),
        "num_samples": dataset["e"].shape[0],
        "num_train": train_data["e"].shape[0],
        "num_val": val_data["e"].shape[0],
        "history": history,
        "smt_history": smt_history,
        "counterexample_buffer_size": len(counterexample_buffer),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
