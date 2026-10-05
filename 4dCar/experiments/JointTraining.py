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
    steps: int = 7500
    batch_size: int = 200
    smt_steps: tuple[int, ...] = ()
    counterexample_fraction: float = 0.0
    counterexample_neighborhood_samples: int = 32
    log_every: int = 40
    validation_size: int = 1024
    device: str = "cpu"
    use_success_only: bool = True
    train_fraction: float = 0.95
    controller_u_min: tuple[float, float] = (-1.0, -1.0)
    controller_u_max: tuple[float, float] = (1.0, 1.0)
    smt_e_low: tuple[float, float, float, float] = (-1.5, -1.5, -1.0, -1.5)
    smt_e_high: tuple[float, float, float, float] = (1.5, 1.5, 1.0, 1.5)
    smt_z_low: tuple[float, float] = (-4.0, -4.0)
    smt_z_high: tuple[float, float] = (4.0, 4.0)
    smt_v_low: tuple[float, float] = (-0.5, -0.5)
    smt_v_high: tuple[float, float] = (0.5, 0.5)


def resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def load_expert_dataset(
    config: JointTrainingConfig,
    error_model: ErrorDynamics | None = None,
) -> dict[str, torch.Tensor]:
    data_path = resolve_project_path(config.data_path)
    data = np.load(data_path, allow_pickle=False)

    required_keys = ("e", "z", "v", "u_expert")
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise KeyError(f"Missing required dataset keys: {missing}")
    if error_model is not None:
        metadata_keys = ("error_coordinates", "error_dynamics_version")
        missing_metadata = [key for key in metadata_keys if key not in data]
        if missing_metadata:
            raise ValueError(
                f"Dataset is missing ErrorDynamics metadata {missing_metadata}. "
                "Regenerate the expert dataset with the current ErrorDynamics."
            )
        error_coordinates = str(data["error_coordinates"].reshape(()))
        error_dynamics_version = int(data["error_dynamics_version"].reshape(()))
        if (
            error_coordinates != error_model.error_coordinates
            or error_dynamics_version != error_model.error_dynamics_version
        ):
            raise ValueError(
                "Dataset ErrorDynamics metadata does not match the current model: "
                f"coordinates={error_coordinates!r}, version={error_dynamics_version}. "
                "Regenerate expert data."
            )

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
    if error_model is not None:
        expected_dims = {
            "e": error_model.error_dim,
            "z": error_model.planning_model.state_dim,
            "v": error_model.planning_model.input_dim,
            "u_expert": error_model.tracking_model.input_dim,
        }
        sample_count = dataset["e"].shape[0]
        for key, expected_dim in expected_dims.items():
            value = dataset[key]
            if value.ndim != 2 or value.shape != (sample_count, expected_dim):
                raise ValueError(
                    f"Expected dataset['{key}'] shape ({sample_count}, {expected_dim}), "
                    f"got {tuple(value.shape)}. Regenerate the expert dataset with "
                    "the current Planning, Tracking, and Error Models."
                )
            if not torch.all(torch.isfinite(value)):
                raise ValueError(f"Expected dataset['{key}'] to contain only finite values")
    return dataset


def split_dataset(
    dataset: dict[str, torch.Tensor],
    train_fraction: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    num_samples = dataset["e"].shape[0]
    if num_samples < 2:
        raise ValueError("Expected at least two expert samples for train/validation split")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_samples, generator=generator)
    num_train = min(max(int(round(train_fraction * num_samples)), 1), num_samples - 1)
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
    if num_samples == 0:
        raise ValueError("Cannot sample from an empty dataset")
    indices = torch.randint(num_samples, (batch_size,))
    return {key: value[indices].to(device) for key, value in dataset.items()}


def make_models(
    error_model: ErrorDynamics,
    device: str,
    u_min=(-1.0, -1.0),
    u_max=(1.0, 1.0),
):
    lyapunov = NeuralLyapunovFunction(
        error_dim=error_model.error_dim,
        hidden_sizes=(32, 32),
        feature_dim=8,
        delta=0.5,
    ).to(device)
    controller = NeuralTrackingController(
        error_dim=error_model.error_dim,
        planning_input_dim=error_model.planning_model.input_dim,
        control_dim=error_model.tracking_model.input_dim,
        hidden_sizes=(32, 32),
        u_min=u_min,
        u_max=u_max,
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
    batch = sample_batch(dataset, batch_size, config.device)
    controller.eval()
    u_pred = controller(batch["e"], batch["v"])
    return float(torch.mean(torch.sum((u_pred - batch["u_expert"]).square(), dim=-1)).cpu())


def main() -> None:
    config = JointTrainingConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    error_model = ErrorDynamics()
    dataset = load_expert_dataset(config, error_model)
    train_data, val_data = split_dataset(dataset, config.train_fraction, config.seed)
    print(
        f"loaded {dataset['e'].shape[0]} expert samples "
        f"({train_data['e'].shape[0]} train, {val_data['e'].shape[0]} val)"
    )

    lyapunov, controller, ellipsoid = make_models(
        error_model,
        config.device,
        u_min=config.controller_u_min,
        u_max=config.controller_u_max,
    )

    training_config = TrainingConfig(
        learning_rate=1e-3,
        gradient_clip_norm=10.0,
        level=1.0,
        decrease_margin=0.01,
        device=config.device,
    )
    weights = LossWeights(
        lyapunov=4,
        error_bound=2,
        behavior_cloning=1.0,
        origin=2.0,
        logdet=0.008,
    )
    optimizer = make_optimizer(lyapunov, controller, ellipsoid, training_config)
    smt_config = SMTConfig(
        decrease_margin=training_config.decrease_margin,
        device=config.device,
        e_low=config.smt_e_low,
        e_high=config.smt_e_high,
        z_low=config.smt_z_low,
        z_high=config.smt_z_high,
        v_low=config.smt_v_low,
        v_high=config.smt_v_high,
    )
    smt_solver = SMTSolver(smt_config)
    counterexample_buffer = CounterexampleBuffer(
        CounterexampleBufferConfig(
            neighborhood_samples=config.counterexample_neighborhood_samples,
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
    for step in range(config.steps):
        batch = sample_batch(train_data, config.batch_size, config.device)
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
            "total_loss": float(info["total_loss"].detach().cpu()),
            "lyapunov_loss": float(info["lyapunov_loss"].detach().cpu()),
            "error_bound_loss": float(info["error_bound_loss"].detach().cpu()),
            "behavior_cloning_loss": float(info["behavior_cloning_loss"].detach().cpu()),
            "u_error_norm_mean": float(info["u_error_norm_mean"].detach().cpu()),
            "V_dot_mean": float(info["V_dot_mean"].detach().cpu()),
            "decrease_violation_mean": float(info["decrease_violation_mean"].detach().cpu()),
            "outside_level_fraction": float(info["outside_level_fraction"].detach().cpu()),
            "E_min_eig": float(info["E_min_eig"].detach().cpu()),
            "E_max_eig": float(info["E_max_eig"].detach().cpu()),
        }

        if step % config.log_every == 0 or step == config.steps - 1:
            val_bc = behavior_cloning_mse(controller, val_data, config)
            record["val_behavior_cloning_loss"] = val_bc
            print(
                f"step={step:04d}, "
                f"loss={record['total_loss']:.5f}, "
                f"L_V={record['lyapunov_loss']:.5f}, "
                f"L_E={record['error_bound_loss']:.5f}, "
                f"L_BC={record['behavior_cloning_loss']:.5f}, "
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
    checkpoint = {
        "lyapunov_state_dict": lyapunov.state_dict(),
        "controller_state_dict": controller.state_dict(),
        "ellipsoid_state_dict": ellipsoid.state_dict(),
        "E": ellipsoid.matrix().detach().cpu(),
        "joint_training_config": asdict(config),
        "training_config": asdict(training_config),
        "loss_weights": asdict(weights),
        "num_samples": dataset["e"].shape[0],
        "num_train": train_data["e"].shape[0],
        "num_val": val_data["e"].shape[0],
        "history": history,
        "smt_history": smt_history,
        "counterexample_buffer_size": len(counterexample_buffer),
        "model_signature": error_model.model_signature(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
