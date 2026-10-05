import os
import sys
import json
import copy
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
    data_path: str = "data/expert_dataset_bc.npz"
    results_dir: str = "results/JointTraining"
    steps: int = 7500
    batch_size: int = 200
    smt_steps: tuple[int, ...] = (1500, 3000, 4500, 6000)
    counterexample_fraction: float = 0.2
    counterexample_neighborhood_samples: int = 32
    log_every: int = 40
    validation_size: int = 1024
    device: str = "cpu"
    use_success_only: bool = True
    train_fraction: float = 0.95
    behavior_cloning_only: bool = False
    pretrained_controller_path: str = "results/BehaviorCloning/checkpoint.pt"
    freeze_pretrained_controller: bool = True
    lyapunov_weight: float = 4.0
    error_bound_weight: float = 2.0
    logdet_weight: float = 0.001
    equilibrium_anchor_fraction: float = 0.25
    smt_e_low: tuple[float, ...] = (-0.5, -0.3, -0.05, 5.0, -0.1, -0.05)
    smt_e_high: tuple[float, ...] = (0.5, 0.3, 0.05, 10.5, 0.1, 0.05)
    smt_z_low: tuple[float, ...] = (-20.0, -20.0, -np.pi)
    smt_z_high: tuple[float, ...] = (20.0, 20.0, np.pi)
    smt_v_low: tuple[float, ...] = (4.0, -1.0)
    smt_v_high: tuple[float, ...] = (10.0, 1.0)
    require_final_search_pass: bool = False
    require_closed_loop_dataset: bool = True
    final_cegis_rounds: int = 8
    final_cegis_steps_per_round: int = 1000
    final_cegis_counterexample_fraction: float = 0.75


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
    if config.require_closed_loop_dataset:
        if "config_json" not in data:
            raise ValueError(
                "Dataset has no generation metadata. Regenerate it with "
                "experiments/DataGeneration.py."
            )
        generation_config = json.loads(str(data["config_json"].item()))
        if not generation_config.get("closed_loop_tracking", False):
            raise ValueError(
                "Dataset was not generated with closed_loop_tracking=True. "
                "Regenerate it with experiments/DataGeneration.py."
            )
        if not config.behavior_cloning_only:
            minimum_planning_speed = generation_config.get(
                "min_planning_speed_for_tracking"
            )
            if minimum_planning_speed is None or config.smt_v_low[0] < minimum_planning_speed:
                raise ValueError(
                    "SMT planning-speed lower bound must be at least the dataset's "
                    "min_planning_speed_for_tracking."
                )

    dataset = {
        "e": torch.as_tensor(data["e"][mask], dtype=torch.float32),
        "z": torch.as_tensor(data["z"][mask], dtype=torch.float32),
        "v": torch.as_tensor(data["v"][mask], dtype=torch.float32),
        "u_expert": torch.as_tensor(data["u_expert"][mask], dtype=torch.float32),
    }
    expected_shapes = {
        "e": (6,),
        "z": (3,),
        "v": (2,),
        "u_expert": (2,),
    }
    for key, expected_tail in expected_shapes.items():
        if dataset[key].shape[1:] != expected_tail:
            raise ValueError(
                f"Expected dataset['{key}'] shape (samples, {expected_tail[0]}), "
                f"got {tuple(dataset[key].shape)}. Regenerate the Bicycle dataset."
            )
        if not torch.all(torch.isfinite(dataset[key])):
            raise ValueError(f"Dataset field '{key}' contains non-finite values")

    if torch.any(dataset["e"][:, 3] <= 0.0):
        raise ValueError(
            "Expected positive longitudinal speed e[:, 3] for the dynamic bicycle model"
        )
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
        raise ValueError("Expected at least two samples for train/validation splitting")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_samples, generator=generator)
    num_train = int(round(train_fraction * num_samples))
    num_train = min(max(num_train, 1), num_samples - 1)
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


def add_equilibrium_anchors(
    batch: dict[str, torch.Tensor],
    fraction: float,
) -> dict[str, torch.Tensor]:
    if fraction <= 0.0:
        return batch
    count = max(1, int(round(batch["e"].shape[0] * fraction)))
    device = batch["e"].device
    dtype = batch["e"].dtype
    speeds = 4.0 + 6.0 * torch.rand(count, device=device, dtype=dtype)
    e = torch.zeros(count, 6, device=device, dtype=dtype)
    v = torch.zeros(count, 2, device=device, dtype=dtype)
    e[:, 3] = speeds
    v[:, 0] = speeds
    anchors = {
        "e": e,
        "z": torch.zeros(count, 3, device=device, dtype=dtype),
        "v": v,
        "u_expert": torch.zeros(count, 2, device=device, dtype=dtype),
    }
    return {
        key: torch.cat([value, anchors[key]], dim=0)
        for key, value in batch.items()
    }


def make_models(error_model: ErrorDynamics, device: str):
    lyapunov = NeuralLyapunovFunction(
        error_dim=error_model.error_dim,
        hidden_sizes=(64, 64),
        feature_dim=16,
        delta=0.5,
    ).to(device)
    controller = NeuralTrackingController(
        hidden_sizes=(64, 64),
    ).to(device)
    ellipsoid = LearnableEllipsoid(
        dim=error_model.error_dim,
        init_scale=1.0,
    ).to(device)

    return lyapunov, controller, ellipsoid


def load_pretrained_controller(controller, config: JointTrainingConfig) -> None:
    if config.behavior_cloning_only or not config.pretrained_controller_path:
        return
    path = resolve_project_path(config.pretrained_controller_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Joint training requires the pretrained BC controller: {path}"
        )
    checkpoint = torch.load(path, map_location=config.device, weights_only=False)
    if "control_smoothness" in checkpoint.get("loss_weights", {}):
        raise ValueError(
            "Pretrained controller was produced by the later control-smoothing "
            "experiment. Rerun experiments/BehaviorCloning.py with the restored "
            "pointwise behavior-cloning code before joint training."
        )
    controller.load_state_dict(checkpoint["controller_state_dict"])
    print(f"loaded pretrained BC controller: {path}")


@torch.no_grad()
def behavior_cloning_mse(controller, dataset: dict[str, torch.Tensor], config: JointTrainingConfig) -> float:
    if dataset["e"].shape[0] == 0:
        return float("nan")

    if dataset["e"].shape[0] <= config.validation_size:
        batch = {key: value.to(config.device) for key, value in dataset.items()}
    else:
        batch = sample_batch(dataset, config.validation_size, config.device)
    controller.eval()
    u_pred = controller(batch["e"], batch["v"])
    return float(torch.mean(torch.sum((u_pred - batch["u_expert"]).square(), dim=-1)).cpu())


@torch.no_grad()
def reference_control_error(controller, device: str) -> float:
    speeds = torch.linspace(4.0, 10.0, 25, device=device)
    v = torch.stack([speeds, torch.zeros_like(speeds)], dim=-1)
    e = torch.zeros(len(speeds), 6, device=device)
    e[:, 3] = speeds
    u = controller(e, v)
    return float(torch.linalg.norm(u, dim=-1).mean().cpu())


def training_record(step: int, info: dict) -> dict:
    return {
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


def verification_record(result, backend: str, after_training_steps: int) -> dict:
    return {
        "after_training_steps": after_training_steps,
        "status": result.status,
        "success": result.success,
        "backend": backend,
        "formal_certificate": False,
        "counterexamples": [
            {
                "kind": counterexample.kind,
                "violation": counterexample.violation,
                "values": counterexample.values,
            }
            for counterexample in result.counterexamples
        ],
    }


def run_joint_training(config: JointTrainingConfig) -> str:
    if config.final_cegis_rounds < 0:
        raise ValueError("final_cegis_rounds must be nonnegative")
    if config.final_cegis_steps_per_round <= 0:
        raise ValueError("final_cegis_steps_per_round must be positive")
    if not 0.0 <= config.final_cegis_counterexample_fraction <= 1.0:
        raise ValueError("final_cegis_counterexample_fraction must be between 0 and 1")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    dataset = load_expert_dataset(config)
    train_data, val_data = split_dataset(dataset, config.train_fraction, config.seed)
    print(
        f"loaded {dataset['e'].shape[0]} expert samples "
        f"({train_data['e'].shape[0]} train, {val_data['e'].shape[0]} val)"
    )

    error_model = ErrorDynamics()
    lyapunov, controller, ellipsoid = make_models(error_model, config.device)
    load_pretrained_controller(controller, config)
    if config.behavior_cloning_only:
        train_features = torch.cat([train_data["e"], train_data["v"]], dim=-1)
        controller.set_feature_normalization(
            train_features.mean(dim=0),
            train_features.std(dim=0),
            min_scale=1e-2,
        )

    training_config = TrainingConfig(
        learning_rate=1e-3,
        gradient_clip_norm=10.0,
        level=1.0,
        decrease_margin=0.01,
        device=config.device,
    )
    weights = LossWeights(
        lyapunov=0.0 if config.behavior_cloning_only else config.lyapunov_weight,
        error_bound=0.0 if config.behavior_cloning_only else config.error_bound_weight,
        behavior_cloning=1.0,
        origin=0.0 if config.behavior_cloning_only else 2.0,
        logdet=0.0 if config.behavior_cloning_only else config.logdet_weight,
    )
    if config.behavior_cloning_only:
        optimizer = torch.optim.Adam(
            controller.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
    elif config.freeze_pretrained_controller:
        for parameter in controller.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam(
            list(lyapunov.parameters()) + list(ellipsoid.parameters()),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
    else:
        optimizer = make_optimizer(lyapunov, controller, ellipsoid, training_config)
    smt_config = None
    smt_solver = None
    counterexample_buffer = None
    if not config.behavior_cloning_only:
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
    final_verification = None
    best_val_bc = float("inf")
    best_val_step = None
    best_controller_state_dict = None
    for step in range(config.steps):
        batch = sample_batch(train_data, config.batch_size, config.device)
        batch = add_equilibrium_anchors(batch, config.equilibrium_anchor_fraction)
        if counterexample_buffer is not None:
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

        record = training_record(step, info)

        if step % config.log_every == 0 or step == config.steps - 1:
            val_bc = behavior_cloning_mse(controller, val_data, config)
            record["val_behavior_cloning_loss"] = val_bc
            if config.behavior_cloning_only and val_bc < best_val_bc:
                best_val_bc = val_bc
                best_val_step = step
                best_controller_state_dict = copy.deepcopy(controller.state_dict())
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
        if smt_solver is not None and completed_steps in config.smt_steps:
            print(f"running SMT counterexample search after {completed_steps} training steps")
            result = smt_solver.verify_all(
                lyapunov=lyapunov,
                controller=controller,
                error_model=error_model,
                E=ellipsoid.matrix(),
            )
            counterexample_buffer.add_many(result.counterexamples)
            smt_record = verification_record(result, smt_config.backend, completed_steps)
            smt_record["buffer_size"] = len(counterexample_buffer)
            smt_history.append(smt_record)
            print(
                f"SMT status={result.status}, "
                f"counterexamples={len(result.counterexamples)}, "
                f"buffer_size={len(counterexample_buffer)}"
            )

    total_training_steps = config.steps
    if config.behavior_cloning_only and best_controller_state_dict is not None:
        controller.load_state_dict(best_controller_state_dict)
        print(
            f"restored best behavior cloning controller from step {best_val_step} "
            f"(val_BC={best_val_bc:.6f})"
        )
    equilibrium_control_error = (
        reference_control_error(controller, config.device)
        if config.behavior_cloning_only
        else None
    )
    if smt_solver is not None:
        for cegis_round in range(config.final_cegis_rounds + 1):
            print(f"running final counterexample search, round {cegis_round}")
            final_verification = smt_solver.verify_all(
                lyapunov=lyapunov,
                controller=controller,
                error_model=error_model,
                E=ellipsoid.matrix(),
            )
            final_record = verification_record(
                final_verification,
                smt_config.backend,
                total_training_steps,
            )
            final_record["cegis_round"] = cegis_round
            smt_history.append(final_record)
            if final_verification.success or cegis_round == config.final_cegis_rounds:
                break

            counterexample_buffer.add_many(final_verification.counterexamples)
            print(
                f"final search found {len(final_verification.counterexamples)} "
                f"counterexamples; retraining for {config.final_cegis_steps_per_round} steps"
            )
            for _ in range(config.final_cegis_steps_per_round):
                batch = sample_batch(train_data, config.batch_size, config.device)
                batch = add_equilibrium_anchors(batch, config.equilibrium_anchor_fraction)
                batch = counterexample_buffer.mix_batch(
                    batch,
                    config.final_cegis_counterexample_fraction,
                )
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
                history.append(training_record(total_training_steps, info))
                total_training_steps += 1

        final_verification_record = verification_record(
            final_verification,
            smt_config.backend,
            total_training_steps,
        )
        final_verification_record["cegis_rounds_completed"] = cegis_round
    else:
        final_verification_record = {
            "status": "not_run_behavior_cloning_only",
            "success": False,
            "backend": None,
            "formal_certificate": False,
            "counterexamples": [],
        }

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
        "total_training_steps": total_training_steps,
        "best_val_behavior_cloning_loss": (
            best_val_bc if config.behavior_cloning_only else None
        ),
        "best_val_behavior_cloning_step": (
            best_val_step if config.behavior_cloning_only else None
        ),
        "reference_control_error": equilibrium_control_error,
        "history": history,
        "smt_history": smt_history,
        "counterexample_buffer_size": (
            len(counterexample_buffer) if counterexample_buffer is not None else 0
        ),
        "training_mode": (
            "behavior_cloning_only"
            if config.behavior_cloning_only
            else "certified_joint_training"
        ),
        "certificate_coordinates": (
            None
            if config.behavior_cloning_only
            else "moving_reference_local_exponential_v2"
        ),
        "final_verification": final_verification_record,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"saved checkpoint: {checkpoint_path}")
    if (
        not config.behavior_cloning_only
        and config.require_final_search_pass
        and not final_verification.success
    ):
        raise RuntimeError(
            "Final counterexample search found violations; checkpoint was saved for "
            "diagnostics but must not be treated as verified."
        )
    return checkpoint_path


def main() -> None:
    run_joint_training(JointTrainingConfig())


if __name__ == "__main__":
    main()
