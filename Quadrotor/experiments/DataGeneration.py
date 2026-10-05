import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.MPCPlanning import LinearizedPlanningMPCController
from controllers.MPCTracking import LinearizedErrorMPCController
from models.error import ErrorDynamics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class DataGenerationConfig:
    seed: int = 1
    num_trajectories: int = 40
    rollout_seconds: float = 8.0
    dt: float = 0.04 
    planner_mpc_horizon: int = 20
    planner_z_min: tuple[float, float, float] = (-21.0, -17.0, 0.0)
    planner_z_max: tuple[float, float, float] = (21.0, 17.0, 11.0)
    planner_v_min: tuple[float, float, float] = (-0.5, -0.5, -0.5)
    planner_v_max: tuple[float, float, float] = (0.5, 0.5, 0.5)
    mpc_horizon: int = 20
    tracking_u_min: tuple[float, float, float] = (
        -np.pi / 18.0,
        -np.pi / 18.0,
        0.0,
    )
    tracking_u_max: tuple[float, float, float] = (
        np.pi / 18.0,
        np.pi / 18.0,
        1.5 * 9.81,
    )
    z0_low: tuple[float, float, float] = (-1.2, -1.2, 0.0)
    z0_high: tuple[float, float, float] = (1.2, 1.2, 1.2)
    z_goal_low: tuple[float, float, float] = (-20.0, -16.0, 0.2)
    z_goal_high: tuple[float, float, float] = (20.0, 16.0, 10.0)
    e_scale: tuple[float, ...] = (
        0.625,
        0.375,
        0.1,
        0.2,
        0.625,
        0.375,
        0.1,
        0.2,
        0.45,
        0.3,
    )
    error_scale_factors: tuple[float, ...] = (1.0, 0.25, 0.05)
    error_scale_probabilities: tuple[float, ...] = (0.5, 0.3, 0.2)
    closed_loop_trajectory_fraction: float = 0.5
    output_path: str = "data/expert_dataset.npz"
    progress_every: int = 10
    progress_every_samples: int = 25
    reset_mpc_each_sample: bool = True


def sample_uniform(rng: np.random.Generator, low, high) -> np.ndarray:
    return rng.uniform(np.asarray(low, dtype=float), np.asarray(high, dtype=float))


def sample_error(
    rng: np.random.Generator,
    e_scale: np.ndarray,
    scale_factors: np.ndarray,
    scale_probabilities: np.ndarray,
) -> np.ndarray:
    scale_factor = rng.choice(scale_factors, p=scale_probabilities)
    return rng.uniform(-1.0, 1.0, size=e_scale.shape) * e_scale * scale_factor


def projection_numpy(error_model: ErrorDynamics, z: np.ndarray) -> np.ndarray:
    z_t = torch.tensor(z, dtype=torch.float32)
    with torch.no_grad():
        x_hat = error_model.projection(z_t)
    return x_hat.detach().cpu().numpy()


def planning_step_numpy(planning_model, z: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
    z_t = torch.tensor(z, dtype=torch.float32)
    v_t = torch.tensor(v, dtype=torch.float32)
    with torch.no_grad():
        z_next = planning_model.step(z_t, v_t, dt)
    return z_next.detach().cpu().numpy()


def rollout_planner_trajectory(
    z0: np.ndarray,
    z_goal: np.ndarray,
    config: DataGenerationConfig,
    planning_mpc: LinearizedPlanningMPCController,
) -> tuple[np.ndarray, np.ndarray]:
    num_steps = int(round(config.rollout_seconds / config.dt))
    z_traj = np.zeros((num_steps + 1, 3), dtype=float)
    v_traj = np.zeros((num_steps, 3), dtype=float)

    z_traj[0] = z0
    planning_mpc.last_solution = np.tile(planning_mpc.v_ref, (planning_mpc.N, 1))
    for k in range(num_steps):
        v, info = planning_mpc.solve(z_traj[k], z_goal)
        if not info["success"]:
            raise RuntimeError(
                f"Planning MPC failed at planner step {k}: {info['status']}"
            )
        v_traj[k] = v
        z_traj[k + 1] = planning_step_numpy(
            planning_mpc.planning_model,
            z_traj[k],
            v,
            config.dt,
        )

    return z_traj, v_traj


def make_v_sequence(v_traj: np.ndarray, start: int, horizon: int) -> np.ndarray:
    end = start + horizon
    if end <= len(v_traj):
        return v_traj[start:end]

    v_seq = np.zeros((horizon, v_traj.shape[1]), dtype=float)
    available = len(v_traj) - start
    if available > 0:
        v_seq[:available] = v_traj[start:]
        v_seq[available:] = v_traj[-1]
    else:
        v_seq[:] = v_traj[-1]
    return v_seq


def reset_mpc_warm_start(mpc: LinearizedErrorMPCController) -> None:
    mpc.last_solution = np.tile(mpc.u_ref, (mpc.N, 1))


def DataGeneration(config: DataGenerationConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    error_model = ErrorDynamics()
    planning_mpc = LinearizedPlanningMPCController(
        planning_model=error_model.planning_model,
        dt=config.dt,
        horizon=config.planner_mpc_horizon,
        z_min=config.planner_z_min,
        z_max=config.planner_z_max,
        v_min=config.planner_v_min,
        v_max=config.planner_v_max,
    )
    mpc = LinearizedErrorMPCController(
        error_model=error_model,
        dt=config.dt,
        horizon=config.mpc_horizon,
        u_min=config.tracking_u_min,
        u_max=config.tracking_u_max,
    )

    e_scale = np.asarray(config.e_scale, dtype=float)
    error_scale_factors = np.asarray(config.error_scale_factors, dtype=float)
    error_scale_probabilities = np.asarray(config.error_scale_probabilities, dtype=float)
    if (
        error_scale_factors.ndim != 1
        or error_scale_probabilities.shape != error_scale_factors.shape
        or np.any(error_scale_factors <= 0.0)
        or np.any(error_scale_probabilities < 0.0)
        or not np.isclose(error_scale_probabilities.sum(), 1.0)
    ):
        raise ValueError("Expected valid error scale factors and probabilities")
    if not 0.0 <= config.closed_loop_trajectory_fraction <= 1.0:
        raise ValueError("closed_loop_trajectory_fraction must be between 0 and 1")
    z_values = []
    v_values = []
    v_seq_values = []
    e_values = []
    x_hat_values = []
    x_values = []
    u_expert_values = []
    success_values = []
    cost_values = []
    status_values = []
    traj_id_values = []
    time_id_values = []
    z_goal_values = []
    closed_loop_values = []

    samples_per_trajectory = int(round(config.rollout_seconds / config.dt))
    total_samples = config.num_trajectories * samples_per_trajectory
    start_time = time.perf_counter()
    failure_count = 0
    print(
        f"starting data generation: {config.num_trajectories} trajectories, "
        f"{samples_per_trajectory} samples each, {total_samples} planning MPC "
        f"solves and {total_samples} tracking MPC solves",
        flush=True,
    )

    for traj_id in range(config.num_trajectories):
        z0 = sample_uniform(rng, config.z0_low, config.z0_high)
        z_goal = sample_uniform(rng, config.z_goal_low, config.z_goal_high)
        use_closed_loop_error = rng.random() < config.closed_loop_trajectory_fraction
        z_traj, v_traj = rollout_planner_trajectory(
            z0,
            z_goal,
            config,
            planning_mpc,
        )
        closed_loop_x = projection_numpy(error_model, z0) + sample_error(
            rng,
            e_scale,
            error_scale_factors,
            error_scale_probabilities,
        )
        if use_closed_loop_error:
            reset_mpc_warm_start(mpc)

        for time_id in range(len(v_traj)):
            z = z_traj[time_id]
            v = v_traj[time_id]
            v_seq = make_v_sequence(v_traj, time_id, config.mpc_horizon)
            x_hat = projection_numpy(error_model, z)
            if use_closed_loop_error:
                x = closed_loop_x
                e = x - x_hat
            else:
                e = sample_error(
                    rng,
                    e_scale,
                    error_scale_factors,
                    error_scale_probabilities,
                )
                x = x_hat + e

            if config.reset_mpc_each_sample and not use_closed_loop_error:
                reset_mpc_warm_start(mpc)
            u_expert, info = mpc.solve(e0=e, z0=z, v_seq=v_seq)
            if not info["success"]:
                failure_count += 1
                if failure_count == 1:
                    print(
                        f"warning: first MPC solve failed with status: {info['status']}",
                        flush=True,
                    )

            if use_closed_loop_error:
                x_t = torch.as_tensor(closed_loop_x, dtype=torch.float32)
                u_t = torch.as_tensor(u_expert, dtype=torch.float32)
                with torch.no_grad():
                    closed_loop_x = (
                        x_t + config.dt * error_model.tracking_model.dynamics(x_t, u_t)
                    ).cpu().numpy()

            z_values.append(z)
            v_values.append(v)
            v_seq_values.append(v_seq)
            e_values.append(e)
            x_hat_values.append(x_hat)
            x_values.append(x)
            u_expert_values.append(u_expert)
            success_values.append(bool(info["success"]))
            cost_values.append(float(info["cost"]))
            status_values.append(str(info["status"]))
            traj_id_values.append(traj_id)
            time_id_values.append(time_id)
            z_goal_values.append(z_goal)
            closed_loop_values.append(use_closed_loop_error)

            completed = len(z_values)
            if (
                config.progress_every_samples
                and completed % config.progress_every_samples == 0
            ):
                elapsed = time.perf_counter() - start_time
                solves_per_second = completed / elapsed
                eta_seconds = (total_samples - completed) / solves_per_second
                print(
                    f"generated {completed}/{total_samples} samples "
                    f"({failure_count} MPC failures, {elapsed:.1f}s elapsed, "
                    f"ETA {eta_seconds / 60:.1f} min)",
                    flush=True,
                )

        if config.progress_every and (traj_id + 1) % config.progress_every == 0:
            print(
                f"generated {traj_id + 1}/{config.num_trajectories} trajectories",
                flush=True,
            )

    return {
        "z": np.asarray(z_values, dtype=np.float32),
        "v": np.asarray(v_values, dtype=np.float32),
        "v_seq": np.asarray(v_seq_values, dtype=np.float32),
        "e": np.asarray(e_values, dtype=np.float32),
        "x_hat": np.asarray(x_hat_values, dtype=np.float32),
        "x": np.asarray(x_values, dtype=np.float32),
        "u_expert": np.asarray(u_expert_values, dtype=np.float32),
        "success": np.asarray(success_values, dtype=bool),
        "cost": np.asarray(cost_values, dtype=np.float32),
        "status": np.asarray(status_values),
        "traj_id": np.asarray(traj_id_values, dtype=np.int32),
        "time_id": np.asarray(time_id_values, dtype=np.int32),
        "z_goal": np.asarray(z_goal_values, dtype=np.float32),
        "is_closed_loop": np.asarray(closed_loop_values, dtype=bool),
        "config_json": np.asarray(json.dumps(asdict(config))),
    }


def save_dataset(dataset: dict[str, np.ndarray], output_path: str) -> None:
    if not os.path.isabs(output_path):
        output_path = os.path.join(PROJECT_ROOT, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **dataset)


def main() -> None:
    config = DataGenerationConfig()
    dataset = DataGeneration(config)
    save_dataset(dataset, config.output_path)

    num_samples = dataset["z"].shape[0]
    success_rate = np.mean(dataset["success"])
    print(f"saved {num_samples} samples to {config.output_path}")
    print(f"mpc success rate: {success_rate:.3f}")


if __name__ == "__main__":
    main()
