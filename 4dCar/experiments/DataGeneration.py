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
    num_trajectories: int = 50
    rollout_seconds: float = 5.0
    dt: float = 0.04
    planner_mpc_horizon: int = 20
    planner_z_min: tuple[float, float] = (-4.0, -4.0)
    planner_z_max: tuple[float, float] = (4.0, 4.0)
    planner_v_min: tuple[float, float] = (-0.5, -0.5)
    planner_v_max: tuple[float, float] = (0.5, 0.5)
    mpc_horizon: int = 30
    tracking_u_min: tuple[float, float] = (-1.0, -1.0)
    tracking_u_max: tuple[float, float] = (1.0, 1.0)
    tracking_q: tuple[float, float, float, float] = (10.0, 10.0, 0.5, 0.5)
    tracking_r: tuple[float, float] = (0.5, 0.2)
    tracking_qf: tuple[float, float, float, float] = (20.0, 20.0, 10.0, 10.0)
    z0_low: tuple[float, float] = (-1.2, -1.2)
    z0_high: tuple[float, float] = (1.2, 1.2)
    z_goal_low: tuple[float, float] = (-4.0, -4.0)
    z_goal_high: tuple[float, float] = (4.0, 4.0)
    # World-frame error order: [e_x, e_y, e_theta, e_v].
    # Match the joint-training certificate search domain so behavior cloning
    # covers the errors where Lyapunov and ellipsoid conditions are checked.
    e_scale: tuple[float, float, float, float] = (1.5, 1.5, 1.0, 1.5)
    near_origin_fraction: float = 0.5
    near_origin_e_scale: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.5)
    stationary_planner_fraction: float = 0.0
    use_planner_preview: bool = False
    output_path: str = "data/expert_dataset.npz"
    progress_every: int = 10
    progress_every_samples: int = 25
    reset_mpc_each_sample: bool = True


def sample_uniform(rng: np.random.Generator, low, high) -> np.ndarray:
    return rng.uniform(np.asarray(low, dtype=float), np.asarray(high, dtype=float))


def sample_error(rng: np.random.Generator, e_scale: np.ndarray) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=e_scale.shape) * e_scale


def projection_numpy(error_model: ErrorDynamics, z: np.ndarray) -> np.ndarray:
    z_t = torch.tensor(z, dtype=torch.float32)
    with torch.no_grad():
        x_hat = error_model.projection(z_t)
    return x_hat.detach().cpu().numpy()


def state_from_error_numpy(
    error_model: ErrorDynamics,
    e: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    e_t = torch.tensor(e, dtype=torch.float32)
    z_t = torch.tensor(z, dtype=torch.float32)
    with torch.no_grad():
        x = error_model.state_from_error(e_t, z_t)
    return x.detach().cpu().numpy()


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
    planning_model = planning_mpc.planning_model
    z_traj = np.zeros((num_steps + 1, planning_model.state_dim), dtype=float)
    v_traj = np.zeros((num_steps, planning_model.input_dim), dtype=float)

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
        Q=np.diag(config.tracking_q),
        R=np.diag(config.tracking_r),
        Qf=np.diag(config.tracking_qf),
        u_min=config.tracking_u_min,
        u_max=config.tracking_u_max,
    )

    e_scale = np.asarray(config.e_scale, dtype=float)
    near_origin_e_scale = np.asarray(config.near_origin_e_scale, dtype=float)
    if e_scale.shape != (error_model.error_dim,):
        raise ValueError(
            f"Expected e_scale shape ({error_model.error_dim},), got {e_scale.shape}"
        )
    if not np.all(np.isfinite(e_scale)) or np.any(e_scale <= 0.0):
        raise ValueError("Expected every e_scale element to be finite and positive")
    if near_origin_e_scale.shape != (error_model.error_dim,):
        raise ValueError(
            f"Expected near_origin_e_scale shape ({error_model.error_dim},), "
            f"got {near_origin_e_scale.shape}"
        )
    if not np.all(np.isfinite(near_origin_e_scale)) or np.any(near_origin_e_scale <= 0.0):
        raise ValueError("Expected every near_origin_e_scale element to be finite and positive")
    if not 0.0 <= config.near_origin_fraction <= 1.0:
        raise ValueError("Expected near_origin_fraction to be between zero and one")
    if not 0.0 <= config.stationary_planner_fraction <= 1.0:
        raise ValueError("Expected stationary_planner_fraction to be between zero and one")
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
        z_traj, v_traj = rollout_planner_trajectory(
            z0,
            z_goal,
            config,
            planning_mpc,
        )

        for time_id in range(len(v_traj)):
            z = z_traj[time_id]
            v = v_traj[time_id]
            if rng.random() < config.stationary_planner_fraction:
                v = np.zeros_like(v)
            if config.use_planner_preview:
                v_seq = make_v_sequence(v_traj, time_id, config.mpc_horizon)
            else:
                # The learned policy receives only current (e, v), so its expert
                # label must not depend on an unobserved future planner sequence.
                v_seq = np.tile(v, (config.mpc_horizon, 1))
            sample_scale = (
                near_origin_e_scale
                if rng.random() < config.near_origin_fraction
                else e_scale
            )
            e = sample_error(rng, sample_scale)
            x_hat = projection_numpy(error_model, z)
            x = state_from_error_numpy(error_model, e, z)

            if config.reset_mpc_each_sample:
                reset_mpc_warm_start(mpc)
            u_expert, info = mpc.solve(e0=e, z0=z, v_seq=v_seq)
            if not info["success"]:
                failure_count += 1
                if failure_count == 1:
                    print(
                        f"warning: first MPC solve failed with status: {info['status']}",
                        flush=True,
                    )

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
        "error_coordinates": np.asarray(error_model.error_coordinates),
        "error_dynamics_version": np.asarray(error_model.error_dynamics_version),
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
