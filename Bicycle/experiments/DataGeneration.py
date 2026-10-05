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
    num_trajectories: int = 20
    rollout_seconds: float = 5.0
    dt: float = 0.04 
    planner_mpc_horizon: int = 80
    planner_z_min: tuple[float, float, float] = (-10.0, -10.0, -np.inf)
    planner_z_max: tuple[float, float, float] = (60.0, 50.0, np.inf)
    planner_v_min: tuple[float, float] = (0.0, -0.3)
    planner_v_max: tuple[float, float] = (10.0, 0.3)
    planner_initial_input: tuple[float, float] = (4.0, 0.0)
    planner_max_input_rate: tuple[float, float] = (2.0, 0.5)
    reference_mode: str = "planner_mpc"
    straight_reference_fraction: float = 0.5
    max_reference_yaw_rate: float = 0.3
    mpc_horizon: int = 20
    z0_low: tuple[float, float, float] = (-4.0, -4.0, -np.pi)
    z0_high: tuple[float, float, float] = (4.0, 4.0, np.pi)
    fixed_z_goal: tuple[float, float, float] | None = (
        40.0,
        30.0,
        np.arctan2(30.0, 40.0),
    )
    z_goal_low: tuple[float, float, float] = (-15.0, -15.0, -np.pi)
    z_goal_high: tuple[float, float, float] = (15.0, 15.0, np.pi)
    e_scale: tuple[float, ...] = (
        0.5,
        0.3,
        0.05,
        0.5,
        0.1,
        0.05,
    )
    min_tracking_speed: float = 5.0
    min_planning_speed_for_tracking: float = 4.0
    rollout_speed_floor: float = 1.0
    output_path: str = "data/expert_dataset_bc.npz"
    progress_every: int = 10
    progress_every_samples: int = 25
    reset_mpc_each_sample: bool = False
    closed_loop_tracking: bool = True
    restart_tracking_after_failure: bool = True
    max_tracking_segment_steps: int = 125
    max_failure_warnings: int = 5


def sample_uniform(rng: np.random.Generator, low, high) -> np.ndarray:
    return rng.uniform(np.asarray(low, dtype=float), np.asarray(high, dtype=float))


def sample_error(
    rng: np.random.Generator,
    e_scale: np.ndarray,
    planning_input: np.ndarray,
    min_tracking_speed: float,
) -> np.ndarray:
    """
    Sample around the moving error reference [0, 0, 0, speed_hat, 0, omega_hat].
    """
    e_ref = np.zeros(6, dtype=float)
    e_ref[3] = planning_input[0]
    e_ref[5] = planning_input[1]
    e = e_ref + rng.uniform(-1.0, 1.0, size=e_scale.shape) * e_scale
    e[3] = max(e[3], min_tracking_speed)
    return e


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


def tracking_step_numpy(error_model: ErrorDynamics, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    x_t = torch.tensor(x, dtype=torch.float32)
    u_t = torch.tensor(u, dtype=torch.float32)
    with torch.no_grad():
        x_next = error_model.tracking_model.step(x_t, u_t, dt)
    return x_next.detach().cpu().numpy()


def rollout_planner_trajectory(
    z0: np.ndarray,
    z_goal: np.ndarray,
    config: DataGenerationConfig,
    planning_mpc: LinearizedPlanningMPCController,
) -> tuple[np.ndarray, np.ndarray]:
    num_steps = int(round(config.rollout_seconds / config.dt))
    z_traj = np.zeros((num_steps + 1, planning_mpc.nz), dtype=float)
    v_traj = np.zeros((num_steps, planning_mpc.nv), dtype=float)

    z_traj[0] = z0
    planning_mpc.last_solution = np.tile(planning_mpc.v_ref, (planning_mpc.N, 1))
    planning_mpc.previous_v = np.asarray(config.planner_initial_input, dtype=float)
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


def rollout_constant_reference_trajectory(
    rng: np.random.Generator,
    z0: np.ndarray,
    config: DataGenerationConfig,
    planning_model,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a Markov-compatible BC reference.

    The neural controller observes only the current planning input v, whereas
    the expert MPC also sees the future v sequence. Keeping v constant over a
    trajectory removes conflicting labels caused by unseen future commands.
    """
    num_steps = int(round(config.rollout_seconds / config.dt))
    min_speed = max(config.min_planning_speed_for_tracking, config.planner_v_min[0])
    speed = rng.uniform(min_speed, config.planner_v_max[0])
    if rng.random() < config.straight_reference_fraction:
        yaw_rate = 0.0
    else:
        yaw_rate = rng.uniform(
            -config.max_reference_yaw_rate,
            config.max_reference_yaw_rate,
        )

    v = np.array([speed, yaw_rate], dtype=float)
    v_traj = np.tile(v, (num_steps, 1))
    z_traj = np.zeros((num_steps + 1, planning_model.state_dim), dtype=float)
    z_traj[0] = z0
    for k in range(num_steps):
        z_traj[k + 1] = planning_step_numpy(
            planning_model,
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
        max_v_rate=config.planner_max_input_rate,
        initial_v=config.planner_initial_input,
    )
    mpc = LinearizedErrorMPCController(
        error_model=error_model,
        dt=config.dt,
        horizon=config.mpc_horizon,
    )

    e_scale = np.asarray(config.e_scale, dtype=float)
    if e_scale.shape != (error_model.error_dim,):
        raise ValueError(
            f"Expected e_scale shape ({error_model.error_dim},), got {e_scale.shape}"
        )
    if config.min_tracking_speed <= 0.0:
        raise ValueError("min_tracking_speed must be positive")
    if not 0.0 < config.rollout_speed_floor < config.min_tracking_speed:
        raise ValueError("rollout_speed_floor must be between 0 and min_tracking_speed")
    if not 0.0 < config.min_planning_speed_for_tracking <= config.min_tracking_speed:
        raise ValueError(
            "min_planning_speed_for_tracking must be between 0 and min_tracking_speed"
        )
    if config.max_tracking_segment_steps <= 0:
        raise ValueError("max_tracking_segment_steps must be positive")
    planner_initial_input = np.asarray(config.planner_initial_input, dtype=float)
    planner_max_input_rate = np.asarray(config.planner_max_input_rate, dtype=float)
    if planner_initial_input.shape != (error_model.planning_model.input_dim,):
        raise ValueError("planner_initial_input must match the planning input dimension")
    if planner_max_input_rate.shape != (error_model.planning_model.input_dim,):
        raise ValueError("planner_max_input_rate must match the planning input dimension")
    if np.any(planner_max_input_rate <= 0.0):
        raise ValueError("planner_max_input_rate must be positive")
    if np.any(planner_initial_input < np.asarray(config.planner_v_min)) or np.any(
        planner_initial_input > np.asarray(config.planner_v_max)
    ):
        raise ValueError("planner_initial_input must lie within planner input bounds")
    if config.max_failure_warnings < 0:
        raise ValueError("max_failure_warnings must be nonnegative")
    if config.reference_mode not in ("constant_segments", "planner_mpc"):
        raise ValueError("reference_mode must be 'constant_segments' or 'planner_mpc'")
    if config.fixed_z_goal is not None:
        fixed_z_goal = np.asarray(config.fixed_z_goal, dtype=float)
        if fixed_z_goal.shape != (error_model.planning_model.state_dim,):
            raise ValueError(
                "fixed_z_goal must contain planning position X, Y, and heading psi"
            )
        if np.any(fixed_z_goal < np.asarray(config.planner_z_min, dtype=float)) or np.any(
            fixed_z_goal > np.asarray(config.planner_z_max, dtype=float)
        ):
            raise ValueError("fixed_z_goal must lie within planner_z_min and planner_z_max")
    if not 0.0 <= config.straight_reference_fraction <= 1.0:
        raise ValueError("straight_reference_fraction must be between 0 and 1")
    if not 0.0 <= config.max_reference_yaw_rate <= config.planner_v_max[1]:
        raise ValueError(
            "max_reference_yaw_rate must be nonnegative and no larger than planner_v_max[1]"
        )
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
    skipped_low_speed_count = 0
    processed_steps = 0

    def report_failure(message: str) -> None:
        if failure_count <= config.max_failure_warnings:
            print(f"warning: {message}", flush=True)
        elif failure_count == config.max_failure_warnings + 1:
            print(
                "warning: additional tracking failures will be summarized only",
                flush=True,
            )
    print(
        f"starting data generation: {config.num_trajectories} trajectories, "
        f"{samples_per_trajectory} samples each, reference_mode={config.reference_mode}, "
        f"up to {total_samples} tracking MPC solves",
        flush=True,
    )

    for traj_id in range(config.num_trajectories):
        z0 = sample_uniform(rng, config.z0_low, config.z0_high)
        if config.reference_mode == "constant_segments":
            z_traj, v_traj = rollout_constant_reference_trajectory(
                rng,
                z0,
                config,
                error_model.planning_model,
            )
            z_goal = z_traj[-1]
        else:
            if config.fixed_z_goal is None:
                z_goal = sample_uniform(rng, config.z_goal_low, config.z_goal_high)
            else:
                z_goal = np.asarray(config.fixed_z_goal, dtype=float)
            z_traj, v_traj = rollout_planner_trajectory(
                z0,
                z_goal,
                config,
                planning_mpc,
            )
        x = None
        tracking_segment_steps = 0
        reset_mpc_warm_start(mpc)

        for time_id in range(len(v_traj)):
            processed_steps += 1
            z = z_traj[time_id]
            v = v_traj[time_id]
            v_seq = make_v_sequence(v_traj, time_id, config.mpc_horizon)
            if v[0] < config.min_planning_speed_for_tracking:
                skipped_low_speed_count += 1
                x = None
                tracking_segment_steps = 0
                reset_mpc_warm_start(mpc)
                continue
            if tracking_segment_steps >= config.max_tracking_segment_steps:
                x = None
                tracking_segment_steps = 0
                reset_mpc_warm_start(mpc)
            if x is None or not config.closed_loop_tracking:
                e = sample_error(rng, e_scale, v, config.min_tracking_speed)
                x = state_from_error_numpy(error_model, e, z)
            else:
                e = error_model.error(
                    torch.tensor(x, dtype=torch.float32),
                    torch.tensor(z, dtype=torch.float32),
                ).detach().cpu().numpy()

            if not np.all(np.isfinite(e)) or e[3] <= config.rollout_speed_floor:
                failure_count += 1
                report_failure(
                    f"tracking segment {traj_id} failed at step {time_id} "
                    f"because tracking state left the safe speed domain "
                    f"(v_x={e[3]:.4f})"
                )
                if config.restart_tracking_after_failure:
                    x = None
                    tracking_segment_steps = 0
                    reset_mpc_warm_start(mpc)
                    continue
                break

            x_hat = projection_numpy(error_model, z)

            reconstructed_e = error_model.error(
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(z, dtype=torch.float32),
            ).detach().cpu().numpy()
            if not np.allclose(reconstructed_e, e, atol=1e-5):
                raise RuntimeError("state_from_error and error mappings are inconsistent")

            if config.reset_mpc_each_sample:
                reset_mpc_warm_start(mpc)
            try:
                u_expert, info = mpc.solve(e0=e, z0=z, v_seq=v_seq)
            except ValueError as error:
                # A stale warm start can drive the nominal rollout through the
                # dynamic bicycle singularity even while the real state is safe.
                reset_mpc_warm_start(mpc)
                try:
                    u_expert, info = mpc.solve(e0=e, z0=z, v_seq=v_seq)
                except ValueError as retry_error:
                    failure_count += 1
                    report_failure(
                        f"tracking segment {traj_id} failed at step {time_id} "
                        f"after tracking MPC nominal rollout failed twice: "
                        f"{retry_error}"
                    )
                    if config.restart_tracking_after_failure:
                        x = None
                        tracking_segment_steps = 0
                        continue
                    break
            if not info["success"]:
                failure_count += 1
                report_failure(
                    f"tracking segment {traj_id} failed at step {time_id} "
                    f"because tracking MPC failed: {info['status']}"
                )
                if config.restart_tracking_after_failure:
                    x = None
                    tracking_segment_steps = 0
                    reset_mpc_warm_start(mpc)
                    continue
                break

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

            if config.closed_loop_tracking:
                x = tracking_step_numpy(error_model, x, u_expert, config.dt)
                tracking_segment_steps += 1
                if not np.all(np.isfinite(x)) or x[3] <= config.rollout_speed_floor:
                    failure_count += 1
                    report_failure(
                        f"tracking segment {traj_id} failed after step {time_id} "
                        f"because the next tracking state left the safe speed domain "
                        f"(v_x={x[3]:.4f})"
                    )
                    if config.restart_tracking_after_failure:
                        x = None
                        tracking_segment_steps = 0
                        reset_mpc_warm_start(mpc)
                        continue
                    break

            completed = len(z_values)
            if (
                config.progress_every_samples
                and processed_steps % config.progress_every_samples == 0
            ):
                elapsed = time.perf_counter() - start_time
                steps_per_second = processed_steps / elapsed
                eta_seconds = (total_samples - processed_steps) / steps_per_second
                print(
                    f"processed {processed_steps}/{total_samples} candidate steps, "
                    f"generated {completed} valid samples "
                    f"({failure_count} MPC failures, {skipped_low_speed_count} "
                    f"low-speed planner steps skipped, {elapsed:.1f}s elapsed, "
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
        "config_json": np.asarray(json.dumps(asdict(config))),
        "failure_count": np.asarray(failure_count, dtype=np.int32),
        "skipped_low_speed_count": np.asarray(skipped_low_speed_count, dtype=np.int32),
        "processed_steps": np.asarray(processed_steps, dtype=np.int32),
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
    print(f"tracking MPC failures skipped: {int(dataset['failure_count'])}")
    print(f"low-speed planner steps skipped: {int(dataset['skipped_low_speed_count'])}")


if __name__ == "__main__":
    main()
