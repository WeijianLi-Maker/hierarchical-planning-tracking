# tests/SafetyTest.py

import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import CubicSpline, splev, splprep

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.MPCPlanning import LinearizedPlanningMPCController
from controllers.TrackingControl import NeuralTrackingController
from experiments.ErrorBound import load_ellipsoid_matrix, position_error_bounds
from learning.Training import LearnableEllipsoid
from lyapunov.LyapunovNetwork import NeuralLyapunovFunction
from models.error import ErrorDynamics


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "results", "JointTraining", "checkpoint.pt")
TRAJECTORY_FIGURE_PATH = os.path.join(PROJECT_ROOT, "results", "SafetyTrajectory3D.png")
ERROR_FIGURE_PATH = os.path.join(PROJECT_ROOT, "results", "SafetyErrorBound.png")
PLOT_DIR = os.path.join(PROJECT_ROOT, "data", "plot")
POSITION_INDICES = (0, 4, 8)


@dataclass(frozen=True)
class Obstacle:
    center: np.ndarray
    radii: np.ndarray

    @property
    def name(self) -> str:
        return f"obstacle@{np.array2string(self.center, precision=1)}"


@dataclass
class SafetyFigureConfig:
    dt: float = 0.08
    rollout_duration: float = 100.0
    planner_mpc_horizon: int = 24
    planner_v_rate_weight: float = 4.0
    reference_path_samples: int = 1000
    reference_lookahead_distance: float = 5.0
    plot_curve_samples: int = 1400
    plot_curve_smoothing: float = 180.0
    waypoint_tolerance: float = 0.75
    final_goal_tolerance: float = 0.85
    waypoint_margin: float = 2.2
    v_limit: float = 0.7
    z_min: tuple[float, float, float] = (-4.0, -12.0, 0.0)
    z_max: tuple[float, float, float] = (54.0, 50.0, 54.0)
    z0: tuple[float, float, float] = (0.0, 0.0, 1.0)
    z_goal: tuple[float, float, float] = (50.0, 45.0, 50.0)
    trajectory_save_path: str = TRAJECTORY_FIGURE_PATH
    error_save_path: str = ERROR_FIGURE_PATH
    save_plot_data: bool = True
    plot_dir: str = PLOT_DIR
    show_plot: bool = False


def load_models_from_checkpoint(path: str = CHECKPOINT_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run experiments\\JointTraining.py first: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    error_model = ErrorDynamics()
    tracking_model = error_model.tracking_model

    lyapunov = NeuralLyapunovFunction(
        error_dim=error_model.error_dim,
        hidden_sizes=(64, 64),
        feature_dim=16,
        delta=0.02,
    )
    controller = NeuralTrackingController(
        hidden_sizes=(64, 64),
        g=tracking_model.g,
        kT=tracking_model.kT,
    )
    ellipsoid = LearnableEllipsoid(dim=error_model.error_dim, init_scale=1.0)

    lyapunov.load_state_dict(checkpoint["lyapunov_state_dict"])
    controller.load_state_dict(checkpoint["controller_state_dict"])
    ellipsoid.load_state_dict(checkpoint["ellipsoid_state_dict"])

    lyapunov.eval()
    controller.eval()
    ellipsoid.eval()

    return error_model, controller, ellipsoid


def make_obstacles() -> list[Obstacle]:
    return [
        Obstacle(center=np.array([12.0, 15.2, 8.5]), radii=np.array([6.0, 5.3, 4.9])),
        Obstacle(center=np.array([29.0, 17.5, 31.0]), radii=np.array([7.2, 6.3, 5.7])),
        Obstacle(center=np.array([39.0, 41.0, 31.5]), radii=np.array([6.4, 7.3, 5.4])),
    ]


def expanded_radii(obstacle: Obstacle, position_error_bounds_value: np.ndarray) -> np.ndarray:
    return obstacle.radii + position_error_bounds_value


def segment_intersects_ellipsoid(
    start: np.ndarray,
    end: np.ndarray,
    center: np.ndarray,
    radii: np.ndarray,
) -> bool:
    start_scaled = (start - center) / radii
    direction_scaled = (end - start) / radii
    a = float(direction_scaled @ direction_scaled)
    b = float(2.0 * start_scaled @ direction_scaled)
    c = float(start_scaled @ start_scaled - 1.0)
    if a <= 1e-12:
        return c <= 0.0
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return False
    root = np.sqrt(discriminant)
    t0 = (-b - root) / (2.0 * a)
    t1 = (-b + root) / (2.0 * a)
    return (0.0 <= t0 <= 1.0) or (0.0 <= t1 <= 1.0) or (t0 < 0.0 and t1 > 1.0)


def build_obstacle_aware_waypoints(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: list[Obstacle],
    position_error_bounds_value: np.ndarray,
    margin: float,
) -> np.ndarray:
    waypoints = [start]
    route_start = start.copy()
    route_direction = goal - start
    route_direction /= np.linalg.norm(route_direction)
    side_axis = np.array([0.0, 1.0, 0.25])
    side_axis -= route_direction * float(side_axis @ route_direction)
    side_axis /= np.linalg.norm(side_axis)

    obstacle_order = sorted(
        obstacles,
        key=lambda obstacle: float((obstacle.center - start) @ route_direction),
    )
    for index, obstacle in enumerate(obstacle_order):
        radii = expanded_radii(obstacle, position_error_bounds_value)
        if not segment_intersects_ellipsoid(route_start, goal, obstacle.center, radii):
            continue
        side = -1.0 if index % 2 == 0 else 1.0
        avoidance_offset = side * side_axis * (np.max(radii) + margin)
        waypoint = obstacle.center + avoidance_offset
        waypoint[2] = np.clip(waypoint[2] + 0.4, 0.8, goal[2] + 1.0)
        waypoints.append(waypoint)
        route_start = waypoint

    waypoints.append(goal)
    return np.asarray(waypoints, dtype=float)


def smooth_reference_path(waypoints: np.ndarray, samples: int) -> tuple[np.ndarray, np.ndarray]:
    segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    arc_length = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    if arc_length[-1] <= 1e-8:
        return waypoints.copy(), arc_length

    unique_indices = np.concatenate([[True], np.diff(arc_length) > 1e-8])
    arc_length = arc_length[unique_indices]
    waypoints = waypoints[unique_indices]

    query = np.linspace(0.0, arc_length[-1], int(samples))
    reference = np.zeros((query.shape[0], waypoints.shape[1]))
    for dim in range(waypoints.shape[1]):
        spline = CubicSpline(arc_length, waypoints[:, dim], bc_type="natural")
        reference[:, dim] = spline(query)

    reference[0] = waypoints[0]
    reference[-1] = waypoints[-1]
    return reference, query


def select_reference_target(
    current_z: np.ndarray,
    reference_path: np.ndarray,
    reference_s: np.ndarray,
    previous_index: int,
    lookahead_distance: float,
) -> tuple[np.ndarray, int]:
    search_start = max(0, previous_index - 8)
    distances = np.linalg.norm(reference_path[search_start:] - current_z, axis=1)
    closest_index = search_start + int(np.argmin(distances))
    closest_index = max(previous_index, closest_index)
    target_s = reference_s[closest_index] + lookahead_distance
    target_index = int(np.searchsorted(reference_s, target_s, side="left"))
    target_index = min(target_index, len(reference_path) - 1)
    return reference_path[target_index], closest_index


def smooth_curve_for_plot(points: np.ndarray, samples: int, smoothing: float) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc_length = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    if arc_length[-1] <= 1e-8 or len(points) < 4:
        return points.copy()

    unique_indices = np.concatenate([[True], np.diff(arc_length) > 1e-8])
    arc_length = arc_length[unique_indices]
    points = points[unique_indices]
    normalized_s = arc_length / arc_length[-1]
    query_u = np.linspace(0.0, 1.0, int(samples))

    try:
        spline, _ = splprep(
            points.T,
            u=normalized_s,
            s=float(smoothing),
            k=min(3, len(points) - 1),
        )
        smooth_points = np.asarray(splev(query_u, spline)).T
        smooth_points[0] = points[0]
        smooth_points[-1] = points[-1]
        return smooth_points
    except ValueError:
        pass

    query = query_u * arc_length[-1]

    smooth_points = np.zeros((query.shape[0], points.shape[1]))
    for dim in range(points.shape[1]):
        spline = CubicSpline(arc_length, points[:, dim], bc_type="natural")
        smooth_points[:, dim] = spline(query)

    smooth_points[0] = points[0]
    smooth_points[-1] = points[-1]
    return smooth_points


def tangent_obstacle_constraints(
    z: np.ndarray,
    target: np.ndarray,
    obstacles: list[Obstacle],
    position_error_bounds_value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    lower = []
    upper = []
    for obstacle in obstacles:
        radii = expanded_radii(obstacle, position_error_bounds_value)
        scaled = (z - obstacle.center) / radii
        scaled_norm = float(np.linalg.norm(scaled))
        if scaled_norm <= 1e-8:
            continue
        surface_point = obstacle.center + radii * scaled / max(scaled_norm, 1.0)
        normal = (surface_point - obstacle.center) / (radii * radii)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-8:
            continue
        normal = normal / normal_norm
        offset = float(normal @ surface_point)

        current_margin = float(normal @ z - offset)
        target_margin = float(normal @ target - offset)
        if current_margin >= -1e-8 and target_margin >= -1e-8:
            rows.append(normal)
            lower.append(offset)
            upper.append(np.inf)

    if not rows:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0)
    return np.vstack(rows), np.asarray(lower), np.asarray(upper)


def tracking_state_from_position(position: np.ndarray) -> torch.Tensor:
    x = torch.zeros(10, dtype=torch.float32)
    x[0] = float(position[0])
    x[4] = float(position[1])
    x[8] = float(position[2])
    return x


def rollout_safety_trajectory(config: SafetyFigureConfig):
    error_model, controller, ellipsoid = load_models_from_checkpoint()
    E = load_ellipsoid_matrix()
    position_bounds, _ = position_error_bounds(E)
    position_bounds_np = position_bounds.detach().cpu().numpy().astype(float)

    obstacles = make_obstacles()
    start = np.asarray(config.z0, dtype=float)
    goal = np.asarray(config.z_goal, dtype=float)
    waypoints = build_obstacle_aware_waypoints(
        start=start,
        goal=goal,
        obstacles=obstacles,
        position_error_bounds_value=position_bounds_np,
        margin=config.waypoint_margin,
    )
    reference_path, reference_s = smooth_reference_path(
        waypoints,
        samples=config.reference_path_samples,
    )

    z = torch.tensor(start, dtype=torch.float32)
    x = tracking_state_from_position(start)
    reference_index = 0
    last_solution = None
    previous_v = None

    z_hist = [z.detach().clone()]
    x_hist = [x.detach().clone()]
    e_hist = []
    u_hist = []
    v_hist = []
    active_waypoint_hist = []
    planning_success = []

    z_min = np.asarray(config.z_min, dtype=float)
    z_max = np.asarray(config.z_max, dtype=float)
    v_min = -config.v_limit * np.ones(3)
    v_max = config.v_limit * np.ones(3)
    rollout_steps = int(round(config.rollout_duration / config.dt))

    for step in range(rollout_steps):
        current_z_np = z.detach().cpu().numpy()
        active_waypoint, reference_index = select_reference_target(
            current_z_np,
            reference_path,
            reference_s,
            reference_index,
            config.reference_lookahead_distance,
        )
        A_obs, lower_obs, upper_obs = tangent_obstacle_constraints(
            current_z_np,
            active_waypoint,
            obstacles,
            position_bounds_np,
        )
        planning_mpc = LinearizedPlanningMPCController(
            planning_model=error_model.planning_model,
            dt=config.dt,
            horizon=config.planner_mpc_horizon,
            Q=np.diag([6.0, 6.0, 8.0]),
            R=0.04 * np.eye(3),
            Rd=config.planner_v_rate_weight * np.eye(3),
            Qf=25.0 * np.eye(3),
            z_min=z_min,
            z_max=z_max,
            z_constraint_matrix=A_obs if A_obs.shape[0] else None,
            z_constraint_lower=lower_obs if A_obs.shape[0] else None,
            z_constraint_upper=upper_obs if A_obs.shape[0] else None,
            v_min=v_min,
            v_max=v_max,
        )
        if last_solution is not None:
            planning_mpc.last_solution = last_solution
        if previous_v is not None:
            planning_mpc.previous_v = previous_v

        v_np, info = planning_mpc.solve(current_z_np, active_waypoint)
        if not info["success"]:
            raise AssertionError(f"Planning MPC failed at step {step}: {info['status']}")
        last_solution = planning_mpc.last_solution
        previous_v = planning_mpc.previous_v
        v = torch.as_tensor(v_np, dtype=torch.float32)

        with torch.no_grad():
            e = error_model.error(x, z)
            u = controller(e, v)
            x = x + config.dt * error_model.tracking_model.dynamics(x, u)
            z = z + config.dt * error_model.planning_model.dynamics(z, v)

        x_hist.append(x.detach().clone())
        z_hist.append(z.detach().clone())
        e_hist.append(e.detach().clone())
        u_hist.append(u.detach().clone())
        v_hist.append(v.detach().clone())
        active_waypoint_hist.append(reference_index)
        planning_success.append(info["success"])

    traj = {
        "x": torch.stack(x_hist),
        "z": torch.stack(z_hist),
        "e": torch.stack(e_hist),
        "u": torch.stack(u_hist),
        "v": torch.stack(v_hist),
        "waypoints": waypoints,
        "reference_path": reference_path,
        "active_waypoint": np.asarray(active_waypoint_hist),
        "planning_success": np.asarray(planning_success, dtype=bool),
        "position_bounds": position_bounds_np,
        "position_error_bound_norm": float(np.linalg.norm(position_bounds_np)),
        "obstacles": obstacles,
        "ellipsoid": ellipsoid,
    }
    return traj


def ellipsoid_margin(points: np.ndarray, obstacle: Obstacle, radii: np.ndarray) -> np.ndarray:
    scaled = (points - obstacle.center) / radii
    return np.linalg.norm(scaled, axis=1) - 1.0


def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = np.mean(limits, axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def plot_ellipsoid(ax, center, radii, color, alpha, label=None, wire=False):
    u = np.linspace(0.0, 2.0 * np.pi, 42)
    v = np.linspace(0.0, np.pi, 22)
    xs = center[0] + radii[0] * np.outer(np.cos(u), np.sin(v))
    ys = center[1] + radii[1] * np.outer(np.sin(u), np.sin(v))
    zs = center[2] + radii[2] * np.outer(np.ones_like(u), np.cos(v))
    if wire:
        ax.plot_wireframe(xs, ys, zs, color=color, alpha=alpha, linewidth=0.45, label=label)
    else:
        ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0.0, label=label)


def plot_safety_figure(traj, config: SafetyFigureConfig):
    z = traj["z"].detach().cpu().numpy()
    x = traj["x"].detach().cpu().numpy()
    x_pos = x[:, POSITION_INDICES]
    z_plot = smooth_curve_for_plot(
        z,
        config.plot_curve_samples,
        config.plot_curve_smoothing,
    )
    x_pos_plot = smooth_curve_for_plot(
        x_pos,
        config.plot_curve_samples,
        config.plot_curve_smoothing,
    )
    time_error = np.arange(traj["e"].shape[0]) * config.dt
    position_error = traj["e"].detach().cpu().numpy()[:, POSITION_INDICES]
    position_bounds = traj["position_bounds"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.grid": True,
            "grid.color": "0.78",
            "grid.linewidth": 0.7,
        }
    )
    trajectory_fig = plt.figure(figsize=(8.2, 7.0))
    ax_traj = trajectory_fig.add_subplot(1, 1, 1, projection="3d")

    for index, obstacle in enumerate(traj["obstacles"]):
        plot_ellipsoid(
            ax_traj,
            obstacle.center,
            expanded_radii(obstacle, traj["position_bounds"]),
            color="#f2b01e",
            alpha=0.16,
            label="Expanded unsafe region" if index == 0 else None,
            wire=True,
        )
        plot_ellipsoid(
            ax_traj,
            obstacle.center,
            obstacle.radii,
            color="#e6550d",
            alpha=0.58,
            label="Obstacle" if index == 0 else None,
        )

    ax_traj.plot(
        z_plot[:, 0],
        z_plot[:, 1],
        z_plot[:, 2],
        color="#0065b3",
        linewidth=2.5,
        label="Planner",
    )
    ax_traj.plot(
        x_pos_plot[:, 0],
        x_pos_plot[:, 1],
        x_pos_plot[:, 2],
        color="#7b3294",
        linestyle="--",
        linewidth=2.2,
        label="Tracker",
    )
    ax_traj.scatter(z[0, 0], z[0, 1], z[0, 2], marker="D", color="#a50026", s=48, label="Start")
    ax_traj.scatter(
        traj["waypoints"][-1, 0],
        traj["waypoints"][-1, 1],
        traj["waypoints"][-1, 2],
        marker="s",
        color="#66a61e",
        s=72,
        label="Goal",
    )
    ax_traj.set_xlabel("X Coordinate (m)", labelpad=8)
    ax_traj.set_ylabel("Y Coordinate (m)", labelpad=8)
    ax_traj.set_zlabel("Z Coordinate (m)", labelpad=8)
    ax_traj.view_init(elev=24, azim=-56)
    ax_traj.legend(loc="upper left", frameon=True, framealpha=0.95)
    set_axes_equal(ax_traj)

    trajectory_fig.tight_layout()
    os.makedirs(os.path.dirname(config.trajectory_save_path), exist_ok=True)
    trajectory_fig.savefig(config.trajectory_save_path, dpi=300, bbox_inches="tight")

    error_fig, ax_err = plt.subplots(1, 1, figsize=(8.2, 5.2))
    error_specs = [
        (r"$e_x$", 0, "#0065b3"),
        (r"$e_y$", 1, "#7b3294"),
        (r"$e_z$", 2, "#1b9e77"),
    ]
    for label, index, color in error_specs:
        ax_err.plot(time_error, position_error[:, index], color=color, linewidth=2.2, label=label)
        bound = float(position_bounds[index])
        ax_err.axhline(bound, color=color, linestyle="--", linewidth=1.1, alpha=0.75)
        ax_err.axhline(-bound, color=color, linestyle="--", linewidth=1.1, alpha=0.75)
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Position Error (m)")
    ax_err.set_xlim(time_error[0], time_error[-1] if len(time_error) else config.dt)
    ax_err.legend(loc="upper right", ncol=3, frameon=True, framealpha=0.95)

    error_fig.tight_layout()
    os.makedirs(os.path.dirname(config.error_save_path), exist_ok=True)
    error_fig.savefig(config.error_save_path, dpi=300, bbox_inches="tight")

    if config.show_plot:
        plt.show()
    else:
        plt.close(trajectory_fig)
        plt.close(error_fig)


def save_plot_data(traj, config: SafetyFigureConfig):
    os.makedirs(config.plot_dir, exist_ok=True)

    z = traj["z"].detach().cpu().numpy()
    x = traj["x"].detach().cpu().numpy()
    e = traj["e"].detach().cpu().numpy()
    planner_xyz = z
    tracker_xyz = x[:, POSITION_INDICES]
    position_error = e[:, POSITION_INDICES]
    position_bounds = traj["position_bounds"]
    time_state = np.arange(z.shape[0]) * config.dt
    time_sample = np.arange(e.shape[0]) * config.dt
    obstacle_centers = np.asarray([obstacle.center for obstacle in traj["obstacles"]])
    obstacle_radii = np.asarray([obstacle.radii for obstacle in traj["obstacles"]])
    expanded_obstacle_radii = np.asarray(
        [expanded_radii(obstacle, position_bounds) for obstacle in traj["obstacles"]]
    )

    np.savez_compressed(
        os.path.join(config.plot_dir, "safety_plot_data.npz"),
        time_state=time_state,
        time_sample=time_sample,
        planner_xyz=planner_xyz,
        tracker_xyz=tracker_xyz,
        planner_xyz_plot=smooth_curve_for_plot(
            planner_xyz,
            config.plot_curve_samples,
            config.plot_curve_smoothing,
        ),
        tracker_xyz_plot=smooth_curve_for_plot(
            tracker_xyz,
            config.plot_curve_samples,
            config.plot_curve_smoothing,
        ),
        position_error=position_error,
        position_error_bounds=position_bounds,
        waypoints=traj["waypoints"],
        reference_path=traj["reference_path"],
        obstacle_centers=obstacle_centers,
        obstacle_radii=obstacle_radii,
        expanded_obstacle_radii=expanded_obstacle_radii,
        z0=np.asarray(config.z0, dtype=float),
        z_goal=np.asarray(config.z_goal, dtype=float),
    )

    np.savetxt(
        os.path.join(config.plot_dir, "safety_trajectory_xyz.csv"),
        np.column_stack([time_state, tracker_xyz, planner_xyz]),
        delimiter=",",
        header="time,tracker_x,tracker_y,tracker_z,planner_x,planner_y,planner_z",
        comments="",
    )
    np.savetxt(
        os.path.join(config.plot_dir, "safety_position_errors.csv"),
        np.column_stack(
            [
                time_sample,
                position_error,
                np.tile(position_bounds, (time_sample.shape[0], 1)),
                -np.tile(position_bounds, (time_sample.shape[0], 1)),
            ]
        ),
        delimiter=",",
        header=(
            "time,e_x,e_y,e_z,"
            "bound_x,bound_y,bound_z,"
            "neg_bound_x,neg_bound_y,neg_bound_z"
        ),
        comments="",
    )
    np.savetxt(
        os.path.join(config.plot_dir, "safety_obstacles.csv"),
        np.column_stack([obstacle_centers, obstacle_radii, expanded_obstacle_radii]),
        delimiter=",",
        header=(
            "center_x,center_y,center_z,"
            "radius_x,radius_y,radius_z,"
            "expanded_radius_x,expanded_radius_y,expanded_radius_z"
        ),
        comments="",
    )

    print(f"saved SafetyTrajectory plot data to: {config.plot_dir}")


def test_safety_figure_rollout():
    config = SafetyFigureConfig(show_plot=False)
    traj = rollout_safety_trajectory(config)
    z = traj["z"].detach().cpu().numpy()
    x = traj["x"].detach().cpu().numpy()[:, POSITION_INDICES]

    if not np.all(traj["planning_success"]):
        raise AssertionError("Expected every planning MPC solve to succeed")
    final_goal_error = np.linalg.norm(z[-1] - np.asarray(config.z_goal, dtype=float))
    if final_goal_error > config.final_goal_tolerance:
        raise AssertionError(
            f"Planner did not reach the goal: final_error={final_goal_error:.3f} m"
        )
    for obstacle in traj["obstacles"]:
        planner_margin = ellipsoid_margin(
            z,
            obstacle,
            expanded_radii(obstacle, traj["position_bounds"]),
        )
        tracker_margin = ellipsoid_margin(x, obstacle, obstacle.radii)
        if np.min(planner_margin) < -1e-3:
            raise AssertionError(f"Planner entered expanded unsafe region for {obstacle.name}")
        if np.min(tracker_margin) < -1e-3:
            raise AssertionError(f"Tracker entered obstacle for {obstacle.name}")

    plot_safety_figure(traj, config)
    if config.save_plot_data:
        save_plot_data(traj, config)
    print(f"saved safety trajectory figure to: {config.trajectory_save_path}")
    print(f"saved safety error figure to: {config.error_save_path}")
    return traj


def main():
    show_plot = os.environ.get("SHOW_SAFETY_PLOT", "0") == "1"
    config = SafetyFigureConfig(show_plot=show_plot)
    traj = rollout_safety_trajectory(config)
    plot_safety_figure(traj, config)
    if config.save_plot_data:
        save_plot_data(traj, config)
    print(f"saved safety trajectory figure to: {config.trajectory_save_path}")
    print(f"saved safety error figure to: {config.error_save_path}")


if __name__ == "__main__":
    main()
