import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(PROJECT_ROOT, "data", "plot")


def require_data_file(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing plot data file: {path}\n"
            "Run tests\\TrainingTest.py and tests\\SafetyTrajectory.py first."
        )
    return path


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 24,
            "axes.titlesize": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "axes.grid": True,
            "grid.color": "0.78",
            "grid.linewidth": 0.7,
        }
    )


def plot_training_tracking_and_controls(data) -> None:
    time_state = data["time_state"]
    tracker_xyz = data["tracker_xyz"]
    planner_xyz = data["planner_xyz"]
    time_sample = data["time_sample"]
    control_inputs = data["tracker_control_inputs"]
    displayed_control_inputs = np.column_stack(
        (
            np.rad2deg(control_inputs[:, 0]),
            np.rad2deg(control_inputs[:, 1]),
            0.91 * control_inputs[:, 2],
        )
    )

    fig = plt.figure(figsize=(30, 9))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.08, 1.0), wspace=0.28)
    position_grid = grid[0, 0].subgridspec(
        3, 1, height_ratios=(0.54, 3.0, 0.54), hspace=0.0
    )
    position_ax = fig.add_subplot(position_grid[1, 0])
    control_grid = grid[0, 1].subgridspec(
        7,
        1,
        height_ratios=(0.54, 0.80, 0.30, 0.80, 0.30, 0.80, 0.54),
        hspace=0.0,
    )
    control_axes = [
        fig.add_subplot(control_grid[row, 0]) for row in (1, 3, 5)
    ]

    specs = [
        ("x", 0, "tab:blue"),
        ("y", 1, "tab:orange"),
        ("z", 2, "tab:green"),
    ]
    for label, index, color in specs:
        position_ax.plot(
            time_state,
            tracker_xyz[:, index],
            color=color,
            linestyle="-",
            linewidth=3.0,
            label=rf"Tracker $p_{{{label}}}$",
        )
        position_ax.plot(
            time_state,
            planner_xyz[:, index],
            color=color,
            linestyle="--",
            linewidth=3.0,
            label=rf"Planner $\hat{{p}}_{{{label}}}$",
        )

    position_ax.set_xlabel("Time (sec)")
    position_ax.set_ylabel("Position (m)")
    position_ax.set_xlim(0, 120)
    position_ax.set_xticks([0, 30, 60, 90, 120])
    position_ax.set_ylim(0, 32)
    position_ax.set_yticks([0, 8, 16, 24, 32])
    position_ax.grid(True, linewidth=1.0)
    position_ax.tick_params(width=1.0)
    for spine in position_ax.spines.values():
        spine.set_linewidth(1.0)
    handles, labels = position_ax.get_legend_handles_labels()
    legend_order = [0, 2, 4, 1, 3, 5]
    position_ax.legend(
        [handles[index] for index in legend_order],
        [labels[index] for index in legend_order],
        ncol=2,
        loc="upper left",
    )

    control_labels = [
        r"$a_x$ (deg)",
        r"$a_y$ (deg)",
        r"$a_z$ (m/sec$^2$)",
    ]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    control_limits = [(-0.2, 0.6), (-0.1, 0.3), (9.78, 9.96)]
    control_ticks = [
        [-0.2, 0.2, 0.6],
        [-0.1, 0.1, 0.3],
        [9.78, 9.87, 9.96],
    ]
    for index, (ax, label, color) in enumerate(
        zip(control_axes, control_labels, colors)
    ):
        ax.plot(
            time_sample,
            displayed_control_inputs[:, index],
            color=color,
            linewidth=3.0,
        )
        ax.set_ylabel(label)
        ax.set_xlim(0, 120)
        ax.set_xticks([0, 30, 60, 90, 120])
        ax.set_ylim(*control_limits[index])
        ax.set_yticks(control_ticks[index])
        ax.grid(True, linewidth=1.0)
        ax.tick_params(width=1.0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        if index < 2:
            ax.tick_params(labelbottom=False)

    control_axes[-1].set_xlabel("Time (sec)")

    fig.text(0.275, 0.025, "(a)", ha="center")
    fig.text(0.755, 0.025, "(b)", ha="center")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.13)
    save_figure(fig, "plot_training_tracking_and_control_inputs.pdf")
    plt.close(fig)


def save_figure(fig, filename: str) -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"saved figure: {path}")


def plot_training_position(data) -> None:
    time_state = data["time_state"]
    tracker_xyz = data["tracker_xyz"]
    planner_xyz = data["planner_xyz"]

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    specs = [
        ("x", 0, "tab:blue"),
        ("y", 1, "tab:orange"),
        ("z", 2, "tab:green"),
    ]
    for label, index, color in specs:
        ax.plot(
            time_state,
            tracker_xyz[:, index],
            color=color,
            linestyle="-",
            label=rf"Tracker $p_{{{label}}}$",
        )
        ax.plot(
            time_state,
            planner_xyz[:, index],
            color=color,
            linestyle="--",
            label=f"Planner {label}",
        )

    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Position (m)")
    ax.set_title("Training Position Trajectories")
    ax.legend(ncol=2)
    fig.tight_layout()
    save_figure(fig, "plot_training_position_trajectories.png")
    plt.close(fig)


def plot_position_errors(
    time_sample, position_error, bounds, title: str, filename: str
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    specs = [
        (r"$e_x$", 0, "tab:blue"),
        (r"$e_y$", 1, "tab:orange"),
        (r"$e_z$", 2, "tab:green"),
    ]
    for label, index, color in specs:
        ax.plot(
            time_sample,
            position_error[:, index],
            color=color,
            linewidth=1.8,
            label=label,
        )
        ax.axhline(
            float(bounds[index]),
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
        )
        ax.axhline(
            -float(bounds[index]),
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
        )

    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title(title)
    ax.legend(ncol=3)
    fig.tight_layout()
    save_figure(fig, filename)
    plt.close(fig)


def plot_training_ellipsoid_lyapunov(data) -> None:
    time_sample = data["time_sample"]
    E_value = data["E_value"]
    V = data["V"]
    V_display = np.minimum(V, E_value)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(time_sample, E_value, color="tab:blue", label=r"$e^T E e$")
    ax.plot(time_sample, V_display, color="tab:orange", label=r"$V(e)$")
    ax.axhline(1.0, color="tab:red", linestyle="--", label="level 1")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Value")
    ax.set_title("Ellipsoid and Lyapunov Values")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "plot_training_ellipsoid_lyapunov.png")
    plt.close(fig)


def plot_training_control_inputs(data) -> None:
    time_sample = data["time_sample"]
    control_inputs = data["tracker_control_inputs"]
    displayed_control_inputs = np.column_stack(
        (
            np.rad2deg(control_inputs[:, 0]),
            np.rad2deg(control_inputs[:, 1]),
            0.91 * control_inputs[:, 2],
        )
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    labels = [r"$u_x$", r"$u_y$", r"$u_z$"]
    for index, label in enumerate(labels):
        axes[index].plot(time_sample, control_inputs[:, index], label=label)
        axes[index].set_ylabel(label)
        axes[index].legend()
        axes[index].grid(True)

    axes[-1].set_xlabel("Time (sec)")
    fig.suptitle("Tracker Control Inputs")
    fig.tight_layout()
    save_figure(fig, "plot_training_tracker_control_inputs.png")
    plt.close(fig)


def plot_ellipsoid(ax, center, radii, color, alpha, label=None, wire=False):
    u = np.linspace(0.0, 2.0 * np.pi, 42)
    v = np.linspace(0.0, np.pi, 22)
    xs = center[0] + radii[0] * np.outer(np.cos(u), np.sin(v))
    ys = center[1] + radii[1] * np.outer(np.sin(u), np.sin(v))
    zs = center[2] + radii[2] * np.outer(np.ones_like(u), np.cos(v))
    if wire:
        ax.plot_wireframe(
            xs, ys, zs, color=color, alpha=alpha, linewidth=1.0, label=label
        )
        return
    ax.plot_surface(
        xs, ys, zs, color=color, alpha=alpha, linewidth=0.0, label=label
    )


def set_axes_equal(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = np.mean(limits, axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def plot_safety_trajectory(data) -> None:
    planner_xyz = (
        data["planner_xyz_plot"]
        if "planner_xyz_plot" in data
        else data["planner_xyz"]
    )
    tracker_xyz = (
        data["tracker_xyz_plot"]
        if "tracker_xyz_plot" in data
        else data["tracker_xyz"]
    )
    obstacle_centers = data["obstacle_centers"]
    obstacle_radii = data["obstacle_radii"]
    expanded_obstacle_radii = data["expanded_obstacle_radii"]
    z0 = data["z0"]
    z_goal = data["z_goal"]

    fig = plt.figure(figsize=(8.2, 7.0))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    for index, center in enumerate(obstacle_centers):
        plot_ellipsoid(
            ax,
            center,
            expanded_obstacle_radii[index],
            color="#f2b01e",
            alpha=0.16,
            label="Expanded unsafe region" if index == 0 else None,
            wire=True,
        )
        plot_ellipsoid(
            ax,
            center,
            obstacle_radii[index],
            color="#e6550d",
            alpha=0.58,
            label="Obstacle" if index == 0 else None,
        )

    ax.plot(
        planner_xyz[:, 0],
        planner_xyz[:, 1],
        planner_xyz[:, 2],
        color="#0065b3",
        linewidth=2.5,
        label="Planner",
    )
    ax.plot(
        tracker_xyz[:, 0],
        tracker_xyz[:, 1],
        tracker_xyz[:, 2],
        color="#7b3294",
        linestyle="--",
        linewidth=2.2,
        label="Tracker",
    )
    ax.scatter(
        z0[0], z0[1], z0[2], marker="D", color="#a50026", s=48, label="Start"
    )
    ax.scatter(
        z_goal[0],
        z_goal[1],
        z_goal[2],
        marker="s",
        color="#66a61e",
        s=72,
        label="Goal",
    )
    ax.set_xlabel("X Coordinate (m)", labelpad=8)
    ax.set_ylabel("Y Coordinate (m)", labelpad=8)
    ax.set_zlabel("Z Coordinate (m)", labelpad=8)
    ax.view_init(elev=24, azim=-56)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    set_axes_equal(ax)

    fig.tight_layout()
    save_figure(fig, "plot_safety_trajectory3d.png")
    plt.close(fig)



def plot_safety_tracking_and_errors(data) -> None:
    planner_xyz = (
        data["planner_xyz_plot"]
        if "planner_xyz_plot" in data
        else data["planner_xyz"]
    )
    tracker_xyz = (
        data["tracker_xyz_plot"]
        if "tracker_xyz_plot" in data
        else data["tracker_xyz"]
    )
    obstacle_centers = data["obstacle_centers"]
    obstacle_radii = data["obstacle_radii"]
    expanded_obstacle_radii = data["expanded_obstacle_radii"]
    z0 = data["z0"]
    z_goal = data["z_goal"]
    time_sample = data["time_sample"]
    position_error = np.abs(data["position_error"])
    position_error_bounds = data["position_error_bounds"]

    fig = plt.figure(figsize=(30, 12))
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.0), wspace=0.24)
    trajectory_ax = fig.add_subplot(grid[0, 0], projection="3d")
    error_grid = grid[0, 1].subgridspec(
        3, 1, height_ratios=(0.275, 0.450, 0.275), hspace=0.0
    )
    error_ax = fig.add_subplot(error_grid[1, 0])
    trajectory_ax.set_facecolor("white")
    error_ax.set_facecolor("white")
    for axis in (trajectory_ax.xaxis, trajectory_ax.yaxis, trajectory_ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((0.75, 0.75, 0.75, 1.0))
        axis.pane.set_alpha(1.0)

    for index, center in enumerate(obstacle_centers):
        plot_ellipsoid(
            trajectory_ax,
            center,
            expanded_obstacle_radii[index],
            color="#f2b01e",
            alpha=0.16,
            label="Expanded unsafe region" if index == 0 else None,
            wire=True,
        )
        plot_ellipsoid(
            trajectory_ax,
            center,
            obstacle_radii[index],
            color="#e6550d",
            alpha=0.58,
            label="Obstacle" if index == 0 else None,
        )

    trajectory_ax.plot(
        planner_xyz[:, 0],
        planner_xyz[:, 1],
        planner_xyz[:, 2],
        color="#0065b3",
        linewidth=3.0,
        label="Planner",
    )
    trajectory_ax.plot(
        tracker_xyz[:, 0],
        tracker_xyz[:, 1],
        tracker_xyz[:, 2],
        color="#7b3294",
        linestyle="--",
        linewidth=3.0,
        label="Tracker",
    )
    trajectory_ax.scatter(
        z0[0], z0[1], z0[2], marker="D", color="#a50026", s=90, label="Start"
    )
    trajectory_ax.scatter(
        z_goal[0],
        z_goal[1],
        z_goal[2],
        marker="s",
        color="#66a61e",
        s=110,
        label="Goal",
    )
    trajectory_ax.set_xlabel(r"$p_x$ (m)", labelpad=12)
    trajectory_ax.set_ylabel(r"$p_y$ (m)", labelpad=12)
    trajectory_ax.set_zlabel(r"$p_z$ (m)", labelpad=12)
    trajectory_ax.set_xlim(0, 60)
    trajectory_ax.set_ylim(0, 60)
    trajectory_ax.set_zlim(0, 60)
    trajectory_ax.set_xticks([0, 15, 30, 45, 60])
    trajectory_ax.set_yticks([0, 15, 30, 45, 60])
    trajectory_ax.set_zticks([0, 15, 30, 45, 60])
    trajectory_ax.view_init(elev=24, azim=-56)
    trajectory_ax.grid(True)
    for axis in (trajectory_ax.xaxis, trajectory_ax.yaxis, trajectory_ax.zaxis):
        axis._axinfo["grid"]["linewidth"] = 1.0
        axis.line.set_linewidth(1.0)
    handles, labels = trajectory_ax.get_legend_handles_labels()
    legend_items = dict(zip(labels, handles))
    blank_handle = Line2D([], [], linestyle="none", alpha=0.0)
    legend_handles = [
        legend_items["Obstacle"],
        Line2D([], [], color="#d99200", linewidth=1.0),
        legend_items["Planner"],
        legend_items["Start"],
        blank_handle,
        blank_handle,
        legend_items["Tracker"],
        legend_items["Goal"],
    ]
    legend_labels = [
        "Obstacle",
        "Expanded unsafe region",
        "Planner",
        "Start",
        "",
        "",
        "Tracker",
        "Goal",
    ]
    trajectory_ax.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.050, 0.82),
        ncol=2,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        handlelength=1.45,
        handletextpad=0.45,
        columnspacing=-5.53,
        labelspacing=0.20,
        borderpad=0.30,
        borderaxespad=0.25,
    )

    error_specs = [
        ("x", 0, "tab:blue"),
        ("y", 1, "tab:orange"),
        ("z", 2, "tab:green"),
    ]
    for coordinate, index, color in error_specs:
        error_ax.plot(
            time_sample,
            position_error[:, index],
            color=color,
            linewidth=3.0,
            label=rf"$e_{{{coordinate}}}$",
        )
        error_ax.axhline(
            float(position_error_bounds[index]),
            color=color,
            linestyle="--",
            linewidth=3.0,
            label=rf"$\bar{{e}}_{{{coordinate}}}$",
        )

    error_ax.set_xlabel("Time (sec)")
    error_ax.set_xlim(0, 100)
    error_ax.set_ylim(0, 3.6)
    error_ax.set_xticks([0, 25, 50, 75, 100])
    error_ax.set_yticks([0, 1.2, 2.4, 3.6])
    error_ax.grid(True, linewidth=1.0)
    error_ax.tick_params(width=1.0)
    for spine in error_ax.spines.values():
        spine.set_linewidth(1.0)
    error_handles, _ = error_ax.get_legend_handles_labels()
    error_legend_handles = [
        blank_handle,
        blank_handle,
        error_handles[0],
        error_handles[1],
        error_handles[2],
        error_handles[3],
        error_handles[4],
        error_handles[5],
    ]
    error_legend_labels = [
        "Error",
        "ErrorBound",
        r"$e_x$",
        r"$\bar{e}_x$",
        r"$e_y$",
        r"$\bar{e}_y$",
        r"$e_z$",
        r"$\bar{e}_z$",
    ]
    error_legend = error_ax.legend(
        error_legend_handles,
        error_legend_labels,
        ncol=4,
        loc="upper center",
        handlelength=0.90,
        handletextpad=0.15,
        columnspacing=0.70,
        labelspacing=0.05,
        borderpad=0.05,
        borderaxespad=0.25,
    )
    first_legend_column = error_legend._legend_handle_box.get_children()[0]
    for legend_row in first_legend_column.get_children():
        legend_row.get_children()[0].set_width(0.0)
    fig.text(0.275, 0.025, "(a)", ha="center")
    fig.text(0.755, 0.025, "(b)", ha="center")
    fig.subplots_adjust(left=0.045, right=0.985, top=0.92, bottom=0.13)
    save_figure(fig, "plot_safety_tracking_and_position_errors.pdf")
    plt.close(fig)
def plot_training_figures() -> None:
    data = np.load(
        require_data_file(os.path.join(PLOT_DIR, "training_plot_data.npz"))
    )
    plot_training_tracking_and_controls(data)


def plot_safety_figures() -> None:
    data = np.load(require_data_file(os.path.join(PLOT_DIR, "safety_plot_data.npz")))
    plot_safety_tracking_and_errors(data)


def main() -> None:
    configure_matplotlib()
    plot_training_figures()
    plot_safety_figures()


if __name__ == "__main__":
    main()
