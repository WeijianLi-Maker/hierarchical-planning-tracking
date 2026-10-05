# tests/PlotTest.py

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "plot", "plot_data.npz")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

TITLE_FONT_SIZE = 32
LABEL_FONT_SIZE = 26
TICK_FONT_SIZE = 22
LEGEND_FONT_SIZE = 22
LINE_WIDTH = 3.6
DASHED_LINE_WIDTH = 3.2
GRID_LINE_WIDTH = 1.0

plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.labelsize": LABEL_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
    }
)


def load_plot_data(path=PLOT_DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Plot data not found: {path}. Run tests\\TrainingTest.py first."
        )
    return np.load(path)


def draw_position_legend(ax):
    x0, y0 = 0.57, 0.04
    width, height = 0.39, 0.17
    row_y = [y0 + 0.125, y0 + 0.045]
    text_x = x0 + 0.02
    blue_line_x = [x0 + 0.145, x0 + 0.20]
    blue_label_x = x0 + 0.215
    orange_line_x = [x0 + 0.27, x0 + 0.325]
    orange_label_x = x0 + 0.34

    box = Rectangle(
        (x0, y0),
        width,
        height,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
        clip_on=False,
        zorder=5,
    )
    ax.add_patch(box)

    rows = [
        ("Tracker", "-", r"$p_x$", r"$p_y$"),
        ("Planner", "--", r"$\hat{p}_x$", r"$\hat{p}_y$"),
    ]
    for y, (name, linestyle, x_label, y_label) in zip(row_y, rows):
        ax.text(
            text_x,
            y,
            name,
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=LEGEND_FONT_SIZE,
            zorder=6,
        )
        ax.plot(
            blue_line_x,
            [y, y],
            transform=ax.transAxes,
            color="tab:blue",
            linestyle=linestyle,
            linewidth=DASHED_LINE_WIDTH if linestyle == "--" else LINE_WIDTH,
            clip_on=False,
            zorder=6,
        )
        ax.text(
            blue_label_x,
            y,
            x_label,
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=LEGEND_FONT_SIZE,
            zorder=6,
        )
        ax.plot(
            orange_line_x,
            [y, y],
            transform=ax.transAxes,
            color="tab:orange",
            linestyle=linestyle,
            linewidth=DASHED_LINE_WIDTH if linestyle == "--" else LINE_WIDTH,
            clip_on=False,
            zorder=6,
        )
        ax.text(
            orange_label_x,
            y,
            y_label,
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=LEGEND_FONT_SIZE,
            zorder=6,
        )

def plot_certificate_values(data, results_dir=RESULTS_DIR):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(
        data["time_value"],
        data["e_T_E_e"],
        color="tab:blue",
        linewidth=LINE_WIDTH,
        label=r"$e^T E e$",
    )
    ax.plot(
        data["time_value"],
        data["V_e"],
        color="tab:orange",
        linewidth=LINE_WIDTH,
        label=r"$V(e)$",
    )
    ax.plot(
        data["time_value"],
        data["one"],
        color="tab:red",
        linestyle="--",
        linewidth=DASHED_LINE_WIDTH,
        label="_nolegend_",
    )
    ax.set_xlim(0.0, 12.0)
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_ylim(0.0, 1.2)
    ax.set_yticks([0, 0.3, 0.6, 0.9, 1.2])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("")
    ax.set_title("Certificate Values")
    ax.legend(loc="center right", frameon=True)
    ax.grid(True, linewidth=GRID_LINE_WIDTH, alpha=0.75)
    fig.tight_layout()

    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "CertificateValues.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_position_trajectories(data, results_dir=RESULTS_DIR):
    fig, ax = plt.subplots(figsize=(12, 7))
    tracker_x_line, = ax.plot(
        data["time_position"],
        data["tracker_x"],
        color="tab:blue",
        linewidth=LINE_WIDTH,
    )
    planner_x_line, = ax.plot(
        data["time_position"],
        data["planner_x"],
        color="tab:blue",
        linestyle="--",
        linewidth=DASHED_LINE_WIDTH,
    )
    tracker_y_line, = ax.plot(
        data["time_position"],
        data["tracker_y"],
        color="tab:orange",
        linewidth=LINE_WIDTH,
    )
    planner_y_line, = ax.plot(
        data["time_position"],
        data["planner_y"],
        color="tab:orange",
        linestyle="--",
        linewidth=DASHED_LINE_WIDTH,
    )
    ax.set_xlim(0.0, 12.0)
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_ylim(0.0, 4.5)
    ax.set_yticks([0, 1.5, 3, 4.5])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position")
    ax.set_title("Tracker and Planner Position Trajectories")
    draw_position_legend(ax)

    ax.grid(True, linewidth=GRID_LINE_WIDTH, alpha=0.75)
    fig.tight_layout()

    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "TrackingPlanningPositions.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_combined_figure(data, results_dir=RESULTS_DIR):
    """Plot Fig. 2 and Fig. 3 side by side and save them as a vector PDF."""
    fig, (position_ax, certificate_ax) = plt.subplots(1, 2, figsize=(24, 7))

    position_ax.plot(data["time_position"], data["tracker_x"], color="tab:blue", linewidth=LINE_WIDTH)
    position_ax.plot(data["time_position"], data["planner_x"], color="tab:blue", linestyle="--", linewidth=DASHED_LINE_WIDTH)
    position_ax.plot(data["time_position"], data["tracker_y"], color="tab:orange", linewidth=LINE_WIDTH)
    position_ax.plot(data["time_position"], data["planner_y"], color="tab:orange", linestyle="--", linewidth=DASHED_LINE_WIDTH)
    position_ax.set(xlim=(0.0, 12.0), ylim=(0.0, 4.5), xlabel="Time (s)", ylabel="Position", title="Tracker and Planner Position Trajectories")
    position_ax.set_xticks([0, 3, 6, 9, 12])
    position_ax.set_yticks([0, 1.5, 3, 4.5])
    draw_position_legend(position_ax)
    position_ax.grid(True, linewidth=GRID_LINE_WIDTH, alpha=0.75)

    certificate_ax.plot(data["time_value"], data["e_T_E_e"], color="tab:blue", linewidth=LINE_WIDTH, label=r"$e^T E e$")
    certificate_ax.plot(data["time_value"], data["V_e"], color="tab:orange", linewidth=LINE_WIDTH, label=r"$V(e)$")
    certificate_ax.plot(data["time_value"], data["one"], color="tab:red", linestyle="--", linewidth=DASHED_LINE_WIDTH, label="_nolegend_")
    certificate_ax.set(xlim=(0.0, 12.0), ylim=(0.0, 1.2), xlabel="Time (s)", title="Certificate Values")
    certificate_ax.set_xticks([0, 3, 6, 9, 12])
    certificate_ax.set_yticks([0, 0.3, 0.6, 0.9, 1.2])
    certificate_ax.legend(loc="center right", frameon=True)
    certificate_ax.grid(True, linewidth=GRID_LINE_WIDTH, alpha=0.75)

    fig.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "TrackingAndCertification.pdf")
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return output_path

def main():
    data = load_plot_data()
    combined_path = plot_combined_figure(data)
    print(f"saved: {combined_path}")


if __name__ == "__main__":
    main()














