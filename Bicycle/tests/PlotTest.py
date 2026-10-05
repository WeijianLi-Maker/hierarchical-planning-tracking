import os
import sys

import matplotlib.pyplot as plt
import numpy as np
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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "plot")
XY_DATA_PATH = os.path.join(PLOT_DATA_DIR, "planner_tracker_xy.csv")
INPUT_DATA_PATH = os.path.join(PLOT_DATA_DIR, "neural_network_input.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
XY_FIGURE_PATH = os.path.join(RESULTS_DIR, "planner_tracker_xy.png")
INPUT_FIGURE_PATH = os.path.join(RESULTS_DIR, "tracker_control_inputs.png")
COMBINED_FIGURE_PATH = os.path.join(RESULTS_DIR, "control_and_tracking.pdf")


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected plot data file: {path}")
    return np.genfromtxt(path, delimiter=",", names=True)


def plot_planner_tracker_xy(data):
    time = data["time"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 8.4), sharex=True)

    axes[0].plot(time, data["planner_x"], color="tab:blue", linewidth=3.0, linestyle="--", label=r"Planner $X$")
    axes[0].plot(time, data["tracker_x"], color="tab:orange", linewidth=3.0, label=r"Tracker $\hat{X}$")
    axes[0].set_ylabel("")
    axes[0].set_xlim(0.0, 20.0)
    axes[0].set_ylim(0.0, 60.0)
    axes[0].set_xticks([0, 5, 10, 15, 20])
    axes[0].set_yticks([0, 20, 40, 60])
    axes[0].legend(loc="best")

    axes[1].plot(time, data["planner_y"], color="tab:blue", linewidth=3.0, linestyle="--", label=r"Planner $Y$")
    axes[1].plot(time, data["tracker_y"], color="tab:orange", linewidth=3.0, label=r"Tracker $\hat{Y}$")
    axes[1].set_ylabel("")
    axes[1].set_xlabel("Time (sec)")
    axes[1].set_xlim(0.0, 20.0)
    axes[1].set_ylim(0.0, 120.0)
    axes[1].set_xticks([0, 5, 10, 15, 20])
    axes[1].set_yticks([0, 40, 80, 120])
    axes[1].legend(loc="best")

    for axis in axes:
        axis.grid(True, color="0.75", linewidth=1.0, alpha=0.7)
        axis.tick_params(direction="in")
        for spine in axis.spines.values():
            spine.set_color("0.35")

    fig.tight_layout(pad=0.25)
    return fig


def plot_tracker_control_inputs(data):
    time = data["time"]
    steering_deg = np.rad2deg(data["delta_f"])
    acceleration = data["a_x"]

    fig, axes = plt.subplots(2, 1, figsize=(15, 8.4), sharex=True)

    axes[0].plot(time, steering_deg, color="tab:blue", linewidth=3.0)
    axes[0].set_ylabel(r"$\delta_f$ (deg)")

    axes[1].plot(time, acceleration, color="tab:orange", linewidth=3.0)
    axes[1].set_ylabel(r"$a_x$ (m/s$^2$)")
    axes[1].set_xlabel("Time (sec)")
    axes[1].set_xlim(0.0, 20.0)
    axes[1].set_ylim(-4, 4)
    axes[1].set_xticks([0, 5, 10, 15, 20])
    axes[1].set_yticks([-4, -2, 0, 2, 4])

    for axis in axes:
        axis.grid(True, color="0.75", linewidth=1.0, alpha=0.7)
        axis.tick_params(direction="in")
        for spine in axis.spines.values():
            spine.set_color("0.35")

    fig.tight_layout(pad=0.25)
    return fig



def plot_combined_figure(input_data, xy_data):
    """Place the control inputs on the left and tracking results on the right."""
    fig, axes = plt.subplots(2, 2, figsize=(30, 8.4), sharex="col")
    control_axes = axes[:, 0]
    position_axes = axes[:, 1]

    input_time = input_data["time"]
    control_axes[0].plot(
        input_time, np.rad2deg(input_data["delta_f"]),
        color="tab:blue", linewidth=3.0,
    )
    control_axes[0].set_ylabel(r"$\delta_f$ (deg)")
    control_axes[1].plot(
        input_time, input_data["a_x"],
        color="tab:orange", linewidth=3.0,
    )
    control_axes[1].set_ylabel(r"$a_x$ (m/s$^2$)")
    control_axes[1].set_xlabel("Time (sec)")
    control_axes[1].set_xlim(0.0, 20.0)
    control_axes[1].set_ylim(-4, 4)
    control_axes[1].set_xticks([0, 5, 10, 15, 20])
    control_axes[1].set_yticks([-4, -2, 0, 2, 4])

    xy_time = xy_data["time"]
    position_axes[0].plot(
        xy_time, xy_data["planner_x"], color="tab:blue",
        linewidth=3.0, linestyle="--", label=r"Planner $X$",
    )
    position_axes[0].plot(
        xy_time, xy_data["tracker_x"], color="tab:orange",
        linewidth=3.0, label=r"Tracker $\hat{X}$",
    )
    position_axes[0].set_ylabel("")
    position_axes[0].set_xlim(0.0, 20.0)
    position_axes[0].set_ylim(0.0, 60.0)
    position_axes[0].set_xticks([0, 5, 10, 15, 20])
    position_axes[0].set_yticks([0, 20, 40, 60])
    position_axes[0].legend(loc="best")
    position_axes[1].plot(
        xy_time, xy_data["planner_y"], color="tab:blue",
        linewidth=3.0, linestyle="--", label=r"Planner $Y$",
    )
    position_axes[1].plot(
        xy_time, xy_data["tracker_y"], color="tab:orange",
        linewidth=3.0, label=r"Tracker $\hat{Y}$",
    )
    position_axes[1].set_ylabel("")
    position_axes[1].set_xlabel("Time (sec)")
    position_axes[1].set_xlim(0.0, 20.0)
    position_axes[1].set_ylim(0.0, 120.0)
    position_axes[1].set_xticks([0, 5, 10, 15, 20])
    position_axes[1].set_yticks([0, 40, 80, 120])
    position_axes[1].legend(loc="best")

    for axis in axes.flat:
        axis.grid(True, color="0.75", linewidth=1.0, alpha=0.7)
        axis.tick_params(direction="in")
        for spine in axis.spines.values():
            spine.set_color("0.35")

    fig.tight_layout(pad=0.5, w_pad=2.0, rect=(0.0, 0.10, 1.0, 1.0))
    fig.text(0.25, 0.015, "(a)", ha="center", va="bottom")
    fig.text(0.75, 0.015, "(b)", ha="center", va="bottom")
    return fig

def save_plot_figures(show=False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    xy_data = load_csv(XY_DATA_PATH)
    input_data = load_csv(INPUT_DATA_PATH)

    xy_figure = plot_planner_tracker_xy(xy_data)
    input_figure = plot_tracker_control_inputs(input_data)
    combined_figure = plot_combined_figure(input_data, xy_data)

    xy_figure.savefig(XY_FIGURE_PATH, dpi=200)
    input_figure.savefig(INPUT_FIGURE_PATH, dpi=200)
    combined_figure.savefig(COMBINED_FIGURE_PATH, format="pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(xy_figure)
        plt.close(input_figure)
        plt.close(combined_figure)

    return XY_FIGURE_PATH, INPUT_FIGURE_PATH


def test_plot_data_files_generate_figures():
    xy_path, input_path = save_plot_figures(show=False)
    for path in (xy_path, input_path):
        if not os.path.exists(path):
            raise AssertionError(f"Expected figure to be saved: {path}")
        if os.path.getsize(path) == 0:
            raise AssertionError(f"Saved figure is empty: {path}")

    print(f"saved XY trajectory figure: {xy_path}")
    print(f"saved input trajectory figure: {input_path}")


if __name__ == "__main__":
    save_plot_figures(show=True)












