"""Controlled ablation of SMT counterexample replay for certificate learning."""

import copy
import os
import sys
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from experiments.JointTraining import (
    JointTrainingConfig,
    add_equilibrium_anchors,
    load_expert_dataset,
    make_models,
    split_dataset,
)
from learning.LossFunction import LossWeights, moving_reference_error
from learning.Training import TrainingConfig, make_optimizer, train_step
from models.error import ErrorDynamics
from SMT.CounterexampleBuffer import CounterexampleBuffer, CounterexampleBufferConfig
from SMT.SMTSolver import SMTConfig, SMTSolver

BC_CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, "results", "BehaviorCloning", "checkpoint.pt"
)
CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, "results", "CompareSMT", "checkpoint.pt"
)
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "Compare_SMT.pdf")


@dataclass
class CompareSMTConfig:
    seed: int = 17
    steps: int = 3000
    batch_size: int = 200
    evaluation_every: int = 100
    probe_samples: int = 20000
    probe_chunk_size: int = 1000
    smt_steps: tuple[int, ...] = (500, 1000, 1500, 2000, 2500)
    # Keep searched samples a small part of each batch. Replacing half of a
    # batch with correlated neighborhoods destabilizes certificate learning.
    counterexample_fraction: float = 0.05
    ellipsoid_counterexample_fraction: float = 0.10
    counterexample_warmup_steps: int = 200
    counterexample_neighborhood_samples: int = 8
    equilibrium_anchor_fraction: float = 0.25
    learning_rate: float = 5e-4
    gradient_clip_norm: float = 2.0
    lyapunov_weight: float = 4.0
    error_bound_weight: float = 2.0
    logdet_weight: float = 1e-3
    level: float = 1.0
    decrease_margin: float = 0.01
    origin_epsilon: float = 1e-4
    smt_random_samples: int = 4096
    smt_refinement_steps: int = 120
    smt_refinement_restarts: int = 8
    dt: float = 0.02
    rollout_seconds: float = 12.0
    planning_input: tuple[float, float] = (8.0, 0.15)
    initial_reference_error: tuple[float, ...] = (
        0.35, -0.20, 0.03, 0.50, 0.08, 0.03
    )
    minimum_tracking_speed: float = 1.0
    enforce_monotone_refinement: bool = True
    device: str = "cpu"
    show_plot: bool = True


def load_frozen_bc_controller(controller) -> None:
    if not os.path.exists(BC_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Run experiments/BehaviorCloning.py first: {BC_CHECKPOINT_PATH}"
        )
    checkpoint = torch.load(
        BC_CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    if checkpoint.get("training_mode") != "behavior_cloning_only":
        raise ValueError("Expected a behavior-cloning-only checkpoint")
    controller.load_state_dict(checkpoint["controller_state_dict"])
    controller.eval()
    for parameter in controller.parameters():
        parameter.requires_grad_(False)


def make_identical_variants(error_model, device):
    torch.manual_seed(0)
    base = make_models(error_model, device)
    load_frozen_bc_controller(base[1])
    # NeuralLyapunovFunction initializes C's last layer to exactly zero.
    # Since V contains ||C(r)||^2, that makes the neural term's parameter
    # gradient exactly zero.  Use one shared small nonzero initialization so
    # both ablation variants can actually learn while remaining identical.
    final_layer = base[0].C[-1]
    torch.nn.init.xavier_uniform_(final_layer.weight, gain=0.1)
    torch.nn.init.zeros_(final_layer.bias)
    return (
        tuple(copy.deepcopy(model) for model in base),
        tuple(copy.deepcopy(model) for model in base),
    )


def sample_box(low, high, count, generator):
    low = torch.tensor(low, dtype=torch.float32)
    high = torch.tensor(high, dtype=torch.float32)
    return low + torch.rand(count, len(low), generator=generator) * (high - low)


def make_fixed_probe_set(smt_config, count, seed):
    generator = torch.Generator().manual_seed(seed)
    return {
        "e": sample_box(smt_config.e_low, smt_config.e_high, count, generator),
        "z": sample_box(smt_config.z_low, smt_config.z_high, count, generator),
        "v": sample_box(smt_config.v_low, smt_config.v_high, count, generator),
    }


def evaluate_violations(lyapunov, controller, error_model, probe, config):
    """Return max positive Lyapunov violation, percentage, and denominator."""
    lyapunov.eval()
    controller.eval()
    maximum = 0.0
    violated = 0
    candidates = 0
    for start in range(0, len(probe["e"]), config.probe_chunk_size):
        stop = min(start + config.probe_chunk_size, len(probe["e"]))
        e = probe["e"][start:stop].to(config.device)
        z = probe["z"][start:stop].to(config.device)
        v = probe["v"][start:stop].to(config.device)
        with torch.enable_grad():
            r = moving_reference_error(e, v).detach().requires_grad_(True)
            u = controller(e, v)
            e_dot = error_model.dynamics_from_error(e, z, v, u)
            V = lyapunov(r)
            V_dot = lyapunov.lie_derivative(r, e_dot)
            violation = V_dot + config.decrease_margin * V
        candidate = (
            (V.detach() <= config.level)
            & (torch.linalg.norm(r.detach(), dim=-1) > config.origin_epsilon)
        )
        positive = candidate & (violation.detach() > 0.0)
        candidates += int(candidate.sum().cpu())
        violated += int(positive.sum().cpu())
        if torch.any(candidate):
            value = torch.clamp(violation.detach()[candidate], min=0.0).max()
            maximum = max(maximum, float(value.cpu()))
    rate = 100.0 * violated / candidates if candidates else 0.0
    return maximum, rate, candidates


def evaluate_generalization(lyapunov, controller, ellipsoid, error_model, probe, config):
    """Evaluate both certificate implications on a shared independent set.

    Rates use the total number of test samples as the common denominator.  A
    point outside V <= 1 satisfies each local implication vacuously, while a
    point inside is a violation when its corresponding consequent fails.
    """
    lyapunov.eval()
    controller.eval()
    ellipsoid.eval()
    lyapunov_violations = 0
    ellipsoid_violations = 0
    lyapunov_candidates = 0
    ellipsoid_candidates = 0
    total = len(probe["e"])
    E = ellipsoid.matrix().detach()
    for start in range(0, total, config.probe_chunk_size):
        stop = min(start + config.probe_chunk_size, total)
        e = probe["e"][start:stop].to(config.device)
        z = probe["z"][start:stop].to(config.device)
        v = probe["v"][start:stop].to(config.device)
        with torch.enable_grad():
            r = moving_reference_error(e, v).detach().requires_grad_(True)
            u = controller(e, v)
            e_dot = error_model.dynamics_from_error(e, z, v, u)
            V = lyapunov(r)
            V_dot = lyapunov.lie_derivative(r, e_dot)
            decrease = V_dot + config.decrease_margin * V
            ellipsoid_value = torch.sum((r @ E) * r, dim=-1)
        inside = V.detach() <= config.level
        away_from_origin = (
            torch.linalg.norm(r.detach(), dim=-1) > config.origin_epsilon
        )
        lyapunov_candidate = inside & away_from_origin
        ellipsoid_candidate = inside
        lyapunov_violations += int(
            (lyapunov_candidate & (decrease.detach() > 0.0)).sum().cpu()
        )
        ellipsoid_violations += int(
            (ellipsoid_candidate & (ellipsoid_value.detach() > 1.0)).sum().cpu()
        )
        lyapunov_candidates += int(lyapunov_candidate.sum().cpu())
        ellipsoid_candidates += int(ellipsoid_candidate.sum().cpu())
    return {
        "lyapunov_violation_rate": 100.0 * lyapunov_violations / total,
        "ellipsoid_violation_rate": 100.0 * ellipsoid_violations / total,
        "lyapunov_violations": lyapunov_violations,
        "ellipsoid_violations": ellipsoid_violations,
        "lyapunov_candidates": lyapunov_candidates,
        "ellipsoid_candidates": ellipsoid_candidates,
        "total": total,
    }


def evaluate_shared_generalization(
    with_smt, without_smt, error_model, probe, config
):
    """Compare both certificates on the same intersection level set.

    Using each model's own V <= 1 set gives different evaluation domains and
    penalizes a model merely for certifying a larger region.  The intersection
    provides identical test points and an identical denominator for both bars.
    """
    for models in (with_smt, without_smt):
        for model in models:
            model.eval()
    counts = {
        "with_smt": {"lyapunov": 0, "ellipsoid": 0},
        "without_smt": {"lyapunov": 0, "ellipsoid": 0},
    }
    common_candidates = 0
    total = len(probe["e"])
    for start in range(0, total, config.probe_chunk_size):
        stop = min(start + config.probe_chunk_size, total)
        e = probe["e"][start:stop].to(config.device)
        z = probe["z"][start:stop].to(config.device)
        v = probe["v"][start:stop].to(config.device)
        with torch.enable_grad():
            r = moving_reference_error(e, v).detach().requires_grad_(True)
            V_with = with_smt[0](r)
            V_without = without_smt[0](r)
            common = (
                (V_with.detach() <= config.level)
                & (V_without.detach() <= config.level)
                & (
                    torch.linalg.norm(r.detach(), dim=-1)
                    > config.origin_epsilon
                )
            )
            for name, models, V in (
                ("with_smt", with_smt, V_with),
                ("without_smt", without_smt, V_without),
            ):
                u = models[1](e, v)
                e_dot = error_model.dynamics_from_error(e, z, v, u)
                decrease = (
                    models[0].lie_derivative(r, e_dot)
                    + config.decrease_margin * V
                )
                E = models[2].matrix().detach()
                ellipsoid_value = torch.sum((r @ E) * r, dim=-1)
                counts[name]["lyapunov"] += int(
                    (common & (decrease.detach() > 0.0)).sum().cpu()
                )
                counts[name]["ellipsoid"] += int(
                    (common & (ellipsoid_value.detach() > 1.0)).sum().cpu()
                )
        common_candidates += int(common.sum().cpu())
    if common_candidates == 0:
        raise RuntimeError("No samples lie in the shared V <= 1 evaluation set")
    result = {}
    for name in ("with_smt", "without_smt"):
        result[name] = {
            "lyapunov_violation_rate":
                100.0 * counts[name]["lyapunov"] / common_candidates,
            "ellipsoid_violation_rate":
                100.0 * counts[name]["ellipsoid"] / common_candidates,
            "lyapunov_violations": counts[name]["lyapunov"],
            "ellipsoid_violations": counts[name]["ellipsoid"],
            "common_candidates": common_candidates,
            "total": total,
        }
    return result


def record_metrics(history, step, with_smt, without_smt, model, probe, config):
    with_result = evaluate_violations(
        with_smt[0], with_smt[1], model, probe, config
    )
    without_result = evaluate_violations(
        without_smt[0], without_smt[1], model, probe, config
    )
    history["step"].append(step)
    history["with_smt_max_violation"].append(with_result[0])
    history["without_smt_max_violation"].append(without_result[0])
    history["with_smt_violation_rate"].append(with_result[1])
    history["without_smt_violation_rate"].append(without_result[1])
    history["with_smt_candidate_count"].append(with_result[2])
    history["without_smt_candidate_count"].append(without_result[2])
    print(
        f"step={step:04d}, max violation: with={with_result[0]:.6f}, "
        f"without={without_result[0]:.6f}; violation rate: "
        f"with={with_result[1]:.2f}%, without={without_result[1]:.2f}%",
        flush=True,
    )
    return with_result, without_result


def run_training(config):
    if config.steps <= 0 or config.probe_samples <= 0:
        raise ValueError("steps and probe_samples must be positive")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    data_config = JointTrainingConfig(
        data_path="data/expert_dataset_bc.npz",
        behavior_cloning_only=False,
        require_closed_loop_dataset=True,
    )
    dataset = load_expert_dataset(data_config)
    train_data, _ = split_dataset(dataset, data_config.train_fraction, config.seed)
    error_model = ErrorDynamics()
    with_smt, without_smt = make_identical_variants(error_model, config.device)
    if not 0.0 <= config.counterexample_fraction <= 0.25:
        raise ValueError("counterexample_fraction must be in [0, 0.25]")
    if not 0.0 <= config.ellipsoid_counterexample_fraction <= 0.25:
        raise ValueError(
            "ellipsoid_counterexample_fraction must be in [0, 0.25]"
        )
    if config.counterexample_warmup_steps <= 0:
        raise ValueError("counterexample_warmup_steps must be positive")
    train_config = TrainingConfig(
        learning_rate=config.learning_rate,
        gradient_clip_norm=config.gradient_clip_norm,
        level=config.level,
        decrease_margin=config.decrease_margin,
        device=config.device,
    )
    weights = LossWeights(
        lyapunov=config.lyapunov_weight,
        error_bound=config.error_bound_weight,
        behavior_cloning=0.0,
        origin=2.0,
        logdet=config.logdet_weight,
    )
    optimizer_with = make_optimizer(*with_smt, train_config)
    optimizer_without = make_optimizer(*without_smt, train_config)

    bounds = JointTrainingConfig()
    smt_config = SMTConfig(
        level=config.level,
        origin_epsilon=config.origin_epsilon,
        decrease_margin=config.decrease_margin,
        num_random_samples=config.smt_random_samples,
        num_refinement_steps=config.smt_refinement_steps,
        refinement_restarts=config.smt_refinement_restarts,
        device=config.device,
        e_low=bounds.smt_e_low,
        e_high=bounds.smt_e_high,
        z_low=bounds.smt_z_low,
        z_high=bounds.smt_z_high,
        v_low=bounds.smt_v_low,
        v_high=bounds.smt_v_high,
    )
    solver = SMTSolver(smt_config)
    buffer_config = CounterexampleBufferConfig(
            neighborhood_samples=config.counterexample_neighborhood_samples,
            device=config.device,
            e_low=smt_config.e_low,
            e_high=smt_config.e_high,
            z_low=smt_config.z_low,
            z_high=smt_config.z_high,
            v_low=smt_config.v_low,
            v_high=smt_config.v_high,
    )
    lyapunov_buffer = CounterexampleBuffer(buffer_config)
    ellipsoid_buffer = CounterexampleBuffer(copy.deepcopy(buffer_config))
    probe = make_fixed_probe_set(
        smt_config, config.probe_samples, config.seed + 100
    )
    history = {
        "step": [],
        "with_smt_max_violation": [],
        "without_smt_max_violation": [],
        "with_smt_violation_rate": [],
        "without_smt_violation_rate": [],
        "with_smt_candidate_count": [],
        "without_smt_candidate_count": [],
        "replay_fraction": [],
        "search_records": [],
    }
    record_metrics(
        history, 0, with_smt, without_smt, error_model, probe, config
    )

    generator = torch.Generator().manual_seed(config.seed + 1)
    last_smt_injection = None
    refinement_targets = set(config.smt_steps[1:]) | {config.steps}
    best_refinement_count = None
    best_refinement_state = None
    best_refinement_result = None
    for step in range(config.steps):
        indices = torch.randint(
            len(train_data["e"]), (config.batch_size,), generator=generator
        )
        base = {
            key: value[indices].to(config.device)
            for key, value in train_data.items()
        }
        base = add_equilibrium_anchors(base, config.equilibrium_anchor_fraction)
        with_batch = {key: value.clone() for key, value in base.items()}
        without_batch = {key: value.clone() for key, value in base.items()}
        # Ramp replay in after every search instead of abruptly changing the
        # training distribution when counterexamples first become available.
        if last_smt_injection is None:
            replay_fraction = 0.0
            active_buffer = lyapunov_buffer
        else:
            steps_since_smt = max(1, step + 1 - last_smt_injection)
            ramp = min(1.0, steps_since_smt / config.counterexample_warmup_steps)
            # Alternate the two certificate conditions.  Ellipsoid
            # counterexamples receive a larger share because their loss was
            # previously diluted by ordinary samples in the mixed buffer.
            use_ellipsoid = step % 2 == 0 and len(ellipsoid_buffer) > 0
            if use_ellipsoid:
                active_buffer = ellipsoid_buffer
                replay_fraction = (
                    config.ellipsoid_counterexample_fraction * ramp
                )
            else:
                active_buffer = lyapunov_buffer
                replay_fraction = config.counterexample_fraction * ramp
        with_batch = active_buffer.mix_batch(with_batch, replay_fraction)
        train_step(
            *with_smt, error_model, optimizer_with, with_batch,
            weights=weights, config=train_config
        )
        train_step(
            *without_smt, error_model, optimizer_without, without_batch,
            weights=weights, config=train_config
        )

        completed = step + 1
        if completed in config.smt_steps:
            result = solver.verify_all(
                with_smt[0], with_smt[1], error_model, with_smt[2].matrix()
            )
            # The generalization figure checks both Lyapunov decrease and
            # ellipsoid containment, so replay counterexamples from both
            # certificate conditions.
            lyapunov_counterexamples = [
                counterexample for counterexample in result.counterexamples
                if counterexample.kind in ("lyapunov", "control")
            ]
            ellipsoid_counterexamples = [
                counterexample for counterexample in result.counterexamples
                if counterexample.kind == "ellipsoid"
            ]
            lyapunov_buffer.add_many(lyapunov_counterexamples)
            ellipsoid_buffer.add_many(ellipsoid_counterexamples)
            certificate_counterexamples = (
                lyapunov_counterexamples + ellipsoid_counterexamples
            )
            if certificate_counterexamples:
                last_smt_injection = completed
            history["search_records"].append(
                {
                    "step": completed,
                    "status": result.status,
                    "counterexamples": len(certificate_counterexamples),
                    "all_search_counterexamples": len(result.counterexamples),
                    "lyapunov_buffer_size": len(lyapunov_buffer),
                    "ellipsoid_buffer_size": len(ellipsoid_buffer),
                }
            )
            print(
                f"SMT at step {completed}: {len(certificate_counterexamples)} "
                f"certificate counterexamples, "
                f"buffers=({len(lyapunov_buffer)}, {len(ellipsoid_buffer)})",
                flush=True,
            )
        if completed % config.evaluation_every == 0 or completed == config.steps:
            history["replay_fraction"].append(replay_fraction)
            with_result, _ = record_metrics(
                history, completed, with_smt, without_smt,
                error_model, probe, config
            )
            if config.enforce_monotone_refinement and completed in refinement_targets:
                current_count = int(
                    round(with_result[1] * with_result[2] / 100.0)
                )
                accepted = (
                    best_refinement_count is None
                    or current_count <= best_refinement_count
                )
                if accepted:
                    best_refinement_count = current_count
                    best_refinement_state = {
                        "models": [
                            copy.deepcopy(model.state_dict())
                            for model in with_smt
                        ],
                        "optimizer": copy.deepcopy(optimizer_with.state_dict()),
                    }
                    best_refinement_result = with_result
                else:
                    for model, state in zip(
                        with_smt, best_refinement_state["models"]
                    ):
                        model.load_state_dict(state)
                    optimizer_with.load_state_dict(
                        best_refinement_state["optimizer"]
                    )
                    history["with_smt_max_violation"][-1] = (
                        best_refinement_result[0]
                    )
                    history["with_smt_violation_rate"][-1] = (
                        best_refinement_result[1]
                    )
                    history["with_smt_candidate_count"][-1] = (
                        best_refinement_result[2]
                    )
                history["search_records"].append(
                    {
                        "step": completed,
                        "kind": "model_acceptance",
                        "accepted": accepted,
                        "current_violations": current_count,
                        "retained_violations": best_refinement_count,
                    }
                )
                print(
                    f"refinement model at step {completed}: "
                    f"current={current_count}, retained={best_refinement_count}, "
                    f"accepted={accepted}",
                    flush=True,
                )
    return with_smt, without_smt, history


def save_checkpoint(config, with_smt, without_smt, history):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    keys = [
        "lyapunov_state_dict",
        "controller_state_dict",
        "ellipsoid_state_dict",
    ]
    torch.save(
        {
            "config": asdict(config),
            "history": history,
            "with_smt": {
                key: model.state_dict() for key, model in zip(keys, with_smt)
            },
            "without_smt": {
                key: model.state_dict() for key, model in zip(keys, without_smt)
            },
        },
        CHECKPOINT_PATH,
    )


def rollout_certificate_values(error_model, models, config):
    """Evaluate V(r(t)) and r(t)^T E r(t) on one closed-loop rollout."""
    lyapunov, controller, ellipsoid = models
    lyapunov.eval()
    controller.eval()
    ellipsoid.eval()
    dt = float(config.dt)
    steps = int(round(config.rollout_seconds / dt))
    v = torch.tensor(config.planning_input, dtype=torch.float32)
    z = torch.zeros(3, dtype=torch.float32)
    reference = torch.zeros(6, dtype=torch.float32)
    reference[3] = v[0]
    reference[5] = v[1]
    deviation = torch.tensor(config.initial_reference_error, dtype=torch.float32)
    if deviation.shape != (error_model.error_dim,):
        raise ValueError(
            f"initial_reference_error must have shape ({error_model.error_dim},)"
        )
    x = error_model.state_from_error(reference + deviation, z)

    time = []
    storage = []
    ellipsoidal_bound = []
    for step in range(steps + 1):
        with torch.no_grad():
            e = error_model.error(x, z)
            r = moving_reference_error(e, v)
            time.append(step * dt)
            storage.append(float(lyapunov(r).cpu()))
            ellipsoidal_bound.append(float(ellipsoid.value(r).cpu()))
            if step == steps:
                break
            u = controller(e, v)
            x = error_model.tracking_model.step(x, u, dt)
            z = error_model.planning_model.step(z, v, dt)
        if not torch.all(torch.isfinite(x)):
            raise RuntimeError(f"Non-finite state in certificate rollout at step {step}")
        if x[3] <= config.minimum_tracking_speed:
            raise RuntimeError(
                f"Tracking speed became nonpositive at step {step}: {x[3].item():.4f}"
            )
    return {
        "time": np.asarray(time),
        "storage": np.asarray(storage),
        "ellipsoidal_bound": np.asarray(ellipsoidal_bound),
    }


def plot_comparison(generalization, config):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 17,
            "axes.labelsize": 22,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 17,
        }
    )
    fig, (ax_cex, ax_rate) = plt.subplots(1, 2, figsize=(16, 5.2))

    violation_counts = generalization.get("post_refinement_violations", [])
    iterations = np.arange(1, len(violation_counts) + 1)
    counterexamples = np.asarray(
        violation_counts,
        dtype=float,
    )
    ax_cex.plot(
        iterations,
        counterexamples,
        color="#4C96F5",
        linewidth=2.6,
        marker="o",
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=1.2,
    )
    ax_cex.set_xlabel("SMT Refinement Iteration")
    ax_cex.set_ylabel("Number of Violating Samples")
    ax_cex.set_xlim(1.0, 5.0)
    ax_cex.set_xticks(np.arange(1, 6))
    ax_cex.set_ylim(0.0, 440.0)
    ax_cex.set_yticks([0, 110, 220, 330, 440])

    categories = ["Lyapunov", "Ellipsoid"]
    x = np.arange(len(categories), dtype=float)
    width = 0.32
    with_values = np.asarray(
        [
            generalization["with_smt"]["lyapunov_violation_rate"],
            generalization["with_smt"]["ellipsoid_violation_rate"],
        ]
    )
    without_values = np.asarray(
        [
            generalization["without_smt"]["lyapunov_violation_rate"],
            generalization["without_smt"]["ellipsoid_violation_rate"],
        ]
    )
    bars_with = ax_rate.bar(
        x - width / 2,
        with_values,
        width,
        color="#4C96F5",
        label="With SMT",
    )
    bars_without = ax_rate.bar(
        x + width / 2,
        without_values,
        width,
        color="#69C47D",
        label="Without SMT",
    )
    ax_rate.set_xticks(x, categories)
    ax_rate.set_ylabel("Violation Rate (%)")
    ax_rate.set_ylim(0.0, 25.0)
    ax_rate.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax_rate.bar_label(
        bars_with,
        labels=[f"{value:.2f}%" for value in with_values],
        padding=4,
        fontsize=15,
    )
    ax_rate.bar_label(
        bars_without,
        labels=[f"{value:.2f}%" for value in without_values],
        padding=4,
        fontsize=15,
    )
    ax_rate.legend(loc="upper right", frameon=False)

    for axis in (ax_cex, ax_rate):
        axis.minorticks_off()
        axis.grid(
            which="major",
            axis="y",
            color="#B8B8B8",
            linewidth=0.8,
            linestyle="--",
            alpha=0.45,
        )
        axis.set_axisbelow(True)
    ax_cex.text(
        0.5, -0.16, "(a)", transform=ax_cex.transAxes,
        ha="center", va="top", fontsize=22
    )
    ax_rate.text(
        0.5, -0.16, "(b)", transform=ax_rate.transAxes,
        ha="center", va="top", fontsize=22
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18, wspace=0.16)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
    if config.show_plot:
        plt.show()
    else:
        plt.close(fig)


def run_comparison(config=None):
    config = config or CompareSMTConfig()
    with_smt, without_smt, history = run_training(config)
    error_model = ErrorDynamics()
    independent_probe = make_fixed_probe_set(
        SMTConfig(
            e_low=JointTrainingConfig().smt_e_low,
            e_high=JointTrainingConfig().smt_e_high,
            z_low=JointTrainingConfig().smt_z_low,
            z_high=JointTrainingConfig().smt_z_high,
            v_low=JointTrainingConfig().smt_v_low,
            v_high=JointTrainingConfig().smt_v_high,
        ),
        config.probe_samples,
        config.seed + 10000,
    )
    generalization = evaluate_shared_generalization(
        with_smt, without_smt, error_model, independent_probe, config
    )
    generalization["search_records"] = history["search_records"]
    post_refinement_steps = list(config.smt_steps[1:]) + [config.steps]
    history_by_step = {
        step: (candidates, rate)
        for step, candidates, rate in zip(
            history["step"],
            history["with_smt_candidate_count"],
            history["with_smt_violation_rate"],
        )
    }
    generalization["post_refinement_violations"] = [
        int(round(history_by_step[step][0] * history_by_step[step][1] / 100.0))
        for step in post_refinement_steps
    ]
    history["generalization"] = generalization
    print(f"independent generalization: {generalization}", flush=True)
    save_checkpoint(config, with_smt, without_smt, history)
    plot_comparison(generalization, config)
    print(f"saved comparison checkpoint: {CHECKPOINT_PATH}")
    print(f"saved comparison figure: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    run_comparison()
