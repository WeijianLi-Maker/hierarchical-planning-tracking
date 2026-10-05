import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SMT.CounterexampleBuffer import CounterexampleBuffer, CounterexampleBufferConfig
from SMT.SMTSolver import Counterexample, SMTConfig, SMTSolver
from controllers.TrackingControl import NeuralTrackingController
from learning.Training import LearnableEllipsoid
from lyapunov.LyapunovNetwork import NeuralLyapunovFunction
from models.error import ErrorDynamics


def test_bicycle_default_search_domain():
    config = SMTConfig()
    solver = SMTSolver(config)

    if len(config.e_low) != 6 or len(config.e_high) != 6:
        raise AssertionError("Expected six-dimensional error search bounds")
    if len(config.z_low) != 3 or len(config.z_high) != 3:
        raise AssertionError("Expected three-dimensional planning-state bounds")
    if len(config.v_low) != 2 or len(config.v_high) != 2:
        raise AssertionError("Expected two-dimensional planning-input bounds")
    if config.e_low[3] <= 0.0:
        raise AssertionError("Expected the SMT domain to avoid the v_x=0 singularity")

    e = solver._sample_box(config.e_low, config.e_high, 128)
    z = solver._sample_box(config.z_low, config.z_high, 128)
    v = solver._sample_box(config.v_low, config.v_high, 128)
    if e.shape != (128, 6) or z.shape != (128, 3) or v.shape != (128, 2):
        raise AssertionError(f"Unexpected sampled shapes: {e.shape}, {z.shape}, {v.shape}")
    if torch.any(e[:, 3] <= 0.0):
        raise AssertionError("Sampled SMT errors include nonpositive longitudinal speed")

    print("Bicycle default SMT search domain: ok")


def test_rejects_legacy_dimensions_and_singular_domain():
    invalid_configs = (
        SMTConfig(e_low=(-1.0,) * 10, e_high=(1.0,) * 10),
        SMTConfig(v_low=(-1.0, -1.0, -1.0), v_high=(1.0, 1.0, 1.0)),
        SMTConfig(
            e_low=(-0.5, -0.3, -0.05, 0.0, -0.1, -0.05),
            e_high=(0.5, 0.3, 0.05, 10.5, 0.1, 0.05),
        ),
    )
    for config in invalid_configs:
        try:
            SMTSolver(config)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid Bicycle SMT config to be rejected: {config}")

    invalid_counterexamples = (
        {"e": torch.zeros(10)},
        {"e": torch.zeros(6), "z": torch.zeros(4)},
        {"e": torch.zeros(6), "v": torch.zeros(3)},
    )
    for values in invalid_counterexamples:
        try:
            Counterexample(kind="lyapunov", **values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid counterexample to be rejected: {values}")

    print("legacy dimensions and singular SMT domain rejected: ok")


def test_ellipsoid_counterexample_search():
    torch.manual_seed(1)
    lyapunov = NeuralLyapunovFunction(
        error_dim=6,
        hidden_sizes=(16,),
        feature_dim=4,
        delta=1e-3,
    )
    E = torch.eye(6)
    e_low = (-2.0, -2.0, -0.2, 5.0, -0.5, -0.2)
    e_high = (2.0, 2.0, 0.2, 10.0, 0.5, 0.2)

    solver = SMTSolver(
        SMTConfig(
            num_random_samples=512,
            num_refinement_steps=30,
            refinement_restarts=2,
            e_low=e_low,
            e_high=e_high,
        )
    )
    counterexamples = solver.find_ellipsoid_counterexamples(lyapunov, E)

    if not counterexamples:
        raise AssertionError("Expected an ellipsoid counterexample for small quadratic V and E=I")
    if len(counterexamples) > solver.config.max_counterexamples_per_kind:
        raise AssertionError(f"Too many counterexamples returned: {len(counterexamples)}")
    for counterexample in counterexamples:
        if counterexample.kind != "ellipsoid":
            raise AssertionError(f"Unexpected counterexample kind: {counterexample.kind}")
        if counterexample.e.shape != (6,):
            raise AssertionError(f"Unexpected e shape: {counterexample.e.shape}")
        if counterexample.v is None:
            raise AssertionError("Ellipsoid counterexample must include its moving reference")
        if counterexample.values["V"] > solver.config.level + 1e-5:
            raise AssertionError(f"Counterexample is outside V level set: {counterexample.values}")
        if counterexample.values["ellipsoid_value"] <= 1.0:
            raise AssertionError(f"Counterexample does not violate ellipsoid bound: {counterexample.values}")

    if len(counterexamples) > 1:
        e = torch.stack([counterexample.e for counterexample in counterexamples])
        low = torch.tensor(e_low)
        high = torch.tensor(e_high)
        normalized_e = (e - low) / (high - low)
        distances = torch.cdist(normalized_e, normalized_e)
        distances.fill_diagonal_(float("inf"))
        if distances.min().item() < solver.config.min_counterexample_distance - 1e-6:
            raise AssertionError(
                f"Counterexamples are not sufficiently diverse: {distances.min().item()}"
            )

    print("ellipsoid counterexample search: ok")


def test_verify_all_and_control_bounds():
    error_model = ErrorDynamics()
    lyapunov = NeuralLyapunovFunction(
        error_dim=6,
        hidden_sizes=(16,),
        feature_dim=4,
        delta=1e-3,
    )
    controller = NeuralTrackingController(
        hidden_sizes=(16,),
    )
    ellipsoid = LearnableEllipsoid(dim=6, init_scale=1.0)

    solver = SMTSolver(
        SMTConfig(
            num_random_samples=128,
            num_refinement_steps=5,
            refinement_restarts=1,
        )
    )
    result = solver.verify_all(lyapunov, controller, error_model, ellipsoid.matrix())

    if "control" not in result.status:
        raise AssertionError(f"Missing control status: {result.status}")
    if result.status["control"] != "no_counterexample_found":
        raise AssertionError("Controller output bounds should hold by construction")

    print("verify_all and control bounds: ok")


def test_counterexample_buffer_sampling_and_mixing():
    counterexample = Counterexample(
        kind="lyapunov",
        e=torch.tensor([0.1, -0.1, 0.02, 8.0, 0.05, 0.01]),
        z=torch.tensor([0.1, 0.2, 0.3]),
        v=torch.tensor([8.0, 0.1]),
        violation=0.5,
    )
    buffer = CounterexampleBuffer(
        CounterexampleBufferConfig(
            neighborhood_samples=8,
            e_radius=1e-3,
            z_radius=1e-3,
            v_radius=1e-3,
        )
    )
    buffer.add(counterexample)

    if len(buffer) != 8:
        raise AssertionError(f"Expected 8 expanded samples, got {len(buffer)}")

    batch = buffer.sample(4)
    expected_shapes = {
        "e": (4, 6),
        "z": (4, 3),
        "v": (4, 2),
        "is_counterexample": (4,),
        "use_lyapunov_loss": (4,),
        "use_error_bound_loss": (4,),
        "use_behavior_cloning_loss": (4,),
    }
    for key, shape in expected_shapes.items():
        if key not in batch or tuple(batch[key].shape) != shape:
            raise AssertionError(f"Unexpected batch[{key}] shape: {batch.get(key)}")
    if torch.any(batch["e"][:, 3] <= 0.0):
        raise AssertionError("Counterexample neighborhoods crossed the v_x=0 singularity")

    base_batch = {
        "e": torch.zeros(10, 6),
        "z": torch.zeros(10, 3),
        "v": torch.zeros(10, 2),
    }
    base_batch["e"][:, 3] = 8.0
    base_batch["v"][:, 0] = 8.0
    mixed = buffer.mix_batch(base_batch, counterexample_fraction=0.3)
    if mixed["is_counterexample"].sum().item() != 3:
        raise AssertionError("Expected three counterexample rows in mixed batch")
    if mixed["use_behavior_cloning_loss"].sum().item() != 7:
        raise AssertionError("Expected behavior cloning to exclude counterexample rows")

    print("counterexample buffer sampling and mixing: ok")


def test_counterexample_buffer_clamps_to_bicycle_domain():
    config = CounterexampleBufferConfig(
        neighborhood_samples=64,
        e_radius=10.0,
        z_radius=100.0,
        v_radius=100.0,
    )
    buffer = CounterexampleBuffer(config)
    counterexample = Counterexample(
        kind="lyapunov",
        e=torch.tensor([0.5, 0.3, 0.05, 5.0, 0.1, 0.05]),
        z=torch.tensor([20.0, 20.0, torch.pi]),
        v=torch.tensor([10.0, 1.0]),
    )
    buffer.add(counterexample)
    batch = buffer.sample(64)

    for key, low, high in (
        ("e", config.e_low, config.e_high),
        ("z", config.z_low, config.z_high),
        ("v", config.v_low, config.v_high),
    ):
        low_t = torch.tensor(low)
        high_t = torch.tensor(high)
        if torch.any(batch[key] < low_t) or torch.any(batch[key] > high_t):
            raise AssertionError(f"Counterexample buffer failed to clamp {key} to its domain")

    print("counterexample buffer Bicycle-domain clamping: ok")


def main():
    test_bicycle_default_search_domain()
    test_rejects_legacy_dimensions_and_singular_domain()
    test_ellipsoid_counterexample_search()
    test_verify_all_and_control_bounds()
    test_counterexample_buffer_sampling_and_mixing()
    test_counterexample_buffer_clamps_to_bicycle_domain()
    print("all SMT solver tests passed")


if __name__ == "__main__":
    main()
