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


def test_ellipsoid_counterexample_search():
    lyapunov = NeuralLyapunovFunction(
        error_dim=10,
        hidden_sizes=(16,),
        feature_dim=4,
        delta=1e-3,
    )
    E = torch.eye(10)

    solver = SMTSolver(
        SMTConfig(
            num_random_samples=512,
            num_refinement_steps=30,
            refinement_restarts=2,
            e_low=(-2.0,) * 10,
            e_high=(2.0,) * 10,
        )
    )
    counterexamples = solver.find_ellipsoid_counterexamples(lyapunov, E)

    if not counterexamples:
        raise AssertionError("Expected an ellipsoid counterexample for small quadratic V and E=I")
    if len(counterexamples) < 2:
        raise AssertionError("Expected diverse search to return multiple ellipsoid counterexamples")
    if len(counterexamples) > solver.config.max_counterexamples_per_kind:
        raise AssertionError(f"Too many counterexamples returned: {len(counterexamples)}")
    for counterexample in counterexamples:
        if counterexample.kind != "ellipsoid":
            raise AssertionError(f"Unexpected counterexample kind: {counterexample.kind}")
        if counterexample.e.shape != (10,):
            raise AssertionError(f"Unexpected e shape: {counterexample.e.shape}")
        if counterexample.values["V"] > solver.config.level + 1e-5:
            raise AssertionError(f"Counterexample is outside V level set: {counterexample.values}")
        if counterexample.values["ellipsoid_value"] <= 1.0:
            raise AssertionError(f"Counterexample does not violate ellipsoid bound: {counterexample.values}")

    e = torch.stack([counterexample.e for counterexample in counterexamples])
    normalized_e = (e + 2.0) / 4.0
    distances = torch.cdist(normalized_e, normalized_e)
    distances.fill_diagonal_(float("inf"))
    if distances.min().item() < solver.config.min_counterexample_distance - 1e-6:
        raise AssertionError(f"Counterexamples are not sufficiently diverse: {distances.min().item()}")

    print("ellipsoid counterexample search: ok")


def test_verify_all_and_control_bounds():
    error_model = ErrorDynamics()
    tracking_model = error_model.tracking_model
    lyapunov = NeuralLyapunovFunction(
        error_dim=10,
        hidden_sizes=(16,),
        feature_dim=4,
        delta=1e-3,
    )
    controller = NeuralTrackingController(
        hidden_sizes=(16,),
        g=tracking_model.g,
        kT=tracking_model.kT,
    )
    ellipsoid = LearnableEllipsoid(dim=10, init_scale=1.0)

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
        e=torch.arange(10, dtype=torch.float32),
        z=torch.tensor([0.1, 0.2, 0.3]),
        v=torch.tensor([0.0, 0.1, -0.1]),
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
        "e": (4, 10),
        "z": (4, 3),
        "v": (4, 3),
        "is_counterexample": (4,),
        "use_lyapunov_loss": (4,),
        "use_error_bound_loss": (4,),
        "use_behavior_cloning_loss": (4,),
    }
    for key, shape in expected_shapes.items():
        if key not in batch or tuple(batch[key].shape) != shape:
            raise AssertionError(f"Unexpected batch[{key}] shape: {batch.get(key)}")

    base_batch = {
        "e": torch.zeros(10, 10),
        "z": torch.zeros(10, 3),
        "v": torch.zeros(10, 3),
    }
    mixed = buffer.mix_batch(base_batch, counterexample_fraction=0.3)
    if mixed["is_counterexample"].sum().item() != 3:
        raise AssertionError("Expected three counterexample rows in mixed batch")
    if mixed["use_behavior_cloning_loss"].sum().item() != 7:
        raise AssertionError("Expected behavior cloning to exclude counterexample rows")

    base_batch["use_behavior_cloning_loss"] = torch.zeros(10, dtype=torch.bool)
    mixed = buffer.mix_batch(base_batch, counterexample_fraction=0.3)
    if mixed["use_behavior_cloning_loss"].any():
        raise AssertionError("Expected counterexample mixing to preserve existing loss masks")

    print("counterexample buffer sampling and mixing: ok")


def main():
    test_ellipsoid_counterexample_search()
    test_verify_all_and_control_bounds()
    test_counterexample_buffer_sampling_and_mixing()
    print("all SMT solver tests passed")


if __name__ == "__main__":
    main()
