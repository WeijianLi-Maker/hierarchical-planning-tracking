from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from learning.LossFunction import moving_reference_error, quadratic_form


CounterexampleKind = Literal["lyapunov", "ellipsoid", "control"]
ERROR_DIM = 6
PLANNING_STATE_DIM = 3
PLANNING_INPUT_DIM = 2


@dataclass
class Counterexample:
    """
    A concrete point that violates one of the desired certificate conditions.

    The tensors are stored as one-dimensional CPU tensors so they can be safely
    serialized, added to buffers, and rechecked with the PyTorch models.
    """

    kind: CounterexampleKind
    e: torch.Tensor
    z: torch.Tensor | None = None
    v: torch.Tensor | None = None
    violation: float = 0.0
    values: dict[str, float] = field(default_factory=dict)
    source: str = "torch_search"

    def __post_init__(self):
        self.e = _as_1d_cpu_float(self.e, "e")
        if self.e.shape != (ERROR_DIM,):
            raise ValueError(f"Expected Bicycle e shape ({ERROR_DIM},), got {tuple(self.e.shape)}")
        if self.z is not None:
            self.z = _as_1d_cpu_float(self.z, "z")
            if self.z.shape != (PLANNING_STATE_DIM,):
                raise ValueError(
                    f"Expected Bicycle z shape ({PLANNING_STATE_DIM},), got {tuple(self.z.shape)}"
                )
        if self.v is not None:
            self.v = _as_1d_cpu_float(self.v, "v")
            if self.v.shape != (PLANNING_INPUT_DIM,):
                raise ValueError(
                    f"Expected Bicycle v shape ({PLANNING_INPUT_DIM},), got {tuple(self.v.shape)}"
                )

    def as_batch(self) -> dict[str, torch.Tensor]:
        batch = {"e": self.e.unsqueeze(0)}
        if self.z is not None:
            batch["z"] = self.z.unsqueeze(0)
        if self.v is not None:
            batch["v"] = self.v.unsqueeze(0)
        return batch


@dataclass
class VerificationResult:
    """
    Summary returned by verify_all().
    """

    counterexamples: list[Counterexample]
    status: dict[str, str]

    @property
    def success(self) -> bool:
        return len(self.counterexamples) == 0


@dataclass
class SMTConfig:
    """
    Search ranges and numerical margins for counterexample generation.

    The first implementation exposes a stable PyTorch search path. It keeps the
    same public interface that a dReal/Z3 backend can use later, while avoiding
    a hard dependency on solvers that are often unavailable on Windows.
    """

    backend: Literal["torch", "dreal", "z3"] = "torch"
    level: float = 1.0
    origin_epsilon: float = 1e-4
    decrease_margin: float = 0.01
    bound_margin: float = 1e-4
    control_margin: float = 1e-6
    num_random_samples: int = 4096
    num_refinement_steps: int = 120
    refinement_restarts: int = 8
    refinement_lr: float = 5e-2
    max_counterexamples_per_kind: int = 10
    min_counterexample_distance: float = 0.1
    device: str = "cpu"
    e_low: tuple[float, ...] = (-0.5, -0.3, -0.05, 5.0, -0.1, -0.05)
    e_high: tuple[float, ...] = (0.5, 0.3, 0.05, 10.5, 0.1, 0.05)
    z_low: tuple[float, ...] = (-20.0, -20.0, -torch.pi)
    z_high: tuple[float, ...] = (20.0, 20.0, torch.pi)
    v_low: tuple[float, ...] = (4.0, -1.0)
    v_high: tuple[float, ...] = (10.0, 1.0)


class SMTSolver:
    """
    Counterexample generator for the learned certificate conditions.

    This class is intentionally conservative: every candidate is rechecked by
    the actual PyTorch models before it is returned. The current executable
    backend is a random plus gradient search over bounded domains. The class
    name and interfaces are kept solver-oriented so a symbolic dReal encoder can
    be added without changing the training loop.
    """

    def __init__(self, config: SMTConfig | None = None):
        self.config = config or SMTConfig()
        if self.config.backend != "torch":
            raise NotImplementedError(
                f"Backend '{self.config.backend}' is not implemented yet. "
                "Use backend='torch' for counterexample-guided training now."
            )
        self._validate_config()

    def _validate_config(self) -> None:
        expected_dims = (
            ("e", self.config.e_low, self.config.e_high, ERROR_DIM),
            ("z", self.config.z_low, self.config.z_high, PLANNING_STATE_DIM),
            ("v", self.config.v_low, self.config.v_high, PLANNING_INPUT_DIM),
        )
        for name, low, high, expected in expected_dims:
            if len(low) != expected or len(high) != expected:
                raise ValueError(
                    f"Expected Bicycle {name} bounds dimension {expected}, "
                    f"got low={len(low)}, high={len(high)}"
                )
            self._bounds(low, high)
        if self.config.e_low[3] <= 0.0:
            raise ValueError("Expected e_low[3] > 0 to avoid the bicycle v_x=0 singularity")
        if self.config.num_random_samples <= 0:
            raise ValueError("num_random_samples must be positive")
        if self.config.refinement_restarts <= 0:
            raise ValueError("refinement_restarts must be positive")

    def find_ellipsoid_counterexamples(self, lyapunov, E: torch.Tensor) -> list[Counterexample]:
        """
        Search for diverse points satisfying V(e) <= level and e^T E e > 1.
        """
        lyapunov.eval()
        E = E.detach().to(self.device)

        def score(e, _z, v):
            r = moving_reference_error(e, v)
            V = lyapunov(r)
            ellipsoid_value = quadratic_form(r, E)
            feasible = self.config.level - V
            violation = ellipsoid_value - 1.0 - self.config.bound_margin
            return torch.minimum(feasible, violation), {
                "V": V,
                "ellipsoid_value": ellipsoid_value,
                "violation": violation,
            }

        results = []
        for e, _z, v, values in self._search_e_z_v_many(score):
            violation = float(values["violation"].detach().cpu())
            if violation > 0.0:
                results.append(
                    Counterexample(
                        kind="ellipsoid",
                        e=e,
                        v=v,
                        violation=violation,
                        values=_float_values(values),
                    )
                )
        return results

    def find_ellipsoid_counterexample(self, lyapunov, E: torch.Tensor) -> Counterexample | None:
        counterexamples = self.find_ellipsoid_counterexamples(lyapunov, E)
        return counterexamples[0] if counterexamples else None

    def find_lyapunov_counterexamples(self, lyapunov, controller, error_model) -> list[Counterexample]:
        """
        Search for points inside the candidate level set that violate
        exponential Lyapunov decrease.
        """
        lyapunov.eval()
        controller.eval()

        def score(e, z, v):
            u = controller(e, v)
            e_dot = error_model.dynamics_from_error(e, z, v, u)
            r = moving_reference_error(e, v)
            V = lyapunov(r)
            V_dot = lyapunov.lie_derivative(r, e_dot)
            norm_gap = torch.linalg.norm(r, dim=-1) - self.config.origin_epsilon
            level_gap = self.config.level - V
            violation = V_dot + self.config.decrease_margin * V
            return torch.minimum(torch.minimum(level_gap, norm_gap), violation), {
                "V": V,
                "V_dot": V_dot,
                "violation": violation,
                "reference_error_norm": torch.linalg.norm(r, dim=-1),
            }

        results = []
        for e, z, v, values in self._search_e_z_v_many(score):
            violation = float(values["violation"].detach().cpu())
            if violation > 0.0:
                results.append(
                    Counterexample(
                        kind="lyapunov",
                        e=e,
                        z=z,
                        v=v,
                        violation=violation,
                        values=_float_values(values),
                    )
                )
        return results

    def find_lyapunov_counterexample(self, lyapunov, controller, error_model) -> Counterexample | None:
        counterexamples = self.find_lyapunov_counterexamples(lyapunov, controller, error_model)
        return counterexamples[0] if counterexamples else None

    def find_control_counterexamples(self, controller) -> list[Counterexample]:
        """
        Search for diverse violations of the controller's declared input bounds.

        The current NeuralTrackingController is parameterized to satisfy this by
        construction, so this mainly protects future controller variants.
        """
        controller.eval()

        def score(e, z, v):
            u = controller(e, v)
            lower_violation = controller.u_min.to(u) - u
            upper_violation = u - controller.u_max.to(u)
            violation = torch.maximum(lower_violation.max(dim=-1).values, upper_violation.max(dim=-1).values)
            return violation - self.config.control_margin, {
                "violation": violation,
                "u_min_gap": lower_violation.max(dim=-1).values,
                "u_max_gap": upper_violation.max(dim=-1).values,
            }

        results = []
        for e, z, v, values in self._search_e_z_v_many(score):
            violation = float(values["violation"].detach().cpu())
            if violation > 0.0:
                results.append(
                    Counterexample(
                        kind="control",
                        e=e,
                        z=z,
                        v=v,
                        violation=violation,
                        values=_float_values(values),
                    )
                )
        return results

    def find_control_counterexample(self, controller) -> Counterexample | None:
        counterexamples = self.find_control_counterexamples(controller)
        return counterexamples[0] if counterexamples else None

    def verify_all(self, lyapunov, controller, error_model, E: torch.Tensor) -> VerificationResult:
        if lyapunov.error_dim != ERROR_DIM or error_model.error_dim != ERROR_DIM:
            raise ValueError("Expected six-dimensional Bicycle Lyapunov and error models")
        if controller.error_dim != ERROR_DIM:
            raise ValueError("Expected controller error_dim=6")
        if controller.planning_input_dim != PLANNING_INPUT_DIM:
            raise ValueError("Expected controller planning_input_dim=2")
        if E.shape != (ERROR_DIM, ERROR_DIM):
            raise ValueError(f"Expected E shape ({ERROR_DIM}, {ERROR_DIM}), got {tuple(E.shape)}")

        counterexamples = []
        status = {}

        checks = (
            ("ellipsoid", lambda: self.find_ellipsoid_counterexamples(lyapunov, E)),
            ("lyapunov", lambda: self.find_lyapunov_counterexamples(lyapunov, controller, error_model)),
            ("control", lambda: self.find_control_counterexamples(controller)),
        )
        for name, fn in checks:
            found = fn()
            if not found:
                status[name] = "no_counterexample_found"
            else:
                status[name] = "counterexample_found"
                counterexamples.extend(found)

        return VerificationResult(counterexamples=counterexamples, status=status)

    @property
    def device(self) -> torch.device:
        return torch.device(self.config.device)

    def _bounds(self, low, high) -> tuple[torch.Tensor, torch.Tensor]:
        low_t = torch.tensor(low, dtype=torch.float32, device=self.device)
        high_t = torch.tensor(high, dtype=torch.float32, device=self.device)
        if low_t.shape != high_t.shape:
            raise ValueError(f"Bounds shape mismatch: {low_t.shape} vs {high_t.shape}")
        if torch.any(low_t >= high_t):
            raise ValueError("Expected each lower bound to be smaller than upper bound")
        return low_t, high_t

    def _sample_box(self, low, high, count: int) -> torch.Tensor:
        low_t, high_t = self._bounds(low, high)
        unit = torch.rand(count, low_t.numel(), device=self.device)
        return low_t + unit * (high_t - low_t)

    def _select_diverse_indices(self, scores, features, low, high) -> list[int]:
        valid_indices = torch.nonzero(
            torch.isfinite(scores) & (scores > 0.0),
            as_tuple=False,
        ).reshape(-1)
        if valid_indices.numel() == 0:
            return []

        low_t = torch.cat([self._bounds(lo, hi)[0] for lo, hi in zip(low, high)])
        high_t = torch.cat([self._bounds(lo, hi)[1] for lo, hi in zip(low, high)])
        normalized = (features - low_t) / (high_t - low_t)
        ordered = valid_indices[torch.argsort(scores[valid_indices], descending=True)]

        selected = []
        min_distance = float(self.config.min_counterexample_distance)
        for index_tensor in ordered:
            index = int(index_tensor.item())
            if selected:
                distances = torch.linalg.norm(normalized[index] - normalized[selected], dim=-1)
                if torch.any(distances < min_distance):
                    continue
            selected.append(index)
            if len(selected) >= self.config.max_counterexamples_per_kind:
                break
        return selected

    def _search_e_only_many(self, score_fn):
        e = self._sample_box(self.config.e_low, self.config.e_high, self.config.num_random_samples)
        with torch.enable_grad():
            scores, _ = score_fn(e)
        seed_indices = self._select_diverse_indices(
            scores.detach(),
            e.detach(),
            (self.config.e_low,),
            (self.config.e_high,),
        )

        refined = []
        for index in seed_indices:
            candidate = self._refine_e_only(score_fn, e[index].detach())
            refined_e, refined_score, refined_values = candidate
            if refined_score > 0.0:
                refined.append((refined_e, refined_score, refined_values))
        return self._select_diverse_refined_e(refined)

    def _search_e_z_v_many(self, score_fn):
        count = self.config.num_random_samples
        e = self._sample_box(self.config.e_low, self.config.e_high, count)
        z = self._sample_box(self.config.z_low, self.config.z_high, count)
        v = self._sample_box(self.config.v_low, self.config.v_high, count)
        with torch.enable_grad():
            scores, _ = score_fn(e, z, v)
        features = torch.cat([e, z, v], dim=-1)
        seed_indices = self._select_diverse_indices(
            scores.detach(),
            features.detach(),
            (self.config.e_low, self.config.z_low, self.config.v_low),
            (self.config.e_high, self.config.z_high, self.config.v_high),
        )

        refined = []
        for index in seed_indices:
            candidate = self._refine_e_z_v(
                score_fn,
                e[index].detach(),
                z[index].detach(),
                v[index].detach(),
            )
            refined_e, refined_z, refined_v, refined_score, refined_values = candidate
            if refined_score > 0.0:
                refined.append((refined_e, refined_z, refined_v, refined_score, refined_values))
        return self._select_diverse_refined_e_z_v(refined)

    def _select_diverse_refined_e(self, refined):
        if not refined:
            return []
        scores = torch.stack([item[1] for item in refined])
        features = torch.stack([item[0] for item in refined])
        indices = self._select_diverse_indices(
            scores,
            features,
            (self.config.e_low,),
            (self.config.e_high,),
        )
        return [(refined[index][0].detach().cpu(), refined[index][2]) for index in indices]

    def _select_diverse_refined_e_z_v(self, refined):
        if not refined:
            return []
        scores = torch.stack([item[3] for item in refined])
        features = torch.stack(
            [torch.cat([item[0], item[1], item[2]], dim=-1) for item in refined]
        )
        indices = self._select_diverse_indices(
            scores,
            features,
            (self.config.e_low, self.config.z_low, self.config.v_low),
            (self.config.e_high, self.config.z_high, self.config.v_high),
        )
        return [
            (
                refined[index][0].detach().cpu(),
                refined[index][1].detach().cpu(),
                refined[index][2].detach().cpu(),
                refined[index][4],
            )
            for index in indices
        ]

    def _search_e_only(self, score_fn):
        e = self._sample_box(self.config.e_low, self.config.e_high, self.config.num_random_samples)
        with torch.enable_grad():
            scores, values = score_fn(e)
        best_idx = torch.argmax(scores)
        best_score = scores[best_idx].detach()
        best_e = e[best_idx].detach()
        best_values = {key: value[best_idx].detach() for key, value in values.items()}

        refined_e, refined_score, refined_values = self._refine_e_only(score_fn, best_e)
        if refined_score > best_score:
            best_e, best_score, best_values = refined_e, refined_score, refined_values

        if best_score <= 0.0:
            return None, {}
        return best_e.detach().cpu(), best_values

    def _search_e_z_v(self, score_fn):
        count = self.config.num_random_samples
        e = self._sample_box(self.config.e_low, self.config.e_high, count)
        z = self._sample_box(self.config.z_low, self.config.z_high, count)
        v = self._sample_box(self.config.v_low, self.config.v_high, count)
        with torch.enable_grad():
            scores, values = score_fn(e, z, v)
        best_idx = torch.argmax(scores)
        best_score = scores[best_idx].detach()
        best_e = e[best_idx].detach()
        best_z = z[best_idx].detach()
        best_v = v[best_idx].detach()
        best_values = {key: value[best_idx].detach() for key, value in values.items()}

        refined = self._refine_e_z_v(score_fn, best_e, best_z, best_v)
        refined_e, refined_z, refined_v, refined_score, refined_values = refined
        if refined_score > best_score:
            best_e, best_z, best_v = refined_e, refined_z, refined_v
            best_score, best_values = refined_score, refined_values

        if best_score <= 0.0:
            return None, None, None, {}
        return best_e.detach().cpu(), best_z.detach().cpu(), best_v.detach().cpu(), best_values

    def _refine_e_only(self, score_fn, seed_e):
        e_low, e_high = self._bounds(self.config.e_low, self.config.e_high)
        return_values = None
        best_e = seed_e.to(self.device)
        best_score = torch.tensor(float("-inf"), device=self.device)

        for _ in range(self.config.refinement_restarts):
            e = (seed_e.to(self.device) + 0.05 * torch.randn_like(seed_e.to(self.device))).clamp(e_low, e_high)
            e = e.detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([e], lr=self.config.refinement_lr)

            for _ in range(self.config.num_refinement_steps):
                optimizer.zero_grad(set_to_none=True)
                score, _ = score_fn(e.unsqueeze(0))
                loss = -score.mean()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    e.clamp_(e_low, e_high)

            with torch.no_grad():
                score, values = score_fn(e.unsqueeze(0))
                score = score.squeeze(0)
                if score > best_score:
                    best_score = score.detach()
                    best_e = e.detach().clone()
                    return_values = {key: value.squeeze(0).detach() for key, value in values.items()}

        return best_e, best_score, return_values or {}

    def _refine_e_z_v(self, score_fn, seed_e, seed_z, seed_v):
        e_low, e_high = self._bounds(self.config.e_low, self.config.e_high)
        z_low, z_high = self._bounds(self.config.z_low, self.config.z_high)
        v_low, v_high = self._bounds(self.config.v_low, self.config.v_high)
        return_values = None
        best_e = seed_e.to(self.device)
        best_z = seed_z.to(self.device)
        best_v = seed_v.to(self.device)
        best_score = torch.tensor(float("-inf"), device=self.device)

        for _ in range(self.config.refinement_restarts):
            e = (seed_e.to(self.device) + 0.05 * torch.randn_like(seed_e.to(self.device))).clamp(e_low, e_high)
            z = (seed_z.to(self.device) + 0.05 * torch.randn_like(seed_z.to(self.device))).clamp(z_low, z_high)
            v = (seed_v.to(self.device) + 0.05 * torch.randn_like(seed_v.to(self.device))).clamp(v_low, v_high)
            e = e.detach().clone().requires_grad_(True)
            z = z.detach().clone().requires_grad_(True)
            v = v.detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([e, z, v], lr=self.config.refinement_lr)

            for _ in range(self.config.num_refinement_steps):
                optimizer.zero_grad(set_to_none=True)
                score, _ = score_fn(e.unsqueeze(0), z.unsqueeze(0), v.unsqueeze(0))
                loss = -score.mean()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    e.clamp_(e_low, e_high)
                    z.clamp_(z_low, z_high)
                    v.clamp_(v_low, v_high)

            with torch.enable_grad():
                score, values = score_fn(e.unsqueeze(0), z.unsqueeze(0), v.unsqueeze(0))
            score = score.squeeze(0)
            if score > best_score:
                best_score = score.detach()
                best_e = e.detach().clone()
                best_z = z.detach().clone()
                best_v = v.detach().clone()
                return_values = {key: value.squeeze(0).detach() for key, value in values.items()}

        return best_e, best_z, best_v, best_score, return_values or {}


def _as_1d_cpu_float(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.tensor(value, dtype=torch.float32)
    value = value.detach().cpu().to(dtype=torch.float32).reshape(-1)
    if value.numel() == 0:
        raise ValueError(f"Expected non-empty tensor for {name}")
    return value


def _float_values(values: dict[str, torch.Tensor]) -> dict[str, float]:
    result = {}
    for key, value in values.items():
        if torch.is_tensor(value):
            result[key] = float(value.detach().cpu().reshape(()))
        else:
            result[key] = float(value)
    return result
