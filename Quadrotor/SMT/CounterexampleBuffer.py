from __future__ import annotations

from dataclasses import dataclass

import torch

from SMT.SMTSolver import Counterexample


@dataclass
class CounterexampleBufferConfig:
    capacity: int = 20000
    neighborhood_samples: int = 32
    e_radius: float = 1e-2
    z_radius: float = 1e-2
    v_radius: float = 1e-2
    device: str = "cpu"
    e_low: tuple[float, ...] | None = None
    e_high: tuple[float, ...] | None = None
    z_low: tuple[float, ...] | None = None
    z_high: tuple[float, ...] | None = None
    v_low: tuple[float, ...] | None = None
    v_high: tuple[float, ...] | None = None


class CounterexampleBuffer:
    """
    Stores counterexamples and small neighborhoods around them for retraining.
    """

    def __init__(self, config: CounterexampleBufferConfig | None = None):
        self.config = config or CounterexampleBufferConfig()
        self._samples: list[dict[str, torch.Tensor | str | float]] = []

    def __len__(self) -> int:
        return len(self._samples)

    def clear(self) -> None:
        self._samples.clear()

    def add(self, counterexample: Counterexample, expand: bool = True) -> None:
        count = self.config.neighborhood_samples if expand else 1
        samples = self.expand_neighborhood(counterexample, count)
        self._samples.extend(samples)
        overflow = len(self._samples) - self.config.capacity
        if overflow > 0:
            del self._samples[:overflow]

    def add_many(self, counterexamples: list[Counterexample], expand: bool = True) -> None:
        for counterexample in counterexamples:
            self.add(counterexample, expand=expand)

    def expand_neighborhood(self, counterexample: Counterexample, count: int | None = None) -> list[dict]:
        count = int(self.config.neighborhood_samples if count is None else count)
        if count <= 0:
            return []

        e = self._jitter(counterexample.e, self.config.e_radius, count, self.config.e_low, self.config.e_high)
        z = None
        v = None
        if counterexample.z is not None:
            z = self._jitter(counterexample.z, self.config.z_radius, count, self.config.z_low, self.config.z_high)
        if counterexample.v is not None:
            v = self._jitter(counterexample.v, self.config.v_radius, count, self.config.v_low, self.config.v_high)

        samples = []
        for index in range(count):
            sample = {
                "kind": counterexample.kind,
                "e": e[index].detach().cpu(),
                "violation": float(counterexample.violation),
            }
            if z is not None:
                sample["z"] = z[index].detach().cpu()
            if v is not None:
                sample["v"] = v[index].detach().cpu()
            samples.append(sample)
        return samples

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if not self._samples:
            raise ValueError("Cannot sample from an empty CounterexampleBuffer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        device = torch.device(self.config.device)
        indices = torch.randint(len(self._samples), (batch_size,))
        selected = [self._samples[int(index)] for index in indices]

        e = torch.stack([sample["e"] for sample in selected]).to(device)
        batch = {
            "e": e,
            "is_counterexample": torch.ones(batch_size, dtype=torch.bool, device=device),
            "violation": torch.tensor(
                [float(sample["violation"]) for sample in selected],
                dtype=torch.float32,
                device=device,
            ),
        }

        if all("z" in sample for sample in selected):
            batch["z"] = torch.stack([sample["z"] for sample in selected]).to(device)
        if all("v" in sample for sample in selected):
            batch["v"] = torch.stack([sample["v"] for sample in selected]).to(device)

        kinds = [sample["kind"] for sample in selected]
        batch["use_lyapunov_loss"] = torch.tensor(
            [kind in ("lyapunov", "control") for kind in kinds],
            dtype=torch.bool,
            device=device,
        )
        batch["use_error_bound_loss"] = torch.tensor(
            [kind == "ellipsoid" for kind in kinds],
            dtype=torch.bool,
            device=device,
        )
        batch["use_behavior_cloning_loss"] = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=device,
        )
        return batch

    def mix_batch(self, base_batch: dict[str, torch.Tensor], counterexample_fraction: float) -> dict[str, torch.Tensor]:
        """
        Return a batch with a fraction of rows replaced by counterexample samples.

        This helper assumes base_batch already contains e, z, and v. Ellipsoid
        counterexamples do not carry z/v, so their missing z/v rows are borrowed
        from the base batch being replaced.
        """
        if not self._samples or counterexample_fraction <= 0.0:
            return base_batch
        if "e" not in base_batch:
            raise KeyError("base_batch must contain 'e'")

        total = base_batch["e"].shape[0]
        cex_count = max(1, min(total, int(round(total * counterexample_fraction))))
        cex_batch = self.sample(cex_count)
        mixed = {key: value.clone() if torch.is_tensor(value) else value for key, value in base_batch.items()}
        target = torch.randperm(total, device=base_batch["e"].device)[:cex_count]

        mixed["e"][target] = cex_batch["e"].to(mixed["e"])
        for key in ("z", "v"):
            if key in mixed and key in cex_batch:
                mixed[key][target] = cex_batch[key].to(mixed[key])

        mixed["is_counterexample"] = torch.zeros(total, dtype=torch.bool, device=base_batch["e"].device)
        mixed["is_counterexample"][target] = True
        for key in (
            "use_lyapunov_loss",
            "use_error_bound_loss",
            "use_behavior_cloning_loss",
            "use_interior_loss",
            "use_utilization_loss",
        ):
            if key in base_batch:
                mixed[key] = base_batch[key].clone()
            else:
                mixed[key] = torch.ones(
                    total,
                    dtype=torch.bool,
                    device=base_batch["e"].device,
                )
        if "use_boundary_loss" in base_batch:
            mixed["use_boundary_loss"] = base_batch["use_boundary_loss"].clone()
        else:
            mixed["use_boundary_loss"] = torch.zeros(
                total,
                dtype=torch.bool,
                device=base_batch["e"].device,
            )
        mixed["use_lyapunov_loss"][target] = cex_batch["use_lyapunov_loss"].to(mixed["use_lyapunov_loss"])
        mixed["use_error_bound_loss"][target] = cex_batch["use_error_bound_loss"].to(mixed["use_error_bound_loss"])
        mixed["use_behavior_cloning_loss"][target] = False
        mixed["use_interior_loss"][target] = False
        mixed["use_utilization_loss"][target] = False
        mixed["use_boundary_loss"][target] = False
        return mixed

    @staticmethod
    def _jitter(
        center: torch.Tensor,
        radius: float,
        count: int,
        low: tuple[float, ...] | None,
        high: tuple[float, ...] | None,
    ) -> torch.Tensor:
        center = center.detach().cpu().to(dtype=torch.float32).reshape(1, -1)
        noise = torch.empty(count, center.shape[-1]).uniform_(-float(radius), float(radius))
        samples = center + noise
        samples[0] = center

        if low is not None:
            low_t = torch.tensor(low, dtype=torch.float32).reshape(1, -1)
            samples = torch.maximum(samples, low_t)
        if high is not None:
            high_t = torch.tensor(high, dtype=torch.float32).reshape(1, -1)
            samples = torch.minimum(samples, high_t)
        return samples
