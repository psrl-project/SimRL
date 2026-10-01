"""Versioned cost artifact envelope shared by rollout and training models."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dryrun.profiler.schema import Parallelism, utc_now

ARTIFACT_VERSION = "1.0"


@dataclass(frozen=True)
class FitMetrics:
    """Summarize prediction quality for a fitted artifact entry."""

    r_squared: float
    rmse: float
    mre_percent: float
    max_error: float
    n_samples: int
    n_outliers: int = 0
    design_rank: int | None = None

    @classmethod
    def from_predictions(
        cls,
        actual: np.ndarray,
        predicted: np.ndarray,
        *,
        n_outliers: int = 0,
        design_rank: int | None = None,
    ) -> FitMetrics:
        """Compute standard regression diagnostics."""
        if actual.size == 0:
            return cls(
                r_squared=0.0,
                rmse=0.0,
                mre_percent=0.0,
                max_error=0.0,
                n_samples=0,
                n_outliers=n_outliers,
                design_rank=design_rank,
            )
        residual = actual - predicted
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
        r_squared = 1.0 if ss_tot == 0 and ss_res == 0 else 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
        denominator = np.maximum(np.abs(actual), np.finfo(np.float64).eps)
        return cls(
            r_squared=float(r_squared),
            rmse=float(np.sqrt(np.mean(residual**2))),
            mre_percent=float(np.mean(np.abs(residual) / denominator) * 100),
            max_error=float(np.max(np.abs(residual))),
            n_samples=int(actual.size),
            n_outliers=n_outliers,
            design_rank=design_rank,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FitMetrics:
        return cls(
            r_squared=float(value["r_squared"]),
            rmse=float(value["rmse"]),
            mre_percent=float(value["mre_percent"]),
            max_error=float(value["max_error"]),
            n_samples=int(value["n_samples"]),
            n_outliers=int(value.get("n_outliers", 0)),
            design_rank=value.get("design_rank"),
        )


@dataclass(frozen=True)
class ArtifactEntry:
    """Store coefficients and diagnostics for one explicit discrete cell."""

    parallelism: Parallelism
    selector: dict[str, Any]
    coefficients: dict[str, float]
    train_metrics: FitMetrics
    validation_metrics: FitMetrics
    feature_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ArtifactEntry:
        return cls(
            parallelism=Parallelism.from_dict(value["parallelism"]),
            selector=dict(value.get("selector", {})),
            coefficients={str(key): float(item) for key, item in value["coefficients"].items()},
            train_metrics=FitMetrics.from_dict(value["train_metrics"]),
            validation_metrics=FitMetrics.from_dict(value["validation_metrics"]),
            feature_ranges={
                str(key): (float(bounds[0]), float(bounds[1]))
                for key, bounds in value.get("feature_ranges", {}).items()
            },
            diagnostics=dict(value.get("diagnostics", {})),
        )


@dataclass(frozen=True)
class CostArtifact:
    """Serialize fitted cost models without importing their implementation."""

    kind: str
    model_type: str
    model: dict[str, Any]
    hardware: dict[str, Any]
    profile_digest: str
    entries: tuple[ArtifactEntry, ...]
    artifact_version: str = ARTIFACT_VERSION
    created_at: str = field(default_factory=utc_now)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.artifact_version != ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported artifact_version {self.artifact_version!r}; expected {ARTIFACT_VERSION!r}."
            )
        if self.kind not in {"rollout", "training"}:
            raise ValueError(f"Unsupported artifact kind {self.kind!r}.")
        if not self.entries:
            raise ValueError("A cost artifact must contain at least one entry.")

    def to_dict(self) -> dict[str, Any]:
        """Convert the artifact into JSON-compatible containers."""
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        """Atomically save the artifact as formatted JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def entry_for(
        self,
        parallelism: Parallelism,
        *,
        selector: dict[str, Any] | None = None,
        ignore_pp: bool = False,
    ) -> ArtifactEntry:
        """Return an exact discrete entry and reject silent extrapolation.

        Args:
            parallelism: The mesh to look up.
            selector: Discrete keys the entry must also carry.
            ignore_pp: Match on TP and CP only, leaving the pipeline degree to the
                cost model's own arithmetic. The training latency form divides its
                workload terms by `pp` and multiplies the slot count by
                `nmb + pp - 1`, both from the topology alone, so a PP cell is
                derived rather than looked up. Leave False for memory, which is not
                a simple per-stage division: the 1F1B peak depends on which stage
                is in flight.
        """
        requested = selector or {}
        matches = [
            entry
            for entry in self.entries
            if self._parallelism_matches(entry.parallelism, parallelism, ignore_pp=ignore_pp)
            and all(entry.selector.get(key) == value for key, value in requested.items())
        ]
        if len(matches) != 1:
            mesh = f"tp={parallelism.tp} cp={parallelism.cp}" if ignore_pp else repr(parallelism)
            raise KeyError(
                f"Expected exactly one artifact entry for parallelism {mesh} "
                f"and selector {requested!r}, found {len(matches)}."
            )
        return matches[0]

    @staticmethod
    def _parallelism_matches(
        entry: Parallelism,
        requested: Parallelism,
        *,
        ignore_pp: bool,
    ) -> bool:
        if not ignore_pp:
            return entry == requested
        return entry.tp == requested.tp and entry.cp == requested.cp

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CostArtifact:
        return cls(
            artifact_version=str(value["artifact_version"]),
            kind=str(value["kind"]),
            model_type=str(value["model_type"]),
            model=dict(value["model"]),
            hardware=dict(value["hardware"]),
            profile_digest=str(value["profile_digest"]),
            entries=tuple(ArtifactEntry.from_dict(item) for item in value["entries"]),
            created_at=str(value["created_at"]),
            diagnostics=dict(value.get("diagnostics", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> CostArtifact:
        """Load and validate an artifact envelope."""
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
        return cls.from_dict(value)
