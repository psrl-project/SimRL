"""Training workload, parallelism, and composable cost interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from dryrun.profiler.schema import TRAINING_SCHEDULES, Parallelism, SequenceStats


class TrainingOutOfDomainError(ValueError):
    """Indicate that a query is outside a fitted training domain."""


@dataclass(frozen=True)
class TrainingParallelism:
    """Describe training model and data parallel dimensions."""

    tp: int = 1
    pp: int = 1
    dp: int = 1
    cp: int = 1

    def __post_init__(self) -> None:
        if min(self.tp, self.pp, self.dp, self.cp) <= 0:
            raise ValueError("Training parallelism dimensions must be positive.")
        if self.cp != 1:
            raise TrainingOutOfDomainError("Hydraulis V1 only supports CP=1.")

    @property
    def world_size(self) -> int:
        return self.tp * self.pp * self.dp * self.cp

    def to_profile_parallelism(self, *, canonical_dp: bool = False) -> Parallelism:
        """Convert to the artifact's shared explicit parallelism type."""
        dp = 1 if canonical_dp else self.dp
        return Parallelism(
            tp=self.tp,
            pp=self.pp,
            dp=dp,
            cp=self.cp,
            world_size=self.tp * self.pp * dp * self.cp,
        )

    @classmethod
    def from_profile_parallelism(cls, value: Parallelism) -> TrainingParallelism:
        return cls(tp=value.tp, pp=value.pp, dp=value.dp, cp=value.cp)


@dataclass(frozen=True)
class TrainingWorkload:
    """Represent one global RL training batch using its actual trajectory lengths."""

    sequence_lengths: tuple[int, ...]
    micro_batch_size: int
    schedule: str = "1f1b"
    recompute: str = "none"
    optimizer: str = "adam"

    def __post_init__(self) -> None:
        if not self.sequence_lengths or any(length <= 0 for length in self.sequence_lengths):
            raise ValueError("Training sequence lengths must be non-empty and positive.")
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive.")
        if self.schedule.lower() not in TRAINING_SCHEDULES:
            raise TrainingOutOfDomainError(
                f"Training schedule must be one of {sorted(TRAINING_SCHEDULES)}, "
                f"got {self.schedule!r}."
            )

    @property
    def global_batch_size(self) -> int:
        return len(self.sequence_lengths)

    @property
    def total_tokens(self) -> int:
        return sum(self.sequence_lengths)

    def rank_microbatches(
        self,
        parallelism: TrainingParallelism,
    ) -> tuple[tuple[SequenceStats, ...], ...]:
        """Balance lengths across DP ranks and split each rank into microbatches."""
        divisor = parallelism.dp * self.micro_batch_size
        if self.global_batch_size % divisor != 0:
            raise TrainingOutOfDomainError(
                "global_batch_size must be divisible by dp * micro_batch_size, "
                f"got {self.global_batch_size} % {divisor}."
            )
        ordered = sorted(self.sequence_lengths, reverse=True)
        rank_lengths = [ordered[index:: parallelism.dp] for index in range(parallelism.dp)]
        ranks: list[tuple[SequenceStats, ...]] = []
        for lengths in rank_lengths:
            microbatches = tuple(
                SequenceStats.from_lengths(lengths[start : start + self.micro_batch_size])
                for start in range(0, len(lengths), self.micro_batch_size)
            )
            ranks.append(microbatches)
        return tuple(ranks)

    def critical_rank_microbatches(
        self,
        parallelism: TrainingParallelism,
    ) -> tuple[SequenceStats, ...]:
        """Return the DP rank with the largest quadratic attention workload."""
        ranks = self.rank_microbatches(parallelism)
        return max(
            ranks,
            key=lambda microbatches: (
                sum(item.sum_squared_tokens for item in microbatches),
                sum(item.sum_tokens for item in microbatches),
            ),
        )


@dataclass(frozen=True)
class TrainingCostEstimate:
    """Return latency and memory together for search and simulation."""

    latency_s: float
    peak_memory_bytes: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.latency_s <= 0:
            raise ValueError("Training latency must be positive.")
        if self.peak_memory_bytes < 0:
            raise ValueError("Training peak memory must be non-negative.")


class TrainingLatencyModel(ABC):
    """Predict training step latency."""

    @abstractmethod
    def predict(
        self,
        workload: TrainingWorkload,
        parallelism: TrainingParallelism,
    ) -> float:
        """Predict wall-clock latency in seconds."""


class TrainingMemoryModel(ABC):
    """Predict peak per-rank memory."""

    @abstractmethod
    def predict(
        self,
        workload: TrainingWorkload,
        parallelism: TrainingParallelism,
    ) -> int:
        """Predict peak memory in bytes on the most loaded rank."""


class TrainingCostModel:
    """Compose independently selectable latency and memory models."""

    def __init__(
        self,
        latency_model: TrainingLatencyModel,
        memory_model: TrainingMemoryModel,
    ):
        self.latency_model = latency_model
        self.memory_model = memory_model

    def estimate(
        self,
        workload: TrainingWorkload,
        parallelism: TrainingParallelism,
    ) -> TrainingCostEstimate:
        """Estimate both outputs without mutating model or workload state."""
        return TrainingCostEstimate(
            latency_s=self.latency_model.predict(workload, parallelism),
            peak_memory_bytes=self.memory_model.predict(workload, parallelism),
            diagnostics={
                "tp": parallelism.tp,
                "pp": parallelism.pp,
                "dp": parallelism.dp,
                "cp": parallelism.cp,
                "micro_batch_size": workload.micro_batch_size,
                "schedule": workload.schedule,
                "recompute": workload.recompute,
            },
        )
