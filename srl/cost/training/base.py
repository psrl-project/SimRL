"""Training workload, parallelism, and composable cost interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from srl.profiler.schema import (
    TRAINING_SCHEDULES,
    ModelDescriptor,
    Parallelism,
    SequenceStats,
)


class TrainingOutOfDomainError(ValueError):
    """Indicate that a query is outside a fitted training domain."""


@dataclass(frozen=True)
class TrainingParallelism:
    """Describe training model and data parallel dimensions."""

    tp: int = 1
    pp: int = 1
    dp: int = 1
    cp: int = 1
    # Expert parallelism. Not a factor of `world_size` (Megatron carves the expert groups
    # out of the data-parallel dimension), but it IS part of the cell key, because two
    # candidates with the same mesh and different EP run different all-to-all traffic and
    # hold a different number of experts per rank. See `profiler/schema.Parallelism.ep`.
    ep: int = 1

    def __post_init__(self) -> None:
        if min(self.tp, self.pp, self.dp, self.cp, self.ep) <= 0:
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
            ep=self.ep,
            world_size=self.tp * self.pp * dp * self.cp,
        )

    @classmethod
    def from_profile_parallelism(cls, value: Parallelism) -> TrainingParallelism:
        return cls(tp=value.tp, pp=value.pp, dp=value.dp, cp=value.cp, ep=value.ep)


def _attention_workload(length: int) -> int:
    """Return the per-sequence cost proxy the balancer sorts on, `24576 * s + s^2`.

    This is `12 * hidden^2 * s + 2 * hidden * s^2` at hidden=4096, divided through.
    It matches the linear-plus-quadratic shape of this module's own latency form,
    which is why balancing on it yields microbatches of roughly equal cost.
    """
    return 24576 * length + length * length


def _pack_to_token_budget(lengths: list[int], budget: int) -> list[list[int]]:
    """Partition `lengths` the way `rearrange_micro_batches` does.

    Not greedy bin packing. The microbatch *count* is fixed first from the token
    total,

        n = min(batch_size, ceil(sum(lengths) / budget))

    and the sequences are then split into exactly `n` groups balanced on
    `_attention_workload`, rather than filling one group at a time until it
    overflows. The distinction matters because greedy first-fit produces
    noticeably more groups on a ragged batch, and the count sets the pipeline slot
    count that every per-slot cost is multiplied by.

    Padding is not modelled: a packed microbatch costs the tokens actually present,
    and attention stays inside each `cu_seqlens` boundary, which is why
    `SequenceStats` keeps the per-sequence squares rather than squaring the total.

    Args:
        lengths: Sequence lengths for one DP rank, in any order.
        budget: `ppo_max_token_len_per_gpu * sp_size`. Sets the count, and is not
            a hard per-group cap once the count is fixed.

    Returns:
        One list of lengths per microbatch.
    """
    if not lengths:
        return []
    count = min(len(lengths), max(1, -(-sum(lengths) // budget)))
    groups: list[list[int]] = [[] for _ in range(count)]
    loads = [0] * count
    # Longest-processing-time first, the standard approximation to the balanced
    # partition the real balancer solves, and close enough that the slot count matches.
    for length in sorted(lengths, reverse=True):
        target = loads.index(min(loads))
        groups[target].append(length)
        loads[target] += _attention_workload(length)
    return [group for group in groups if group]


@dataclass(frozen=True)
class TrainingWorkload:
    """Represent one global RL training batch using its actual trajectory lengths."""

    sequence_lengths: tuple[int, ...]
    micro_batch_size: int
    schedule: str = "1f1b"
    recompute: str = "none"
    optimizer: str = "adam"
    max_tokens_per_microbatch: int | None = None
    """Token budget per microbatch, which replaces `micro_batch_size` when set.

    A real RL trainer does not group a ragged batch by sequence *count*: with dynamic
    batching the microbatch count falls out of the lengths instead. The difference is
    large, because a ragged batch packs into far fewer microbatches than it has
    sequences and every per-slot cost is charged once per slot.

    The budget is the per-GPU token limit times the ulysses degree, which is not
    Megatron's `sequence_parallel` flag. The caller passes the product.
    """

    def __post_init__(self) -> None:
        if not self.sequence_lengths or any(length <= 0 for length in self.sequence_lengths):
            raise ValueError("Training sequence lengths must be non-empty and positive.")
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive.")
        if self.max_tokens_per_microbatch is not None and self.max_tokens_per_microbatch <= 0:
            raise ValueError("max_tokens_per_microbatch must be positive when set.")
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
        """Balance lengths across DP ranks and split each rank into microbatches.

        Grouping is by token budget when `max_tokens_per_microbatch` is set, and
        by sequence count otherwise. See that field for why the distinction
        changes the prediction by a factor, not a few percent.
        """
        if self.max_tokens_per_microbatch is None:
            divisor = parallelism.dp * self.micro_batch_size
            if self.global_batch_size % divisor != 0:
                raise TrainingOutOfDomainError(
                    "global_batch_size must be divisible by dp * micro_batch_size, "
                    f"got {self.global_batch_size} % {divisor}."
                )
        ordered = sorted(self.sequence_lengths, reverse=True)
        rank_lengths = [ordered[index :: parallelism.dp] for index in range(parallelism.dp)]
        ranks: list[tuple[SequenceStats, ...]] = []
        for lengths in rank_lengths:
            if self.max_tokens_per_microbatch is None:
                groups = [
                    lengths[start : start + self.micro_batch_size]
                    for start in range(0, len(lengths), self.micro_batch_size)
                ]
            else:
                groups = _pack_to_token_budget(lengths, self.max_tokens_per_microbatch)
            ranks.append(tuple(SequenceStats.from_lengths(group) for group in groups))
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
        model: ModelDescriptor | None = None,
    ):
        self.latency_model = latency_model
        self.memory_model = memory_model
        self.model = model
        """Architecture the fit describes, when it came from an artifact.

        Anything deriving a quantity from the model's geometry must read it from
        here rather than being configured separately, so it cannot describe a
        different model than the curve it is scaling. None for the analytical
        and constant models, which are not tied to a checkpoint.
        """

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
                "ep": parallelism.ep,
                "micro_batch_size": workload.micro_batch_size,
                "schedule": workload.schedule,
                "recompute": workload.recompute,
            },
        )
