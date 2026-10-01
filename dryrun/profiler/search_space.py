"""Constrained candidate generation for Megatron-Bridge profiling sweeps."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

from .schema import Parallelism, ProfileValidationError


@dataclass(frozen=True)
class ModelConstraints:
    """Describe architecture divisibility constraints used by Megatron-Core."""

    num_layers: int
    num_attention_heads: int
    num_query_groups: int

    def __post_init__(self) -> None:
        for name in ("num_layers", "num_attention_heads", "num_query_groups"):
            if getattr(self, name) <= 0:
                raise ProfileValidationError(f"{name} must be positive.")


_PACKING_MODES = frozenset({"bshd", "thd"})


@dataclass(frozen=True)
class TrainingCandidate:
    """Describe one independently executable training profiling candidate."""

    parallelism: Parallelism
    micro_batch_size: int
    global_batch_size: int
    num_microbatches: int
    sequence_shape: tuple[int, ...]
    schedule: str = "1f1b"
    recompute: str = "none"
    repeat: int = 0
    packing: str = "bshd"
    """Sequence layout for the training forward pass.

    ``bshd`` pads every sequence in a microbatch to the longest, so attention costs
    ``O(max_len²)`` per sequence. ``thd`` concatenates them behind ``cu_seqlens`` as
    verl does with ``use_remove_padding=True``, costing ``O(Σsᵢ²)``. Those are
    different functions of the same batch, so a model fitted on one does not price
    the other: use ``thd`` to predict verl-style steps.
    """

    def __post_init__(self) -> None:
        if self.parallelism.cp != 1:
            raise ProfileValidationError("Training candidate V1 only supports CP=1.")
        if self.schedule.lower() != "1f1b":
            raise ProfileValidationError("Training candidate V1 only supports the 1F1B schedule.")
        if self.micro_batch_size <= 0 or self.global_batch_size <= 0 or self.num_microbatches <= 0:
            raise ProfileValidationError("Batch sizes and num_microbatches must be positive.")
        expected = self.micro_batch_size * self.parallelism.dp * self.num_microbatches
        if self.global_batch_size != expected:
            raise ProfileValidationError(
                "global_batch_size must equal MBS * DP * num_microbatches, "
                f"got {self.global_batch_size} != {expected}."
            )
        # Only BSHD gives each sequence its own padded row; a THD pack is one sample.
        if self.packing != "thd" and len(self.sequence_shape) != self.micro_batch_size:
            raise ProfileValidationError(
                "sequence_shape must contain one length per microbatch sequence."
            )
        if self.packing == "thd" and self.micro_batch_size != 1:
            raise ProfileValidationError(
                "THD packing trains one packed row per microbatch, so micro_batch_size "
                f"must be 1, got {self.micro_batch_size}."
            )
        if any(length <= 0 for length in self.sequence_shape):
            raise ProfileValidationError("All sequence lengths must be positive.")
        if self.repeat < 0:
            raise ProfileValidationError("repeat must be non-negative.")
        if self.packing not in _PACKING_MODES:
            raise ProfileValidationError(
                f"packing must be one of {sorted(_PACKING_MODES)}, got {self.packing!r}."
            )

    @property
    def candidate_id(self) -> str:
        """Return a compact, deterministic identity without encoding a lookup protocol."""
        value = json.dumps(self.to_dict(include_id=False), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = asdict(self)
        if include_id:
            value["candidate_id"] = self.candidate_id
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainingCandidate:
        return cls(
            parallelism=Parallelism.from_dict(value["parallelism"]),
            micro_batch_size=int(value["micro_batch_size"]),
            global_batch_size=int(value["global_batch_size"]),
            num_microbatches=int(value["num_microbatches"]),
            sequence_shape=tuple(int(item) for item in value["sequence_shape"]),
            schedule=str(value.get("schedule", "1f1b")),
            recompute=str(value.get("recompute", "none")),
            repeat=int(value.get("repeat", 0)),
            packing=str(value.get("packing", "bshd")),
        )


def tensor_parallel_is_valid(tp: int, constraints: ModelConstraints) -> bool:
    """Check attention head and grouped-query attention divisibility."""
    if constraints.num_attention_heads % tp != 0:
        return False
    return constraints.num_query_groups % tp == 0 or tp % constraints.num_query_groups == 0


def enumerate_training_candidates(
    *,
    world_sizes: list[int],
    tensor_parallel_sizes: list[int],
    pipeline_parallel_sizes: list[int],
    micro_batch_sizes: list[int],
    num_microbatches: list[int],
    sequence_shapes: list[list[int] | tuple[int, ...]],
    constraints: ModelConstraints,
    repeats: int = 1,
    context_parallel_sizes: list[int] | None = None,
    schedules: list[str] | None = None,
    recompute_modes: list[str] | None = None,
    data_parallel_sizes: list[int] | None = None,
    packing: str = "bshd",
) -> list[TrainingCandidate]:
    """Enumerate only candidates that satisfy V1 topology and batch constraints.

    Args:
        data_parallel_sizes: Pin DP instead of letting it absorb whatever GPUs
            TP and PP leave over. Leave None for the residual behaviour
            `dp = world_size // (tp * pp * cp)`.

            Pinning matters whenever a sweep is meant to isolate TP or PP. With
            DP as a residual and a fixed world size, `tp1/pp1` lands on dp8
            while `tp1/pp4` lands on dp2, so every TP or PP comparison is also a
            DP comparison, and DP changes both the gradient all-reduce volume
            and the per-rank token count. Pinning DP=1 leaves GPUs idle at the
            smaller meshes, which costs nothing and keeps the comparison clean.
    """
    if repeats <= 0:
        raise ProfileValidationError("repeats must be positive.")
    if any(value <= 0 for value in num_microbatches):
        raise ProfileValidationError("num_microbatches values must be positive.")
    if data_parallel_sizes is not None and any(value <= 0 for value in data_parallel_sizes):
        raise ProfileValidationError("data_parallel_sizes values must be positive.")
    cp_values = context_parallel_sizes or [1]
    schedule_values = schedules or ["1f1b"]
    recompute_values = recompute_modes or ["full"]
    candidates: list[TrainingCandidate] = []

    # `None` is the residual marker, kept inside the product so the pinned and
    # residual paths enumerate through one loop instead of two.
    dp_values: list[int | None] = (
        sorted(set(data_parallel_sizes)) if data_parallel_sizes is not None else [None]
    )

    dimensions = product(
        sorted(set(world_sizes)),
        sorted(set(tensor_parallel_sizes)),
        sorted(set(pipeline_parallel_sizes)),
        sorted(set(cp_values)),
        dp_values,
        sorted(set(micro_batch_sizes)),
        sorted(set(num_microbatches)),
        sequence_shapes,
        schedule_values,
        recompute_values,
    )
    for world_size, tp, pp, cp, pinned_dp, mbs, nmb, raw_shape, schedule, recompute in dimensions:
        if cp != 1 or schedule.lower() != "1f1b":
            continue
        model_parallel = tp * pp * cp
        if not tensor_parallel_is_valid(tp, constraints):
            continue
        if constraints.num_layers % pp != 0:
            continue
        if pinned_dp is None:
            if world_size % model_parallel != 0:
                continue
            dp = world_size // model_parallel
            mesh_size = world_size
        else:
            # A pinned mesh only has to fit in the world, not fill it. The
            # leftover GPUs stay idle rather than inflating DP.
            dp = pinned_dp
            mesh_size = model_parallel * dp
            if mesh_size > world_size:
                continue
        shape = tuple(int(item) for item in raw_shape)
        if any(length <= 0 for length in shape):
            continue
        # A THD pack is one sample however many sequences it holds, so `mbs` stays 1
        # and the shape is free to vary in length. Setting it to the sequence count
        # makes Megatron expect `count * seq_length` tokens, which leaves the ranks
        # disagreeing on collective sizes and deadlocks the step in NCCL.
        effective_mbs = 1 if packing == "thd" else mbs
        if packing != "thd" and len(shape) != mbs:
            continue

        # `world_size` is the mesh's own rank count, because that is what the
        # launcher turns into `--nproc-per-node`. For a pinned DP it is the mesh
        # product rather than the hardware GPU count, so the idle GPUs are never
        # handed ranks.
        parallelism = Parallelism(tp=tp, pp=pp, dp=dp, cp=cp, world_size=mesh_size)
        for repeat in range(repeats):
            candidates.append(
                TrainingCandidate(
                    parallelism=parallelism,
                    micro_batch_size=effective_mbs,
                    global_batch_size=effective_mbs * dp * nmb,
                    num_microbatches=nmb,
                    sequence_shape=shape,
                    schedule=schedule,
                    recompute=recompute,
                    repeat=repeat,
                    packing=packing,
                )
            )
    return candidates
