"""Candidate generation for FSDP2 and HSDP profiling sweeps.

An FSDP candidate is described by a device mesh of two integers rather than the
tensor, pipeline, and context degrees Megatron needs. Data parallelism is the whole
topology: HSDP replicates *parameters* across mesh rows, not data, so every rank
still consumes a distinct microbatch and the profile topology is pure DP.

Every microbatch a rank runs is identically shaped, which is what keeps the measured
GEMM shapes clean. The *number* of them is nonetheless a swept axis, because the
latency design row is `[1, nmb*tokens, nmb]` and at a fixed `nmb` the third column is
collinear with the intercept: the design caps at rank two where the model needs three,
so `token_cost` is identifiable but `microbatch_overhead` and `per_step_overhead` are
not separable. Running `nmb` identically-shaped microbatches with gradient accumulation
is what real FSDP training does anyway, so the axis costs no fidelity. Forming *ragged*
microbatches from real trajectory lengths remains the simulate-side planner's job.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

from .schema import Parallelism, ProfileValidationError


@dataclass(frozen=True)
class FsdpCandidate:
    """Describe one independently executable FSDP profiling candidate."""

    shard_size: int
    replicate_size: int
    n_seqs_per_rank: int
    sequence_length: int
    gradient_checkpointing: bool
    optimizer_offload: bool
    reshard_after_forward: bool
    num_microbatches: int = 1
    sequence_parallel: int = 1
    repeat: int = 0

    def __post_init__(self) -> None:
        for name in (
            "shard_size",
            "replicate_size",
            "n_seqs_per_rank",
            "sequence_length",
            "num_microbatches",
        ):
            if getattr(self, name) <= 0:
                raise ProfileValidationError(f"{name} must be positive.")
        if self.sequence_parallel != 1:
            raise ProfileValidationError(
                "FSDP profiling V1 requires sequence_parallel=1. Ulysses needs an "
                "all-to-all attention wrapper that is not implemented."
            )
        if self.repeat < 0:
            raise ProfileValidationError("repeat must be non-negative.")

    @property
    def world_size(self) -> int:
        return self.shard_size * self.replicate_size

    @property
    def micro_batch_size(self) -> int:
        return self.n_seqs_per_rank

    @property
    def global_batch_size(self) -> int:
        return self.n_seqs_per_rank * self.world_size * self.num_microbatches

    @property
    def sequence_shape(self) -> tuple[int, ...]:
        """Return one microbatch's shape; every microbatch a rank runs is identical."""
        return (self.sequence_length,) * self.n_seqs_per_rank

    @property
    def parallelism(self) -> Parallelism:
        """
        Return the topology as the shared schema expresses it.

        Sharding and replication are configuration, not data partitioning, so they
        are carried by the artifact selector rather than by this type. Every rank
        holds different data, which is what `dp` means here.
        """
        return Parallelism(tp=1, pp=1, dp=self.world_size, cp=1, world_size=self.world_size)

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
    def from_dict(cls, value: dict[str, Any]) -> FsdpCandidate:
        return cls(
            shard_size=int(value["shard_size"]),
            replicate_size=int(value["replicate_size"]),
            n_seqs_per_rank=int(value["n_seqs_per_rank"]),
            sequence_length=int(value["sequence_length"]),
            gradient_checkpointing=bool(value["gradient_checkpointing"]),
            optimizer_offload=bool(value["optimizer_offload"]),
            reshard_after_forward=bool(value["reshard_after_forward"]),
            num_microbatches=int(value["num_microbatches"]),
            sequence_parallel=int(value.get("sequence_parallel", 1)),
            repeat=int(value.get("repeat", 0)),
        )


def enumerate_fsdp_candidates(
    *,
    world_size: int,
    shard_sizes: list[int],
    n_seqs_per_rank: list[int],
    sequence_lengths: list[int],
    gradient_checkpointing: list[bool],
    optimizer_offload: list[bool],
    reshard_after_forward: list[bool],
    num_microbatches: list[int] | None = None,
    sequence_parallel_sizes: list[int] | None = None,
    repeats: int = 1,
) -> list[FsdpCandidate]:
    """
    Enumerate the candidates a fixed world size admits.

    The replicate degree is derived rather than swept: the mesh must cover the world
    exactly, so a shard degree that does not divide it describes no runnable topology.

    Args:
        num_microbatches (list[int] | None): Gradient-accumulation depths to sweep.
            Defaults to `[1]`, which leaves the latency design at rank two. Pass at
            least two values for the fit to separate the per-microbatch cost from the
            per-step one.
    """
    if repeats <= 0:
        raise ProfileValidationError("repeats must be positive.")
    if world_size <= 0:
        raise ProfileValidationError("world_size must be positive.")
    sp_values = sequence_parallel_sizes or [1]
    # Reject Ulysses before touching the product loop so callers get a clear error
    # rather than an empty sweep that looks like a config typo.
    if any(sp != 1 for sp in sp_values):
        raise ProfileValidationError(
            "FSDP profiling V1 requires sequence_parallel=1. Ulysses needs an "
            "all-to-all attention wrapper that is not implemented."
        )
    # A non-positive shard_size is never a valid topology: surface it immediately
    # instead of silently filtering it out and masking the caller's mistake.
    if any(s <= 0 for s in shard_sizes):
        raise ProfileValidationError("shard_size must be positive.")
    nmb_values = num_microbatches or [1]
    if any(value <= 0 for value in nmb_values):
        raise ProfileValidationError("num_microbatches must be positive.")
    candidates: list[FsdpCandidate] = []
    dimensions = product(
        sorted(set(shard_sizes)),
        sorted(set(n_seqs_per_rank)),
        sorted(set(sequence_lengths)),
        sorted(set(nmb_values)),
        sorted(set(gradient_checkpointing)),
        sorted(set(optimizer_offload)),
        sorted(set(reshard_after_forward)),
        sorted(set(sp_values)),
    )
    for shard, n_seqs, length, nmb, checkpointing, offload, reshard, sp in dimensions:
        # A shard degree that does not divide the world describes no runnable topology;
        # the caller may intentionally pass a superset of sizes, so skip rather than raise.
        if world_size % shard != 0:
            continue
        for repeat in range(repeats):
            candidates.append(
                FsdpCandidate(
                    shard_size=shard,
                    replicate_size=world_size // shard,
                    n_seqs_per_rank=n_seqs,
                    sequence_length=length,
                    num_microbatches=nmb,
                    gradient_checkpointing=checkpointing,
                    optimizer_offload=offload,
                    reshard_after_forward=reshard,
                    sequence_parallel=sp,
                    repeat=repeat,
                )
            )
    return candidates
