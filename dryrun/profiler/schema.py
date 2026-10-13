"""Versioned profiling records shared by adapters and cost model fitters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, TypeAlias

SCHEMA_VERSION = "1.0"
BYTES_PER_MIB = 1024 * 1024
RecordType: TypeAlias = Literal["rollout", "training"]
RunStatus: TypeAlias = Literal["success", "oom", "failed", "skipped"]
RolloutPhase: TypeAlias = Literal["prefill", "decode", "mixed"]


class ProfileValidationError(ValueError):
    """Indicate that a profiling record violates the versioned contract."""


def utc_now() -> str:
    """Return an ISO 8601 timestamp in UTC."""
    return datetime.now(UTC).isoformat()


def mib_to_bytes(mib: int) -> int:
    """Convert a binary mebibyte count to bytes."""
    return int(mib) * BYTES_PER_MIB


def _positive(value: int | float, name: str, *, allow_zero: bool = False) -> None:
    valid = value >= 0 if allow_zero else value > 0
    if not valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ProfileValidationError(f"{name} must be {qualifier}, got {value!r}.")


@dataclass(frozen=True)
class Parallelism:
    """Describe one explicit model and data parallel topology."""

    tp: int = 1
    pp: int = 1
    dp: int = 1
    cp: int = 1
    # Expert parallelism, for mixture-of-experts blocks. Deliberately NOT a factor of
    # `world_size`: Megatron carves the expert groups out of the data-parallel dimension
    # (`parallel_state.initialize_model_parallel` derives
    # `expert_data_parallel_size = world_size // (etp * ep * pp)`), so a rank count is
    # already fully described by tp * pp * dp * cp. EP only changes how many experts each
    # rank owns and therefore the all-to-all traffic, which is exactly why it has to be a
    # cell key: two candidates differing only in EP run different collectives for the same
    # mesh, and sharing one fitted cell would average them.
    #
    # 1 means every rank holds every expert, which is the correct default for a dense model
    # and the only valid value when `num_moe_experts` is absent.
    ep: int = 1
    world_size: int | None = None

    def __post_init__(self) -> None:
        for name in ("tp", "pp", "dp", "cp", "ep"):
            _positive(getattr(self, name), name)
        product = self.tp * self.pp * self.dp * self.cp
        if self.world_size is None:
            object.__setattr__(self, "world_size", product)
        elif self.world_size != product:
            raise ProfileValidationError(
                "world_size must equal tp * pp * dp * cp, "
                f"got {self.world_size} != {self.tp} * {self.pp} * {self.dp} * {self.cp}."
            )
        # Megatron builds the expert mesh as `expert_tensor_parallel * ep * pp` over the
        # whole world. The Megatron launcher pins `expert_tensor_parallel_size = 1` (see
        # `adapters/megatron/launch.py` for why: following TP would force ep=1 at every
        # memory-feasible mesh), so a real mesh must satisfy `ep * pp | world_size`.
        #
        # Enforced only when `dp > 1`, which is what distinguishes a real mesh from a
        # canonical artifact cell key. A `Parallelism` is used for both:
        #
        # - A real mesh, where the constraint is Megatron's own and violating it raises a
        #   RuntimeError minutes into distributed init. Such a mesh always has `dp > 1` for
        #   this model, since EP is carved out of the data-parallel dimension and `ep > 1`
        #   needs ranks to carve from.
        # - A canonical cell key, where `to_profile_parallelism(canonical_dp=True)` collapses
        #   DP to 1 so one fitted cell describes every DP degree. That key has
        #   `world_size = tp * pp`, smaller than `ep * pp` whenever EP exceeds TP. It is
        #   legitimate, and it round-trips through `from_dict` when the artifact is reloaded,
        #   so rejecting it would make a correct artifact unloadable.
        expert_mesh = self.ep * self.pp
        if self.ep > 1 and self.dp > 1 and self.world_size % expert_mesh:
            raise ProfileValidationError(
                f"world_size ({self.world_size}) must be divisible by "
                f"ep * pp ({self.ep} * {self.pp} = {expert_mesh}); "
                "Megatron derives expert data parallelism from that quotient."
            )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Parallelism:
        return cls(
            tp=int(value.get("tp", 1)),
            pp=int(value.get("pp", 1)),
            dp=int(value.get("dp", 1)),
            cp=int(value.get("cp", 1)),
            ep=int(value.get("ep", 1)),
            world_size=int(value["world_size"]) if value.get("world_size") is not None else None,
        )


@dataclass(frozen=True)
class RunDescriptor:
    """Identify one independently launched benchmark repetition."""

    run_id: str
    repeat: int = 0
    status: RunStatus = "success"
    started_at: str = field(default_factory=utc_now)
    duration_s: float | None = None
    raw_artifact: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ProfileValidationError("run_id must not be empty.")
        _positive(self.repeat, "repeat", allow_zero=True)
        if self.status not in {"success", "oom", "failed", "skipped"}:
            raise ProfileValidationError(f"Unsupported run status {self.status!r}.")
        if self.duration_s is not None:
            _positive(self.duration_s, "duration_s", allow_zero=True)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunDescriptor:
        return cls(
            run_id=str(value["run_id"]),
            repeat=int(value.get("repeat", 0)),
            status=value.get("status", "success"),
            started_at=str(value.get("started_at", utc_now())),
            duration_s=float(value["duration_s"]) if value.get("duration_s") is not None else None,
            raw_artifact=value.get("raw_artifact"),
            error=value.get("error"),
        )


@dataclass(frozen=True)
class ModelDescriptor:
    """Capture model identity and architecture values used by a fit."""

    name: str
    revision: str | None = None
    architecture: str | None = None
    parameter_count: int | None = None
    num_layers: int | None = None
    hidden_size: int | None = None
    num_attention_heads: int | None = None
    num_query_groups: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ProfileValidationError("Model name must not be empty.")
        for name in (
            "parameter_count",
            "num_layers",
            "hidden_size",
            "num_attention_heads",
            "num_query_groups",
        ):
            value = getattr(self, name)
            if value is not None:
                _positive(value, name)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelDescriptor:
        return cls(
            name=str(value["name"]),
            revision=value.get("revision"),
            architecture=value.get("architecture"),
            parameter_count=value.get("parameter_count"),
            num_layers=value.get("num_layers"),
            hidden_size=value.get("hidden_size"),
            num_attention_heads=value.get("num_attention_heads"),
            num_query_groups=value.get("num_query_groups"),
            extra=dict(value.get("extra", {})),
        )


@dataclass(frozen=True)
class HardwareDescriptor:
    """Capture hardware identity and topology for reproducible fits."""

    gpu_name: str
    gpu_count: int
    gpu_memory_mib: int | None = None
    node_count: int = 1
    gpus_per_node: int | None = None
    interconnect: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.gpu_name:
            raise ProfileValidationError("gpu_name must not be empty.")
        _positive(self.gpu_count, "gpu_count")
        _positive(self.node_count, "node_count")
        if self.gpu_memory_mib is not None:
            _positive(self.gpu_memory_mib, "gpu_memory_mib")
        if self.gpus_per_node is not None:
            _positive(self.gpus_per_node, "gpus_per_node")

    @property
    def gpu_memory_bytes(self) -> int | None:
        if self.gpu_memory_mib is None:
            return None
        return mib_to_bytes(self.gpu_memory_mib)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> HardwareDescriptor:
        memory_mib = value.get("gpu_memory_mib")
        return cls(
            gpu_name=str(value["gpu_name"]),
            gpu_count=int(value["gpu_count"]),
            gpu_memory_mib=None if memory_mib is None else int(memory_mib),
            node_count=int(value.get("node_count", 1)),
            gpus_per_node=value.get("gpus_per_node"),
            interconnect=value.get("interconnect"),
            extra=dict(value.get("extra", {})),
        )


@dataclass(frozen=True)
class SoftwareDescriptor:
    """Capture the benchmark backend and relevant software versions."""

    backend: str
    backend_version: str
    source_commit: str | None = None
    python_version: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    command: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.backend or not self.backend_version:
            raise ProfileValidationError("Software backend and backend_version must not be empty.")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SoftwareDescriptor:
        return cls(
            backend=str(value["backend"]),
            backend_version=str(value["backend_version"]),
            source_commit=value.get("source_commit"),
            python_version=value.get("python_version"),
            torch_version=value.get("torch_version"),
            cuda_version=value.get("cuda_version"),
            command=tuple(str(item) for item in value.get("command", [])),
            extra=dict(value.get("extra", {})),
        )


@dataclass(frozen=True)
class RolloutPayload:
    """Store one prefill, decode, or mixed batch-step observation."""

    phase: RolloutPhase
    request_count: int
    input_tokens: int
    output_tokens: int
    n_tokens: int
    sum_l: float
    sum_l2: float
    ctxsum: float
    latency_s: float
    ttft_s: tuple[float, ...] = ()
    itl_s: tuple[float, ...] = ()
    request_input_lengths: tuple[int, ...] = ()
    request_output_lengths: tuple[int, ...] = ()
    source_request_ids: tuple[int, ...] = ()
    batch_step_index: int | None = None

    def __post_init__(self) -> None:
        if self.phase not in {"prefill", "decode", "mixed"}:
            raise ProfileValidationError(f"Unsupported rollout phase {self.phase!r}.")
        for name in ("request_count", "n_tokens"):
            _positive(getattr(self, name), name)
        for name in ("input_tokens", "output_tokens", "sum_l", "sum_l2", "ctxsum", "latency_s"):
            _positive(getattr(self, name), name, allow_zero=True)
        if self.request_input_lengths and len(self.request_input_lengths) != self.request_count:
            raise ProfileValidationError("request_input_lengths must have one entry per request.")
        if self.request_output_lengths and len(self.request_output_lengths) != self.request_count:
            raise ProfileValidationError("request_output_lengths must have one entry per request.")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RolloutPayload:
        return cls(
            phase=value["phase"],
            request_count=int(value["request_count"]),
            input_tokens=int(value["input_tokens"]),
            output_tokens=int(value["output_tokens"]),
            n_tokens=int(value["n_tokens"]),
            sum_l=float(value["sum_l"]),
            sum_l2=float(value["sum_l2"]),
            ctxsum=float(value["ctxsum"]),
            latency_s=float(value["latency_s"]),
            ttft_s=tuple(float(item) for item in value.get("ttft_s", [])),
            itl_s=tuple(float(item) for item in value.get("itl_s", [])),
            request_input_lengths=tuple(int(item) for item in value.get("request_input_lengths", [])),
            request_output_lengths=tuple(int(item) for item in value.get("request_output_lengths", [])),
            source_request_ids=tuple(int(item) for item in value.get("source_request_ids", [])),
            batch_step_index=value.get("batch_step_index"),
        )


@dataclass(frozen=True)
class SequenceStats:
    """Summarize one training microbatch while optionally preserving lengths."""

    count: int
    sum_tokens: int
    sum_squared_tokens: int
    max_tokens: int
    lengths: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _positive(self.count, "count")
        _positive(self.sum_tokens, "sum_tokens")
        _positive(self.sum_squared_tokens, "sum_squared_tokens")
        _positive(self.max_tokens, "max_tokens")
        if self.lengths:
            if len(self.lengths) != self.count:
                raise ProfileValidationError("SequenceStats.lengths must match count.")
            if any(length <= 0 for length in self.lengths):
                raise ProfileValidationError("Sequence lengths must be positive.")
            if sum(self.lengths) != self.sum_tokens:
                raise ProfileValidationError("SequenceStats.sum_tokens does not match lengths.")
            if sum(length * length for length in self.lengths) != self.sum_squared_tokens:
                raise ProfileValidationError("SequenceStats.sum_squared_tokens does not match lengths.")
            if max(self.lengths) != self.max_tokens:
                raise ProfileValidationError("SequenceStats.max_tokens does not match lengths.")

    @classmethod
    def from_lengths(cls, lengths: list[int] | tuple[int, ...]) -> SequenceStats:
        normalized = tuple(int(length) for length in lengths)
        if not normalized:
            raise ProfileValidationError("A microbatch must contain at least one sequence.")
        return cls(
            count=len(normalized),
            sum_tokens=sum(normalized),
            sum_squared_tokens=sum(length * length for length in normalized),
            max_tokens=max(normalized),
            lengths=normalized,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SequenceStats:
        return cls(
            count=int(value["count"]),
            sum_tokens=int(value["sum_tokens"]),
            sum_squared_tokens=int(value["sum_squared_tokens"]),
            max_tokens=int(value["max_tokens"]),
            lengths=tuple(int(item) for item in value.get("lengths", [])),
        )


@dataclass(frozen=True)
class RankMemory:
    """Store allocated and reserved CUDA peaks for one rank."""

    rank: int
    allocated_bytes: int
    reserved_bytes: int

    def __post_init__(self) -> None:
        _positive(self.rank, "rank", allow_zero=True)
        _positive(self.allocated_bytes, "allocated_bytes", allow_zero=True)
        _positive(self.reserved_bytes, "reserved_bytes", allow_zero=True)
        if self.reserved_bytes < self.allocated_bytes:
            raise ProfileValidationError("reserved_bytes must be at least allocated_bytes.")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RankMemory:
        return cls(
            rank=int(value["rank"]),
            allocated_bytes=int(value["allocated_bytes"]),
            reserved_bytes=int(value["reserved_bytes"]),
        )


# Pipeline schedule identifiers a training payload may carry. `1f1b` is Megatron's
# interleaved-free pipeline schedule; `fsdp` marks a backend with no pipeline at all,
# where the schedule axis degenerates rather than taking another value.
TRAINING_SCHEDULES = frozenset({"1f1b", "fsdp"})


@dataclass(frozen=True)
class TrainingPayload:
    """Store one training candidate's workload, timings, and memory peaks."""

    micro_batch_size: int
    global_batch_size: int
    num_microbatches: int
    schedule: str
    recompute: str
    optimizer: str
    microbatches: tuple[SequenceStats, ...]
    step_times_s: tuple[float, ...] = ()
    timers_s: dict[str, tuple[float, ...]] = field(default_factory=dict)
    rank_memory: tuple[RankMemory, ...] = ()
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    parameter_dtype: str = "bf16"
    gradient_dtype: str = "fp32"
    optimizer_state_dtype: str = "fp32"
    distributed_optimizer: bool = True
    parameter_count: int | None = None
    warmup_steps: int = 0
    packing: str = "bshd"
    """Sequence layout the measurement was taken under.

    Recorded because the two layouts price attention differently: BSHD charges the
    padded length per sequence, THD the sum of per-sequence squares. A fit cannot
    mix them.
    """

    def __post_init__(self) -> None:
        for name in ("micro_batch_size", "global_batch_size", "num_microbatches"):
            _positive(getattr(self, name), name)
        if self.schedule.lower() not in TRAINING_SCHEDULES:
            raise ProfileValidationError(
                f"Training schedule must be one of {sorted(TRAINING_SCHEDULES)}, "
                f"got {self.schedule!r}."
            )
        if len(self.microbatches) != self.num_microbatches:
            raise ProfileValidationError("microbatches must have num_microbatches entries.")
        # A THD pack is a single sample whatever its sequence count.
        if self.packing != "thd" and any(
            item.count != self.micro_batch_size for item in self.microbatches
        ):
            raise ProfileValidationError("Every microbatch count must equal micro_batch_size.")
        if any(value <= 0 for value in self.step_times_s):
            raise ProfileValidationError("step_times_s values must be positive.")
        if self.parameter_count is not None:
            _positive(self.parameter_count, "parameter_count")
        _positive(self.warmup_steps, "warmup_steps", allow_zero=True)
        if self.rank_memory:
            allocated = max(item.allocated_bytes for item in self.rank_memory)
            reserved = max(item.reserved_bytes for item in self.rank_memory)
            if self.peak_allocated_bytes is not None and self.peak_allocated_bytes != allocated:
                raise ProfileValidationError("peak_allocated_bytes must equal the maximum rank peak.")
            if self.peak_reserved_bytes is not None and self.peak_reserved_bytes != reserved:
                raise ProfileValidationError("peak_reserved_bytes must equal the maximum rank peak.")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainingPayload:
        return cls(
            micro_batch_size=int(value["micro_batch_size"]),
            global_batch_size=int(value["global_batch_size"]),
            num_microbatches=int(value["num_microbatches"]),
            schedule=str(value["schedule"]),
            recompute=str(value["recompute"]),
            optimizer=str(value["optimizer"]),
            microbatches=tuple(SequenceStats.from_dict(item) for item in value["microbatches"]),
            step_times_s=tuple(float(item) for item in value.get("step_times_s", [])),
            timers_s={
                str(name): tuple(float(item) for item in samples)
                for name, samples in value.get("timers_s", {}).items()
            },
            rank_memory=tuple(RankMemory.from_dict(item) for item in value.get("rank_memory", [])),
            peak_allocated_bytes=value.get("peak_allocated_bytes"),
            peak_reserved_bytes=value.get("peak_reserved_bytes"),
            parameter_dtype=str(value.get("parameter_dtype", "bf16")),
            gradient_dtype=str(value.get("gradient_dtype", "fp32")),
            optimizer_state_dtype=str(value.get("optimizer_state_dtype", "fp32")),
            distributed_optimizer=bool(value.get("distributed_optimizer", True)),
            parameter_count=value.get("parameter_count"),
            warmup_steps=int(value.get("warmup_steps", 0)),
            packing=str(value.get("packing", "bshd")),
        )


ProfilePayload: TypeAlias = RolloutPayload | TrainingPayload


@dataclass(frozen=True)
class ProfileRecord:
    """Represent one validated JSONL profiling observation."""

    record_type: RecordType
    run: RunDescriptor
    model: ModelDescriptor
    hardware: HardwareDescriptor
    software: SoftwareDescriptor
    parallelism: Parallelism
    payload: ProfilePayload
    schema_version: str = SCHEMA_VERSION
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ProfileValidationError(
                f"Unsupported schema_version {self.schema_version!r}; expected {SCHEMA_VERSION!r}."
            )
        expected_type = "rollout" if isinstance(self.payload, RolloutPayload) else "training"
        if self.record_type != expected_type:
            raise ProfileValidationError(
                f"record_type {self.record_type!r} does not match {type(self.payload).__name__}."
            )
        if self.parallelism.world_size != self.hardware.gpu_count:
            raise ProfileValidationError(
                f"Hardware gpu_count {self.hardware.gpu_count} does not match world_size "
                f"{self.parallelism.world_size}."
            )
        if isinstance(self.payload, TrainingPayload):
            expected_gbs = (
                self.payload.micro_batch_size
                * self.parallelism.dp
                * self.payload.num_microbatches
            )
            if self.payload.global_batch_size != expected_gbs:
                raise ProfileValidationError(
                    "global_batch_size must equal micro_batch_size * dp * num_microbatches, "
                    f"got {self.payload.global_batch_size} != {expected_gbs}."
                )

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to JSON-compatible built-in containers."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProfileRecord:
        record_type = value["record_type"]
        if record_type == "rollout":
            payload: ProfilePayload = RolloutPayload.from_dict(value["payload"])
        elif record_type == "training":
            payload = TrainingPayload.from_dict(value["payload"])
        else:
            raise ProfileValidationError(f"Unsupported record_type {record_type!r}.")
        return cls(
            schema_version=str(value.get("schema_version", "")),
            record_type=record_type,
            run=RunDescriptor.from_dict(value["run"]),
            model=ModelDescriptor.from_dict(value["model"]),
            hardware=HardwareDescriptor.from_dict(value["hardware"]),
            software=SoftwareDescriptor.from_dict(value["software"]),
            parallelism=Parallelism.from_dict(value["parallelism"]),
            payload=payload,
            tags={str(key): item for key, item in value.get("tags", {}).items()},
        )
