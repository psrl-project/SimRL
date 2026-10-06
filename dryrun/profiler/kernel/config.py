"""Configuration for direct kernel profiling of attention and GEMM ops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...cost.kernel.hardware import gpu_slug
from ...cost.kernel.ops.base import families, op_for, ops_for_family
from ..hf import model_descriptor_from_yaml
from ..schema import HardwareDescriptor, ModelDescriptor, ProfileValidationError

# Raggedness budgets, sized against the hardware this profiler targets rather than set to
# a round number. Both count KV *elements*, which cost 4 bytes each at bfloat16 because K
# and V are both stored.
#
# A context point writes `batch_size * seq_len` tokens of KV, so 262144 tokens is a 1 GiB
# write. A generation point reads `batch_size * step * num_kv_heads * head_dim` elements,
# so 1 << 34 is a 64 GiB cache. The largest decode point the H20 sweeps configure
# (batch 64, step 32768, 8 KV heads, head_dim 128) is 8 GiB, which leaves ample headroom
# inside the ~75 GiB an H20 has free after Qwen3-8B's bfloat16 weights.
#
# These defaults are deliberately generous: a budget that is too small silently drops
# sweep points, and a sparse grid starves the holdout gate of the interior points it
# needs. Lower them explicitly in a config when profiling a smaller card.
DEFAULT_MAX_CONTEXT_TOKENS = 262144
DEFAULT_MAX_GENERATION_KV_TOKENS = 1 << 34


def _positive_int_tuple(value: Any, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ProfileValidationError(f"{field} must be a non-empty list of positive integers.")
    parsed = tuple(int(item) for item in value)
    if any(item <= 0 for item in parsed):
        raise ProfileValidationError(f"{field} entries must be positive.")
    return parsed


def _non_negative_int_tuple(value: Any, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ProfileValidationError(f"{field} must be a non-empty list of non-negative integers.")
    parsed = tuple(int(item) for item in value)
    if any(item < 0 for item in parsed):
        raise ProfileValidationError(f"{field} entries must not be negative.")
    return parsed


def model_head_dim(model: ModelDescriptor) -> int:
    """Return the per-head dimension, inferring it when the checkpoint omits it."""
    head_dim = model.extra.get("head_dim")
    if head_dim is not None:
        return int(head_dim)
    if model.hidden_size is None or model.num_attention_heads is None:
        raise ProfileValidationError("Model is missing hidden_size or num_attention_heads.")
    return model.hidden_size // model.num_attention_heads


def model_intermediate_size(model: ModelDescriptor) -> int:
    """Return the MLP intermediate size, falling back to the common 4x-hidden rule."""
    intermediate_size = model.extra.get("intermediate_size")
    if intermediate_size is not None:
        return int(intermediate_size)
    if model.hidden_size is None:
        raise ProfileValidationError("Model is missing hidden_size.")
    return 4 * model.hidden_size


def model_vocab_size(model: ModelDescriptor) -> int:
    """Return the vocabulary size from the checkpoint's extra fields."""
    vocab_size = model.extra.get("vocab_size")
    if vocab_size is None:
        raise ProfileValidationError("Model is missing vocab_size in extra fields.")
    return int(vocab_size)


@dataclass(frozen=True)
class GemmSweep:
    """Token counts to measure every per-rank GEMM shape at."""

    token_counts: tuple[int, ...]
    logits_request_counts: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GemmSweep:
        return cls(
            token_counts=_positive_int_tuple(value["token_counts"], field="gemm.token_counts"),
            logits_request_counts=_positive_int_tuple(
                value.get("logits_request_counts", [1, 2, 4, 8, 16, 32, 48, 64]),
                field="gemm.logits_request_counts",
            ),
        )


@dataclass(frozen=True)
class ElementwiseSweep:
    """Token counts to measure every elementwise kernel at."""

    token_counts: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ElementwiseSweep:
        return cls(
            token_counts=_positive_int_tuple(value["token_counts"], field="elementwise.token_counts")
        )


@dataclass(frozen=True)
class TrainingAttentionSweep:
    """Packed token counts and head counts to measure THD attention at, both directions.

    `total_tokens` rather than `(batch_size, seq_len)`: a THD microbatch is one packed row,
    so batch size is always 1 and the pack length is the axis. That is also why the rollout
    `context_attention` table cannot serve this op even where its shapes overlap, since its
    grid varies the batch dimension that a packed layout does not have.
    """

    total_tokens: tuple[int, ...]
    num_heads: tuple[int, ...]
    mask_types: tuple[str, ...] = ("padding_causal",)

    def __post_init__(self) -> None:
        from ...cost.kernel.ops.attention_training import TRAINING_MASK_TYPES

        unknown = sorted(set(self.mask_types) - set(TRAINING_MASK_TYPES))
        if unknown:
            known = ", ".join(TRAINING_MASK_TYPES)
            raise ProfileValidationError(
                f"attention_training.mask_types holds unknown values {unknown}. "
                f"Known types: {known}. Plain 'causal' is rejected by Transformer Engine "
                f"for qkv_format='thd'."
            )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainingAttentionSweep:
        return cls(
            total_tokens=_positive_int_tuple(
                value["total_tokens"], field="attention_training.total_tokens"
            ),
            num_heads=_positive_int_tuple(
                value["num_heads"], field="attention_training.num_heads"
            ),
            mask_types=tuple(str(item) for item in value.get("mask_types", ("padding_causal",))),
        )


@dataclass(frozen=True)
class OptimizerSweep:
    """Parameter-shard sizes to measure the fused Adam step at.

    Parameter counts rather than token counts: an Adam step's cost is independent of how
    many tokens produced the gradients. This is the only sweep in the tree with no token
    axis, and that is a property of the op rather than an omission.

    The grid should bracket `total_parameters / (tp * pp * dp)` for the configurations being
    modelled, since that is the shard one rank updates. Qwen3-8B at tp=2, pp=3 gives
    1.365B, and a dp=8 distributed optimizer would divide that again.
    """

    param_counts: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OptimizerSweep:
        return cls(
            param_counts=_positive_int_tuple(
                value["param_counts"], field="optimizer.param_counts"
            )
        )


@dataclass(frozen=True)
class CollectiveSweep:
    """Message sizes and topologies to measure every collective at.

    World size is declared here rather than read from `parallelism.tensor_parallel`, which
    is how the original all-reduce sweep worked. That coupling does not extend: a
    data-parallel gradient reduction at 24 ranks has nothing to do with a tensor-parallel
    degree, and writing `tensor_parallel: 24` to measure one would misdescribe what was
    measured. It also forced one config file per degree, which
    `examples/profile/kernel/profile_qwen3_8b.sh` works around by re-running with
    `FAMILY=collective`.
    """

    message_bytes: tuple[int, ...]
    """Per-rank message sizes in bytes, swept directly.

    Not derived from token counts, because the five collectives do not all carry the same
    size at the same token count: an SP-sharded reduce-scatter moves `tokens*hidden/tp`
    where an all-reduce moves `tokens*hidden`, and a gradient bucket reduction is unrelated
    to tokens entirely (Megatron buckets at `max(40M, 1M*dp)` *parameters*, so an 8.19B
    model reduces ~205 buckets of ~153 MiB). Sweeping bytes covers all three without
    teaching this config each caller's formula.
    """

    topologies: tuple[tuple[int, int], ...]
    """`(world_size, nodes)` pairs to measure.

    An explicit pair list rather than a cross product of two axes: `world_size=2` with
    `nodes=3` describes no placement, and a cross product would spawn runs that cannot form
    a mesh.
    """

    hostnames: tuple[str, ...] = ()
    """Hosts available for multi-node rendezvous, rank 0 first.

    Empty is valid and means intra-node only. A topology asking for more nodes than there
    are hostnames is rejected at parse time rather than silently measuring fewer, because a
    cross-node row that was secretly intra-node is worse than no row.
    """

    dtypes: tuple[str, ...] = ("bfloat16",)
    """Dtypes to sweep. `float32` is required to price Megatron's fp32 gradient reduction."""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CollectiveSweep:
        message_bytes = _positive_int_tuple(
            value["message_bytes"], field="collective.message_bytes"
        )
        raw_topologies = value.get("topologies")
        if not isinstance(raw_topologies, (list, tuple)) or not raw_topologies:
            raise ProfileValidationError(
                "collective.topologies must be a non-empty list of [world_size, nodes] pairs."
            )
        topologies: list[tuple[int, int]] = []
        for item in raw_topologies:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ProfileValidationError(
                    f"collective.topologies entries must be [world_size, nodes] pairs, "
                    f"got {item!r}."
                )
            world_size, nodes = int(item[0]), int(item[1])
            if world_size <= 0 or nodes <= 0:
                raise ProfileValidationError(
                    f"collective.topologies entries must be positive, got {item!r}."
                )
            if nodes > world_size:
                raise ProfileValidationError(
                    f"collective.topologies entry {item!r} places {nodes} nodes in a "
                    f"{world_size}-rank group, which describes no rank placement."
                )
            if world_size % nodes:
                raise ProfileValidationError(
                    f"collective.topologies entry {item!r} does not divide ranks evenly "
                    f"across nodes. An uneven placement measures a topology no launcher builds."
                )
            topologies.append((world_size, nodes))
        hostnames = tuple(str(item) for item in value.get("hostnames", ()))
        required_nodes = max(nodes for _, nodes in topologies)
        if required_nodes > 1 and len(hostnames) < required_nodes:
            raise ProfileValidationError(
                f"collective.topologies needs {required_nodes} nodes but collective.hostnames "
                f"lists {len(hostnames)}. Add the missing hosts rather than letting a "
                f"cross-node sweep run on fewer nodes than it claims."
            )
        dtypes = tuple(str(item) for item in value.get("dtypes", ("bfloat16",)))
        if not dtypes:
            raise ProfileValidationError("collective.dtypes must be non-empty when present.")
        return cls(
            message_bytes=message_bytes,
            topologies=tuple(topologies),
            hostnames=hostnames,
            dtypes=dtypes,
        )


@dataclass(frozen=True)
class AttentionSweep:
    """Grid of attention shapes, with the memory budgets that truncate its corners."""

    batch_sizes: tuple[int, ...]
    context_seq_lens: tuple[int, ...]
    generation_steps: tuple[int, ...]
    block_size: int = 16
    window_size: int = 0
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    max_generation_kv_tokens: int = DEFAULT_MAX_GENERATION_KV_TOKENS

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ProfileValidationError(f"attention.block_size must be positive, got {self.block_size!r}.")
        if self.window_size < 0:
            raise ProfileValidationError(f"attention.window_size must not be negative, got {self.window_size!r}.")
        if self.max_context_tokens <= 0:
            raise ProfileValidationError("attention.max_context_tokens must be positive.")
        if self.max_generation_kv_tokens <= 0:
            raise ProfileValidationError("attention.max_generation_kv_tokens must be positive.")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AttentionSweep:
        return cls(
            batch_sizes=_positive_int_tuple(value["batch_sizes"], field="attention.batch_sizes"),
            context_seq_lens=_positive_int_tuple(
                value["context_seq_lens"], field="attention.context_seq_lens"
            ),
            generation_steps=_non_negative_int_tuple(
                value["generation_steps"], field="attention.generation_steps"
            ),
            block_size=int(value.get("block_size", 16)),
            window_size=int(value.get("window_size", 0)),
            max_context_tokens=int(value.get("max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS)),
            max_generation_kv_tokens=int(
                value.get("max_generation_kv_tokens", DEFAULT_MAX_GENERATION_KV_TOKENS)
            ),
        )


@dataclass(frozen=True)
class MeasurementConfig:
    """Warmup/active iteration counts and dtype shared by every kernel profiler."""

    warmup_iterations: int = 5
    active_iterations: int = 20
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.warmup_iterations < 0:
            raise ProfileValidationError("measurement.warmup_iterations must not be negative.")
        if self.active_iterations <= 0:
            raise ProfileValidationError("measurement.active_iterations must be positive.")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MeasurementConfig:
        return cls(
            warmup_iterations=int(value.get("warmup_iterations", 5)),
            active_iterations=int(value.get("active_iterations", 20)),
            dtype=str(value.get("dtype", "bfloat16")),
            kv_cache_dtype=str(value.get("kv_cache_dtype", "bfloat16")),
        )


@dataclass(frozen=True)
class KernelProfileConfig:
    """Configuration required to profile kernels, with one sweep per op."""

    model: ModelDescriptor
    hardware: HardwareDescriptor
    output_dir: Path
    experiment_name: str
    tensor_parallel: int
    measurement: MeasurementConfig
    sweeps: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.experiment_name:
            raise ProfileValidationError("experiment_name must not be empty.")
        if self.tensor_parallel <= 0:
            raise ProfileValidationError("parallelism.tensor_parallel must be positive.")

    def sweep(self, op_name: str) -> Any:
        """
        Return the parsed sweep grid for one op.

        Args:
            op_name (str): A registered op name.

        Returns:
            Any: The op's own sweep type, e.g. `GemmSweep` or `AttentionSweep`.

        Raises:
            ProfileValidationError: If the config carries no sweep for this op's family.
        """
        try:
            return self.sweeps[op_name]
        except KeyError as exc:
            available = ", ".join(sorted(self.sweeps)) or "none"
            raise ProfileValidationError(
                f"No sweep configured for op {op_name!r}. Configured ops: {available}. "
                f"Add a `{op_for(op_name).family}:` block to the kernel profile config."
            ) from exc

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, base_dir: Path | None = None) -> KernelProfileConfig:
        root = base_dir or Path.cwd()
        output_dir = Path(value.get("output_dir", "profile_output"))
        if not output_dir.is_absolute():
            output_dir = (root / output_dir).resolve()
        parallelism = value.get("parallelism", {})
        # Each op parses its own family's subtree, so adding an op adds no field here. A family
        # absent from the config yields no entry, and `sweep()` raises naming the missing block
        # only if something actually asks for it: profiling one family must not require
        # configuring them all.
        sweeps: dict[str, Any] = {}
        for family in families():
            subtree = value.get(family)
            if subtree is None:
                continue
            for op in ops_for_family(family):
                sweeps[op.name] = op.parse_sweep(subtree)
        return cls(
            model=model_descriptor_from_yaml(value["model"], base_dir=root),
            hardware=HardwareDescriptor.from_dict(value["hardware"]),
            output_dir=output_dir,
            experiment_name=str(value.get("experiment_name", "kernel")),
            tensor_parallel=int(parallelism.get("tensor_parallel", 1)),
            measurement=MeasurementConfig.from_dict(value.get("measurement", {})),
            sweeps=sweeps,
        )

    def table_root(self) -> Path:
        """Return the root of the parquet perf-table tree."""
        return self.output_dir / "kernel"

    def gpu(self) -> str:
        """Return the perf-table directory slug for this hardware."""
        return gpu_slug(self.hardware.gpu_name)
