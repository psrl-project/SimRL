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
class CollectiveSweep:
    """Token counts to measure all-reduce message sizes at."""

    token_counts: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CollectiveSweep:
        return cls(
            token_counts=_positive_int_tuple(
                value["token_counts"], field="collective.token_counts"
            )
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
