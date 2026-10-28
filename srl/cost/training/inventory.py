"""Static parameter, gradient, and optimizer state inventory.

Sizing follows Megatron's distributed-optimizer layout, which was measured directly
rather than assumed. For Qwen3-8B at TP1/PP1/DP8 the allocator reported
`param_data` at 15.256 GiB and `grad_data` at 30.513 GiB: exactly `2p` and `4p`
with no data-parallel division, while the fp32 master shard was 3.814 GiB, exactly
`4p/DP`. The gradient buffer is the reduce-scatter *source*, so every rank holds it
at full local size. Only optimizer state is sharded.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import TrainingParallelism, stage_layer_counts

_DTYPE_BYTES = {
    "fp32": 4,
    "float32": 4,
    "tf32": 4,
    "fp16": 2,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "fp8": 1,
    "int8": 1,
}


def dtype_bytes(dtype: str) -> int:
    """Return storage bytes for a supported training dtype."""
    normalized = dtype.lower()
    if normalized not in _DTYPE_BYTES:
        raise ValueError(f"Unsupported training dtype {dtype!r}.")
    return _DTYPE_BYTES[normalized]


@dataclass(frozen=True)
class ParameterInventory:
    """Break down per-rank static state bytes."""

    model_parameters: float
    gradients: float
    master_parameters: float
    optimizer_moments: float

    @property
    def total_bytes(self) -> float:
        return (
            self.model_parameters
            + self.gradients
            + self.master_parameters
            + self.optimizer_moments
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "model_parameters": self.model_parameters,
            "gradients": self.gradients,
            "master_parameters": self.master_parameters,
            "optimizer_moments": self.optimizer_moments,
            "total_bytes": self.total_bytes,
        }


def heaviest_stage_parameters(
    *,
    parameter_count: int,
    num_layers: int,
    hidden_size: int,
    vocab_size: int,
    parallelism: TrainingParallelism,
    tied_embeddings: bool = False,
) -> float:
    """
    Return the parameter count on the most loaded pipeline stage.

    Megatron splits transformer layers evenly across pipeline stages, but the
    embedding sits on the first stage and the output projection on the last, and
    neither is a transformer layer. A uniform `parameter_count / (tp * pp)` therefore
    understates the stage that carries one of them, and peak memory is set by the
    worst stage. Measured at PP=4 on Qwen3-8B, the uniform assumption predicted
    15.26 GiB of static state where the heaviest rank held 19.32 GiB. Accounting for
    the embedding brings the prediction to 17.57 GiB.

    Tensor parallelism splits every tensor, including the embedding, so it divides the
    result uniformly.

    `parallelism.pipeline_layout`, when set, carves the layers unevenly instead of
    `num_layers / pp` each: every stage's own total (its share of `layer_parameters`
    plus an embedding copy if it is stage 0 or the last stage) is computed and the
    worst of them returned, so a deliberately rebalanced end stage can lose the
    "heaviest" title to a middle stage carrying more layers, which a single
    uniform-share-plus-embedding number could never represent.
    """
    if parameter_count <= 0 or num_layers <= 0:
        raise ValueError("parameter_count and num_layers must be positive.")
    if parallelism.pp == 1:
        return parameter_count / parallelism.tp
    stage_layers = stage_layer_counts(num_layers, parallelism)
    embedding = hidden_size * vocab_size
    # Untied weights mean two separate vocabulary-sized tensors, one per end stage.
    embedding_copies = 1 if tied_embeddings else 2
    layer_parameters = parameter_count - embedding * embedding_copies
    if layer_parameters <= 0:
        raise ValueError("Embeddings cannot exceed the total parameter count.")
    per_layer = layer_parameters / num_layers
    stage_totals = [per_layer * count for count in stage_layers]
    stage_totals[0] += embedding
    stage_totals[-1] += embedding
    return max(stage_totals) / parallelism.tp


def parameter_inventory(
    *,
    parameter_count: int,
    parallelism: TrainingParallelism,
    parameter_dtype: str,
    gradient_dtype: str,
    optimizer_state_dtype: str,
    distributed_optimizer: bool,
    local_parameters: float | None = None,
) -> ParameterInventory:
    """
    Compute per-rank static state bytes.

    Model parameters and gradients are held at full model-parallel-local size on
    every rank. The distributed optimizer shards state, not the gradient buffer:
    gradients are reduce-scattered *from* a full-size buffer, so dividing them by
    the data-parallel degree understates the true footprint roughly twofold.

    `local_parameters` overrides the uniform `parameter_count / (tp * pp)` share. Pass
    `heaviest_stage_parameters` for a pipelined topology, where the stage holding the
    embedding or output projection exceeds a uniform split.
    """
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive.")
    model_parallel = parallelism.tp * parallelism.pp
    if local_parameters is None:
        local_parameters = parameter_count / model_parallel
    # Megatron reduces gradients over the combined DP and CP groups, and the distributed
    # optimizer shards over that same group, so context parallel ranks split one state copy.
    optimizer_shard = parallelism.dp * parallelism.cp if distributed_optimizer else 1
    parameter_bytes = dtype_bytes(parameter_dtype)
    gradient_bytes = dtype_bytes(gradient_dtype)
    optimizer_bytes = dtype_bytes(optimizer_state_dtype)
    # A separate fp32 master copy exists only when parameters are kept narrower.
    master_bytes = optimizer_bytes if parameter_bytes < optimizer_bytes else 0
    return ParameterInventory(
        model_parameters=local_parameters * parameter_bytes,
        gradients=local_parameters * gradient_bytes,
        master_parameters=local_parameters * master_bytes / optimizer_shard,
        # Adam keeps two moments. They allocate lazily on the first optimizer step,
        # so a snapshot taken before that step reports less than steady state.
        optimizer_moments=local_parameters * 2 * optimizer_bytes / optimizer_shard,
    )
