"""
Per-rank GEMM shapes of a transformer block, sharded by tensor parallelism.

Separate from `gemm.py` because these are pure integer arithmetic over a `ModelDescriptor`,
while `gemm.py` also holds the CUDA timing loop and so imports `torch`. The cost model
composes a step from these shapes and must not pay a 4.2-second GPU-stack import to do it.

Shapes are sharded at construction time, so the perf table itself is TP-agnostic: it stores
`(m, n, k)` and nothing else. That is what lets a later phase price backward dgrad and wgrad
from the same rows by permuting the axes, with no extra measurement.

Which shapes a layer runs is a property of the architecture, so this module no longer decides
it: `transformer_gemm_shapes` and `logits_gemm_shape` dispatch to a `ModelFamily` chosen by
`ModelDescriptor.architecture`. See `families.py` for why an unregistered architecture is
refused rather than priced with the dense arithmetic. The sweep helpers below are family-agnostic
-- they take whatever shapes the family returns and cross them with token counts.
"""

from __future__ import annotations

from ..schema import ModelDescriptor
from .config import GemmSweep
from .families import family_for
from .shape import GemmShape


def transformer_gemm_shapes(
    model: ModelDescriptor,
    *,
    tensor_parallel: int,
) -> tuple[GemmShape, ...]:
    """
    List one layer's per-rank projection shapes, as the model's family declares them.

    Does not include the logits projection, which runs once per step rather than once per layer
    and over a different m dimension. Use `logits_gemm_shape` for that.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.
        tensor_parallel (int): Ranks the block is sharded across.

    Returns:
        tuple[GemmShape, ...]: One shape per projection, already sharded.

    Raises:
        ProfileValidationError: If no family is registered for the architecture, if the family
            cannot price this model, if the architecture is incomplete, or if a dimension is
            not divisible by `tensor_parallel`.
    """
    return tuple(family_for(model).gemm_shapes(model, tensor_parallel=tensor_parallel))


def logits_gemm_shape(
    model: ModelDescriptor,
    *,
    tensor_parallel: int,
) -> GemmShape:
    """
    Return the per-rank logits projection shape, as the model's family declares it.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.
        tensor_parallel (int): Ranks the head is sharded across.

    Returns:
        GemmShape: One per-rank logits projection shape.

    Raises:
        ProfileValidationError: If no family is registered for the architecture, if the family
            cannot price this model, or if vocab_size is not divisible by `tensor_parallel`.
    """
    return family_for(model).logits_shape(model, tensor_parallel=tensor_parallel)


def gemm_sweep_shapes(
    model: ModelDescriptor,
    sweep: GemmSweep,
    *,
    tensor_parallel: int,
) -> tuple[tuple[int, int, int], ...]:
    """
    Cross the token-count sweep with every per-rank per-layer shape.

    Does not include the logits projection: that shape uses a different m
    sweep (`sweep.logits_request_counts`) and is enumerated by `logits_sweep_shapes`.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.
        sweep (GemmSweep): Token counts to measure at.
        tensor_parallel (int): Ranks the block is sharded across.

    Returns:
        tuple[tuple[int, int, int], ...]: Deduplicated `(m, n, k)` triples, sorted.
    """
    shapes = transformer_gemm_shapes(model, tensor_parallel=tensor_parallel)
    triples = {
        (token_count, shape.n, shape.k)
        for token_count in sweep.token_counts
        for shape in shapes
    }
    return tuple(sorted(triples))


def logits_sweep_shapes(
    model: ModelDescriptor,
    sweep: GemmSweep,
    *,
    tensor_parallel: int,
) -> tuple[tuple[int, int, int], ...]:
    """
    Cross the logits-request-count sweep with the per-rank LM-head shape.

    The logits projection runs once per step over the request count (one sampled
    token per request in decode, one per sequence in prefill), not over the full
    token count. Its sweep uses `sweep.logits_request_counts` to avoid measuring
    `m` values that this kernel can never reach in practice.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.
        sweep (GemmSweep): Provides `logits_request_counts` to measure at.
        tensor_parallel (int): Ranks the block is sharded across.

    Returns:
        tuple[tuple[int, int, int], ...]: Deduplicated `(m, n, k)` triples, sorted.
    """
    shape = logits_gemm_shape(model, tensor_parallel=tensor_parallel)
    triples = {
        (request_count, shape.n, shape.k)
        for request_count in sweep.logits_request_counts
    }
    return tuple(sorted(triples))


def gemm_wgrad_shapes(
    model: ModelDescriptor,
    sweep: GemmSweep,
    *,
    tensor_parallel: int,
) -> tuple[tuple[int, int, int], ...]:
    """
    Enumerate the wgrad GEMM shapes for every forward projection and token count.

    For a forward linear `activation[m, k] @ weight^T[k, n] -> output[m, n]`, the
    weight-gradient GEMM is `dY^T[n, m] @ X[m, k] -> dW[n, k]`, i.e. the shape
    `(m_wgrad, n_wgrad, k_wgrad) = (n_fwd, m, k_fwd)`.  After per-(n,k) slicing
    the wgrad's n_wgrad equals the token count, which is not one of the measured
    forward (n, k) pairs, so these shapes must be measured explicitly.

    Only the per-layer projections are included. The logits projection is inference-
    only and has no wgrad.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.
        sweep (GemmSweep): Token counts to measure at.
        tensor_parallel (int): Ranks the block is sharded across.

    Returns:
        tuple[tuple[int, int, int], ...]: Deduplicated `(m, n, k)` triples, sorted.
    """
    shapes = transformer_gemm_shapes(model, tensor_parallel=tensor_parallel)
    triples = {
        (shape.n, token_count, shape.k)
        for token_count in sweep.token_counts
        for shape in shapes
    }
    return tuple(sorted(triples))


def gemm_all_shapes(
    model: ModelDescriptor,
    sweep: GemmSweep,
    *,
    tensor_parallel: int,
) -> tuple[tuple[int, int, int], ...]:
    """
    Return the union of forward, wgrad, and logits GEMM shapes, deduplicated.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.
        sweep (GemmSweep): Token counts and logits request counts to measure at.
        tensor_parallel (int): Ranks the block is sharded across.

    Returns:
        tuple[tuple[int, int, int], ...]: Deduplicated `(m, n, k)` triples, sorted.
    """
    forward = set(gemm_sweep_shapes(model, sweep, tensor_parallel=tensor_parallel))
    wgrad = set(gemm_wgrad_shapes(model, sweep, tensor_parallel=tensor_parallel))
    logits = set(logits_sweep_shapes(model, sweep, tensor_parallel=tensor_parallel))
    return tuple(sorted(forward | wgrad | logits))
