"""
Per-rank GEMM shapes of a transformer block, sharded by tensor parallelism.

Separate from `runner/gemm.py` because these are pure integer arithmetic over a
`ModelDescriptor`, while the runner also holds the CUDA timing loop and so imports `torch`.
The cost model composes a step from these shapes and must not pay a 4.2-second GPU-stack
import to do it.

Shapes are sharded at construction time, so the perf table itself is TP-agnostic: it stores
`(m, n, k)` and nothing else. Forward, dgrad, and wgrad all land in that one table, which is
what lets one surface price both a training and an inference step. They are nonetheless three
separate enumerations rather than one: per-`(n, k)` slicing makes a dgrad's swapped `(k, n)`
pair a *different slice* from its forward, so "backward reuses forward for free" stopped being
true and three of four Qwen3 projections would otherwise query a slice no sweep wrote.

Which shapes a layer runs is a property of the architecture, so this module no longer decides
it: `transformer_gemm_shapes` and `logits_gemm_shape` dispatch to a `ModelFamily` chosen by
`ModelDescriptor.architecture`. See `families.py` for why an unregistered architecture is
refused rather than priced with the dense arithmetic. The sweep helpers below are family-agnostic
-- they take whatever shapes the family returns and cross them with token counts.
"""

from __future__ import annotations

from ...schema import ModelDescriptor
from ..sweep.gemm import GemmSweep
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


def linear_attention_gemm_shapes(
    model: ModelDescriptor,
    *,
    tensor_parallel: int,
) -> tuple[GemmShape, ...]:
    """
    List one gated-delta-net layer's per-rank projections, empty for a pure-softmax model.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.
        tensor_parallel (int): Ranks the block is sharded across.

    Returns:
        tuple[GemmShape, ...]: The linear layer's projections, or an empty tuple when the
            model has no linear-attention layers.

    Raises:
        ProfileValidationError: If no family is registered for the architecture, or if the
            family cannot price the model.
    """
    family = family_for(model)
    getter = getattr(family, "linear_attention_gemm_shapes", None)
    if getter is None:
        return ()
    shapes = getter(model, tensor_parallel=tensor_parallel)
    return () if shapes is None else tuple(shapes)


def per_layer_gemm_shapes(
    model: ModelDescriptor,
    *,
    tensor_parallel: int,
) -> tuple[GemmShape, ...]:
    """
    List every distinct per-layer projection shape in the model's block.

    A hybrid model runs two blocks, so a sweep that enumerated only `transformer_gemm_shapes`
    would leave the linear layer's projections unmeasured and every query for them would
    raise `KernelCoverageError`.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.
        tensor_parallel (int): Ranks the block is sharded across.

    Returns:
        tuple[GemmShape, ...]: Softmax-layer shapes followed by linear-layer shapes.
    """
    return transformer_gemm_shapes(model, tensor_parallel=tensor_parallel) + (
        linear_attention_gemm_shapes(model, tensor_parallel=tensor_parallel)
    )


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
    shapes = per_layer_gemm_shapes(model, tensor_parallel=tensor_parallel)
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


def gemm_dgrad_shapes(
    model: ModelDescriptor,
    sweep: GemmSweep,
    *,
    tensor_parallel: int,
) -> tuple[tuple[int, int, int], ...]:
    """
    Enumerate the dgrad GEMM shapes for every forward projection and token count.

    For a forward linear `activation[m, k] @ weight^T[k, n] -> output[m, n]`, the
    input-gradient GEMM is `dY[m, n] @ W[n, k] -> dX[m, k]`, i.e. the shape
    `(m_dgrad, n_dgrad, k_dgrad) = (m, k_fwd, n_fwd)`: the forward's `n` and `k` swap.

    This module's docstring used to claim backward needed no extra measurement, and that
    was true while `gemm` sliced on `(gemm_dtype,)` alone, because a dgrad's `(m, n, k)`
    was then just another point on the same surface. Per-`(n, k)` slicing ended it: the
    swapped pair `(k_fwd, n_fwd)` is a *different slice*, and for Qwen3-8B at tp=2 three of
    the four projections land in a slice no sweep ever wrote. Measured: `qkv_proj`,
    `gate_up_proj` and `down_proj` dgrads all raise `KernelCoverageError`, and only
    `o_proj` resolves, because its `n` and `k` happen to be measured in both orders.

    Leaving them to clamp onto a neighbouring slice would price three of four backward
    GEMMs per layer from a different projection's curve, which is the exact defect
    per-`(n, k)` slicing was introduced to fix (it took GEMM holdout from 8.48% to 4.08%).

    Only the per-layer projections are included. The logits projection has no dgrad in the
    measured set, because training's loss path produces it through a different fused
    kernel than a plain linear backward.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.
        sweep (GemmSweep): Token counts to measure at.
        tensor_parallel (int): Ranks the block is sharded across.

    Returns:
        tuple[tuple[int, int, int], ...]: Deduplicated `(m, n, k)` triples, sorted.
    """
    shapes = per_layer_gemm_shapes(model, tensor_parallel=tensor_parallel)
    triples = {
        (token_count, shape.k, shape.n)
        for token_count in sweep.token_counts
        for shape in shapes
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

    The table's row `(m, n, k)` means `F.linear(A[m, k], W[n, k]) -> C[m, n]`, which is what
    `gemm.py:69-72` actually times. A forward projection is therefore `(tokens, n_f, k_f)`.

    The weight gradient has the same shape as the weight, `[n_f, k_f]`, and is formed as
    `dW[n_f, k_f] = dY^T[n_f, tokens] @ X[tokens, k_f]`. Expressed in the table's
    convention, the output rows are `n_f`, the output columns are `k_f`, and the contracted
    length is the token count, so the row is `(n_f, k_f, tokens)`.

    NOTE(claude): The swapped triple `(n_f, tokens, k_f)` is a different slice whenever
    `tokens != k_f`, with an identical FLOP count since the product is symmetric, so the
    bug surfaces as a lookup into the wrong `(n, k)` slice rather than an implausible cost.

    Only the per-layer projections are included. The logits projection is inference-
    only and has no wgrad.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.
        sweep (GemmSweep): Token counts to measure at.
        tensor_parallel (int): Ranks the block is sharded across.

    Returns:
        tuple[tuple[int, int, int], ...]: Deduplicated `(m, n, k)` triples, sorted.
    """
    shapes = per_layer_gemm_shapes(model, tensor_parallel=tensor_parallel)
    triples = {
        (shape.n, shape.k, token_count)
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
    Return the union of forward, dgrad, wgrad, and logits GEMM shapes, deduplicated.

    All four families go into one table, which is what lets a training step and an
    inference step read the same surface. Dgrad is included because per-`(n, k)` slicing
    made the swapped pair a distinct slice. See `gemm_dgrad_shapes`.

    Args:
        model (ModelDescriptor): Architecture read from the checkpoint.
        sweep (GemmSweep): Token counts and logits request counts to measure at.
        tensor_parallel (int): Ranks the block is sharded across.

    Returns:
        tuple[tuple[int, int, int], ...]: Deduplicated `(m, n, k)` triples, sorted.
    """
    forward = set(gemm_sweep_shapes(model, sweep, tensor_parallel=tensor_parallel))
    dgrad = set(gemm_dgrad_shapes(model, sweep, tensor_parallel=tensor_parallel))
    wgrad = set(gemm_wgrad_shapes(model, sweep, tensor_parallel=tensor_parallel))
    logits = set(logits_sweep_shapes(model, sweep, tensor_parallel=tensor_parallel))
    return tuple(sorted(forward | dgrad | wgrad | logits))
