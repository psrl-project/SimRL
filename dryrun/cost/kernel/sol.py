"""Analytical speed-of-light latency per kernel family.

Every formula is a roofline: `max(compute_time, memory_time)` from vendor peak
throughput, in seconds. SOL is never a prediction on its own. It is the normalizer
that turns a measured latency into a dimensionless utility
(`SOL / measured`), which is what interpolates and transfers across shapes and
hardware. A utility above ~1.2 means the formula here is missing work the kernel
actually does, not that the GPU beat its spec.

## Attention dispatch floor

Attention calls carry a per-call dispatch cost that the pure roofline does not represent.
Real H20 data (fitted 2026-08-24) shows decode latency flat across a 64x increase in KV
work at small batch sizes: batch 1 measures ~70 us at step 128 and still ~70 us at step
8192. That is conclusive evidence the cost is `max(floor, roofline_work)`, not
`floor + roofline_work`. Adding a constant before deriving utility barely helps; placing the
floor inside the max (as `floor_s`) restores utility to order 1 in the overhead-limited
regime and lifts the holdout gate from >20% MRE to <10%.

The floor is a property of the backend's dispatch path, not of the GPU silicon, so it
cannot live in `GpuSpec`. A module-level mapping keyed by backend and op name is honest: these
numbers are fitted from real measurements (see `FALLBACK_FLOORS_S`), their origin is
visible here alongside the evidence, and the location avoids a schema change to the parquet
provenance. The mapping is the simplest arrangement that makes the measurement traceable.

GEMM does not need a floor: its holdout MRE is 3.91% and the regime is compute- or
bandwidth-limited, not dispatch-limited. `floor_s` defaults to 0.0 so GEMM and every
existing call site are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass

from .hardware import GpuSpec, dtype_bytes


def _require_positive(**values: int) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value!r}.")


# Per-backend, per-op dispatch floor in seconds, used only when a fitted floors.json is absent.
# Fitted for H20 + Qwen3-8B TP=1 (2026-08-24, elementwise and all_reduce 2026-08-26): these are
# the MRE-minimising floors for that hardware and those backend versions, not general truths. Run
# `dryrun-profile kernels fit-floor` to generate floors.json for the actual deployment; it is
# loaded in place of these values.
#
# Every op declaring `fits_floor = True` must appear here under at least one backend, which
# `tests/test_kernel_sol.py` enforces off the registry. An op with a floor policy but no fallback
# degrades to a zero floor when floors.json is absent, and for `all_reduce` that is a 220x SOL
# error at 4096 bytes (2.000 us fitted against 0.0091 us of ring transfer) which the fitted
# utility then partly launders into a plausible-looking MRE.
#
# Keyed by op name rather than by phase. The phase spelling forced four separate translations
# between `generation` and `generation_attention` (in floor fitting, in two holdout sites, and
# in the database), each of which was a place a new floor-taking op had to be threaded through.
# See `dryrun/cost/kernel/floor.py` for the fitting procedure.
# The four collectives added alongside `all_reduce` share its 2 us fallback. They are not
# separately fitted here on purpose: a fallback exists to keep a missing floors.json loud
# rather than silently zero, and inventing per-op constants nobody measured would dress a
# guess up as a measurement. `fit-floor` replaces all of them from the real sweep, and
# `floor.py` fits each op independently because their launch costs are not actually equal.
_COLLECTIVE_FALLBACK_S: dict[str, float] = {
    "all_reduce": 2e-6,
    "reduce_scatter": 2e-6,
    "all_gather": 2e-6,
    "broadcast": 2e-6,
    "p2p": 2e-6,
}

FALLBACK_FLOORS_S: dict[str, dict[str, float]] = {
    "vllm": {
        "context_attention": 32e-6,
        "generation_attention": 22e-6,
        "elementwise": 2e-6,
        # Gated delta net, refitted on H20 against Qwen3.5-35B-A3B (2026-10-08) after the
        # runner moved to CUDA graph replay. The earlier 80 us / 22 us pair was fitted
        # against event-timed rows that were measuring FlashInfer's CUTLASS DSL python
        # marshaling rather than the GPU: 491 us of "kernel" at 128 tokens against 57 us of
        # actual device work. A real engine replays a captured decode graph and pays none of
        # that. See `profiler/kernel/runner/gdn.py` for the evidence and the phase caveat.
        "gdn_context": 16e-6,
        "gdn_decode": 32e-6,
        **_COLLECTIVE_FALLBACK_S,
    },
    "flashinfer": {
        "context_attention": 4e-6,
        "generation_attention": 6e-6,
    },
    # The collective family is measured twice: `vllm` times the dispatcher a server actually
    # uses, and `nccl` times `torch.distributed.all_reduce` as a control. Both need a floor,
    # because a launch cost dominates entirely at the message sizes a decode step produces.
    "nccl": dict(_COLLECTIVE_FALLBACK_S),
    # `megatron` names the framework whose kernels are timed: TE's `FusedAdam` and its
    # `DotProductAttention`, which is what megatron-core calls. vLLM has neither.
    "megatron": {
        # Least consequential floor in the tree: a real shard is 10^8-10^9 parameters and
        # the measured step is bandwidth-bound at 2703 GB/s well before a launch cost shows.
        "optimizer_step": 2e-6,
        # Both directions fit to the candidate grid's **top** value, 80 us, which means the
        # true floor is above the grid rather than at its edge. The measured evidence is
        # unambiguous: at 4 heads the forward takes 200.1, 199.6 and 193.8 us at 512, 1024
        # and 2048 tokens, i.e. flat across 16x more work, and only starts scaling at 3072.
        # So ~195 us is the real dispatch cost of a TE attention call on this machine,
        # roughly 2.4x the fitted value.
        #
        # Extending `candidates_us` past 80 us is the correct fix and is deliberately not
        # done here: the candidate grid is shared by every floor-taking op, and widening it
        # would re-fit the attention, elementwise and collective floors that
        # `tests/test_collective_baseline_regression.py` pins. The consequence is bounded
        # and in the safe direction - a floor fitted too low under-predicts only shapes
        # that are entirely dispatch-bound, and the smallest real training microbatch in
        # the validation set is 4096 tokens, already past the flat region.
        "training_attention_forward": 80e-6,
        "training_attention_backward": 80e-6,
    },
}


@dataclass(frozen=True)
class SolComponents:
    """Compute-bound, bandwidth-bound, and dispatch-floor components of one roofline."""

    math_s: float
    memory_s: float
    floor_s: float = 0.0

    @property
    def latency_s(self) -> float:
        """Return the binding component of the roofline, in seconds."""
        return max(self.math_s, self.memory_s, self.floor_s)


def gemm_sol(spec: GpuSpec, *, m: int, n: int, k: int, dtype: str) -> SolComponents:
    """
    Return the roofline latency of one `(m, k) @ (k, n)` GEMM.

    Args:
        spec (GpuSpec): Target hardware.
        m (int): Token count, the row dimension of the activation.
        n (int): Output feature count.
        k (int): Input feature count.
        dtype (str): Compute dtype, e.g. `bfloat16`.

    Returns:
        SolComponents: Compute and memory halves in seconds.
    """
    _require_positive(m=m, n=n, k=k)
    element_bytes = dtype_bytes(dtype)
    math_s = 2.0 * m * n * k / spec.flops_for(dtype)
    memory_s = element_bytes * (m * k + k * n + m * n) / spec.memory_bandwidth_bytes_per_s
    return SolComponents(math_s=math_s, memory_s=memory_s)


def context_attention_sol(
    spec: GpuSpec,
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    attn_dtype: str,
    kv_cache_dtype: str,
    window_size: int = 0,
    floor_s: float = 0.0,
) -> SolComponents:
    """
    Return the roofline latency of one causal prefill attention call.

    Args:
        spec (GpuSpec): Target hardware.
        batch_size (int): Requests in the batch.
        seq_len (int): Prompt length per request, uniform across the batch.
        num_heads (int): Query heads on this rank.
        num_kv_heads (int): Key/value heads on this rank.
        head_dim (int): Per-head dimension.
        attn_dtype (str): Query, output, and compute dtype.
        kv_cache_dtype (str): KV cache dtype, which may be narrower.
        window_size (int): Sliding window span, or 0 for full causal attention.
            A `window_size` at or above `seq_len` yields a strictly larger estimate
            than 0 does, because the causal triangular saving does not apply within a window.
        floor_s (float): Per-backend dispatch floor in seconds; see `FALLBACK_FLOORS_S`.
            Placed inside the `max`, not added, because dispatch cost is a lower bound on
            latency: when work is small the kernel is entirely dispatch-limited, and
            `floor + work` would double-count it. Defaults to 0.0 (no floor).

    Returns:
        SolComponents: Compute, memory, and dispatch-floor components in seconds.
    """
    _require_positive(
        batch_size=batch_size, seq_len=seq_len, num_heads=num_heads,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
    )
    attended = seq_len * 0.5 if window_size <= 0 else float(min(seq_len, window_size))
    # Two matmuls, QK^T and PV, each 2 * b * h * s * attended * d FLOP.
    math_s = 2.0 * (2.0 * batch_size * num_heads * seq_len * attended * head_dim) / spec.flops_for(
        attn_dtype
    )
    attn_bytes = dtype_bytes(attn_dtype)
    kv_bytes = dtype_bytes(kv_cache_dtype)
    query_and_output = 2 * attn_bytes * batch_size * seq_len * num_heads * head_dim
    kv_written = kv_bytes * 2 * batch_size * seq_len * num_kv_heads * head_dim
    memory_s = (query_and_output + kv_written) / spec.memory_bandwidth_bytes_per_s
    return SolComponents(math_s=math_s, memory_s=memory_s, floor_s=floor_s)


def generation_attention_sol(
    spec: GpuSpec,
    *,
    batch_size: int,
    step: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    attn_dtype: str,
    kv_cache_dtype: str,
    window_size: int = 0,
    floor_s: float = 0.0,
) -> SolComponents:
    """
    Return the roofline latency of one paged decode attention call.

    The memory half includes writing this step's KV entry, because vLLM's
    FlashAttention backend reports `forward_includes_kv_cache_update = False` and the
    profiler therefore times `do_kv_cache_update` alongside `forward`.

    Args:
        spec (GpuSpec): Target hardware.
        batch_size (int): Requests decoding concurrently.
        step (int): KV history length per request before this step; may be 0.
        num_heads (int): Query heads on this rank.
        num_kv_heads (int): Key/value heads on this rank.
        head_dim (int): Per-head dimension.
        attn_dtype (str): Query, output, and compute dtype.
        kv_cache_dtype (str): KV cache dtype, which may be narrower.
        window_size (int): Sliding window span, or 0 for unbounded history.
        floor_s (float): Per-backend dispatch floor in seconds; see `FALLBACK_FLOORS_S`.
            Placed inside the `max`, not added, because dispatch cost is a lower bound on
            latency: when work is small the kernel is entirely dispatch-limited, and
            `floor + work` would double-count it. Defaults to 0.0 (no floor).

    Returns:
        SolComponents: Compute, memory, and dispatch-floor components in seconds.
    """
    _require_positive(
        batch_size=batch_size, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    if step < 0:
        raise ValueError(f"step must not be negative, got {step!r}.")
    attended = step if window_size <= 0 else min(step, window_size)
    math_s = 2.0 * (2.0 * batch_size * num_heads * attended * head_dim) / spec.flops_for(attn_dtype)
    attn_bytes = dtype_bytes(attn_dtype)
    kv_bytes = dtype_bytes(kv_cache_dtype)
    history_read = kv_bytes * 2 * batch_size * attended * num_kv_heads * head_dim
    kv_write = kv_bytes * 2 * batch_size * num_kv_heads * head_dim
    query_and_output = 2 * attn_bytes * batch_size * num_heads * head_dim
    memory_s = (history_read + kv_write + query_and_output) / spec.memory_bandwidth_bytes_per_s
    return SolComponents(math_s=math_s, memory_s=memory_s, floor_s=floor_s)


def gdn_context_sol(
    spec: GpuSpec,
    *,
    total_tokens: int,
    num_key_heads: int,
    num_value_heads: int,
    head_dim: int,
    attn_dtype: str,
    chunk_size: int = 64,
    floor_s: float = 0.0,
) -> SolComponents:
    """
    Return the roofline latency of one chunked gated-delta-net prefill call.

    Linear in `total_tokens`, not quadratic, and that is the whole point of the layer: a
    gated delta net carries a fixed-size recurrent state of `num_value_heads * head_dim^2`
    instead of a growing KV cache, so a token attends to the state rather than to every
    earlier token. Pricing it with `context_attention_sol` would overstate a long prompt by
    roughly `seq_len / chunk_size` -- 128x at 32k tokens.

    The chunked algorithm (`fla.ops.gated_delta_rule.chunk_gated_delta_rule`) splits the
    sequence into chunks of `chunk_size` and, per chunk, does intra-chunk attention plus a
    state update, so the work has two parts:

    - intra-chunk, quadratic *within* a chunk and therefore linear overall:
      `total_tokens * chunk * head_dim` per value head.
    - state propagation, linear in tokens: each token touches the `head_dim x head_dim`
      state once, giving `total_tokens * head_dim^2` per state block. The state is indexed
      by VALUE heads (`gated_delta_net_state_shape` is `(num_v_heads, head_v_dim,
      head_k_dim)`), so the block count is `num_value_heads`.

    Measured on H20 (Qwen3.5-35B-A3B, 16 key / 32 value heads, head_dim 128, 2026-10-08) by
    CUDA graph replay, the cost scales from 57.8 us at 128 tokens to 3106 us at 32768, clean
    doublings once past ~2048. An earlier version of this docstring reported the kernel as
    flat at ~368 us to 4096 tokens; that plateau was FlashInfer's CUTLASS DSL python
    marshaling, which an engine replaying a captured graph never pays, and it is gone from
    the measurement now. The fitted floor is correspondingly small, 16 us.

    Args:
        spec (GpuSpec): Target hardware.
        total_tokens (int): Tokens in the packed batch, summed over sequences.
        num_key_heads (int): Key/query heads on this rank.
        num_value_heads (int): Value heads on this rank, which also index the state.
        head_dim (int): Per-head dimension, shared by key and value here.
        attn_dtype (str): Input and output dtype.
        chunk_size (int): The kernel's chunk length. 64 is `fla`'s default for this op.
        floor_s (float): Fitted dispatch floor in seconds, applied inside the `max`.

    Returns:
        SolComponents: Compute, memory, and dispatch-floor components in seconds.
    """
    _require_positive(
        total_tokens=total_tokens,
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        head_dim=head_dim,
        chunk_size=chunk_size,
    )
    attended = float(min(chunk_size, total_tokens))
    intra = 2.0 * (2.0 * num_value_heads * total_tokens * attended * head_dim)
    state = 2.0 * (2.0 * num_value_heads * total_tokens * head_dim * head_dim)
    math_s = (intra + state) / spec.flops_for(attn_dtype)

    nbytes = dtype_bytes(attn_dtype)
    qk = 2 * nbytes * total_tokens * num_key_heads * head_dim
    vo = 2 * nbytes * total_tokens * num_value_heads * head_dim
    gates = (4 + nbytes) * total_tokens * num_value_heads
    state_bytes = 2 * nbytes * num_value_heads * head_dim * head_dim
    memory_s = (qk + vo + gates + state_bytes) / spec.memory_bandwidth_bytes_per_s
    return SolComponents(math_s=math_s, memory_s=memory_s, floor_s=floor_s)


def gdn_decode_sol(
    spec: GpuSpec,
    *,
    batch_size: int,
    num_key_heads: int,
    num_value_heads: int,
    head_dim: int,
    attn_dtype: str,
    floor_s: float = 0.0,
) -> SolComponents:
    """
    Return the roofline latency of one recurrent gated-delta-net decode step.

    There is no `step` axis, which is the structural difference from
    `generation_attention_sol` and the reason this op has one fewer dimension: the recurrent
    state is a fixed `num_value_heads x head_dim x head_dim` per request regardless of how
    many tokens preceded it, so a decode step costs the same at KV depth 128 as at 32768. A
    cost model that interpolated GDN decode over history depth would be fitting noise.

    That also makes the step overwhelmingly memory-bound: it reads and writes the whole state
    to produce one token, so `state_bytes / bandwidth` dominates unless the batch is large.

    Measured on H20 (2026-10-08) by CUDA graph replay, the step scales from 3.9 us at batch 1
    to 348 us at batch 256, roughly linear throughout (12.8 us at 8, 50.0 us at 32, 91.4 us
    at 64). The flat ~25 us plateau an earlier version of this docstring reported below batch
    16 was python dispatch, not device work, and is absent once the marshaling is excluded.

    Args:
        spec (GpuSpec): Target hardware.
        batch_size (int): Requests decoding concurrently.
        num_key_heads (int): Key heads on this rank.
        num_value_heads (int): Value heads on this rank, which index the state.
        head_dim (int): Per-head dimension.
        attn_dtype (str): Input and output dtype.
        floor_s (float): Fitted dispatch floor in seconds, applied inside the `max`.

    Returns:
        SolComponents: Compute, memory, and dispatch-floor components in seconds.
    """
    _require_positive(
        batch_size=batch_size,
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        head_dim=head_dim,
    )
    # One token against a head_dim x head_dim state: an outer-product update and a readout,
    # two matmuls per state block, and the state is indexed by value heads.
    math_s = (
        2.0 * (2.0 * batch_size * num_value_heads * head_dim * head_dim)
    ) / spec.flops_for(attn_dtype)
    nbytes = dtype_bytes(attn_dtype)
    # The state is read and written in full. fp32 because the recurrence accumulates.
    state_bytes = 2 * 4 * batch_size * num_value_heads * head_dim * head_dim
    qk = 2 * nbytes * batch_size * num_key_heads * head_dim
    vo = 2 * nbytes * batch_size * num_value_heads * head_dim
    memory_s = (state_bytes + qk + vo) / spec.memory_bandwidth_bytes_per_s
    return SolComponents(math_s=math_s, memory_s=memory_s, floor_s=floor_s)
