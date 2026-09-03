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
FALLBACK_FLOORS_S: dict[str, dict[str, float]] = {
    "vllm": {
        "context_attention": 32e-6,
        "generation_attention": 22e-6,
        "elementwise": 2e-6,
        "all_reduce": 2e-6,
    },
    "flashinfer": {
        "context_attention": 4e-6,
        "generation_attention": 6e-6,
    },
    # The collective family is measured twice: `vllm` times the dispatcher a server actually
    # uses, and `nccl` times `torch.distributed.all_reduce` as a control. Both need a floor,
    # because a launch cost dominates entirely at the message sizes a decode step produces.
    "nccl": {
        "all_reduce": 2e-6,
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
