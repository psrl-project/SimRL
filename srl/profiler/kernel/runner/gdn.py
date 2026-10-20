"""Time gated-delta-net (linear attention) kernels, both phases.

This is the layer type a hybrid-attention model uses for most of its depth (30 of
Qwen3.5-35B-A3B's 40 layers), and it is priced by different kernels from softmax attention,
so it is a separate family rather than a slice of the `attention` one.

Two kernels, matching what vLLM dispatches in `qwen_gdn_linear_attn.py`:

- prefill: `fi_chunk_gated_delta_rule`, which is vLLM's wrapper over FlashInfer's
  `gdn_prefill.chunk_gated_delta_rule`, varlen over a packed batch behind `cu_seqlens` and
  linear in the token total.

  This is the kernel `_resolve_gdn_prefill_backend` selects on this hardware for an `'auto'`
  request. Timing `fla.ops.gated_delta_rule` instead measures a kernel the engine never runs,
  the same failure mode as reading the FlashInfer attention table for a server that
  dispatches FA3.
- decode: `fused_recurrent_gated_delta_rule_packed_decode` from vLLM's own ops, one token per
  request against a fixed-size recurrent state.

The decode kernel takes a *fused* `mixed_qkv` of width `2 * key_dim + value_dim` rather than
separate q/k/v, plus the `A_log` and `dt_bias` parameters the gating is derived from, so its
call shape is dictated by the layer rather than by this module's own choices. Those are read
from the architecture, never guessed: `_require` raises instead of defaulting, because a wrong
head count here would write a table nothing can query.

## Why this family times CUDA graph replay

The only family that does. FlashInfer dispatches both GDN kernels through the CUTLASS DSL,
which re-marshals every tensor argument in Python on each call, and that host work costs
several times the device work at small shapes. Timing back-to-back calls with CUDA events
measures the interpreter, not the GPU, rather than the kernel the engine actually dispatches.

A live 35B server resolves `cudagraph_mode=FULL_AND_PIECEWISE` and captures the whole decode
forward pass, GDN included, so decode pays none of the marshaling. Capture was checked for
validity, not assumed: replayed output matches eager output for both phases.

Prefill is the weaker case and is a deliberate approximation. `vllm::qwen_gdn_attention_core`
is in `splitting_ops`, so a piecewise-captured prefill batch leaves this op outside the graph
and does pay some host cost. It is timed by replay anyway, because the alternative
over-predicts far more and prefill is a small fraction of engine time against decode's.

GEMM, elementwise and attention stay on `time_cuda_call`. See `utils/timing.py` for the
measurements behind that split and for why attention is not converted despite showing the
same pattern.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from ..grid.gdn import gdn_head_counts

# `fla`'s chunk length for this op. Recorded as a slice column rather than hardcoded into the
# SOL alone, because a future version changing it would change the intra-chunk term and the
# table must not silently mix the two regimes.
FLA_CHUNK_SIZE = 64


def gdn_kernel_source() -> str:
    """Return the kernel-source tag for every row this module writes."""
    import vllm

    return f"vllm_gdn_{vllm.__version__}"


def _time_for(op_name: str) -> Callable[..., float]:
    """
    Return the timing helper the engine's execution mode implies for one op.

    Read from the op rather than hardcoded, because the two GDN phases disagree: decode sits
    inside vLLM's FULL decode graph while prefill is a splitting op run eagerly. Keeping the
    decision in `cost/kernel/ops/gdn.py` puts it next to the `sol()` and floor policy it has
    to stay consistent with.

    Args:
        op_name (str): A registered op name, `gdn_context` or `gdn_decode`.

    Returns:
        Callable[..., float]: Either `time_cuda_graph_replay` or `time_cuda_call`.
    """
    from ....cost.kernel.ops.base import op_for
    from ..utils.timing import time_cuda_call, time_cuda_graph_replay

    return time_cuda_graph_replay if op_for(op_name).graph_replayed else time_cuda_call


def time_gdn_context_point(
    *,
    total_tokens: int,
    num_key_heads: int,
    num_value_heads: int,
    head_dim: int,
    config: Any,
) -> float:
    """
    Time one chunked GDN prefill call over a packed batch.

    Args:
        total_tokens (int): Tokens in the pack, summed over sequences.
        num_key_heads (int): Key heads on this rank.
        num_value_heads (int): Value heads on this rank.
        head_dim (int): Per-head dimension.
        config (Any): A `KernelProfileConfig` carrying dtypes and iteration counts.

    Returns:
        float: Mean per-call latency in seconds.
    """
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        fi_chunk_gated_delta_rule,
    )

    from ..utils.timing import torch_dtype

    dtype = torch_dtype(config.measurement.dtype)
    device = torch.device("cuda")

    # Varlen layout: one packed row behind `cu_seqlens`, which is why there is no batch axis.
    # The leading singleton is the batch dimension the kernel still expects.
    query = torch.randn(1, total_tokens, num_key_heads, head_dim, dtype=dtype, device=device)
    key = torch.randn(1, total_tokens, num_key_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn(1, total_tokens, num_value_heads, head_dim, dtype=dtype, device=device)
    # The decay gate is consumed in log space and must be negative, so a log of U(0,1) is
    # both the right sign and the right magnitude. fp32 because the recurrence accumulates it.
    gate = torch.rand(1, total_tokens, num_value_heads, dtype=torch.float32, device=device).log()
    beta = torch.rand(1, total_tokens, num_value_heads, dtype=dtype, device=device)
    cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.long, device=device)

    # fp32 recurrent state, shaped as vLLM allocates it and indexed by VALUE heads.
    state = torch.zeros(
        1, num_value_heads, head_dim, head_dim, dtype=torch.float32, device=device
    )

    def _call() -> None:
        fi_chunk_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=gate,
            beta=beta,
            initial_state=state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )

    return _time_for("gdn_context")(
        _call,
        warmup_iterations=config.measurement.warmup_iterations,
        active_iterations=config.measurement.active_iterations,
    )


def time_gdn_decode_point(
    *,
    batch_size: int,
    num_key_heads: int,
    num_value_heads: int,
    head_dim: int,
    config: Any,
) -> float:
    """
    Time one recurrent GDN decode step over a batch of requests.

    There is no history-depth argument, which is the structural point: the state is a fixed
    `num_value_heads x head_dim x head_dim` per request however many tokens came before.

    Args:
        batch_size (int): Requests decoding concurrently. One token each.
        num_key_heads (int): Key heads on this rank.
        num_value_heads (int): Value heads on this rank.
        head_dim (int): Per-head dimension.
        config (Any): A `KernelProfileConfig` carrying dtypes and iteration counts.

    Returns:
        float: Mean per-call latency in seconds.
    """
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    from ..utils.timing import torch_dtype

    dtype = torch_dtype(config.measurement.dtype)
    device = torch.device("cuda")

    # The kernel takes q, k and v already concatenated, exactly as the layer's conv output
    # arrives: `2 * key_dim + value_dim` wide, with q and k at key-head width.
    key_dim = num_key_heads * head_dim
    value_dim = num_value_heads * head_dim
    mixed_qkv = torch.randn(batch_size, 2 * key_dim + value_dim, dtype=dtype, device=device)
    a = torch.randn(batch_size, num_value_heads, dtype=dtype, device=device)
    b = torch.randn(batch_size, num_value_heads, dtype=dtype, device=device)
    # Gating parameters, per value head. `A_log` is negative in the real layer, but the magnitude
    # does not change the kernel's work, only its numerics.
    a_log = torch.rand(num_value_heads, dtype=torch.float32, device=device).log()
    dt_bias = torch.randn(num_value_heads, dtype=torch.float32, device=device)
    # The recurrent state is indexed by VALUE heads, not key heads:
    # `MambaStateShapeCalculator.gated_delta_net_state_shape` returns
    # `(num_v_heads, head_v_dim, head_k_dim)`, and the kernel reads `HV` off
    # `initial_state.shape[-3]` to validate `a` and `b` against it. Using key heads here
    # raises `a/b must have shape [B, HV] with HV=16`. fp32 because the recurrence
    # accumulates, which is also how vLLM allocates it.
    state = torch.zeros(
        batch_size, num_value_heads, head_dim, head_dim, dtype=torch.float32, device=device
    )
    out = torch.zeros(batch_size, 1, num_value_heads, head_dim, dtype=dtype, device=device)
    state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)

    def _call() -> None:
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=a_log,
            dt_bias=dt_bias,
            scale=head_dim**-0.5,
            initial_state=state,
            out=out,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )

    return _time_for("gdn_decode")(
        _call,
        warmup_iterations=config.measurement.warmup_iterations,
        active_iterations=config.measurement.active_iterations,
    )


def profile_gdn(
    model: Any,
    config: Any,
    *,
    writer_factory: Callable[[str, str, str], Any],
) -> tuple[Path, ...]:
    """
    Measure both gated-delta-net phases and write one table each.

    Args:
        model (Any): A `ModelDescriptor` read from the checkpoint.
        config (Any): A `KernelProfileConfig` carrying the `gdn:` grid and dtypes.
        writer_factory (Callable[[str, str, str], PerfTableWriter]): Builds a writer from
            `(framework, version, kernel_source)`.

    Returns:
        tuple[Path, ...]: The written context and decode table paths.
    """
    import vllm

    tp: int = config.tensor_parallel
    key_heads, value_heads, head_dim = gdn_head_counts(model, tensor_parallel=tp)
    sweep = config.sweep("gdn_context")
    writer = writer_factory("vllm", vllm.__version__, gdn_kernel_source())

    shared = dict(
        num_key_heads=key_heads,
        num_value_heads=value_heads,
        head_dim=head_dim,
        attn_dtype=config.measurement.dtype,
        chunk_size=FLA_CHUNK_SIZE,
    )

    context_rows: list[dict[str, object]] = []
    for total_tokens in sweep.total_tokens:
        latency_s = time_gdn_context_point(
            total_tokens=total_tokens,
            num_key_heads=key_heads,
            num_value_heads=value_heads,
            head_dim=head_dim,
            config=config,
        )
        context_rows.append({**shared, "total_tokens": int(total_tokens), "latency_s": latency_s})

    decode_rows: list[dict[str, object]] = []
    for batch_size in sweep.batch_sizes:
        latency_s = time_gdn_decode_point(
            batch_size=batch_size,
            num_key_heads=key_heads,
            num_value_heads=value_heads,
            head_dim=head_dim,
            config=config,
        )
        decode_rows.append({**shared, "batch_size": int(batch_size), "latency_s": latency_s})

    return (
        writer.write("gdn_context", context_rows),
        writer.write("gdn_decode", decode_rows),
    )
