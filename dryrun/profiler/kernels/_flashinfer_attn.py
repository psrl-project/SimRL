"""Profile attention through FlashInfer's paged wrappers (`BatchPrefillWithPagedKVCacheWrapper`
and `BatchDecodeWithPagedKVCacheWrapper`), using the KV layout vLLM would allocate.

**Critical measurement decision — plan() is called OUTSIDE the timed region.**

FlashInfer's `plan()` performs host-side scheduling (indptr construction, tile splitting,
work-item generation) that vLLM amortizes once per decode step across all attention layers
in the model.  If `plan()` were inside the timed region, every FlashInfer sample would
include a fixed host cost that vLLM distributes across N layers; the per-layer comparison
would then overstate FlashInfer's latency relative to vLLM by up to N×.  Only `.run(...)`
belongs in the timed region.

**KV cache layout alignment.**

The KV cache tensor is allocated in FlashInfer's NHD layout
(`[num_blocks, 2, page_size, num_kv_heads, head_dim]`), which matches the NHD layout
`get_kv_cache_layout()` returns for this machine.  Because both backends use the same
logical NHD layout, the two row sets differ by kernel behaviour and not by a memory-layout
transpose.  The `get_kv_cache_layout()` call reaches `get_current_vllm_config()`, so it
must be placed inside a `set_current_vllm_config(...)` context (same vLLM-config as
Task 11).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import torch


def flashinfer_kernel_source() -> str:
    """Return the kernel-source tag for all rows written by this module."""
    import flashinfer

    return f"flashinfer_{flashinfer.__version__}"


def time_flashinfer_context_point(
    point: object,
    *,
    config: object,
    workspace_buffer: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    kv_layout: str,
) -> float:
    """
    Time one prefill shape through `BatchPrefillWithPagedKVCacheWrapper`.

    Args:
        point: An `AttentionPoint` with `batch_size`, `seq_len`, `step`.
        config: A `KernelProfileConfig` carrying measurement iteration counts and dtypes.
        workspace_buffer (torch.Tensor): 256-MiB uint8 buffer reused across points.
        num_heads (int): Query head count on this rank.
        num_kv_heads (int): KV head count on this rank.
        head_dim (int): Per-head dimension.
        kv_layout (str): `NHD` or `HND`, sourced from `get_kv_cache_layout()`.

    Returns:
        float: Mean per-call latency in seconds.
    """
    import flashinfer

    from ._timing import time_cuda_call, torch_dtype

    batch_size: int = point.batch_size
    seq_len: int = point.seq_len
    block_size: int = config.sweep("context_attention").block_size
    attn_dtype_str: str = config.measurement.dtype
    kv_cache_dtype_str: str = config.measurement.kv_cache_dtype
    warmup: int = config.measurement.warmup_iterations
    active: int = config.measurement.active_iterations

    dtype = torch_dtype(attn_dtype_str)
    device = torch.device("cuda")

    blocks_per_req = math.ceil(seq_len / block_size)
    # One extra block per request as a margin guard; block 0 is the null/padding block.
    num_blocks = blocks_per_req * batch_size + batch_size + 1

    # Allocate KV cache in FlashInfer NHD layout: [num_blocks, 2, page_size, num_kv_heads, head_dim].
    # This is NHD as returned by get_kv_cache_layout(), so both backends are layout-comparable.
    kv_cache = torch.zeros(num_blocks, 2, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)

    num_tokens = batch_size * seq_len

    # qo_indptr: cumulative token counts per request.
    qo_indptr = torch.tensor(
        [seq_len * r for r in range(batch_size + 1)],
        dtype=torch.int32,
        device=device,
    )

    # paged_kv_indptr: cumulative page counts per request.
    paged_kv_indptr = torch.tensor(
        [blocks_per_req * r for r in range(batch_size + 1)],
        dtype=torch.int32,
        device=device,
    )

    # paged_kv_indices: flat array of physical block indices (1-based, block 0 is null).
    paged_kv_indices = torch.tensor(
        [1 + r * blocks_per_req + i for r in range(batch_size) for i in range(blocks_per_req)],
        dtype=torch.int32,
        device=device,
    )

    # Last-page token count: every request fills to exactly seq_len tokens.
    last_page_fill = seq_len % block_size or block_size
    paged_kv_last_page_len = torch.full(
        (batch_size,), last_page_fill, dtype=torch.int32, device=device
    )

    # Pre-populate the KV cache OUTSIDE the timed region.
    k_all = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_all = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    batch_indices = torch.repeat_interleave(
        torch.arange(batch_size, device=device), seq_len
    )
    positions = torch.cat([torch.arange(seq_len, device=device) for _ in range(batch_size)])
    flashinfer.append_paged_kv_cache(
        k_all, v_all,
        batch_indices, positions,
        kv_cache, paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len,
        kv_layout=kv_layout,
    )
    torch.cuda.synchronize()

    query = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=device)
    output = torch.zeros(num_tokens, num_heads, head_dim, dtype=dtype, device=device)

    # plan() is called OUTSIDE the timed region — see module docstring.
    prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, kv_layout=kv_layout)
    prefill_wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        num_qo_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=block_size,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype if kv_cache_dtype_str == "auto" else torch_dtype(kv_cache_dtype_str),
        sm_scale=head_dim**-0.5,
    )

    def _timed_call(
        _wrapper=prefill_wrapper,
        _q=query,
        _kv=kv_cache,
        _out=output,
    ) -> None:
        _wrapper.run(_q, _kv, out=_out)

    latency_s = time_cuda_call(
        _timed_call,
        warmup_iterations=warmup,
        active_iterations=active,
    )

    del kv_cache, query, k_all, v_all, output
    torch.cuda.empty_cache()
    return latency_s


def time_flashinfer_generation_point(
    point: object,
    *,
    config: object,
    workspace_buffer: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    kv_layout: str,
) -> float:
    """
    Time one decode shape through `BatchDecodeWithPagedKVCacheWrapper`.

    Args:
        point: An `AttentionPoint` with `batch_size`, `seq_len`, `step`.
        config: A `KernelProfileConfig` carrying measurement iteration counts and dtypes.
        workspace_buffer (torch.Tensor): 256-MiB uint8 buffer reused across points.
        num_heads (int): Query head count on this rank.
        num_kv_heads (int): KV head count on this rank.
        head_dim (int): Per-head dimension.
        kv_layout (str): `NHD` or `HND`, sourced from `get_kv_cache_layout()`.

    Returns:
        float: Mean per-call latency in seconds.
    """
    import flashinfer

    from ._timing import time_cuda_call, torch_dtype

    batch_size: int = point.batch_size
    step: int = point.step
    block_size: int = config.sweep("generation_attention").block_size
    attn_dtype_str: str = config.measurement.dtype
    kv_cache_dtype_str: str = config.measurement.kv_cache_dtype
    warmup: int = config.measurement.warmup_iterations
    active: int = config.measurement.active_iterations

    dtype = torch_dtype(attn_dtype_str)
    device = torch.device("cuda")

    # KV history is `step + 1` after this decode step.  Use max(step, 1) to avoid
    # zero-length allocations when step == 0 in a generation sweep point.
    total_kv_per_req = max(step + 1, 1)
    blocks_per_req = math.ceil(total_kv_per_req / block_size)
    # One extra block per request as a margin guard; block 0 is the null/padding block.
    num_blocks = blocks_per_req * batch_size + batch_size + 1

    # Allocate KV cache in FlashInfer NHD layout: [num_blocks, 2, page_size, num_kv_heads, head_dim].
    kv_cache = torch.zeros(num_blocks, 2, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)

    # paged_kv_indptr: cumulative page counts per request.
    paged_kv_indptr = torch.tensor(
        [blocks_per_req * r for r in range(batch_size + 1)],
        dtype=torch.int32,
        device=device,
    )

    # paged_kv_indices: flat array of physical block indices (1-based, block 0 is null).
    paged_kv_indices = torch.tensor(
        [1 + r * blocks_per_req + i for r in range(batch_size) for i in range(blocks_per_req)],
        dtype=torch.int32,
        device=device,
    )

    # Last-page token count: total_kv_per_req tokens per request.
    last_page_fill = total_kv_per_req % block_size or block_size
    paged_kv_last_page_len = torch.full(
        (batch_size,), last_page_fill, dtype=torch.int32, device=device
    )

    # Pre-populate KV history OUTSIDE the timed region.
    if step > 0:
        _prepopulate_generation_kv(
            kv_cache=kv_cache,
            paged_kv_indices=paged_kv_indices,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_last_page_len=paged_kv_last_page_len,
            batch_size=batch_size,
            step=step,
            blocks_per_req=blocks_per_req,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            kv_layout=kv_layout,
        )

    # query: [batch_size, num_heads, head_dim] for decode.
    query = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)
    output = torch.zeros(batch_size, num_heads, head_dim, dtype=dtype, device=device)

    # plan() is called OUTSIDE the timed region — see module docstring.
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout=kv_layout)
    decode_wrapper.plan(
        indptr=paged_kv_indptr,
        indices=paged_kv_indices,
        last_page_len=paged_kv_last_page_len,
        num_qo_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=block_size,
        q_data_type=dtype,
        kv_data_type=dtype if kv_cache_dtype_str == "auto" else torch_dtype(kv_cache_dtype_str),
        sm_scale=head_dim**-0.5,
    )

    def _timed_call(
        _wrapper=decode_wrapper,
        _q=query,
        _kv=kv_cache,
        _out=output,
    ) -> None:
        _wrapper.run(_q, _kv, out=_out)

    latency_s = time_cuda_call(
        _timed_call,
        warmup_iterations=warmup,
        active_iterations=active,
    )

    del kv_cache, query, output
    torch.cuda.empty_cache()
    return latency_s


def _prepopulate_generation_kv(
    *,
    kv_cache: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    batch_size: int,
    step: int,
    blocks_per_req: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    kv_layout: str,
) -> None:
    """Write `step` historical KV tokens into the FlashInfer paged cache for each request."""
    import flashinfer

    # Write in chunks to avoid large transient allocations.
    chunk_size = min(step, 256)
    written = 0
    while written < step:
        this_chunk = min(chunk_size, step - written)
        chunk_tokens = batch_size * this_chunk
        k_chunk = torch.randn(chunk_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
        v_chunk = torch.randn(chunk_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, device=device), this_chunk
        )
        positions = torch.cat(
            [torch.arange(written, written + this_chunk, device=device) for _ in range(batch_size)]
        )
        flashinfer.append_paged_kv_cache(
            k_chunk, v_chunk,
            batch_indices, positions,
            kv_cache, paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len,
            kv_layout=kv_layout,
        )
        written += this_chunk
    torch.cuda.synchronize()


def profile_flashinfer_attention(
    model: object,
    config: object,
    *,
    writer_factory: Callable[[str, str, str], object],
) -> tuple[Path, Path]:
    """
    Measure both attention phases through FlashInfer's paged wrappers.

    Args:
        model: A `ModelDescriptor` read from the checkpoint.
        config: A `KernelProfileConfig` carrying sweep grids, dtypes, and iteration counts.
        writer_factory (Callable[[str, str, str], PerfTableWriter]): Builds a writer from
            `(framework, version, kernel_source)`.

    Returns:
        tuple[Path, Path]: The written context and generation table paths.
    """
    import flashinfer
    from vllm.config import CacheConfig, ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.utils import get_kv_cache_layout

    from .attention import (
        AttentionPoint,
        attention_head_dim,
        context_sweep_points,
        generation_sweep_points,
        rank_head_counts,
    )

    tp: int = config.tensor_parallel
    num_heads, num_kv_heads = rank_head_counts(model, tensor_parallel=tp)
    head_dim: int = attention_head_dim(model)
    attn_dtype_str: str = config.measurement.dtype
    kv_cache_dtype_str: str = config.measurement.kv_cache_dtype
    # Both attention ops parse the same `attention:` subtree, so the block and window sizes are
    # one value; they are read off the context op because the KV cache layout precedes both phases.
    context_sweep = config.sweep("context_attention")
    generation_sweep = config.sweep("generation_attention")
    block_size: int = context_sweep.block_size
    window_size: int = context_sweep.window_size
    model_path: str = model.name

    kernel_source = flashinfer_kernel_source()
    writer = writer_factory("flashinfer", flashinfer.__version__, kernel_source)

    all_context_points = context_sweep_points(context_sweep)
    all_gen_points = generation_sweep_points(
        generation_sweep, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    max_kv_len = max(
        (pt.seq_len for pt in all_context_points),
        default=1,
    )
    max_step = max(
        (pt.step for pt in all_gen_points),
        default=0,
    )
    max_model_len = max(max_kv_len, max_step) + 1

    # Build VllmConfig so get_kv_cache_layout() can reach get_current_vllm_config().
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=model_path,
            tokenizer=model_path,
            trust_remote_code=False,
            dtype=attn_dtype_str,
            seed=0,
            max_model_len=max_model_len,
        ),
        cache_config=CacheConfig(block_size=block_size, cache_dtype=kv_cache_dtype_str),
    )

    with set_current_vllm_config(vllm_config):
        kv_layout = get_kv_cache_layout()

    # Allocate the workspace buffer once and reuse across all points.
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    def _point_kwargs(point: AttentionPoint) -> dict:
        return dict(
            point=point,
            config=config,
            workspace_buffer=workspace_buffer,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_layout=kv_layout,
        )

    # The two phases interpolate over different third axes -- `isl` for context, `step` for
    # generation -- so each row carries only the axis its own op declares.
    shared = dict(
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        attn_dtype=attn_dtype_str,
        kv_cache_dtype=kv_cache_dtype_str,
        block_size=block_size,
        window_size=window_size,
    )

    # --- Context phase ---
    context_rows: list[dict[str, object]] = []
    for point in all_context_points:
        latency_s = time_flashinfer_context_point(**_point_kwargs(point))
        context_rows.append(
            {**shared, "batch_size": point.batch_size, "isl": point.seq_len, "latency_s": latency_s}
        )

    # --- Generation phase ---
    generation_rows: list[dict[str, object]] = []
    for point in all_gen_points:
        latency_s = time_flashinfer_generation_point(**_point_kwargs(point))
        generation_rows.append(
            {**shared, "batch_size": point.batch_size, "step": point.step, "latency_s": latency_s}
        )

    context_path = writer.write("context_attention", context_rows)
    generation_path = writer.write("generation_attention", generation_rows)
    return context_path, generation_path
