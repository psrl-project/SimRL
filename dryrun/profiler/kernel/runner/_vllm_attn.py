"""Profile attention through the backend vLLM's own selector picks, against a real paged KV cache.

This module is the only place in the profiler that imports vLLM internals.  It is kept
isolated here so that every vLLM API bump only requires touching this file.

Call sequence (established by direct experiment on 8x NVIDIA H20, vLLM 0.26.1.dev0):

1. Everything must run inside `set_current_vllm_config(vllm_config)`.
2. `slot_mapping` must be int64; `block_table_tensor` stays int32.
3. The mock attention layer needs the full scale set: `_q_scale`, `_k_scale`, `_v_scale`,
   `_prob_scale` (tensors) and `_q_scale_float`, `_k_scale_float`, `_v_scale_float` (floats).
4. Q/K/V are flat `[num_tokens, num_heads, head_size]`; output is `[num_tokens, num_heads, head_size]`.
5. For FA3 (`forward_includes_kv_cache_update = False`), the timed region is
   `do_kv_cache_update` then `forward`.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class SelectedBackend:
    """vLLM attention backend resolved for one set of head/dtype/block_size parameters."""

    name: str
    kernel_source: str
    backend_cls: type
    builder_cls: type
    impl_cls: type
    forward_includes_kv_cache_update: bool


def select_attention_backend(
    *,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    attn_dtype: str,
    kv_cache_dtype: str,
    block_size: int,
    model_path: str,
) -> SelectedBackend:
    """
    Resolve the vLLM attention backend for one set of model parameters.

    Args:
        head_dim (int): Per-head dimension.
        num_heads (int): Query head count.
        num_kv_heads (int): Key/value head count.
        attn_dtype (str): Compute dtype string, e.g. `bfloat16`.
        kv_cache_dtype (str): KV cache element dtype string.
        block_size (int): Paged KV cache block size in tokens.
        model_path (str): Path to a HuggingFace checkpoint; used only for `ModelConfig`.

    Returns:
        SelectedBackend: Resolved backend class, builder, impl, and source tag.
    """
    from vllm.config import CacheConfig, ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.platforms import current_platform
    from vllm.utils.import_utils import resolve_obj_by_qualname
    from vllm.v1.attention.selector import AttentionSelectorConfig

    dtype = _torch_dtype(attn_dtype)
    model_config = ModelConfig(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=False,
        dtype=attn_dtype,
        seed=0,
        max_model_len=65536,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(block_size=block_size, cache_dtype=kv_cache_dtype),
    )

    with set_current_vllm_config(vllm_config):
        selector = AttentionSelectorConfig(
            head_size=head_dim,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            block_size=block_size,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            use_mm_prefix=False,
        )
        backend_cls = resolve_obj_by_qualname(
            current_platform.get_attn_backend_cls(None, selector, num_heads=num_heads)
        )
        name = backend_cls.get_name()

        # Instantiate the impl to read the FA version attribute.
        impl = backend_cls.get_impl_cls()(
            num_heads=num_heads,
            head_size=head_dim,
            scale=head_dim**-0.5,
            num_kv_heads=num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=kv_cache_dtype,
        )
        if name == "FLASH_ATTN":
            kernel_source = f"vllm_flash_attn_fa{impl.vllm_flash_attn_version}"
        else:
            kernel_source = f"vllm_{name.lower()}"

    return SelectedBackend(
        name=name,
        kernel_source=kernel_source,
        backend_cls=backend_cls,
        builder_cls=backend_cls.get_builder_cls(),
        impl_cls=backend_cls.get_impl_cls(),
        forward_includes_kv_cache_update=backend_cls.forward_includes_kv_cache_update,
    )


def time_attention_point(
    point: object,
    *,
    phase: str,
    selected: SelectedBackend,
    config: object,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    vllm_config: object,
) -> float:
    """
    Time one attention shape through the selected vLLM backend.

    Args:
        point: An `AttentionPoint` with `batch_size`, `seq_len`, `step`.
        phase (str): Either `context` or `generation`.
        selected (SelectedBackend): Resolved backend from `select_attention_backend`.
        config: A `KernelProfileConfig` carrying measurement iteration counts and dtypes.
        num_heads (int): Query head count on this rank.
        num_kv_heads (int): KV head count on this rank.
        head_dim (int): Per-head dimension.
        vllm_config: An active `VllmConfig` from `select_attention_backend`'s context.

    Returns:
        float: Mean per-call latency in seconds.
    """
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.utils import set_kv_cache_layout
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    from ._timing import time_cuda_call, torch_dtype

    batch_size: int = point.batch_size
    seq_len: int = point.seq_len
    step: int = point.step
    block_size: int = config.sweep(f"{phase}_attention").block_size
    attn_dtype_str: str = config.measurement.dtype
    kv_cache_dtype_str: str = config.measurement.kv_cache_dtype
    warmup: int = config.measurement.warmup_iterations
    active: int = config.measurement.active_iterations

    dtype = torch_dtype(attn_dtype_str)
    device = torch.device("cuda")

    # Context phase: each request has `seq_len` tokens; generation phase has 1 query token.
    if phase == "context":
        query_len_per_req = seq_len
        total_kv_per_req = seq_len
    else:
        query_len_per_req = 1
        # KV history is `step + 1` after this decode step.
        total_kv_per_req = step + 1

    num_tokens = batch_size * query_len_per_req

    # Allocate the paged KV cache.  Use max(step, 1) to avoid zero-length allocations when
    # step == 0 in a generation sweep point.
    kv_per_req_for_alloc = max(total_kv_per_req, 1)
    blocks_per_req = math.ceil(kv_per_req_for_alloc / block_size)
    # One extra block per request as a margin guard.
    num_blocks = blocks_per_req * batch_size + batch_size

    with set_current_vllm_config(vllm_config):
        spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            dtype=dtype,
        )
        kv_shape = selected.backend_cls.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_dim
        )
        kv_cache = torch.zeros(kv_shape, dtype=dtype, device=device)

        # Apply the stride permutation required by the backend.
        stride_order = selected.backend_cls.get_kv_cache_stride_order()
        if tuple(stride_order) != tuple(range(len(kv_shape))):
            inverse = [list(stride_order).index(i) for i in range(len(kv_shape))]
            kv_cache = kv_cache.permute(*stride_order).contiguous().permute(*inverse)

        # Restore the global KV cache layout override on exit.
        stack = contextlib.ExitStack()
        required_layout = selected.backend_cls.get_required_kv_cache_layout()
        stack.callback(set_kv_cache_layout, None)
        set_kv_cache_layout(required_layout)

        with stack:
            # Assign each request a contiguous block range starting from block 1.
            # Block 0 is conventionally reserved as the null/padding block.
            block_table = torch.tensor(
                [
                    [1 + r * blocks_per_req + i for i in range(blocks_per_req)]
                    for r in range(batch_size)
                ],
                dtype=torch.int32,
                device=device,
            )

            # Build slot_mapping: the flat KV slot index of every query token.
            # slot_mapping MUST be int64 (vLLM fact 2).
            if phase == "context":
                slot_mapping = torch.cat(
                    [
                        block_table[r, 0].to(torch.int64) * block_size
                        + torch.arange(seq_len, dtype=torch.int64, device=device)
                        for r in range(batch_size)
                    ]
                )
            else:
                # One new KV slot at position `step` within the KV history.
                # block_table[r, step // block_size] = 1 + r * blocks_per_req + step // block_size,
                # so the slot simplifies to (block_size + step) + r * blocks_per_req * block_size.
                r_idx = torch.arange(batch_size, dtype=torch.int64, device=device)
                slot_mapping = (block_size + step) + r_idx * (blocks_per_req * block_size)

            if phase == "context":
                query_start_loc_vals = [0] + [seq_len * (r + 1) for r in range(batch_size)]
                seq_lens_vals = [seq_len] * batch_size
                max_query_len = seq_len
                max_seq_len = seq_len
                num_computed_tokens = [0] * batch_size
            else:
                query_start_loc_vals = [0] + [r + 1 for r in range(batch_size)]
                seq_lens_vals = [step + 1] * batch_size
                max_query_len = 1
                max_seq_len = step + 1
                num_computed_tokens = [step] * batch_size

            query_start_loc_cpu = torch.tensor(query_start_loc_vals, dtype=torch.int32)
            seq_lens_cpu = torch.tensor(seq_lens_vals, dtype=torch.int32)
            num_computed_cpu = torch.tensor(num_computed_tokens, dtype=torch.int32)

            common = CommonAttentionMetadata(
                query_start_loc=query_start_loc_cpu.to(device),
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens_cpu.to(device),
                num_reqs=batch_size,
                num_actual_tokens=num_tokens,
                max_query_len=max_query_len,
                max_seq_len=max_seq_len,
                block_table_tensor=block_table,
                slot_mapping=slot_mapping,
                _seq_lens_cpu=seq_lens_cpu,
                _num_computed_tokens_cpu=num_computed_cpu,
            )

            builder = selected.builder_cls(spec, ["layer.0"], vllm_config, device)
            attn_metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)

            impl = selected.impl_cls(
                num_heads=num_heads,
                head_size=head_dim,
                scale=head_dim**-0.5,
                num_kv_heads=num_kv_heads,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype=kv_cache_dtype_str,
            )

            layer = _MockAttentionLayer(device)

            query = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=device)
            key = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
            value = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
            output = torch.zeros(num_tokens, num_heads, head_dim, dtype=dtype, device=device)

            # Pre-populate KV history OUTSIDE the timed region for generation phase.
            if phase == "generation" and step > 0:
                _prepopulate_kv_history(
                    impl=impl,
                    layer=layer,
                    kv_cache=kv_cache,
                    block_table=block_table,
                    batch_size=batch_size,
                    step=step,
                    block_size=block_size,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    dtype=dtype,
                    device=device,
                    vllm_config=vllm_config,
                )

            forward_includes_update = selected.forward_includes_kv_cache_update

            def _timed_call(
                _impl=impl,
                _layer=layer,
                _q=query,
                _k=key,
                _v=value,
                _kv=kv_cache,
                _meta=attn_metadata,
                _out=output,
                _slot=slot_mapping,
                _fwd_includes=forward_includes_update,
            ) -> None:
                if not _fwd_includes:
                    _impl.do_kv_cache_update(_layer, _k, _v, _kv, _slot)
                _impl.forward(_layer, _q, _k, _v, _kv, _meta, output=_out)

            latency_s = time_cuda_call(
                _timed_call,
                warmup_iterations=warmup,
                active_iterations=active,
            )

    del kv_cache, query, key, value, output
    torch.cuda.empty_cache()
    return latency_s


def _prepopulate_kv_history(
    *,
    impl: object,
    layer: object,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    batch_size: int,
    step: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    vllm_config: object,
) -> None:
    """Write `step` historical KV tokens into the paged cache for each request in the batch."""
    from vllm.config import set_current_vllm_config

    # Derive blocks_per_req from the same arithmetic used to build `block_table` so that
    # block_table[r, token_pos // block_size] = 1 + r * blocks_per_req + token_pos // block_size.
    # The slot for token `token_pos` of request `r` then simplifies to:
    #   slot = (1 + r*B + token_pos//bs) * bs + token_pos%bs = bs + r*B*bs + token_pos
    # where B = blocks_per_req.  This lets us build the mapping with one arange broadcast
    # and a single host-to-device transfer per chunk, with zero .item() calls.
    blocks_per_req: int = block_table.shape[1]
    req_offsets = torch.arange(batch_size, dtype=torch.int64, device=device) * (blocks_per_req * block_size)

    chunk_size = min(step, 256)
    written = 0
    while written < step:
        this_chunk = min(chunk_size, step - written)
        chunk_tokens = batch_size * this_chunk
        k_hist = torch.randn(chunk_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
        v_hist = torch.randn(chunk_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
        # Build slot_mapping entirely on-device: shape [batch_size, this_chunk], then flatten.
        token_offsets = torch.arange(this_chunk, dtype=torch.int64, device=device) + written
        slot_mapping_hist = (block_size + req_offsets.unsqueeze(1) + token_offsets.unsqueeze(0)).reshape(-1)

        with set_current_vllm_config(vllm_config):
            impl.do_kv_cache_update(layer, k_hist, v_hist, kv_cache, slot_mapping_hist)

        written += this_chunk
    torch.cuda.synchronize()


class _MockAttentionLayer(torch.nn.Module):
    """Stand-in for a real attention layer, carrying only the scale attributes `impl` reads."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        for name in ("_q_scale", "_k_scale", "_v_scale", "_prob_scale"):
            setattr(self, name, torch.tensor(1.0, device=device))
        for name in ("_q_scale_float", "_k_scale_float", "_v_scale_float"):
            setattr(self, name, 1.0)


def profile_vllm_attention(
    model: object,
    config: object,
    *,
    writer_factory: Callable[[str, str, str], object],
) -> tuple[Path, Path]:
    """
    Measure both attention phases through the backend vLLM selects.

    Args:
        model: A `ModelDescriptor` read from the checkpoint.
        config: A `KernelProfileConfig` carrying sweep grids, dtypes, and iteration counts.
        writer_factory (Callable[[str, str, str], PerfTableWriter]): Builds a writer from
            `(framework, version, kernel_source)`. The kernel source is only known after
            backend selection, so the writer cannot be passed in directly.

    Returns:
        tuple[Path, Path]: The written context and generation table paths.
    """
    import vllm
    from vllm.config import CacheConfig, ModelConfig, VllmConfig

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
    # one value; they are read off the context op because backend selection precedes both phases.
    context_sweep = config.sweep("context_attention")
    generation_sweep = config.sweep("generation_attention")
    block_size: int = context_sweep.block_size
    window_size: int = context_sweep.window_size
    model_path: str = model.name

    selected = select_attention_backend(
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        attn_dtype=attn_dtype_str,
        kv_cache_dtype=kv_cache_dtype_str,
        block_size=block_size,
        model_path=model_path,
    )

    vllm_version = vllm.__version__
    writer = writer_factory("vllm", vllm_version, selected.kernel_source)

    # Build the VllmConfig that all timing calls will use.
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

    def _point_kwargs(point: AttentionPoint, phase: str) -> dict:
        return dict(
            point=point,
            phase=phase,
            selected=selected,
            config=config,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            vllm_config=vllm_config,
        )

    # The two phases interpolate over different third axes: `isl` for context, `step` for
    # generation, so each row carries only the axis its own op declares.
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
        latency_s = time_attention_point(**_point_kwargs(point, "context"))
        context_rows.append(
            {**shared, "batch_size": point.batch_size, "isl": point.seq_len, "latency_s": latency_s}
        )

    # --- Generation phase ---
    generation_rows: list[dict[str, object]] = []
    for point in all_gen_points:
        latency_s = time_attention_point(**_point_kwargs(point, "generation"))
        generation_rows.append(
            {**shared, "batch_size": point.batch_size, "step": point.step, "latency_s": latency_s}
        )

    context_path = writer.write("context_attention", context_rows)
    generation_path = writer.write("generation_attention", generation_rows)
    return context_path, generation_path


def _torch_dtype(name: str) -> torch.dtype:
    """Resolve a dtype string to a `torch.dtype`."""
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}.") from exc
