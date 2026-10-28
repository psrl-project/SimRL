"""Does KV-block fragmentation (scattered, non-contiguous block assignment, as a long real run
with many admit/evict cycles produces) make generation attention slower than the kernel table's
clean, freshly-allocated, contiguous-block profiling methodology would predict?

Reuses `srl.profiler.kernel.runner._vllm_attn.select_attention_backend` for backend resolution
(identical to how the shipped table was built), then times generation attention at a fixed
(batch_size, step) shape under two block-table layouts:
  - contiguous: exactly what `_vllm_attn.time_attention_point` builds (the profiling baseline).
  - scattered: each request's blocks interleaved across a much larger pool, simulating a KV pool
    that has cycled through many admit/evict cycles.

Also checks for a sustained-execution (thermal/allocator) drift: replays ONE captured graph
continuously (no gaps -- a short gapped loop never keeps the GPU busy long enough to reach
thermal steady state) for several minutes and tracks latency against measured GPU temperature.

**Result (2026-10-26, see `memory/lpt-admission-loses-on-real-vllm-under-kv-pressure.md` for the
full writeup this was run to settle): both hypotheses ruled out.** Scattered vs contiguous block
layout: <0.4% difference at every depth tested (2000-16000), noise-level. Sustained replay over
7+ minutes: GPU temperature rises normally (36C -> 59C, plateaus) but per-call latency stays flat
throughout (no monotonic trend, ~1% peak-to-peak noise) -- no thermal throttling on this hardware
at this load. Neither explains the ~5-6.5% real-vs-predicted divergence found at deep context
(~9000-11000 tokens) in the real-vs-sim comparison. Raw results from the run that produced the
numbers above are in `outputs/vllm_vs_sim/kv_fragmentation_probe.json`.

Usage:
    CUDA_VISIBLE_DEVICES=0 python examples/analysis/vllm_vs_sim/probe_kv_fragmentation.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

MODEL_PATH = "/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B"


def _build_common(batch_size, step, block_table, device, dtype):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    r_idx = torch.arange(batch_size, dtype=torch.int64, device=device)
    query_start_loc_vals = [0] + [r + 1 for r in range(batch_size)]
    seq_lens_vals = [step + 1] * batch_size
    num_computed_tokens = [step] * batch_size
    query_start_loc_cpu = torch.tensor(query_start_loc_vals, dtype=torch.int32)
    seq_lens_cpu = torch.tensor(seq_lens_vals, dtype=torch.int32)
    num_computed_cpu = torch.tensor(num_computed_tokens, dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc_cpu.to(device),
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens_cpu.to(device),
        num_reqs=batch_size,
        num_actual_tokens=batch_size,
        max_query_len=1,
        max_seq_len=step + 1,
        block_table_tensor=block_table,
        slot_mapping=None,  # filled by caller
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_cpu,
    )


def measure(batch_size: int, step: int, block_size: int, layout: str, warmup: int, active: int,
            num_heads: int, num_kv_heads: int, head_dim: int, dtype, selected, vllm_config) -> float:
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backends.utils import set_kv_cache_layout
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    import contextlib

    device = torch.device("cuda")
    blocks_per_req = math.ceil((step + 1) / block_size)

    if layout == "contiguous":
        num_blocks = blocks_per_req * batch_size + batch_size
        block_table = torch.tensor(
            [[1 + r * blocks_per_req + i for i in range(blocks_per_req)] for r in range(batch_size)],
            dtype=torch.int32, device=device,
        )
    elif layout == "scattered":
        # A much bigger pool (10x), each request's blocks drawn from a shuffled permutation so
        # consecutive blocks for one request land far apart in physical block-id space -- what a
        # KV pool that has cycled through many admit/evict/preempt rounds looks like, since the
        # free list hands out whatever was returned most recently, not a contiguous run.
        pool_size = blocks_per_req * batch_size * 10 + batch_size
        perm = torch.randperm(pool_size - 1, device=device) + 1  # avoid block 0 (null/padding)
        needed = blocks_per_req * batch_size
        block_table = perm[:needed].reshape(batch_size, blocks_per_req).to(torch.int32)
        num_blocks = pool_size
    else:
        raise ValueError(layout)

    with set_current_vllm_config(vllm_config):
        spec = FullAttentionSpec(block_size=block_size, num_kv_heads=num_kv_heads, head_size=head_dim, dtype=dtype)
        kv_shape = selected.backend_cls.get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_dim)
        kv_cache = torch.zeros(kv_shape, dtype=dtype, device=device)
        stride_order = selected.backend_cls.get_kv_cache_stride_order()
        if tuple(stride_order) != tuple(range(len(kv_shape))):
            inverse = [list(stride_order).index(i) for i in range(len(kv_shape))]
            kv_cache = kv_cache.permute(*stride_order).contiguous().permute(*inverse)

        stack = contextlib.ExitStack()
        required_layout = selected.backend_cls.get_required_kv_cache_layout()
        stack.callback(set_kv_cache_layout, None)
        set_kv_cache_layout(required_layout)

        with stack:
            common = _build_common(batch_size, step, block_table, device, dtype)
            # slot for the NEW token: block_table[r, step // block_size] * block_size + step % block_size
            blk_idx = step // block_size
            blk_off = step % block_size
            slot_mapping = block_table[:, blk_idx].to(torch.int64) * block_size + blk_off
            common.slot_mapping = slot_mapping

            builder = selected.builder_cls(spec, ["layer.0"], vllm_config, device)
            attn_metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)

            impl = selected.impl_cls(
                num_heads=num_heads, head_size=head_dim, scale=head_dim**-0.5,
                num_kv_heads=num_kv_heads, alibi_slopes=None, sliding_window=None,
                kv_cache_dtype="bfloat16",
            )

            class _Layer(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    for n in ("_q_scale", "_k_scale", "_v_scale", "_prob_scale"):
                        setattr(self, n, torch.tensor(1.0, device=device))
                    for n in ("_q_scale_float", "_k_scale_float", "_v_scale_float"):
                        setattr(self, n, 1.0)

            layer = _Layer()
            query = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)
            key = torch.randn(batch_size, num_kv_heads, head_dim, dtype=dtype, device=device)
            value = torch.randn(batch_size, num_kv_heads, head_dim, dtype=dtype, device=device)
            output = torch.zeros(batch_size, num_heads, head_dim, dtype=dtype, device=device)

            # Pre-populate KV history so the read path touches real (if random) data at every block.
            chunk = min(step, 256)
            written = 0
            while written < step:
                this_chunk = min(chunk, step - written)
                k_hist = torch.randn(batch_size * this_chunk, num_kv_heads, head_dim, dtype=dtype, device=device)
                v_hist = torch.randn(batch_size * this_chunk, num_kv_heads, head_dim, dtype=dtype, device=device)
                offs = torch.arange(this_chunk, device=device) + written
                blk = (offs.unsqueeze(0) // block_size)  # [1, chunk] broadcast per request below
                slots = []
                for r in range(batch_size):
                    bt = block_table[r]
                    s = bt[offs // block_size].to(torch.int64) * block_size + offs % block_size
                    slots.append(s)
                slot_hist = torch.cat(slots)
                with set_current_vllm_config(vllm_config):
                    impl.do_kv_cache_update(layer, k_hist, v_hist, kv_cache, slot_hist)
                written += this_chunk
            torch.cuda.synchronize()

            forward_includes_update = selected.forward_includes_kv_cache_update

            def _call():
                if not forward_includes_update:
                    impl.do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
                impl.forward(layer, query, key, value, kv_cache, attn_metadata, output=output)

            # Graph-replay timing, matching the shipped table's methodology for generation attention.
            g = torch.cuda.CUDAGraph()
            for _ in range(3):
                _call()
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                _call()
            for _ in range(warmup):
                g.replay()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(active):
                g.replay()
            end.record()
            torch.cuda.synchronize()
            latency_s = start.elapsed_time(end) / 1000.0 / active

    del kv_cache, query, key, value, output
    torch.cuda.empty_cache()
    return latency_s


def sustained_drift(batch_size: int, step: int, block_size: int, checkpoint_every: int,
                     n_checkpoints: int, num_heads: int, num_kv_heads: int, head_dim: int,
                     dtype, selected, vllm_config) -> list[dict]:
    """
    Replay one captured graph continuously (no gaps) for `checkpoint_every * n_checkpoints`
    iterations, timing each checkpoint block with CUDA events and sampling GPU temperature.

    A short, gapped loop (build graph, time a handful of replays, repeat) never keeps the GPU
    busy long enough to reach thermal steady state -- this keeps one graph alive and replays it
    back-to-back so the device sees the same sustained load a real 30+ minute decode run does.
    """
    import subprocess
    import contextlib
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.utils import set_kv_cache_layout
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    device = torch.device("cuda")
    blocks_per_req = math.ceil((step + 1) / block_size)
    num_blocks = blocks_per_req * batch_size + batch_size
    block_table = torch.tensor(
        [[1 + r * blocks_per_req + i for i in range(blocks_per_req)] for r in range(batch_size)],
        dtype=torch.int32, device=device,
    )

    with set_current_vllm_config(vllm_config):
        spec = FullAttentionSpec(block_size=block_size, num_kv_heads=num_kv_heads, head_size=head_dim, dtype=dtype)
        kv_shape = selected.backend_cls.get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_dim)
        kv_cache = torch.zeros(kv_shape, dtype=dtype, device=device)
        stride_order = selected.backend_cls.get_kv_cache_stride_order()
        if tuple(stride_order) != tuple(range(len(kv_shape))):
            inverse = [list(stride_order).index(i) for i in range(len(kv_shape))]
            kv_cache = kv_cache.permute(*stride_order).contiguous().permute(*inverse)
        stack = contextlib.ExitStack()
        required_layout = selected.backend_cls.get_required_kv_cache_layout()
        stack.callback(set_kv_cache_layout, None)
        set_kv_cache_layout(required_layout)
        with stack:
            query_start_loc_vals = [0] + [r + 1 for r in range(batch_size)]
            seq_lens_vals = [step + 1] * batch_size
            num_computed_tokens = [step] * batch_size
            query_start_loc_cpu = torch.tensor(query_start_loc_vals, dtype=torch.int32)
            seq_lens_cpu = torch.tensor(seq_lens_vals, dtype=torch.int32)
            num_computed_cpu = torch.tensor(num_computed_tokens, dtype=torch.int32)
            blk_idx, blk_off = step // block_size, step % block_size
            slot_mapping = block_table[:, blk_idx].to(torch.int64) * block_size + blk_off
            common = CommonAttentionMetadata(
                query_start_loc=query_start_loc_cpu.to(device), query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens_cpu.to(device), num_reqs=batch_size, num_actual_tokens=batch_size,
                max_query_len=1, max_seq_len=step + 1, block_table_tensor=block_table,
                slot_mapping=slot_mapping, _seq_lens_cpu=seq_lens_cpu, _num_computed_tokens_cpu=num_computed_cpu,
            )
            builder = selected.builder_cls(spec, ["layer.0"], vllm_config, device)
            attn_metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)
            impl = selected.impl_cls(num_heads=num_heads, head_size=head_dim, scale=head_dim**-0.5,
                                      num_kv_heads=num_kv_heads, alibi_slopes=None, sliding_window=None,
                                      kv_cache_dtype="bfloat16")

            class _Layer(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    for n in ("_q_scale", "_k_scale", "_v_scale", "_prob_scale"):
                        setattr(self, n, torch.tensor(1.0, device=device))
                    for n in ("_q_scale_float", "_k_scale_float", "_v_scale_float"):
                        setattr(self, n, 1.0)

            layer = _Layer()
            query = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)
            key = torch.randn(batch_size, num_kv_heads, head_dim, dtype=dtype, device=device)
            value = torch.randn(batch_size, num_kv_heads, head_dim, dtype=dtype, device=device)
            output = torch.zeros(batch_size, num_heads, head_dim, dtype=dtype, device=device)
            chunk = min(step, 256)
            written = 0
            while written < step:
                this_chunk = min(chunk, step - written)
                k_hist = torch.randn(batch_size * this_chunk, num_kv_heads, head_dim, dtype=dtype, device=device)
                v_hist = torch.randn(batch_size * this_chunk, num_kv_heads, head_dim, dtype=dtype, device=device)
                offs = torch.arange(this_chunk, device=device) + written
                slots = [
                    block_table[r][offs // block_size].to(torch.int64) * block_size + offs % block_size
                    for r in range(batch_size)
                ]
                impl.do_kv_cache_update(layer, k_hist, v_hist, kv_cache, torch.cat(slots))
                written += this_chunk
            torch.cuda.synchronize()
            forward_includes_update = selected.forward_includes_kv_cache_update

            def _call():
                if not forward_includes_update:
                    impl.do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
                impl.forward(layer, query, key, value, kv_cache, attn_metadata, output=output)

            g = torch.cuda.CUDAGraph()
            for _ in range(3):
                _call()
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                _call()

            def _gpu_temp() -> str:
                return subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                    capture_output=True, text=True,
                ).stdout.strip().splitlines()[0]

            rows = []
            t0 = time.perf_counter()
            for cp in range(n_checkpoints):
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()
                for _ in range(checkpoint_every):
                    g.replay()
                end_ev.record()
                torch.cuda.synchronize()
                per_call_us = start_ev.elapsed_time(end_ev) / checkpoint_every * 1000.0
                row = {"checkpoint": cp, "t": time.perf_counter() - t0, "latency_us": per_call_us, "gpu_temp_c": _gpu_temp()}
                rows.append(row)
                print(f"  checkpoint={cp:2d}  t={row['t']:6.1f}s  latency={per_call_us:8.3f}us  gpu_temp={row['gpu_temp_c']}", flush=True)

    del kv_cache, query, key, value, output
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    from srl.profiler.kernel.runner._vllm_attn import select_attention_backend
    from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig

    num_heads, num_kv_heads, head_dim = 32, 8, 128  # Qwen3-8B, tp=1
    block_size = 16
    dtype = torch.bfloat16

    selected = select_attention_backend(
        head_dim=head_dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
        attn_dtype="bfloat16", kv_cache_dtype="bfloat16", block_size=block_size,
        model_path=MODEL_PATH,
    )
    print(f"selected backend: {selected.name} ({selected.kernel_source})", flush=True)

    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=MODEL_PATH, tokenizer=MODEL_PATH, trust_remote_code=False,
            dtype="bfloat16", seed=0, max_model_len=40000,
        ),
        cache_config=CacheConfig(block_size=block_size, cache_dtype="bfloat16"),
        parallel_config=ParallelConfig(tensor_parallel_size=1),
    )

    results = {}

    print("\n=== test 1: contiguous vs scattered block layout, batch=40 ===", flush=True)
    for step in (2000, 5000, 9022, 10927, 16000):
        row = {}
        for layout in ("contiguous", "scattered"):
            lat = measure(40, step, block_size, layout, warmup=20, active=200,
                          num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
                          dtype=dtype, selected=selected, vllm_config=vllm_config)
            row[layout] = lat
            print(f"  step={step:6d} layout={layout:11s} latency={lat*1e6:8.2f}us", flush=True)
        ratio = row["scattered"] / row["contiguous"]
        print(f"  step={step:6d} scattered/contiguous = {ratio:.4f} ({(ratio-1)*100:+.2f}%)", flush=True)
        results[f"frag_step{step}"] = row

    print("\n=== test 2: sustained-execution drift, one graph replayed continuously (no gaps), "
          "contiguous batch=40 step=10000 ===", flush=True)
    results["drift"] = sustained_drift(
        40, 10000, block_size, checkpoint_every=20000, n_checkpoints=20,
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
        dtype=dtype, selected=selected, vllm_config=vllm_config,
    )

    out = ROOT / "outputs" / "vllm_vs_sim" / "kv_fragmentation_probe.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
