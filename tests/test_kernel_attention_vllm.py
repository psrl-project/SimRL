"""Attention profiling through the backend vLLM's own selector picks."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from dryrun.cost.kernel.ops.attention import (
    CONTEXT_ATTENTION_OP,
    GENERATION_ATTENTION_OP,
)
from dryrun.cost.kernel.table import KernelTable
from dryrun.profiler.hf import model_descriptor_from_yaml
from dryrun.profiler.kernels.config import KernelProfileConfig
from dryrun.profiler.kernels.measurements import PerfTableWriter

_FIXTURE_MODEL = Path("tests/fixtures/hf/qwen3_8b").resolve()
_CUDA = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(not _CUDA, reason="CUDA device is not available.")


def _config(tmp_path: Path) -> KernelProfileConfig:
    return KernelProfileConfig.from_dict(
        {
            "model": str(_FIXTURE_MODEL),
            "hardware": {"gpu_name": torch.cuda.get_device_name(0), "gpu_count": 1},
            "output_dir": str(tmp_path),
            "experiment_name": "test",
            "parallelism": {"tensor_parallel": 1},
            "gemm": {"token_counts": [64]},
            "attention": {
                "batch_sizes": [1, 4, 16],
                "context_seq_lens": [1024, 2048],
                "generation_steps": [1024, 2048],
                "block_size": 16,
            },
            "measurement": {"warmup_iterations": 2, "active_iterations": 5},
        },
        base_dir=tmp_path,
    )


def test_selected_backend_is_a_real_vllm_backend():
    from dryrun.profiler.kernels._vllm_attn import select_attention_backend

    selected = select_attention_backend(
        head_dim=128,
        num_heads=32,
        num_kv_heads=8,
        attn_dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        block_size=16,
        model_path=str(_FIXTURE_MODEL),
    )
    assert re.match(r"^vllm_", selected.kernel_source), selected.kernel_source
    assert selected.name in {"FLASH_ATTN", "FLASHINFER", "TRITON_ATTN"}


def test_profile_writes_both_phase_tables(tmp_path):
    from dryrun.profiler.kernels.attention import profile_vllm_attention

    config = _config(tmp_path)
    model = model_descriptor_from_yaml(str(_FIXTURE_MODEL), base_dir=Path.cwd())

    def writer_factory(framework: str, version: str, kernel_source: str) -> PerfTableWriter:
        return PerfTableWriter(
            output_root=config.table_root(),
            device=torch.cuda.get_device_name(0),
            framework=framework,
            version=version,
            kernel_source=kernel_source,
        )

    context_path, generation_path = profile_vllm_attention(model, config, writer_factory=writer_factory)
    context = KernelTable.load(context_path, CONTEXT_ATTENTION_OP)
    generation = KernelTable.load(generation_path, GENERATION_ATTENTION_OP)
    assert context.row_count == 6
    assert generation.row_count == 6
    assert context.kernel_sources()[0].startswith("vllm_")


def test_context_latency_grows_superlinearly_in_sequence_length(tmp_path):
    from dryrun.profiler.kernels.attention import profile_vllm_attention

    config = _config(tmp_path)
    model = model_descriptor_from_yaml(str(_FIXTURE_MODEL), base_dir=Path.cwd())
    context_path, _ = profile_vllm_attention(
        model,
        config,
        writer_factory=lambda framework, version, kernel_source: PerfTableWriter(
            output_root=config.table_root(),
            device=torch.cuda.get_device_name(0),
            framework=framework,
            version=version,
            kernel_source=kernel_source,
        ),
    )
    rows = dict(KernelTable.load(context_path, CONTEXT_ATTENTION_OP).rows((8, 128, "bfloat16", "bfloat16", 16, 0)))
    short = rows[(32.0, 1.0, 1024.0)]
    long = rows[(32.0, 1.0, 2048.0)]
    assert long > 1.5 * short, "Causal attention is quadratic in sequence length."


def test_generation_latency_grows_with_kv_history(tmp_path):
    from dryrun.profiler.kernels.attention import profile_vllm_attention

    config = _config(tmp_path)
    model = model_descriptor_from_yaml(str(_FIXTURE_MODEL), base_dir=Path.cwd())
    _, generation_path = profile_vllm_attention(
        model,
        config,
        writer_factory=lambda framework, version, kernel_source: PerfTableWriter(
            output_root=config.table_root(),
            device=torch.cuda.get_device_name(0),
            framework=framework,
            version=version,
            kernel_source=kernel_source,
        ),
    )
    rows = dict(
        KernelTable.load(generation_path, GENERATION_ATTENTION_OP).rows(
            (8, 128, "bfloat16", "bfloat16", 16, 0)
        )
    )
    # At small batch (e.g. 4) a constant per-call overhead inside vLLM's attention path swamps
    # the KV-length signal, so the ordering there is within jitter.  At batch 16 the KV-length
    # signal is large enough (measured ~43% gap) to beat the overhead reliably.
    short = rows[(32.0, 16.0, 1024.0)]
    long = rows[(32.0, 16.0, 2048.0)]
    assert long > short, "Longer KV history must take more time at batch 16."
    assert long > 1.20 * short, (
        f"KV-length scaling signal too small at batch 16: long={long * 1e3:.4f} ms, "
        f"short={short * 1e3:.4f} ms, ratio={long / short:.3f}x."
    )
