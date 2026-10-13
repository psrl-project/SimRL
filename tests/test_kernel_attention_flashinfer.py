"""Attention profiling through FlashInfer's paged wrappers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dryrun.cost.kernel.database import KernelDatabase
from dryrun.cost.kernel.ops.attention import GENERATION_ATTENTION_OP
from dryrun.cost.kernel.table import KernelTable
from dryrun.profiler.hf import model_descriptor_from_yaml
from dryrun.profiler.kernel.config import KernelProfileConfig
from dryrun.profiler.kernel.measurements import PerfTableWriter

_FIXTURE_MODEL = Path("tests/fixtures/hf/qwen3_8b").resolve()

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device is not available."
)


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
                "batch_sizes": [4],
                "context_seq_lens": [1024],
                "generation_steps": [1024, 2048],
                "block_size": 16,
            },
            "measurement": {"warmup_iterations": 2, "active_iterations": 5},
        },
        base_dir=tmp_path,
    )


def _factory(config):
    def writer_factory(framework: str, version: str, kernel_source: str) -> PerfTableWriter:
        return PerfTableWriter(
            output_root=config.table_root(),
            device=torch.cuda.get_device_name(0),
            framework=framework,
            version=version,
            kernel_source=kernel_source,
        )

    return writer_factory


def test_kernel_source_reports_the_flashinfer_version():
    import flashinfer

    from dryrun.profiler.kernel.runner._flashinfer_attn import flashinfer_kernel_source

    assert flashinfer_kernel_source() == f"flashinfer_{flashinfer.__version__}"


def test_profile_writes_a_loadable_generation_table(tmp_path):
    from dryrun.profiler.kernel.runner._flashinfer_attn import profile_flashinfer_attention

    config = _config(tmp_path)
    model = model_descriptor_from_yaml(str(_FIXTURE_MODEL), base_dir=Path.cwd())
    _, generation_path = profile_flashinfer_attention(model, config, writer_factory=_factory(config))
    table = KernelTable.load(generation_path, GENERATION_ATTENTION_OP)
    assert table.row_count == 2
    assert table.kernel_sources()[0].startswith("flashinfer_")


def test_both_backends_are_visible_to_the_database(tmp_path):
    from dryrun.profiler.kernel.runner._flashinfer_attn import profile_flashinfer_attention
    from dryrun.profiler.kernel.runner._vllm_attn import profile_vllm_attention

    config = _config(tmp_path)
    model = model_descriptor_from_yaml(str(_FIXTURE_MODEL), base_dir=Path.cwd())
    profile_vllm_attention(model, config, writer_factory=_factory(config))
    profile_flashinfer_attention(model, config, writer_factory=_factory(config))
    database = KernelDatabase.from_root(
        config.table_root(), gpu=torch.cuda.get_device_name(0)
    )
    assert database.backends("generation_attention") == ("flashinfer", "vllm")
