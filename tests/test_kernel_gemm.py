"""Per-rank GEMM shape enumeration and, when a GPU is present, timing."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dryrun.cost.kernel.ops.gemm import GEMM_OP
from dryrun.cost.kernel.table import KernelTable
from dryrun.profiler.hf import model_descriptor_from_yaml
from dryrun.profiler.kernels.config import GemmSweep, KernelProfileConfig
from dryrun.profiler.kernels.gemm import (
    gemm_all_shapes,
    gemm_sweep_shapes,
    gemm_wgrad_shapes,
    logits_gemm_shape,
    logits_sweep_shapes,
    profile_gemm,
    transformer_gemm_shapes,
)
from dryrun.profiler.kernels.measurements import PerfTableWriter
from dryrun.profiler.schema import ProfileValidationError

_FIXTURE_MODEL = Path("tests/fixtures/hf/qwen3_8b").resolve()
_CUDA = torch.cuda.is_available()


@pytest.fixture
def model():
    return model_descriptor_from_yaml(str(_FIXTURE_MODEL), base_dir=Path.cwd())


def test_tp1_shapes_match_qwen3_8b(model):
    shapes = {shape.name: (shape.n, shape.k) for shape in transformer_gemm_shapes(model, tensor_parallel=1)}
    assert shapes == {
        "qkv_proj": (6144, 4096),
        "o_proj": (4096, 4096),
        "gate_up_proj": (24576, 4096),
        "down_proj": (4096, 12288),
    }


def test_tp2_shards_heads_and_intermediate(model):
    shapes = {shape.name: (shape.n, shape.k) for shape in transformer_gemm_shapes(model, tensor_parallel=2)}
    assert shapes == {
        "qkv_proj": (3072, 4096),
        "o_proj": (4096, 2048),
        "gate_up_proj": (12288, 4096),
        "down_proj": (4096, 6144),
    }


def test_indivisible_tensor_parallel_is_rejected(model):
    with pytest.raises(ProfileValidationError):
        transformer_gemm_shapes(model, tensor_parallel=3)


def test_logits_gemm_shape_tp1(model):
    # Qwen3-8B: vocab_size=151936, hidden_size=4096, TP=1 so n=151936, k=4096.
    shape = logits_gemm_shape(model, tensor_parallel=1)
    assert shape.name == "lm_head"
    assert shape.n == 151936
    assert shape.k == 4096


def test_logits_gemm_shape_tp2(model):
    # vLLM shards the LM head over TP, so n = vocab_size / tp = 151936 / 2 = 75968.
    shape = logits_gemm_shape(model, tensor_parallel=2)
    assert shape.name == "lm_head"
    assert shape.n == 75968
    assert shape.k == 4096


def test_logits_name_distinguishes_from_per_layer_projections(model):
    # The lm_head name must not appear among per-layer projection names.
    per_layer_names = {shape.name for shape in transformer_gemm_shapes(model, tensor_parallel=1)}
    logits_name = logits_gemm_shape(model, tensor_parallel=1).name
    assert logits_name not in per_layer_names


def test_logits_sweep_shapes_uses_request_counts(model):
    sweep = GemmSweep(token_counts=(1, 64), logits_request_counts=(1, 8, 64))
    shapes = logits_sweep_shapes(model, sweep, tensor_parallel=1)
    assert len(shapes) == len(set(shapes))
    assert (1, 151936, 4096) in shapes
    assert (8, 151936, 4096) in shapes
    assert (64, 151936, 4096) in shapes
    # Only request counts appear in the logits shapes, not arbitrary token counts.
    assert len(shapes) == 3


def test_logits_sweep_shapes_not_in_gemm_sweep_shapes(model):
    # The forward token-count sweep must not contain the logits (n=151936, k=4096) shape.
    sweep = GemmSweep(token_counts=(1, 64), logits_request_counts=(1, 64))
    forward = set(gemm_sweep_shapes(model, sweep, tensor_parallel=1))
    logits = set(logits_sweep_shapes(model, sweep, tensor_parallel=1))
    assert forward.isdisjoint(logits)


def test_sweep_shapes_cross_tokens_with_shapes_and_deduplicate(model):
    sweep = GemmSweep(token_counts=(1, 1024), logits_request_counts=(1, 64))
    shapes = gemm_sweep_shapes(model, sweep, tensor_parallel=1)
    assert len(shapes) == len(set(shapes))
    assert (1, 6144, 4096) in shapes
    assert (1024, 4096, 12288) in shapes
    assert len(shapes) == 2 * 4


def test_wgrad_shapes_are_derived_from_forward_projections(model):
    # For each forward (n_fwd, k_fwd) and token count m, the wgrad GEMM is
    # (m_wgrad, n_wgrad, k_wgrad) = (n_fwd, m, k_fwd).  Verify presence for a
    # representative (n_fwd=6144, k_fwd=4096) projection with m=1024.
    sweep = GemmSweep(token_counts=(64, 1024), logits_request_counts=(1, 64))
    shapes = gemm_wgrad_shapes(model, sweep, tensor_parallel=1)
    assert len(shapes) == len(set(shapes))
    # qkv_proj forward (n=6144, k=4096) → wgrad (6144, 64, 4096) and (6144, 1024, 4096)
    assert (6144, 64, 4096) in shapes
    assert (6144, 1024, 4096) in shapes
    # down_proj forward (n=4096, k=12288) → wgrad (4096, 64, 12288)
    assert (4096, 64, 12288) in shapes


def test_wgrad_shapes_cover_all_forward_projections(model):
    # For each forward (n, k) projection, every token count must appear as a wgrad slice.
    sweep = GemmSweep(token_counts=(1, 32, 512), logits_request_counts=(1, 32))
    forward = transformer_gemm_shapes(model, tensor_parallel=1)
    wgrad = set(gemm_wgrad_shapes(model, sweep, tensor_parallel=1))
    for shape in forward:
        for m in sweep.token_counts:
            expected = (shape.n, m, shape.k)
            assert expected in wgrad, (
                f"Wgrad shape {expected} missing for forward projection "
                f"{shape.name} (n={shape.n}, k={shape.k}) at token count {m}."
            )


def test_gemm_all_shapes_is_union_of_forward_wgrad_and_logits(model):
    sweep = GemmSweep(token_counts=(64, 1024), logits_request_counts=(1, 64))
    forward = set(gemm_sweep_shapes(model, sweep, tensor_parallel=1))
    wgrad = set(gemm_wgrad_shapes(model, sweep, tensor_parallel=1))
    logits = set(logits_sweep_shapes(model, sweep, tensor_parallel=1))
    combined = gemm_all_shapes(model, sweep, tensor_parallel=1)
    assert set(combined) == forward | wgrad | logits
    assert len(combined) == len(set(combined))  # deduplicated


@pytest.mark.skipif(not _CUDA, reason="CUDA device is not available.")
def test_profile_gemm_writes_a_loadable_table(tmp_path, model):
    config = KernelProfileConfig.from_dict(
        {
            "model": str(_FIXTURE_MODEL),
            "hardware": {"gpu_name": torch.cuda.get_device_name(0), "gpu_count": 1},
            "output_dir": str(tmp_path),
            "experiment_name": "test",
            "parallelism": {"tensor_parallel": 1},
            "gemm": {"token_counts": [64, 1024], "logits_request_counts": [1, 8]},
            "attention": {
                "batch_sizes": [1],
                "context_seq_lens": [256],
                "generation_steps": [0],
            },
            "measurement": {"warmup_iterations": 1, "active_iterations": 2},
        },
        base_dir=tmp_path,
    )
    writer = PerfTableWriter(
        output_root=config.table_root(),
        device=torch.cuda.get_device_name(0),
        framework="vllm",
        version="0.26.1.dev0",
        kernel_source="torch.nn.functional.linear",
    )
    path = profile_gemm(model, config, writer=writer)
    table = KernelTable.load(path, GEMM_OP)
    # 8 forward shapes + 8 wgrad shapes (no overlaps for these token counts) + 2 logits shapes.
    assert table.row_count == 18
    # With per-(n,k) slicing each projection's rows live in a distinct slice; sample one.
    latencies_qkv = dict(table.rows(("bfloat16", 6144, 4096)))
    assert all(value > 0.0 for value in latencies_qkv.values())
    assert latencies_qkv[(1024.0,)] > latencies_qkv[(64.0,)]
    # Logits projection must be present and queryable.
    latencies_lm_head = dict(table.rows(("bfloat16", 151936, 4096)))
    assert all(value > 0.0 for value in latencies_lm_head.values())
