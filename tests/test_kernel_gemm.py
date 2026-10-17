"""Per-rank GEMM shape enumeration and, when a GPU is present, timing."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from srl.cost.kernel.ops.gemm import GEMM_OP
from srl.cost.kernel.table import KernelTable
from srl.profiler.hf import model_descriptor_from_yaml
from srl.profiler.kernel.config import KernelProfileConfig
from srl.profiler.kernel.grid.gemm import (
    gemm_all_shapes,
    gemm_dgrad_shapes,
    gemm_sweep_shapes,
    gemm_wgrad_shapes,
    logits_gemm_shape,
    logits_sweep_shapes,
    transformer_gemm_shapes,
)
from srl.profiler.kernel.measurements import PerfTableWriter
from srl.profiler.kernel.runner.gemm import GemmRunner
from srl.profiler.kernel.sweep import GemmSweep
from srl.profiler.schema import ProfileValidationError

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
    # The table row (m, n, k) means F.linear(A[m,k], W[n,k]) -> C[m,n], which is what
    # gemm.py times. dW has the weight's shape [n_fwd, k_fwd] and is dY^T @ X, so the
    # contracted length is the token count: (m, n, k) = (n_fwd, k_fwd, tokens).
    #
    # These expectations were (n_fwd, m, k_fwd) until 2026-10-05, matching a bug in
    # gemm_wgrad_shapes. The FLOP count is identical either way because the product is
    # symmetric, so neither the code nor this test could catch it by cost; the test below
    # pins the convention against F.linear's actual output shape instead.
    sweep = GemmSweep(token_counts=(64, 1024), logits_request_counts=(1, 64))
    shapes = gemm_wgrad_shapes(model, sweep, tensor_parallel=1)
    assert len(shapes) == len(set(shapes))
    # qkv_proj forward (n=6144, k=4096) → wgrad (6144, 4096, 64) and (6144, 4096, 1024)
    assert (6144, 4096, 64) in shapes
    assert (6144, 4096, 1024) in shapes
    # down_proj forward (n=4096, k=12288) → wgrad (4096, 12288, 64)
    assert (4096, 12288, 64) in shapes


def test_the_wgrad_triple_produces_the_weight_gradients_own_shape():
    """Pin the convention against the kernel being timed, not against FLOP arithmetic.

    This is the check that would have caught the swapped triple. A wgrad row must, when
    run as the table's own call, produce a tensor the shape of the weight it gradients.
    FLOPs cannot distinguish the orderings; output shape can.
    """
    import torch

    token_count, n_fwd, k_fwd = 4096, 4096, 2048  # o_proj at tp=2, where n_fwd != k_fwd
    reference = (torch.randn(token_count, n_fwd).T @ torch.randn(token_count, k_fwd)).shape
    assert tuple(reference) == (n_fwd, k_fwd)

    m, n, k = n_fwd, k_fwd, token_count  # the triple gemm_wgrad_shapes emits
    produced = torch.nn.functional.linear(torch.randn(m, k), torch.randn(n, k)).shape
    assert tuple(produced) == tuple(reference)


def test_wgrad_shapes_cover_all_forward_projections(model):
    # For each forward (n, k) projection, every token count must appear as a wgrad slice.
    sweep = GemmSweep(token_counts=(1, 32, 512), logits_request_counts=(1, 32))
    forward = transformer_gemm_shapes(model, tensor_parallel=1)
    wgrad = set(gemm_wgrad_shapes(model, sweep, tensor_parallel=1))
    for shape in forward:
        for m in sweep.token_counts:
            expected = (shape.n, shape.k, m)
            assert expected in wgrad, (
                f"Wgrad shape {expected} missing for forward projection "
                f"{shape.name} (n={shape.n}, k={shape.k}) at token count {m}."
            )


def test_gemm_all_shapes_is_union_of_forward_dgrad_wgrad_and_logits(model):
    sweep = GemmSweep(token_counts=(64, 1024), logits_request_counts=(1, 64))
    forward = set(gemm_sweep_shapes(model, sweep, tensor_parallel=1))
    dgrad = set(gemm_dgrad_shapes(model, sweep, tensor_parallel=1))
    wgrad = set(gemm_wgrad_shapes(model, sweep, tensor_parallel=1))
    logits = set(logits_sweep_shapes(model, sweep, tensor_parallel=1))
    combined = gemm_all_shapes(model, sweep, tensor_parallel=1)
    assert set(combined) == forward | dgrad | wgrad | logits
    assert len(combined) == len(set(combined))  # deduplicated


def test_dgrad_shapes_swap_the_forward_n_and_k(model):
    """dgrad is dY @ W, so the forward's n and k exchange roles.

    That swap is why backward is not free: per-(n, k) slicing makes (k_fwd, n_fwd) a
    different slice from (n_fwd, k_fwd), and three of Qwen3-8B's four projections land in
    a slice no forward sweep writes. Measured before this was added: qkv_proj, gate_up_proj
    and down_proj dgrads all raised KernelCoverageError.
    """
    sweep = GemmSweep(token_counts=(64, 1024), logits_request_counts=(1, 64))
    dgrad = set(gemm_dgrad_shapes(model, sweep, tensor_parallel=1))
    for shape in transformer_gemm_shapes(model, tensor_parallel=1):
        for m in sweep.token_counts:
            assert (m, shape.k, shape.n) in dgrad, (
                f"Dgrad shape {(m, shape.k, shape.n)} missing for {shape.name}."
            )


def test_dgrad_output_has_the_activation_gradients_shape():
    """Same convention check as wgrad: validate against the call, not against FLOPs."""
    import torch

    token_count, n_fwd, k_fwd = 4096, 3072, 4096  # qkv_proj at tp=2
    # dX has the activation's shape [tokens, k_fwd], from dY[tokens, n_fwd] @ W[n_fwd, k_fwd].
    m, n, k = token_count, k_fwd, n_fwd
    produced = torch.nn.functional.linear(torch.randn(m, k), torch.randn(n, k)).shape
    assert tuple(produced) == (token_count, k_fwd)


@pytest.mark.skipif(not _CUDA, reason="CUDA device is not available.")
def test_gemm_runner_writes_a_loadable_table(tmp_path, model):
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
    # The runner calls `writer_factory` itself, because the families that select a backend at
    # runtime can only name their kernel source mid-sweep. GEMM's is known up front, so the
    # factory here ignores its arguments and hands back the writer built above.
    (path,) = GemmRunner(model, config).run(writer_factory=lambda *_: writer)
    table = KernelTable.load(path, GEMM_OP)
    # 8 forward + 8 dgrad + 8 wgrad (no overlaps at these token counts) + 2 logits = 26,
    # minus the 2 forward/dgrad coincidences where a projection's n equals its k
    # (o_proj at tp=1 is 4096x4096, so its dgrad triple is its forward triple).
    expected = len(gemm_all_shapes(model, config.sweep("gemm"), tensor_parallel=1))
    assert table.row_count == expected
    assert expected == 24
    # With per-(n,k) slicing each projection's rows live in a distinct slice; sample one.
    latencies_qkv = dict(table.rows(("bfloat16", 6144, 4096)))
    assert all(value > 0.0 for value in latencies_qkv.values())
    assert latencies_qkv[(1024.0,)] > latencies_qkv[(64.0,)]
    # Logits projection must be present and queryable.
    latencies_lm_head = dict(table.rows(("bfloat16", 151936, 4096)))
    assert all(value > 0.0 for value in latencies_lm_head.values())
