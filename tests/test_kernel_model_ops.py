"""The declarative op list, and its reuse of the profiler's TP sharding."""

from __future__ import annotations

from pathlib import Path

import pytest

from srl.cost.kernel.model_ops import ModelOps
from srl.profiler.schema import ProfileValidationError

_CHECKPOINT = Path("/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B")

pytestmark = pytest.mark.skipif(
    not (_CHECKPOINT / "config.json").exists(),
    reason="Needs the real Qwen3-8B checkpoint.",
)


@pytest.fixture(scope="module")
def ops():
    return ModelOps.from_checkpoint(str(_CHECKPOINT), tensor_parallel=1)


def test_the_architecture_comes_from_the_checkpoint(ops):
    """Seven interrelated integers. Hand-writing them is silently wrong when wrong."""
    assert ops.num_layers == 36
    assert ops.num_heads == 32
    assert ops.num_kv_heads == 8
    assert ops.head_dim == 128


def test_the_projection_shapes_match_the_profiled_sweep(ops):
    """These must be the shapes the sweep measured, or the queries miss the table."""
    assert {op.name: (op.n, op.k) for op in ops.projection_ops} == {
        "qkv_proj": (6144, 4096),
        "o_proj": (4096, 4096),
        "gate_up_proj": (24576, 4096),
        "down_proj": (4096, 12288),
    }


def test_every_projection_runs_once_per_layer(ops):
    assert all(op.per_layer for op in ops.projection_ops)


def test_the_logits_projection_runs_once_per_step_not_once_per_layer(ops):
    """vLLM computes logits for the last token of each sequence, outside the layer loop."""
    assert ops.logits_op.per_layer is False
    assert (ops.logits_op.n, ops.logits_op.k) == (151936, 4096)


@pytest.mark.parametrize(
    "tensor_parallel,heads,kv_heads,qkv_n,gate_up_n,logits_n",
    [(1, 32, 8, 6144, 24576, 151936), (2, 16, 4, 3072, 12288, 75968), (4, 8, 2, 1536, 6144, 37984)],
)
def test_sharding_follows_the_profiler(tensor_parallel, heads, kv_heads, qkv_n, gate_up_n, logits_n):
    ops = ModelOps.from_checkpoint(str(_CHECKPOINT), tensor_parallel=tensor_parallel)
    assert (ops.num_heads, ops.num_kv_heads) == (heads, kv_heads)
    shapes = {op.name: op.n for op in ops.projection_ops}
    assert shapes["qkv_proj"] == qkv_n
    assert shapes["gate_up_proj"] == gate_up_n
    assert ops.logits_op.n == logits_n


def test_an_indivisible_degree_is_rejected_rather_than_floored():
    """Flooring would query a shape the model never runs. Qwen3-8B has 8 KV heads."""
    with pytest.raises(ProfileValidationError):
        ModelOps.from_checkpoint(str(_CHECKPOINT), tensor_parallel=16)


def test_the_gemm_queries_carry_their_multipliers(ops):
    """Five queries: four at num_layers, the logits one at 1."""
    queries = ops.gemm_queries(token_count=512, request_count=8)
    assert len(queries) == 5
    multipliers = {}
    for query, multiplier in queries:
        multipliers[(query.n, query.k)] = (query.m, multiplier)
    assert multipliers[(6144, 4096)] == (512, 36)
    assert multipliers[(4096, 12288)] == (512, 36)
    # The logits projection's m is the request count, not the token count.
    assert multipliers[(151936, 4096)] == (8, 1)


def test_a_prefill_step_and_a_decode_step_differ_only_in_the_token_count(ops):
    """In decode every request contributes one token, so token_count == request_count."""
    decode = dict(
        ((query.n, query.k), query.m) for query, _ in ops.gemm_queries(token_count=8, request_count=8)
    )
    prefill = dict(
        ((query.n, query.k), query.m) for query, _ in ops.gemm_queries(token_count=4096, request_count=8)
    )
    assert decode[(6144, 4096)] == 8
    assert prefill[(6144, 4096)] == 4096
    assert decode[(151936, 4096)] == prefill[(151936, 4096)] == 8


def test_the_attention_query_carries_the_sharded_head_counts(ops):
    query = ops.attention_query("generation", batch_size=16, step=2048)
    assert (query.num_heads, query.num_kv_heads, query.head_dim) == (32, 8, 128)
    assert query.phase == "generation"
    assert query.step == 2048
    assert query.seq_len == 1


def test_a_context_attention_query_carries_the_sequence_length(ops):
    query = ops.attention_query("context", batch_size=4, seq_len=1024)
    assert query.phase == "context"
    assert query.seq_len == 1024
    assert query.step == 0


def test_no_collective_runs_at_tensor_parallel_one(ops):
    """This emptiness is what keeps every pinned tp=1 accuracy number unchanged."""
    assert ops.comm_queries(token_count=16) == ()
    assert ops.tensor_parallel == 1


def test_a_tensor_parallel_step_pays_two_all_reduces_per_layer():
    """One after o_proj and one after down_proj. An inline count would be a 36x error."""
    tp2 = ModelOps.from_checkpoint(str(_CHECKPOINT), tensor_parallel=2)
    queries = tp2.comm_queries(token_count=16)
    assert len(queries) == 1
    query, multiplier = queries[0]
    assert multiplier == 2 * tp2.num_layers == 72
    assert query.world_size == 2
    # The all-reduce carries the *unsharded* hidden size: TP shards the projections and then
    # reduces to reassemble the full activation, so the message does not shrink with the degree.
    assert query.message_bytes == 16 * 4096 * 2


def test_the_all_reduce_message_grows_with_the_token_count():
    """Prefill reduces every prompt token. Decode reduces one per request."""
    tp2 = ModelOps.from_checkpoint(str(_CHECKPOINT), tensor_parallel=2)
    decode = tp2.comm_queries(token_count=8)[0][0].message_bytes
    prefill = tp2.comm_queries(token_count=4096)[0][0].message_bytes
    assert prefill == decode * 512


def test_a_relative_checkpoint_path_resolves_against_base_dir():
    """A config-driven caller must not depend on the process working directory."""
    ops = ModelOps.from_checkpoint(
        "models/Qwen3-8B",
        tensor_parallel=1,
        base_dir=Path("/apdcephfs_zwfy/share_303760348/lhy"),
    )
    assert ops.num_layers == 36
