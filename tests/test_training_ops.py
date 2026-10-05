"""`TrainingOps` is a declarative op list, so its multipliers are what the tests check.

A composed model's failure mode is a plausible-looking MRE produced by a wrong count: 36
layers charged twice reads as a 2x miss, which looks like a hardware or overlap effect
rather than an arithmetic error. These tests assert the counts directly against the
per-layer inventory in `docs/superpowers/specs/2026-10-05-composed-training-design.md` §2,
which was derived from megatron-core and Transformer Engine source with file:line.

No GPU and no measured table: `TrainingOps` emits queries, it does not price them.
"""

from __future__ import annotations

import pytest

from dryrun.cost.kernel.database import ElementwiseQuery
from dryrun.cost.training.training_ops import TrainingOps
from dryrun.profiler.schema import ModelDescriptor, ProfileValidationError

# Qwen3-8B. `qk_layernorm` is in `extra` because that is where `load_hf_model_descriptor`
# puts it, and a test model that spelled it differently would not exercise the read.
_QWEN3_8B = ModelDescriptor(
    name="qwen3-8b-test",
    architecture="Qwen3ForCausalLM",
    num_layers=36,
    hidden_size=4096,
    num_attention_heads=32,
    num_query_groups=8,
    extra={"intermediate_size": 12288, "head_dim": 128, "qk_layernorm": True, "vocab_size": 151936},
)
# Same architecture without the Qwen3 per-head norms, to prove they are conditional.
_NO_QK_NORM = ModelDescriptor(
    name="llama-like-test",
    architecture="LlamaForCausalLM",
    num_layers=36,
    hidden_size=4096,
    num_attention_heads=32,
    num_query_groups=8,
    extra={"intermediate_size": 12288, "head_dim": 128, "qk_layernorm": False, "vocab_size": 151936},
)

_PROJECTIONS_PER_LAYER = 4


def _ops(model: ModelDescriptor = _QWEN3_8B, **kwargs: object) -> TrainingOps:
    kwargs.setdefault("tensor_parallel", 2)
    return TrainingOps(model, **kwargs)  # type: ignore[arg-type]


def test_forward_gemm_queries_charge_one_per_projection_per_layer():
    ops = _ops(recompute="none")
    queries = ops.forward_gemm_queries(4096)
    assert len(queries) == _PROJECTIONS_PER_LAYER
    assert all(multiplier == 36 for _, multiplier in queries)


def test_full_recompute_doubles_the_forward_gemms_but_not_the_backward_ones():
    """`CheckpointFunction.backward` replays the forward; it does not replay the backward."""
    plain = dict(_ops(recompute="none").forward_gemm_queries(4096))
    recomputed = dict(_ops(recompute="full").forward_gemm_queries(4096))
    assert set(plain) == set(recomputed)
    for query, multiplier in plain.items():
        assert recomputed[query] == 2 * multiplier

    plain_bwd = dict(_ops(recompute="none").backward_gemm_queries(4096))
    recomputed_bwd = dict(_ops(recompute="full").backward_gemm_queries(4096))
    assert plain_bwd == recomputed_bwd


def test_backward_emits_a_dgrad_and_a_wgrad_for_every_projection():
    ops = _ops()
    queries = ops.backward_gemm_queries(4096)
    assert len(queries) == 2 * _PROJECTIONS_PER_LAYER
    assert all(multiplier == 36 for _, multiplier in queries)


def test_the_dgrad_and_wgrad_triples_are_the_two_distinct_permutations():
    """`F.linear(A[m,k], W[n,k]) -> C[m,n]`, so each gradient has its own slice key.

    This is the convention §1.2(b) of the design doc found wrong in `gemm_wgrad_shapes`:
    the FLOP count is symmetric under the permutation, so only the shape catches it.
    """
    token_count = 7510
    ops = _ops()
    forward = {(q.n, q.k) for q, _ in ops.forward_gemm_queries(token_count)}
    backward = [q for q, _ in ops.backward_gemm_queries(token_count)]

    for n_fwd, k_fwd in forward:
        # dgrad computes dX[tokens, k_fwd] from dY[tokens, n_fwd] @ W[n_fwd, k_fwd].
        assert any(
            q.m == token_count and q.n == k_fwd and q.k == n_fwd for q in backward
        ), f"no dgrad for forward (n={n_fwd}, k={k_fwd})"
        # wgrad computes dW, which has the weight's own shape [n_fwd, k_fwd].
        assert any(
            q.m == n_fwd and q.n == k_fwd and q.k == token_count for q in backward
        ), f"no wgrad for forward (n={n_fwd}, k={k_fwd})"


def test_sequence_parallel_collectives_are_counted_per_layer_not_per_model_twice():
    """The count is per-layer x num_layers exactly once.

    `ModelOps.comm_queries` carries the same check. Multiplying a per-layer count by the
    layer count twice yields 36x, which in a composed sum looks like a hardware effect.
    """
    ops = _ops(recompute="full")
    queries = dict(ops.tp_comm_queries(4096))
    by_op = {query.op: multiplier for query, multiplier in queries.items()}
    assert by_op == {"all_gather": 8 * 36, "reduce_scatter": 6 * 36}


def test_without_recompute_the_collective_count_drops_by_the_forward_pass():
    """Forward is 2 AG + 2 RS, so omitting the replay removes exactly that."""
    ops = _ops(recompute="none")
    by_op = {query.op: multiplier for query, multiplier in ops.tp_comm_queries(4096)}
    assert by_op == {"all_gather": 6 * 36, "reduce_scatter": 4 * 36}


def test_pipeline_sends_cross_every_boundary_in_both_directions():
    """Under 1F1B each microbatch crosses `pp - 1` boundaries forward and the same back."""
    ops = _ops()
    queries = ops.pp_comm_queries(4096, pipeline_parallel=3)
    assert len(queries) == 1
    query, multiplier = queries[0]
    assert query.op == "p2p"
    assert multiplier == 2 * (3 - 1)


def test_the_pipeline_send_carries_the_sequence_parallel_shard():
    """The handoff tensor is sequence-sharded under SP, so each rank sends tokens/tp rows."""
    token_count = 8192
    tp = 2
    ops = _ops(tensor_parallel=tp)
    (query, _), = ops.pp_comm_queries(token_count, pipeline_parallel=2)
    assert query.message_bytes == (token_count // tp) * 4096 * 2


def test_a_single_stage_pipeline_sends_nothing():
    """There is no boundary to cross, so the term must be absent rather than zero-valued."""
    assert _ops().pp_comm_queries(4096, pipeline_parallel=1) == ()


def test_tp1_issues_no_tensor_parallel_collectives():
    assert _ops(tensor_parallel=1).tp_comm_queries(4096) == ()


def test_the_collective_message_carries_the_full_hidden_size():
    """`tp` shards the sequence dimension under SP, not the hidden one."""
    token_count = 4096
    ops = _ops(tensor_parallel=2)
    for query, _ in ops.tp_comm_queries(token_count):
        assert query.message_bytes == token_count * 4096 * 2


def test_the_hidden_size_norms_are_charged_on_the_sequence_parallel_shard():
    """Two RMSNorms per layer, each over `token_count / tp` rows of `hidden_size`.

    They sit outside the all_gather, so charging them the full sequence overstates them by
    `tp`. This is the one place where the elementwise inventory's sharding differs from the
    attention-side kernels', which run post-gather.
    """
    token_count = 8192
    tp = 2
    ops = _ops(tensor_parallel=tp)
    norms = [
        (query, multiplier)
        for query, multiplier in ops.elementwise_queries(token_count)
        if query.op_kind == "rms_norm"
    ]
    assert len(norms) == 2, "input_layernorm and pre_mlp_layernorm"
    for query, _ in norms:
        assert query.n_elements == (token_count // tp) * 4096


def test_the_attention_side_elementwise_kernels_see_the_full_sequence():
    """RoPE and the q/k norms run inside the TP region, after the all_gather.

    Their sharding is on heads, which `rank_head_counts` already applied, so passing the
    full token count is correct and dividing again would halve them twice.
    """
    token_count = 8192
    ops = _ops(tensor_parallel=2)
    queries = {
        (query.op_kind, query.n_elements)
        for query, _ in ops.elementwise_queries(token_count)
    }
    # 32 heads / tp=2 = 16 query heads; 8 kv heads / tp=2 = 4; head_dim 128.
    assert ("rotary_embedding", token_count * 16 * 128) in queries
    assert ("rotary_embedding", token_count * 4 * 128) in queries
    assert ("qk_norm", token_count * 16 * 128) in queries
    assert ("qk_norm", token_count * 4 * 128) in queries


def test_swiglu_is_charged_over_the_tensor_parallel_sharded_intermediate():
    token_count = 8192
    ops = _ops(tensor_parallel=2)
    swiglu = [
        query
        for query, _ in ops.elementwise_queries(token_count)
        if query.op_kind == "silu_and_mul"
    ]
    assert len(swiglu) == 1, "chunk + silu + mul are one fused kernel"
    assert swiglu[0].n_elements == token_count * (12288 // 2)


def test_a_model_without_qk_layernorm_is_charged_no_per_head_norms():
    """Qwen3 applies `q_norm`/`k_norm`; earlier architectures do not.

    Charging them unconditionally would add two kernels per layer to every model, and
    `qk_norm` is the most expensive kind per element (122.9 us against `rms_norm`'s 40.8 us
    at the same count on H20), so the error would not be small.
    """
    kinds = [query.op_kind for query, _ in _ops(_NO_QK_NORM).elementwise_queries(4096)]
    assert "qk_norm" not in kinds
    assert kinds.count("rms_norm") == 2
    assert kinds.count("rotary_embedding") == 2
    assert kinds.count("silu_and_mul") == 1


def test_elementwise_multipliers_count_forward_recompute_and_backward():
    """Three forward-shaped passes under full recompute, two without."""
    for recompute, passes in (("full", 3.0), ("none", 2.0)):
        ops = _ops(recompute=recompute)
        multipliers = {m for _, m in ops.elementwise_queries(4096)}
        assert multipliers == {36 * passes}, recompute


def test_every_elementwise_query_names_a_measured_kind():
    """A typo'd kind must raise at construction, not resolve to a neighbouring curve."""
    from dryrun.cost.kernel.ops.elementwise import ELEMENTWISE_KINDS

    for query, _ in _ops().elementwise_queries(4096):
        assert query.op_kind in ELEMENTWISE_KINDS


def test_an_unknown_elementwise_kind_is_refused():
    from dryrun.cost.kernel.interp import KernelCoverageError

    with pytest.raises(KernelCoverageError, match="Unknown elementwise op_kind"):
        ElementwiseQuery(op_kind="layer_norm", n_elements=4096)


def test_attention_passes_counts_only_the_forward_shaped_passes():
    """Backward is priced separately through `attn_bwd_fwd_ratio`, so it is not counted here."""
    assert _ops(recompute="full").attention_passes() == 2
    assert _ops(recompute="none").attention_passes() == 1


def test_the_logits_gemm_runs_over_every_token_and_only_once_per_step():
    """Inference computes one logit row per request; training computes all of them."""
    token_count = 12288
    queries = _ops().logits_queries(token_count)
    assert all(multiplier == 1 for _, multiplier in queries)
    forward = queries[0][0]
    assert forward.m == token_count, "training logits are not per-request"


def test_an_invalid_recompute_mode_is_refused_at_construction():
    with pytest.raises(ProfileValidationError, match="recompute must be"):
        _ops(recompute="selective")


def test_dp_collectives_are_absent_at_dp_one_rather_than_raising():
    """The term is inactive rather than unimplemented at the degree that is validated."""
    assert _ops(dp=1).dp_comm_queries() == ()


def test_dp_collectives_above_one_refuse_rather_than_returning_a_guess():
    """Spec B8: no `dp>1` THD record exists, so the term must not look validated."""
    with pytest.raises(NotImplementedError, match="dp>1"):
        _ops(dp=4).dp_comm_queries()
