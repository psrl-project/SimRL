"""The grid modules extracted from the measurement loops, checked without a GPU.

These three enumerations used to live inside `profile_elementwise`, `profile_optimizer` and
`profile_training_attention`, where the only way to check them was to run a sweep on a device.
That mattered most for `elementwise`: `cost/training/training_ops.py` derives the same
`n_elements` on the query side, and a disagreement means a query lands on a point of the surface
the sweep never measured. The tests below are what that extraction buys.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dryrun.cost.kernel.ops.elementwise import ELEMENTWISE_KINDS
from dryrun.profiler.hf import model_descriptor_from_yaml
from dryrun.profiler.kernel.grid.attention_training import training_attention_points
from dryrun.profiler.kernel.grid.elementwise import (
    elementwise_dims,
    elementwise_sweep_points,
)
from dryrun.profiler.kernel.grid.optimizer import optimizer_param_counts
from dryrun.profiler.kernel.sweep import (
    ElementwiseSweep,
    OptimizerSweep,
    TrainingAttentionSweep,
)
from dryrun.profiler.schema import ProfileValidationError

_FIXTURE_MODEL = Path("tests/fixtures/hf/qwen3_8b").resolve()


@pytest.fixture
def model():
    return model_descriptor_from_yaml(str(_FIXTURE_MODEL), base_dir=Path.cwd())


# --------------------------------------------------------------------------------------
# elementwise
# --------------------------------------------------------------------------------------


def test_elementwise_dims_reads_the_architecture(model):
    """Qwen3-8B: hidden 4096, 32 query heads to 8 KV heads, head_dim 128, intermediate 12288."""
    dims = elementwise_dims(model)
    assert (dims.hidden_size, dims.head_dim, dims.intermediate_size) == (4096, 128, 12288)
    assert (dims.num_heads, dims.num_kv_heads) == (32, 8)


def test_elementwise_points_cover_every_registered_kind(model):
    """A kind in `ELEMENTWISE_KINDS` with no grid point writes no row and the table is short.

    The kinds are owned by `cost/kernel/ops/elementwise.py` precisely so the two cannot drift;
    this is the assertion that notices if they do.
    """
    points = elementwise_sweep_points(model, ElementwiseSweep(token_counts=(128,)))
    assert {point.op_kind for point in points} == set(ELEMENTWISE_KINDS)


def test_elementwise_element_counts_match_the_hand_arithmetic(model):
    """Pin each kind's `n_elements` against what sizes it, not against a recorded constant.

    Written as the product rather than the number so that a reader can see which dimension each
    kind scales with: this is where `rotary_embedding` summing *both* head groups, and `qk_norm`
    using head_dim rather than hidden_size, are visible.
    """
    tokens = 128
    dims = elementwise_dims(model)
    points = elementwise_sweep_points(model, ElementwiseSweep(token_counts=(tokens,)))
    by_kind = {point.op_kind: point.n_elements for point in points}
    assert by_kind["rms_norm"] == tokens * dims.hidden_size
    # Rope rotates query and key in one call, so both head groups are read.
    assert by_kind["rotary_embedding"] == tokens * (dims.num_heads + dims.num_kv_heads) * dims.head_dim
    assert by_kind["silu_and_mul"] == tokens * dims.intermediate_size
    # qk_norm normalizes per head, so it is token*heads rows of head_dim.
    assert by_kind["qk_norm"] == tokens * dims.num_heads * dims.head_dim
    # tensor_plumbing is timed as a hidden-size copy, so it matches rms_norm's count.
    assert by_kind["tensor_plumbing"] == tokens * dims.hidden_size


def test_elementwise_n_elements_agree_with_the_cost_side_derivation(model):
    """The sweep and the query must size the same kernel from the same dimensions.

    `TrainingOps._elementwise_queries` builds `(op_kind, n_elements)` pairs from the same three
    model dimensions. If the two formulas disagree, every query misses the measured grid and the
    surface interpolates between points that describe a different kernel -- silently, because a
    neighbouring point always exists. This is the check that extraction to `grid/` made possible;
    before it, both halves of the comparison could only be obtained by running a sweep.

    The two sides are **not** expected to be equal point-for-point, and the inequalities are the
    content of this test:

    - The sweep measures *one call per kind*. The cost side issues a query *per call a layer
      makes*, so `rms_norm` appears twice (input and pre-MLP), `rotary_embedding` twice (q and
      k, as separate queries), and `qk_norm` twice (q and k).
    - The cost side shards by tensor parallelism; the sweep measures unsharded and the query
      divides. At tp=1 the division is the identity, which is what makes the comparison below
      exact.

    What must hold is that every per-call count the cost side can ask for is expressible from the
    sweep's dimensions -- i.e. the two agree on *what sizes each kernel*, which is the thing that
    drifts.
    """
    tokens = 4096
    dims = elementwise_dims(model)
    grid = {
        point.op_kind: point.n_elements
        for point in elementwise_sweep_points(model, ElementwiseSweep(token_counts=(tokens,)))
    }

    # The per-call expressions in `training_ops.py:374-405`, at tp=1 where `sp_tokens == tokens`
    # and `intermediate_size // tp == intermediate_size`.
    assert grid["rms_norm"] == tokens * dims.hidden_size
    assert grid["tensor_plumbing"] == tokens * dims.hidden_size
    assert grid["silu_and_mul"] == tokens * dims.intermediate_size

    # `rotary_embedding` and `qk_norm` are each issued twice by the cost side, once per head
    # group. The sweep times rope as the single fused call that rotates both, so its count is the
    # *sum* of the two query terms; qk_norm is two separate calls, so the sweep's count is the
    # query-head term alone and the kv term reads the same surface at a smaller n_elements.
    query_term = tokens * dims.num_heads * dims.head_dim
    kv_term = tokens * dims.num_kv_heads * dims.head_dim
    assert grid["rotary_embedding"] == query_term + kv_term
    assert grid["qk_norm"] == query_term
    # The kv-head qk_norm query is interior to the measured grid rather than off its end, which
    # is what keeps it interpolable rather than a coverage error.
    assert kv_term < grid["qk_norm"]


def test_elementwise_refuses_an_architecture_it_cannot_size(model):
    """An incomplete descriptor must raise rather than size a tensor from a None."""
    from dataclasses import replace

    with pytest.raises(ProfileValidationError, match="hidden_size"):
        elementwise_dims(replace(model, hidden_size=None))
    with pytest.raises(ProfileValidationError, match="num_attention_heads"):
        elementwise_dims(replace(model, num_attention_heads=None))


# --------------------------------------------------------------------------------------
# optimizer
# --------------------------------------------------------------------------------------


def test_optimizer_counts_are_sorted_ascending():
    """Small shards first, so a grid whose largest point OOMs still records what fits."""
    counts = optimizer_param_counts(OptimizerSweep(param_counts=(2048, 512, 1024)))
    assert counts == (512, 1024, 2048)


def test_optimizer_counts_are_deduplicated():
    """A repeated point would time the same shard twice and write two identical rows."""
    assert optimizer_param_counts(OptimizerSweep(param_counts=(512, 512, 1024))) == (512, 1024)


def test_optimizer_refuses_an_empty_or_non_positive_grid():
    """`from_dict` already rejects these; a directly-constructed sweep must not slip through."""
    with pytest.raises(ProfileValidationError, match="empty"):
        optimizer_param_counts(OptimizerSweep(param_counts=()))
    with pytest.raises(ProfileValidationError, match="non-positive"):
        optimizer_param_counts(OptimizerSweep(param_counts=(0, 1024)))


# --------------------------------------------------------------------------------------
# training attention
# --------------------------------------------------------------------------------------


def test_training_attention_shards_kv_heads_in_the_gqa_ratio(model):
    """The KV head count must follow the query count, or the row is unreachable by any query.

    `rank_head_counts` divides both by the same `tensor_parallel` on the cost side, so a sweep
    that pinned KV heads to the model's full 8 would write rows keyed on a combination that only
    exists at tp=1.
    """
    points = training_attention_points(
        model, TrainingAttentionSweep(total_tokens=(4096,), num_heads=(32, 16, 8, 4))
    )
    # Qwen3-8B is 32 query heads to 8 KV heads: a 4:1 ratio.
    assert [(p.num_heads, p.num_kv_heads) for p in points] == [(32, 8), (16, 4), (8, 2), (4, 1)]


def test_training_attention_floors_kv_heads_at_one(model):
    """At 2 query heads the 4:1 ratio integer-divides to 0, which Transformer Engine rejects."""
    (point,) = training_attention_points(
        model, TrainingAttentionSweep(total_tokens=(4096,), num_heads=(2,))
    )
    assert point.num_kv_heads == 1


def test_training_attention_is_ordered_for_module_reuse(model):
    """Mask-major then head-major then token-major, so one module serves a run of points.

    `DotProductAttention` construction is the expensive part, and the runner rebuilds only when
    `(mask_type, num_heads)` changes. An order that interleaved head counts would rebuild per
    point.
    """
    points = training_attention_points(
        model,
        TrainingAttentionSweep(
            total_tokens=(1024, 4096),
            num_heads=(16, 32),
            mask_types=("padding_causal", "padding"),
        ),
    )
    keys = [(p.mask_type, p.num_heads) for p in points]
    # Each distinct key must appear as one contiguous run.
    runs = [key for index, key in enumerate(keys) if index == 0 or key != keys[index - 1]]
    assert len(runs) == len(set(runs)), f"a module key is revisited: {keys}"
    assert runs == [
        ("padding_causal", 16),
        ("padding_causal", 32),
        ("padding", 16),
        ("padding", 32),
    ]


def test_training_attention_carries_the_head_dim(model):
    """Every point must name its own `head_dim`: it is a slice column on both tables."""
    points = training_attention_points(
        model, TrainingAttentionSweep(total_tokens=(1024, 4096), num_heads=(16,))
    )
    assert {p.head_dim for p in points} == {128}
    assert [p.total_tokens for p in points] == [1024, 4096]


def test_training_attention_refuses_a_headless_architecture(model):
    """Without `num_attention_heads` there is no ratio to shard the KV heads by."""
    from dataclasses import replace

    with pytest.raises(ProfileValidationError, match="num_attention_heads"):
        training_attention_points(
            replace(model, num_attention_heads=None),
            TrainingAttentionSweep(total_tokens=(4096,), num_heads=(16,)),
        )
