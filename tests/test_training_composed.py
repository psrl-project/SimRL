"""`ComposedTraining` sums kernel costs, so the tests check how the sum is assembled.

The arithmetic that matters is not in any single query but in the scaling between them:
pipeline stages, slot counts, and whether a per-layer multiplier gets the layer count once
or twice. A stub database makes each of those checkable without a measured table, by
returning a known constant per family so the composition's own structure is what the
assertions see.

`tests/test_training_composed_accuracy.py` scores the real tables against real step times;
this file is about structure and runs anywhere.
"""

from __future__ import annotations

import pytest

from srl.cost.kernel.database import (
    CommQuery,
    ElementwiseQuery,
    GemmQuery,
    OptimizerQuery,
    TrainingAttentionQuery,
)
from srl.cost.kernel.interp import KernelCoverageError
from srl.cost.training.base import TrainingParallelism, TrainingWorkload
from srl.cost.training.composed import ComposedTraining
from srl.cost.training.training_ops import TrainingOps
from srl.profiler.schema import ModelDescriptor, SequenceStats

_MODEL = ModelDescriptor(
    name="qwen3-8b-test",
    architecture="Qwen3ForCausalLM",
    num_layers=36,
    hidden_size=4096,
    num_attention_heads=32,
    num_query_groups=8,
    parameter_count=8_190_735_360,
    extra={"intermediate_size": 12288, "head_dim": 128, "qk_layernorm": True, "vocab_size": 151936},
)

# One second per call, per family. Unit costs make every assertion a count rather than a
# product, so a scaling error reads as the wrong integer instead of the wrong float.
_UNIT = 1.0


class _StubDatabase:
    """Return a fixed cost per family and record what was asked.

    Not a MagicMock: the point is to count queries per family, and a mock that answered
    every attribute would also answer a method `ComposedTraining` should not be calling.
    """

    def __init__(self, *, gemm: float = _UNIT, attention_forward: float = _UNIT,
                 attention_backward: float = _UNIT, elementwise: float = _UNIT,
                 comm: float = _UNIT, optimizer: float = _UNIT,
                 missing_logits_dgrad: bool = False) -> None:
        self._costs = {
            "gemm": gemm, "attention_forward": attention_forward,
            "attention_backward": attention_backward, "elementwise": elementwise,
            "comm": comm, "optimizer": optimizer,
        }
        self._missing_logits_dgrad = missing_logits_dgrad
        self.calls: list[tuple[str, object]] = []

    def query_gemm(self, query: GemmQuery) -> float:
        # The logits wgrad is the slice the measured table genuinely lacks; this models it
        # so the skip path is exercised rather than assumed.
        if self._missing_logits_dgrad and query.k > 100_000:
            raise KernelCoverageError("stub: no slice for the logits wgrad")
        self.calls.append(("gemm", query))
        return self._costs["gemm"]

    attention_cost_is_quadratic = False
    """Price attention as `(total_tokens / 4096)^2` instead of a unit cost.

    Off by default, because a unit cost makes every other assertion a count rather than a
    product. One test needs the quadratic shape: splitting a pack must *lower* the
    attention term, and three unit-cost calls are more than one, not less.
    """

    def query_training_attention(self, query: TrainingAttentionQuery) -> float:
        # Keyed on direction, because forward and backward read separate measured tables
        # and a single cost would hide a composition charging the wrong one.
        family = f"attention_{query.direction}"
        self.calls.append((family, query))
        if self.attention_cost_is_quadratic:
            return self._costs[family] * (query.total_tokens / 4096.0) ** 2
        return self._costs[family]

    def query_elementwise(self, query: ElementwiseQuery) -> float:
        self.calls.append(("elementwise", query))
        return self._costs["elementwise"]

    def query_comm(self, query: CommQuery) -> float:
        self.calls.append(("comm", query))
        return self._costs["comm"]

    def query_optimizer(self, query: OptimizerQuery) -> float:
        self.calls.append(("optimizer", query))
        return self._costs["optimizer"]

    def count(self, family: str) -> int:
        """Return how many queries of one family were issued."""
        return sum(1 for name, _ in self.calls if name == family)


_DB_KEYS = (
    "gemm", "attention_forward", "attention_backward", "elementwise", "comm", "optimizer",
    "missing_logits_dgrad",
)
_MODEL_KEYS = ("charge_optimizer",)


def _model(**kwargs: object) -> tuple[ComposedTraining, _StubDatabase]:
    db_kwargs = {k: v for k, v in kwargs.items() if k in _DB_KEYS}
    model_kwargs = {k: v for k, v in kwargs.items() if k in _MODEL_KEYS}
    ops_kwargs = {
        k: v for k, v in kwargs.items() if k not in _DB_KEYS and k not in _MODEL_KEYS
    }
    ops_kwargs.setdefault("tensor_parallel", 2)
    ops_kwargs.setdefault("recompute", "full")
    db = _StubDatabase(**db_kwargs)  # type: ignore[arg-type]
    ops = TrainingOps(_MODEL, **ops_kwargs)  # type: ignore[arg-type]
    return ComposedTraining(db, ops, **model_kwargs), db  # type: ignore[arg-type]


def _plan(*sum_tokens: int) -> tuple[SequenceStats, ...]:
    """Build a microbatch plan with the given per-microbatch token sums."""
    return tuple(SequenceStats.from_lengths((tokens,)) for tokens in sum_tokens)


def _token_counts(db: _StubDatabase) -> set[int]:
    """Return the token counts the model priced, read off the attention queries.

    The attention query's `total_tokens` is the unambiguous carrier: a GEMM's `m` is the
    token count for a forward or dgrad call but is `n_fwd` for a wgrad, so a set of `m`
    values cannot distinguish "priced 4096 tokens" from "priced a projection 4096 wide".
    """
    return {
        query.total_tokens
        for name, query in db.calls
        if isinstance(query, TrainingAttentionQuery)
    }


def test_the_prediction_equals_the_sum_of_its_decomposition():
    """The two must be one computation: the decomposition is what diagnoses the sum.

    Two parallel code paths summing the same kernels would be free to drift, and then the
    instrument you would reach for to find the drift is itself the thing that drifted.
    """
    model, _ = _model()
    plan = _plan(4096, 4096, 4096)
    parallelism = TrainingParallelism(tp=2, pp=3, dp=1)
    terms = model.decompose_microbatches(plan, parallelism)
    assert model.predict_microbatches(plan, parallelism) == pytest.approx(sum(terms.values()))


def test_each_pipeline_stage_is_charged_its_own_layers_not_the_whole_model():
    """At pp=3 a slot runs 12 of 36 layers, so the step is 36 layer-equivalents, not 108.

    This was a measured 3x over-prediction: `pipeline_slots` correctly returns `nmb+pp-1`,
    but the slot's unit is one stage's work, and pricing all 36 layers per slot charged the
    model `pp` times over.
    """
    plan = _plan(4096, 4096, 4096)
    model, _ = _model()
    at_pp3 = model.decompose_microbatches(plan, TrainingParallelism(tp=2, pp=3, dp=1))
    at_pp1 = model.decompose_microbatches(plan, TrainingParallelism(tp=2, pp=1, dp=1))

    # Per-layer work is per microbatch, not per slot (traced: kernel counts scale 4.00x
    # from nmb=1 to nmb=4). So 3 microbatches x 12 layers at pp=3 against 3 x 36 at pp=1 is
    # exactly 1/pp, and the bubble is a separate term rather than folded into these.
    for term in ("gemm_fwd", "gemm_bwd", "attn_fwd", "elementwise", "tp_collective"):
        assert at_pp3[term] / at_pp1[term] == pytest.approx(1.0 / 3.0), term
    # pp=1 has no boundary to wait at, so it has no bubble at all.
    assert at_pp1["bubble"] == 0.0
    assert at_pp3["bubble"] > 0.0


def test_the_bubble_is_a_named_term_costing_pp_minus_one_idle_slots():
    """`pp - 1` slots in which this stage has no microbatch, whatever the microbatch count.

    It is a separate term rather than folded into the per-layer costs because the two scale
    differently: compute goes as the microbatch count, the bubble as `pp - 1`. Tracing
    showed the bubble is not an abstraction — it appears on the GPU as
    `ncclDevKernel_SendRecv` blocking at the boundary, 2.859 s across 2 calls for a 48 MiB
    handoff, which is 35 MB/s on a 450 GB/s link.
    """
    model, _ = _model()
    parallelism = TrainingParallelism(tp=2, pp=3, dp=1)
    one_mb = model.decompose_microbatches(_plan(4096), parallelism)
    three_mb = model.decompose_microbatches(_plan(4096, 4096, 4096), parallelism)

    # Compute triples with the microbatch count.
    assert three_mb["gemm_fwd"] / one_mb["gemm_fwd"] == pytest.approx(3.0)
    # The bubble is `pp - 1 = 2` idle slots either way, so it does not.
    assert three_mb["bubble"] == pytest.approx(one_mb["bubble"])


def test_per_step_terms_are_not_multiplied_by_the_slot_count():
    """Logits and the optimizer run once per step; the per-layer terms run once per slot."""
    model, _ = _model()
    parallelism = TrainingParallelism(tp=2, pp=3, dp=1)
    one = model.decompose_microbatches(_plan(4096), parallelism)
    many = model.decompose_microbatches(_plan(4096, 4096, 4096, 4096), parallelism)
    assert one["optimizer"] == many["optimizer"]
    assert one["logits"] == many["logits"]
    assert many["gemm_fwd"] > one["gemm_fwd"]


def test_the_optimizer_term_is_omitted_when_the_caller_disables_it():
    """A configuration whose `optimizer_step` table is unswept should say so, not guess."""
    model, _ = _model(charge_optimizer=False)
    terms = model.decompose_microbatches(_plan(4096), TrainingParallelism(tp=2, pp=3, dp=1))
    assert terms["optimizer"] == 0.0


def test_attention_is_priced_per_sequence_while_the_gemms_see_the_whole_pack():
    """The two must differ, because only attention stops at a `cu_seqlens` boundary.

    A pack of three 4096-token sequences runs its GEMMs over 12288 rows but does three
    independent 4096-token attentions. Pricing attention at the pack total charged
    `(sum_i s_i)^2` where the kernel does `sum_i s_i^2`, which over-predicted packed records
    by the sequence count in the quadratic term.
    """
    model, db = _model()
    packed = (SequenceStats.from_lengths((4096, 4096, 4096)),)
    model.predict_microbatches(packed, TrainingParallelism(tp=2, pp=3, dp=1))

    assert _token_counts(db) == {4096}, "attention must see each sequence, not the pack"
    gemm_m = {query.m for name, query in db.calls if name == "gemm"}
    assert 12288 in gemm_m, "the GEMMs must see the whole pack"


def test_splitting_a_pack_lowers_the_attention_term_but_not_the_gemm_term():
    """Holding tokens fixed and splitting the pack is what the validation set measures.

    Five records hold 12288 tokens as (12288), (10240,2048), (6144,6144), (4096x3) and
    (3072x4), and measured step time falls monotonically across them. Only the attention
    term may move: every other term is linear in tokens, which is unchanged.
    """
    parallelism = TrainingParallelism(tp=2, pp=3, dp=1)
    # A unit-cost stub cannot express this: three calls at 1.0 each exceed one call at 1.0,
    # which is the arithmetic the real quadratic kernel inverts. So this one test uses a
    # cost quadratic in sequence length, which is what attention actually is.
    db = _StubDatabase()
    db.attention_cost_is_quadratic = True
    model = ComposedTraining(db, TrainingOps(_MODEL, tensor_parallel=2, recompute="full"))
    whole = model.decompose_microbatches(
        (SequenceStats.from_lengths((12288,)),), parallelism
    )
    split = model.decompose_microbatches(
        (SequenceStats.from_lengths((4096, 4096, 4096)),), parallelism
    )
    assert split["attn_fwd"] < whole["attn_fwd"]
    assert split["attn_bwd"] < whole["attn_bwd"]
    for term in ("gemm_fwd", "gemm_bwd", "elementwise", "tp_collective", "logits"):
        assert split[term] == pytest.approx(whole[term]), f"{term} moved with the packing"


def test_repeated_sequence_lengths_are_charged_once_with_a_multiplier():
    """Four 3072-token sequences are one table lookup times four, not four lookups.

    The table read is identical either way; collapsing them keeps the query count
    independent of the pack's sequence count, which matters for a 64-sequence pack.
    """
    ops = TrainingOps(_MODEL, tensor_parallel=2, recompute="full")
    queries = ops.attention_queries((3072,) * 4, direction="forward")
    assert len(queries) == 1
    query, repeats = queries[0]
    assert (query.total_tokens, repeats) == (3072, 4)


def test_a_packed_microbatch_is_priced_at_its_summed_tokens_not_its_longest_sequence():
    """Under THD packing one row holds several sequences and the GEMMs run over all of them.

    Pricing `max_tokens` understated the packed records by the packing factor: a microbatch
    of three 4096-token sequences is 12288 GEMM rows, and charging 4096 was a measured 3x
    miss.
    """
    model, db = _model()
    packed = (SequenceStats.from_lengths((4096, 4096, 4096)),)
    model.predict_microbatches(packed, TrainingParallelism(tp=2, pp=3, dp=1))
    # Asserted on the GEMM `m` values, since attention is now priced per sequence and so no
    # longer carries the pack total. 3072 and 75968 are the qkv `n` and the logits `n`.
    gemm_m = {query.m for name, query in db.calls if name == "gemm"}
    assert 12288 in gemm_m


def test_a_ragged_plan_is_priced_at_the_mean_rather_than_the_longest_microbatch():
    """Every microbatch occupies its own slot, so the step cost is the sum over them.

    `slots * mean` reproduces that sum, and is exact when the plan is uniform. `slots * max`
    would charge every slot for the longest microbatch, over-predicting a ragged RL batch by
    exactly its raggedness.
    """
    model, db = _model()
    model.predict_microbatches(
        _plan(2048, 4096, 6144), TrainingParallelism(tp=2, pp=1, dp=1)
    )
    assert _token_counts(db) == {4096}


def test_an_empty_plan_is_refused_rather_than_priced_as_zero():
    model, _ = _model()
    with pytest.raises(ValueError, match="at least one microbatch"):
        model.predict_microbatches((), TrainingParallelism(tp=2, pp=1, dp=1))


def test_a_parallelism_whose_tp_disagrees_with_the_ops_is_refused():
    """The ops were sharded at construction, so a mismatched tp prices unmeasured shapes."""
    model, _ = _model(tensor_parallel=2)
    with pytest.raises(ValueError, match="does not match ops.tensor_parallel"):
        model.predict_microbatches(_plan(4096), TrainingParallelism(tp=4, pp=1, dp=1))


def test_a_missing_logits_slice_is_skipped_rather_than_taking_a_neighbours_curve():
    """The measured table has no logits wgrad row, and the nearest slice is different work.

    Clamping onto it would produce a number that looks like a measurement, so the term is
    dropped and the omission is stated. The rest of the prediction must still resolve.
    """
    db = _StubDatabase(missing_logits_dgrad=True)
    ops = TrainingOps(_MODEL, tensor_parallel=2, recompute="full")
    model = ComposedTraining(db, ops)
    terms = model.decompose_microbatches(_plan(4096), TrainingParallelism(tp=2, pp=3, dp=1))
    assert terms["logits"] > 0.0, "the forward logits GEMM still resolves"
    assert terms["gemm_fwd"] > 0.0


def test_attention_reads_its_own_measured_table_in_each_direction():
    """Backward is a separate table, not a multiple of forward.

    The measured bwd/fwd ratio is 2.24-2.42 and drifts with sequence length, so no scalar
    is right; `ComposedTraining` previously used 1.0 and understated the term. With unit
    costs per direction, the forward is charged twice (recompute replays it) and the
    backward once, so `attn_fwd == 2 * attn_bwd` here. The point of the assertion is that
    the two come from *different* queries: a composition reading the forward table for both
    would still satisfy a ratio check but would price the wrong kernel.
    """
    parallelism = TrainingParallelism(tp=2, pp=3, dp=1)
    model, db = _model(attention_forward=1.0, attention_backward=4.0)
    terms = model.decompose_microbatches(_plan(4096), parallelism)

    directions = {
        query.direction for _, query in db.calls if isinstance(query, TrainingAttentionQuery)
    }
    assert directions == {"forward", "backward"}
    # 2 forward passes at 1.0 against 1 backward pass at 4.0.
    assert terms["attn_bwd"] == pytest.approx(2.0 * terms["attn_fwd"])


def test_the_elementwise_term_is_present_and_removable():
    """Summed for Megatron and deliberately omitted from `ComposedRollout`.

    The rollout utilities were fitted against a CUDA-graph-replayed server whose elementwise
    cost is already inside them (measured: adding it took decode 4.84% -> 11.18% MRE).
    Megatron runs no CUDA graphs here, so the cost is genuinely exposed.
    """
    model, _ = _model()
    terms = model.decompose_microbatches(_plan(4096), TrainingParallelism(tp=2, pp=3, dp=1))
    assert terms["elementwise"] > 0.0

    zeroed = ComposedTraining(
        _StubDatabase(elementwise=0.0),
        TrainingOps(_MODEL, tensor_parallel=2, recompute="full"),
    ).decompose_microbatches(_plan(4096), TrainingParallelism(tp=2, pp=3, dp=1))
    assert zeroed["elementwise"] == 0.0


def test_pipeline_sends_scale_with_microbatches_not_with_slots():
    """The bubble slots carry no microbatch, so they send nothing.

    Every per-layer term is charged once per slot including the bubble, because a stage is
    idle during the bubble but the slot is still on the critical path. A p2p send is
    different: it exists only because a microbatch crossed a boundary.
    """
    model, _ = _model()
    parallelism = TrainingParallelism(tp=2, pp=3, dp=1)
    one = model.decompose_microbatches(_plan(4096), parallelism)
    three = model.decompose_microbatches(_plan(4096, 4096, 4096), parallelism)
    # Both the sends and the per-layer work are per microbatch, so both triple. The bubble
    # is what does not, and `test_the_bubble_is_a_named_term...` covers that.
    assert three["pp_comm"] == pytest.approx(3.0 * one["pp_comm"])
    assert three["gemm_fwd"] == pytest.approx(3.0 * one["gemm_fwd"])


def test_a_single_stage_pipeline_charges_no_sends():
    model, _ = _model()
    terms = model.decompose_microbatches(_plan(4096), TrainingParallelism(tp=2, pp=1, dp=1))
    assert terms["pp_comm"] == 0.0


def test_tp1_contributes_no_collective_term():
    model, _ = _model(tensor_parallel=1)
    terms = model.decompose_microbatches(_plan(4096), TrainingParallelism(tp=1, pp=1, dp=1))
    assert terms["tp_collective"] == 0.0


def test_predict_plans_the_workload_and_agrees_with_the_planned_entry_point():
    """`predict` is the abstract interface; `predict_microbatches` is what validation uses."""
    model, _ = _model()
    workload = TrainingWorkload(
        sequence_lengths=(4096,) * 4, micro_batch_size=1, recompute="full"
    )
    parallelism = TrainingParallelism(tp=2, pp=3, dp=1)
    assert model.predict(workload, parallelism) == pytest.approx(
        model.predict_microbatches(_plan(4096, 4096, 4096, 4096), parallelism)
    )


def test_recompute_raises_the_prediction_without_touching_the_backward_gemms():
    """Full recompute replays the forward, so only forward-shaped terms move."""
    plan = _plan(4096)
    parallelism = TrainingParallelism(tp=2, pp=3, dp=1)
    full, _ = _model(recompute="full")
    none, _ = _model(recompute="none")
    with_replay = full.decompose_microbatches(plan, parallelism)
    without = none.decompose_microbatches(plan, parallelism)

    assert with_replay["gemm_fwd"] == pytest.approx(2.0 * without["gemm_fwd"])
    assert with_replay["attn_fwd"] == pytest.approx(2.0 * without["attn_fwd"])
    assert with_replay["gemm_bwd"] == pytest.approx(without["gemm_bwd"])
    assert with_replay["attn_bwd"] == pytest.approx(without["attn_bwd"])
    # The backward table is read once per layer whatever the recompute mode: the replay
    # re-runs the forward, not the backward.
    assert sum(with_replay.values()) > sum(without.values())
