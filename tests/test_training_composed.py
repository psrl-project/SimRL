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

from dryrun.cost.kernel.database import (
    AttentionQuery,
    CommQuery,
    ElementwiseQuery,
    GemmQuery,
    OptimizerQuery,
)
from dryrun.cost.kernel.interp import KernelCoverageError
from dryrun.cost.training.base import TrainingParallelism, TrainingWorkload
from dryrun.cost.training.composed import ComposedTraining
from dryrun.cost.training.training_ops import TrainingOps
from dryrun.profiler.schema import ModelDescriptor, SequenceStats

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

    def __init__(self, *, gemm: float = _UNIT, attention: float = _UNIT,
                 elementwise: float = _UNIT, comm: float = _UNIT,
                 optimizer: float = _UNIT, missing_logits_dgrad: bool = False) -> None:
        self._costs = {
            "gemm": gemm, "attention": attention, "elementwise": elementwise,
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

    def query_attention(self, query: AttentionQuery) -> float:
        self.calls.append(("attention", query))
        return self._costs["attention"]

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


_DB_KEYS = ("gemm", "attention", "elementwise", "comm", "optimizer", "missing_logits_dgrad")
_MODEL_KEYS = ("attn_bwd_fwd_ratio", "charge_optimizer")


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

    The attention query's `seq_len` is the unambiguous carrier: a GEMM's `m` is the token
    count for a forward or dgrad call but is `n_fwd` for a wgrad, so a set of `m` values
    cannot distinguish "priced 4096 tokens" from "priced a projection 4096 wide".
    """
    return {
        query.seq_len
        for name, query in db.calls
        if name == "attention" and isinstance(query, AttentionQuery)
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

    # pp=1: 3 slots x 36 layers = 108 layer-slots.
    # pp=3: 5 slots (3+3-1) x 12 layers = 60 layer-slots. Ratio 60/108.
    for term in ("gemm_fwd", "gemm_bwd", "attn_fwd", "elementwise", "tp_collective"):
        assert at_pp3[term] / at_pp1[term] == pytest.approx(60.0 / 108.0), term


def test_the_pipeline_bubble_appears_as_slots_exceeding_microbatches():
    """At pp=3 with 3 microbatches the critical path is 5 slots, not 3.

    The bubble is not a separate additive term here: it falls out of the slot count, which
    is why there is no `bubble` key to drift from the terms it is derived from.
    """
    model, _ = _model()
    one_mb = model.decompose_microbatches(_plan(4096), TrainingParallelism(tp=2, pp=3, dp=1))
    three_mb = model.decompose_microbatches(
        _plan(4096, 4096, 4096), TrainingParallelism(tp=2, pp=3, dp=1)
    )
    # 1 microbatch -> 1+3-1 = 3 slots; 3 microbatches -> 3+3-1 = 5 slots.
    assert three_mb["gemm_fwd"] / one_mb["gemm_fwd"] == pytest.approx(5.0 / 3.0)


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


def test_a_packed_microbatch_is_priced_at_its_summed_tokens_not_its_longest_sequence():
    """Under THD packing one row holds several sequences and the GEMMs run over all of them.

    Pricing `max_tokens` understated the packed records by the packing factor: a microbatch
    of three 4096-token sequences is 12288 GEMM rows, and charging 4096 was a measured 3x
    miss.
    """
    model, db = _model()
    packed = (SequenceStats.from_lengths((4096, 4096, 4096)),)
    model.predict_microbatches(packed, TrainingParallelism(tp=2, pp=3, dp=1))
    # The attention query's seq_len is the one unambiguous carrier of the token count: a
    # GEMM's `m` is also the wgrad's `n_fwd`, so asserting on it would match by accident.
    assert _token_counts(db) == {12288}


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


def test_attention_backward_is_charged_as_a_multiple_of_the_measured_forward():
    """Measured 0.83-0.99x for cuDNN FusedAttention on H20, so the default is 1.0.

    A zero ratio must remove the term entirely rather than falling back to a default, since
    that is how a caller states that backward is unmeasured.
    """
    parallelism = TrainingParallelism(tp=2, pp=3, dp=1)
    plan = _plan(4096)
    default = ComposedTraining(
        _StubDatabase(), TrainingOps(_MODEL, tensor_parallel=2, recompute="full")
    ).decompose_microbatches(plan, parallelism)
    assert default["attn_bwd"] == pytest.approx(default["attn_fwd"] / 2.0), (
        "forward is charged twice under full recompute, backward once"
    )

    omitted = ComposedTraining(
        _StubDatabase(),
        TrainingOps(_MODEL, tensor_parallel=2, recompute="full"),
        attn_bwd_fwd_ratio=0.0,
    ).decompose_microbatches(plan, parallelism)
    assert omitted["attn_bwd"] == 0.0


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
    assert sum(with_replay.values()) > sum(without.values())
