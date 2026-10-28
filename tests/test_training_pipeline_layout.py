"""
Non-uniform pipeline layer splits, and cross-PP calibration borrowing, in the
training cost model.

Two independent things live here, mirroring `test_training_context_parallel.py`'s
split between "the formula generalizes" and "crossing an unmeasured axis needs an
opt-in":

1. `TrainingParallelism.pipeline_layout` lets a stage hold a different number of
   transformer layers than its neighbours (Megatron's own
   `--pipeline-model-parallel-layout`). The static-parameter, activation, and
   latency formulas already divide by `pp` uniformly; this generalizes that
   division to each stage's own share of the layers, unconditionally -- it is a
   formula correction, not a new measured regime, so it needs no opt-in, and
   `pipeline_layout=None` (the default) must stay byte-for-byte identical to
   before this existed.

2. Querying a `pp` degree the artifact never measured is a *different* gap, the
   same one `allow_context_parallel` already exists for: `HydraulisLatencyModel`
   already crosses it unconditionally (pooled per-layer fit, see
   `latency.py::latency_features`'s own docstring on why PP is "arithmetic rather
   than a cell to measure"). `HydraulisMemoryModel` does not pool PP, so it
   refuses by default and only crosses it with `allow_pp_extrapolation=True`,
   borrowing another PP's `activation_scale`/`residual_constant` for the same TP
   and flagging every such prediction with an extrapolation note.
"""

from __future__ import annotations

import pytest

from srl.cost.artifact import CostArtifact
from srl.cost.training import HydraulisTrainingCost, TrainingParallelism, TrainingWorkload
from srl.cost.training.activation import ActivationGeometry, activation_bytes
from srl.cost.training.base import stage_layer_counts
from srl.cost.training.inventory import heaviest_stage_parameters, parameter_inventory
from srl.cost.training.latency import latency_features
from srl.profiler.schema import (
    HardwareDescriptor,
    ModelDescriptor,
    Parallelism,
    ProfileRecord,
    RankMemory,
    RunDescriptor,
    SequenceStats,
    SoftwareDescriptor,
    TrainingPayload,
)

QWEN3_8B_PARAMS = 8_190_735_360


def _qwen3_8b_geometry(num_layers: int = 36) -> ActivationGeometry:
    return ActivationGeometry(num_layers=num_layers, hidden_size=4096, vocab_size=151936)


def _microbatches(seq_length: int, count: int) -> tuple[SequenceStats, ...]:
    return tuple(SequenceStats.from_lengths([seq_length]) for _ in range(count))


# --- TrainingParallelism.pipeline_layout validation -------------------------


def test_pipeline_layout_defaults_to_none():
    assert TrainingParallelism(tp=2, pp=5).pipeline_layout is None


def test_pipeline_layout_length_must_match_pp():
    with pytest.raises(ValueError, match="pipeline_layout"):
        TrainingParallelism(tp=2, pp=5, pipeline_layout=(8, 7, 7, 7))


def test_pipeline_layout_rejects_a_non_positive_stage():
    with pytest.raises(ValueError, match="at least one layer"):
        TrainingParallelism(tp=2, pp=3, pipeline_layout=(10, 0, 26))


def test_heaviest_stage_share_is_uniform_without_a_layout():
    assert TrainingParallelism(tp=1, pp=4).heaviest_stage_share == pytest.approx(0.25)


def test_heaviest_stage_share_reads_the_layout_s_own_largest_stage():
    # 36 layers over 5 stages: Qwen3-8B does not divide evenly, so the shape this
    # whole module exists for is [8, 7, 7, 7, 7] (stage 0 absorbs the remainder).
    parallelism = TrainingParallelism(tp=2, pp=5, pipeline_layout=(8, 7, 7, 7, 7))
    assert parallelism.heaviest_stage_share == pytest.approx(8 / 36)


# --- stage_layer_counts -------------------------------------------------


def test_stage_layer_counts_divides_evenly_without_a_layout():
    assert stage_layer_counts(36, TrainingParallelism(tp=1, pp=4)) == (9, 9, 9, 9)


def test_stage_layer_counts_rejects_indivisible_layers_without_a_layout():
    """This is the existing `heaviest_stage_parameters` refusal, unmoved."""
    with pytest.raises(ValueError, match="not divisible"):
        stage_layer_counts(36, TrainingParallelism(tp=1, pp=5))


def test_stage_layer_counts_returns_the_explicit_layout_verbatim():
    layout = (8, 7, 7, 7, 7)
    parallelism = TrainingParallelism(tp=2, pp=5, pipeline_layout=layout)
    assert stage_layer_counts(36, parallelism) == layout


def test_stage_layer_counts_rejects_a_layout_for_the_wrong_architecture():
    """A layout summing to the wrong total is a mismatched model, not a typo to paper over."""
    parallelism = TrainingParallelism(tp=2, pp=5, pipeline_layout=(8, 7, 7, 7, 7))
    with pytest.raises(ValueError, match="sums to 36"):
        stage_layer_counts(40, parallelism)


# --- heaviest_stage_parameters: non-uniform layouts -------------------------


class TestHeaviestStageParametersNonUniform:
    def test_uniform_layout_is_unaffected(self):
        """The existing PP=4 fact (`test_pipeline_stages_are_not_uniform`) must not move."""
        parallelism = TrainingParallelism(tp=1, pp=4, dp=2, cp=1)
        embedding = 4096 * 151936
        expected = (QWEN3_8B_PARAMS - 2 * embedding) / 4 + embedding
        assert heaviest_stage_parameters(
            parameter_count=QWEN3_8B_PARAMS,
            num_layers=36,
            hidden_size=4096,
            vocab_size=151936,
            parallelism=parallelism,
        ) == pytest.approx(expected)

    def test_rebalancing_the_ends_can_move_the_heaviest_stage_to_the_middle(self):
        """
        Giving each end stage fewer layers to make room for its embedding can push
        the heaviest stage into the middle, which a formula that only ever compares
        "uniform share + embedding" could not represent at all.
        """
        # 4 end stages of 2 layers (plus embedding) vs. one middle stage of 28 layers
        # (36 - 2*2 - 2*2 = 28): the middle stage's own 28-layer share dwarfs either
        # end's 2-layer-plus-embedding share for this geometry.
        parallelism = TrainingParallelism(
            tp=1, pp=5, dp=1, cp=1, pipeline_layout=(2, 2, 28, 2, 2)
        )
        embedding = 4096 * 151936
        layer_parameters = QWEN3_8B_PARAMS - 2 * embedding
        per_layer = layer_parameters / 36
        middle_stage_total = per_layer * 28
        end_stage_total = per_layer * 2 + embedding
        assert middle_stage_total > end_stage_total
        assert heaviest_stage_parameters(
            parameter_count=QWEN3_8B_PARAMS,
            num_layers=36,
            hidden_size=4096,
            vocab_size=151936,
            parallelism=parallelism,
        ) == pytest.approx(middle_stage_total)

    def test_the_qwen3_8b_pp5_shape_uses_the_first_stage_remainder(self):
        """36 does not divide by 5; the shape this module exists for is [8,7,7,7,7]."""
        parallelism = TrainingParallelism(
            tp=2, pp=5, dp=1, cp=1, pipeline_layout=(8, 7, 7, 7, 7)
        )
        embedding = 4096 * 151936
        layer_parameters = QWEN3_8B_PARAMS - 2 * embedding
        per_layer = layer_parameters / 36
        expected = (per_layer * 8 + embedding) / 2
        assert heaviest_stage_parameters(
            parameter_count=QWEN3_8B_PARAMS,
            num_layers=36,
            hidden_size=4096,
            vocab_size=151936,
            parallelism=parallelism,
        ) == pytest.approx(expected)


# --- activation_bytes: non-uniform layouts ----------------------------------


class TestActivationBytesNonUniform:
    def test_uniform_layout_is_unaffected(self):
        """The existing PP=4 peak-stage fact must not move."""
        geometry = _qwen3_8b_geometry()
        parallelism = TrainingParallelism(tp=1, pp=4, dp=2, cp=1)
        with_layout = TrainingParallelism(
            tp=1, pp=4, dp=2, cp=1, pipeline_layout=(9, 9, 9, 9)
        )
        plain = activation_bytes(geometry, _microbatches(512, 4), parallelism)
        explicit = activation_bytes(geometry, _microbatches(512, 4), with_layout)
        assert explicit == pytest.approx(plain)

    def test_a_heavy_middle_stage_can_exceed_both_ends(self):
        """
        A middle stage given most of the layers must be able to dominate the two
        ends the uniform formula only ever compared against each other.
        """
        geometry = _qwen3_8b_geometry()
        # Stage 2 (middle, index 2 of 5) gets 28 of 36 layers.
        parallelism = TrainingParallelism(
            tp=1, pp=5, dp=1, cp=1, pipeline_layout=(2, 2, 28, 2, 2)
        )
        microbatches = _microbatches(512, 5)
        peak = activation_bytes(geometry, microbatches, parallelism)
        # The middle stage's own share, at its own 1F1B in-flight depth (pp - 2 = 3).
        per_token_all_layers = geometry.checkpoint_bytes_per_token(1)
        middle_share = per_token_all_layers * (28 / 36) * 512
        middle_concurrency = min(len(microbatches), 5 - 2)
        assert peak == pytest.approx(middle_share * middle_concurrency)

    def test_the_qwen3_8b_pp5_shape_matches_the_first_stage_warmup_path(self):
        """[8,7,7,7,7]: the warmup-heavy first stage, not the logit-heavy last one, wins."""
        geometry = _qwen3_8b_geometry()
        parallelism = TrainingParallelism(
            tp=2, pp=5, dp=1, cp=1, pipeline_layout=(8, 7, 7, 7, 7)
        )
        microbatches = _microbatches(512, 8)
        peak = activation_bytes(geometry, microbatches, parallelism)
        per_token_all_layers = geometry.checkpoint_bytes_per_token(2)
        first_stage = per_token_all_layers * (8 / 36) * 512 * min(len(microbatches), 5)
        last_stage = per_token_all_layers * (7 / 36) * 512 * 1 + geometry.logit_bytes_per_token() * 512
        assert peak == pytest.approx(max(first_stage, last_stage))


# --- latency_features: non-uniform layouts ----------------------------------


class TestLatencyFeaturesNonUniform:
    def test_uniform_layout_is_unaffected(self):
        parallelism = TrainingParallelism(tp=1, pp=4, dp=1, cp=1)
        with_layout = TrainingParallelism(
            tp=1, pp=4, dp=1, cp=1, pipeline_layout=(9, 9, 9, 9)
        )
        microbatches = _microbatches(512, 4)
        plain = latency_features(microbatches, parallelism)
        explicit = latency_features(microbatches, with_layout)
        assert explicit.weighted_tokens == pytest.approx(plain.weighted_tokens)
        assert explicit.weighted_slots == pytest.approx(plain.weighted_slots)

    def test_the_heaviest_stage_sets_the_share_not_the_pipeline_degree(self):
        """
        At [8,7,7,7,7] the bottleneck stage holds 8/36, not the uniform 1/5 -- the
        stage that sets the step's critical path is this bigger share.
        """
        parallelism = TrainingParallelism(
            tp=2, pp=5, dp=1, cp=1, pipeline_layout=(8, 7, 7, 7, 7)
        )
        microbatches = _microbatches(512, 8)
        features = latency_features(microbatches, parallelism)
        uniform_parallelism = TrainingParallelism(tp=2, pp=5, dp=1, cp=1)
        uniform_features = latency_features(microbatches, uniform_parallelism)
        # 8/36 > 1/5, so the non-uniform shape must cost strictly more.
        assert features.weighted_tokens > uniform_features.weighted_tokens


# --- HydraulisMemoryModel: crossing an unmeasured PP needs an opt-in --------


LATENCY_COEFFICIENTS = (0.010, 1e-7, 0.002, 1e-11)
ACTIVATION_SCALE = 1.0
MEMORY_RESIDUAL = 5_000_000.0
PARAMETER_COUNT = 10_000_000
# Divisible by both the measured pp=2 and the extrapolated-to pp=5, so the
# uniform (no `pipeline_layout`) path is exercised at both without tripping the
# unrelated "not divisible" refusal `stage_layer_counts` raises for that case.
NUM_LAYERS = 10
HIDDEN_SIZE = 1024
VOCAB_SIZE = 2_048


def _static_bytes(parallelism: TrainingParallelism) -> float:
    return parameter_inventory(
        parameter_count=PARAMETER_COUNT,
        parallelism=parallelism,
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        distributed_optimizer=True,
        local_parameters=heaviest_stage_parameters(
            parameter_count=PARAMETER_COUNT,
            num_layers=NUM_LAYERS,
            hidden_size=HIDDEN_SIZE,
            vocab_size=VOCAB_SIZE,
            parallelism=parallelism,
        ),
    ).total_bytes


def _geometry() -> ActivationGeometry:
    return ActivationGeometry(num_layers=NUM_LAYERS, hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE)


def _fit_pp2_artifact() -> CostArtifact:
    """Fit a tp=1/pp=2 cell only -- pp=5 is never measured, by construction."""
    model = ModelDescriptor(
        name="synthetic-pp-fixture",
        parameter_count=PARAMETER_COUNT,
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=16,
        num_query_groups=4,
        extra={"vocab_size": VOCAB_SIZE},
    )
    software = SoftwareDescriptor(backend="megatron_bridge", backend_version="0.4.2")
    hardware = HardwareDescriptor(gpu_name="H100", gpu_count=2, gpu_memory_mib=81920)
    parallelism = Parallelism(tp=1, pp=2, dp=1, cp=1)
    training_parallelism = TrainingParallelism(tp=1, pp=2, dp=1, cp=1)
    static = _static_bytes(training_parallelism)
    per_step, token_cost, microbatch_overhead, quadratic_cost = LATENCY_COEFFICIENTS
    records: list[ProfileRecord] = []
    for sequence_length in (384, 512, 768, 1536):
        for num_microbatches in (1, 4):
            microbatches = tuple(
                SequenceStats.from_lengths([sequence_length]) for _ in range(num_microbatches)
            )
            bubble = training_parallelism.pp - 1
            max_tokens = sequence_length
            max_sq = sequence_length * sequence_length
            weighted_tokens = (num_microbatches * max_tokens + bubble * max_tokens) / 2
            weighted_sq = (num_microbatches * max_sq + bubble * max_sq) / 2
            weighted_slots = (num_microbatches + bubble) / 2
            latency = (
                per_step
                + token_cost * weighted_tokens
                + quadratic_cost * weighted_sq
                + microbatch_overhead * weighted_slots
            )
            activation_feature = activation_bytes(_geometry(), microbatches, training_parallelism)
            peak = int(static + ACTIVATION_SCALE * activation_feature + MEMORY_RESIDUAL)
            payload = TrainingPayload(
                micro_batch_size=1,
                global_batch_size=num_microbatches,
                num_microbatches=num_microbatches,
                schedule="1f1b",
                recompute="none",
                optimizer="adam",
                microbatches=microbatches,
                step_times_s=(latency, latency),
                rank_memory=(RankMemory(rank=0, allocated_bytes=peak, reserved_bytes=peak + 1024),),
                peak_allocated_bytes=peak,
                peak_reserved_bytes=peak + 1024,
                parameter_count=PARAMETER_COUNT,
            )
            records.append(
                ProfileRecord(
                    record_type="training",
                    run=RunDescriptor(run_id=f"seq{sequence_length}-nmb{num_microbatches}"),
                    model=model,
                    hardware=hardware,
                    software=software,
                    parallelism=parallelism,
                    payload=payload,
                )
            )
    return HydraulisTrainingCost.fit(records, validation_fraction=0.0)


@pytest.fixture(scope="module")
def artifact() -> CostArtifact:
    return _fit_pp2_artifact()


@pytest.fixture()
def workload() -> TrainingWorkload:
    return TrainingWorkload(sequence_lengths=(1536,) * 4, micro_batch_size=1)


class TestMemoryRefusedByDefault:
    def test_memory_model_refuses_an_unmeasured_pp_without_opt_in(self, artifact, workload):
        model = HydraulisTrainingCost.from_artifact(artifact)
        with pytest.raises(KeyError):
            model.memory_model.predict(workload, TrainingParallelism(tp=1, pp=5, dp=1, cp=1))

    def test_measured_pp_never_raises(self, artifact, workload):
        model = HydraulisTrainingCost.from_artifact(artifact)
        model.estimate(workload, TrainingParallelism(tp=1, pp=2, dp=1, cp=1))  # must not raise


class TestMeasuredPpIsUnaffectedByTheOptIn:
    def test_memory_prediction_at_the_measured_pp_is_unchanged(self, artifact, workload):
        strict = HydraulisTrainingCost.from_artifact(artifact)
        permissive = HydraulisTrainingCost.from_artifact(artifact, allow_pp_extrapolation=True)
        parallelism = TrainingParallelism(tp=1, pp=2, dp=1, cp=1)
        assert strict.memory_model.predict(workload, parallelism) == permissive.memory_model.predict(
            workload, parallelism
        )

    def test_measured_pp_adds_no_extrapolation_note(self, artifact, workload):
        model = HydraulisTrainingCost.from_artifact(artifact, allow_pp_extrapolation=True)
        _, notes = model.memory_model.predict_with_notes(
            workload, TrainingParallelism(tp=1, pp=2, dp=1, cp=1)
        )
        assert not any("never been measured at pp" in note for note in notes)


class TestMemoryPpExtrapolation:
    def test_opt_in_borrows_the_same_tp_s_calibration_and_flags_the_prediction(
        self, artifact, workload
    ):
        model = HydraulisTrainingCost.from_artifact(artifact, allow_pp_extrapolation=True)
        memory, notes = model.memory_model.predict_with_notes(
            workload, TrainingParallelism(tp=1, pp=5, dp=1, cp=1)
        )
        assert memory > 0
        assert any("never been measured at pp" in note for note in notes)

    def test_non_uniform_layout_at_the_unmeasured_pp_also_works(self, artifact, workload):
        """
        The two features (layout and PP-extrapolation) must compose.

        A 4-layer first stage against 1-layer end stages pushes `activation_bytes`
        outside the pp=2 fixture's own fitted range, which is the separate
        `extrapolate` feature-range gate correctly doing its job -- a real query
        this far from anything measured needs both escape hatches, not evidence
        that PP-extrapolation alone is broken.
        """
        model = HydraulisTrainingCost.from_artifact(
            artifact, allow_pp_extrapolation=True, extrapolate=["activation_bytes"]
        )
        memory, notes = model.memory_model.predict_with_notes(
            workload,
            TrainingParallelism(tp=1, pp=5, dp=1, cp=1, pipeline_layout=(4, 2, 2, 1, 1)),
        )
        assert memory > 0
        assert any("never been measured at pp" in note for note in notes)
        assert any("extrapolated" in note for note in notes)
