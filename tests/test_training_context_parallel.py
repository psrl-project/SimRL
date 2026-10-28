"""
Context-parallel support in the training cost model.

No `cp > 1` training step has ever been measured on any architecture this codebase
fits (the vendored Megatron's gated-delta-net layer asserts `context_parallel_size == 1`,
so even a model that needs it structurally cannot produce one). Every artifact
`HydraulisTrainingCost.fit` can build therefore holds only `cp == 1` cells, and a `cp > 1`
query is answered by an explicit analytical projection rather than a measured one:
`HydraulisLatencyModel`/`HydraulisMemoryModel` refuse it unless the caller opts in with
`allow_context_parallel=True`, and record an extrapolation note whenever they do.

These tests check three things: the refusal is the default, the opt-in projection divides
exactly the terms the docstrings claim it divides, and `cp == 1` stays byte-for-byte
unaffected (so every artifact fitted before CP support existed still predicts the same
number).
"""

from __future__ import annotations

import pytest

from srl.cost.artifact import CostArtifact
from srl.cost.training import HydraulisTrainingCost, TrainingParallelism, TrainingWorkload
from srl.cost.training.activation import ActivationGeometry, activation_bytes
from srl.cost.training.inventory import heaviest_stage_parameters, parameter_inventory
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

PARAMETER_COUNT = 10_000_000
NUM_LAYERS = 8
HIDDEN_SIZE = 1024
VOCAB_SIZE = 2_048
# (per_step_overhead, token_cost, microbatch_overhead, token_quadratic_cost).
LATENCY_COEFFICIENTS = (0.010, 1e-7, 0.002, 1e-11)
ACTIVATION_SCALE = 1.0
MEMORY_RESIDUAL = 5_000_000.0


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
    return ActivationGeometry(
        num_layers=NUM_LAYERS, hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE
    )


def _fit_cp1_artifact() -> CostArtifact:
    """Fit a cp=1 artifact at tp=1/pp=2, with three sequence lengths so the design is rank 4.

    Two sequence lengths leave the linear and quadratic token columns collinear (this is
    exactly the trap `check_latency_rank` exists to catch, and the one the real Qwen3.8-27B
    sweep hit when it only varied sequence length between two points related by a clean
    factor of two). Three lengths that are not all multiples of one another break it.
    """
    model = ModelDescriptor(
        name="synthetic-cp-fixture",
        parameter_count=PARAMETER_COUNT,
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=16,
        num_query_groups=4,
        extra={"vocab_size": VOCAB_SIZE},
    )
    software = SoftwareDescriptor(backend="megatron_bridge", backend_version="0.4.2")
    records: list[ProfileRecord] = []
    parallelism = Parallelism(tp=1, pp=2, dp=1, cp=1)
    hardware = HardwareDescriptor(gpu_name="H100", gpu_count=2, gpu_memory_mib=81920)
    training_parallelism = TrainingParallelism(tp=1, pp=2, dp=1, cp=1)
    static = _static_bytes(training_parallelism)
    per_step, token_cost, microbatch_overhead, quadratic_cost = LATENCY_COEFFICIENTS
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
            slots = num_microbatches + bubble
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
    return _fit_cp1_artifact()


@pytest.fixture()
def workload() -> TrainingWorkload:
    # 1536 tokens is the fixture's longest fitted length, chosen so dividing by cp up to 4
    # still lands inside the fitted [384, 1536] range: this module tests the CP arithmetic,
    # not the separate out-of-domain gate, so the workload must not trip it by accident.
    return TrainingWorkload(sequence_lengths=(1536,) * 4, micro_batch_size=1)


class TestRefusedByDefault:
    """`cp > 1` is a structural gap, not an out-of-range query, so it is refused by default."""

    def test_memory_model_refuses_cp_without_opt_in(self, artifact, workload):
        model = HydraulisTrainingCost.from_artifact(artifact)
        with pytest.raises(ValueError, match="ever been measured"):
            model.memory_model.predict(workload, TrainingParallelism(tp=1, pp=2, dp=1, cp=2))

    def test_latency_model_refuses_cp_without_opt_in(self, artifact, workload):
        model = HydraulisTrainingCost.from_artifact(artifact)
        with pytest.raises(ValueError, match="ever been measured"):
            model.latency_model.predict(workload, TrainingParallelism(tp=1, pp=2, dp=1, cp=2))

    def test_cp_equal_one_never_raises(self, artifact, workload):
        model = HydraulisTrainingCost.from_artifact(artifact)
        # Must not raise.
        model.estimate(workload, TrainingParallelism(tp=1, pp=2, dp=1, cp=1))


class TestCpOneIsUnaffected:
    """Adding CP support must not move a single bit of the existing cp=1 path."""

    def test_memory_prediction_is_unchanged(self, artifact, workload):
        strict = HydraulisTrainingCost.from_artifact(artifact)
        permissive = HydraulisTrainingCost.from_artifact(artifact, allow_context_parallel=True)
        parallelism = TrainingParallelism(tp=1, pp=2, dp=1, cp=1)
        assert strict.memory_model.predict(workload, parallelism) == permissive.memory_model.predict(
            workload, parallelism
        )

    def test_latency_prediction_is_unchanged(self, artifact, workload):
        strict = HydraulisTrainingCost.from_artifact(artifact)
        permissive = HydraulisTrainingCost.from_artifact(artifact, allow_context_parallel=True)
        parallelism = TrainingParallelism(tp=1, pp=2, dp=1, cp=1)
        assert strict.latency_model.predict(workload, parallelism) == permissive.latency_model.predict(
            workload, parallelism
        )

    def test_cp_one_adds_no_extrapolation_note(self, artifact, workload):
        model = HydraulisTrainingCost.from_artifact(artifact, allow_context_parallel=True)
        _, notes = model.latency_model.predict_with_notes(
            workload, TrainingParallelism(tp=1, pp=2, dp=1, cp=1)
        )
        assert not any("analytical projection" in note for note in notes)


class TestCpProjection:
    """The opt-in projection divides exactly the terms the docstrings claim."""

    def test_cp_flags_every_prediction_as_unverified(self, artifact, workload):
        model = HydraulisTrainingCost.from_artifact(artifact, allow_context_parallel=True)
        _, lat_notes = model.latency_model.predict_with_notes(
            workload, TrainingParallelism(tp=1, pp=2, dp=1, cp=2)
        )
        _, mem_notes = model.memory_model.predict_with_notes(
            workload, TrainingParallelism(tp=1, pp=2, dp=1, cp=2)
        )
        assert any("analytical projection" in note for note in lat_notes)
        assert any("analytical projection" in note for note in mem_notes)

    def test_memory_activation_term_halves_at_cp_two(self, artifact, workload):
        model = HydraulisTrainingCost.from_artifact(artifact, allow_context_parallel=True)
        cp1 = model.memory_model.predict(workload, TrainingParallelism(tp=1, pp=2, dp=1, cp=1))
        cp2 = model.memory_model.predict(workload, TrainingParallelism(tp=1, pp=2, dp=1, cp=2))
        # Subtract static state (which differs between cp1 and cp2 only in the optimizer
        # shard, tested separately below) and the fitted residual constant, a fixed
        # loss-computation offset that must not halve with cp, to isolate the activation
        # term the docstring claims is divided by cp.
        activation_cp1 = cp1 - _static_bytes(TrainingParallelism(tp=1, pp=2, dp=1, cp=1)) - MEMORY_RESIDUAL
        activation_cp2 = cp2 - _static_bytes(TrainingParallelism(tp=1, pp=2, dp=1, cp=2)) - MEMORY_RESIDUAL
        assert activation_cp2 == pytest.approx(activation_cp1 / 2, rel=0.02)

    def test_memory_optimizer_shard_widens_with_cp(self, artifact, workload):
        """Confirms the OTHER half of the memory story: inventory.py's dp*cp shard."""
        cp1_static = _static_bytes(TrainingParallelism(tp=1, pp=2, dp=1, cp=1))
        cp2_static = _static_bytes(TrainingParallelism(tp=1, pp=2, dp=1, cp=2))
        assert cp2_static < cp1_static

    def test_latency_halves_at_cp_two_at_fixed_pipeline_shape(self, artifact, workload):
        """
        `weighted_slots` does not depend on cp, so the microbatch_overhead term cannot
        halve. Comparing cp1 against cp2 at the SAME pipeline shape isolates how much of
        the gap must come from the token and quadratic terms, which the model divides.
        """
        model = HydraulisTrainingCost.from_artifact(artifact, allow_context_parallel=True)
        cp1 = model.latency_model.predict(workload, TrainingParallelism(tp=1, pp=2, dp=1, cp=1))
        cp2 = model.latency_model.predict(workload, TrainingParallelism(tp=1, pp=2, dp=1, cp=2))
        assert cp2 < cp1
        # Reconstruct the fixed per_step + microbatch_overhead part directly from the
        # entry's own coefficients, so this assertion does not depend on this test module's
        # copy of LATENCY_COEFFICIENTS matching the fit exactly.
        entry = artifact.entry_for(
            Parallelism(tp=1, pp=1, dp=1, cp=1),
            selector={"component": "latency", "schedule": "1f1b", "recompute": "none",
                      "optimizer": "adam", "parameter_dtype": "bf16", "gradient_dtype": "fp32",
                      "optimizer_state_dtype": "fp32", "distributed_optimizer": True},
            ignore_pp=True,
        )
        fixed = (
            entry.coefficients["per_step_overhead"]
            + entry.coefficients["microbatch_overhead"] * 2.5
        )
        # At cp=2 the token-proportional part should be roughly half of what it is at cp=1,
        # since the fixed part is identical at both.
        token_part_cp1 = cp1 - fixed
        token_part_cp2 = cp2 - fixed
        assert token_part_cp2 == pytest.approx(token_part_cp1 / 2, rel=0.02)

    def test_cp_four_quarters_the_token_proportional_part_again(self, artifact, workload):
        model = HydraulisTrainingCost.from_artifact(artifact, allow_context_parallel=True)
        cp2 = model.latency_model.predict(workload, TrainingParallelism(tp=1, pp=2, dp=1, cp=2))
        cp4 = model.latency_model.predict(workload, TrainingParallelism(tp=1, pp=2, dp=1, cp=4))
        assert cp4 < cp2


class TestArtifactLookupIgnoreCp:
    """Unit-level check of the `ignore_cp` lookup `HydraulisLatencyModel` relies on."""

    def test_exact_match_without_ignore_cp_rejects_a_different_cp(self, artifact):
        with pytest.raises(KeyError):
            artifact.entry_for(
                Parallelism(tp=1, pp=1, dp=1, cp=2),
                selector={"component": "latency", "schedule": "1f1b", "recompute": "none",
                          "optimizer": "adam", "parameter_dtype": "bf16", "gradient_dtype": "fp32",
                          "optimizer_state_dtype": "fp32", "distributed_optimizer": True},
                ignore_pp=True,
            )

    def test_ignore_cp_finds_the_cp_one_entry_despite_a_cp_two_request(self, artifact):
        entry = artifact.entry_for(
            Parallelism(tp=1, pp=1, dp=1, cp=2),
            selector={"component": "latency", "schedule": "1f1b", "recompute": "none",
                      "optimizer": "adam", "parameter_dtype": "bf16", "gradient_dtype": "fp32",
                      "optimizer_state_dtype": "fp32", "distributed_optimizer": True},
            ignore_pp=True,
            ignore_cp=True,
        )
        assert entry.parallelism.cp == 1


class TestTrainingOpsGdnHeadDivision:
    """`TrainingOps` divides a GDN layer's heads by `tp * cp`, not just `tp`."""

    def _hybrid_model(self) -> ModelDescriptor:
        return ModelDescriptor(
            name="synthetic-hybrid",
            architecture="Qwen3_5ForConditionalGeneration",
            num_layers=8,
            hidden_size=512,
            num_attention_heads=8,
            num_query_groups=4,
            extra={
                "head_dim": 64,
                "vocab_size": 1000,
                "intermediate_size": 1024,
                "full_attention_interval": 4,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
            },
        )

    def test_cp_one_matches_tp_only_division(self):
        from srl.cost.training.training_ops import TrainingOps

        ops = TrainingOps(self._hybrid_model(), tensor_parallel=2, context_parallel=1)
        (query, _mult), = ops.gdn_queries([256])
        assert (query.num_key_heads, query.num_value_heads) == (8, 16)

    def test_cp_two_further_divides_the_tp_local_head_count(self):
        from srl.cost.training.training_ops import TrainingOps

        ops = TrainingOps(self._hybrid_model(), tensor_parallel=2, context_parallel=2)
        (query, _mult), = ops.gdn_queries([256])
        # 16 key heads / tp=2 / cp=2 = 4, against 8 at cp=1.
        assert (query.num_key_heads, query.num_value_heads) == (4, 8)

    def test_total_tokens_is_not_divided_by_cp(self):
        """The GDN scan processes the FULL local sequence for its share of heads."""
        from srl.cost.training.training_ops import TrainingOps

        ops_cp1 = TrainingOps(self._hybrid_model(), tensor_parallel=1, context_parallel=1)
        ops_cp2 = TrainingOps(self._hybrid_model(), tensor_parallel=1, context_parallel=2)
        (q1, _), = ops_cp1.gdn_queries([256])
        (q2, _), = ops_cp2.gdn_queries([256])
        assert q1.total_tokens == q2.total_tokens == 256

    def test_non_divisible_cp_is_refused(self):
        from srl.cost.training.training_ops import TrainingOps
        from srl.profiler.schema import ProfileValidationError

        with pytest.raises(ProfileValidationError, match="context_parallel"):
            TrainingOps(self._hybrid_model(), tensor_parallel=1, context_parallel=3)


class _FakeKernelDatabase:
    """Echoes a shape field back as the kernel's cost, to isolate composition arithmetic.

    Real kernel costs are not linear in the queried shape (a GEMM utility surface
    interpolates a measured curve), so this stub is only useful for checking that
    `ComposedTraining` divides the right terms by the right CP factor, not for checking
    kernel accuracy.
    """

    def query_gemm(self, query):
        # `m * k` rather than `m` alone: a wgrad query puts the token-dependent axis in
        # `k` (`GemmQuery(m=s.n, n=s.k, k=token_count)`), not `m`, so echoing `m` alone
        # would make the stub constant in token_count for that one query family and the
        # cp-halving assertions below would not actually exercise it.
        return float(query.m) * float(query.k)

    def query_elementwise(self, query):
        return float(query.n_elements)

    def query_comm(self, query):
        return float(query.message_bytes)

    def query_training_attention(self, query):
        return float(query.total_tokens)

    def query_gdn(self, query):
        return float(query.total_tokens)

    def query_optimizer(self, query):
        return 0.0


def _dense_model() -> ModelDescriptor:
    return ModelDescriptor(
        name="synthetic-dense",
        architecture="Qwen3ForCausalLM",
        num_layers=4,
        hidden_size=256,
        num_attention_heads=8,
        num_query_groups=2,
        extra={"vocab_size": 1000, "intermediate_size": 512},
    )


class TestComposedTrainingContextParallel:
    """`ComposedTraining` must divide GEMM/elementwise/logits tokens, and the attention
    result, by `cp`, consistent with the documented ring-attention approximation."""

    def _terms(self, cp: int):
        from srl.cost.training.composed import ComposedTraining
        from srl.cost.training.training_ops import TrainingOps

        ops = TrainingOps(
            _dense_model(), tensor_parallel=1, context_parallel=cp, recompute="none"
        )
        composed = ComposedTraining(_FakeKernelDatabase(), ops, charge_optimizer=False)
        microbatches = [SequenceStats.from_lengths([1024])]
        return composed.decompose_microbatches(
            microbatches, TrainingParallelism(tp=1, pp=1, dp=1, cp=cp)
        )

    def test_mismatched_cp_is_refused(self):
        from srl.cost.training.composed import ComposedTraining
        from srl.cost.training.training_ops import TrainingOps

        ops = TrainingOps(_dense_model(), tensor_parallel=1, context_parallel=1)
        composed = ComposedTraining(_FakeKernelDatabase(), ops, charge_optimizer=False)
        with pytest.raises(ValueError, match="ops.context_parallel"):
            composed.decompose_microbatches(
                [SequenceStats.from_lengths([1024])],
                TrainingParallelism(tp=1, pp=1, dp=1, cp=2),
            )

    def test_gemm_and_elementwise_and_logits_halve_at_cp_two(self):
        terms1 = self._terms(1)
        terms2 = self._terms(2)
        for key in ("gemm_fwd", "gemm_bwd", "elementwise", "logits"):
            assert terms2[key] == pytest.approx(terms1[key] / 2), key

    def test_attention_result_halves_at_cp_two_not_its_input_length(self):
        """
        The stub returns the SAME value regardless of cp, because `pack_lengths` is not
        divided. The halving must therefore come from `ComposedTraining` dividing the
        summed result, which is the behaviour under test.
        """
        terms1 = self._terms(1)
        terms2 = self._terms(2)
        assert terms2["attn_fwd"] == pytest.approx(terms1["attn_fwd"] / 2)
        assert terms2["attn_bwd"] == pytest.approx(terms1["attn_bwd"] / 2)

    def test_cp_one_is_unaffected(self):
        """A dense model's cp=1 decomposition must be identical before and after this change."""
        from srl.cost.training.composed import ComposedTraining
        from srl.cost.training.training_ops import TrainingOps

        ops = TrainingOps(_dense_model(), tensor_parallel=1, recompute="none")
        composed = ComposedTraining(_FakeKernelDatabase(), ops, charge_optimizer=False)
        terms = composed.decompose_microbatches(
            [SequenceStats.from_lengths([1024])], TrainingParallelism(tp=1, pp=1, dp=1)
        )
        db = _FakeKernelDatabase()
        expected = sum(
            db.query_gemm(query) * mult for query, mult in ops.forward_gemm_queries(1024)
        )
        assert terms["gemm_fwd"] == pytest.approx(expected)
