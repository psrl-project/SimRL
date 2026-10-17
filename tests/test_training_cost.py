"""Synthetic Hydraulis V1 latency, memory, OOD, and simulator tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from srl.cost.artifact import CostArtifact
from srl.cost.rollout import UnifiedRoofline
from srl.cost.training import (
    AnalyticalTrainingCost,
    HydraulisTrainingCost,
    LinearTrainingCost,
    TrainingOutOfDomainError,
    TrainingParallelism,
    TrainingWorkload,
    parameter_inventory,
)
from srl.cost.training.activation import ActivationGeometry, activation_bytes
from srl.cost.training.inventory import heaviest_stage_parameters
from srl.policy.psrl import PSRLPolicy
from srl.profiler.schema import (
    BYTES_PER_MIB,
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
from srl.simulator import SimConfig, Simulator
from srl.telemetry.store import JsonlStore
from srl.workload.distributions import uniform
from srl.workload.grouped import GroupedWorkload

PARAMETER_COUNT = 10_000_000
# (per_step_overhead, token_cost, microbatch_overhead) for the physical latency form
# `per_step_overhead + slots * (token_cost * tokens + microbatch_overhead)`.
LATENCY_COEFFICIENTS = (0.010, 1e-7, 0.002)
# The activation term is architecture-derived bytes, so the synthetic peaks are built
# from the same geometry the fit reconstructs and the recovered scale must be one.
ACTIVATION_SCALE = 1.0
MEMORY_RESIDUAL = 5_000_000.0
NUM_LAYERS = 8
HIDDEN_SIZE = 1024
# Small enough that two untied embeddings fit inside PARAMETER_COUNT. The memory model
# partitions a pipeline by putting the embedding on the first stage and the output
# projection on the last, so a fixture whose embeddings exceed its own parameter count
# is not a physically consistent architecture and is rejected.
VOCAB_SIZE = 2_048


def _synthetic_static_bytes(parallelism: TrainingParallelism) -> float:
    """Mirror the model's heaviest-stage share so the fit must recover a scale of one."""
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


def _training_records() -> tuple[list[ProfileRecord], ProfileRecord]:
    records: list[ProfileRecord] = []
    model = ModelDescriptor(
        name="synthetic-qwen",
        parameter_count=PARAMETER_COUNT,
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=16,
        num_query_groups=4,
        extra={"vocab_size": VOCAB_SIZE},
    )
    software = SoftwareDescriptor(
        backend="megatron_bridge",
        backend_version="0.4.2",
    )
    for dp in (1, 2):
        parallelism = Parallelism(tp=1, pp=2, dp=dp, cp=1)
        hardware = HardwareDescriptor(
            gpu_name="H100",
            gpu_count=int(parallelism.world_size),
            gpu_memory_mib=81920,
        )
        for micro_batch_size in (1, 2):
            for num_microbatches in (2, 4, 8):
                for sequence_length in (256, 512, 1024):
                    microbatches = tuple(
                        SequenceStats.from_lengths(
                            [sequence_length] * micro_batch_size
                        )
                        for _ in range(num_microbatches)
                    )
                    q_values = [item.sum_squared_tokens for item in microbatches]
                    t_values = [item.sum_tokens for item in microbatches]
                    q_effective = sum(q_values) + max(q_values)
                    slots = num_microbatches + 1
                    peak_tokens = max(t_values)
                    per_step, token_cost, microbatch_overhead = LATENCY_COEFFICIENTS
                    latency = per_step + slots * (
                        token_cost * peak_tokens + microbatch_overhead
                    )
                    training_parallelism = TrainingParallelism(
                        tp=1,
                        pp=2,
                        dp=dp,
                        cp=1,
                    )
                    static = _synthetic_static_bytes(training_parallelism)
                    activation_feature = activation_bytes(
                        ActivationGeometry(
                            num_layers=NUM_LAYERS,
                            hidden_size=HIDDEN_SIZE,
                            vocab_size=VOCAB_SIZE,
                        ),
                        microbatches,
                        training_parallelism,
                    )
                    peak = int(
                        static
                        + ACTIVATION_SCALE * activation_feature
                        + MEMORY_RESIDUAL
                    )
                    payload = TrainingPayload(
                        micro_batch_size=micro_batch_size,
                        global_batch_size=micro_batch_size * dp * num_microbatches,
                        num_microbatches=num_microbatches,
                        schedule="1f1b",
                        recompute="none",
                        optimizer="adam",
                        microbatches=microbatches,
                        step_times_s=(latency, latency * 1.001),
                        rank_memory=(
                            RankMemory(
                                rank=0,
                                allocated_bytes=peak,
                                reserved_bytes=peak + 1024,
                            ),
                        ),
                        peak_allocated_bytes=peak,
                        peak_reserved_bytes=peak + 1024,
                        parameter_count=PARAMETER_COUNT,
                    )
                    run_id = (
                        f"dp{dp}-mbs{micro_batch_size}-nmb{num_microbatches}"
                        f"-seq{sequence_length}"
                    )
                    records.append(
                        ProfileRecord(
                            record_type="training",
                            run=RunDescriptor(run_id=run_id),
                            model=model,
                            hardware=hardware,
                            software=software,
                            parallelism=parallelism,
                            payload=payload,
                        )
                    )

    oom_parallelism = Parallelism(tp=1, pp=2, dp=1, cp=1)
    oom_microbatches = tuple(
        SequenceStats.from_lengths([1024])
        for _ in range(2)
    )
    oom_static = _synthetic_static_bytes(TrainingParallelism(tp=1, pp=2, dp=1, cp=1))
    oom_activation = activation_bytes(
        ActivationGeometry(
            num_layers=NUM_LAYERS,
            hidden_size=HIDDEN_SIZE,
            vocab_size=VOCAB_SIZE,
        ),
        oom_microbatches,
        TrainingParallelism(tp=1, pp=2, dp=1, cp=1),
    )
    unconstrained_peak = int(
        oom_static + ACTIVATION_SCALE * oom_activation + MEMORY_RESIDUAL
    )
    oom_hardware = HardwareDescriptor(
        gpu_name="H100",
        gpu_count=int(oom_parallelism.world_size),
        gpu_memory_mib=(unconstrained_peak + 2_000_000 + BYTES_PER_MIB - 1) // BYTES_PER_MIB,
    )
    oom_payload = TrainingPayload(
        micro_batch_size=1,
        global_batch_size=2,
        num_microbatches=2,
        schedule="1f1b",
        recompute="none",
        optimizer="adam",
        microbatches=oom_microbatches,
        parameter_count=PARAMETER_COUNT,
    )
    oom_record = ProfileRecord(
        record_type="training",
        run=RunDescriptor(run_id="oom", status="oom", error="CUDA out of memory."),
        model=model,
        hardware=oom_hardware,
        software=software,
        parallelism=oom_parallelism,
        payload=oom_payload,
    )
    records.append(oom_record)
    return records, oom_record


def test_hydraulis_fit_roundtrip_and_known_latency(tmp_path: Path):
    records, _ = _training_records()
    artifact = HydraulisTrainingCost.fit(records)
    path = artifact.save(tmp_path / "hydraulis.json")
    loaded_artifact = CostArtifact.load(path)
    model = HydraulisTrainingCost.from_artifact(loaded_artifact)
    workload = TrainingWorkload(
        sequence_lengths=(512,) * 8,
        micro_batch_size=2,
    )
    parallelism = TrainingParallelism(tp=1, pp=2, dp=1, cp=1)
    estimate = model.estimate(workload, parallelism)

    # Four microbatches of two 512-token sequences, plus the PP=2 bubble slot.
    slots = 5
    peak_tokens = 2 * 512
    per_step, token_cost, microbatch_overhead = LATENCY_COEFFICIENTS
    expected = per_step + slots * (token_cost * peak_tokens + microbatch_overhead)
    # The fit must recover the generating coefficients, not merely interpolate.
    assert estimate.latency_s == pytest.approx(expected, rel=5e-3)
    assert estimate.peak_memory_bytes > 0
    assert {entry.selector["component"] for entry in artifact.entries} == {
        "latency",
        "memory",
    }
    # Latency is rank 4 (it carries a quadratic token term), memory is rank 2 or 3.
    ranks = {e.selector["component"]: e.train_metrics.design_rank for e in artifact.entries}
    assert ranks["latency"] == 4
    assert ranks["memory"] in {2, 3}


def test_hydraulis_records_whether_validation_was_held_out():
    """
    The same trap the FSDP model carries: `validation_metrics` may be in-sample.

    `split_records_by_run` returns the same list twice at fraction zero, so an entry's
    `r_squared` is then a training score under an out-of-sample name. Both backends record
    the fraction and the flag so a reader of either artifact can tell the difference.
    """
    records, _ = _training_records()
    in_sample = HydraulisTrainingCost.fit(records, validation_fraction=0.0)
    for entry in in_sample.entries:
        assert entry.diagnostics["validation_fraction"] == 0.0
        assert entry.diagnostics["validation_is_in_sample"] is True

    held_out = HydraulisTrainingCost.fit(records, validation_fraction=0.2)
    for entry in held_out.entries:
        assert entry.diagnostics["validation_fraction"] == 0.2
        assert entry.diagnostics["validation_is_in_sample"] is False


def test_hydraulis_uses_oom_as_memory_inequality():
    records, oom_record = _training_records()
    model = HydraulisTrainingCost.from_artifact(
        HydraulisTrainingCost.fit(records)
    )
    workload = TrainingWorkload(
        sequence_lengths=(1024, 1024),
        micro_batch_size=1,
    )
    predicted = model.memory_model.predict(
        workload,
        TrainingParallelism(tp=1, pp=2, dp=1, cp=1),
    )
    assert oom_record.hardware.gpu_memory_bytes is not None
    assert predicted > oom_record.hardware.gpu_memory_bytes


def test_hydraulis_dp_optimizer_sharding_reduces_memory():
    records, _ = _training_records()
    model = HydraulisTrainingCost.from_artifact(
        HydraulisTrainingCost.fit(records)
    )
    workload = TrainingWorkload(
        sequence_lengths=(512,) * 8,
        micro_batch_size=1,
    )
    dp1 = model.memory_model.predict(
        workload,
        TrainingParallelism(tp=1, pp=2, dp=1, cp=1),
    )
    dp2 = model.memory_model.predict(
        workload,
        TrainingParallelism(tp=1, pp=2, dp=2, cp=1),
    )
    assert dp2 < dp1


def test_hydraulis_rejects_numerical_extrapolation():
    records, _ = _training_records()
    model = HydraulisTrainingCost.from_artifact(
        HydraulisTrainingCost.fit(records)
    )
    workload = TrainingWorkload(
        sequence_lengths=(8192,) * 8,
        micro_batch_size=2,
    )
    with pytest.raises(TrainingOutOfDomainError, match="outside the profiled range"):
        model.estimate(
            workload,
            TrainingParallelism(tp=1, pp=2, dp=1, cp=1),
        )


def test_hydraulis_rejects_rank_deficient_design():
    """
    Varying only the sequence length cannot separate the per-microbatch cost.

    At a fixed microbatch count the slot count is constant, so the intercept and the
    per-microbatch column become collinear and the design falls to rank two. Both axes
    have to move, which is exactly the coverage the error message asks for.
    """
    records, _ = _training_records()
    restricted = [
        record
        for record in records
        if isinstance(record.payload, TrainingPayload)
        and record.run.status == "success"
        and record.parallelism.dp == 1
        # A single microbatch count, so only the sequence length varies.
        and record.payload.num_microbatches == 2
    ]
    assert len(restricted) >= 3, "the subset must be large enough to reach the rank check"
    with pytest.raises(ValueError, match="design matrix is rank"):
        HydraulisTrainingCost.fit(restricted)


def test_fitted_training_cost_integrates_with_simulator(tmp_path: Path):
    records, _ = _training_records()
    training_cost = HydraulisTrainingCost.from_artifact(
        HydraulisTrainingCost.fit(records)
    )
    store = JsonlStore(tmp_path / "logs")
    simulator = Simulator(
        rollout_cost=UnifiedRoofline(
            F=0.005,
            W=0.001,
            G=0.0001,
            A_p=0,
            A_d=1e-7,
        ),
        training_cost=training_cost,
        policy=PSRLPolicy(),
        cfg=SimConfig(
            batch_size=8,
            n_versions=1,
            staleness_horizon=2,
            training_pp=2,
            training_micro_batch_size=2,
            workload=GroupedWorkload.from_lengths([512], uniform(32, 50, 100, seed=1)),
        ),
        store=store,
    )
    result = simulator.run()
    rows = list(store.iter("train"))
    assert result.versions_done == 1
    assert result.train_peak_memory_bytes[0] > 0
    assert rows[0]["peak_memory_bytes"] == result.train_peak_memory_bytes[0]
    assert rows[0]["training_pp"] == 2


def test_linear_latency_scales_with_tokens_and_memory_with_gpus():
    model = LinearTrainingCost(time_per_token=1e-4, base_memory_bytes=80_000_000_000)
    short = TrainingWorkload(sequence_lengths=(256,) * 4, micro_batch_size=1)
    long = TrainingWorkload(sequence_lengths=(512,) * 4, micro_batch_size=1)
    one_gpu = TrainingParallelism()
    four_gpus = TrainingParallelism(tp=2, pp=1, dp=2, cp=1)

    short_estimate = model.estimate(short, one_gpu)
    long_estimate = model.estimate(long, one_gpu)
    sharded = model.estimate(long, four_gpus)

    assert short_estimate.latency_s == pytest.approx(0.1024)
    assert long_estimate.latency_s == pytest.approx(2 * short_estimate.latency_s)
    assert long_estimate.peak_memory_bytes == 80_000_000_000
    assert sharded.peak_memory_bytes == 20_000_000_000
    assert sharded.latency_s == long_estimate.latency_s


def test_analytical_roofline_uses_compute_and_shards_memory():
    model = AnalyticalTrainingCost(
        model_params=8_000_000_000,
        peak_flops=312e12,
        memory_bandwidth=2e12,
        overhead_factor=1.0,
        bytes_per_parameter=16.0,
    )
    workload = TrainingWorkload(sequence_lengths=(1000,) * 8, micro_batch_size=1)
    one_gpu = model.estimate(workload, TrainingParallelism())
    two_gpu = model.estimate(workload, TrainingParallelism(tp=2, pp=1, dp=1, cp=1))
    expected_compute = 6 * 8_000_000_000 * 8000 / 312e12
    assert one_gpu.latency_s == pytest.approx(expected_compute)
    assert two_gpu.latency_s == pytest.approx(expected_compute / 2)
    assert two_gpu.peak_memory_bytes == one_gpu.peak_memory_bytes // 2


class TestDynamicBatching:
    """Grouping microbatches by token budget rather than by sequence count."""

    def test_packs_to_the_budget_instead_of_a_fixed_count(self):
        from srl.cost.training.base import TrainingParallelism, TrainingWorkload

        # Four sequences, budget fits two of them per microbatch.
        workload = TrainingWorkload(
            sequence_lengths=(1000, 1000, 1000, 1000),
            micro_batch_size=1,
            max_tokens_per_microbatch=2000,
        )
        microbatches = workload.critical_rank_microbatches(
            TrainingParallelism(tp=1, pp=1, dp=1, cp=1)
        )
        assert len(microbatches) == 2
        assert [m.sum_tokens for m in microbatches] == [2000, 2000]

    def test_count_based_grouping_is_unchanged_when_no_budget_is_set(self):
        from srl.cost.training.base import TrainingParallelism, TrainingWorkload

        workload = TrainingWorkload(
            sequence_lengths=(1000, 1000, 1000, 1000),
            micro_batch_size=1,
        )
        microbatches = workload.critical_rank_microbatches(
            TrainingParallelism(tp=1, pp=1, dp=1, cp=1)
        )
        assert len(microbatches) == 4

    def test_ragged_batch_packs_greedily(self):
        from srl.cost.training.base import TrainingParallelism, TrainingWorkload

        # First-fit on the descending order 900, 600, 500, 400 under a 1000 budget:
        # 900 alone (600 would overflow), then 600 alone (500 would overflow), then
        # 500 + 400. Greedy and order-dependent, not optimal bin packing.
        workload = TrainingWorkload(
            sequence_lengths=(500, 900, 400, 600),
            micro_batch_size=1,
            max_tokens_per_microbatch=1000,
        )
        microbatches = workload.critical_rank_microbatches(
            TrainingParallelism(tp=1, pp=1, dp=1, cp=1)
        )
        assert [m.sum_tokens for m in microbatches] == [900, 600, 900]

    def test_attention_does_not_cross_sequences_inside_a_pack(self):
        """A packed microbatch sums each sequence's square, not the square of the total."""
        from srl.cost.training.base import TrainingParallelism, TrainingWorkload

        workload = TrainingWorkload(
            sequence_lengths=(1000, 1000),
            micro_batch_size=1,
            max_tokens_per_microbatch=2000,
        )
        (packed,) = workload.critical_rank_microbatches(
            TrainingParallelism(tp=1, pp=1, dp=1, cp=1)
        )
        assert packed.sum_squared_tokens == 2 * 1000**2
        assert packed.sum_squared_tokens != 2000**2

    def test_budget_sets_the_count_and_is_not_a_hard_cap(self):
        """verl fixes the microbatch count from the budget, then balances.

        A single sequence longer than the budget is not an error: the count
        formula floors at one group, and that group holds the sequence. Treating
        the budget as a hard per-group cap was wrong, and it also produced the
        wrong count: greedy first-fit gave 60 microbatches on the measured
        PSRL batch where the real trainer produced 44.
        """
        from srl.cost.training.base import TrainingParallelism, TrainingWorkload

        workload = TrainingWorkload(
            sequence_lengths=(5000,),
            micro_batch_size=1,
            max_tokens_per_microbatch=1000,
        )
        microbatches = workload.critical_rank_microbatches(
            TrainingParallelism(tp=1, pp=1, dp=1, cp=1)
        )
        assert len(microbatches) == 1
        assert microbatches[0].sum_tokens == 5000

    def test_microbatch_count_follows_verls_formula(self):
        """count = min(batch_size, ceil(total_tokens / budget)).

        Checked against the real run: instrumenting PSRL measured 44
        microbatches, and ceil(563200 / 12800) is 44.
        """
        from srl.cost.training.base import TrainingParallelism, TrainingWorkload

        one_rank = TrainingParallelism(tp=1, pp=1, dp=1, cp=1)
        # 64 sequences summing to 563_200 at a 12_800 budget -> 44 groups.
        lengths = tuple([8800] * 64)
        workload = TrainingWorkload(
            sequence_lengths=lengths,
            micro_batch_size=1,
            max_tokens_per_microbatch=12_800,
        )
        assert sum(lengths) == 563_200
        assert len(workload.critical_rank_microbatches(one_rank)) == 44

        # The count is capped by the sequence count: a budget larger than the
        # whole batch still cannot produce fewer than one group.
        single = TrainingWorkload(
            sequence_lengths=(100, 100),
            micro_batch_size=1,
            max_tokens_per_microbatch=1_000_000,
        )
        assert len(single.critical_rank_microbatches(one_rank)) == 1
