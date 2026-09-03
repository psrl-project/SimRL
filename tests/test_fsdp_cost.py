"""Fitting and evaluating the FSDP training cost model."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dryrun.cost.artifact import CostArtifact
from dryrun.cost.training import TRAINING_MODEL_REGISTRY
from dryrun.cost.training.base import (
    TrainingOutOfDomainError,
    TrainingParallelism,
    TrainingWorkload,
)
from dryrun.cost.training.fsdp import FsdpTrainingCost
from dryrun.cost.training.fsdp_inventory import fsdp_inventory
from dryrun.profiler.schema import (
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

PARAMETER_COUNT = 8_190_735_360
# A synthetic ground truth so a fitted coefficient can be checked against a known value.
PER_STEP = 0.20
TOKEN_COST = 5e-6
MICROBATCH_OVERHEAD = 0.05
# Memory ground-truth coefficients for testing both features.
MICROBATCH_SLOPE = 6e8  # bytes per additional microbatch (autograd-graph accumulation)
ACTIVATION_SLOPE = 1000.0  # bytes per token (activation residency when not checkpointed)


def _record(
    *,
    n_seqs: int,
    length: int,
    num_microbatches: int,
    shard_size: int = 8,
    replicate_size: int = 1,
    offload: bool = False,
    status: str = "success",
    peak_allocated: int | None = None,
    peak_reserved: int | None = None,
    run_id: str | None = None,
) -> ProfileRecord:
    world_size = shard_size * replicate_size
    microbatches = tuple(
        SequenceStats.from_lengths([length] * n_seqs) for _ in range(num_microbatches)
    )
    tokens = float(max(item.sum_tokens for item in microbatches))
    slots = float(num_microbatches)
    step = PER_STEP + slots * (TOKEN_COST * tokens + MICROBATCH_OVERHEAD)
    static = fsdp_inventory(
        parameter_count=PARAMETER_COUNT,
        world_size=world_size,
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=offload,
    ).total_bytes
    # Memory depends on both num_microbatches (autograd-graph accumulation under gradient
    # checkpointing) and tokens (resident activations when checkpointing is off). Using
    # both here means the fixture matches the real physics and the fit must recover both
    # coefficients to pass.
    allocated = peak_allocated if peak_allocated is not None else int(
        static + MICROBATCH_SLOPE * num_microbatches + ACTIVATION_SLOPE * tokens
    )
    # `peak_reserved` defaults to `allocated + 1`, an affine shift both regressions absorb
    # exactly, producing a tie at r²=1.0.  Pass an explicit value to break that tie.
    reserved = peak_reserved if peak_reserved is not None else allocated + 1
    return ProfileRecord(
        record_type="training",
        run=RunDescriptor(
            run_id=run_id or f"fsdp-{n_seqs}-{length}-{num_microbatches}-{shard_size}-{offload}",
            repeat=0,
            status=status,
            duration_s=10.0,
        ),
        model=ModelDescriptor(
            name="Qwen3-8B",
            architecture="Qwen3ForCausalLM",
            parameter_count=PARAMETER_COUNT,
            num_layers=36,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=8,
        ),
        hardware=HardwareDescriptor(
            gpu_name="H20",
            gpu_count=world_size,
            gpu_memory_mib=97871,
            node_count=1,
            gpus_per_node=8,
        ),
        software=SoftwareDescriptor(backend="fsdp", backend_version="2.11.0"),
        parallelism=Parallelism(tp=1, pp=1, dp=world_size, cp=1, world_size=world_size),
        payload=TrainingPayload(
            micro_batch_size=n_seqs,
            global_batch_size=n_seqs * world_size * num_microbatches,
            num_microbatches=num_microbatches,
            schedule="fsdp",
            recompute="full",
            optimizer="adam",
            microbatches=microbatches,
            step_times_s=(step, step * 1.001, step * 0.999) if status == "success" else (),
            rank_memory=(
                (RankMemory(rank=0, allocated_bytes=allocated, reserved_bytes=reserved),)
                if status == "success"
                else ()
            ),
            peak_allocated_bytes=allocated if status == "success" else None,
            peak_reserved_bytes=reserved if status == "success" else None,
            parameter_dtype="bf16",
            gradient_dtype="fp32",
            optimizer_state_dtype="fp32",
            parameter_count=PARAMETER_COUNT,
            warmup_steps=4,
        ),
        tags={
            "shard_size": str(shard_size),
            "replicate_size": str(replicate_size),
            "optimizer_offload": str(offload),
            "reshard_after_forward": "True",
            "sequence_parallel": "1",
        },
    )


def _records(shard_size: int = 8) -> list[ProfileRecord]:
    """Vary both axes so the design reaches rank three."""
    records = []
    for length in (512, 1024, 2048):
        for num_microbatches in (1, 2, 4):
            records.append(
                _record(
                    n_seqs=1,
                    length=length,
                    num_microbatches=num_microbatches,
                    shard_size=shard_size,
                )
            )
    return records


def _cost(*, shard_size: int = 8, **overrides) -> FsdpTrainingCost:
    artifact = FsdpTrainingCost.fit(_records(shard_size), validation_fraction=0.0)
    defaults = {"batching_mode": "pad_to_max"}
    return FsdpTrainingCost.from_artifact(artifact, **{**defaults, **overrides})


def test_registered_under_the_fsdp_name():
    assert TRAINING_MODEL_REGISTRY["fsdp"] is FsdpTrainingCost


def test_fit_recovers_the_known_coefficients():
    artifact = FsdpTrainingCost.fit(_records(), validation_fraction=0.0)
    entry = next(e for e in artifact.entries if e.selector["component"] == "latency")
    assert entry.coefficients["per_step_overhead"] == pytest.approx(PER_STEP, rel=0.02)
    assert entry.coefficients["token_cost"] == pytest.approx(TOKEN_COST, rel=0.02)
    assert entry.coefficients["microbatch_overhead"] == pytest.approx(
        MICROBATCH_OVERHEAD, rel=0.02
    )


def test_fit_recovers_both_memory_coefficients():
    """
    The memory model must recover both `microbatch_slope` and `activation_slope`.

    The fixture encodes memory as `static + MICROBATCH_SLOPE * nmb + ACTIVATION_SLOPE * tokens`,
    and the fit must separate both terms to within 5%. A single-feature fit against only
    tokens would collapse the nmb contribution into the constant and mis-size every
    prediction that varies the microbatch count.
    """
    artifact = FsdpTrainingCost.fit(_records(), validation_fraction=0.0)
    entry = next(e for e in artifact.entries if e.selector["component"] == "memory")
    assert entry.coefficients["microbatch_slope"] == pytest.approx(MICROBATCH_SLOPE, rel=0.05)
    assert entry.coefficients["activation_slope"] == pytest.approx(ACTIVATION_SLOPE, rel=0.05)


def test_fit_memory_nmb_only_drives_coefficient_and_not_token_term():
    """
    Memory that depends only on `num_microbatches` must fit with high r² and near-zero
    `activation_slope`.

    This is the regression the old single-feature design failed: with `gradient_checkpointing=true`
    the real profiled data is flat across tokens (to four decimal places across a 4x range)
    but varies strongly with `num_microbatches`. Fitting only `max_tokens` against that data
    produced `r²` in the range of -2.6e9 because the single feature explained nothing.

    The test builds a synthetic version of that scenario and asserts that the two-term design
    handles it correctly: the token coefficient is driven near zero by the non-negative constraint
    rather than distorting the microbatch slope.
    """
    # Build records where memory depends only on nmb (token coefficient is truly zero).
    static = fsdp_inventory(
        parameter_count=PARAMETER_COUNT,
        world_size=8,
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=False,
    ).total_bytes
    nmb_only_slope = 5e8  # 500 MiB per microbatch, matching the measured ~580 MiB per nmb
    records = []
    for length in (512, 1024, 2048):
        for num_microbatches in (1, 2, 4):
            allocated = int(static + nmb_only_slope * num_microbatches)
            records.append(
                _record(
                    n_seqs=1,
                    length=length,
                    num_microbatches=num_microbatches,
                    peak_allocated=allocated,
                    peak_reserved=allocated + 1,
                )
            )
    artifact = FsdpTrainingCost.fit(records, validation_fraction=0.0)
    entry = next(e for e in artifact.entries if e.selector["component"] == "memory")
    coefficients = entry.coefficients
    # The fit must recover the microbatch slope and drive the token slope to approximately zero.
    assert coefficients["microbatch_slope"] == pytest.approx(nmb_only_slope, rel=0.05)
    # The activation slope is driven near zero by the non-negative constraint; the threshold
    # is 0.1% of the microbatch slope so a collapsed token term cannot pass as a real signal.
    assert coefficients["activation_slope"] < nmb_only_slope * 0.001
    # The validation r² must be positive since the microbatch signal is strong and clean.
    r2 = entry.diagnostics["candidate_r_squared"][entry.diagnostics["memory_metric"]]
    assert r2 > 0.99, f"Expected r² > 0.99 for nmb-only memory, got {r2:.4f}."


def test_fit_records_no_pipeline_term():
    """
    FSDP has no pipeline, so slots reduce to the microbatch count.

    Asserting `"c_pipeline" not in coefficients` cannot fail, since nothing anywhere writes
    that key. What is worth pinning is the exact coefficient set, so a fifth fitted term
    cannot appear unnoticed.

    The set is four rather than three because FSDP carries a quadratic token term that
    Megatron does not. FSDP is profiled to 16384 tokens per microbatch, where measured
    per-token cost is U-shaped (918 us/token at 512, 525 at 8192, 608 at 16384) and a form
    linear in tokens under-predicts by 28% at the low end and 17% at the high end. Megatron
    stops at 2048, where the same term fails leave-one-out.
    """
    artifact = FsdpTrainingCost.fit(_records(), validation_fraction=0.0)
    entry = next(e for e in artifact.entries if e.selector["component"] == "latency")
    assert set(entry.coefficients) == {
        "per_step_overhead",
        "token_cost",
        "microbatch_overhead",
        "token_quadratic_cost",
    }
    assert entry.diagnostics["slots"] == "num_microbatches"


def test_artifact_round_trips(tmp_path: Path):
    artifact = FsdpTrainingCost.fit(_records(), validation_fraction=0.0)
    path = artifact.save(tmp_path / "fsdp.json")
    reloaded = CostArtifact.load(path)
    assert reloaded.kind == "training"
    assert reloaded.model_type == "fsdp"
    assert len(reloaded.entries) == len(artifact.entries)


def test_predicts_a_known_step_time():
    cost = _cost()
    # One sequence per rank on 8 ranks, so dp=8 and each rank holds one 1024 sequence.
    workload = TrainingWorkload(
        sequence_lengths=(1024,) * 8,
        micro_batch_size=1,
        schedule="fsdp",
        recompute="full",
        optimizer="adam",
    )
    estimate = cost.estimate(workload, TrainingParallelism(dp=8))
    expected = PER_STEP + 1.0 * (TOKEN_COST * 1024 + MICROBATCH_OVERHEAD)
    assert estimate.latency_s == pytest.approx(expected, rel=0.05)


def test_memory_uses_the_fsdp_inventory_not_megatron_s():
    cost = _cost()
    workload = TrainingWorkload(
        sequence_lengths=(1024,) * 8,
        micro_batch_size=1,
        schedule="fsdp",
        recompute="full",
        optimizer="adam",
    )
    estimate = cost.estimate(workload, TrainingParallelism(dp=8))
    static = fsdp_inventory(
        parameter_count=PARAMETER_COUNT,
        world_size=8,
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=False,
    ).total_bytes
    assert estimate.peak_memory_bytes >= static
    # Megatron's inventory would be roughly twice this, so a factor-two error is caught.
    assert estimate.peak_memory_bytes < static * 1.6


def test_memory_target_selection_is_recorded():
    artifact = FsdpTrainingCost.fit(_records(), validation_fraction=0.0)
    entry = next(e for e in artifact.entries if e.selector["component"] == "memory")
    diagnostics = entry.diagnostics
    assert diagnostics["memory_metric"] in ("peak_allocated_bytes", "peak_reserved_bytes")
    # Both candidates must be reported so the choice is auditable.
    assert "peak_allocated_bytes" in diagnostics["candidate_r_squared"]
    assert "peak_reserved_bytes" in diagnostics["candidate_r_squared"]


def _records_discriminating(*, noisy_target: str, seed: int = 42) -> list[ProfileRecord]:
    """
    Build a record set where the two memory targets are NOT affine images of each other.

    The existing `_records()` fixture sets `reserved = allocated + 1`.  That constant
    shift is an affine transform, so both linear regressions absorb it perfectly and both
    candidates reach r²=1.0.  `test_memory_target_selection_is_recorded` therefore
    passes even if the selection always returns a hardcoded answer -- a tie makes the
    winner irrelevant.

    This helper breaks the tie by injecting large deterministic noise into one target
    while keeping the other on a clean linear function of the memory features.  The
    noise amplitude (3x the signal range) ensures the noisy target's r² is well below
    zero, making the winner unambiguous and the ordering checkable.

    The allocator invariant (`reserved_bytes >= allocated_bytes`) is preserved by always
    expressing the noisy target relative to the clean one in the direction that satisfies
    it: when allocated is noisy, `allocated = clean - |noise|`; when reserved is noisy,
    `reserved = clean + |noise|`.

    The microbatch counts are chosen so that `num_microbatches` and `max_tokens` are NOT
    perfectly collinear, which is required for the three-column memory design to reach
    rank 3.

    Args:
        noisy_target (str): Which metric receives noise: ``"peak_allocated_bytes"`` or
            ``"peak_reserved_bytes"``.  The other metric is a clean linear function.
        seed (int): RNG seed for the noise so every run produces the same values.
    """
    if noisy_target not in ("peak_allocated_bytes", "peak_reserved_bytes"):
        raise ValueError(f"noisy_target must be one of the two memory metric names, got {noisy_target!r}.")

    rng = np.random.default_rng(seed)
    # Use lengths and microbatch counts that are NOT proportional, so `num_microbatches`
    # and `max_tokens` vary independently and the three-column design reaches rank 3.
    # (512,1), (1024,1), (2048,2): tokens double twice but nmb only changes once, so the
    # two columns are linearly independent.
    # A fourth pair is needed because the latency design is [1, nmb*tok, nmb, nmb*tok^2] and
    # rank 4 requires three distinct lengths AND two distinct microbatch counts. The memory
    # design this fixture exists to discriminate is unaffected by the extra row.
    lengths = (512, 1024, 2048, 4096)
    microbatches_per_length = (1, 1, 2, 4)
    # Signal range: 800 bytes/token * (2048 - 512) = 1_228_800 bytes.
    # Noise magnitude is 3x the signal range so the noisy target is dominated by noise
    # and its r² will be strongly negative.
    noise_magnitude = int(800 * (max(lengths) - min(lengths)) * 3)
    # Draw strictly positive displacements so the direction constraint stays clean.
    noise = rng.integers(noise_magnitude // 2, noise_magnitude, size=len(lengths), endpoint=True)

    static = fsdp_inventory(
        parameter_count=PARAMETER_COUNT,
        world_size=8,
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=False,
    ).total_bytes

    records = []
    for idx, (length, num_mb) in enumerate(zip(lengths, microbatches_per_length)):
        clean = int(static + 800 * length)
        if noisy_target == "peak_allocated_bytes":
            # Reserved is the clean signal; allocated is depressed by noise to preserve
            # the `reserved >= allocated` invariant while decorrelating it from length.
            reserved = clean
            allocated = clean - int(noise[idx])
        else:
            # Allocated is the clean signal; reserved is inflated by noise to preserve
            # the `reserved >= allocated` invariant while decorrelating it from length.
            allocated = clean
            reserved = clean + int(noise[idx])
        records.append(
            _record(
                n_seqs=1,
                length=length,
                num_microbatches=num_mb,
                peak_allocated=allocated,
                peak_reserved=reserved,
                run_id=f"disc-{noisy_target}-{length}",
            )
        )
    return records


def test_memory_target_selection_prefers_reserved_when_allocated_is_noisy():
    """
    Assert that the model selects `peak_reserved_bytes` when `peak_allocated_bytes`
    carries noise uncorrelated with the activation feature.

    The existing `test_memory_target_selection_is_recorded` cannot prove real selection
    because its fixture sets `reserved = allocated + 1`, an affine transform that both
    linear fits absorb perfectly, yielding r²=1.0 for both candidates.  That tie always
    resolves to `peak_allocated_bytes` by the stable `max` fallback, so a hardcoded
    winner would pass that test.  Here the allocated metric is deliberately noisy so the
    r² ordering contradicts the default fallback: only real data-driven selection picks
    the correct winner.
    """
    records = _records_discriminating(noisy_target="peak_allocated_bytes", seed=42)
    artifact = FsdpTrainingCost.fit(records, validation_fraction=0.0)
    entry = next(e for e in artifact.entries if e.selector["component"] == "memory")
    diagnostics = entry.diagnostics
    r2 = diagnostics["candidate_r_squared"]
    assert diagnostics["memory_metric"] == "peak_reserved_bytes", (
        f"Expected peak_reserved_bytes to win when allocated is noisy, "
        f"got {diagnostics['memory_metric']!r} (r²={r2})."
    )
    # The winner must genuinely outscore the loser, not just be picked by some other rule.
    assert r2["peak_reserved_bytes"] > r2["peak_allocated_bytes"], (
        f"Reserved r²={r2['peak_reserved_bytes']:.4f} must exceed "
        f"allocated r²={r2['peak_allocated_bytes']:.4f}."
    )


def test_memory_target_selection_prefers_allocated_when_reserved_is_noisy():
    """
    Assert that the model selects `peak_allocated_bytes` when `peak_reserved_bytes`
    carries noise uncorrelated with the activation feature.

    This is the direction that confirms the expected production outcome (allocated winning
    on real hardware, as the Megatron profiling found), but expressed as a data-driven
    assertion rather than a default-fallback assertion.  Together with
    `test_memory_target_selection_prefers_reserved_when_allocated_is_noisy`, the pair
    proves that selection follows the r² evidence in both directions.
    """
    records = _records_discriminating(noisy_target="peak_reserved_bytes", seed=42)
    artifact = FsdpTrainingCost.fit(records, validation_fraction=0.0)
    entry = next(e for e in artifact.entries if e.selector["component"] == "memory")
    diagnostics = entry.diagnostics
    r2 = diagnostics["candidate_r_squared"]
    assert diagnostics["memory_metric"] == "peak_allocated_bytes", (
        f"Expected peak_allocated_bytes to win when reserved is noisy, "
        f"got {diagnostics['memory_metric']!r} (r²={r2})."
    )
    assert r2["peak_allocated_bytes"] > r2["peak_reserved_bytes"], (
        f"Allocated r²={r2['peak_allocated_bytes']:.4f} must exceed "
        f"reserved r²={r2['peak_reserved_bytes']:.4f}."
    )


def test_an_in_sample_validation_score_is_recorded_as_such():
    """
    An in-sample `r_squared=1.0` is indistinguishable from a held-out one on inspection.

    Every entry stores its score under the name `validation_metrics`, and
    `split_records_by_run` returns the same list twice at fraction zero, so the artifact
    has to say which it is or a reader will over-trust it.
    """
    artifact = FsdpTrainingCost.fit(_records(), validation_fraction=0.0)
    for component in ("latency", "memory"):
        entry = next(e for e in artifact.entries if e.selector["component"] == component)
        assert entry.diagnostics["validation_fraction"] == 0.0
        assert entry.diagnostics["validation_is_in_sample"] is True, component


def test_a_held_out_validation_score_is_recorded_as_out_of_sample():
    """With nine distinct runs the default fraction genuinely holds some of them back."""
    artifact = FsdpTrainingCost.fit(_records(), validation_fraction=0.2)
    for component in ("latency", "memory"):
        entry = next(e for e in artifact.entries if e.selector["component"] == component)
        assert entry.diagnostics["validation_fraction"] == 0.2
        assert entry.diagnostics["validation_is_in_sample"] is False, component


def test_an_uninformative_memory_selection_is_recorded_as_such():
    """
    `FitMetrics.from_predictions` returns `r_squared=0.0` whenever `ss_tot == 0`, which is
    unconditionally true at `n_samples == 1` no matter how large the residual.

    Both targets then score equally and the winner falls to MEMORY_TARGETS order while the
    artifact presents the pair as evidence. The selection rule is unchanged, so this only
    asserts that the audit trail marks the comparison as uninformative.
    """
    # `reserved = allocated + 1` is an affine shift both linear fits absorb exactly, so
    # both candidates reach r-squared 1.0 and the comparison decides nothing.
    artifact = FsdpTrainingCost.fit(_records(), validation_fraction=0.0)
    entry = next(e for e in artifact.entries if e.selector["component"] == "memory")
    scores = entry.diagnostics["candidate_r_squared"]
    assert scores["peak_allocated_bytes"] == scores["peak_reserved_bytes"]
    assert entry.diagnostics["validation_selection_informative"] is False


def test_an_informative_memory_selection_is_recorded_as_such():
    """When the two targets genuinely disagree the flag must not cry wolf."""
    records = _records_discriminating(noisy_target="peak_reserved_bytes", seed=42)
    artifact = FsdpTrainingCost.fit(records, validation_fraction=0.0)
    entry = next(e for e in artifact.entries if e.selector["component"] == "memory")
    assert entry.diagnostics["validation_selection_informative"] is True


def test_memory_prediction_is_compared_against_reserved():
    """
    `allocated` excludes fragmentation, so the capacity check needs the reserved ratio.

    Asserting `reserved_over_allocated >= 1.0` cannot fail, because `_reserved_over_allocated`
    ends in `max(1.0, ...)`, and it says nothing about whether `predict` applies the ratio at
    all. The prediction is therefore checked against the raw fitted value times the ratio.
    """
    # A 2.2x ratio, close to the 20.928/9.478 measured on 8x H20, so dropping the multiply
    # is a twofold error rather than a rounding difference.
    # Use (length, nmb) pairs that are NOT proportional so the three-column memory design
    # reaches rank 3. (512,1), (1024,1), (2048,2) keeps tokens varying while nmb only
    # changes once, giving independent columns.
    # A fourth pair is added because the latency design is rank 4; the memory design this
    # test is about needs only rank 3 and is unaffected by the extra row.
    records = []
    for length, num_microbatches in zip((512, 1024, 2048, 4096), (1, 1, 2, 4)):
        static = fsdp_inventory(
            parameter_count=PARAMETER_COUNT,
            world_size=8,
            parameter_dtype="bf16",
            gradient_dtype="fp32",
            optimizer_state_dtype="fp32",
            optimizer_offload=False,
        ).total_bytes
        allocated = int(static + 1000 * length)
        records.append(
            _record(
                n_seqs=1,
                length=length,
                num_microbatches=num_microbatches,
                peak_allocated=allocated,
                peak_reserved=int(allocated * 2.2),
                run_id=f"ratio-{length}",
            )
        )
    artifact = FsdpTrainingCost.fit(records, validation_fraction=0.0)
    entry = next(e for e in artifact.entries if e.selector["component"] == "memory")
    # The clean target here is allocated, so the ratio must be the one that gets applied.
    assert entry.diagnostics["memory_metric"] == "peak_allocated_bytes"
    ratio = entry.diagnostics["reserved_over_allocated"]
    assert ratio == pytest.approx(2.2, rel=1e-6)

    cost = FsdpTrainingCost.from_artifact(artifact, batching_mode="pad_to_max")
    workload = TrainingWorkload(
        sequence_lengths=(1024,) * 8,
        micro_batch_size=1,
        schedule="fsdp",
        recompute="full",
        optimizer="adam",
    )
    predicted = cost.memory_model.predict(workload, TrainingParallelism(dp=8))
    static = fsdp_inventory(
        parameter_count=PARAMETER_COUNT,
        world_size=8,
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=False,
    ).total_bytes
    raw = (
        static
        + entry.coefficients["microbatch_slope"] * 1.0  # one microbatch for the 1024-token workload
        + entry.coefficients["activation_slope"] * 1024
        + entry.coefficients["residual_constant"]
    )
    assert predicted == pytest.approx(raw * ratio, rel=1e-9)


def test_rejects_a_dp_that_is_not_the_fitted_mesh():
    """
    A cell is a measurement of one mesh, so answering for another mesh is out of domain.

    The mesh never enters the fitted form, so before this check every dp returned the same
    memory prediction while the true sharded inventory scales as `1/dp`. Measured on the
    8-rank fixture, a `dp=4` query came back at 0.50x of the true static pool and a `dp=2`
    query at 0.25x, so the simulator's capacity comparison could not trip.
    """
    cost = _cost()
    # Sixteen sequences so both dp=8 and dp=4 can be planned from one workload.
    workload = TrainingWorkload(
        sequence_lengths=(1024,) * 16,
        micro_batch_size=1,
        schedule="fsdp",
        recompute="full",
        optimizer="adam",
    )
    with pytest.raises(TrainingOutOfDomainError, match="dp=4"):
        cost.estimate(workload, TrainingParallelism(dp=4))
    with pytest.raises(TrainingOutOfDomainError, match="dp=16"):
        cost.estimate(workload, TrainingParallelism(dp=16))


def test_the_fitted_mesh_still_predicts():
    """The mesh check must reject only a mismatch, not the mesh that was profiled."""
    cost = _cost()
    workload = TrainingWorkload(
        sequence_lengths=(1024,) * 8,
        micro_batch_size=1,
        schedule="fsdp",
        recompute="full",
        optimizer="adam",
    )
    estimate = cost.estimate(workload, TrainingParallelism(dp=8))
    static = fsdp_inventory(
        parameter_count=PARAMETER_COUNT,
        world_size=8,
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=False,
    ).total_bytes
    assert estimate.peak_memory_bytes >= static


def test_batching_mode_changes_the_prediction():
    lengths = (10240, 10240, 900, 1100, 950, 1050, 800, 1200)
    # micro_batch_size=4: each rank's 4 sequences are fused into one microbatch.
    # pad_to_max then pads all 4 to the rank-maximum, which is expensive on a bimodal
    # distribution. packed_no_pad charges only real tokens and is measurably cheaper.
    workload = TrainingWorkload(
        sequence_lengths=lengths,
        micro_batch_size=4,
        schedule="fsdp",
        recompute="full",
        optimizer="adam",
    )
    # A two-rank cell, because the mesh check pins the query's dp to the profiled mesh.
    _all = ["num_microbatches", "tokens", "sequences_per_microbatch", "activation_tokens"]
    padded = _cost(shard_size=2, batching_mode="pad_to_max", extrapolate=_all)
    packed = _cost(shard_size=2, batching_mode="packed_no_pad", extrapolate=_all)
    parallelism = TrainingParallelism(dp=2)
    padded_latency = padded.estimate(workload, parallelism).latency_s
    packed_latency = packed.estimate(workload, parallelism).latency_s
    assert packed_latency < padded_latency


def test_rejects_a_query_outside_the_profiled_range():
    cost = _cost()
    workload = TrainingWorkload(
        sequence_lengths=(65536,) * 8,
        micro_batch_size=1,
        schedule="fsdp",
        recompute="full",
        optimizer="adam",
    )
    with pytest.raises(TrainingOutOfDomainError):
        cost.estimate(workload, TrainingParallelism(dp=8))


def test_rejects_a_rank_deficient_design():
    """Sequence length alone cannot reach rank four, however many lengths are measured.

    Four records rather than three: the candidate-count guard fires first below four, and it
    would mask the rank message this test is about.
    """
    records = [
        _record(n_seqs=1, length=length, num_microbatches=1)
        for length in (512, 1024, 2048, 4096)
    ]
    with pytest.raises(ValueError, match="rank"):
        FsdpTrainingCost.fit(records, validation_fraction=0.0)


def test_oom_records_raise_the_memory_floor():
    """
    The floor must land at capacity, and the entry's metrics must describe it.

    Asserting only `oom_constraints == 1` would pass with the floor anywhere, including at
    0.45x or 2.208x capacity, which is exactly how a units error in the inequality hid
    once already. The prediction at the OOM configuration is therefore checked directly.
    """
    records = _records()
    oom = _record(
        n_seqs=8,
        length=2048,
        num_microbatches=4,
        status="oom",
        run_id="fsdp-oom",
    )
    artifact = FsdpTrainingCost.fit([*records, oom], validation_fraction=0.0)
    entry = next(e for e in artifact.entries if e.selector["component"] == "memory")
    assert entry.diagnostics["oom_constraints"] == 1

    cost = FsdpTrainingCost.from_artifact(
        artifact,
        batching_mode="token_budget",
        max_tokens_per_rank=16384,
        extrapolate=["num_microbatches", "tokens", "sequences_per_microbatch", "activation_tokens"],
    )
    # The OOM candidate had n_seqs=8 sequences of 2048 tokens across num_microbatches=4, giving
    # features (nmb=4, max_tokens=16384). With token_budget at 16384 tokens and 256 total
    # sequences on dp=8, each rank processes 4 microbatches of 8 sequences at 2048 tokens,
    # reproducing (nmb=4, max_tokens=16384) exactly so the floor check is at the OOM point.
    workload = TrainingWorkload(
        sequence_lengths=(2048,) * 256,
        micro_batch_size=8,
        schedule="fsdp",
        recompute="full",
        optimizer="adam",
    )
    predicted = cost.memory_model.predict(workload, TrainingParallelism(dp=8))
    capacity = oom.hardware.gpu_memory_bytes
    ratio = predicted / capacity
    assert 1.0 <= ratio < 1.01, f"The OOM floor must sit at capacity, got {ratio:.4f}x."


def test_the_oom_bumped_metrics_describe_the_shipped_coefficients():
    """
    An entry must not advertise the fit quality of a constant it does not carry.

    The OOM inequality raises the intercept after the regression converges. Measured on
    this fixture the constant moves from 6.03e-10 to 8.62e10, and metrics computed before
    the bump still read `r_squared=1.0, mre_percent=0.00` while the honest mean relative
    error of the shipped coefficients against the same successful records is 526%.
    """
    records = _records()
    oom = _record(
        n_seqs=8,
        length=2048,
        num_microbatches=4,
        status="oom",
        run_id="fsdp-oom",
    )
    artifact = FsdpTrainingCost.fit([*records, oom], validation_fraction=0.0)
    entry = next(e for e in artifact.entries if e.selector["component"] == "memory")
    assert entry.diagnostics["oom_intercept_increase"] > 0

    static = fsdp_inventory(
        parameter_count=PARAMETER_COUNT,
        world_size=8,
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        optimizer_offload=False,
    ).total_bytes
    metric = entry.diagnostics["memory_metric"]
    nmb_features = np.asarray([float(r.payload.num_microbatches) for r in records])
    token_features = np.asarray(
        [float(max(m.sum_tokens for m in r.payload.microbatches)) for r in records]
    )
    targets = np.asarray([float(getattr(r.payload, metric)) for r in records])
    predicted = (
        static
        + entry.coefficients["microbatch_slope"] * nmb_features
        + entry.coefficients["activation_slope"] * token_features
        + entry.coefficients["residual_constant"]
    )
    honest_mre = float(np.mean(np.abs(targets - predicted) / targets) * 100)
    # The bump is enormous relative to the measured peaks, so an honest metric cannot be
    # near zero. Comparing to the recomputed value pins the two together exactly.
    assert honest_mre > 100
    assert entry.train_metrics.mre_percent == pytest.approx(honest_mre, rel=1e-6)
    assert entry.train_metrics.r_squared < 0


def test_rejects_megatron_records():
    """A 1F1B record must never be fitted by the FSDP model."""
    record = _record(n_seqs=1, length=512, num_microbatches=1)
    megatron = ProfileRecord(
        record_type="training",
        run=record.run,
        model=record.model,
        hardware=record.hardware,
        software=record.software,
        parallelism=record.parallelism,
        payload=TrainingPayload(
            micro_batch_size=1,
            global_batch_size=8,
            num_microbatches=1,
            schedule="1f1b",
            recompute="full",
            optimizer="adam",
            microbatches=record.payload.microbatches,
            step_times_s=(1.0,),
            parameter_count=PARAMETER_COUNT,
        ),
        tags=record.tags,
    )
    with pytest.raises(ValueError, match="fsdp"):
        FsdpTrainingCost.fit([megatron], validation_fraction=0.0)


def test_satisfies_the_contract_the_simulator_calls():
    """
    The simulator's only requirement is `estimate(workload, parallelism)`.

    It reads `latency_s` to advance the clock and `peak_memory_bytes` to flag OOM
    against configured capacity, so both must be positive and finite for a workload
    built the way `Simulator._train` builds one.
    """
    import math

    from dryrun.cost.training.base import TrainingCostEstimate

    cost = _cost()
    workload = TrainingWorkload(
        sequence_lengths=(1024,) * 8,
        micro_batch_size=1,
        schedule="fsdp",
        recompute="full",
        optimizer="adam",
    )
    estimate = cost.estimate(workload, TrainingParallelism(dp=8))
    assert isinstance(estimate, TrainingCostEstimate)
    assert math.isfinite(estimate.latency_s) and estimate.latency_s > 0
    assert isinstance(estimate.peak_memory_bytes, int)
    assert estimate.peak_memory_bytes > 0
    # The diagnostics the simulator writes to telemetry must name the mesh.
    assert estimate.diagnostics["dp"] == 8


# ---------------------------------------------------------------------------
# extrapolate interface tests (replaces allow_extrapolation: bool)
# ---------------------------------------------------------------------------


def test_empty_extrapolate_rejects_out_of_range_for_all_features():
    """An empty list (the default) must reject any out-of-range query."""
    cost = _cost()
    workload = TrainingWorkload(
        sequence_lengths=(65536,) * 8,  # tokens way beyond profiled [512, 2048]
        micro_batch_size=1,
        schedule="fsdp",
        recompute="full",
        optimizer="adam",
    )
    with pytest.raises(TrainingOutOfDomainError):
        cost.estimate(workload, TrainingParallelism(dp=8))


def test_null_extrapolate_rejects_out_of_range():
    """Passing extrapolate=None must behave identically to an empty list."""
    artifact = FsdpTrainingCost.fit(_records(), validation_fraction=0.0)
    cost = FsdpTrainingCost.from_artifact(
        artifact, batching_mode="pad_to_max", extrapolate=None
    )
    workload = TrainingWorkload(
        sequence_lengths=(65536,) * 8,
        micro_batch_size=1,
        schedule="fsdp",
        recompute="full",
        optimizer="adam",
    )
    with pytest.raises(TrainingOutOfDomainError):
        cost.estimate(workload, TrainingParallelism(dp=8))


def test_listed_feature_permits_that_feature_and_rejects_others():
    """Listing one feature permits only that feature; unlisted features still raise."""
    artifact = FsdpTrainingCost.fit(_records(), validation_fraction=0.0)
    # The profiled range for num_microbatches is [1, 4] (from _records using 1, 2, 4).
    # We construct a workload whose num_microbatches is 8 (out of range) but whose tokens
    # are in-range. Listing num_microbatches must let it through; tokens out-of-range must
    # still raise when tokens is not listed.
    cost_nmb_only = FsdpTrainingCost.from_artifact(
        artifact,
        batching_mode="pad_to_max",
        extrapolate=["num_microbatches"],
    )
    # num_microbatches=8 out of [1,4], tokens in-range — must succeed because nmb is listed.
    workload_nmb_oor = TrainingWorkload(
        sequence_lengths=(512,) * 8 * 8,  # 8 seqs/rank * 8 ranks, tokens=512 in range
        micro_batch_size=1,  # micro_batch_size=1 → 8 microbatches per rank (out of [1,4])
        schedule="fsdp",
        recompute="full",
        optimizer="adam",
    )
    # This must NOT raise (num_microbatches is permitted):
    estimate = cost_nmb_only.estimate(workload_nmb_oor, TrainingParallelism(dp=8))
    assert estimate.latency_s > 0
    # tokens way out of range — must still raise because tokens is not listed:
    cost_tokens_unlisted = FsdpTrainingCost.from_artifact(
        artifact,
        batching_mode="pad_to_max",
        extrapolate=["num_microbatches"],
    )
    workload_tokens_oor = TrainingWorkload(
        sequence_lengths=(65536,) * 8,
        micro_batch_size=1,
        schedule="fsdp",
        recompute="full",
        optimizer="adam",
    )
    with pytest.raises(TrainingOutOfDomainError, match="tokens"):
        cost_tokens_unlisted.estimate(workload_tokens_oor, TrainingParallelism(dp=8))


def test_unknown_extrapolate_name_raises_at_construction():
    """A typo in the extrapolate list must raise ValueError naming the valid names."""
    artifact = FsdpTrainingCost.fit(_records(), validation_fraction=0.0)
    with pytest.raises(ValueError, match="num_microbatch"):  # typo without the final 's'
        FsdpTrainingCost.from_artifact(
            artifact,
            batching_mode="pad_to_max",
            extrapolate=["num_microbatch"],  # typo: should be num_microbatches
        )


def test_unknown_extrapolate_name_message_lists_valid_names():
    """The error message for an unknown name must include the valid names."""
    artifact = FsdpTrainingCost.fit(_records(), validation_fraction=0.0)
    with pytest.raises(ValueError) as exc_info:
        FsdpTrainingCost.from_artifact(
            artifact,
            batching_mode="pad_to_max",
            extrapolate=["totally_made_up"],
        )
    message = str(exc_info.value)
    # At least one valid name must appear in the message.
    assert "num_microbatches" in message or "tokens" in message


def test_permitted_extrapolation_is_recorded_in_diagnostics():
    """An out-of-range query on a permitted feature must leave a note in diagnostics."""
    artifact = FsdpTrainingCost.fit(_records(), validation_fraction=0.0)
    cost = FsdpTrainingCost.from_artifact(
        artifact,
        batching_mode="pad_to_max",
        extrapolate=["num_microbatches"],
    )
    # 8 microbatches per rank: micro_batch_size=1 and 8 seqs/rank → 8 microbatches (> [1,4])
    workload = TrainingWorkload(
        sequence_lengths=(512,) * 64,  # 64 total, dp=8 → 8 per rank → 8 microbatches
        micro_batch_size=1,
        schedule="fsdp",
        recompute="full",
        optimizer="adam",
    )
    estimate = cost.estimate(workload, TrainingParallelism(dp=8))
    notes = estimate.diagnostics.get("extrapolation_notes")
    assert notes is not None, "Permitted extrapolation must be recorded in diagnostics."
    assert any("num_microbatches" in note for note in notes), (
        f"Expected a note mentioning num_microbatches, got: {notes}"
    )


def test_in_range_values_produce_no_extrapolation_notes():
    """When all features are in range the diagnostics must not contain extrapolation_notes."""
    cost = _cost()
    workload = TrainingWorkload(
        sequence_lengths=(1024,) * 8,
        micro_batch_size=1,
        schedule="fsdp",
        recompute="full",
        optimizer="adam",
    )
    estimate = cost.estimate(workload, TrainingParallelism(dp=8))
    assert "extrapolation_notes" not in estimate.diagnostics


def test_the_quadratic_term_sums_per_sequence_squares_not_the_packed_square():
    """Attention does not cross sequence boundaries, so packing must not be charged its square.

    The shared `_records` fixture generates step times from a purely linear formula and uses one
    sequence per microbatch, so it cannot discriminate here. This builds records with a known
    quadratic ground truth expressed in `Sum(s^2)` and with packing, then checks the fit recovers
    that coefficient. Charging `(Sum s)^2` instead would need a coefficient smaller by the
    sequence count to fit the same data, so recovering the true value is the discriminating
    assertion.
    """
    quad_true = 2.0e-8
    records = []
    for length, n_seqs, num_microbatches in (
        (512, 1, 1),
        (512, 1, 4),
        (1024, 1, 1),
        (2048, 1, 1),
        (2048, 1, 2),
        (1024, 2, 1),
        (1024, 2, 2),
        (512, 4, 1),
    ):
        microbatches = tuple(
            SequenceStats.from_lengths([length] * n_seqs) for _ in range(num_microbatches)
        )
        tokens = float(max(item.sum_tokens for item in microbatches))
        sum_sq = float(max(item.sum_squared_tokens for item in microbatches))
        slots = float(num_microbatches)
        step = PER_STEP + slots * (
            TOKEN_COST * tokens + quad_true * sum_sq + MICROBATCH_OVERHEAD
        )
        record = _record(
            n_seqs=n_seqs, length=length, num_microbatches=num_microbatches
        )
        payload = replace(record.payload, step_times_s=(step,) * 8)
        records.append(replace(record, payload=payload))

    artifact = FsdpTrainingCost.fit(records, validation_fraction=0.0)
    entry = next(e for e in artifact.entries if e.selector["component"] == "latency")
    assert entry.coefficients["token_quadratic_cost"] == pytest.approx(quad_true, rel=0.05), (
        "the fitted quadratic coefficient must match the Sum(s^2) ground truth; a mismatch "
        "means the design row squared the packed total instead"
    )
