"""Normalizing raw FSDP candidate output into profile records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dryrun.profiler.adapters.fsdp import (
    FsdpProfileConfig,
    collect_fsdp_record,
    collect_fsdp_records,
    prepare_fsdp_candidate,
)
from dryrun.profiler.fsdp_search_space import FsdpCandidate
from dryrun.profiler.schema import (
    HardwareDescriptor,
    ModelDescriptor,
    ProfileValidationError,
    TrainingPayload,
)


def _model() -> ModelDescriptor:
    return ModelDescriptor(
        name="Qwen3-8B",
        architecture="Qwen3ForCausalLM",
        parameter_count=8_190_735_360,
        num_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        extra={"model_type": "qwen3", "vocab_size": 151936},
    )


def _config(tmp_path: Path, **overrides) -> FsdpProfileConfig:
    defaults = {
        "output_dir": tmp_path,
        "model": _model(),
        "hardware": HardwareDescriptor(
            gpu_name="H20", gpu_count=8, gpu_memory_mib=97871, node_count=1, gpus_per_node=8
        ),
        "model_path": "/models/Qwen3-8B",
    }
    return FsdpProfileConfig(**{**defaults, **overrides})


def _candidate(**overrides) -> FsdpCandidate:
    defaults = {
        "shard_size": 8,
        "replicate_size": 1,
        "n_seqs_per_rank": 2,
        "sequence_length": 1024,
        "gradient_checkpointing": True,
        "optimizer_offload": False,
        "reshard_after_forward": True,
    }
    return FsdpCandidate(**{**defaults, **overrides})


def _sidecar(candidate: FsdpCandidate, **overrides) -> dict:
    defaults = {
        "step_times_s": [0.5, 0.4, 0.35] + [0.30, 0.31, 0.30, 0.29, 0.30, 0.31, 0.30, 0.29, 0.30],
        "timers_s": {},
        "rank_memory": [
            {"rank": rank, "allocated_bytes": 1_000_000 + rank, "reserved_bytes": 2_000_000 + rank}
            for rank in range(candidate.world_size)
        ],
        "microbatch_sequence_lengths": [
            [candidate.sequence_length] * candidate.n_seqs_per_rank
        ]
        * candidate.num_microbatches,
        "parameter_count": 8_190_735_360,
        "observed_mesh": {
            "replicate_size": candidate.replicate_size,
            "shard_size": candidate.shard_size,
        },
    }
    return {**defaults, **overrides}


def _stage(config, candidate, *, status="success", sidecar=None) -> Path:
    _, run_dir = prepare_fsdp_candidate(config, candidate)
    (run_dir / "status.json").write_text(
        json.dumps({"status": status, "duration_s": 30.0}), encoding="utf-8"
    )
    if sidecar is not None:
        (run_dir / "dryrun_metrics.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return run_dir


def test_collect_produces_a_training_record_with_the_fsdp_schedule(tmp_path: Path):
    config, candidate = _config(tmp_path), _candidate()
    _stage(config, candidate, sidecar=_sidecar(candidate))
    record = collect_fsdp_record(config, candidate)
    assert record.record_type == "training"
    assert isinstance(record.payload, TrainingPayload)
    assert record.payload.schedule == "fsdp"
    assert record.software.backend == "fsdp"
    assert record.run.status == "success"


def test_collect_slices_the_warmup_steps(tmp_path: Path):
    config, candidate = _config(tmp_path, warmup_steps=4), _candidate()
    _stage(config, candidate, sidecar=_sidecar(candidate))
    record = collect_fsdp_record(config, candidate)
    # Twelve measured steps minus four warmup leaves eight.
    assert len(record.payload.step_times_s) == 8
    assert record.payload.warmup_steps == 4


def test_collect_takes_the_worst_rank_peak(tmp_path: Path):
    config, candidate = _config(tmp_path), _candidate()
    _stage(config, candidate, sidecar=_sidecar(candidate))
    record = collect_fsdp_record(config, candidate)
    assert record.payload.peak_allocated_bytes == 1_000_007
    assert record.payload.peak_reserved_bytes == 2_000_007


def test_collect_reports_dp_equal_to_the_world_size(tmp_path: Path):
    config, candidate = _config(tmp_path), _candidate(shard_size=4, replicate_size=2)
    _stage(config, candidate, sidecar=_sidecar(candidate))
    record = collect_fsdp_record(config, candidate)
    assert record.parallelism.dp == 8
    assert record.hardware.gpu_count == 8


def test_collect_tags_the_mesh_so_the_cell_key_can_read_it(tmp_path: Path):
    config, candidate = _config(tmp_path), _candidate(shard_size=4, replicate_size=2)
    _stage(config, candidate, sidecar=_sidecar(candidate))
    record = collect_fsdp_record(config, candidate)
    assert record.tags["shard_size"] == "4"
    assert record.tags["replicate_size"] == "2"
    assert record.tags["optimizer_offload"] == "False"
    assert record.tags["reshard_after_forward"] == "True"


def test_collect_rejects_a_mesh_that_differs_from_the_candidate(tmp_path: Path):
    """A silently wrong mesh would attribute the measurement to the wrong topology."""
    config, candidate = _config(tmp_path), _candidate(shard_size=8, replicate_size=1)
    sidecar = _sidecar(candidate, observed_mesh={"replicate_size": 2, "shard_size": 4})
    _stage(config, candidate, sidecar=sidecar)
    with pytest.raises(ProfileValidationError, match="mesh"):
        collect_fsdp_record(config, candidate)


def test_collect_uses_the_measured_parameter_count(tmp_path: Path):
    config, candidate = _config(tmp_path), _candidate()
    _stage(config, candidate, sidecar=_sidecar(candidate, parameter_count=1234))
    record = collect_fsdp_record(config, candidate)
    assert record.payload.parameter_count == 1234


def test_collect_describes_an_overridden_layer_count(tmp_path: Path):
    """The memory model derives activation scale from the layer count."""
    config = _config(tmp_path, num_layers_override=2)
    candidate = _candidate()
    _stage(config, candidate, sidecar=_sidecar(candidate, parameter_count=500_000))
    record = collect_fsdp_record(config, candidate)
    assert record.model.num_layers == 2
    assert record.model.parameter_count == 500_000


def test_collect_carries_an_accumulated_microbatch_count_into_the_payload(tmp_path: Path):
    """The latency fit reads `num_microbatches` as the slots feature, so it must survive."""
    config, candidate = _config(tmp_path), _candidate(num_microbatches=3)
    _stage(config, candidate, sidecar=_sidecar(candidate))
    record = collect_fsdp_record(config, candidate)
    assert record.payload.num_microbatches == 3
    assert len(record.payload.microbatches) == 3
    # The schema's own invariant, which a mis-derived count would break.
    assert record.payload.global_batch_size == (
        candidate.n_seqs_per_rank * candidate.world_size * 3
    )


def test_collect_rejects_a_microbatch_count_that_differs_from_the_candidate(tmp_path: Path):
    """A sidecar reporting fewer microbatches than were requested is a mis-run candidate."""
    config, candidate = _config(tmp_path), _candidate(num_microbatches=3)
    sidecar = _sidecar(
        candidate,
        microbatch_sequence_lengths=[[candidate.sequence_length] * candidate.n_seqs_per_rank],
    )
    _stage(config, candidate, sidecar=sidecar)
    with pytest.raises(ProfileValidationError, match="num_microbatches"):
        collect_fsdp_record(config, candidate)


def test_collect_requires_step_times_for_a_successful_run(tmp_path: Path):
    config, candidate = _config(tmp_path), _candidate()
    _stage(config, candidate, sidecar=_sidecar(candidate, step_times_s=[]))
    with pytest.raises(ProfileValidationError, match="step time"):
        collect_fsdp_record(config, candidate)


def test_collect_tolerates_a_missing_sidecar_when_allowed(tmp_path: Path):
    config = _config(tmp_path, allow_incomplete=True)
    candidate = _candidate()
    _stage(config, candidate)
    record = collect_fsdp_record(config, candidate)
    assert record.payload.step_times_s == ()


def test_collect_keeps_an_oom_candidate_as_a_memory_lower_bound(tmp_path: Path):
    config, candidate = _config(tmp_path), _candidate()
    _, run_dir = prepare_fsdp_candidate(config, candidate)
    (run_dir / "status.json").write_text(
        json.dumps({"status": "failed", "duration_s": 5.0}), encoding="utf-8"
    )
    (run_dir / "oom_rank3.json").write_text(json.dumps({"rank": 3}), encoding="utf-8")
    record = collect_fsdp_record(config, candidate)
    assert record.run.status == "oom"
    assert "rank(s) 3" in record.run.error
    # The activation shape comes from the candidate, so the record is usable unmeasured.
    assert record.payload.microbatches[0].count == candidate.n_seqs_per_rank


def test_collect_records_iterates_every_prepared_candidate(tmp_path: Path):
    config = _config(tmp_path)
    candidates = [_candidate(), _candidate(sequence_length=2048)]
    for candidate in candidates:
        _stage(config, candidate, sidecar=_sidecar(candidate))
    records = collect_fsdp_records(config, candidates)
    assert len(records) == 2
    assert {record.run.run_id for record in records} == {
        f"fsdp-{candidate.candidate_id}" for candidate in candidates
    }


def test_collect_records_raises_on_a_missing_candidate(tmp_path: Path):
    config = _config(tmp_path)
    with pytest.raises(FileNotFoundError):
        collect_fsdp_records(config, [_candidate()])


def test_collect_round_trips_the_measured_smoke_sidecar(tmp_path: Path):
    """The collector must accept the document the launcher actually writes.

    The fixture was captured from a real 8×H20 smoke run. Staging it through
    ``collect_fsdp_record`` pins the round trip end-to-end, so any breakage in
    field parsing surfaces immediately rather than at fit time.
    """
    fixture = Path("tests/fixtures/fsdp/dryrun_metrics.json")
    # The candidate parameters here must match the fixture's observed_mesh and
    # microbatch_sequence_lengths exactly, or the mesh cross-check will reject the record.
    candidate = _candidate(
        shard_size=8,
        replicate_size=1,
        n_seqs_per_rank=1,
        sequence_length=512,
        gradient_checkpointing=True,
        optimizer_offload=False,
        reshard_after_forward=True,
    )
    config = _config(tmp_path, warmup_steps=4)
    _, run_dir = prepare_fsdp_candidate(config, candidate)
    (run_dir / "status.json").write_text(
        json.dumps({"status": "success", "duration_s": 60.0}), encoding="utf-8"
    )
    (run_dir / "dryrun_metrics.json").write_text(fixture.read_text(), encoding="utf-8")

    record = collect_fsdp_record(config, candidate)

    # Twelve raw steps minus four warmup must yield eight retained.
    assert len(record.payload.step_times_s) == 8
    assert record.payload.schedule == "fsdp"
    assert record.parallelism.dp == 8
    # Both memory peaks must be populated — the fixture records all eight ranks.
    assert record.payload.peak_allocated_bytes is not None
    assert record.payload.peak_reserved_bytes is not None


def test_validate_observed_mesh_raises_when_absent_on_success(tmp_path: Path):
    """A successful run without observed_mesh cannot be verified and must be rejected.

    The cell key is keyed on topology; silently accepting an unverifiable record would
    contaminate the wrong cost-model cell with no indication that the topology was never
    cross-checked.
    """
    config, candidate = _config(tmp_path), _candidate()
    sidecar = _sidecar(candidate)
    del sidecar["observed_mesh"]
    _stage(config, candidate, sidecar=sidecar)
    with pytest.raises(ProfileValidationError, match="observed_mesh"):
        collect_fsdp_record(config, candidate)


def test_validate_observed_mesh_tolerates_absent_key_when_allow_incomplete(tmp_path: Path):
    """A partial sweep may not have reached mesh-recording yet; incomplete mode must pass."""
    config = _config(tmp_path, allow_incomplete=True)
    candidate = _candidate()
    sidecar = _sidecar(candidate)
    del sidecar["observed_mesh"]
    _stage(config, candidate, sidecar=sidecar)
    # Must not raise.
    record = collect_fsdp_record(config, candidate)
    assert record.run.status == "success"


def test_validate_observed_mesh_tolerates_absent_key_on_non_success_run(tmp_path: Path):
    """An OOM or failed run may have died before writing the sidecar; the record is still wanted."""
    config, candidate = _config(tmp_path), _candidate()
    # Stage without a sidecar (launcher died before writing dryrun_metrics.json).
    _, run_dir = prepare_fsdp_candidate(config, candidate)
    (run_dir / "status.json").write_text(
        json.dumps({"status": "failed", "duration_s": 5.0}), encoding="utf-8"
    )
    (run_dir / "oom_rank0.json").write_text(json.dumps({"rank": 0}), encoding="utf-8")
    # Must not raise even though there is no sidecar and therefore no observed_mesh.
    record = collect_fsdp_record(config, candidate)
    assert record.run.status == "oom"
