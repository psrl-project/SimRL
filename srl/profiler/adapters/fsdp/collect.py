"""Normalize raw FSDP candidate output into versioned profile records."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ...fsdp_search_space import FsdpCandidate
from ...schema import (
    ModelDescriptor,
    ProfileRecord,
    ProfileValidationError,
    RankMemory,
    RunDescriptor,
    SequenceStats,
    SoftwareDescriptor,
    TrainingPayload,
)
from ..base import (
    iter_candidate_dirs,
    load_sidecar_metrics,
    rank_memory_peaks,
    read_status,
    slice_warmup,
)
from .config import FsdpProfileConfig

# The schedule identifier for a backend with no pipeline. The schedule axis degenerates
# here rather than taking another value, and Hydraulis filters on it to keep Megatron
# and FSDP records from being fitted together.
FSDP_SCHEDULE = "fsdp"


def _validate_observed_mesh(
    metrics: dict[str, Any],
    candidate: FsdpCandidate,
    *,
    status: str,
    allow_incomplete: bool,
) -> None:
    """
    Cross-check the mesh the launcher built against the one requested.

    The launcher derives its mesh from the spec, but a mismatch would produce a record
    attributed to the wrong topology, and the cell key is keyed on that topology. This
    is where such a mismatch becomes a hard error instead of a plausible number.

    Absent-key contract:
    - If ``observed_mesh`` is missing and the run succeeded and ``allow_incomplete`` is
      false, a ``ProfileValidationError`` is raised.  Without the key there is no way to
      verify the topology, so the record must not silently enter the fit as though it were
      verified: that is the exact failure this function exists to prevent.
    - If ``allow_incomplete`` is true, a missing key is tolerated: the caller is
      deliberately collecting a partial sweep where the launcher may not have reached
      mesh-recording yet.
    - If the run did not succeed (e.g. ``failed`` or ``oom``), the launcher may have died
      before writing the sidecar at all, and that OOM record is still wanted data.
    """
    observed = metrics.get("observed_mesh")
    if observed is None:
        if status == "success" and not allow_incomplete:
            raise ProfileValidationError(
                f"Successful candidate {candidate.candidate_id!r} has no observed_mesh "
                f"in its sidecar. The topology cannot be verified, so the record cannot "
                f"be collected as though it were."
            )
        return
    expected = {
        "replicate_size": candidate.replicate_size,
        "shard_size": candidate.shard_size,
    }
    if {key: int(value) for key, value in observed.items()} != expected:
        raise ProfileValidationError(
            f"Launcher built mesh {observed!r} but the candidate requested {expected!r}."
        )


def _effective_model(
    config: FsdpProfileConfig,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
) -> ModelDescriptor:
    """
    Return the architecture that actually trained, not the checkpoint's.

    A layer-count override changes both the parameter count and the activation
    footprint, and the memory model derives its activation term from the layer count.
    A two-layer smoke run described as thirty-six layers would fit an activation scale
    near one eighteenth of the truth.
    """
    override = manifest.get("num_layers_override")
    if override is None:
        return config.model
    measured = metrics.get("parameter_count")
    return replace(
        config.model,
        num_layers=int(override),
        parameter_count=int(measured) if measured else config.model.parameter_count,
    )


def collect_fsdp_record(
    config: FsdpProfileConfig,
    candidate: FsdpCandidate,
    *,
    run_dir: str | Path | None = None,
) -> ProfileRecord:
    """Normalize one candidate's manifest, measurements, and status."""
    source_dir = Path(run_dir) if run_dir is not None else config.raw_root / candidate.candidate_id
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"FSDP manifest does not exist: {manifest_path}.")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    status, duration_s, error = read_status(source_dir)
    metrics = load_sidecar_metrics(source_dir) or {}
    _validate_observed_mesh(metrics, candidate, status=status, allow_incomplete=config.allow_incomplete)

    step_times = slice_warmup(
        tuple(float(item) for item in metrics.get("step_times_s", [])),
        config.warmup_steps,
    )
    if status == "success" and not step_times and not config.allow_incomplete:
        raise ProfileValidationError(
            f"No measured step time was found for successful candidate "
            f"{candidate.candidate_id!r}. The launcher must write srl_metrics.json."
        )

    rank_memory = tuple(
        RankMemory.from_dict(item) for item in metrics.get("rank_memory", [])
    )
    if status == "success" and not rank_memory and not config.allow_incomplete:
        raise ProfileValidationError(
            f"No rank memory peak was found for successful candidate "
            f"{candidate.candidate_id!r}. The launcher must write srl_metrics.json."
        )
    peak_allocated, peak_reserved = rank_memory_peaks(rank_memory)

    # An out-of-memory candidate has no sidecar: the launcher died before writing one. Its
    # shape comes from the candidate instead, repeated once per microbatch, because
    # `num_microbatches` is a swept axis and a single entry would only ever match nmb=1.
    # These records are wanted data (they bound the memory model from below) so a
    # fallback that mismatched would discard exactly the high-memory end of the sweep.
    shapes = metrics.get("microbatch_sequence_lengths") or [
        list(candidate.sequence_shape) for _ in range(candidate.num_microbatches)
    ]
    microbatches = tuple(
        SequenceStats.from_lengths([int(item) for item in shape]) for shape in shapes
    )
    if len(microbatches) != candidate.num_microbatches:
        raise ProfileValidationError(
            "Collected microbatch sequence shapes do not match num_microbatches."
        )

    payload = TrainingPayload(
        micro_batch_size=candidate.micro_batch_size,
        global_batch_size=candidate.global_batch_size,
        num_microbatches=candidate.num_microbatches,
        schedule=FSDP_SCHEDULE,
        recompute="full" if candidate.gradient_checkpointing else "none",
        optimizer=config.optimizer,
        microbatches=microbatches,
        step_times_s=step_times,
        timers_s={
            str(name): tuple(float(item) for item in values)
            for name, values in metrics.get("timers_s", {}).items()
        },
        rank_memory=rank_memory,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        parameter_dtype=config.parameter_dtype,
        gradient_dtype=config.gradient_dtype,
        optimizer_state_dtype=config.optimizer_state_dtype,
        # FSDP2 shards optimizer state unconditionally: there is no flag to record,
        # and unlike Megatron it also shards parameters and gradients.
        distributed_optimizer=True,
        parameter_count=(
            int(metrics.get("parameter_count", config.model.parameter_count or 0)) or None
        ),
        warmup_steps=config.warmup_steps,
    )
    return ProfileRecord(
        record_type="training",
        run=RunDescriptor(
            run_id=f"fsdp-{candidate.candidate_id}",
            repeat=candidate.repeat,
            status=status,
            duration_s=duration_s,
            raw_artifact=str(source_dir.relative_to(config.output_dir)),
            error=error,
        ),
        model=_effective_model(config, manifest, metrics),
        hardware=replace(config.hardware, gpu_count=candidate.world_size),
        software=SoftwareDescriptor(
            backend="fsdp",
            backend_version=config.backend_version,
            command=tuple(str(item) for item in manifest["command"]),
        ),
        parallelism=candidate.parallelism,
        payload=payload,
        # The mesh is configuration rather than data partitioning, so it travels as tags
        # and reaches the artifact through the cell selector.
        tags={
            "source": "srl-fsdp2-launcher",
            "shard_size": str(candidate.shard_size),
            "replicate_size": str(candidate.replicate_size),
            "optimizer_offload": str(candidate.optimizer_offload),
            "reshard_after_forward": str(candidate.reshard_after_forward),
            "sequence_parallel": str(candidate.sequence_parallel),
            "attn_implementation": config.attn_implementation,
        },
    )


def collect_fsdp_records(
    config: FsdpProfileConfig,
    candidates: list[FsdpCandidate],
) -> list[ProfileRecord]:
    """Collect all candidate directories that have adapter manifests."""
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    return [
        collect_fsdp_record(config, by_id[candidate_id], run_dir=run_dir)
        for candidate_id, run_dir in iter_candidate_dirs(
            config.raw_root,
            [candidate.candidate_id for candidate in candidates],
            allow_incomplete=config.allow_incomplete,
        )
    ]
