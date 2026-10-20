"""Normalize raw Megatron candidate output into versioned profile records."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

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
from ...search_space import TrainingCandidate
from ..base import (
    iter_candidate_dirs,
    load_sidecar_metrics,
    rank_memory_peaks,
    read_status,
    slice_warmup,
)
from .config import MegatronBridgeConfig


def _validate_resolved_config(
    run_dir: Path,
    manifest: dict[str, Any],
    candidate: TrainingCandidate,
) -> Path | None:
    """
    Cross-check the config Megatron actually resolved against the candidate.

    The launcher mutates a `ConfigContainer` in Python. This reads back what rank
    zero serialized, so a mutation that failed to land cannot pass unnoticed.
    """
    configured = Path(manifest.get("resolved_config", run_dir / "ConfigContainer.yaml"))
    path = configured if configured.exists() else run_dir / "ConfigContainer.yaml"
    if not path.exists():
        return None
    from omegaconf import OmegaConf

    resolved = OmegaConf.load(path)
    # Derived the same way as `prepare.build_launch_spec`.
    expected_seq_length = (
        sum(candidate.sequence_shape)
        if candidate.packing == "thd"
        else max(candidate.sequence_shape)
    )
    expected: dict[str, Any] = {
        "model.tensor_model_parallel_size": candidate.parallelism.tp,
        "model.pipeline_model_parallel_size": candidate.parallelism.pp,
        "model.context_parallel_size": candidate.parallelism.cp,
        "model.seq_length": expected_seq_length,
        "train.micro_batch_size": candidate.micro_batch_size,
        "train.global_batch_size": candidate.global_batch_size,
    }
    # Verified only for a MoE run. A dense recipe leaves `expert_model_parallel_size` at
    # the library default rather than at the candidate's `ep`, and comparing them would
    # fail every dense candidate.
    #
    # `expert_tensor_parallel_size` is checked against 1, not against `tp`: the launcher
    # pins it there on purpose, because Megatron's default of following TP would force
    # `ep=1` at every mesh whose memory fits this model and the EP axis would vanish. See
    # `launch.py` for the arithmetic. Asserting `tp` here instead rejected a sweep whose
    # configs were exactly right.
    if OmegaConf.select(resolved, "model.num_moe_experts"):
        expected["model.expert_model_parallel_size"] = candidate.parallelism.ep
        expected["model.expert_tensor_parallel_size"] = 1
    for key, expected_value in expected.items():
        actual = OmegaConf.select(resolved, key)
        if actual is not None and int(actual) != int(expected_value):
            raise ProfileValidationError(
                f"Resolved ConfigContainer value {key}={actual!r} does not match "
                f"candidate value {expected_value!r}."
            )
    granularity = OmegaConf.select(resolved, "model.recompute_granularity")
    if candidate.recompute == "full" and granularity != "full":
        raise ProfileValidationError(
            f"Candidate requested full recompute but the resolved config has "
            f"recompute_granularity={granularity!r}."
        )
    return path


def _effective_model(
    config: MegatronBridgeConfig,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
) -> ModelDescriptor:
    """
    Return the architecture that actually trained, not the checkpoint's.

    A layer-count override changes both the parameter count and the activation
    footprint, and the memory model derives its activation term from the layer count.
    Leaving the descriptor at the checkpoint's depth would silently misprice the run:
    a two-layer smoke run described as thirty-six layers fits an activation scale
    near one half rather than near one.
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


def collect_bridge_record(
    config: MegatronBridgeConfig,
    candidate: TrainingCandidate,
    *,
    run_dir: str | Path | None = None,
) -> ProfileRecord:
    """Normalize one candidate's resolved config, measurements, and status."""
    source_dir = Path(run_dir) if run_dir is not None else config.raw_root / candidate.candidate_id
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Megatron-Bridge manifest does not exist: {manifest_path}.")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    resolved_config_path = _validate_resolved_config(source_dir, manifest, candidate)
    status, duration_s, error = read_status(source_dir)
    metrics = load_sidecar_metrics(source_dir) or {}

    step_times = slice_warmup(
        tuple(float(item) for item in metrics.get("step_times_s", [])),
        config.warmup_steps,
    )
    if status == "success" and not step_times and not config.allow_incomplete:
        raise ProfileValidationError(
            f"No measured step time was found for successful candidate "
            f"{candidate.candidate_id!r}. The launcher must write srl_metrics.json."
        )

    if "rank_memory" in metrics:
        rank_memory = tuple(RankMemory.from_dict(item) for item in metrics["rank_memory"])
    else:
        rank_memory = ()
    if status == "success" and not rank_memory and not config.allow_incomplete:
        raise ProfileValidationError(
            f"No rank memory peak was found for successful candidate "
            f"{candidate.candidate_id!r}. The launcher must write srl_metrics.json."
        )

    effective_shape = tuple(int(item) for item in manifest["effective_sequence_shape"])
    if "microbatch_sequence_lengths" in metrics:
        shapes = [
            tuple(int(item) for item in shape)
            for shape in metrics["microbatch_sequence_lengths"]
        ]
    else:
        shapes = [effective_shape] * candidate.num_microbatches
    microbatches = tuple(SequenceStats.from_lengths(shape) for shape in shapes)
    if len(microbatches) != candidate.num_microbatches:
        raise ProfileValidationError(
            "Collected microbatch sequence shapes do not match num_microbatches."
        )

    peak_allocated, peak_reserved = rank_memory_peaks(rank_memory)
    timers = {
        str(name): tuple(float(item) for item in values)
        for name, values in metrics.get("timers_s", {}).items()
    }
    hardware = replace(config.hardware, gpu_count=int(candidate.parallelism.world_size))
    model = _effective_model(config, manifest, metrics)
    run = RunDescriptor(
        run_id=f"megatron-bridge-{candidate.candidate_id}",
        repeat=candidate.repeat,
        status=status,
        duration_s=duration_s,
        raw_artifact=str(source_dir.relative_to(config.output_dir)),
        error=error,
    )
    payload = TrainingPayload(
        micro_batch_size=candidate.micro_batch_size,
        global_batch_size=candidate.global_batch_size,
        num_microbatches=candidate.num_microbatches,
        schedule=candidate.schedule,
        recompute=candidate.recompute,
        optimizer=config.optimizer,
        microbatches=microbatches,
        step_times_s=step_times,
        timers_s=timers,
        rank_memory=rank_memory,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        parameter_dtype=config.parameter_dtype,
        gradient_dtype=config.gradient_dtype,
        optimizer_state_dtype=config.optimizer_state_dtype,
        distributed_optimizer=config.distributed_optimizer,
        parameter_count=(
            int(metrics.get("parameter_count", config.model.parameter_count or 0)) or None
        ),
        warmup_steps=config.warmup_steps,
        packing=candidate.packing,
    )
    software = SoftwareDescriptor(
        backend="megatron_bridge",
        backend_version=config.backend_version,
        source_commit=config.source_commit,
        command=tuple(str(item) for item in manifest["command"]),
    )
    return ProfileRecord(
        record_type="training",
        run=run,
        model=model,
        hardware=hardware,
        software=software,
        parallelism=candidate.parallelism,
        payload=payload,
        tags={
            "source": "megatron-bridge-library-recipe",
            "recompute_method": str(manifest.get("recompute_method", "")),
            "resolved_config": (
                str(resolved_config_path.relative_to(config.output_dir))
                if resolved_config_path is not None
                and resolved_config_path.is_relative_to(config.output_dir)
                else ""
            ),
        },
    )


def collect_bridge_records(
    config: MegatronBridgeConfig,
    candidates: list[TrainingCandidate],
) -> list[ProfileRecord]:
    """Collect all candidate directories that have adapter manifests."""
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    return [
        collect_bridge_record(config, by_id[candidate_id], run_dir=run_dir)
        for candidate_id, run_dir in iter_candidate_dirs(
            config.raw_root,
            [candidate.candidate_id for candidate in candidates],
            allow_incomplete=config.allow_incomplete,
        )
    ]
