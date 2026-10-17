"""Job planning for the official ``vllm bench sweep serve`` workflow.

Prepares timing sweep shards, allocates GPUs and ports for concurrent
execution, and writes the parameter/manifest files the official
``vllm bench sweep serve`` CLI reads.
"""

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ...schema import ProfileValidationError
from .config import VLLMSweepConfig, _max_parallel_value


def _mapping_to_cli(value: dict[str, Any]) -> list[str]:
    command: list[str] = []
    for key, raw in value.items():
        option = f"--{key.replace('_', '-')}"
        if isinstance(raw, bool):
            if raw:
                command.append(option)
            continue
        if raw is None:
            continue
        if isinstance(raw, list):
            command.append(option)
            command.extend(str(item) for item in raw)
        elif isinstance(raw, dict):
            command.extend((option, json.dumps(raw, separators=(",", ":"))))
        else:
            command.extend((option, str(raw)))
    return command


def _mapping_with_override(
    value: dict[str, Any],
    key: str,
    override: Any,
) -> dict[str, Any]:
    normalized_key = key.replace("_", "-")
    result = {
        item_key: item
        for item_key, item in value.items()
        if item_key.replace("_", "-") != normalized_key
    }
    result[normalized_key] = override
    return result


def _prepare_vllm_sweep(
    config: VLLMSweepConfig,
    phase: str,
    *,
    serve_parameters: Sequence[dict[str, Any]],
    bench_parameters: Sequence[dict[str, Any]],
    shard_id: str | None = None,
    port: int | None = None,
    gpu_ids: Sequence[str] = (),
) -> tuple[list[str], Path]:
    if phase not in config.phase_parameters:
        raise ProfileValidationError(f"Phase {phase!r} is not configured.")
    if not serve_parameters or not bench_parameters:
        raise ProfileValidationError("A vLLM sweep shard must contain serve and bench parameters.")
    measurement = "timing"
    raw_root = config.output_dir / "raw" / f"vllm_{measurement}"
    control_dir = config.output_dir / "control" / "vllm" / phase
    raw_root.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)

    file_suffix = f"_{shard_id}" if shard_id else ""
    serialized_serve_parameters = [dict(item) for item in serve_parameters]
    serialized_bench_parameters = [dict(item) for item in bench_parameters]
    if port is not None:
        serialized_serve_parameters = [
            {
                key: value
                for key, value in item.items()
                if str(key).replace("_", "-") != "port"
            }
            for item in serialized_serve_parameters
        ]
        serialized_bench_parameters = [
            {
                key: value
                for key, value in item.items()
                if str(key).replace("_", "-") not in {"base-url", "port"}
            }
            for item in serialized_bench_parameters
        ]
    serve_params_path = control_dir / f"{measurement}{file_suffix}_serve_params.json"
    bench_params_path = control_dir / f"{measurement}{file_suffix}_bench_params.json"
    serve_params_path.write_text(
        json.dumps(serialized_serve_parameters, indent=2),
        encoding="utf-8",
    )
    bench_params_path.write_text(
        json.dumps(serialized_bench_parameters, indent=2),
        encoding="utf-8",
    )

    serve_args = {
        **config.serve_args,
        "enable-logging-iteration-details": True,
        # Async scheduling overlaps a step with its neighbours, so the iteration
        # log's elapsed time stops describing that step's own forward pass: a
        # 65k-token prefill can report microseconds while its cost lands on a
        # later step. Timing collection needs one measurement per step.
        "no-async-scheduling": True,
    }
    if port is not None:
        serve_args = _mapping_with_override(serve_args, "port", port)
    serve_command = [
        config.executable,
        "serve",
        config.model.name,
        *_mapping_to_cli(serve_args),
    ]
    default_bench_args: dict[str, Any] = {
        "model": config.model.name,
        "backend": "vllm",
        "endpoint": "/v1/completions",
        "dataset-name": "random",
        "request-rate": "inf",
        "save-detailed": True,
        "random-range-ratio": 0.0,
    }
    default_bench_args.update(config.bench_args)
    if port is not None:
        default_bench_args = _mapping_with_override(
            default_bench_args,
            "base-url",
            f"http://127.0.0.1:{port}",
        )
    bench_command = [
        config.executable,
        "bench",
        "serve",
        *_mapping_to_cli(default_bench_args),
    ]
    command = [
        config.executable,
        "bench",
        "sweep",
        "serve",
        "--serve-cmd",
        shlex.join(serve_command),
        "--bench-cmd",
        shlex.join(bench_command),
        "--serve-params",
        str(serve_params_path),
        "--bench-params",
        str(bench_params_path),
        "--output-dir",
        str(raw_root),
        "--experiment-name",
        f"{config.experiment_name}-{phase}-{measurement}"
        f"{f'-{shard_id}' if shard_id else ''}",
        "--num-runs",
        str(config.num_runs),
        "--server-ready-timeout",
        str(config.server_ready_timeout),
        "--show-stdout",
    ]
    manifest = {
        "adapter": "vllm",
        "phase": phase,
        "measurement": measurement,
        "shard_id": shard_id,
        "command": command,
        "serve_parameters": serialized_serve_parameters,
        "bench_parameters": serialized_bench_parameters,
        "gpu_ids": list(gpu_ids),
        "port": port,
        "backend_version": config.backend_version,
        "source_commit": config.source_commit,
        "measurement_source": "vllm-scheduler-iteration-log",
    }
    (control_dir / f"{measurement}{file_suffix}_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return command, config.raw_experiment_dir(phase, measurement, shard_id)


def prepare_vllm_sweep(
    config: VLLMSweepConfig,
    phase: str,
) -> tuple[list[str], Path]:
    """Write official sweep parameter files and return the executable command."""
    return _prepare_vllm_sweep(
        config,
        phase,
        serve_parameters=config.serve_parameters,
        bench_parameters=config.phase_parameters.get(phase, ()),
    )


@dataclass(frozen=True)
class VLLMSweepJob:
    """One isolated official sweep process in a deterministic GPU wave."""

    phase: str
    measurement: str
    shard_id: str | None
    wave: int
    gpu_ids: tuple[str, ...]
    port: int | None
    command: tuple[str, ...]
    experiment_dir: Path
    log_path: Path
    world_size: int
    expected_runs: int


@dataclass(frozen=True)
class _VLLMJobSpec:
    phase: str
    shard_id: str
    serve_parameters: tuple[dict[str, Any], ...]
    bench_parameters: tuple[dict[str, Any], ...]
    world_size: int


def _serve_world_size(
    config: VLLMSweepConfig,
    parameters: dict[str, Any],
) -> int:
    values = {
        str(key).replace("_", "-"): value
        for mapping in (config.serve_args, parameters)
        for key, value in mapping.items()
    }
    tp = int(values.get("tensor-parallel-size", config.hardware.gpu_count))
    pp = int(values.get("pipeline-parallel-size", 1))
    dp = int(values.get("data-parallel-size", 1))
    world_size = tp * pp * dp
    if min(tp, pp, dp) <= 0:
        raise ProfileValidationError(
            f"vLLM parallel sizes must be positive, got TP={tp}, PP={pp}, DP={dp}."
        )
    return world_size


def _visible_gpu_ids(
    config: VLLMSweepConfig,
    environment: dict[str, str],
    explicit: Sequence[str] | None,
) -> tuple[str, ...]:
    if explicit is not None:
        gpu_ids = tuple(str(item).strip() for item in explicit if str(item).strip())
    elif "CUDA_VISIBLE_DEVICES" in environment:
        value = environment["CUDA_VISIBLE_DEVICES"].strip()
        gpu_ids = () if value in {"", "-1"} else tuple(
            item.strip() for item in value.split(",") if item.strip()
        )
    else:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            gpu_ids = tuple(
                line.strip() for line in result.stdout.splitlines() if line.strip()
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            fallback_count = int(
                config.hardware.gpus_per_node or config.hardware.gpu_count
            )
            gpu_ids = tuple(str(index) for index in range(fallback_count))
    if not gpu_ids:
        raise ProfileValidationError("No visible GPUs are available for the vLLM profile.")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ProfileValidationError(
            f"CUDA_VISIBLE_DEVICES contains duplicate devices: {gpu_ids!r}."
        )
    return gpu_ids


def _available_ports(count: int, *, start: int) -> tuple[int, ...]:
    ports: list[int] = []
    candidate = start
    while len(ports) < count and candidate <= 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", candidate))
            except OSError:
                candidate += 1
                continue
        ports.append(candidate)
        candidate += 1
    if len(ports) != count:
        raise ProfileValidationError(
            f"Could not find {count} available TCP ports starting at {start}."
        )
    return tuple(ports)


def _partition_parameters(
    parameters: Sequence[dict[str, Any]],
    count: int,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    return tuple(
        tuple(parameters[index::count])
        for index in range(count)
    )


def _parallel_job_specs(
    config: VLLMSweepConfig,
    phases: Sequence[str],
    *,
    visible_gpu_count: int,
    max_parallel: int,
) -> list[_VLLMJobSpec]:
    specs: list[_VLLMJobSpec] = []
    for phase in phases:
        if phase not in config.phase_parameters:
            raise ProfileValidationError(f"Phase {phase!r} is not configured.")
        bench_parameters = config.phase_parameters[phase]
        if not bench_parameters:
            raise ProfileValidationError(f"vLLM phase {phase!r} has no benchmark points.")
        phase_shards: list[
            tuple[str, tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], int]
        ] = []
        shard_number = 0
        for serve_parameters in config.serve_parameters:
            world_size = _serve_world_size(config, serve_parameters)
            if world_size > visible_gpu_count:
                raise ProfileValidationError(
                    f"vLLM serve point needs {world_size} GPUs but only "
                    f"{visible_gpu_count} are visible."
                )
            shard_count = min(
                len(bench_parameters),
                max_parallel,
                visible_gpu_count // world_size,
            )
            for bench_shard in _partition_parameters(bench_parameters, shard_count):
                phase_shards.append(
                    (
                        f"s{shard_number:03d}",
                        (serve_parameters,),
                        bench_shard,
                        world_size,
                    )
                )
                shard_number += 1
        for shard_id, serve_shard, bench_shard, world_size in phase_shards:
            specs.append(
                _VLLMJobSpec(
                    phase=phase,
                    shard_id=shard_id,
                    serve_parameters=serve_shard,
                    bench_parameters=bench_shard,
                    world_size=world_size,
                )
            )
    return specs


def _configured_base_port(config: VLLMSweepConfig) -> int:
    normalized = {
        str(key).replace("_", "-"): value
        for key, value in config.serve_args.items()
    }
    return int(normalized.get("port", 8000))


def _write_jobs_manifest(
    config: VLLMSweepConfig,
    phase: str,
    jobs: Sequence[VLLMSweepJob],
    *,
    visible_devices: Sequence[str],
    max_parallel: int,
) -> None:
    path = (
        config.output_dir
        / "control"
        / "vllm"
        / phase
        / "jobs_manifest.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "visible_devices": list(visible_devices),
                "max_parallel": max_parallel,
                "jobs": [
                    {
                        "phase": job.phase,
                        "measurement": job.measurement,
                        "shard_id": job.shard_id,
                        "wave": job.wave,
                        "gpu_ids": list(job.gpu_ids),
                        "port": job.port,
                        "world_size": job.world_size,
                        "experiment_dir": str(job.experiment_dir),
                        "log_path": str(job.log_path),
                        "command": list(job.command),
                    }
                    for job in jobs
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def prepare_vllm_jobs(
    config: VLLMSweepConfig,
    phases: Sequence[str],
    *,
    max_parallel: int | str | None = None,
    visible_devices: Sequence[str] | None = None,
) -> list[VLLMSweepJob]:
    """Prepare isolated sweep jobs and deterministic GPU/port assignments."""
    phases = tuple(phases)
    if not phases:
        raise ProfileValidationError("At least one vLLM phase must be selected.")
    environment = {**os.environ, **config.environment}
    gpu_ids = _visible_gpu_ids(config, environment, visible_devices)
    requested = (
        config.max_parallel
        if max_parallel is None
        else _max_parallel_value(max_parallel)
    )
    world_sizes = [
        _serve_world_size(config, parameters)
        for parameters in config.serve_parameters
    ]
    if not world_sizes:
        raise ProfileValidationError("At least one vLLM serve point is required.")
    if max(world_sizes) > len(gpu_ids):
        raise ProfileValidationError(
            f"A vLLM serve point needs {max(world_sizes)} GPUs but only "
            f"{len(gpu_ids)} are visible."
        )
    hardware_limit = max(len(gpu_ids) // world_size for world_size in world_sizes)
    effective_max_parallel = min(requested or hardware_limit, hardware_limit)

    if effective_max_parallel <= 1:
        jobs: list[VLLMSweepJob] = []
        required_gpus = max(world_sizes)
        for wave, phase in enumerate(phases):
            command, experiment_dir = prepare_vllm_sweep(config, phase)
            jobs.append(
                VLLMSweepJob(
                    phase=phase,
                    measurement="timing",
                    shard_id=None,
                    wave=wave,
                    gpu_ids=gpu_ids[:required_gpus],
                    port=None,
                    command=tuple(command),
                    experiment_dir=experiment_dir,
                    log_path=config.sweep_log_path(phase, "timing"),
                    world_size=required_gpus,
                    expected_runs=(
                        len(config.serve_parameters)
                        * len(config.phase_parameters[phase])
                        * config.num_runs
                    ),
                )
            )
        for phase in phases:
            _write_jobs_manifest(
                config,
                phase,
                [job for job in jobs if job.phase == phase],
                visible_devices=gpu_ids,
                max_parallel=1,
            )
        return jobs

    specs = _parallel_job_specs(
        config,
        phases,
        visible_gpu_count=len(gpu_ids),
        max_parallel=effective_max_parallel,
    )
    concurrency_limit = min(effective_max_parallel, len(specs))
    ports = _available_ports(
        concurrency_limit,
        start=_configured_base_port(config),
    )
    waves: list[dict[str, Any]] = []
    assignments: list[tuple[_VLLMJobSpec, int, tuple[str, ...], int]] = []
    for spec in specs:
        selected_wave: int | None = None
        for wave_index, wave in enumerate(waves):
            if (
                len(wave["assignments"]) < concurrency_limit
                and len(wave["available_gpus"]) >= spec.world_size
            ):
                selected_wave = wave_index
                break
        if selected_wave is None:
            selected_wave = len(waves)
            waves.append(
                {
                    "available_gpus": list(gpu_ids),
                    "assignments": [],
                }
            )
        wave = waves[selected_wave]
        assigned_gpu_ids = tuple(wave["available_gpus"][: spec.world_size])
        del wave["available_gpus"][: spec.world_size]
        worker_index = len(wave["assignments"])
        wave["assignments"].append(spec)
        assignments.append(
            (spec, selected_wave, assigned_gpu_ids, ports[worker_index])
        )

    jobs = []
    for spec, wave, assigned_gpu_ids, port in assignments:
        command, experiment_dir = _prepare_vllm_sweep(
            config,
            spec.phase,
            serve_parameters=spec.serve_parameters,
            bench_parameters=spec.bench_parameters,
            shard_id=spec.shard_id,
            port=port,
            gpu_ids=assigned_gpu_ids,
        )
        jobs.append(
            VLLMSweepJob(
                phase=spec.phase,
                measurement="timing",
                shard_id=spec.shard_id,
                wave=wave,
                gpu_ids=assigned_gpu_ids,
                port=port,
                command=tuple(command),
                experiment_dir=experiment_dir,
                log_path=config.sweep_log_path(
                    spec.phase,
                    "timing",
                    spec.shard_id,
                ),
                world_size=spec.world_size,
                expected_runs=len(spec.bench_parameters) * config.num_runs,
            )
        )
    for phase in phases:
        _write_jobs_manifest(
            config,
            phase,
            [job for job in jobs if job.phase == phase],
            visible_devices=gpu_ids,
            max_parallel=concurrency_limit,
        )
    return jobs


def format_vllm_job_command(
    job: VLLMSweepJob,
    *,
    dry_run: bool = False,
) -> str:
    command = [*job.command, *(("--dry-run",) if dry_run else ())]
    devices = ",".join(job.gpu_ids)
    return f"CUDA_VISIBLE_DEVICES={shlex.quote(devices)} {shlex.join(command)}"
