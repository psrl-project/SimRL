"""Subprocess execution of official ``vllm bench sweep serve`` jobs.

Startups are admitted through a gate: a shared filesystem (e.g. CephFS) serving torch,
vLLM and the checkpoint to several servers booting at once slows every import to a crawl,
so each server misses the sweep's own ``server_ready_timeout`` and the shard dies before
it benchmarks anything. A shard therefore holds a startup permit from launch until its
server reports ready, and only `VLLMSweepConfig.max_concurrent_startups` shards may hold
one. Benchmarking itself stays fully parallel because the permit is released as soon as
the server serves traffic.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable, Sequence

from ...._logging import srl_logger
from .config import VLLMSweepConfig
from .prepare import VLLMSweepJob, format_vllm_job_command, prepare_vllm_jobs

_LEVELS_WITHOUT_ITERATION_LOGS = frozenset(
    {"WARN", "WARNING", "ERROR", "CRITICAL", "FATAL"}
)

# The official sweep prints `[BEGIN BENCHMARK]` only after `wait_until_ready` returns, so
# it marks the end of a shard's startup. The uvicorn banner is a fallback for sweep
# revisions that word the section header differently.
_SERVER_READY_MARKERS = (
    b"[BEGIN BENCHMARK]",
    b"Application startup complete",
)
_READY_MARKER_OVERLAP = max(len(marker) for marker in _SERVER_READY_MARKERS) - 1


def _sweep_runtime_environment(base: dict[str, str]) -> dict[str, str]:
    """Force settings required to capture scheduler iteration logs from a piped sweep."""
    environment = dict(base)
    environment["PYTHONUNBUFFERED"] = "1"
    level = environment.get("VLLM_LOGGING_LEVEL", "INFO").upper()
    if level in _LEVELS_WITHOUT_ITERATION_LOGS:
        srl_logger.warning(
            f"[vllm-profile] Overriding VLLM_LOGGING_LEVEL={level!r} to INFO so "
            "vLLM emits scheduler iteration timings."
        )
        environment["VLLM_LOGGING_LEVEL"] = "INFO"
    return environment


def _prefix_console_bytes(
    data: bytes,
    prefix: bytes,
    *,
    at_line_start: bool,
) -> tuple[bytes, bool]:
    """Insert a shard prefix after each newline or carriage return."""
    output = bytearray()
    for byte in data:
        if at_line_start:
            output += prefix
            at_line_start = False
        output.append(byte)
        if byte in {10, 13}:
            at_line_start = True
    return bytes(output), at_line_start


def _tee_subprocess_output(
    process: subprocess.Popen[bytes],
    log_handle,
    *,
    prefix: str,
    on_server_ready: Callable[[], None] | None = None,
) -> int:
    """Copy child output to the log file and stdout, reading chunks not lines.

    Line iteration deadlocks when several vLLM servers fill the 64KiB pipe with
    ``\\r`` progress bars. ``os.read`` returns as soon as any bytes are available.
    """
    stdout = process.stdout
    if stdout is None:
        raise RuntimeError("Failed to capture vLLM sweep output.")
    marker = f"[{prefix}] ".encode()
    at_line_start = True
    fd = stdout.fileno()
    pending_ready = on_server_ready is not None
    # A ready marker can straddle two reads, so re-scan it with the previous tail.
    tail = b""
    while True:
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        log_handle.write(chunk)
        log_handle.flush()
        if pending_ready:
            window = tail + chunk
            if any(item in window for item in _SERVER_READY_MARKERS):
                pending_ready = False
                tail = b""
                on_server_ready()
            else:
                tail = window[-_READY_MARKER_OVERLAP:]
        console, at_line_start = _prefix_console_bytes(
            chunk,
            marker,
            at_line_start=at_line_start,
        )
        sys.stdout.buffer.write(console)
        sys.stdout.buffer.flush()
    return process.wait()


class _StartupGate:
    """Bounds how many shards may sit in their pre-ready startup phase at once."""

    def __init__(self, permits: int, *, timeout: float) -> None:
        self._semaphore = threading.BoundedSemaphore(permits)
        self._timeout = timeout

    def acquire(self, label: str) -> bool:
        if self._semaphore.acquire(timeout=self._timeout):
            return True
        srl_logger.warning(
            f"[vllm-profile] Waited {self._timeout:.0f}s for a startup slot before "
            f"{label}; launching it alongside the server still booting."
        )
        return False

    def release(self) -> None:
        self._semaphore.release()


def _vllm_job_is_complete(job: VLLMSweepJob) -> bool:
    if not job.experiment_dir.exists():
        return False
    if len(list(job.experiment_dir.rglob("run=*.json"))) < job.expected_runs:
        return False
    return True


def _run_vllm_job(
    config: VLLMSweepConfig,
    job: VLLMSweepJob,
    *,
    resume: bool,
    gate: _StartupGate | None = None,
) -> None:
    label = f"{job.phase}/{job.measurement}/{job.shard_id or 'legacy'}"
    if _vllm_job_is_complete(job):
        srl_logger.info(f"[vllm-profile] Skipping complete {label}.")
        return
    command = list(job.command)
    if job.experiment_dir.exists() and "--resume" not in command:
        command.append("--resume")
    environment = _sweep_runtime_environment(
        {
            **os.environ,
            **config.environment,
            "CUDA_VISIBLE_DEVICES": ",".join(job.gpu_ids),
        }
    )
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_mode = "ab" if job.log_path.exists() else "wb"

    holds_permit = gate.acquire(label) if gate is not None else False
    released = threading.Event()

    def release_startup() -> None:
        # Invoked on readiness and again when the shard exits; release exactly once.
        if holds_permit and not released.is_set():
            released.set()
            gate.release()

    srl_logger.info(
        f"[vllm-profile] Starting {label} on GPUs {','.join(job.gpu_ids)}"
        f"{f' port {job.port}' if job.port is not None else ''}; "
        f"logs: {job.log_path}"
    )
    try:
        with job.log_path.open(log_mode) as log_handle:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
            return_code = _tee_subprocess_output(
                process,
                log_handle,
                prefix=label,
                on_server_ready=release_startup,
            )
    finally:
        release_startup()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _run_vllm_job_pool(
    config: VLLMSweepConfig,
    jobs: Sequence[VLLMSweepJob],
    *,
    resume: bool,
) -> None:
    """Run every shard whose GPUs/port are free, admitting startups through the gate.

    A shard that dies no longer cancels the queue behind it: shards are independent and
    ``--resume`` skips finished ones, so surviving the failure keeps a multi-hour sweep
    from being thrown away. Every failure is reported once the pool drains.
    """
    pending = list(jobs)
    running: dict[Future[None], VLLMSweepJob] = {}
    failures: list[tuple[VLLMSweepJob, BaseException]] = []
    used_gpus: set[str] = set()
    used_ports: set[int] = set()
    worker_count = max(
        sum(job.wave == wave for job in jobs)
        for wave in {job.wave for job in jobs}
    )
    gate = _StartupGate(
        min(config.max_concurrent_startups, worker_count),
        timeout=config.server_ready_timeout + config.startup_stagger_seconds,
    )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        while pending or running:
            for job in list(pending):
                if len(running) >= worker_count:
                    break
                if used_gpus.intersection(job.gpu_ids):
                    continue
                if job.port is not None and job.port in used_ports:
                    continue
                pending.remove(job)
                used_gpus.update(job.gpu_ids)
                if job.port is not None:
                    used_ports.add(job.port)
                running[
                    executor.submit(_run_vllm_job, config, job, resume=resume, gate=gate)
                ] = job
            if not running:
                raise RuntimeError(
                    "vLLM GPU scheduler could not place any pending sweep job."
                )
            completed, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in completed:
                job = running.pop(future)
                used_gpus.difference_update(job.gpu_ids)
                if job.port is not None:
                    used_ports.remove(job.port)
                error = future.exception()
                if error is None:
                    continue
                failures.append((job, error))
                srl_logger.error(
                    f"[vllm-profile] {job.phase}/{job.measurement}"
                    f"/{job.shard_id or 'legacy'} failed: {error}. "
                    f"See {job.log_path}."
                )
    if failures:
        summary = ", ".join(
            f"{job.phase}/{job.measurement}/{job.shard_id or 'legacy'}"
            for job, _ in failures
        )
        raise RuntimeError(
            f"{len(failures)} of {len(jobs)} vLLM sweep shards failed: {summary}. "
            "Rerun the same command to retry only the unfinished shards."
        ) from failures[0][1]


def run_vllm_profiles(
    config: VLLMSweepConfig,
    phases: Sequence[str],
    *,
    max_parallel: int | str | None = None,
    dry_run: bool = False,
    resume: bool = False,
    visible_devices: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Run all selected phases, packing isolated sweep shards onto visible GPUs."""
    jobs = prepare_vllm_jobs(
        config,
        phases,
        max_parallel=max_parallel,
        visible_devices=visible_devices,
    )
    if dry_run:
        for job in jobs:
            srl_logger.info(
                f"Prepared vLLM {job.phase!r} {job.measurement} dry run: "
                f"{format_vllm_job_command(job, dry_run=True)}."
            )
        return {
            phase: config.raw_experiment_dir(phase, "timing")
            for phase in phases
        }

    _run_vllm_job_pool(config, jobs, resume=resume)
    return {
        phase: config.raw_experiment_dir(phase, "timing")
        for phase in phases
    }


def run_vllm_sweep(
    config: VLLMSweepConfig,
    phase: str,
    *,
    dry_run: bool = False,
    resume: bool = False,
    max_parallel: int | str | None = None,
    visible_devices: Sequence[str] | None = None,
) -> Path:
    """Run one phase; retained as a compatibility wrapper."""
    run_vllm_profiles(
        config,
        [phase],
        max_parallel=max_parallel,
        dry_run=dry_run,
        resume=resume,
        visible_devices=visible_devices,
    )
    return config.raw_experiment_dir(phase, "timing")
