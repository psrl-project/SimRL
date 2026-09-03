"""Measure step latency and per-rank memory peaks inside the training loop.

Imports torch and megatron, so this module is only ever imported by the torchrun
child, never from the adapter package namespace.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.bridge.training.callbacks import Callback, CallbackContext

SIDECAR_NAME = "dryrun_metrics.json"


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Replace `path` in one step so a reader never sees a partial document."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class DryRunMeasurementCallback(Callback):
    """
    Record per-step wall time and per-rank CUDA peaks for one candidate.

    Timing uses a synchronized wall clock rather than CUDA events because the
    target model has an explicit update-overhead intercept: the measurement must
    include the host-side optimizer step and the data-parallel gradient reduce,
    not just GPU stream time. The bracket around `on_train_step_start` and
    `on_train_step_end` already spans the whole `train_step`.

    No barrier is taken on the timing path. A barrier would charge every rank for
    the slowest one, turning a per-rank measurement into a straggler measurement.
    """

    def __init__(self, *, run_dir: Path, spec: dict[str, Any]):
        self._run_dir = run_dir
        self._spec = spec
        self._seq_length = int(spec["seq_length"])
        self._micro_batch_size = int(spec["micro_batch_size"])
        self._expected_microbatches = int(spec["num_microbatches"])
        self._step_times: list[float] = []
        self._shapes: list[list[int]] = []
        self._parameter_count: int | None = None
        self._local_parameter_count: int | None = None
        self._static_allocated_bytes: int | None = None
        self._static_reserved_bytes: int | None = None
        self._started_at: float | None = None

    # -- training loop hooks ------------------------------------------------

    def on_train_start(self, context: CallbackContext) -> None:
        """
        Reset peak counters once the model and optimizer exist.

        Resetting here discards the transient construction spike (weight init and
        checkpoint conversion scratch) while keeping the static pool: resetting
        peak statistics does not free memory, so the already-reserved parameter,
        gradient, and optimizer state is immediately re-counted. That matters
        because the memory model fits the peak minus an analytic static inventory,
        so the peak has to include that inventory.
        """
        torch.cuda.synchronize()
        # Recorded per rank, not just on rank zero: with pipeline parallelism the
        # stages hold different layer counts, and the stage carrying the embedding or
        # output projection is heavier than a uniform split predicts.
        self._static_allocated_bytes = int(torch.cuda.memory_allocated())
        self._static_reserved_bytes = int(torch.cuda.max_memory_reserved())
        torch.cuda.reset_peak_memory_stats()
        self._local_parameter_count = sum(
            parameter.numel() for chunk in context.model for parameter in chunk.parameters()
        )
        self._parameter_count = _global_parameter_count(context)

    def on_train_step_start(self, context: CallbackContext) -> None:
        torch.cuda.synchronize()
        self._started_at = time.perf_counter()

    def on_train_step_end(self, context: CallbackContext) -> None:
        torch.cuda.synchronize()
        if self._started_at is None:
            raise RuntimeError("on_train_step_end fired without a matching start.")
        self._step_times.append(time.perf_counter() - self._started_at)
        self._started_at = None
        self._record_microbatch_shape()

    def on_train_end(self, context: CallbackContext) -> None:
        """Gather every rank's peak and let rank zero write one consistent document."""
        torch.cuda.synchronize()
        local = self.local_memory()
        if dist.is_initialized():
            gathered: list[Any] = [None] * _world_size()
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]
        if _rank() == 0:
            _write_json_atomic(self._run_dir / SIDECAR_NAME, self.sidecar(gathered))

    # -- measurement accessors ---------------------------------------------

    def local_memory(self) -> dict[str, int]:
        """Return this rank's peak allocated and reserved bytes."""
        allocated = int(torch.cuda.max_memory_allocated())
        reserved = int(torch.cuda.max_memory_reserved())
        return {
            "rank": _rank(),
            # Per-rank static state, so a pipeline stage's own share is visible rather
            # than only rank zero's.
            "static_allocated_bytes": self._static_allocated_bytes or 0,
            "local_parameter_count": self._local_parameter_count or 0,
            "allocated_bytes": allocated,
            # The schema requires reserved >= allocated. They are read from the
            # same allocator, so this only guards against a driver-level surprise.
            "reserved_bytes": max(reserved, allocated),
        }

    def sidecar(self, rank_memory: list[dict[str, int]]) -> dict[str, Any]:
        """
        Build the collector's sidecar document.

        `step_times_s` keeps its warmup entries. The collector owns the warmup
        policy and slices them off, so the launcher stays a pure measurer.
        """
        payload: dict[str, Any] = {
            "step_times_s": list(self._step_times),
            "timers_s": {},
            "rank_memory": sorted(
                (item for item in rank_memory if item is not None),
                key=lambda item: item["rank"],
            ),
            "microbatch_sequence_lengths": self._shapes,
        }
        if self._parameter_count:
            payload["parameter_count"] = int(self._parameter_count)
        if self._static_reserved_bytes is not None:
            # Diagnostic only. Comparing this against the analytic parameter
            # inventory is how we confirm the peak reset kept the static pool.
            payload["static_reserved_bytes_at_train_start"] = self._static_reserved_bytes
        return payload

    # -- internals ---------------------------------------------------------

    def _record_microbatch_shape(self) -> None:
        """
        Record the shape trained on and verify the derived microbatch count.

        Megatron derives `num_microbatches` from global batch, micro batch, and
        data-parallel size; it cannot be set directly. This is where a mismatch
        against the candidate becomes a hard error instead of a fit silently
        attributed to the wrong pipeline depth.
        """
        from megatron.core.num_microbatches_calculator import get_num_microbatches

        observed = int(get_num_microbatches())
        if observed != self._expected_microbatches:
            raise RuntimeError(
                f"Megatron derived {observed} microbatches but the candidate "
                f"requested {self._expected_microbatches}. Check global_batch_size, "
                "micro_batch_size, and the data-parallel size."
            )
        self._shapes = [[self._seq_length] * self._micro_batch_size] * observed


def _global_parameter_count(context: CallbackContext) -> int | None:
    """
    Sum parameters across the model-parallel groups.

    The count must be invariant across tensor and pipeline degrees so that cells
    differing only in topology share one key. Vocabulary padding depends on the
    tensor degree, so this is measured rather than derived.
    """
    local = sum(
        parameter.numel() for chunk in context.model for parameter in chunk.parameters()
    )
    if not dist.is_initialized():
        return int(local)
    try:
        from megatron.core import parallel_state

        group = parallel_state.get_model_parallel_group()
    except Exception:
        return None
    total = torch.tensor([local], dtype=torch.long, device="cuda")
    dist.all_reduce(total, group=group)
    return int(total.item())


def write_oom_sidecar(
    run_dir: Path,
    spec: dict[str, Any],
    error: BaseException,
) -> None:
    """
    Mark this rank as having run out of memory.

    Every rank writes its own marker and no collective is taken. An out-of-memory
    failure is usually local to one rank while the others sit in a collective, so
    any gather here would block until the communicator times out.

    Out-of-memory candidates are wanted data: they bound the memory model from
    below. The activation feature comes from the candidate itself, so the record
    stays useful without any measurement.
    """
    rank = _rank()
    try:
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        payload = {
            "rank": rank,
            "torch_oom": True,
            "device_free_bytes": int(free_bytes),
            "device_total_bytes": int(total_bytes),
            "device_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "seq_length": int(spec["seq_length"]),
            "num_microbatches": int(spec["num_microbatches"]),
            "message": str(error)[:2000],
        }
        _write_json_atomic(run_dir / f"oom_rank{rank}.json", payload)
    except Exception as marker_error:  # noqa: BLE001 - must never mask the OOM
        print(f"[dryrun] rank {rank} failed to write its OOM marker: {marker_error!r}")
