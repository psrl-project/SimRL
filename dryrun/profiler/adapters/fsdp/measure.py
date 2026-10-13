"""Measure step latency and per-rank memory peaks inside the FSDP training loop.

Takes collectives and reads the CUDA allocator, so this module is only ever imported
by the torchrun child, never from the adapter package namespace.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ..base import SIDECAR_NAME

__all__ = ["SIDECAR_NAME", "FsdpMeasurement", "is_nccl_memory_exhaustion", "write_oom_sidecar"]


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Replace `path` in one step so a reader never sees a partial document."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class FsdpMeasurement:
    """
    Record per-step wall time and per-rank CUDA peaks for one candidate.

    Timing uses a synchronized wall clock rather than CUDA events because the fitted
    model has an explicit per-step overhead term: the measurement must include the
    host-side optimizer step and the gradient reduce-scatter, not just stream time.

    No barrier is taken on the timing path. A barrier would charge every rank for the
    slowest one, turning a per-rank measurement into a straggler measurement.
    """

    def __init__(self, *, run_dir: Path, spec: dict[str, Any]):
        self._run_dir = run_dir
        self._spec = spec
        self._sequence_length = int(spec["sequence_length"])
        self._n_seqs_per_rank = int(spec["n_seqs_per_rank"])
        self._num_microbatches = int(spec["num_microbatches"])
        self._step_times: list[float] = []
        self._started_at: float | None = None
        self._static_allocated_bytes: int | None = None
        self._observed_mesh: dict[str, int] | None = None
        self._parameter_count: int | None = None
        self._collective_timers: dict[str, list[float]] = {}

    def on_train_start(self) -> None:
        """
        Reset peak counters once the model and optimizer exist.

        Resetting here discards the construction spike from weight loading and from the
        pre-shard full-model residency that `fully_shard` has already released, while
        keeping the static pool: resetting peak statistics does not free memory, so
        already-allocated parameter and gradient state is immediately re-counted. The
        memory model fits the peak *including* that inventory.

        The static pool itself is not read here: see `record_static_pool`.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def record_static_pool(self) -> None:
        """
        Record the live allocation that persists across steps.

        Must be called after the first optimizer step. AdamW allocates `exp_avg` and
        `exp_avg_sq` lazily, so before the first step the optimizer holds no state at
        all: measured on 8 ranks, `len(optimizer.state)` was 0 before and 12 after, and
        the reading moved from 24.5 MB to 160.5 MB against a 96.0 MB analytic figure.
        Capturing at train start would understate the pool several-fold.

        The live figure (`memory_allocated`), not the peak: the peak would fold in the
        pre-shard full-model residency and the first step's activations, neither of
        which is static.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self._static_allocated_bytes = int(torch.cuda.memory_allocated())

    def record_topology(
        self,
        *,
        replicate_size: int,
        shard_size: int,
        parameter_count: int,
    ) -> None:
        """
        Record what the process group actually built.

        The collector cross-checks this against the candidate. A mesh that silently
        differs from the requested one would otherwise yield a record attributed to
        the wrong topology, which is worse than no record at all.
        """
        self._observed_mesh = {"replicate_size": replicate_size, "shard_size": shard_size}
        self._parameter_count = parameter_count

    def step_begin(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._started_at = time.perf_counter()

    def record_collective_time(
        self,
        *,
        all_gather_s: float,
        reduce_scatter_s: float,
    ) -> None:
        """
        Record one step's collective kernel occupancy.

        These are summed `self_device_time_total` per rank: how long the NCCL kernels
        occupied the device, *not* how much wall time they exposed. With
        `reshard_after_forward=True` most of the all-gather overlaps compute on another
        stream, so the sum is an upper bound on the step's communication cost. A
        kernel-summation model that added these to compute time would over-predict.

        Not read by the regression. It is ground truth for the kernel-summation model
        that `latency.py` names as the principled next step, and measuring it now costs
        one profiled step where re-profiling the whole sweep later would not.
        """
        self._collective_timers.setdefault("all_gather", []).append(all_gather_s)
        self._collective_timers.setdefault("reduce_scatter", []).append(reduce_scatter_s)

    def step_end(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self._started_at is None:
            raise RuntimeError("step_end fired without a matching start.")
        self._step_times.append(time.perf_counter() - self._started_at)
        self._started_at = None

    def on_train_end(self) -> None:
        """Gather every rank's peak and let rank zero write one consistent document."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        local = self.local_memory()
        if dist.is_initialized():
            gathered: list[Any] = [None] * _world_size()
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]
        if _rank() == 0:
            _write_json_atomic(self._run_dir / SIDECAR_NAME, self.sidecar(gathered))

    def local_memory(self) -> dict[str, int]:
        """Return this rank's peak allocated and reserved bytes."""
        if not torch.cuda.is_available():
            return {"rank": _rank(), "allocated_bytes": 0, "reserved_bytes": 0}
        allocated = int(torch.cuda.max_memory_allocated())
        reserved = int(torch.cuda.max_memory_reserved())
        return {
            "rank": _rank(),
            "allocated_bytes": allocated,
            # The schema requires reserved >= allocated. Both come from the same
            # allocator, so this only guards against a driver-level surprise.
            "reserved_bytes": max(reserved, allocated),
        }

    def sidecar(self, rank_memory: list[dict[str, int]]) -> dict[str, Any]:
        """
        Build the collector's sidecar document.

        `step_times_s` keeps its warmup entries. The collector owns the warmup policy
        and slices them off, so the launcher stays a pure measurer and the retained
        window can be revisited without re-profiling.
        """
        payload: dict[str, Any] = {
            "step_times_s": list(self._step_times),
            "timers_s": {
                name: tuple(values) for name, values in self._collective_timers.items()
            },
            "rank_memory": sorted(
                (item for item in rank_memory if item is not None),
                key=lambda item: item["rank"],
            ),
            "microbatch_sequence_lengths": [
                [self._sequence_length] * self._n_seqs_per_rank
            ]
            * self._num_microbatches,
        }
        if self._parameter_count:
            payload["parameter_count"] = int(self._parameter_count)
        if self._observed_mesh is not None:
            payload["observed_mesh"] = self._observed_mesh
        if self._static_allocated_bytes is not None:
            # Diagnostic only, and the key name predates the capture point: this is read
            # after the first optimizer step, not at train start, because the moments are
            # allocated lazily. It is parameters plus master weights plus optimizer
            # moments at whatever dtype the optimizer actually produced: that dtype
            # follows configuration, so no fixed bytes-per-parameter formula describes it.
            # Compare it against the analytic FSDP inventory; it does not predict one.
            payload["static_allocated_bytes_at_train_start"] = self._static_allocated_bytes
        return payload


_NCCL_MEMORY_PHRASES = (
    "failed to cuda calloc",
    "out of memory",
)


def is_nccl_memory_exhaustion(error: BaseException) -> bool:
    """
    Decide whether a `DistBackendError` is caused by memory exhaustion.

    Returns True only when the error message contains a known memory-allocation
    failure phrase (case-insensitive). A bare `ncclUnhandledCudaError` without
    a memory phrase could be a genuine driver fault and must stay classified as
    `failed`, so it is not sufficient on its own.

    Phrases matched against the lower-cased message:
    - ``failed to cuda calloc``: the NCCL buffer allocation that appears in the
      actual sweep failure (``Failed to CUDA calloc 10485760 bytes``).
    - ``out of memory``: the common CUDA allocator phrasing.
    """
    text = str(error).lower()
    return any(phrase in text for phrase in _NCCL_MEMORY_PHRASES)


def write_oom_sidecar(
    run_dir: Path,
    spec: dict[str, Any],
    error: BaseException,
    *,
    oom_source: str = "torch_oom",
) -> None:
    """
    Mark this rank as having run out of memory.

    Every rank writes its own marker and no collective is taken. An out-of-memory
    failure is usually local to one rank while the others sit in a collective, so any
    gather here would block until the communicator times out.

    Out-of-memory candidates are wanted data: they bound the memory model from below.
    The activation shape comes from the candidate itself, so the record stays useful
    without any measurement.

    Args:
        oom_source (str): Identifies the detection route.
            ``torch_oom`` means a ``torch.OutOfMemoryError`` was caught by exact type.
            ``nccl_alloc`` means a ``DistBackendError`` was classified as memory
            exhaustion via text matching on the error message.
    """
    rank = _rank()
    try:
        payload: dict[str, Any] = {
            "rank": rank,
            "torch_oom": True,
            "oom_source": oom_source,
            "sequence_length": int(spec["sequence_length"]),
            "n_seqs_per_rank": int(spec["n_seqs_per_rank"]),
            "message": str(error)[:2000],
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            payload["device_free_bytes"] = int(free_bytes)
            payload["device_total_bytes"] = int(total_bytes)
            payload["device_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        _write_json_atomic(run_dir / f"oom_rank{rank}.json", payload)
    except Exception as marker_error:  # noqa: BLE001 - must never mask the OOM
        print(f"[dryrun] rank {rank} failed to write its OOM marker: {marker_error!r}")
