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

SIDECAR_NAME = "srl_metrics.json"


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Replace `path` in one step so a reader never sees a partial document."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class SrlMeasurementCallback(Callback):
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
        self._packing = str(spec.get("packing", "bshd"))
        # TP-aligned exactly as the launcher's THD iterator aligns them. Recording
        # the unaligned lengths would describe a buffer that never ran.
        _tp = max(1, int(spec.get("parallelism", {}).get("tp", 1)))
        self._sequence_lengths = [
            length + (-length % _tp)
            for length in (int(item) for item in spec.get("sequence_lengths", ()))
        ]
        self._step_times: list[float] = []
        self._shapes: list[list[int]] = []
        self._parameter_count: int | None = None
        self._local_parameter_count: int | None = None
        self._static_allocated_bytes: int | None = None
        self._static_reserved_bytes: int | None = None
        self._started_at: float | None = None
        self._trace: Any = None

    # --- training loop hooks ------------------------------------------------

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
        self._maybe_start_trace()
        self._started_at = time.perf_counter()

    def on_train_step_end(self, context: CallbackContext) -> None:
        torch.cuda.synchronize()
        if self._started_at is None:
            raise RuntimeError("on_train_step_end fired without a matching start.")
        self._step_times.append(time.perf_counter() - self._started_at)
        self._maybe_stop_trace()
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

    # --- measurement accessors ---------------------------------------------

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

    # --- kernel tracing, off unless asked for -------------------------------

    def _maybe_start_trace(self) -> None:
        """Begin a CUDA kernel trace when `SIMRL_TRACE_STEP` names this step.

        Off by default and deliberately so: the profiler perturbs the very step times
        the latency model is fitted against, so a traced run must not also be a
        measured run. It exists to answer "which kernels does a step actually
        execute", which is the one question `ComposedTraining`'s per-op decomposition
        cannot answer from the cost side. The composition can only report the ops it
        already knows about, and a residual that is *none* of them is invisible to it.

        Only rank 0 traces. The ranks hold different pipeline stages, so a trace per
        rank would need stage-aware aggregation to mean anything, and rank 0's stage
        is the one carrying the embedding and the loss.
        """
        target = os.environ.get("SIMRL_TRACE_STEP")
        if target is None or self._trace is not None or _rank() != 0:
            return
        if len(self._step_times) != int(target):
            return
        self._trace = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        )
        self._trace.__enter__()

    def _maybe_stop_trace(self) -> None:
        """Close the trace and write both a raw export and a kernel-time summary."""
        if self._trace is None:
            return
        self._trace.__exit__(None, None, None)
        destination = self._run_dir / f"trace_step{len(self._step_times) - 1}.json"
        self._trace.export_chrome_trace(str(destination))
        _write_json_atomic(
            self._run_dir / f"kernel_totals_step{len(self._step_times) - 1}.json",
            _kernel_totals(self._trace),
        )
        self._trace = None

    # --- internals ---------------------------------------------------------

    def _record_microbatch_shape(self) -> None:
        """
        Record the shape trained on and verify the derived microbatch count.

        Megatron derives `num_microbatches` from global batch, micro batch, and
        data-parallel size, and it cannot be set directly. This is where a mismatch
        against the candidate becomes a hard error instead of a fit silently
        attributed to the wrong pipeline depth.

        The recorded shape has to be what the attention kernel saw: the collector
        turns it into `SequenceStats` and the quadratic latency term is fitted
        against `sum_squared_tokens`. BSHD pads to the longest sequence, so the
        shape is `[seq_length] * micro_batch_size`. THD keeps the per-sequence
        lengths, whose individual squares are what the pack actually costs.
        """
        from megatron.core.num_microbatches_calculator import get_num_microbatches

        observed = int(get_num_microbatches())
        if observed != self._expected_microbatches:
            raise RuntimeError(
                f"Megatron derived {observed} microbatches but the candidate "
                f"requested {self._expected_microbatches}. Check global_batch_size, "
                "micro_batch_size, and the data-parallel size."
            )
        if self._packing == "thd":
            if not self._sequence_lengths:
                raise RuntimeError(
                    "THD packing requires sequence_lengths in the launch spec so the "
                    "per-sequence squares can be recorded."
                )
            shape = list(self._sequence_lengths)
        else:
            shape = [self._seq_length] * self._micro_batch_size
        self._shapes = [shape] * observed


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
        print(f"[srl] rank {rank} failed to write its OOM marker: {marker_error!r}")


# Substring patterns mapping a CUDA kernel name onto the op family `ComposedTraining`
# charges it under. Ordered: the first match wins, so a more specific pattern must come
# first. `None` means "deliberately not one of the composition's families", which is the
# bucket that answers the question this tracing exists for.
#
# Matching on kernel names is inherently a heuristic, and the summary keeps an
# `unmatched` list naming every kernel that fell through with its total time. A residual
# hiding in `unmatched` is the finding. A residual hiding in a *mis*-matched bucket would
# be a bug in this table, which is why nothing is silently absorbed into "other".
_KERNEL_CATEGORIES: tuple[tuple[str, str | None], ...] = (
    # Attention, before GEMM: flash kernels are matmuls too but are charged separately.
    ("flash_fwd", "attention"),
    ("flash_bwd", "attention"),
    ("fmha", "attention"),
    ("attn", "attention"),
    # GEMM. cutlass and the cublas-internal names both appear.
    ("cutlass", "gemm"),
    ("gemm", "gemm"),
    ("cublas", "gemm"),
    # cuBLAS's Hopper-generation GEMM kernels are named `nvjet_tst_*` and match none of
    # the historical patterns. They were 0.86 s of `unmatched` before this line, which is
    # how the unmatched bucket earns its keep: it named the gap instead of hiding it.
    ("nvjet", "gemm"),
    ("nn_", "gemm"),
    ("s16816", "gemm"),
    ("ampere_", "gemm"),
    ("sm90", "gemm"),
    # Collectives.
    ("nccl", "collective"),
    ("AllReduce", "collective"),
    ("ReduceScatter", "collective"),
    ("AllGather", "collective"),
    ("SendRecv", "collective"),
    # Optimizer.
    ("multi_tensor_adam", "optimizer"),
    ("multi_tensor_l2norm", "optimizer"),
    ("multi_tensor_scale", "optimizer"),
    ("adam", "optimizer"),
    # Elementwise: norms, rope, activation, and the fused residual add.
    ("rmsnorm", "elementwise"),
    ("layernorm", "elementwise"),
    ("layer_norm", "elementwise"),
    ("rms_norm", "elementwise"),
    ("rotary", "elementwise"),
    ("RoPE", "elementwise"),
    ("silu", "elementwise"),
    ("swiglu", "elementwise"),
    ("bias_dropout", "elementwise"),
    ("dropout", "elementwise"),
    ("GeluFunctor", "elementwise"),
    # Loss and the logit path.
    ("cross_entropy", "loss"),
    ("softmax", "loss"),
    ("log_softmax", "loss"),
)


def _categorize_kernel(name: str) -> str | None:
    """Return the op family a CUDA kernel belongs to, or None when unrecognized."""
    lowered = name.lower()
    for pattern, family in _KERNEL_CATEGORIES:
        if pattern.lower() in lowered:
            return family
    return None


def _kernel_totals(trace: Any) -> dict[str, Any]:
    """
    Summarize one step's CUDA kernel time by op family.

    This is the measurement `ComposedTraining`'s decomposition structurally cannot make.
    The composition sums the ops it knows about, so a cost it has no term for is invisible
    from the cost side however carefully the per-term shares are read. Here the total is
    whatever the GPU actually ran, and the `unmatched` list is the part no modelled family
    claims.

    Device time is summed per kernel name rather than taken from the profiler's own
    grouping, because `key_averages()` aggregates by op rather than by kernel and a fused
    TE module appears as one op spanning several kernels.

    **Only `DeviceType.CUDA` events are counted.** `self_device_time_total` is non-zero on
    the *CPU-side* op as well as on the kernel it launched, so filtering merely on "has
    device time" double-counts every collective: `nccl:coalesced` and
    `ncclDevKernel_SendRecv` both reported 2.859 s across 2 calls, and the first version of
    this function summed both. That inflated the collective family to 68% of device time,
    which is exactly the kind of plausible-looking number this tracing exists to avoid
    producing.

    Returns:
        dict[str, Any]: `by_family` in seconds, `total_device_s`, `unmatched` as a
            descending list of `[name, seconds, count]`, and `top_kernels` for context.
    """
    from collections import defaultdict

    per_kernel: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    cuda_device = torch.autograd.DeviceType.CUDA
    for event in trace.events():
        # A CPU op carries the device time of the kernels it launched, so counting both
        # the op and its kernel doubles every collective. Only the kernel is real work.
        if getattr(event, "device_type", None) != cuda_device:
            continue
        name = getattr(event, "name", "")
        duration = getattr(event, "self_device_time_total", 0) or 0
        if duration <= 0:
            continue
        entry = per_kernel[name]
        entry[0] += float(duration) / 1e6  # microseconds -> seconds
        entry[1] += 1

    by_family: dict[str, float] = defaultdict(float)
    unmatched: list[tuple[str, float, int]] = []
    for name, (seconds, count) in per_kernel.items():
        family = _categorize_kernel(name)
        if family is None:
            unmatched.append((name, seconds, count))
            by_family["unmatched"] += seconds
        else:
            by_family[family] += seconds

    unmatched.sort(key=lambda item: -item[1])
    ranked = sorted(per_kernel.items(), key=lambda item: -item[1][0])
    return {
        "by_family_s": {name: by_family[name] for name in sorted(by_family)},
        "total_device_s": sum(by_family.values()),
        "distinct_kernels": len(per_kernel),
        "unmatched": [[name, seconds, count] for name, seconds, count in unmatched[:40]],
        "top_kernels": [
            [name, entry[0], entry[1]] for name, entry in ranked[:40]
        ],
    }
