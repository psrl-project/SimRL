"""
The mechanism every measurement loop shares, without dictating the loop itself.

Six `profile_*` functions each re-implemented the same four things: unpacking
`config.measurement` into a timing call, building a row dict by hand, freeing tensors between
points, and reaching the right sweep. The duplication was not merely verbose. Hand-built rows
meant a misspelled column reached `PerfTableWriter.write`, which validates against the op's
declared columns but only when the whole sweep has finished: on a multi-hour grid the error
arrives hours after the typo. `row()` moves that check to the point of construction.

**`run()` is abstract rather than a template.** The three families do not share a loop shape:

- `AttentionRunner` must select a backend and allocate one KV cache *before* the loop, and both
  phases are timed against that one cache.
- `OptimizerRunner` must issue an untimed `step()` and verify state allocation before timing,
  because `FusedAdam` allocates lazily and an OOM there yields a fast row rather than an error.
- `ElementwiseRunner` runs its whole loop inside `set_current_vllm_config`, without which vLLM
  dispatches a native fallback instead of the fused kernel.

A base class owning the iteration would need a hook for each of those, which is a template
with three escape hatches: more coupling than the duplication it removes.

**There is no tensor-lifetime helper, deliberately.** The obvious one (a context manager taking
the point's tensors and freeing them on exit) cannot work: a callee cannot delete a caller's
names, so `del tensors` inside the manager drops only its own tuple reference and
`torch.cuda.empty_cache()` would then run with the originals still live, reclaiming nothing. The
sweeps are ordered largest-footprint first precisely so that a grid which will not fit fails in
its first seconds, and that ordering depends on each point's memory actually being released, so a
helper that silently did not free would defeat it. Each runner therefore keeps its explicit
`del` followed by `torch.cuda.empty_cache()` as two separate lines, which is correct.

`collective.py` is not a `KernelRunner`. It launches torchrun (and ssh, across nodes) and
aggregates JSON the child ranks wrote, so `time()` has nothing to time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...schema import ModelDescriptor
from ..config import KernelProfileConfig
from ..measurements import PerfTableWriter

WriterFactory = Callable[[str, str, str], PerfTableWriter]
"""Builds a `PerfTableWriter` from `(framework, version, kernel_source)`.

A factory rather than a writer, because the kernel source is only known after the measurement
starts for the families that select a backend at runtime.
"""


@dataclass(frozen=True)
class KernelRunner:
    """One op family's measurement sweep: shared mechanism, family-specific loop.

    Frozen because a runner is constructed per sweep and read from a closure in every timed
    lambda. A mutable `config` would let a point be timed under settings a later point changed.
    """

    model: ModelDescriptor
    config: KernelProfileConfig

    def time(self, call: Callable[[], Any], *, op_name: str | None = None) -> float:
        """
        Time one call the way the engine that will be simulated runs it.

        Args:
            call (Callable[[], Any]): Zero-argument callable that launches GPU work. Bind
                tensors as default arguments rather than capturing them, so the callable holds
                its own reference and the caller's `del` can free the originals.
            op_name (str | None): The op this point belongs to. When given, its
                `graph_replayed` declaration selects the timing mode. When omitted, CUDA
                events are used. Passing the name is how a family whose ops disagree (the
                two attention phases, the two GDN phases) gets each one timed correctly.

        Returns:
            float: Mean per-call latency in seconds.
        """
        # Imported here rather than at module scope: `runner/` is the GPU side of the tree, but
        # a test that only constructs a runner to check its row building should not need torch.
        from ..utils.timing import time_cuda_call, time_cuda_graph_replay

        replay = False
        if op_name is not None:
            from ....cost.kernel.ops.base import op_for

            replay = op_for(op_name).graph_replayed
        measurement = self.config.measurement
        timer = time_cuda_graph_replay if replay else time_cuda_call
        return timer(
            call,
            warmup_iterations=measurement.warmup_iterations,
            active_iterations=measurement.active_iterations,
        )

    def row(self, op_name: str, *, latency_s: float, **fields: Any) -> dict[str, Any]:
        """
        Build one perf-table row, checked against the op's declared columns.

        Args:
            op_name (str): A registered op name. Supplies the expected column set.
            latency_s (float): The measured latency, always the last column.
            **fields (Any): This row's slice and axis values.

        Returns:
            dict[str, Any]: The row, keyed exactly as the op declares.

        Raises:
            ValueError: If `fields` plus `latency_s` do not match the op's columns exactly.
                Raised here rather than at write time so a misspelled key fails on the first
                point instead of after the whole grid has been measured.
            KernelCoverageError: If `op_name` is not registered.
        """
        from ....cost.kernel.ops.base import op_for

        columns = op_for(op_name).table_columns()
        row = {**fields, "latency_s": latency_s}
        missing = sorted(set(columns) - set(row))
        extra = sorted(set(row) - set(columns))
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing {', '.join(missing)}")
            if extra:
                parts.append(f"unexpected {', '.join(extra)}")
            raise ValueError(
                f"A {op_name} row has {' and '.join(parts)}. Expected exactly: "
                f"{', '.join(columns)}."
            )
        return row

    def run(self, *, writer_factory: WriterFactory) -> tuple[Path, ...]:
        """
        Measure this op's whole family and write its perf tables.

        Args:
            writer_factory (WriterFactory): Builds a writer from
                `(framework, version, kernel_source)`.

        Returns:
            tuple[Path, ...]: The written parquet paths.

        Raises:
            NotImplementedError: Always. Each family implements its own loop. See the module
                docstring for why this is not a template method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement run(). See runner/base.py on why the loop "
            f"is not inherited."
        )
