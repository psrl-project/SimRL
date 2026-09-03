"""
Tensor-parallel all-reduce, the one op whose measurement is not a single kernel call.

vLLM issues an all-reduce over per-token hidden activations after each attention block and each
MLP block, so a `tp>1` step pays two per layer. Without this table a composition at `tp>1` would
silently omit every one of them, which is why `ComposedRollout` rejected `tp>1` outright before
this op existed.

`world_size` is a slice column rather than an interpolation axis. A 2-rank all-reduce and an
8-rank one run different ring schedules over different link topologies, so blending them would
produce a number describing neither.

The SOL is the standard ring-all-reduce lower bound: each rank sends and receives
`(N-1)/N * bytes` in the reduce-scatter phase and the same again in the all-gather phase, for
`2*(N-1)/N * bytes` of traffic per rank. That factor is 1.0 at two ranks and approaches 2 as the
world grows, never reaching it. NCCL may choose a tree or a different algorithm at small sizes,
which the fitted utility absorbs in the same way it absorbs every other SOL simplification.

A fitted floor is essential: NCCL pays a per-call launch and handshake cost that dominates
entirely at small message sizes, which is the regime a decode step lives in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..hardware import GpuSpec
from ..sol import SolComponents
from .base import register_op


@dataclass(frozen=True)
class AllReduceKernelOp:
    """Ring all-reduce over per-token hidden activations.

    Frozen because the instance below is a registry singleton shared by every caller, so a
    mutation anywhere would silently change which columns every table is read against.
    """

    name = "all_reduce"
    family = "collective"
    slice_columns = ("dtype", "world_size")
    axis_columns = ("message_bytes",)
    fits_floor = True

    def sol(
        self,
        spec: GpuSpec,
        *,
        slice_key: tuple[Any, ...],
        coords: tuple[float, ...],
        floor_s: float = 0.0,
    ) -> SolComponents:
        """
        Return the roofline of one all-reduce.

        Args:
            spec (GpuSpec): Hardware; `link_bandwidth_bytes_per_s` carries the interconnect.
            slice_key (tuple[Any, ...]): `(dtype, world_size)`.
            coords (tuple[float, ...]): `(message_bytes,)`, the per-rank tensor size in bytes.
            floor_s (float): Fitted NCCL launch floor in seconds, applied inside the `max`.

        Returns:
            SolComponents: A zero compute term, the ring transfer term, and the floor. At
                `world_size` 1 the transfer term is zero, because no collective runs.

        Raises:
            ValueError: If `world_size` or `message_bytes` is not positive.
        """
        world_size = int(slice_key[1])
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size!r}.")
        message_bytes = float(coords[0])
        if message_bytes <= 0.0:
            raise ValueError(f"message_bytes must be positive, got {message_bytes!r}.")
        # Ring all-reduce: reduce-scatter then all-gather, each moving (N-1)/N per rank.
        factor = 2.0 * (world_size - 1) / world_size
        memory_s = factor * message_bytes / spec.link_bandwidth_bytes_per_s
        return SolComponents(math_s=0.0, memory_s=memory_s, floor_s=floor_s)

    def parse_sweep(self, value: dict[str, Any]) -> Any:
        """Parse the `collective:` sweep subtree into a `CollectiveSweep`."""
        from ....profiler.kernels.config import CollectiveSweep

        return CollectiveSweep.from_dict(value)

    def measure(self, model: Any, config: Any, *, writer_factory: Any) -> tuple[Any, ...]:
        """Spawn `tensor_parallel` ranks and time an all-reduce at each message size."""
        from ....profiler.kernels.collective import profile_collective

        return profile_collective(model, config, writer_factory=writer_factory)

    def table_columns(self) -> tuple[str, ...]:
        """Return this op's non-provenance table columns, latency last."""
        return self.slice_columns + self.axis_columns + ("latency_s",)


ALL_REDUCE_OP = AllReduceKernelOp()
register_op(ALL_REDUCE_OP)
