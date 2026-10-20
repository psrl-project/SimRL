"""
The collectives a sharded step issues, each a table keyed by topology as well as size.

Five ops rather than one. The original `all_reduce` was enough for rollout, where vLLM
issues an all-reduce after each attention and MLP block, but it cannot describe training:
`megatron/core/tensor_parallel/layers.py` calls
`reduce_scatter_to_sequence_parallel_region` and `gather_from_sequence_parallel_region`
whenever sequence parallelism is on, and the Megatron profiler turns that on for every
`tp > 1` run. A reduce-scatter over `tokens*hidden/tp` bytes priced from the all-reduce
table would be wrong in op identity, message size, and count at once.

`sol()` lives on `_CollectiveOp` and is written once. The five ops differ only in how much
traffic one rank moves, which each declares as `traffic_factor`. Five copies of the same
roofline binding is precisely the drift `ops/base.py` exists to prevent: its module
docstring records that two independent implementations of this binding could silently
change utility on one side of the holdout round trip.

`world_size` and `nodes` are both **slice** columns, matched exactly rather than
interpolated. A 2-rank all-reduce and an 8-rank one run different ring schedules, and 8
ranks on one node and 8 ranks across two nodes traverse different links in different
counts. Blending either pair would produce a number describing neither, so an unmeasured
topology raises instead.

A fitted floor is essential and not a refinement: a floor-less all-reduce SOL errs by 220x
at 4096 bytes, because every collective pays a launch and handshake cost that dominates
entirely at the small messages both a decode step and an SP-sharded layer live in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..hardware import GpuSpec
from ..sol import SolComponents
from .base import register_op


@dataclass(frozen=True)
class _CollectiveOp:
    """One collective, priced by how much traffic a single rank moves.

    Frozen because each instance below is a registry singleton shared by every caller, so a
    mutation anywhere would silently change which columns every table is read against.

    Subclasses declare `name` and `traffic_factor` and inherit everything else. They do not
    override `sol`: see the module docstring for why that matters.
    """

    family = "collective"
    slice_columns = ("dtype", "world_size", "nodes")
    axis_columns = ("message_bytes",)
    fits_floor = True
    # Not capturable in the first place: `collective.py` is not a `KernelRunner` but a
    # torchrun (and ssh) launcher whose child ranks write JSON, so there is no single
    # callable to capture. The launch cost is also real for the training model.
    graph_replayed = False

    def traffic_factor(self, world_size: int) -> float:
        """
        Return per-rank bytes moved as a multiple of the message size.

        Args:
            world_size (int): Ranks participating in the collective.

        Returns:
            float: The multiplier on `message_bytes`.
        """
        raise NotImplementedError

    def sol(
        self,
        spec: GpuSpec,
        *,
        slice_key: tuple[Any, ...],
        coords: tuple[float, ...],
        floor_s: float = 0.0,
    ) -> SolComponents:
        """
        Return the roofline of one collective, accounting for the node boundary.

        A ring over `world_size` ranks spread across `nodes` nodes crosses a node boundary
        on exactly `nodes` of its `world_size` hops, because the ring enters and leaves
        each node once. So `nodes / world_size` of the traffic traverses a NIC and the rest
        stays on the fast intra-node link, and the slower of the two levels bounds the
        result. `max` rather than a sum: the two levels of a hierarchical collective overlap
        in the best case, matching the optimism already built into
        `max(math_s, memory_s, floor_s)`.

        Args:
            spec (GpuSpec): Hardware. Supplies both the intra-node link bandwidth and the
                per-rank inter-node egress.
            slice_key (tuple[Any, ...]): `(dtype, world_size, nodes)`.
            coords (tuple[float, ...]): `(message_bytes,)`, the per-rank tensor size.
            floor_s (float): Fitted launch floor in seconds, applied inside the `max`.

        Returns:
            SolComponents: A zero compute term, the transfer term, and the floor. The
                transfer term is zero at `world_size` 1, where no collective runs.

        Raises:
            ValueError: If `world_size`, `nodes`, or `message_bytes` is not positive, or if
                `nodes` exceeds `world_size`, which describes no placement.
        """
        world_size = int(slice_key[1])
        nodes = int(slice_key[2])
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size!r}.")
        if nodes <= 0:
            raise ValueError(f"nodes must be positive, got {nodes!r}.")
        if nodes > world_size:
            raise ValueError(
                f"nodes {nodes!r} exceeds world_size {world_size!r}, which describes no rank "
                f"placement. Every node in the group must hold at least one rank."
            )
        message_bytes = float(coords[0])
        if message_bytes <= 0.0:
            raise ValueError(f"message_bytes must be positive, got {message_bytes!r}.")

        traffic_bytes = self.traffic_factor(world_size) * message_bytes
        # Guarded rather than derived: at one node the ring crosses no boundary, but
        # `nodes / world_size` would read 1/8 at eight ranks and charge an intra-node sweep
        # for NIC traffic it never paid. That would perturb the very tp=1 and tp=2 numbers
        # the accuracy gate pins.
        inter_fraction = 0.0 if nodes == 1 else nodes / world_size
        intra_s = traffic_bytes * (1.0 - inter_fraction) / spec.link_bandwidth_bytes_per_s
        inter_s = traffic_bytes * inter_fraction / spec.inter_node_bandwidth_bytes_per_s
        return SolComponents(math_s=0.0, memory_s=max(intra_s, inter_s), floor_s=floor_s)

    def parse_sweep(self, value: dict[str, Any]) -> Any:
        """Parse the `collective:` sweep subtree into a `CollectiveSweep`."""
        from ....profiler.kernel.sweep import CollectiveSweep

        return CollectiveSweep.from_dict(value)

    def measure(self, model: Any, config: Any, *, writer_factory: Any) -> tuple[Any, ...]:
        """
        Time every registered collective at each swept topology.

        Implemented once on the base and inherited by all five ops, because `measure` is
        called once per *family* and because process-group setup dominates a small-message
        measurement: a cross-node rendezvous costs seconds, so amortizing it across all
        five ops is the difference between a feasible sweep and an absurd one. It also
        guarantees every row describes the same communicator and the same topology.
        """
        from ....profiler.kernel.collective import profile_collective

        return profile_collective(model, config, writer_factory=writer_factory)

    def table_columns(self) -> tuple[str, ...]:
        """Return this op's non-provenance table columns, latency last."""
        return self.slice_columns + self.axis_columns + ("latency_s",)


@dataclass(frozen=True)
class AllReduceKernelOp(_CollectiveOp):
    """Ring all-reduce: reduce-scatter then all-gather, so each phase moves `(N-1)/N`."""

    name = "all_reduce"

    def traffic_factor(self, world_size: int) -> float:
        """Return `2(N-1)/N`, which is 1.0 at two ranks and approaches 2 as N grows."""
        return 2.0 * (world_size - 1) / world_size


@dataclass(frozen=True)
class ReduceScatterKernelOp(_CollectiveOp):
    """Half of an all-reduce. Megatron's row-parallel output under sequence parallelism."""

    name = "reduce_scatter"

    def traffic_factor(self, world_size: int) -> float:
        """Return `(N-1)/N`: each rank sends all but its own shard."""
        return (world_size - 1) / world_size


@dataclass(frozen=True)
class AllGatherKernelOp(_CollectiveOp):
    """The other half. Megatron's column-parallel input under sequence parallelism."""

    name = "all_gather"

    def traffic_factor(self, world_size: int) -> float:
        """Return `(N-1)/N`: each rank receives every shard but its own."""
        return (world_size - 1) / world_size


@dataclass(frozen=True)
class BroadcastKernelOp(_CollectiveOp):
    """One rank's payload delivered to all. Used for weight synchronization."""

    name = "broadcast"

    def traffic_factor(self, world_size: int) -> float:
        """
        Return `1.0`, independent of `N`.

        Deliberately **not** `(N-1)/N`. A pipelined broadcast hands the payload down a
        chain, so each rank receives the message once and the bound is one message time
        regardless of how many ranks follow. Giving it a ring reduction's factor would
        understate it by up to 2x at large `N`.
        """
        return 1.0


@dataclass(frozen=True)
class P2PKernelOp(_CollectiveOp):
    """A pipeline stage boundary send/recv, the one op here that is not a collective."""

    name = "p2p"

    def traffic_factor(self, world_size: int) -> float:
        """
        Return `1.0`: a single directed transfer, independent of `N`.

        `world_size` stays a slice column even though the factor ignores it, because the
        *link* a stage boundary traverses is a function of how the mesh places the pair,
        and a pair inside a node and a pair across one are not the same measurement.
        """
        return 1.0


ALL_REDUCE_OP = AllReduceKernelOp()
REDUCE_SCATTER_OP = ReduceScatterKernelOp()
ALL_GATHER_OP = AllGatherKernelOp()
BROADCAST_OP = BroadcastKernelOp()
P2P_OP = P2PKernelOp()

register_op(ALL_REDUCE_OP)
register_op(REDUCE_SCATTER_OP)
register_op(ALL_GATHER_OP)
register_op(BROADCAST_OP)
register_op(P2P_OP)
