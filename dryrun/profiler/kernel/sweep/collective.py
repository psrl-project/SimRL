"""Message sizes and topologies to measure every collective at."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...schema import ProfileValidationError
from ..utils.validators import positive_int_tuple


@dataclass(frozen=True)
class CollectiveSweep:
    """Message sizes and topologies to measure every collective at.

    World size is declared here rather than read from `parallelism.tensor_parallel`, which
    is how the original all-reduce sweep worked. That coupling does not extend: a
    data-parallel gradient reduction at 24 ranks has nothing to do with a tensor-parallel
    degree, and writing `tensor_parallel: 24` to measure one would misdescribe what was
    measured. It also forced one config file per degree, which
    `examples/profile/kernel/profile_qwen3_8b.sh` works around by re-running with
    `FAMILY=collective`.
    """

    message_bytes: tuple[int, ...]
    """Per-rank message sizes in bytes, swept directly.

    Not derived from token counts, because the five collectives do not all carry the same
    size at the same token count: an SP-sharded reduce-scatter moves `tokens*hidden/tp`
    where an all-reduce moves `tokens*hidden`, and a gradient bucket reduction is unrelated
    to tokens entirely (Megatron buckets at `max(40M, 1M*dp)` *parameters*, so an 8.19B
    model reduces ~205 buckets of ~153 MiB). Sweeping bytes covers all three without
    teaching this config each caller's formula.
    """

    topologies: tuple[tuple[int, int], ...]
    """`(world_size, nodes)` pairs to measure.

    An explicit pair list rather than a cross product of two axes: `world_size=2` with
    `nodes=3` describes no placement, and a cross product would spawn runs that cannot form
    a mesh.
    """

    hostnames: tuple[str, ...] = ()
    """Hosts available for multi-node rendezvous, rank 0 first.

    Empty is valid and means intra-node only. A topology asking for more nodes than there
    are hostnames is rejected at parse time rather than silently measuring fewer, because a
    cross-node row that was secretly intra-node is worse than no row.
    """

    dtypes: tuple[str, ...] = ("bfloat16",)
    """Dtypes to sweep. `float32` is required to price Megatron's fp32 gradient reduction."""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CollectiveSweep:
        """
        Parse the `collective:` subtree of a kernel profile config.

        Args:
            value (dict[str, Any]): The `collective:` block.

        Returns:
            CollectiveSweep: The parsed grid, topologies validated against the host list.

        Raises:
            ProfileValidationError: If `message_bytes` is malformed, if a topology is not a
                positive `[world_size, nodes]` pair that places ranks evenly, if the host
                list is shorter than the widest topology needs, or if `dtypes` is empty.
        """
        message_bytes = positive_int_tuple(
            value["message_bytes"], field="collective.message_bytes"
        )
        raw_topologies = value.get("topologies")
        if not isinstance(raw_topologies, (list, tuple)) or not raw_topologies:
            raise ProfileValidationError(
                "collective.topologies must be a non-empty list of [world_size, nodes] pairs."
            )
        topologies: list[tuple[int, int]] = []
        for item in raw_topologies:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ProfileValidationError(
                    f"collective.topologies entries must be [world_size, nodes] pairs, "
                    f"got {item!r}."
                )
            world_size, nodes = int(item[0]), int(item[1])
            if world_size <= 0 or nodes <= 0:
                raise ProfileValidationError(
                    f"collective.topologies entries must be positive, got {item!r}."
                )
            if nodes > world_size:
                raise ProfileValidationError(
                    f"collective.topologies entry {item!r} places {nodes} nodes in a "
                    f"{world_size}-rank group, which describes no rank placement."
                )
            if world_size % nodes:
                raise ProfileValidationError(
                    f"collective.topologies entry {item!r} does not divide ranks evenly "
                    f"across nodes. An uneven placement measures a topology no launcher builds."
                )
            topologies.append((world_size, nodes))
        hostnames = tuple(str(item) for item in value.get("hostnames", ()))
        required_nodes = max(nodes for _, nodes in topologies)
        if required_nodes > 1 and len(hostnames) < required_nodes:
            raise ProfileValidationError(
                f"collective.topologies needs {required_nodes} nodes but collective.hostnames "
                f"lists {len(hostnames)}. Add the missing hosts rather than letting a "
                f"cross-node sweep run on fewer nodes than it claims."
            )
        dtypes = tuple(str(item) for item in value.get("dtypes", ("bfloat16",)))
        if not dtypes:
            raise ProfileValidationError("collective.dtypes must be non-empty when present.")
        return cls(
            message_bytes=message_bytes,
            topologies=tuple(topologies),
            hostnames=hostnames,
            dtypes=dtypes,
        )
