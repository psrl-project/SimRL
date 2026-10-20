"""The request data model shared across the simulator.

A `Request` is the unit of work every component handles: the workload generator
creates it, the router places it, `NativeInstance` generates its tokens, the
staleness policy decides when it is consumed, and the trainer batches it. Its
`Segment` list records which weight version produced each stretch of tokens,
which is what the staleness metrics are computed from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ReqStatus(Enum):
    QUEUED = "queued"
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    RECOMPUTING = "recomputing"
    DONE = "done"
    CONSUMED = "consumed"
    DROPPED = "dropped"


@dataclass
class Segment:
    """
    One contiguous stretch of generation produced under a single weight version.
    """

    version: int
    tokens: int = 0


@dataclass(eq=False)
class Request:
    """
    A single inference request flowing through the RL pipeline.

    `eq=False` keeps the inherited identity-based `__eq__`/`__hash__`. A request
    is a mutable entity, not a value: the queues hold the same objects the
    simulator mutates, and `req in queue` / `queue.remove(req)` mean "this exact
    request". Generated equality would deep-compare every field including
    `segments` on each membership test, which dominated the engine loop.
    """

    rid: int
    prompt_len: int
    target_len: int
    v_traj: int
    group_id: int = 0
    """Monotonic key for the prompt group this response belongs to.

    Not the workload's own group id: a workload shorter than the run repeats, so
    its ids recur and two distinct groups would be treated as one. See
    `source_group_id` for the provenance.
    """
    group_index: int = 0
    """Position within the group, `0..group_size-1`."""
    source_group_id: int = 0
    """The workload group the lengths came from. Provenance only."""
    instance_id: int | None = None
    """The instance most recently serving this request, None before first placement.

    Survives a reclaim, so a re-placement can be told apart as same-instance or
    cross-instance. SMG keeps the same thing as its `x-base-worker-id` hint.
    """
    admit_time: float = 0.0
    dispatch_time: float = 0.0
    segments: list[Segment] = field(default_factory=list)
    generated: int = 0
    prefilled: int = 0
    status: ReqStatus = ReqStatus.QUEUED
    complete_time: float | None = None
    consume_version: int | None = None
    num_aborts: int = 0
    num_preemptions: int = 0
    num_reroutes: int = 0
    """Times the router placed this request after its first placement.

    A weight update or a preemption overflow returns a request to the router,
    which may then send it somewhere else. In the matched run 58.5% of
    loopbacks landed on a different instance, so this is not a rare path.
    """
    router_enqueue_time: float = 0.0
    """When this request most recently entered the router queue."""
    reprefill_tokens: int = 0
    prefix_hit_tokens: int = 0
    """Prompt tokens this request skipped because the KV cache already held them.

    Accumulates across resumptions, so a request interrupted by a weight update
    and re-prefilled records what the cache saved each time.
    """
    deadline_buffer: int = 0
    """The training buffer index this request's group must reach before its deadline.

    Stamped at admission by `inventory_deadline`'s `_commit_admission`. Zero for
    requests admitted by other strategies. Used by the `latest_deadline` eviction
    strategy to evict the least-urgent resident first.
    """

    @property
    def remaining(self) -> int:
        return self.target_len - self.generated

    @property
    def ctx(self) -> int:
        return self.prompt_len + self.generated

    @property
    def v_min(self) -> int:
        return min((s.version for s in self.segments), default=self.v_traj)

    @property
    def v_max(self) -> int:
        return max((s.version for s in self.segments), default=self.v_traj)

    def add_tokens(self, version: int, n: int) -> None:
        if n <= 0:
            return
        if not self.segments or self.segments[-1].version != version:
            self.segments.append(Segment(version=version))
        self.segments[-1].tokens += n
        self.generated += n

    def staleness_true(self) -> int:
        assert self.consume_version is not None, "Request was never consumed."
        return self.consume_version - self.v_min

    def staleness_reported(self) -> int:
        assert self.consume_version is not None, "Request was never consumed."
        return self.consume_version - self.v_max
