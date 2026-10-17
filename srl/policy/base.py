"""Staleness policy plugin interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from ..component.rollout.engine import Request
from .ledger import GroupView


class CompleteAction(Enum):
    KEEP = "keep"
    DROP = "drop"


@dataclass
class SimState:
    """What a policy is allowed to see when making decisions."""

    now: float = 0.0
    train_version: int = 0
    rollout_version: int = 0
    batch_size: int = 0
    """Prompt groups per training batch. Responses is `batch_size * group_size`."""
    group_size: int = 1
    max_staleness: int = 0
    inflight: list[Request] = field(default_factory=list)
    ready: list[Request] = field(default_factory=list)
    """Responses awaiting consumption. Only members of complete groups appear."""
    ready_groups: list[list[Request]] = field(default_factory=list)
    """The same responses grouped, so a batch cannot be sliced mid-group."""
    n_admitted: int = 0
    n_consumed: int = 0
    n_instances: int = 1
    sync_in_progress: bool = False
    groups: tuple[GroupView, ...] = ()
    """Every live group. Order is ledger insertion, not recency."""

    @property
    def responses_per_batch(self) -> int:
        return self.batch_size * self.group_size


def take_complete_groups(
    ready: list[Request],
    n_groups: int,
    group_size: int,
) -> list[Request] | None:
    """
    Return the first `n_groups` fully-finished prompt groups, flattened.

    GRPO scores each response against its own group, so a group missing even one
    response is not trainable. Selecting by response count instead would favour
    short responses, whose long siblings are still generating.

    Groups are returned in the order their last response landed, so a batch is
    the oldest complete work rather than an arbitrary subset.

    Args:
        ready: Finished requests awaiting consumption.
        n_groups: Prompt groups required.
        group_size: Responses that make a group complete.

    Returns:
        `n_groups * group_size` requests, or None if too few groups are complete.
    """
    if n_groups < 1 or group_size < 1:
        raise ValueError(
            f"n_groups and group_size must be >= 1, got {n_groups} and {group_size}."
        )
    buckets: dict[int, list[Request]] = {}
    complete: list[list[Request]] = []
    for req in ready:
        bucket = buckets.setdefault(req.group_id, [])
        bucket.append(req)
        if len(bucket) == group_size:
            complete.append(bucket)
            if len(complete) == n_groups:
                return [req for group in complete for req in group]
    return None


class StalenessPolicy(ABC):
    """
    Base class for staleness control policies.
    """

    name: str = "base"

    @abstractmethod
    def admit_quota(self, st: SimState) -> int:
        """How many new responses may be admitted at this instant.

        Counted in responses, not groups, because the engine's capacity is what
        bounds it. The simulator rounds down to a whole group.
        """

    def on_admit(self, group: list[Request], st: SimState) -> None:
        """Commit policy state when a staged group first reaches an instance.

        Called exactly once per prompt group, immediately after its first
        response is placed. The whole group is admitted atomically; siblings
        may remain staged in the router until instance capacity is available.
        """

    def on_route(self, req: Request, instance_id: int, st: SimState) -> None:
        """Hook for policies tracking which instance serves a request.

        Called every time the router places `req`, including the first
        placement and re-placements after a weight update or preemption.
        """

    def on_group_complete(self, group: list[Request], st: SimState) -> CompleteAction:
        """Decide whether a finished prompt group is kept for training.

        Called once, when the group's last response finishes. Returning `DROP`
        discards the whole group, since a partial group is not trainable.
        """
        return CompleteAction.KEEP

    @abstractmethod
    def peek_batch(self, st: SimState) -> list[Request] | None:
        """Return the next training batch without consuming policy state.

        The batch is `st.batch_size` complete groups, flattened: exactly
        `st.responses_per_batch` responses.
        """

    def take_batch(self, st: SimState) -> list[Request] | None:
        """Consume and return the next training batch."""
        return self.peek_batch(st)

    def on_batch_unavailable(self, st: SimState) -> None:
        """Called when no training batch is currently available."""

    def on_train_version_advance(self, train_version: int, st: SimState) -> None:
        """Called after training finishes and `train_version` is published."""

    def will_sync_after_train(self, st: SimState) -> bool:
        """Whether the current training step ends with a rollout weight sync."""
        return True

    def on_interrupt(self, reqs: list[Request], st: SimState) -> None:
        """Called when in-flight generation is interrupted and will resume."""

    def on_discard(self, reqs: list[Request], st: SimState) -> None:
        """Called when requests are permanently removed from the simulator."""

    def replacement_groups_on_drop(
        self,
        groups: list[list[Request]],
        st: SimState,
    ) -> int:
        """Return dataset groups to stage after permanent policy drops."""
        return 0

    def expire(self, st: SimState, expected_train_version: int) -> list[Request]:
        """Drop requests that would violate staleness at `expected_train_version`.

        Called when a batch is committed, before training actually finishes.
        `st.train_version` is still the last published version, `expected_train_version`
        is the version this batch will produce.
        """
        return []

    def rollout_version_after_train(self, st: SimState) -> int:
        """The weight version the rollout engine should serve after publish."""
        return st.train_version

    def check_invariants(self, st: SimState) -> None:
        """Assert this policy's own contract."""
