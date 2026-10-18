"""OpenRLHF batch-level permit pool staleness policy."""

from __future__ import annotations

from ..component.rollout.engine import Request
from .base import CompleteAction, PauseMode, SimState, StalenessPolicy
from .rules import FIFO


class OpenRLHFPolicy(StalenessPolicy):
    """
    OpenRLHF's generation permit pool.

    Admits exactly one `batch_size`-group lump per free permit, never a
    partial one. `partial_rollout_enable` decides when the permit releases:
    at take (allowing the next lump to generate while this one trains) or
    only after the publish that follows (holding admission closed for the
    whole training step, since nothing can ever be in flight at a sync).
    """

    name = "openrlhf"

    def __init__(
        self,
        async_queue_size: int = 1,
        partial_rollout_enable: bool = False,
        max_inflight: int | None = None,
    ):
        assert async_queue_size > 0, "async_queue_size must be > 0."
        self.queue_size = async_queue_size
        self.partial_rollout_enable = partial_rollout_enable
        self.max_inflight = max_inflight

        self._lumps_in_flight = 0
        self._admitted_in_lump = 0
        self._pending_release = False

    def admit_quota(self, st: SimState) -> int:
        if self._lumps_in_flight >= self.queue_size:
            return 0
        room = st.batch_size - self._admitted_in_lump
        if self.max_inflight is not None:
            room = min(room, self.max_inflight - len(st.inflight) // st.group_size)
        return max(0, room) * st.group_size

    def on_admit(self, group: list[Request], st: SimState) -> None:
        self._admitted_in_lump += 1
        if self._admitted_in_lump >= st.batch_size:
            self._lumps_in_flight += 1
            self._admitted_in_lump = 0

    def on_group_complete(self, group: list[Request], st: SimState) -> CompleteAction:
        return CompleteAction.KEEP

    def peek_batch(self, st: SimState) -> list[Request] | None:
        return FIFO().select(st.ready_groups, st.batch_size)

    def take_batch(self, st: SimState) -> list[Request] | None:
        batch = self.peek_batch(st)
        if batch is None:
            return None
        if self.partial_rollout_enable:
            self._lumps_in_flight -= 1
        else:
            self._pending_release = True
        return batch

    def on_train_version_advance(self, train_version: int, st: SimState) -> None:
        if self._pending_release:
            self._lumps_in_flight -= 1
            self._pending_release = False

    def pause_mode(self, st: SimState) -> PauseMode:
        return "in_place" if self.partial_rollout_enable else "abort"
