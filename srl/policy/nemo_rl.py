"""NeMo-RL single-controller staleness policy: four samplers, one admission gate."""

from __future__ import annotations

from ..component.rollout.engine import Request
from .base import Cancel, CancelAction, CompleteAction, PauseMode, PolicyContext, SimState, StalenessPolicy
from .rules import FIFO, CancelBelowVersion


class NeMoRLPolicy(StalenessPolicy):
    """
    NeMo-RL's `async_rl.sampler` family, selected by `sampler`.

    All four share one admission gate (two buffer semaphores plus a
    dispatch-lag check) and differ only in how they select and, for
    `windowed`, cancel. `windowed` keys on `admit_version` with a strict
    window, cancelling both running and ready groups below the floor.
    `weight_fifo` filters to the same window but never cancels, so a group
    that falls below it sits unconsumed forever. `ready_first` has no
    lower bound at all. `in_order` stamps each admitted cohort with the
    training version it is reserved for and takes only an exact match.
    """

    name = "nemo_rl"

    def __init__(
        self,
        sampler: str = "in_order",
        max_staleness_versions: int = 1,
        max_lookahead_versions: int = 1,
        sample_freshest_first: bool = False,
        max_inflight_prompts: int = 32,
        max_buffered_rollouts: int = 64,
        pause_mode: PauseMode = "abort",
    ):
        assert sampler in ("windowed", "weight_fifo", "ready_first", "in_order"), (
            f"Unsupported sampler {sampler!r}."
        )
        self.sampler = sampler
        self.staleness_window = max_staleness_versions
        self.lookahead = max_lookahead_versions
        self.freshest_first = sample_freshest_first
        self.max_inflight_prompts = max_inflight_prompts
        self.max_buffered_rollouts = max_buffered_rollouts
        self.pause = pause_mode

        self._inflight_prompts = 0
        self._buffered = 0
        self._dispatch_index = 0
        self._admitted_in_cohort = 0
        self._target_step: dict[int, int] = {}
        self.gc_dropped = 0

    def _gate_window(self) -> int:
        return self.lookahead if self.sampler == "in_order" else self.staleness_window

    def admit_quota(self, st: SimState) -> int:
        if self._inflight_prompts >= self.max_inflight_prompts:
            return 0
        if self._buffered >= self.max_buffered_rollouts:
            return 0
        if self._dispatch_index >= st.train_version + self._gate_window():
            return 0
        room = min(
            self.max_inflight_prompts - self._inflight_prompts,
            self.max_buffered_rollouts - self._buffered,
        )
        return max(0, room) * st.group_size

    def on_admit(self, group: list[Request], st: SimState) -> None:
        self._inflight_prompts += 1
        self._buffered += 1
        if self.sampler == "in_order":
            self._target_step[group[0].group_id] = self._dispatch_index
        self._admitted_in_cohort += 1
        if self._admitted_in_cohort >= st.batch_size:
            self._dispatch_index += 1
            self._admitted_in_cohort = 0

    def on_group_complete(self, group: list[Request], st: SimState) -> CompleteAction:
        self._inflight_prompts -= 1
        return CompleteAction.KEEP

    def peek_batch(self, st: SimState) -> list[Request] | None:
        if self.sampler == "in_order":
            eligible = [
                group
                for group in st.ready_groups
                if self._target_step.get(group[0].group_id) == st.train_version
            ]
            if len(eligible) < st.batch_size:
                return None
            return [req for group in eligible[: st.batch_size] for req in group]

        if self.sampler == "ready_first":
            return FIFO().select(st.ready_groups, st.batch_size)

        admit_version = {g.group_id: g.admit_version for g in st.groups}
        floor = max(0, st.train_version - self.staleness_window)
        eligible = [
            group for group in st.ready_groups if admit_version.get(group[0].group_id, 0) >= floor
        ]
        if len(eligible) < st.batch_size:
            return None
        reverse = self.sampler == "windowed" and self.freshest_first
        ordered = sorted(
            eligible, key=lambda group: admit_version.get(group[0].group_id, 0), reverse=reverse
        )
        return [req for group in ordered[: st.batch_size] for req in group]

    def take_batch(self, st: SimState) -> list[Request] | None:
        batch = self.peek_batch(st)
        if batch is None:
            return None
        taken_group_ids = {req.group_id for req in batch}
        self._buffered -= len(taken_group_ids)
        for group_id in taken_group_ids:
            self._target_step.pop(group_id, None)
        return batch

    def on_publish(self, new_version: int, st: SimState) -> list[Cancel]:
        if self.sampler != "windowed":
            return []
        cancels = CancelBelowVersion(horizon=self.staleness_window, action=CancelAction.DROP).on_publish(
            PolicyContext(state=st), new_version
        )
        status_by_id = {g.group_id: g.status for g in st.groups}
        for cancel in cancels:
            if status_by_id.get(cancel.group_id) != "done":
                self._inflight_prompts -= 1
            self._buffered -= 1
            self._target_step.pop(cancel.group_id, None)
        self.gc_dropped += len(cancels)
        return cancels

    def pause_mode(self, st: SimState) -> PauseMode:
        return self.pause
