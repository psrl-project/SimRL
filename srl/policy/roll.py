"""ROLL admission + GC discard staleness policy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..component.rollout.engine import Request
from .base import Cancel, CancelAction, CompleteAction, SimState, StalenessPolicy


@dataclass
class _Cohort:
    """One FIFO of finished prompt groups per admission step."""

    running: set[int] = field(default_factory=set)
    finished: deque = field(default_factory=deque)


class RollPolicy(StalenessPolicy):
    """
    ROLL's RLVR async admission with GC discard enforcement.

    Same token bucket as AReaL for admission, but enforces the bound via:
    1. Consumption floor (min_step) prevents old batches.
    2. GC at weight publication purges groups that aged past the window, as ROLL's `gc()`
       does before the model update. Each purged group is replaced by a fresh prompt.
    """

    name = "roll"

    def __init__(self, max_inflight: int | None = None):
        self.max_inflight = max_inflight
        self.budget = 0
        self.running_prompts = 0
        self.completed_prompts = 0
        self._configured = False
        self.cohorts: dict[int, _Cohort] = {}
        self.gc_dropped = 0

    def _ensure_cohort(self, step: int) -> _Cohort:
        return self.cohorts.setdefault(step, _Cohort())

    def _configure(self, st: SimState) -> None:
        ratio = st.max_staleness
        if ratio > 0:
            self.budget = ratio * st.batch_size
        self.budget += st.batch_size
        self._ensure_cohort(st.train_version)
        self._configured = True

    def admit_quota(self, st: SimState) -> int:
        if not self._configured:
            self._configure(st)
        self._ensure_cohort(st.train_version)
        room = self.budget - (self.running_prompts + self.completed_prompts)
        if self.max_inflight is not None:
            room = min(room, self.max_inflight - len(st.inflight) // st.group_size)
        return max(0, room) * st.group_size

    def on_admit(self, group: list[Request], st: SimState) -> None:
        self.running_prompts += 1
        self._ensure_cohort(group[0].v_traj).running.add(group[0].group_id)

    def on_group_complete(self, group: list[Request], st: SimState) -> CompleteAction:
        group_id = group[0].group_id
        cohort = self.cohorts.get(group[0].v_traj)
        if cohort is None or group_id not in cohort.running:
            return CompleteAction.DROP
        cohort.running.discard(group_id)
        cohort.finished.append(group)
        self.running_prompts -= 1
        self.completed_prompts += 1
        return CompleteAction.KEEP

    def _eligible_steps(self, st: SimState) -> list[int]:
        min_step = st.train_version - st.max_staleness
        steps = [s for s in sorted(self.cohorts) if min_step <= s < st.train_version]
        if st.train_version in self.cohorts:
            steps.append(st.train_version)
        return steps

    def peek_batch(self, st: SimState) -> list[Request] | None:
        steps = self._eligible_steps(st)
        available = sum(len(self.cohorts[s].finished) for s in steps)
        if available < st.batch_size:
            return None

        collected: list[list[Request]] = []
        for step in steps:
            cohort = self.cohorts[step]
            needed = st.batch_size - len(collected)
            collected.extend(list(cohort.finished)[:needed])
            if len(collected) >= st.batch_size:
                break
        return [req for group in collected[: st.batch_size] for req in group]

    def take_batch(self, st: SimState) -> list[Request] | None:
        batch = self.peek_batch(st)
        if batch is None:
            return None

        remaining = st.batch_size
        for step in self._eligible_steps(st):
            cohort = self.cohorts[step]
            take = min(remaining, len(cohort.finished))
            for _ in range(take):
                cohort.finished.popleft()
            remaining -= take
            if remaining == 0:
                break
        return batch

    def on_train_version_advance(self, train_version: int, st: SimState) -> None:
        self.budget += st.batch_size
        self._ensure_cohort(train_version)

    def replacement_groups_on_drop(
        self,
        groups: list[list[Request]],
        st: SimState,
    ) -> int:
        """ROLL replaces groups permanently removed by its expiry rules."""
        return len(groups)

    def on_publish(self, new_version: int, st: SimState) -> list[Cancel]:
        max_gc_step = (new_version - 1) - st.max_staleness
        if max_gc_step < 0:
            return []
        cancels: list[Cancel] = []
        for step in [s for s in self.cohorts if s <= max_gc_step]:
            cohort = self.cohorts.pop(step)
            # Finished groups that aged out of the consume window are dropped, not consumed.
            while cohort.finished:
                group = cohort.finished.popleft()
                cancels.append(Cancel(group[0].group_id, CancelAction.REPLACE))
                self.completed_prompts -= 1
                self.gc_dropped += 1
            for group_id in cohort.running:
                cancels.append(Cancel(group_id, CancelAction.REPLACE))
                self.running_prompts -= 1
                self.gc_dropped += 1
        return cancels

    def check_invariants(self, st: SimState) -> None:
        floor = st.train_version - st.max_staleness
        for step, cohort in self.cohorts.items():
            if step < floor:
                assert not cohort.running and not cohort.finished, (
                    f"Cohort {step} below floor {floor} (version={st.train_version}) "
                    f"still has {len(cohort.running)} running / "
                    f"{len(cohort.finished)} finished."
                )
