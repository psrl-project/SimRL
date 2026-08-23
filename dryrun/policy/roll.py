"""ROLL admission + GC discard staleness policy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..component.rollout.engine import Request
from .base import CompleteAction, SimState, StalenessPolicy


@dataclass
class _Group:
    """One FIFO of finished prompts per admission step."""

    running: set[int] = field(default_factory=set)
    finished: deque = field(default_factory=deque)


class RollPolicy(StalenessPolicy):
    """
    ROLL's RLVR async admission with GC discard enforcement.

    Same token bucket as AReaL for admission, but enforces the bound via:
    1. Consumption floor (min_step) prevents old batches.
    2. Per-step GC purges stragglers whose group aged past the window.
    """

    name = "roll"

    def __init__(self, max_inflight: int | None = None):
        self.max_inflight = max_inflight
        self.budget = 0
        self.running_prompts = 0
        self.completed_prompts = 0
        self._configured = False
        self.groups: dict[int, _Group] = {}
        self.gc_dropped = 0

    def _ensure_group(self, step: int) -> _Group:
        return self.groups.setdefault(step, _Group())

    def _configure(self, st: SimState) -> None:
        ratio = st.max_staleness
        if ratio > 0:
            self.budget = ratio * st.batch_size
        self.budget += st.batch_size
        self._ensure_group(st.train_version)
        self._configured = True

    def admit_quota(self, st: SimState) -> int:
        if not self._configured:
            self._configure(st)
        self._ensure_group(st.train_version)
        room = self.budget - (self.running_prompts + self.completed_prompts)
        if self.max_inflight is not None:
            room = min(room, self.max_inflight - len(st.inflight))
        return max(0, room)

    def on_admit(self, req: Request, st: SimState) -> None:
        self.running_prompts += 1
        self._ensure_group(req.v_traj).running.add(req.rid)

    def on_complete(self, req: Request, st: SimState) -> CompleteAction:
        group = self.groups.get(req.v_traj)
        if group is None or req.rid not in group.running:
            return CompleteAction.DROP
        group.running.discard(req.rid)
        group.finished.append(req)
        self.running_prompts -= 1
        self.completed_prompts += 1
        return CompleteAction.KEEP

    def _eligible_steps(self, st: SimState) -> list[int]:
        min_step = st.train_version - st.max_staleness
        steps = [s for s in sorted(self.groups) if min_step <= s < st.train_version]
        if st.train_version in self.groups:
            steps.append(st.train_version)
        return steps

    def peek_batch(self, st: SimState) -> list[Request] | None:
        steps = self._eligible_steps(st)
        available = sum(len(self.groups[s].finished) for s in steps)
        if available < st.batch_size:
            return None

        collected: list[Request] = []
        for step in steps:
            group = self.groups[step]
            needed = st.batch_size - len(collected)
            collected.extend(list(group.finished)[:needed])
            if len(collected) >= st.batch_size:
                break
        return collected[:st.batch_size]

    def take_batch(self, st: SimState) -> list[Request] | None:
        batch = self.peek_batch(st)
        if batch is None:
            return None

        remaining = len(batch)
        for step in self._eligible_steps(st):
            group = self.groups[step]
            take = min(remaining, len(group.finished))
            for _ in range(take):
                group.finished.popleft()
            remaining -= take
            if remaining == 0:
                break
        return batch

    def on_train_version_advance(self, train_version: int, st: SimState) -> None:
        self.budget += st.batch_size
        self._ensure_group(train_version)

    def expire(self, st: SimState, expected_train_version: int) -> list[Request]:
        max_gc_step = (expected_train_version - 1) - st.max_staleness
        if max_gc_step < 0:
            return []
        by_rid = {r.rid: r for r in (*st.inflight, *st.ready)}
        victims = []
        for step in [s for s in self.groups if s <= max_gc_step]:
            group = self.groups.pop(step)
            # Finished prompts that aged out of the consume window are dropped,
            # not consumed. This can happen when they complete after the last
            # take_batch that was still allowed to pull from this group.
            while group.finished:
                req = group.finished.popleft()
                victims.append(req)
                self.completed_prompts -= 1
                self.gc_dropped += 1
            for rid in group.running:
                req = by_rid.get(rid)
                if req is not None:
                    victims.append(req)
                self.running_prompts -= 1
                self.gc_dropped += 1
        return victims

    def check_invariants(self, st: SimState) -> None:
        floor = st.train_version - st.max_staleness
        for step, group in self.groups.items():
            if step < floor:
                assert not group.running and not group.finished, (
                    f"Group {step} below floor {floor} (version={st.train_version}) "
                    f"still has {len(group.running)} running / "
                    f"{len(group.finished)} finished."
                )
