"""Plan which instance each unplaced request goes to, and how many share it.

A lane is not a configured pool. It is however many requests the plan puts on one instance:
few requests means low per-token latency, many means high aggregate throughput. The planner
therefore emits an assignment, and the instances that ended up holding few requests are the
fast lanes. Nothing here touches the engine, so the whole search is a pure function of the
jobs, the slots, and the inventory snapshot it is given.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from psrl.workers.ps.staleness import InventorySnapshot

from ..component.rollout.engine import blocks_for
from .drain import InstanceForecast
from .pipeline import Pipeline, PipelineForecast, TrainState, group_etas
from .snapshot import InstanceState, Resident


@dataclass(frozen=True, slots=True)
class LaneJob:
    """One request the planner must place."""

    rid: int
    group_id: int
    prompt_key: int
    prompt_len: int
    generated: int
    remaining: int
    """Predicted decode tokens still to generate, at least 1."""

    def __post_init__(self) -> None:
        if self.remaining < 1:
            raise ValueError(f"A job must have at least one token left to generate, got {self.remaining!r}.")

    @property
    def final_ctx(self) -> int:
        """Context the job holds once it finishes, which bounds the blocks it will ever need."""
        return self.prompt_len + self.generated + self.remaining


@dataclass(frozen=True, slots=True)
class LaneSlot:
    """One instance the planner may assign jobs to, and the limits it prices against."""

    instance_id: int
    cost: object
    """The instance's rollout cost model, passed through to `InstanceState`."""
    block_size: int
    kv_blocks: int
    token_budget: int
    max_concurrency: int | None
    tp: int


def lpt_assign(
    jobs: Sequence[LaneJob],
    slot_ids: Sequence[int],
    pinned: Mapping[int, int] | None = None,
) -> dict[int, int]:
    """
    Assign every job to a slot, longest predicted remaining first, onto the least-loaded slot.

    Args:
        jobs (Sequence[LaneJob]): Jobs to place, including the pinned ones.
        slot_ids (Sequence[int]): Instance ids the assignment may use.
        pinned (Mapping[int, int] | None): Request id to the slot it is already on and must
            stay on. Its load counts against that slot, so the free jobs balance around it.

    Returns:
        dict[int, int]: Request id to slot id, covering every job.

    Raises:
        ValueError: If `slot_ids` is empty, or a pinned job names a slot outside `slot_ids`.
    """
    if not slot_ids:
        raise ValueError("lpt_assign needs at least one slot.")
    pinned = pinned or {}
    allowed = set(slot_ids)
    load = {slot_id: 0 for slot_id in slot_ids}
    out: dict[int, int] = {}
    for job in jobs:
        slot_id = pinned.get(job.rid)
        if slot_id is None:
            continue
        if slot_id not in allowed:
            raise ValueError(f"Pinned job {job.rid} names slot {slot_id!r}, which is not in slot_ids.")
        out[job.rid] = slot_id
        load[slot_id] += job.remaining
    free = sorted((j for j in jobs if j.rid not in out), key=lambda j: (-j.remaining, j.rid))
    for job in free:
        slot_id = min(slot_ids, key=lambda s: (load[s], s))
        out[job.rid] = slot_id
        load[slot_id] += job.remaining
    return out


def kv_safe_concurrency(jobs: Sequence[LaneJob], slot: LaneSlot) -> int:
    """
    Return how many of `jobs` may decode at once without the pool ever overflowing.

    A job's block usage grows monotonically to `blocks_for(final_ctx)`, so the sum of those
    bounds over a running set bounds that set's peak. Counting the longest jobs first makes
    the answer independent of the order `jobs` arrives in. The result is at least 1 whenever
    there is any job: an instance holding one request cannot do better by holding none, and
    the engine would preempt rather than stall.

    Args:
        jobs (Sequence[LaneJob]): The jobs assigned to this slot.
        slot (LaneSlot): The slot's block size and pool size.

    Returns:
        int: The largest safe running count, 0 when `jobs` is empty.
    """
    if not jobs:
        return 0
    used = 0
    count = 0
    for job in sorted(jobs, key=lambda j: (-j.final_ctx, j.rid)):
        used += blocks_for(job.final_ctx, slot.block_size)
        if used > slot.kv_blocks:
            break
        count += 1
    return max(1, count)


@dataclass(frozen=True, slots=True)
class Pricing:
    """What one assignment costs the training pipeline."""

    objective: float
    """`J`, the sum of `E_k` over the priced horizon. `inf` when any buffer never becomes ready."""
    pipeline: Pipeline
    member_finish: dict[int, float]
    group_eta: dict[int, float]


def state_for(slot: LaneSlot, jobs: Sequence[LaneJob], now: float) -> InstanceState:
    """
    Project one slot and the jobs assigned to it into the drain forecast's input.

    Every job enters as a waiting resident with no KV computed, which is what a publish leaves
    behind: `abort_all` frees the pool and resets `prefilled`, keeping the generated tokens. The
    forecast's running batch is capped at `kv_safe_concurrency`, so the replay queues what the
    pool cannot hold instead of reporting an overflow that would void its finish times.

    Args:
        slot (LaneSlot): The instance's cost model and limits.
        jobs (Sequence[LaneJob]): Jobs assigned to this slot.
        now (float): Simulated time the forecast starts from.

    Returns:
        InstanceState: The forecast's input state.
    """
    safe = kv_safe_concurrency(jobs, slot)
    cap = safe if slot.max_concurrency is None else min(slot.max_concurrency, safe)
    waiting = tuple(
        Resident(
            key=job.rid,
            group_id=job.group_id,
            prompt_key=job.prompt_key,
            prompt_len=job.prompt_len,
            ctx=job.prompt_len + job.generated,
            prefilled=0,
            remaining=job.remaining,
            running=False,
        )
        for job in jobs
    )
    return InstanceState(
        cost=slot.cost,
        block_size=slot.block_size,
        kv_blocks=slot.kv_blocks,
        token_budget=slot.token_budget,
        max_concurrency=cap if jobs else None,
        chunk_cap=None,
        tp=slot.tp,
        now=now,
        waiting=waiting,
        running=(),
    )


def price(
    assignment: Mapping[int, int],
    jobs: Sequence[LaneJob],
    slots: Sequence[LaneSlot],
    *,
    snapshot: InventorySnapshot,
    train: TrainState,
    members: Mapping[int, Sequence[int]],
    finished: Mapping[int, float],
    now: float,
    horizon_buffers: int,
) -> Pricing:
    """
    Price `assignment` as one training pipeline.

    Each slot's jobs are drained independently, since a closed assignment gives the instances
    nothing to share. The resulting member finish times become group ETAs, the inventory replay
    turns those into each buffer's READY time, and the training chain turns those into `E_k`.

    Args:
        assignment (Mapping[int, int]): Request id to slot id, covering every job.
        jobs (Sequence[LaneJob]): The jobs the assignment covers.
        slots (Sequence[LaneSlot]): Every slot the assignment may name.
        snapshot (InventorySnapshot): The staleness inventory to replay.
        train (TrainState): The training loop's state, priced for the next cycle.
        members (Mapping[int, Sequence[int]]): Group id to its member request ids.
        finished (Mapping[int, float]): Completion time of an already-finished member, by id.
        now (float): Simulated time the forecasts start from.
        horizon_buffers (int): Buffers the objective sums over, from the earliest pending one.

    Returns:
        Pricing: The objective and the pipeline it came from.

    Raises:
        ValueError: If `assignment` names a slot that is not in `slots`, or omits a job.
    """
    by_slot: dict[int, list[LaneJob]] = {slot.instance_id: [] for slot in slots}
    for job in jobs:
        slot_id = assignment.get(job.rid)
        if slot_id is None:
            raise ValueError(f"Assignment omits job {job.rid}.")
        if slot_id not in by_slot:
            raise ValueError(f"Assignment sends job {job.rid} to unknown slot {slot_id!r}.")
        by_slot[slot_id].append(job)

    member_finish: dict[int, float] = {}
    for slot in slots:
        placed = by_slot[slot.instance_id]
        if not placed:
            continue
        result = InstanceForecast(state_for(slot, placed, now)).finish_times()
        member_finish.update(result.finish_times)

    eta = group_etas(member_finish, members, finished)
    pipeline = PipelineForecast(snapshot, train).evaluate(eta)
    horizon = pipeline.train_end[:horizon_buffers]
    objective = math.inf if any(math.isinf(e) for e in horizon) else sum(horizon)
    return Pricing(objective=objective, pipeline=pipeline, member_finish=member_finish, group_eta=eta)
