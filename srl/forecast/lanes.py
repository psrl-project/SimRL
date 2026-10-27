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


@dataclass(frozen=True, slots=True)
class LanePlan:
    """Where every job goes, and which instances ended up as fast lanes."""

    assignment: dict[int, int]
    fast_instances: frozenset[int]
    """Instances the search deliberately thinned out. Empty when the bulk plan was never beaten."""
    objective: float
    moves: int
    """Upgrades the search accepted, for telemetry and for bounding the search in tests."""


def plan_lanes(
    jobs: Sequence[LaneJob],
    slots: Sequence[LaneSlot],
    *,
    snapshot: InventorySnapshot,
    train: TrainState,
    members: Mapping[int, Sequence[int]],
    finished: Mapping[int, float],
    now: float,
    pinned: Mapping[int, int] | None = None,
    horizon_buffers: int = 2,
    c_max: int = 4,
    eps_frac: float = 0.01,
    max_moves: int = 8,
) -> LanePlan:
    """
    Assign jobs to slots, then thin out slots until no late buffer can be improved.

    The search starts from the balanced assignment and repeatedly asks the same question: which
    pending buffer would make the trainer idle, and which group holds it up? It then tries to
    give that group's late members fewer neighbours, either on a slot already thinned out or by
    thinning a fresh one, and keeps the move that lowers the objective most. The number of fast
    lanes and their occupancy are outputs of this loop, never inputs.

    Args:
        jobs (Sequence[LaneJob]): Every job in flight or awaiting placement.
        slots (Sequence[LaneSlot]): The fleet.
        snapshot (InventorySnapshot): The staleness inventory to replay.
        train (TrainState): The training loop's state, held fixed across the search.
        members (Mapping[int, Sequence[int]]): Group id to its member request ids.
        finished (Mapping[int, float]): Completion time of an already-finished member, by id.
        now (float): Simulated time the forecasts start from.
        pinned (Mapping[int, int] | None): Jobs already placed, which keep their slot.
        horizon_buffers (int): Buffers the objective sums over.
        c_max (int): Jobs a fast lane may hold.
        eps_frac (float): Fraction of one training cycle a move must save to be accepted.
        max_moves (int): Upgrades the search may accept.

    Returns:
        LanePlan: The assignment, the fast lanes, the objective, and the moves accepted.

    Raises:
        ValueError: If `c_max` is below 1 or `slots` is empty.
    """
    if c_max < 1:
        raise ValueError(f"c_max must be at least 1, got {c_max!r}.")
    if not slots:
        raise ValueError("plan_lanes needs at least one slot.")
    pinned = pinned or {}
    slot_ids = [slot.instance_id for slot in slots]
    by_rid = {job.rid: job for job in jobs}
    pricing_args = dict(
        snapshot=snapshot,
        train=train,
        members=members,
        finished=finished,
        now=now,
        horizon_buffers=horizon_buffers,
    )

    assignment = lpt_assign(jobs, slot_ids, pinned=pinned)
    best = price(assignment, jobs, slots, **pricing_args)
    eps = eps_frac * train.cycle
    fast: set[int] = set()
    unfixable: set[int] = set()
    moves = 0

    while moves < max_moves:
        late = _first_late_buffer(best.pipeline, train, now, horizon_buffers, unfixable)
        if late is None:
            break
        buffer_id = best.pipeline.buffers[late]
        target = _target_time(best.pipeline, train, now, late)
        critical = _critical_group(snapshot, best.group_eta, members, buffer_id)
        if critical is None:
            unfixable.add(buffer_id)
            continue
        movers = [
            rid
            for rid in members[critical]
            if rid in by_rid and rid not in pinned and best.member_finish.get(rid, math.inf) > target
        ]
        candidates = _candidate_moves(movers, assignment, by_rid, slot_ids, fast, c_max, pinned)
        winner = None
        for cand_assignment, cand_fast in candidates:
            pricing = price(cand_assignment, jobs, slots, **pricing_args)
            improves = pricing.objective < best.objective - eps
            if improves and (winner is None or pricing.objective < winner[0].objective):
                winner = (pricing, cand_assignment, cand_fast)
        if winner is None:
            unfixable.add(buffer_id)
            continue
        best, assignment, fast = winner[0], winner[1], set(winner[2])
        moves += 1

    return LanePlan(
        assignment=dict(assignment),
        fast_instances=frozenset(fast),
        objective=best.objective,
        moves=moves,
    )


def _target_time(pipeline: Pipeline, train: TrainState, now: float, index: int) -> float:
    """When buffer `index` must be ready for the trainer not to idle before it."""
    if index > 0:
        return pipeline.train_end[index - 1]
    return now if train.training_end is None else train.training_end + train.weight_load


def _first_late_buffer(
    pipeline: Pipeline,
    train: TrainState,
    now: float,
    horizon_buffers: int,
    skip: set[int],
) -> int | None:
    """Index of the earliest priced buffer the trainer would wait for, skipping `skip`."""
    for index in range(min(horizon_buffers, len(pipeline.buffers))):
        if pipeline.buffers[index] in skip:
            continue
        if pipeline.ready[index] > _target_time(pipeline, train, now, index):
            return index
    return None


def _critical_group(
    snapshot: InventorySnapshot,
    group_eta: Mapping[int, float],
    members: Mapping[int, Sequence[int]],
    buffer_id: int,
) -> int | None:
    """
    The latest-finishing group that buffer `buffer_id` cannot be ready without.

    A group whose deadline is at or before `buffer_id` must be consumed by it at the latest, so
    the buffer cannot fill until every one of them has finished. The slowest of those is what
    sets the buffer's READY time, and the only group worth speeding up.
    """
    worst_eta = -math.inf
    worst: int | None = None
    for group_id in members:
        try:
            deadline = snapshot.deadline(group_id)
        except KeyError:
            continue
        if deadline > buffer_id:
            continue
        eta = group_eta.get(group_id, math.inf)
        if eta > worst_eta:
            worst_eta, worst = eta, group_id
    return worst


def _candidate_moves(
    movers: Sequence[int],
    assignment: Mapping[int, int],
    by_rid: Mapping[int, LaneJob],
    slot_ids: Sequence[int],
    fast: set[int],
    c_max: int,
    pinned: Mapping[int, int],
) -> list[tuple[dict[int, int], set[int]]]:
    """
    Enumerate the ways `movers` could get fewer neighbours.

    Two shapes only: join a fast lane that still has room, or thin a bulk lane out by
    rebalancing its current jobs over the remaining bulk lanes. Enumerating every bulk lane
    would price N nearly identical options, so only the emptiest one is offered. A pinned
    resident of the thinned slot is never part of what gets rebalanced, even when that slot
    is chosen as the new host.
    """
    if not movers:
        return []
    out: list[tuple[dict[int, int], set[int]]] = []
    load = {slot_id: 0 for slot_id in slot_ids}
    held = {slot_id: 0 for slot_id in slot_ids}
    for rid, slot_id in assignment.items():
        load[slot_id] += by_rid[rid].remaining
        held[slot_id] += 1

    for slot_id in sorted(fast):
        if held[slot_id] + len(movers) > c_max:
            continue
        candidate = dict(assignment)
        for rid in movers:
            candidate[rid] = slot_id
        out.append((candidate, set(fast)))

    bulk = [slot_id for slot_id in slot_ids if slot_id not in fast]
    if len(bulk) > 1 and len(movers) <= c_max:
        host = min(bulk, key=lambda s: (load[s], s))
        remaining_bulk = [slot_id for slot_id in bulk if slot_id != host]
        evicted = [
            by_rid[rid]
            for rid, slot_id in assignment.items()
            if slot_id == host and rid not in movers and rid not in pinned
        ]
        candidate = {
            rid: slot_id for rid, slot_id in assignment.items() if slot_id != host or rid in pinned
        }
        candidate.update(lpt_assign(evicted, remaining_bulk))
        for rid in movers:
            candidate[rid] = host
        out.append((candidate, {*fast, host}))
    return out


def plan_cadence(
    jobs: Sequence[LaneJob],
    slots: Sequence[LaneSlot],
    *,
    snapshot: InventorySnapshot,
    train: TrainState,
    members: Mapping[int, Sequence[int]],
    finished: Mapping[int, float],
    now: float,
    pinned: Mapping[int, int] | None = None,
    horizon_buffers: int = 2,
    c_max: int = 4,
    eps_frac: float = 0.01,
    max_moves: int = 8,
    bisect_steps: int = 8,
) -> LanePlan:
    """
    Plan for the smallest uniform step period the fleet can actually hold.

    `plan_lanes` improves the nearest late buffer first, so it will spend its move budget on
    buffer k and leave k+1 late. This asks the opposite question: what is the smallest period
    `P` such that spacing every deadline `P` apart leaves a plan that meets all of them? The
    answer is found by bisection between `train.cycle` and the bulk (un-upgraded) plan's own
    mean spacing, and the plan that achieved it is returned. The exception is when
    `plan_lanes`'s own plan on the same input happens to price lower, since the two searches
    optimize different things (uniform spacing vs. total pipeline cost) and this function must
    never regress relative to `plan_lanes`.

    Args:
        jobs (Sequence[LaneJob]): Every job in flight or awaiting placement.
        slots (Sequence[LaneSlot]): The fleet.
        snapshot (InventorySnapshot): The staleness inventory to replay.
        train (TrainState): The training loop's state, held fixed across the search.
        members (Mapping[int, Sequence[int]]): Group id to its member request ids.
        finished (Mapping[int, float]): Completion time of an already-finished member, by id.
        now (float): Simulated time the forecasts start from.
        pinned (Mapping[int, int] | None): Jobs already placed, which keep their slot.
        horizon_buffers (int): Buffers the objective sums over.
        c_max (int): Jobs a fast lane may hold.
        eps_frac (float): Fraction of one training cycle a move must save to be accepted.
        max_moves (int): Upgrades each period's search may accept.
        bisect_steps (int): Bisection iterations between `train.cycle` and the bulk plan's own
            mean period. Eight brings the bracket inside 0.4% of its width. Zero skips the
            search entirely and returns `plan_lanes`'s own result.

    Returns:
        LanePlan: Whichever of the smallest-feasible-period plan and `plan_lanes`'s own plan has
            the lower objective. Exactly `plan_lanes`'s result when no period in the bracket is
            feasible, or when `bisect_steps` is 0.

    Raises:
        ValueError: If `c_max` is below 1 or `slots` is empty.
    """
    if c_max < 1:
        raise ValueError(f"c_max must be at least 1, got {c_max!r}.")
    if not slots:
        raise ValueError("plan_cadence needs at least one slot.")

    plan_lanes_args = dict(
        snapshot=snapshot,
        train=train,
        members=members,
        finished=finished,
        now=now,
        pinned=pinned,
        horizon_buffers=horizon_buffers,
        c_max=c_max,
        eps_frac=eps_frac,
        max_moves=max_moves,
    )
    fallback = plan_lanes(jobs, slots, **plan_lanes_args)
    if bisect_steps == 0:
        return fallback

    pinned_map = pinned or {}
    slot_ids = [slot.instance_id for slot in slots]
    by_rid = {job.rid: job for job in jobs}
    pricing_args = dict(
        snapshot=snapshot,
        train=train,
        members=members,
        finished=finished,
        now=now,
        horizon_buffers=horizon_buffers,
    )
    eps = eps_frac * train.cycle

    bulk_assignment = lpt_assign(jobs, slot_ids, pinned=pinned_map)
    bulk = price(bulk_assignment, jobs, slots, **pricing_args)
    n = min(horizon_buffers, len(bulk.pipeline.buffers))
    if n == 0:
        return fallback
    target_0 = _target_time(bulk.pipeline, train, now, 0)
    bulk_mean_period = (bulk.pipeline.ready[n - 1] - target_0) / n

    def deadlines_for(period: float) -> list[float]:
        return [target_0 + (k + 1) * period for k in range(n)]

    def feasible_plan_for(period: float) -> LanePlan | None:
        return _search_for_period(
            jobs,
            slots,
            by_rid,
            slot_ids,
            pinned_map,
            pricing_args,
            snapshot,
            members,
            deadlines_for(period),
            horizon_buffers,
            c_max,
            eps,
            max_moves,
        )

    lo, hi = train.cycle, bulk_mean_period
    best_feasible: LanePlan | None = feasible_plan_for(hi) if hi > lo else None
    if best_feasible is None:
        return fallback

    for _ in range(bisect_steps):
        mid = (lo + hi) / 2
        candidate = feasible_plan_for(mid)
        if candidate is not None:
            best_feasible, hi = candidate, mid
        else:
            lo = mid

    return best_feasible if best_feasible.objective <= fallback.objective else fallback


def _feasible_for_deadlines(pipeline: Pipeline, deadlines: Sequence[float], horizon_buffers: int) -> bool:
    """Whether every priced buffer in the horizon is ready at or before its fixed deadline."""
    n = min(horizon_buffers, len(pipeline.buffers))
    return all(pipeline.ready[k] <= deadlines[k] for k in range(n))


def _overshoot(pipeline: Pipeline, deadlines: Sequence[float], horizon_buffers: int) -> float:
    """Total seconds every priced buffer in the horizon is late past its fixed deadline."""
    n = min(horizon_buffers, len(pipeline.buffers))
    return sum(max(0.0, pipeline.ready[k] - deadlines[k]) for k in range(n))


def _first_late_buffer_for_deadlines(
    pipeline: Pipeline,
    deadlines: Sequence[float],
    horizon_buffers: int,
    skip: set[int],
) -> int | None:
    """Index of the earliest priced buffer in the horizon late past its fixed deadline, skipping `skip`."""
    n = min(horizon_buffers, len(pipeline.buffers))
    for index in range(n):
        if pipeline.buffers[index] in skip:
            continue
        if pipeline.ready[index] > deadlines[index]:
            return index
    return None


def _search_for_period(
    jobs: Sequence[LaneJob],
    slots: Sequence[LaneSlot],
    by_rid: Mapping[int, LaneJob],
    slot_ids: Sequence[int],
    pinned: Mapping[int, int],
    pricing_args: dict,
    snapshot: InventorySnapshot,
    members: Mapping[int, Sequence[int]],
    deadlines: Sequence[float],
    horizon_buffers: int,
    c_max: int,
    eps: float,
    max_moves: int,
) -> LanePlan | None:
    """
    Search for an assignment meeting every fixed `deadlines[k]`, by the same move set
    `plan_lanes` uses (`_candidate_moves`), accepting a move when it strictly reduces total
    overshoot past those deadlines rather than when it lowers the training-pipeline objective.

    Returns:
        LanePlan | None: The feasible plan, or `None` when the search exhausts `max_moves`
            without meeting every deadline in the horizon.
    """
    assignment = lpt_assign(jobs, slot_ids, pinned=pinned)
    best = price(assignment, jobs, slots, **pricing_args)
    best_overshoot = _overshoot(best.pipeline, deadlines, horizon_buffers)
    fast: set[int] = set()
    unfixable: set[int] = set()
    moves = 0

    while best_overshoot > 0 and moves < max_moves:
        late = _first_late_buffer_for_deadlines(best.pipeline, deadlines, horizon_buffers, unfixable)
        if late is None:
            break
        buffer_id = best.pipeline.buffers[late]
        critical = _critical_group(snapshot, best.group_eta, members, buffer_id)
        if critical is None:
            unfixable.add(buffer_id)
            continue
        movers = [
            rid
            for rid in members[critical]
            if rid in by_rid and rid not in pinned and best.member_finish.get(rid, math.inf) > deadlines[late]
        ]
        candidates = _candidate_moves(movers, assignment, by_rid, slot_ids, fast, c_max, pinned)
        winner = None
        for cand_assignment, cand_fast in candidates:
            pricing = price(cand_assignment, jobs, slots, **pricing_args)
            overshoot = _overshoot(pricing.pipeline, deadlines, horizon_buffers)
            if overshoot < best_overshoot - eps and (winner is None or overshoot < winner[0]):
                winner = (overshoot, pricing, cand_assignment, cand_fast)
        if winner is None:
            unfixable.add(buffer_id)
            continue
        best_overshoot, best, assignment, fast = winner[0], winner[1], winner[2], set(winner[3])
        moves += 1

    if not _feasible_for_deadlines(best.pipeline, deadlines, horizon_buffers):
        return None
    return LanePlan(
        assignment=dict(assignment),
        fast_instances=frozenset(fast),
        objective=best.objective,
        moves=moves,
    )
