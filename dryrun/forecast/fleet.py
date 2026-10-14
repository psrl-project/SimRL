"""Fleet-level forward forecast: one training-pipeline projection per placement decision.

A `FleetForecast` bundles a per-instance drain forecast so a single hypothetical placement can be
priced against every instance at once, and `forecast` turns the live simulator into the `Pipeline`
the `inventory_deadline` router strategy compares options on.

Multi-buffer chaining needs no fixed point. Buffer `k0`'s READY time resolves from finish times at
or before `k0`'s own publish, and every publish is strictly later than its own buffer's READY time,
so an `Interrupt` scheduled at `publish_k` cannot move anything resolved for buffer `k`. Evaluating
buffer `k0` with no interrupts fixes `publish_k0`, which fixes the interrupt for `k0 + 1`, and so on.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from psrl.workers.ps.inventory_forecast import InventorySnapshot

from ..component.rollout.engine import Request, TrajView, traj_view
from .drain import InstanceForecast
from .length import LengthPredictor
from .pipeline import Pipeline, PipelineForecast, TrainState, group_etas
from .snapshot import Arrival, Interrupt

if TYPE_CHECKING:
    from ..simulator.runtime import SchedulingRuntime


@dataclass(frozen=True, slots=True)
class FleetForecast:
    """Every instance's drain forecast, so one placement decision updates a single member."""

    instances: Mapping[int, InstanceForecast]

    def with_placement(self, instance_id: int, arrival: Arrival) -> FleetForecast:
        """Return the fleet in which `arrival` also reaches `instance_id`."""
        return FleetForecast({**self.instances, instance_id: self.instances[instance_id].with_arrival(arrival)})

    def with_interrupt(self, interrupt: Interrupt) -> FleetForecast:
        """Return the fleet in which `interrupt` publishes against every instance."""
        return FleetForecast({i: fc.with_interrupt(interrupt) for i, fc in self.instances.items()})

    def finish_times(self) -> dict[int, float]:
        """Union of every instance's completion times, keyed by the globally unique request id."""
        out: dict[int, float] = {}
        for fc in self.instances.values():
            out.update(fc.finish_times().finish_times)
        return out


def members_of(runtime: SchedulingRuntime) -> dict[int, list[int]]:
    """
    Map every group holding a live reservation to its member request ids.

    A group is reserved from its first placement until it is consumed, and its members then live in
    `inflight` while unfinished and in `pending` once finished. Staged groups are not yet reserved
    and are deliberately excluded: the inventory never saw them, so they belong in `new_groups`
    rather than in the snapshot a `PipelineForecast` replays.
    """
    out: dict[int, list[int]] = {}
    for req in runtime.inflight:
        out.setdefault(req.group_id, []).append(req.rid)
    for group_id, members in runtime.pending.items():
        out.setdefault(group_id, []).extend(members.keys())
    return out


def arrival_of(req: Request, now: float, remaining: Callable[[TrajView], float]) -> Arrival:
    """Project a request into the hypothetical arrival the fleet forecast admits."""
    traj = traj_view(req)
    return Arrival(
        key=req.rid,
        arrive_time=now,
        prompt_len=req.prompt_len,
        generated=req.generated,
        remaining=max(1, round(remaining(traj))),
        prompt_key=req.source_group_id,
    )


def placement_order(
    queued_by_group: Mapping[int, Sequence[Request]],
    deadline_of: Callable[[int], int],
    remaining_of: Callable[[TrajView], float],
) -> list[Request]:
    """
    Order queued members by ascending deadline buffer, then longest predicted remaining first.

    This is the greedy order `estimate_queued`, and anything layered on it, folds members in. The
    sort is stable, so two members equal on both keys keep the order the queue offered them.

    Args:
        queued_by_group (Mapping[int, Sequence[Request]]): Queued members by group id, all reserved.
        deadline_of (Callable[[int], int]): A reserved group's deadline buffer, from the snapshot.
        remaining_of (Callable[[TrajView], float]): Predicted tokens left for a member.

    Returns:
        list[Request]: The members in the order they are folded.
    """
    items: list[tuple[int, float, Request]] = []
    for group_id, reqs in queued_by_group.items():
        deadline = deadline_of(group_id)
        for req in reqs:
            items.append((deadline, -float(remaining_of(traj_view(req))), req))
    items.sort(key=lambda item: (item[0], item[1]))
    return [req for _, _, req in items]


def fold_member(
    fleet: FleetForecast,
    req: Request,
    remaining_of: Callable[[TrajView], float],
    *,
    now: float,
) -> tuple[FleetForecast, int | None, float]:
    """
    Fold one queued member onto the instance it would finish on soonest.

    Args:
        fleet (FleetForecast): The committed forecasts, without this member.
        req (Request): The queued member to place.
        remaining_of (Callable[[TrajView], float]): Predicted tokens left for the member.
        now (float): The instant the member would be admitted.

    Returns:
        tuple[FleetForecast, int | None, float]: The fleet with the member folded in (unchanged when
            no instance can admit it without overflowing), the chosen instance id or None, and the
            member's finish time, `inf` when unplaced.
    """
    arrival = arrival_of(req, now, remaining_of)
    best_id: int | None = None
    best_finish = float("inf")
    for instance_id, fc in fleet.instances.items():
        result = fc.with_arrival(arrival).finish_times()
        if result.kv_overflow:
            continue
        candidate = result.finish_times[req.rid]
        if candidate < best_finish:
            best_id, best_finish = instance_id, candidate
    if best_id is None:
        return fleet, None, math.inf
    return fleet.with_placement(best_id, arrival), best_id, best_finish


def estimate_queued(
    fleet: FleetForecast,
    queued_by_group: Mapping[int, Sequence[Request]],
    deadline_of: Callable[[int], int],
    remaining_of: Callable[[TrajView], float],
    *,
    now: float,
) -> tuple[FleetForecast, dict[int, float]]:
    """
    Fold every queued-but-unplaced member onto the instance it would finish on soonest.

    Members are consumed in ascending deadline buffer, then longest predicted remaining first, the
    ordering the headroom study found valuable. Each member is folded into the chosen instance's
    forecast before the next is evaluated, so a later member sees the capacity the earlier one took.
    A member no instance can admit without overflowing its pool is left unplaced, and its group's
    ETA stays infinite.

    Args:
        fleet (FleetForecast): The committed forecasts, without the queued members.
        queued_by_group (Mapping[int, Sequence[Request]]): Queued members by group id, all reserved.
        deadline_of (Callable[[int], int]): A reserved group's deadline buffer, from the snapshot.
        remaining_of (Callable[[TrajView], float]): Predicted tokens left for a member.
        now (float): The instant the queued members would be admitted.

    Returns:
        tuple[FleetForecast, dict[int, float]]: The fleet with every placed member folded in, and
            the finish time of each placed member by request id.
    """
    finish: dict[int, float] = {}
    for req in placement_order(queued_by_group, deadline_of, remaining_of):
        fleet, best_id, best_finish = fold_member(fleet, req, remaining_of, now=now)
        if best_id is None:
            continue
        finish[req.rid] = best_finish
    return fleet, finish


def baseline_fleet(runtime: SchedulingRuntime, remaining_of: Callable[[TrajView], float]) -> FleetForecast:
    """Project every instance into a drain forecast, with nothing hypothetical layered on."""
    return FleetForecast(
        {
            inst.instance_id: InstanceForecast(runtime.instance_state(inst, remaining_of))
            for inst in runtime.instances
        }
    )


def _queued_admitted(runtime: SchedulingRuntime, members: Mapping[int, Sequence[int]]) -> dict[int, list[Request]]:
    """Queued requests whose group already holds a reservation, which the snapshot can replay."""
    return {group_id: reqs for group_id, reqs in runtime.queued().items() if group_id in members}


def sequence_hint(
    group_eta: Mapping[int, float],
    members: Mapping[int, Sequence[int]],
    lengths_by_rid: Mapping[int, int],
    batch_size: int,
    group_size: int,
) -> tuple[int, ...]:
    """Price the next training buffer from the groups predicted to finish first.

    `cycle_time` takes one entry per training response, so the hint is per response, not per group.
    A buffer trains on the groups that finish earliest, not the groups reserved to it: psrl's occupy
    rule moves an entry into whichever buffer has room, so the ready set is the earliest finishers.
    The `batch_size` groups with the smallest ETA are therefore the ones priced. A group already
    finished has no live member left to price, so the priced responses are padded with their mean.
    """
    ranked = sorted(
        (eta, prompt_id) for prompt_id, eta in group_eta.items() if prompt_id in members
    )
    per_response: list[int] = []
    for _, prompt_id in ranked[:batch_size]:
        for rid in members[prompt_id]:
            if rid in lengths_by_rid:
                per_response.append(lengths_by_rid[rid])
    if not per_response:
        return (1,)
    mean = max(1, round(sum(per_response) / len(per_response)))
    batch = batch_size * group_size
    return tuple((per_response + [mean] * batch)[:batch])


def group_etas_from_fleet(
    fleet: FleetForecast,
    members: Mapping[int, Sequence[int]],
    runtime: SchedulingRuntime,
) -> dict[int, float]:
    """Group ETA as the latest of its members, mixing placed forecasts and known completions."""
    finished = {rid: time for group in runtime.pending.values() for rid, time in group.items()}
    return group_etas(fleet.finish_times(), members, finished)


def forecast(
    runtime: SchedulingRuntime,
    length_predictor: LengthPredictor,
    *,
    horizon_buffers: int = 2,
    override: tuple[int, Arrival] | None = None,
    sequence_lengths_hint: Sequence[int] | None = None,
) -> Pipeline:
    """
    Predict the training pipeline from the live simulator, over the next `horizon_buffers` buffers.

    Args:
        runtime (SchedulingRuntime): The live, read-only simulator facade. Its policy must be psrl.
        length_predictor (LengthPredictor): Predicted remaining length for every request.
        horizon_buffers (int): How many buffers to project. Clamped to the buffers the snapshot
            already holds, since opening more would require a future admission this forecast has no
            basis to invent.
        override (tuple[int, Arrival] | None): One hypothetical placement under evaluation as
            `(instance_id, arrival)`. None evaluates the committed state with no new decision.
        sequence_lengths_hint (Sequence[int] | None): Per-response training lengths for the next
            step. None derives them from the groups predicted to finish first.

    Returns:
        Pipeline: `F_k` and `E_k` for buffers `k0 .. k0 + horizon - 1`, `k0` the current buffer.
    """
    snapshot = runtime.inventory()
    remaining_of = length_predictor.remaining
    now = runtime.now
    members = members_of(runtime)

    fleet = baseline_fleet(runtime, remaining_of)
    overridden = None
    if override is not None:
        instance_id, arrival = override
        fleet = fleet.with_placement(instance_id, arrival)
        overridden = arrival.key

    queued = _queued_admitted(runtime, members)
    if overridden is not None:
        queued = {group_id: [r for r in reqs if r.rid != overridden] for group_id, reqs in queued.items()}
    fleet, _ = estimate_queued(fleet, queued, snapshot.deadline, remaining_of, now=now)

    eta = group_etas_from_fleet(fleet, members, runtime)
    if sequence_lengths_hint is None:
        lengths_by_rid = {
            req.rid: max(1, round(req.prompt_len + length_predictor.predict_total(traj_view(req))))
            for req in runtime.inflight
        }
        sequence_lengths_hint = sequence_hint(
            eta, members, lengths_by_rid, runtime.batch_size, runtime.group_size
        )
    train = runtime.train_state(tuple(sequence_lengths_hint))
    horizon = min(horizon_buffers, len(snapshot.buffers))
    return chain_buffers(fleet, snapshot, train, eta, members, runtime, horizon)


def chain_buffers(
    fleet: FleetForecast,
    snapshot: InventorySnapshot,
    train: TrainState,
    eta: Mapping[int, float],
    members: Mapping[int, Sequence[int]],
    runtime: SchedulingRuntime,
    horizon: int,
    new_groups: Sequence[tuple[int, float]] = (),
) -> Pipeline:
    """
    Chain `horizon` buffers through the training pipeline, scheduling an `Interrupt` per publish.

    Shared by `forecast` and any evaluator that needs the same multi-buffer projection on top of a
    different fleet. See the module docstring for why no fixed point is needed.

    Args:
        fleet (FleetForecast): The fleet whose finish times price buffer 0.
        snapshot (InventorySnapshot): The inventory snapshot `eta` and `new_groups` replay over.
        train: The `TrainState` pricing every buffer's cycle.
        eta (Mapping[int, float]): Buffer-0 group ETAs, mixing placed forecasts and completions.
        members (Mapping[int, Sequence[int]]): Group id to its member request ids.
        runtime (SchedulingRuntime): The live simulator facade, for `runtime.pending` completions.
        horizon (int): How many buffers to project, clamped to the snapshot's own buffers.
        new_groups (Sequence[tuple[int, float]]): Hypothetical admissions passed to buffer 0 only.

    Returns:
        Pipeline: `F_k` and `E_k` for buffers `k0 .. k0 + horizon - 1`.
    """
    proj = PipelineForecast(snapshot, train)
    pipeline = proj.evaluate(eta, new_groups)
    interrupts: list[Interrupt] = []
    for step in range(1, horizon):
        # The `step - 1`-th publish is fixed once the buffers before it are priced, so the next
        # interrupt can be scheduled without ever iterating to a fixed point.
        publish = pipeline.publish_times[step - 1]
        interrupts.append(Interrupt(time=publish, resume_delay=train.weight_load + train.reconfig))
        staged = fleet
        for interrupt in interrupts:
            staged = staged.with_interrupt(interrupt)
        pipeline = proj.evaluate(group_etas_from_fleet(staged, members, runtime))

    return Pipeline(
        buffers=pipeline.buffers[:horizon],
        ready=pipeline.ready[:horizon],
        train_end=pipeline.train_end[:horizon],
        publish_times=pipeline.publish_times[:horizon],
    )


def evaluate_placement(
    runtime: SchedulingRuntime,
    length_predictor: LengthPredictor,
    member: Request,
    options: Sequence[int | None],
    *,
    horizon_buffers: int = 2,
    sequence_lengths_hint: Sequence[int] | None = None,
) -> dict[int | None, Pipeline]:
    """
    Price one queued member's placement options, each as its own pipeline projection.

    Args:
        runtime (SchedulingRuntime): The live simulator facade.
        length_predictor (LengthPredictor): Predicted remaining length for every request.
        member (Request): The queued member whose placement is under evaluation.
        options (Sequence[int | None]): Candidate instance ids, plus None for staying queued.

    Returns:
        dict[int | None, Pipeline]: One pipeline per option, keyed by the option.
    """
    arrival = arrival_of(member, runtime.now, length_predictor.remaining)
    out: dict[int | None, Pipeline] = {}
    for option in options:
        override = None if option is None else (option, arrival)
        out[option] = forecast(
            runtime,
            length_predictor,
            horizon_buffers=horizon_buffers,
            override=override,
            sequence_lengths_hint=sequence_lengths_hint,
        )
    return out
