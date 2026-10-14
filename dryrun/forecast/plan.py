"""Incremental placement planning for one router dispatch burst.

`PlanSession.build` snapshots the live simulator once, folds every queued member of an admitted
group into a tentative tail the way `estimate_queued` would, and then lets the router ask about one
member or one staged group's admission at a time. Each `evaluate_*` call recomputes only the
instance or two whose arrival set the option changes, so a burst of hundreds of queued members costs
one fold at session build plus one or two drain replays per option, not a full `forecast` per option.

An `Option` is an instance id to place on now, or `None` for DEFER (stay queued until the next
dispatch, or DEFER_GROUP for a staged group admission).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from ..component.rollout.engine import Request, traj_view
from .drain import InstanceForecast
from .fleet import (
    FleetForecast,
    arrival_of,
    baseline_fleet,
    chain_buffers,
    fold_member,
    group_etas_from_fleet,
    members_of,
    placement_order,
    sequence_hint,
)
from .length import LengthPredictor
from .pipeline import Pipeline, TrainState

if TYPE_CHECKING:
    from ..simulator.runtime import SchedulingRuntime

Option = int | None
"""An instance id to place on now, or `None` for DEFER / DEFER_GROUP."""


def _without_arrival(fc: InstanceForecast, key: int) -> InstanceForecast:
    """Return the forecast in which the arrival keyed `key` never reached the instance."""
    return InstanceForecast(fc.state, [a for a in fc.arrivals if a.key != key], fc.interrupts)


class PlanSession:
    """One dispatch burst's incremental fleet and pipeline state.

    Built once from the live `SchedulingRuntime`, then asked about members and staged groups one at
    a time in the priority order the router offers them, committing each decision before the next is
    evaluated.
    """

    def __init__(
        self,
        runtime: SchedulingRuntime,
        length_predictor: LengthPredictor,
        horizon_buffers: int,
        now: float,
        snapshot,
        rollout_version: int,
        members: dict[int, list[int]],
        staged_groups: dict[int, list[Request]],
        fleet: FleetForecast,
        tail_order: list[Request],
        tail_assignment: dict[int, int],
        defer_until: float | None,
        defer_valid: bool,
        lengths_by_rid: dict[int, int],
        train_cache: dict[tuple[int, ...], TrainState],
        horizon: int,
    ) -> None:
        self.runtime = runtime
        self.length_predictor = length_predictor
        self.horizon_buffers = horizon_buffers
        self.now = now
        self.snapshot = snapshot
        self.rollout_version = rollout_version
        self.members = members
        self.staged_groups = staged_groups
        self.fleet = fleet
        self.tail_order = tail_order
        self.tail_assignment = tail_assignment
        self.defer_until = defer_until
        self.defer_valid = defer_valid
        self.lengths_by_rid = lengths_by_rid
        self.train_cache = train_cache
        self.horizon = horizon

    # --- construction ---------------------------------------------------

    @classmethod
    def build(
        cls,
        runtime: SchedulingRuntime,
        length_predictor: LengthPredictor,
        horizon_buffers: int = 2,
    ) -> PlanSession:
        """
        Snapshot `runtime` and fold its queued admitted-group members into a tentative tail.

        Args:
            runtime (SchedulingRuntime): The live, read-only simulator facade. Its policy must be psrl.
            length_predictor (LengthPredictor): Predicted remaining length for every request.
            horizon_buffers (int): How many buffers `evaluate_*` projects, clamped to the snapshot.
        """
        snapshot = runtime.inventory()
        remaining_of = length_predictor.remaining
        now = runtime.now
        members = members_of(runtime)

        fleet = baseline_fleet(runtime, remaining_of)
        queued = {group_id: reqs for group_id, reqs in runtime.queued().items() if group_id in members}
        tail_order = placement_order(queued, snapshot.deadline, remaining_of)
        tail_assignment: dict[int, int] = {}
        for req in tail_order:
            fleet, chosen, _ = fold_member(fleet, req, remaining_of, now=now)
            if chosen is not None:
                tail_assignment[req.rid] = chosen

        lengths_by_rid = {
            req.rid: max(1, round(req.prompt_len + length_predictor.predict_total(traj_view(req))))
            for req in runtime.inflight
        }
        horizon = min(horizon_buffers, len(snapshot.buffers))
        session = cls(
            runtime=runtime,
            length_predictor=length_predictor,
            horizon_buffers=horizon_buffers,
            now=now,
            snapshot=snapshot,
            rollout_version=runtime.rollout_version,
            members=members,
            staged_groups=dict(runtime.staged_groups),
            fleet=fleet,
            tail_order=tail_order,
            tail_assignment=tail_assignment,
            defer_until=None,
            defer_valid=False,
            lengths_by_rid=lengths_by_rid,
            train_cache={},
            horizon=horizon,
        )
        baseline = session._baseline_pipeline()
        session.defer_until, session.defer_valid = session._compute_defer_until(baseline)
        return session

    def _compute_defer_until(self, baseline: Pipeline) -> tuple[float | None, bool]:
        """
        The earliest future finish time anywhere in the fleet, capped at the first publish.

        `None` with `False` when nothing is resident anywhere, in which case deferral would never
        resolve and is not offered.
        """
        finishes = self.fleet.finish_times()
        future = [t for t in finishes.values() if math.isfinite(t) and t > self.now]
        if not future:
            return None, False
        candidate = min(future)
        if baseline.publish_times:
            candidate = min(candidate, baseline.publish_times[0])
        return candidate, True

    # --- lookups ----------------------------------------------------------

    def deadline(self, group_id: int) -> int:
        """Return the reserved group's deadline buffer, from the snapshot this session was built on."""
        return self.snapshot.deadline(group_id)

    # --- training-cycle pricing, cached per sequence-length hint -----------

    def _price_train(self, eta: Mapping[int, float]) -> TrainState:
        hint = sequence_hint(eta, self.members, self.lengths_by_rid, self.runtime.batch_size, self.runtime.group_size)
        cached = self.train_cache.get(hint)
        if cached is not None:
            return cached
        train = self.runtime.train_state(hint)
        self.train_cache[hint] = train
        return train

    def _baseline_pipeline(self) -> Pipeline:
        """The pipeline for the fleet and members as they stand, with no new hypothetical group."""
        eta = group_etas_from_fleet(self.fleet, self.members, self.runtime)
        train = self._price_train(eta)
        return chain_buffers(self.fleet, self.snapshot, train, eta, self.members, self.runtime, self.horizon)

    # --- member placement ---------------------------------------------------

    def _member_arrival(
        self,
        req: Request,
        option: Option,
        src: int | None,
        base: Mapping[int, InstanceForecast],
    ):
        """
        Resolve `(target instance, arrival)` for placing `req` under `option`, `None` if infeasible.

        An instance option arrives at `now`. DEFER (`option is None`) arrives at `defer_until`, on
        `src` when the tentative fold placed `req` somewhere, otherwise re-selected from `base` since
        there is no tentative instance to keep.
        """
        remaining_of = self.length_predictor.remaining
        if option is not None:
            return option, arrival_of(req, self.now, remaining_of)
        if not self.defer_valid:
            return None
        if src is not None:
            return src, dataclasses.replace(arrival_of(req, self.now, remaining_of), arrive_time=self.defer_until)
        _, chosen, _ = fold_member(FleetForecast(dict(base)), req, remaining_of, now=self.defer_until)
        if chosen is None:
            return None
        return chosen, dataclasses.replace(arrival_of(req, self.now, remaining_of), arrive_time=self.defer_until)

    def _price_with(self, base: Mapping[int, InstanceForecast], resolved) -> Pipeline | None:
        if resolved is None:
            return None
        target, arrival = resolved
        touched = {**base, target: base[target].with_arrival(arrival)}
        if touched[target].finish_times().kv_overflow:
            return None
        fleet_eval = FleetForecast(touched)
        eta = group_etas_from_fleet(fleet_eval, self.members, self.runtime)
        train = self._price_train(eta)
        return chain_buffers(fleet_eval, self.snapshot, train, eta, self.members, self.runtime, self.horizon)

    def evaluate_member(self, req: Request) -> dict[Option, Pipeline | None]:
        """
        Price every placement option for one queued member of an admitted group.

        Args:
            req (Request): A member in this session's tentative tail.

        Returns:
            dict[Option, Pipeline | None]: One pipeline per instance id, plus `None` for DEFER.
                A value of `None` means the option overflows KV or (DEFER only) is not offered.
        """
        src = self.tail_assignment.get(req.rid)
        base = dict(self.fleet.instances)
        if src is not None:
            base[src] = _without_arrival(base[src], req.rid)
        out: dict[Option, Pipeline | None] = {}
        for instance_id in self.fleet.instances:
            resolved = (instance_id, arrival_of(req, self.now, self.length_predictor.remaining))
            out[instance_id] = self._price_with(base, resolved)
        out[None] = self._price_with(base, self._member_arrival(req, None, src, base))
        return out

    def _commit_member(self, req: Request, option: Option) -> None:
        src = self.tail_assignment.pop(req.rid, None)
        base = dict(self.fleet.instances)
        if src is not None:
            base[src] = _without_arrival(base[src], req.rid)
        resolved = self._member_arrival(req, option, src, base)
        if resolved is None:
            raise ValueError(f"Cannot commit request {req.rid!r} to option {option!r}, which is infeasible.")
        target, arrival = resolved
        base[target] = base[target].with_arrival(arrival)
        self.fleet = FleetForecast(base)
        self.tail_order = [r for r in self.tail_order if r.rid != req.rid]

    # --- group admission ---------------------------------------------------

    def _rest_sorted(self, rest: Sequence[Request]) -> list[Request]:
        remaining_of = self.length_predictor.remaining
        return sorted(rest, key=lambda r: -remaining_of(traj_view(r)))

    def _price_admission(self, f: Request, rest_sorted: Sequence[Request], instance_id: int) -> Pipeline | None:
        remaining_of = self.length_predictor.remaining
        fleet_eval = self.fleet.with_placement(instance_id, arrival_of(f, self.now, remaining_of))
        first = fleet_eval.instances[instance_id].finish_times()
        if first.kv_overflow:
            return None
        group_eta = first.finish_times[f.rid]
        for member in rest_sorted:
            fleet_eval, chosen, finish = fold_member(fleet_eval, member, remaining_of, now=self.now)
            if chosen is None:
                return None
            group_eta = max(group_eta, finish)
        eta = group_etas_from_fleet(fleet_eval, self.members, self.runtime)
        train = self._price_train(eta)
        return chain_buffers(
            fleet_eval,
            self.snapshot,
            train,
            eta,
            self.members,
            self.runtime,
            self.horizon,
            new_groups=[(self.rollout_version, group_eta)],
        )

    def evaluate_admission(self, group_requests: Sequence[Request]) -> dict[Option, Pipeline | None]:
        """
        Price admitting a staged group on each instance, plus DEFER_GROUP.

        Args:
            group_requests (Sequence[Request]): The staged group's members, first member first.

        Returns:
            dict[Option, Pipeline | None]: One pipeline per instance id the first member could go
                on, plus `None` for DEFER_GROUP's key when deferral is invalid, which it never is
                for a staged group since admitting nothing is always a well-defined baseline.
        """
        f, *rest = group_requests
        rest_sorted = self._rest_sorted(rest)
        out: dict[Option, Pipeline | None] = {
            instance_id: self._price_admission(f, rest_sorted, instance_id) for instance_id in self.fleet.instances
        }
        out[None] = self._baseline_pipeline()
        return out

    def _commit_admission(self, group: list[Request], option: Option) -> None:
        f, *rest = group
        if option is None:
            return
        remaining_of = self.length_predictor.remaining
        rest_sorted = self._rest_sorted(rest)
        fleet_eval = self.fleet.with_placement(option, arrival_of(f, self.now, remaining_of))
        if fleet_eval.instances[option].finish_times().kv_overflow:
            raise ValueError(f"Cannot admit group {f.group_id!r} on instance {option!r}, which is infeasible.")
        tail_updates: dict[int, int] = {}
        for member in rest_sorted:
            fleet_eval, chosen, _ = fold_member(fleet_eval, member, remaining_of, now=self.now)
            if chosen is not None:
                tail_updates[member.rid] = chosen
        self.fleet = fleet_eval
        self.members = {**self.members, f.group_id: [r.rid for r in group]}
        self.staged_groups.pop(f.group_id, None)
        for member in rest_sorted:
            if member.rid in tail_updates:
                self.tail_assignment[member.rid] = tail_updates[member.rid]
                self.tail_order.append(member)

    # --- commit -------------------------------------------------------------

    def commit(self, req: Request, option: Option) -> None:
        """
        Fold a decision into the session's committed state.

        `req` either names a member of this session's tentative tail, or the first member of a
        staged group not yet admitted this session, which `evaluate_admission` was called on.

        Args:
            req (Request): The request the strategy decided about.
            option (Option): The chosen instance id, or `None` for DEFER / DEFER_GROUP.

        Raises:
            ValueError: If `option` is infeasible, or `req` names neither a tail member nor a staged
                group's first member.
        """
        if req.group_id in self.staged_groups and req.group_id not in self.members:
            group = self.staged_groups[req.group_id]
            if req.rid != group[0].rid:
                raise ValueError(
                    f"Admission commit must be given the staged group's first member, got rid "
                    f"{req.rid!r} for group {req.group_id!r}."
                )
            self._commit_admission(group, option)
            return
        if req.group_id not in self.members:
            raise ValueError(
                f"commit got request {req.rid!r} whose group {req.group_id!r} is neither staged nor admitted."
            )
        self._commit_member(req, option)
