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
        kv_watermark: float = 0.0,
        lookahead_defer: bool = False,
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
        self.kv_watermark = kv_watermark
        self.lookahead_defer = lookahead_defer

    # --- construction ---------------------------------------------------

    @classmethod
    def build(
        cls,
        runtime: SchedulingRuntime,
        length_predictor: LengthPredictor,
        horizon_buffers: int = 2,
        kv_watermark: float = 0.0,
        lookahead_defer: bool = False,
    ) -> PlanSession:
        """
        Snapshot `runtime` and fold its queued admitted-group members into a tentative tail.

        Args:
            runtime (SchedulingRuntime): The live, read-only simulator facade. Its policy must be psrl.
            length_predictor (LengthPredictor): Predicted remaining length for every request.
            horizon_buffers (int): How many buffers `evaluate_*` projects, clamped to the snapshot.
            kv_watermark (float): Fraction of an instance's KV pool a staged group's forecast peak
                may reach before admitting it there is refused, same convention as `length_aware`'s
                knob of the same name. Zero disables the check, leaving literal `kv_overflow` as the
                only admission-feasibility bar. Applies to admission only: an already-admitted
                member already holds a buffer reservation and must still be placed somewhere.
            lookahead_defer (bool): When True, price DEFER_GROUP as 'admit this group at
                defer_until on the then-reduced fleet' rather than as the static baseline. This
                gives DEFER a more realistic value in high-concurrency regimes: the fleet will be
                less loaded by defer_until, so the group will complete faster and the nearest
                training buffer will be ready sooner than the static baseline shows.
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
            kv_watermark=kv_watermark,
            lookahead_defer=lookahead_defer,
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

    def _above_watermark(self, fc: InstanceForecast, result) -> bool:
        if self.kv_watermark <= 0.0:
            return False
        return result.peak_blocks > self.kv_watermark * fc.state.kv_blocks

    def throughput_gain(self, instance_id: int, first: Request) -> float:
        """
        Return the fractional decode-throughput gain of adding `first` to an instance's batch.

        Throughput is `n_d / decode_segment_time(n_d, ctxsum, 1)` over the instance's running
        batch plus the arrivals this burst already placed there. Adding a request to a batch that
        is already past the decode knee barely raises tokens per second yet slows every resident.

        Args:
            instance_id (int): The instance to price.
            first (Request): The group's first member, the request being admitted.

        Returns:
            float: `(after - before) / before`, `inf` when the batch is empty.
        """
        forecast = self.fleet.instances[instance_id]
        state = forecast.state
        residents = [*(r.ctx for r in state.running), *(a.ctx for a in forecast.arrivals)]
        n_d, ctxsum = len(residents), float(sum(residents))
        if n_d == 0:
            return math.inf
        cost = state.cost
        added = arrival_of(first, self.now, self.length_predictor.remaining).ctx
        before = n_d / cost.decode_segment_time(n_d, ctxsum, 1)
        after = (n_d + 1) / cost.decode_segment_time(n_d + 1, ctxsum + added, 1)
        return (after - before) / before

    def _fleet_at_defer(self) -> FleetForecast | None:
        """Return a fleet with all arrivals predicted to finish before `defer_until` removed.

        When the strategy defers admission of a staged group, some admitted members will finish
        before the next dispatch. Removing those arrivals gives a lower-concurrency fleet that
        reflects what the instance will actually look like when the deferred group is reconsidered.
        Returns `None` when `defer_valid` is False (no meaningful defer moment to project to).
        """
        if not self.defer_valid or self.defer_until is None:
            return None
        finishes = self.fleet.finish_times()
        done_before_defer = {rid for rid, t in finishes.items() if math.isfinite(t) and t <= self.defer_until}
        if not done_before_defer:
            return self.fleet
        reduced: dict[int, InstanceForecast] = {}
        for instance_id, fc in self.fleet.instances.items():
            reduced_fc = fc
            for rid in done_before_defer:
                if any(a.key == rid for a in reduced_fc.arrivals):
                    reduced_fc = _without_arrival(reduced_fc, rid)
            reduced[instance_id] = reduced_fc
        return FleetForecast(reduced)

    def _price_admission_on_fleet(
        self,
        f: Request,
        rest_sorted: Sequence[Request],
        instance_id: int,
        fleet: FleetForecast,
        arrive_time: float,
    ) -> Pipeline | None:
        """Price admitting `group` to `instance_id` on an arbitrary `fleet` at `arrive_time`."""
        remaining_of = self.length_predictor.remaining
        arrival = dataclasses.replace(arrival_of(f, self.now, remaining_of), arrive_time=arrive_time)
        fleet_eval = fleet.with_placement(instance_id, arrival)
        first = fleet_eval.instances[instance_id].finish_times()
        if first.kv_overflow or self._above_watermark(fleet_eval.instances[instance_id], first):
            return None
        group_eta = first.finish_times[f.rid]
        for member in rest_sorted:
            fleet_eval, chosen, finish = fold_member(fleet_eval, member, remaining_of, now=arrive_time)
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

    def _lookahead_defer_pipeline(self, f: Request, rest_sorted: Sequence[Request]) -> Pipeline | None:
        """Price the DEFER option as 'admit this group at defer_until on the then-reduced fleet'.

        Instead of pricing DEFER as 'do nothing now' (`_baseline_pipeline`), this finds the best
        instance on the fleet after in-flight members expected to finish before `defer_until` have
        been removed, priced at arrival time `defer_until`. This gives DEFER a more realistic value
        in high-concurrency regimes: the fleet will be less loaded by then, so the group will
        complete faster and the training buffer will be ready sooner than the static baseline shows.
        Returns `None` when no lookahead fleet or no feasible instance is found. Callers should
        then fall back to `_baseline_pipeline`.
        """
        fleet_after = self._fleet_at_defer()
        if fleet_after is None or self.defer_until is None:
            return None
        best: Pipeline | None = None
        for instance_id in fleet_after.instances:
            candidate = self._price_admission_on_fleet(f, rest_sorted, instance_id, fleet_after, self.defer_until)
            if candidate is None:
                continue
            if best is None or candidate.train_end[0] < best.train_end[0]:
                best = candidate
        return best

    def _price_admission(self, f: Request, rest_sorted: Sequence[Request], instance_id: int) -> Pipeline | None:
        remaining_of = self.length_predictor.remaining
        fleet_eval = self.fleet.with_placement(instance_id, arrival_of(f, self.now, remaining_of))
        first = fleet_eval.instances[instance_id].finish_times()
        if first.kv_overflow or self._above_watermark(fleet_eval.instances[instance_id], first):
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
        if self.lookahead_defer:
            out[None] = self._lookahead_defer_pipeline(f, rest_sorted) or self._baseline_pipeline()
        else:
            out[None] = self._baseline_pipeline()
        return out

    def _commit_admission(self, group: list[Request], option: Option) -> None:
        f, *rest = group
        if option is None:
            return
        remaining_of = self.length_predictor.remaining
        rest_sorted = self._rest_sorted(rest)
        fleet_eval = self.fleet.with_placement(option, arrival_of(f, self.now, remaining_of))
        first = fleet_eval.instances[option].finish_times()
        if first.kv_overflow or self._above_watermark(fleet_eval.instances[option], first):
            raise ValueError(f"Cannot admit group {f.group_id!r} on instance {option!r}, which is infeasible.")
        tail_updates: dict[int, int] = {}
        for member in rest_sorted:
            fleet_eval, chosen, _ = fold_member(fleet_eval, member, remaining_of, now=self.now)
            if chosen is not None:
                tail_updates[member.rid] = chosen
        self.fleet = fleet_eval
        self.members = {**self.members, f.group_id: [r.rid for r in group]}
        self.staged_groups.pop(f.group_id, None)
        try:
            db = self.snapshot.deadline(f.group_id)
        except KeyError:
            db = self.rollout_version + self.runtime.horizon
        for member in group:
            member.deadline_buffer = db
        for member in rest_sorted:
            if member.rid in tail_updates:
                self.tail_assignment[member.rid] = tail_updates[member.rid]
                self.tail_order.append(member)

    # --- burst planning -----------------------------------------------------

    def strip(self, requests: Sequence[Request]) -> FleetForecast:
        """Return `self.fleet` with every request in `requests` removed from its tentative slot.

        A request with no tentative slot (not in `self.tail_assignment`) is left alone.
        """
        base = dict(self.fleet.instances)
        for req in requests:
            src = self.tail_assignment.get(req.rid)
            if src is not None:
                base[src] = _without_arrival(base[src], req.rid)
        return FleetForecast(base)

    def price_assignment(
        self,
        requests: Sequence[Request],
        assignment: Mapping[int, int],
        fleet: FleetForecast | None = None,
    ) -> Pipeline | None:
        """
        Price folding every request in `requests` onto the instance `assignment` names for it, all
        at once -- a whole hypothetical fleet state, not one option at a time the way
        `evaluate_member` prices options for a single request. Never mutates `self.fleet`.

        Args:
            requests (Sequence[Request]): Every request this assignment places. Each must be a key
                of `assignment`.
            assignment (Mapping[int, int]): `rid -> instance_id` for every request in `requests`.
            fleet (FleetForecast | None): The fleet to fold onto, defaulting to
                `self.strip(requests)` so a request's own prior tentative placement never counts
                against the candidate.

        Returns:
            Pipeline | None: The resulting pipeline, or `None` if any instance the assignment uses
                overflows its KV pool.
        """
        remaining_of = self.length_predictor.remaining
        base = dict((fleet or self.strip(requests)).instances)
        touched: set[int] = set()
        for req in requests:
            target = assignment[req.rid]
            base[target] = base[target].with_arrival(arrival_of(req, self.now, remaining_of))
            touched.add(target)
        for instance_id in touched:
            if base[instance_id].finish_times().kv_overflow:
                return None
        fleet_eval = FleetForecast(base)
        eta = group_etas_from_fleet(fleet_eval, self.members, self.runtime)
        train = self._price_train(eta)
        return chain_buffers(fleet_eval, self.snapshot, train, eta, self.members, self.runtime, self.horizon)

    def commit_assignment(self, requests: Sequence[Request], assignment: Mapping[int, int]) -> None:
        """
        Commit a priced assignment: fold every request into `self.fleet` and drop it from the tail.

        Args:
            requests (Sequence[Request]): Every request this assignment places.
            assignment (Mapping[int, int]): `rid -> instance_id` for every request in `requests`.

        Raises:
            ValueError: If any request's assigned instance overflows KV -- callers should price
                with `price_assignment` first and only commit a candidate that returned a `Pipeline`.
        """
        remaining_of = self.length_predictor.remaining
        base = dict(self.strip(requests).instances)
        for req in requests:
            target = assignment[req.rid]
            base[target] = base[target].with_arrival(arrival_of(req, self.now, remaining_of))
        for instance_id in {assignment[req.rid] for req in requests}:
            if base[instance_id].finish_times().kv_overflow:
                raise ValueError(f"Cannot commit assignment {assignment!r}: instance {instance_id!r} overflows KV.")
        self.fleet = FleetForecast(base)
        committed = {req.rid for req in requests}
        for rid in committed:
            self.tail_assignment.pop(rid, None)
        self.tail_order = [r for r in self.tail_order if r.rid not in committed]

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
