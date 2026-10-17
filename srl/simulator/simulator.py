"""The simulation driver: rollout generation interleaved with training.

This module provides the Simulator class that orchestrates the full RL
training loop using the event-driven architecture. It mirrors the sim_staleness
driver but with the multi-instance event coordinator infrastructure.
"""

from __future__ import annotations

import heapq
import math

from tqdm import tqdm

from ..component.recompute.cost import FixedRecomputeCost, RecomputeCostModel
from ..component.rollout.engine import EngineEvent, EventKind, NativeInstance, ReqStatus, Request
from ..component.rollout.engine.eviction import build_eviction
from ..component.rollout.router import AdmissionGate, Router, SortKey, build_strategy
from ..component.sync.cost import FixedSyncCost, SyncCostModel
from ..cost.rollout import RolloutCostModel
from ..cost.training import (
    FixedTrainingLatency,
    FixedTrainingMemory,
    TrainingCostEstimate,
    TrainingCostModel,
    TrainingParallelism,
    TrainingWorkload,
)
from ..policy.base import CompleteAction, SimState, StalenessPolicy
from ..policy.ledger import GroupLedger
from ..telemetry.store import JsonlStore
from ..telemetry.writer import SimTelemetry
from .config import SimConfig, SimResult, clip_total_length
from .runtime import SchedulingRuntime


class Simulator:
    """
    Single or multi-instance simulation driver.

    Runs the full RL training loop: admit -> generate -> complete ->
    select batch -> (recompute) -> train -> weight update -> abort/re-prefill.
    """

    def __init__(
        self,
        rollout_cost: RolloutCostModel,
        policy: StalenessPolicy,
        cfg: SimConfig,
        training_cost: TrainingCostModel | None = None,
        sync_cost: SyncCostModel | None = None,
        recompute_cost: RecomputeCostModel | None = None,
        store: JsonlStore | None = None,
        backend: str = "native",
    ):
        self.rollout_cost = rollout_cost
        self.policy = policy
        self.workload = cfg.workload
        self.cfg = cfg
        # Upper-cased to match `EventKind.name`, so the per-event test is one set lookup.
        self._undoable_kinds = frozenset(name.upper() for name in cfg.undoable_events)
        self._iteration_speculative = "ITERATION_DONE" in self._undoable_kinds
        # instance id -> whether `_batches_iterations` allows batching
        self._iteration_batching: dict[int, bool] = {}
        self._backend = backend
        self.training_cost = training_cost or TrainingCostModel(
            FixedTrainingLatency(1.0),
            FixedTrainingMemory(0),
        )
        self.training_parallelism = TrainingParallelism(
            tp=cfg.training_tp,
            pp=cfg.training_pp,
            dp=cfg.training_dp,
            cp=cfg.training_cp,
        )
        self.sync_cost = sync_cost or FixedSyncCost(cfg.sync_time)
        # No recompute model means no log-prob pass is charged. A real PPO/GRPO
        # loop always has one, so an aligned run must pass one explicitly.
        self.recompute_cost = recompute_cost or FixedRecomputeCost(0.0)

        if self._backend == "auto":
            try:
                from srl._engine_cpp import CppInstance  # noqa: F401

                self._backend = "cpp"
            except ImportError:
                self._backend = "native"

        if self._backend == "cpp":
            self._Request = _cpp_request_class()
            cpp_cost = _make_cpp_cost(rollout_cost)
            self.instances: list[NativeInstance] = []
            for i in range(cfg.n_instances):
                self.instances.append(
                    _make_cpp_instance(
                        cpp_cost, cfg, instance_id=i,
                    )
                )
        else:
            self._Request = Request
            self.instances: list[NativeInstance] = []
            for i in range(cfg.n_instances):
                self.instances.append(
                    NativeInstance(
                        rollout_cost,
                        token_budget=cfg.token_budget,
                        kv_blocks=cfg.kv_blocks,
                        block_size=cfg.block_size,
                        max_running=cfg.max_concurrent,
                        max_concurrency=cfg.max_concurrency,
                        instance_id=i,
                        reject_if_kv_full=cfg.reject_if_kv_full,
                        reject_if_waiting=cfg.reject_if_waiting,
                        reject_if_running_full=cfg.reject_if_running_full,
                        enable_prefix_caching=cfg.enable_prefix_caching,
                        eviction=build_eviction(cfg.eviction_strategy),
                    )
                )

        self.router = Router(
            strategy=build_strategy(cfg.routing_strategy, **cfg.routing_strategy_args),
            instances=self.instances,
            gate=AdmissionGate(
                max_concurrent_seqs_per_instance=cfg.max_concurrent_seqs_per_instance,
                reject_on_waiting=cfg.router_reject_on_waiting,
                reject_preempted_on_waiting=cfg.router_reject_preempted_on_waiting,
            ),
            sort_key=SortKey(cfg.router_sort_key),
            queue_window=cfg.router_queue_window,
            enable_group_sticky=cfg.enable_group_sticky,
            max_waiting_after_preempt=cfg.max_waiting_after_preempt,
            # The Python model even on the C++ backend, which holds a translated
            # copy and exposes no handle to it. A strategy pricing a hypothetical
            # batch must reach the same coefficients the engine runs on.
            cost_model=rollout_cost,
        )

        bind_runtime = getattr(self.router.strategy, "bind_runtime", None)
        if callable(bind_runtime):
            bind_runtime(SchedulingRuntime(self))

        self.t: float = 0.0
        self.train_version: int = 0
        self.rollout_version: int = 0
        self.next_rid: int = 0
        self.next_group_id: int = 0
        self.inflight: list[Request] = []
        self._staged_groups: dict[int, list[Request]] = {}
        self._admitted_group_ids: set[int] = set()
        # A flow limit on dataset ingress, not a staleness bound. See `_stage`.
        self._ingress_group_credits = (cfg.max_staleness + 1) * cfg.batch_size
        self.pending: dict[int, list[Request]] = {}
        """Finished responses whose group is not complete yet, by group id.

        A group is the unit of consumption, so a response waits here rather than
        entering `ready` where a policy could select it without its siblings.
        """
        self.ready: list[Request] = []
        self.ready_groups: list[list[Request]] = []
        self.consumed: list[Request] = []
        self.dropped: list[Request] = []
        self.n_admitted: int = 0
        self.n_consumed: int = 0
        self.train_timestamps: list[tuple[float, float]] = []
        self.train_peak_memory_bytes: list[int] = []
        self._ledger = GroupLedger()
        self._training_end: float | None = None
        """Absolute time the in-progress training step publishes, `None` while idle.

        Set in `_train` before `_run_engine(deadline=...)`, cleared after `_publish_weights`. Read by
        `SchedulingRuntime.train_state` for a strategy forecasting the training pipeline.
        """

        # `cfg.log_telemetry=False` disables logging even if a `store` was
        # passed in; see `SimTelemetry` for the stream schemas and the
        # commit-only bookkeeping (all the logging logic lives there, not
        # here, so this class only ever calls high-level telemetry methods).
        self.telemetry = SimTelemetry(store, enabled=cfg.log_telemetry)

    def _state(self) -> SimState:
        return SimState(
            now=self.t,
            train_version=self.train_version,
            rollout_version=self.rollout_version,
            batch_size=self.cfg.batch_size,
            group_size=self.cfg.group_size,
            max_staleness=self.cfg.max_staleness,
            inflight=self.inflight,
            ready=self.ready,
            ready_groups=self.ready_groups,
            n_admitted=self.n_admitted,
            n_consumed=self.n_consumed,
            n_instances=self.cfg.n_instances,
            groups=self._ledger.views(),
        )

    def _next_request_shape(self) -> tuple[int, int, int]:
        """
        Return `(group_id, prompt_len, target_len)` for the next request.

        Both lengths come from the same workload group, so the prompt/response
        pairing is the one that was observed rather than a product of two
        independent modulo lookups.

        The pair is capped on its **total**, not per dimension: the serving
        stack budgets `max_prompt_length + max_output_length` of context, so a
        short prompt leaves the remainder available for generation. See
        `clip_total_length`.

        Returns:
            tuple[int, int, int]: The group id and the system-capped lengths.
        """
        group_id, prompt_len, output_len = self.workload.request(self.next_rid)
        prompt, target = clip_total_length(
            prompt_len,
            output_len,
            self.cfg.max_prompt_length,
            self.cfg.max_output_length,
        )
        return group_id, prompt, target

    def _training_sequence_length(self, req: Request) -> int:
        """Return the length the trainer sees, which is not the length generated.

        Rollout and training cap differently:

        - Generation gets a *shared* budget, `response_length + prompt_length` less the
          prompt actually consumed, so a short prompt buys extra output room and the
          engine can emit well past the nominal response cap.
        - Training gets *per-dimension* caps: the sample handed to the trainer holds at
          most `max_output_length` response tokens whatever the prompt length was.

        So the trained length is `min(prompt, max_prompt) + min(response, max_output)`,
        and the tokens past the cap are charged to rollout but not to training.

        Returns:
            The context length the trainer would see, floored at one token.
        """
        if self.cfg.max_output_length is None:
            return req.ctx
        prompt = min(req.prompt_len, self.cfg.max_prompt_length or req.prompt_len)
        return max(1, prompt + min(req.generated, self.cfg.max_output_length))

    def _dispatch_router(self) -> int:
        """Place whatever the router can place right now.

        Called only after a committed capacity release: a finished request, a
        preemption, or a weight update. A dispatch at any other moment is a
        provable no-op, because `add_request`'s binding gate is a block count
        and nothing else in the attempt varies with time. See
        `component/rollout/router/router.py`.

        Returns:
            How many requests were placed.
        """
        placed = self.router.dispatch(
            self.t,
            can_admit=self._can_place_from_router,
            on_placed=self._commit_router_placement,
        )
        if not placed:
            return 0
        self.telemetry.emit_instance_all(self.instances)
        self.telemetry.emit_gateway(self.t, self.router, self.instances)
        return len(placed)

    def _can_place_from_router(self, req: Request) -> bool:
        """Whether placement may atomically admit `req`'s staged group."""
        if req.group_id in self._admitted_group_ids:
            return True
        return self.policy.admit_quota(self._state()) >= self.cfg.group_size

    def _commit_router_placement(self, req: Request, instance_id: int) -> None:
        """Commit first-placement admission, then report this placement."""
        if req.group_id not in self._admitted_group_ids:
            group = self._staged_groups.pop(req.group_id)
            for member in group:
                member.v_traj = self.rollout_version
                member.admit_time = self.t
            self.router.rekey(group)
            self._admitted_group_ids.add(req.group_id)
            self.inflight.extend(group)
            self.n_admitted += len(group)
            self.policy.on_admit(group, self._state())
            self._ledger.begin(group, self.rollout_version, self.t)
            for member in group:
                self.telemetry.write_request("admit", self.t, member)
        self.policy.on_route(req, instance_id, self._state())

    def _stage(self, dispatch: bool = True) -> int:
        """
        Offer credited dataset groups to the router without admitting them.

        Staging only decides how much of the dataset the router can see. The staleness
        policy gates admission one layer later, in `_can_place_from_router`, and `v_traj` is
        fixed only when a placement commits. The credit must stay independent of the policy:
        the number of staged groups is the candidate set `sort_key` orders, so a
        policy-dependent width would change which request gets placed.

        Returns:
            int: Responses staged by this call. Zero means no credit was left, which `run`
                reads as one of its livelock signals.
        """
        staged = 0
        while self._ingress_group_credits > 0:
            staged += len(self._stage_one_group())
            self._ingress_group_credits -= 1
        if staged:
            self.telemetry.emit_gateway(self.t, self.router, self.instances)
            if dispatch:
                self._dispatch_router()
        return staged

    def _stage_one_group(self) -> list[Request]:
        """Create one prompt group and expose it to placement admission."""
        group: list[Request] = []
        group_id = self.next_group_id
        for index in range(self.cfg.group_size):
            source_group_id, prompt_len, target_len = self._next_request_shape()
            req = self._Request(
                rid=self.next_rid,
                group_id=group_id,
                group_index=index,
                source_group_id=source_group_id,
                prompt_len=prompt_len,
                target_len=target_len,
                v_traj=self.rollout_version,
            )
            self.next_rid += 1
            group.append(req)

        self.next_group_id += 1
        self._staged_groups[group_id] = group
        self.router.enqueue(group, self.t)
        for req in group:
            self.telemetry.write_request("enqueue", self.t, req)
        return group

    def run(self) -> SimResult:
        idle_rounds = 0
        pbar = tqdm(
            total=self.cfg.n_versions,
            desc="simulate versions",
            unit="ver",
            dynamic_ncols=True,
        )
        try:
            while self.train_version < self.cfg.n_versions:
                self._stage()

                batch = self.policy.take_batch(self._state())
                if batch is None:
                    self.policy.on_batch_unavailable(self._state())
                    if not self.inflight and self._stage() == 0:
                        has_waiting = (
                            any(inst.waiting for inst in self.instances)
                            or bool(self.router)
                        )
                        if not has_waiting:
                            idle_rounds += 1
                            if idle_rounds > self.cfg.livelock_rounds:
                                return self._result(livelocked=True)
                            continue
                    before = (self.t, len(self.inflight), len(self.ready), len(self.dropped))
                    self._run_engine(stop_on_batch=True)
                    after = (self.t, len(self.inflight), len(self.ready), len(self.dropped))
                    idle_rounds = 0 if after != before else idle_rounds + 1
                    if idle_rounds > self.cfg.livelock_rounds:
                        return self._result(livelocked=True)
                    continue

                idle_rounds = 0
                self._train(batch)
                pbar.set_postfix(t=f"{self.t:.1f}s", refresh=False)
                pbar.update(1)

            return self._result()
        finally:
            pbar.close()

    def _collect(self, done: list[Request]) -> None:
        """Retire finished responses, releasing a group once it is whole."""
        complete: list[list[Request]] = []
        for req in done:
            if req in self.inflight:
                self.inflight.remove(req)
            # complete_time is already set by _retire via t_local. Do not overwrite.
            self.telemetry.write_request("complete", self.t, req)
            self._observe_completion(req)
            self._ledger.complete(req)
            siblings = self.pending.setdefault(req.group_id, [])
            siblings.append(req)
            if len(siblings) < self.cfg.group_size:
                continue
            complete.append(sorted(self.pending.pop(req.group_id), key=lambda r: r.group_index))
        if not complete:
            return
        # One snapshot after every completion, so each policy call sees this batch.
        st = self._state()
        for group in complete:
            if self.policy.on_group_complete(group, st) is CompleteAction.DROP:
                for member in group:
                    member.status = ReqStatus.DROPPED
                    self.dropped.append(member)
                    self.telemetry.write_request("drop", self.t, member)
                self._ledger.drop(group[0].group_id)
                self._grant_drop_replacements([group])
            else:
                self.ready_groups.append(group)
                self.ready.extend(group)

    def _observe_completion(self, req: Request) -> None:
        """Offer a finished trajectory to a strategy that implements `observe_completion`.

        A capability probe rather than a protocol method, so a strategy that does
        not learn from completions is never asked.
        """
        observe = getattr(self.router.strategy, "observe_completion", None)
        if callable(observe):
            observe(req)

    def _observe_train_cycle(self) -> None:
        """Tell a strategy that implements `observe_train_cycle` that a weight version was published.

        Called from `_publish_weights`, which is where the staleness clock ticks.
        """
        observe = getattr(self.router.strategy, "observe_train_cycle", None)
        if callable(observe):
            observe(self.train_version, self.t)

    def _discard(self, reqs: list[Request]) -> None:
        """Drop requests, widening to whole prompt groups.

        A group that loses one response can never be consumed, so its surviving
        siblings would hold KV for the rest of the run. Dropping the group keeps
        the invariant that everything in flight is still trainable.
        """
        seen: set[int] = set()
        unique_reqs: list[Request] = []
        for req in self._expand_to_groups(reqs):
            if id(req) in seen:
                continue
            seen.add(id(req))
            unique_reqs.append(req)

        dropped_groups = {req.group_id for req in unique_reqs}
        committed = [
            req for req in unique_reqs if req.group_id in self._admitted_group_ids
        ]
        self.policy.on_discard(committed, self._state())
        for group_id in dropped_groups:
            self.pending.pop(group_id, None)
            self._staged_groups.pop(group_id, None)
            self._ledger.drop(group_id)
        self.ready_groups = [
            group for group in self.ready_groups if group[0].group_id not in dropped_groups
        ]

        # Take the dropped requests out of the router queue first. A queued
        # request is inflight but sits on no instance, so the per-instance
        # cancel below would skip it and a later dispatch would place work the
        # simulator has already written off.
        self.router.remove(unique_reqs)
        self.router.release_group_pins(dropped_groups)

        to_cancel: list[Request] = []
        for req in unique_reqs:
            if req in self.inflight:
                self.inflight.remove(req)
                to_cancel.append(req)
            if req in self.ready:
                self.ready.remove(req)
            req.status = ReqStatus.DROPPED
            self.dropped.append(req)
            self.telemetry.write_request("drop", self.t, req)
        if to_cancel:
            for inst in self.instances:
                inst.cancel(
                    [r for r in to_cancel if r.instance_id == inst.instance_id],
                    keep_tokens=False,
                )
        groups = [
            [req for req in unique_reqs if req.group_id == group_id]
            for group_id in dropped_groups
        ]
        self._grant_drop_replacements(groups)
        self._stage()

    def _grant_drop_replacements(self, groups: list[list[Request]]) -> None:
        """Credit policy-requested dataset replacements exactly once per group.

        This is a flow credit, not an admission decision: a policy that expires
        work permanently (ROLL) would otherwise shrink the dataset in flight for
        the rest of the run, because nothing else re-opens ingress for a group
        that was dropped rather than consumed.
        """
        if not groups:
            return
        self._ingress_group_credits += self.policy.replacement_groups_on_drop(
            groups,
            self._state(),
        )

    def _expand_to_groups(self, reqs: list[Request]) -> list[Request]:
        """Add every still-live sibling of the given requests."""
        if self.cfg.group_size == 1:
            return list(reqs)
        wanted = {req.group_id for req in reqs}
        pending = [req for gid in wanted for req in self.pending.get(gid, ())]
        staged = [req for gid in wanted for req in self._staged_groups.get(gid, ())]
        live = [
            req for req in (*self.inflight, *self.ready) if req.group_id in wanted
        ]
        return [*reqs, *pending, *staged, *live]

    def _is_cross_instance(self, event: EngineEvent) -> bool:
        """Whether this event can move work between instances.

        A preemption frees its victim's blocks, so it is one of the three
        moments the router can place something new. It only becomes a *global*
        event when an instance's queue has run deeper than
        `max_waiting_after_preempt` and work is about to leave it, which is
        vLLM's own `preemption_notification_threshold` rule. Below the
        threshold the victim is requeued locally and no other instance needs
        to be told, which is the common case and must stay cheap: there were
        2122 preemptions in the matched run against 404 aborts.
        """
        if self.router.max_waiting_after_preempt is None:
            return False
        if not event.preempted:
            return False
        return (
            len(self.instances[event.instance_id].waiting)
            > self.router.max_waiting_after_preempt
        )

    def _batches_iterations(self, instance_id: int) -> bool:
        """
        Return whether this instance's ITERATION_DONE events are batched in the engine.

        Batching reproduces every committed state and written row except the per-event
        occupancy sample, so it is enabled only where the `instance/*` stream is not written.
        """
        allowed = self._iteration_batching.get(instance_id)
        if allowed is None:
            allowed = self.telemetry.instance_stream_suppressed(instance_id)
            self._iteration_batching[instance_id] = allowed
        return allowed

    def _undo_events(self, events: dict[int, EngineEvent]) -> None:
        """Undo all pending events (one per instance) and clear the events dict.

        Instances whose last event was a committed advance (no snapshot) are
        skipped — their mutations are permanent by design. Any completions they
        produced must be collected here, because the normal collection path in
        `_run_engine` will never see them (the event is being discarded).
        """
        for iid in list(events):
            inst = self.instances[iid]
            if inst.has_pending_event:
                inst.undo_last_event()
            else:
                event = events[iid]
                if event.completed:
                    self._collect(event.completed)
        events.clear()

    def _align_instances(self, t: float, force_align_t_local: bool = False) -> None:
        """Advance every lagging instance to a common global time.

        Alignment is a committed advance, so any undo log still open is closed
        first. `advance_to` would otherwise record committed mutations into a
        log belonging to an event that has already been decided. Another
        instance may complete at exactly the alignment time; that completion
        is committed too and must be collected rather than discarded.

        Instances whose clock is already past ``t`` (possible when selective
        undo left a committed advance in place) are left alone and the global
        clock is widened to include them.
        """
        self.t = max(self.t, t)
        for inst in self.instances:
            inst.commit_last_event()
        for inst in self.instances:
            if inst.t_local > t:
                self.t = max(self.t, inst.t_local)
                continue
            extra_done, _ = inst.advance_to(t, self.rollout_version)
            if extra_done:
                self._collect(extra_done)
            if force_align_t_local:
                inst.t_local = t

    def _undo_and_align(
        self,
        events: dict[int, EngineEvent],
        queue: list[tuple[float, int]],
        t: float,
        force_align_t_local: bool = False,
    ) -> None:
        """Undo pending events, clear their heap entries, and align instance clocks."""
        self._undo_events(events)
        queue.clear()
        self._align_instances(t, force_align_t_local=force_align_t_local)

    def _seed_events(
        self,
        events: dict[int, EngineEvent],
        queue: list[tuple[float, int]],
    ) -> None:
        """Produce one pending event for every active instance."""
        assert not events and not queue, "Events must be empty before seeding."
        for inst in self.instances:
            # `n_active` because reading `running`/`waiting` converts a whole vector on C++.
            if inst.n_active == 0:
                continue
            event = inst.advance_one_event(self.rollout_version)
            if event is not None:
                events[inst.instance_id] = event
                heapq.heappush(queue, (event.t, inst.instance_id))

    def _run_engine(
        self,
        deadline: float | None = None,
        stop_on_batch: bool = False,
    ) -> None:
        """
        Unified engine loop driving all instances in global time order.

        Processes events from a priority queue, always advancing the instance
        with the earliest next event. Stops when either condition is met:
        - deadline is reached (simulation time exceeds it), or
        - stop_on_batch is True and the policy reports a complete batch.

        When a stop condition triggers, all pending events are undone, then all
        instances are advanced to the stop time via `advance_to` to ensure a
        consistent final state.

        Args:
            deadline (float | None): Stop when event time exceeds this value.
                None means no time limit (use stop_on_batch instead).
            stop_on_batch (bool): Stop as soon as a training batch is ready.
        """
        # events maps instance_id -> the pending (not yet committed) event.
        events: dict[int, EngineEvent] = {}
        queue: list[tuple[float, int]] = []  # (event_time, instance_id)

        # Place whatever is waiting before seeding. Entry to the engine loop is
        # a committed point, and without this an instance that went idle while
        # the router still held work would seed no event at all and the run
        # would spin to a false livelock.
        self._dispatch_router()

        # Telemetry reflects committed state only. Log it here, before any
        # speculative advance_one_event call below has a chance to mutate
        # step_stats / occupancy for an event that may still be undone.
        self.telemetry.sync_all(self.instances)

        # Seed: each active instance produces its first event.
        self._seed_events(events, queue)

        while queue:
            t_event, iid = heapq.heappop(queue)
            event = events.pop(iid)
            inst = self.instances[iid]

            # --- Deadline exceeded: undo this event and all remaining ---
            if deadline is not None and t_event > deadline:
                events[iid] = event
                self._undo_and_align(events, queue, deadline, force_align_t_local=True)
                self.telemetry.sync_all(self.instances)
                break

            # --- Cross-instance event: commit on owning instance and align others ---
            cross_instance = self._is_cross_instance(event)
            if cross_instance:
                # The event is already committed on `inst` at t_event. The clock
                # has to move before the others are aligned to it, otherwise
                # `_align_instances` would assert against a stale global time.
                self.t = max(self.t, t_event)
                inst.commit_last_event()
                self._undo_and_align(events, queue, t_event)
                self.telemetry.sync_all(self.instances)
                # An instance whose queue has run past the threshold hands its
                # preempted overflow back, and the router re-places it. This is
                # the only path by which work crosses instances mid-generation.
                reclaimed = self.router.reclaim_preempted(self.t)
                if reclaimed:
                    self._dispatch_router()
                    self.telemetry.sync_all(self.instances)
            else:
                # Event is committed. Update global clock and close its undo
                # log, so a later router dispatch can tell a live speculative
                # event from a finished one.
                self.t = max(self.t, t_event)
                inst.commit_last_event()
                self.telemetry.sync(inst)

            # --- Process completions ---
            if event.kind.name == "REQUEST_COMPLETE" and event.completed:
                self._collect(event.completed)

                if stop_on_batch:
                    if self.policy.peek_batch(self._state()) is not None:
                        if not cross_instance:
                            self._undo_and_align(events, queue, t_event)
                        self.telemetry.sync_all(self.instances)
                        break

                    self.policy.on_batch_unavailable(self._state())

                # A completion frees KV, which is one of the three moments the
                # router can place work, and it can also release policy
                # capacity. Breaking out for either is required: gating on the
                # quota alone would leave a non-empty router queue untouched
                # whenever the policy is saturated, even though this very
                # completion just made room on an instance.
                if self.router.has_dispatchable() or self._ingress_group_credits > 0:
                    if not cross_instance:
                        self._undo_and_align(events, queue, t_event)
                    self.telemetry.sync_all(self.instances)
                    if not self._stage():
                        self._dispatch_router()
                    self._seed_events(events, queue)
                    continue

            if cross_instance:
                self._seed_events(events, queue)
                continue

            # --- Produce next event from this instance ---
            if inst.n_active > 0:
                # A non-undoable event's successor is produced without a snapshot.
                speculative = event.kind.name in self._undoable_kinds
                if self._batches_iterations(inst.instance_id):
                    # NOTE(claude): This loop only commits an ITERATION_DONE and asks for the
                    # next event, so the engine does that itself for every one that would be
                    # popped next anyway. The steps and their order are unchanged.
                    next_event, n_done, t_done, n_stats = inst.advance_iterations(
                        self.rollout_version,
                        speculative,
                        self._iteration_speculative,
                        queue[0] if queue else None,
                        math.inf if deadline is None else deadline,
                    )
                    if n_done:
                        self.t = max(self.t, t_done)
                        # Leave out the trailing row, which belongs to the pending event.
                        self.telemetry.sync(inst, stats_upto=n_stats)
                elif speculative:
                    next_event = inst.advance_one_event(self.rollout_version)
                else:
                    next_event = inst.advance_one_event_committed(self.rollout_version)
                if next_event is not None:
                    events[inst.instance_id] = next_event
                    heapq.heappush(queue, (next_event.t, inst.instance_id))

        # --- Finalize: advance all instances to stop_time ---
        # For deadline, align all the instances to the exact deadline time.
        # For stop_on_batch, algin all the instances to the nearest time before the global time.
        if deadline:
            self._align_instances(deadline, force_align_t_local=True)
        else:
            self._align_instances(self.t, force_align_t_local=False)
        self.telemetry.sync_all(self.instances)

    def _publish_weights(self) -> None:
        """Publish the finished training step, then interrupt and refill.

        `train_version` advances here, after recompute/train/sync actually
        finished. Rollout then loads the new weights, aborts in-flight partials,
        rebases unstarted staged groups, and opens one new dataset batch.
        """
        previous_rollout_version = self.rollout_version
        self.train_version += 1
        self._observe_train_cycle()
        self.policy.on_train_version_advance(self.train_version, self._state())
        self.rollout_version = self.policy.rollout_version_after_train(self._state())
        weights_changed = self.rollout_version != previous_rollout_version
        if weights_changed and self._staged_groups:
            staged = [req for group in self._staged_groups.values() for req in group]
            for req in staged:
                assert req.generated == 0
                req.v_traj = self.rollout_version
            self.router.rekey(staged)
        if weights_changed and self.cfg.partial_rollout and self.inflight:
            aborted: list[Request] = []
            for inst in self.instances:
                aborted.extend(inst.abort_all(requeue=False))
            self.router.enqueue(aborted, self.t)
            self.policy.on_interrupt(list(self.inflight), self._state())
            self.telemetry.emit_instance_all(self.instances)
            self.telemetry.emit_gateway(self.t, self.router, self.instances)
        if weights_changed and self.cfg.weight_load_time > 0:
            load_t = self.cfg.weight_load_time
            self.t += load_t
            for inst in self.instances:
                inst.t_local = max(inst.t_local, self.t)
            self.telemetry.emit_instance_all(self.instances)
            # No dispatch here: `_stage` below enqueues the batch this update
            # unlocks and dispatches once, so both cohorts are in the queue
            # together. SMG polls its routing loop every 500 ms and the matched
            # run shows them arriving within one tick, reaching 444 = 256 newly
            # unlocked + 188 aborted before placing any of them. Placing the
            # aborted set first empties the queue before the new batch exists,
            # which leaves the version-layered priority nothing to arbitrate.
        self._ingress_group_credits += self.cfg.batch_size
        # `_stage` dispatches only when it enqueued something, so keep the
        # fallback for an externally configured zero-credit update.
        if not self._stage():
            self._dispatch_router()

    def _train_cycle(
        self,
        sequence_lengths: tuple[int, ...],
    ) -> tuple[TrainingCostEstimate, TrainingWorkload, float, float]:
        """
        Price one training step: latency plus memory, the workload priced, sync time, and recompute time.

        Shared by `_train` and `SchedulingRuntime.cycle_time` so the two accountings cannot drift.

        Returns:
            tuple[TrainingCostEstimate, TrainingWorkload, float, float]: The latency and memory
                estimate, the `TrainingWorkload` it was priced on, the sync time, and the recompute
                time, in seconds.
        """
        training_workload = TrainingWorkload(
            sequence_lengths=sequence_lengths,
            micro_batch_size=self.cfg.training_micro_batch_size,
            max_tokens_per_microbatch=self.cfg.training_max_tokens_per_microbatch,
            schedule=self.cfg.training_schedule,
            recompute=self.cfg.activation_recompute,
            optimizer=self.cfg.training_optimizer,
        )
        training_estimate = self.training_cost.estimate(
            training_workload,
            self.training_parallelism,
        )
        weight_sync = self.policy.will_sync_after_train(self._state())
        sync_time = self.sync_cost.sync_time(0, self.cfg.n_instances, 0) if weight_sync else 0.0
        recompute_time = self.recompute_cost.recompute_time(
            training_workload,
            self.training_parallelism,
        )
        return training_estimate, training_workload, sync_time, recompute_time

    def _train(self, batch: list[Request]) -> None:
        """Consume `batch`, expire against the expected version, train, then publish.

        Committing this batch means it will become `train_version + 1`. Stale
        in-flight work is expired against that expected version immediately,
        without publishing yet. Rollout keeps the previous `rollout_version`
        until train+sync finish.
        """
        train_start = self.t
        consumed_ids = {req.group_id for req in batch}
        self.ready_groups = [
            group for group in self.ready_groups if group[0].group_id not in consumed_ids
        ]
        taken = {id(req) for req in batch}
        self.ready = [req for req in self.ready if id(req) not in taken]
        for group_id in consumed_ids:
            self._ledger.drop(group_id)
        # A consumed group is finished with its instance, so its sticky pin is
        # released. SMG drops the pin when the group's last request retires;
        # holding it would let a repeating workload be pinned by a group that
        # no longer exists.
        self.router.release_group_pins(consumed_ids)
        for req in batch:
            req.consume_version = self.train_version
            self.consumed.append(req)
            self.n_consumed += 1
            self.telemetry.write_request("consume", self.t, req)

        expected_train_version = self.train_version + 1
        expired = self.policy.expire(self._state(), expected_train_version)
        if expired:
            self._discard(expired)
            self.telemetry.emit_instance_all(self.instances)

        sequence_lengths = tuple(self._training_sequence_length(req) for req in batch)
        total_tokens = sum(sequence_lengths)
        training_estimate, training_workload, sync_time, recompute_time = self._train_cycle(sequence_lengths)
        train_time = training_estimate.latency_s
        peak_memory_bytes = training_estimate.peak_memory_bytes
        if (
            self.cfg.training_gpu_memory_bytes is not None
            and peak_memory_bytes > self.cfg.training_gpu_memory_bytes
        ):
            raise RuntimeError(
                f"Predicted training peak memory {peak_memory_bytes} exceeds configured capacity "
                f"{self.cfg.training_gpu_memory_bytes}."
            )

        total_duration = train_time + sync_time + recompute_time
        deadline = self.t + total_duration
        self._training_end = deadline
        self._run_engine(deadline=deadline)
        self._publish_weights()
        self._training_end = None

        self.train_timestamps.append((train_start, self.t))
        self.train_peak_memory_bytes.append(peak_memory_bytes)
        self.telemetry.write_train(
            t=self.t,
            start=train_start,
            end=self.t,
            version=self.train_version,
            n_groups=len(batch) // self.cfg.group_size,
            group_size=self.cfg.group_size,
            n_reqs=len(batch),
            total_tokens=total_tokens,
            train_time=train_time,
            peak_memory_bytes=peak_memory_bytes,
            training_tp=self.training_parallelism.tp,
            training_pp=self.training_parallelism.pp,
            training_dp=self.training_parallelism.dp,
            training_cp=self.training_parallelism.cp,
            micro_batch_size=training_workload.micro_batch_size,
            schedule=training_workload.schedule,
            activation_recompute=training_workload.recompute,
            sync_time=sync_time,
            recompute_time=recompute_time,
            n_inflight=len(self.inflight),
            n_ready=len(self.ready),
            n_pending=sum(len(group) for group in self.pending.values()),
            n_dropped_step=len(expired or []),
            extrapolation_notes=training_estimate.diagnostics.get("extrapolation_notes"),
        )
        self.policy.check_invariants(self._state())

    def _result(self, livelocked: bool = False) -> SimResult:
        return SimResult(
            consumed=self.consumed,
            dropped=self.dropped,
            sim_time=self.t,
            versions_done=self.train_version,
            livelocked=livelocked,
            train_timestamps=self.train_timestamps,
            train_peak_memory_bytes=self.train_peak_memory_bytes,
        )


# --- C++ backend helpers ---


def _cpp_request_class() -> type:
    """Return the CppRequest class for use as a Request factory."""
    from srl._engine_cpp import CppRequest

    return CppRequest


def _make_cpp_cost(py_cost: RolloutCostModel):
    """
    Wrap a Python cost model so the C++ engine can call it.

    Every model goes through the same trampoline into the Python object. A model that
    publishes `cpp_term_spec` is served from C++-side term memos and is called only on a miss.
    """
    from srl._engine_cpp import PyWrappedCostModel

    return PyWrappedCostModel(py_cost)


def _make_cpp_instance(cpp_cost, cfg: SimConfig, instance_id: int):
    """Build a CppInstance with the same parameters as the Python path."""
    from srl._engine_cpp import CppInstance

    inst = CppInstance(
        cost=cpp_cost,
        token_budget=cfg.token_budget,
        kv_blocks=cfg.kv_blocks,
        block_size=cfg.block_size,
        max_running=-1 if cfg.max_concurrent is None else cfg.max_concurrent,
        max_concurrency=-1 if cfg.max_concurrency is None else cfg.max_concurrency,
        instance_id=instance_id,
        reject_if_kv_full=cfg.reject_if_kv_full,
        reject_if_waiting=cfg.reject_if_waiting,
        reject_if_running_full=cfg.reject_if_running_full,
        enable_prefix_caching=cfg.enable_prefix_caching,
    )
    # The same strategy object the native path builds, so both engines consult one `select`.
    inst.set_eviction(build_eviction(cfg.eviction_strategy))
    return inst
