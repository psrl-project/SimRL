"""The simulation driver: rollout generation interleaved with training.

This module provides the Simulator class that orchestrates the full RL
training loop using the event-driven architecture. It mirrors the sim_staleness
driver but with the multi-instance event coordinator infrastructure.
"""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

from tqdm import tqdm

from ..component.recompute.cost import FixedRecomputeCost, RecomputeCostModel
from ..component.rollout.engine import EngineEvent, EventKind, NativeInstance, ReqStatus, Request
from ..component.rollout.router import AdmissionGate, Router, SortKey, build_strategy
from ..component.sync.cost import FixedSyncCost, SyncCostModel
from ..cost.rollout import RolloutCostModel
from ..cost.training import (
    FixedTrainingLatency,
    FixedTrainingMemory,
    TrainingCostModel,
    TrainingParallelism,
    TrainingWorkload,
)
from ..policy.base import CompleteAction, SimState, StalenessPolicy
from ..telemetry.store import JsonlStore
from ..telemetry.writer import SimTelemetry
from .config import SimConfig, SimResult, clip_total_length

if TYPE_CHECKING:
    from .dynamic_parallel import DynamicParallelFleet, FleetPlan, ParallelismSolver


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
        parallel_cost_models: dict[int, RolloutCostModel] | None = None,
        kv_blocks_per_tp: dict[int, int] | None = None,
        solver: "WaterFillingSolver | None" = None,
    ):
        self.rollout_cost = rollout_cost
        self.policy = policy
        self.workload = cfg.workload
        self.cfg = cfg
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
                )
            )

        self.router = Router(
            strategy=build_strategy(cfg.routing_strategy, **cfg.routing_strategy_args),
            instances=self.instances,
            gate=AdmissionGate(
                max_concurrent_seqs_per_instance=cfg.max_concurrent_seqs_per_instance,
                reject_on_waiting=cfg.router_reject_on_waiting,
            ),
            sort_key=SortKey(cfg.router_sort_key),
            enable_group_sticky=cfg.enable_group_sticky,
            max_waiting_after_preempt=cfg.max_waiting_after_preempt,
        )

        self.t: float = 0.0
        self.train_version: int = 0
        self.rollout_version: int = 0
        self.next_rid: int = 0
        self.next_group_id: int = 0
        self.inflight: list[Request] = []
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

        # `cfg.log_telemetry=False` disables logging even if a `store` was
        # passed in; see `SimTelemetry` for the stream schemas and the
        # commit-only bookkeeping (all the logging logic lives there, not
        # here, so this class only ever calls high-level telemetry methods).
        self.telemetry = SimTelemetry(store, enabled=cfg.log_telemetry)

        # Heterogeneous fleet: replaces the homogeneous instances built above with
        # router this fleet rebinds is the one constructed above; a caller-built fleet
        # would hold a different Router than the one this Simulator dispatches through.
        # Dynamic heterogeneous parallelism. Opt-in: with no solver the fleet keeps
        # the single shape built above and nothing in `dynamic_parallel` is reached.
        #
        # Built here rather than passed in because the router it rebinds is the one
        # constructed above. A caller-built fleet would hold a different Router than
        # this Simulator dispatches through, and the rebind would be a silent no-op.
        self._solver = solver
        self._fleet: DynamicParallelFleet | None = None
        if solver is not None:
            if parallel_cost_models is None or kv_blocks_per_tp is None:
                raise ValueError(
                    "A parallelism solver needs parallel_cost_models and "
                    "kv_blocks_per_tp: the fleet cannot price a step or size a KV pool "
                    "for a TP degree it has no model and no capacity for."
                )
            from .dynamic_parallel import DynamicParallelFleet

            self._fleet = DynamicParallelFleet(
                cost_models=parallel_cost_models,
                kv_blocks_per_tp=kv_blocks_per_tp,
                router=self.router,
                instance_cfg=dict(
                    token_budget=cfg.token_budget,
                    block_size=cfg.block_size,
                    max_running=cfg.max_concurrent,
                    max_concurrency=cfg.max_concurrency,
                    reject_if_kv_full=cfg.reject_if_kv_full,
                    reject_if_waiting=cfg.reject_if_waiting,
                    reject_if_running_full=cfg.reject_if_running_full,
                    enable_prefix_caching=cfg.enable_prefix_caching,
                ),
                initial_plan=solver.solve([], self.rollout_version),
                t=0.0,
            )
            # The fleet owns the instance list now; the uniform one built above is
            # discarded. Rebinding rather than keeping both is what stops the engine
            # loop advancing instances the router never dispatches to.
            self.instances = self._fleet.instances

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
        placed = self.router.dispatch(self.t)
        if not placed:
            return 0
        st = self._state()
        for req, instance_id in placed:
            self.policy.on_route(req, instance_id, st)
        self.telemetry.emit_instance_all(self.instances)
        self.telemetry.emit_gateway(self.t, self.router, self.instances)
        return len(placed)

    def _admit(self, quota: int | None = None) -> int:
        """Admit whole prompt groups into the router, up to a response `quota`.

        A group is admitted all-or-nothing, because a half-admitted group can
        never be consumed and its finished responses would hold KV until the
        run ends. The quota is a response count, rounded down to a whole group.

        Admission no longer touches an instance. The router accepts
        unconditionally and places the request when some instance can take it,
        so a busy fleet produces dwell rather than the rollback this used to
        do. What bounds admission is the policy's own quota, which is what
        bounded it before: the engine refusal was a secondary gate.
        """
        st = self._state()
        if quota is None:
            quota = self.policy.admit_quota(st)
        group_size = self.cfg.group_size
        admitted = 0
        while admitted + group_size <= quota:
            admitted += len(self._admit_one_group(st))
        if admitted:
            self.telemetry.emit_gateway(self.t, self.router, self.instances)
            self._dispatch_router()
        return admitted

    def _admit_one_group(self, st: SimState) -> list[Request]:
        """Create one whole prompt group and hand it to the router."""
        group: list[Request] = []
        group_id = self.next_group_id
        for index in range(self.cfg.group_size):
            source_group_id, prompt_len, target_len = self._next_request_shape()
            req = Request(
                rid=self.next_rid,
                group_id=group_id,
                group_index=index,
                source_group_id=source_group_id,
                prompt_len=prompt_len,
                target_len=target_len,
                v_traj=self.rollout_version,
                admit_time=self.t,
            )
            self.next_rid += 1
            self.inflight.append(req)
            self.n_admitted += 1
            group.append(req)

        self.next_group_id += 1
        self.router.enqueue(group, self.t)
        self.policy.on_admit(group, st)
        for req in group:
            self.telemetry.write_request("admit", self.t, req)
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
                self._admit()

                batch = self.policy.take_batch(self._state())
                if batch is None:
                    self.policy.on_batch_unavailable(self._state())
                    if not self.inflight and self._admit() == 0:
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
        st = self._state()
        for req in done:
            if req in self.inflight:
                self.inflight.remove(req)
            # complete_time is already set by _retire via t_local. Do not overwrite.
            self.telemetry.write_request("complete", self.t, req)
            siblings = self.pending.setdefault(req.group_id, [])
            siblings.append(req)
            if len(siblings) < self.cfg.group_size:
                continue
            group = sorted(self.pending.pop(req.group_id), key=lambda r: r.group_index)
            if self.policy.on_group_complete(group, st) is CompleteAction.DROP:
                for member in group:
                    member.status = ReqStatus.DROPPED
                    self.dropped.append(member)
                    self.telemetry.write_request("drop", self.t, member)
            else:
                self.ready_groups.append(group)
                self.ready.extend(group)

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

        self.policy.on_discard(unique_reqs, self._state())

        dropped_groups = {req.group_id for req in unique_reqs}
        for group_id in dropped_groups:
            self.pending.pop(group_id, None)
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

    def _expand_to_groups(self, reqs: list[Request]) -> list[Request]:
        """Add every still-live sibling of the given requests."""
        if self.cfg.group_size == 1:
            return list(reqs)
        wanted = {req.group_id for req in reqs}
        pending = [req for gid in wanted for req in self.pending.get(gid, ())]
        live = [
            req for req in (*self.inflight, *self.ready) if req.group_id in wanted
        ]
        return [*reqs, *pending, *live]

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
        if event.kind not in (EventKind.KV_PREEMPT, EventKind.DEADLOCK_PREEMPT):
            return False
        return any(
            len(inst.waiting) > self.router.max_waiting_after_preempt
            for inst in self.instances
        )

    def _undo_events(self, events: dict[int, EngineEvent]) -> None:
        """Undo all pending events (one per instance) and clear the events dict."""
        for iid in list(events):
            self.instances[iid].undo_last_event()
        events.clear()

    def _align_instances(self, t: float, force_align_t_local: bool = False) -> None:
        """Advance every lagging instance to a common global time.

        Alignment is a committed advance, so any undo log still open is closed
        first. `advance_to` would otherwise record committed mutations into a
        log belonging to an event that has already been decided.
        """
        for inst in self.instances:
            inst.commit_last_event()
        for inst in self.instances:
            assert inst.t_local <= t, "Instance time must be less than or equal to the target time."
            extra_done, _ = inst.advance_to(t, self.rollout_version)
            assert not extra_done, "No extra completions expected."
            if force_align_t_local:
                inst.t_local = t
        self.t = max(self.t, t)

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
            if not inst.running and not inst.waiting:
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

            # --- Cross-instance event: commit on owning instance, align others, re-seed ---
            if self._is_cross_instance(event):
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
                self._seed_events(events, queue)
                continue

            # Event is committed. Update global clock and close its undo log,
            # so a later router dispatch can tell a live speculative event from
            # a finished one.
            self.t = max(self.t, t_event)
            inst.commit_last_event()
            self.telemetry.sync(inst)

            # --- Process completions ---
            if event.kind == EventKind.REQUEST_COMPLETE and event.completed:
                self._collect(event.completed)

                if stop_on_batch:
                    if self.policy.peek_batch(self._state()) is not None:
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
                quota = self.policy.admit_quota(self._state())
                if quota > 0 or self.router.has_dispatchable():
                    self._undo_and_align(events, queue, t_event)
                    self.telemetry.sync_all(self.instances)
                    self._dispatch_router()
                    if quota > 0:
                        self._admit(quota)
                    self._seed_events(events, queue)
                    continue

            # --- Produce next event from this instance ---
            if inst.running or inst.waiting:
                next_event = inst.advance_one_event(self.rollout_version)
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
        and the policy is asked for a new admit quota.
        """
        previous_rollout_version = self.rollout_version
        self.train_version += 1
        self.policy.on_train_version_advance(self.train_version, self._state())
        self.rollout_version = self.policy.rollout_version_after_train(self._state())
        weights_changed = self.rollout_version != previous_rollout_version
        if weights_changed and self.cfg.partial_rollout and self.inflight:
            if self._fleet is not None and self._solver is not None:
                # Dynamic-parallel path: the solver picks the new layout, the fleet
                # aborts everything in flight and re-shapes the instances.
                # Imported here, not at module scope: with dynamic parallelism off
                # the package must stay untouched, which is what "opt-in" means.
                from .dynamic_parallel import RequestSnapshot

                snapshot = [
                    RequestSnapshot(v_traj=r.v_traj, ctx=r.ctx) for r in self.inflight
                ]
                plan = self._solver.solve(snapshot, self.rollout_version)
                aborted = self._fleet.rebuild(plan, self.t)
                # Tell the router which version now counts as current, so the
                # trajectories that just became the old-version tail are the ones it
                # steers toward high TP. Without this the strategy keeps comparing
                # against a stale version and treats the whole fleet as current work.
                set_reference = getattr(
                    self.router.strategy, "set_reference_version", None
                )
                if callable(set_reference):
                    set_reference(self.rollout_version)
                # The fleet may have constructed new instance objects, so rebind.
                # `self.instances` is what the engine loop iterates and what
                # `_align_instances` asserts clocks against.
                self.instances = self._fleet.instances
            else:
                # Existing homogeneous path — unchanged.
                aborted: list[Request] = []
                for inst in self.instances:
                    aborted.extend(inst.abort_all(requeue=False))
            self.router.enqueue(aborted, self.t)
            self.policy.on_interrupt(list(self.inflight), self._state())
            self.telemetry.emit_instance_all(self.instances)
            self.telemetry.emit_gateway(self.t, self.router, self.instances)
            self._dispatch_router()
        self._admit()  # emits router/instance telemetry itself if it admits anything.

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
        weight_sync = self.policy.will_sync_after_train(self._state())
        sync_time = self.sync_cost.sync_time(0, self.cfg.n_instances, 0) if weight_sync else 0.0
        recompute_time = self.recompute_cost.recompute_time(
            training_workload,
            self.training_parallelism,
        )

        total_duration = train_time + sync_time + recompute_time
        deadline = self.t + total_duration
        self._run_engine(deadline=deadline)
        self._publish_weights()

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
