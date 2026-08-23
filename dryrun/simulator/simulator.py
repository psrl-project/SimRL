"""The simulation driver: rollout generation interleaved with training.

This module provides the Simulator class that orchestrates the full RL
training loop using the event-driven architecture. It mirrors the sim_staleness
driver but with the multi-instance event coordinator infrastructure.
"""

from __future__ import annotations

import heapq

from tqdm import tqdm

from ..component.recompute.cost import FixedRecomputeCost, RecomputeCostModel
from ..component.rollout.engine import EngineEvent, EventKind, NativeInstance, ReqStatus, Request
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
from .config import SimConfig, SimResult, clip_length


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
        lengths: list[int],
        cfg: SimConfig,
        training_cost: TrainingCostModel | None = None,
        sync_cost: SyncCostModel | None = None,
        recompute_cost: RecomputeCostModel | None = None,
        store: JsonlStore | None = None,
    ):
        self.rollout_cost = rollout_cost
        self.policy = policy
        self.lengths = list(lengths)
        self.prompt_lengths = list(cfg.prompt_lengths)
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
        mean_prompt_len = sum(self.prompt_lengths) / len(self.prompt_lengths)
        self.recompute_cost = recompute_cost or FixedRecomputeCost(
            cfg.recompute_time / max(1, cfg.batch_size * mean_prompt_len) if cfg.recompute_time > 0 else 0.0
        )

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
                )
            )

        self.t: float = 0.0
        self.train_version: int = 0
        self.rollout_version: int = 0
        self.next_rid: int = 0
        self.inflight: list[Request] = []
        self.ready: list[Request] = []
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

    def _state(self) -> SimState:
        return SimState(
            now=self.t,
            train_version=self.train_version,
            rollout_version=self.rollout_version,
            batch_size=self.cfg.batch_size,
            max_staleness=self.cfg.max_staleness,
            inflight=self.inflight,
            ready=self.ready,
            n_admitted=self.n_admitted,
            n_consumed=self.n_consumed,
            n_instances=self.cfg.n_instances,
        )

    def _next_length(self) -> int:
        sampled = int(self.lengths[self.next_rid % len(self.lengths)])
        return clip_length(sampled, self.cfg.max_output_length)

    def _next_prompt_length(self) -> int:
        sampled = int(self.prompt_lengths[self.next_rid % len(self.prompt_lengths)])
        return clip_length(sampled, self.cfg.max_prompt_length)

    def _select_instance(self, req: Request) -> int:
        loads = [inst.n_active for inst in self.instances]
        return loads.index(min(loads))

    def _admit(self, quota: int | None = None) -> int:
        st = self._state()
        if quota is None:
            quota = self.policy.admit_quota(st)
        admitted = 0
        while admitted < quota:
            req = Request(
                rid=self.next_rid,
                prompt_len=self._next_prompt_length(),
                target_len=self._next_length(),
                v_traj=self.rollout_version,
                admit_time=self.t,
            )
            instance_id = self._select_instance(req)
            # False means temporarily full (KV / waiting / running), not
            # that this request is oversized.
            if not self.instances[instance_id].add_request(req, self.t):
                break
            self.next_rid += 1
            self.inflight.append(req)
            self.n_admitted += 1
            self.policy.on_admit(req, st)
            self.telemetry.write_request("admit", self.t, req)
            admitted += 1
        if admitted:
            self.telemetry.emit_instance_all(self.instances)
        return admitted

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
                        has_waiting = any(inst.waiting for inst in self.instances)
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
        st = self._state()
        for req in done:
            if req in self.inflight:
                self.inflight.remove(req)
            # complete_time is already set by _retire via t_local. Do not overwrite.
            self.telemetry.write_request("complete", self.t, req)
            if self.policy.on_complete(req, st) is CompleteAction.DROP:
                req.status = ReqStatus.DROPPED
                self.dropped.append(req)
                self.telemetry.write_request("drop", self.t, req)
            else:
                self.ready.append(req)

    def _discard(self, reqs: list[Request]) -> None:
        seen: set[int] = set()
        unique_reqs: list[Request] = []
        for req in reqs:
            if id(req) in seen:
                continue
            seen.add(id(req))
            unique_reqs.append(req)

        self.policy.on_discard(unique_reqs, self._state())

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

    def _is_cross_instance(self, event: EngineEvent) -> bool:
        # TODO(lhy): Return True for KV_PREEMPT once rerouting to other instances is implemented.
        return False

    def _undo_events(self, events: dict[int, EngineEvent]) -> None:
        """Undo all pending events (one per instance) and clear the events dict."""
        for iid in list(events):
            self.instances[iid].undo_last_event()
        events.clear()

    def _align_instances(self, t: float, force_align_t_local: bool = False) -> None:
        """Advance every lagging instance to a common global time."""
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
                # event is already committed on inst (at t_event).
                # Undo all other pending events, advance them to t_event for clock consistency.
                self._undo_and_align(events, queue, t_event)
                self.telemetry.sync_all(self.instances)
                # TODO(lhy): Handle global cross-instance consequence here (e.g. KV rerouting).
                # Re-seed all instances.
                self._seed_events(events, queue)
                continue

            # Event is committed. Update global clock.
            self.t = max(self.t, t_event)
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

                # A completion can release policy capacity. Only invalidate the
                # pending events when the policy permits another admission.
                quota = self.policy.admit_quota(self._state())
                if quota > 0:
                    self._undo_and_align(events, queue, t_event)
                    self.telemetry.sync_all(self.instances)
                    self._admit(quota)  # emits instance telemetry itself if it admits anything.
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
            for inst in self.instances:
                inst.abort_all()
            self.policy.on_interrupt(list(self.inflight), self._state())
            self.telemetry.emit_instance_all(self.instances)
        self._admit()  # emits instance telemetry itself if it admits anything.

    def _train(self, batch: list[Request]) -> None:
        """Consume `batch`, expire against the expected version, train, then publish.

        Committing this batch means it will become `train_version + 1`. Stale
        in-flight work is expired against that expected version immediately,
        without publishing yet. Rollout keeps the previous `rollout_version`
        until train+sync finish.
        """
        train_start = self.t
        for req in batch:
            req.consume_version = self.train_version
            if req in self.ready:
                self.ready.remove(req)
            self.consumed.append(req)
            self.n_consumed += 1
            self.telemetry.write_request("consume", self.t, req)

        expected_train_version = self.train_version + 1
        expired = self.policy.expire(self._state(), expected_train_version)
        if expired:
            self._discard(expired)
            self.telemetry.emit_instance_all(self.instances)

        sequence_lengths = tuple(req.ctx for req in batch)
        total_tokens = sum(sequence_lengths)
        training_workload = TrainingWorkload(
            sequence_lengths=sequence_lengths,
            micro_batch_size=self.cfg.training_micro_batch_size,
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
            len(batch),
            total_tokens,
            self.training_parallelism.dp,
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
            batch_size=len(batch),
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
            n_dropped_step=len(expired or []),
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
