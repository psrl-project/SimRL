"""The router inside a full simulation.

Three things only a whole run can check: that routing did not change the
single-instance result, that a saturated policy does not strand the router
queue, and that capacity is released only at the events the router watches.
"""

from __future__ import annotations

from srl.component.rollout.engine import EngineEvent, EventKind, Request
from srl.cost.rollout import UnifiedRoofline
from srl.cost.training import (
    FixedTrainingLatency,
    FixedTrainingMemory,
    TrainingCostModel,
)
from srl.policy.areal import ArealPolicy
from srl.policy.psrl import PSRLPolicy
from srl.simulator.config import SimConfig
from srl.simulator.simulator import Simulator
from srl.workload.distributions import uniform
from srl.workload.grouped import GroupedWorkload
from srl.workload.trace import RequestGroup


def _sim(cfg: SimConfig, lengths=None, policy=None) -> Simulator:
    cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7)
    if lengths is not None:
        cfg.workload = GroupedWorkload.from_lengths(
            [group.prompt_len for group in cfg.workload.groups], lengths
        )
    return Simulator(
        rollout_cost=cost,
        training_cost=TrainingCostModel(FixedTrainingLatency(0.5), FixedTrainingMemory(0)),
        policy=policy or PSRLPolicy(),
        cfg=cfg,
    )


class TestBehaviourPreservation:
    def test_one_instance_leaves_routing_no_decision(self):
        """With a single instance the strategy cannot matter, so every strategy
        must produce the identical run. A difference here is a plumbing bug."""
        results = {}
        for strategy in ("request_num_balance", "round_robin", "kv_cache_aware"):
            cfg = SimConfig(batch_size=4, max_staleness=2, n_versions=6, n_instances=1)
            cfg.routing_strategy = strategy
            result = _sim(cfg, uniform(100, 50, 100, seed=0)).run()
            results[strategy] = (result.sim_time, result.n_consumed, result.versions_done)
        assert len(set(results.values())) == 1, results

    def test_the_default_strategy_balances_two_instances(self):
        cfg = SimConfig(batch_size=4, max_staleness=2, n_versions=8, n_instances=2)
        result = _sim(cfg, uniform(100, 50, 100, seed=0)).run()
        assert not result.livelocked
        assert result.versions_done == 8

    def test_no_strategy_leaves_an_instance_idle_for_the_whole_run(self):
        """Regression: `kv_cache_aware` read raw free blocks, which are equal
        while requests are still queued, so every placement went to instance 0
        and instance 1 drew a flat zero line for the entire run. A strategy
        that silently collapses onto one instance is worse than useless, since
        the run still completes and only the figure reveals it."""
        for strategy in ("request_num_balance", "kv_cache_aware", "round_robin"):
            cfg = SimConfig(batch_size=8, max_staleness=2, n_versions=5, n_instances=2)
            cfg.routing_strategy = strategy
            sim = _sim(cfg, uniform(100, 100, 300, seed=1))
            sim.run()
            served = [inst.completed for inst in sim.instances]
            assert all(served), f"{strategy} left an instance unused: {[len(s) for s in served]}"


class TestNoStarvation:
    def test_a_saturated_policy_still_drains_the_router(self):
        """A completion frees instance capacity even when the policy grants no
        new quota. Gating dispatch on the quota alone would leave the queue
        untouched until the next training step."""
        cfg = SimConfig(
            batch_size=4,
            max_staleness=1,
            n_versions=6,
            n_instances=2,
            max_concurrency=2,  # force a backlog the router has to hold
        )
        sim = _sim(cfg, uniform(100, 200, 400, seed=3))
        result = sim.run()
        assert not result.livelocked
        assert result.versions_done == 6
        # Everything admitted was eventually placed, so nothing was stranded.
        assert len(sim.router) == 0
        assert sim.router.n_routed >= sim.n_admitted

    def test_requests_are_not_lost_when_the_fleet_is_full(self):
        cfg = SimConfig(
            batch_size=2,
            max_staleness=1,
            n_versions=4,
            n_instances=1,
            max_concurrency=1,
        )
        sim = _sim(cfg, uniform(100, 100, 200, seed=5))
        result = sim.run()
        assert not result.livelocked
        accounted = result.n_consumed + len(result.dropped)
        assert accounted + len(sim.inflight) == sim.n_admitted


class TestPlacementTimeAdmission:
    def test_staging_does_not_consume_policy_or_inflight_capacity(self):
        """A staged group costs nothing until something places it.

        The router cap, not the policy, is what holds the queue back here. A
        zero `max_inflight` would be the wrong instrument: staging is gated by
        `admit_quota`, so a policy that admits nothing correctly stages nothing,
        and the queue this test is about would never form.
        """
        cfg = SimConfig(
            batch_size=2,
            max_staleness=1,
            n_versions=1,
            n_instances=1,
            max_concurrent_seqs_per_instance=1,
        )
        policy = ArealPolicy()
        sim = _sim(cfg, policy=policy)

        assert sim._stage() == 4
        # One response was placed; the cap strands the rest in the router. The
        # default group_size is 1, so a placement commits exactly one response.
        assert len(sim.router) == 3
        assert sim.n_admitted == 1
        assert len(sim.inflight) == 1
        assert policy.running == 1
        # The three queued responses hold quota but no instance.
        assert sim.router.queue_depths == {0: 3}

    def test_first_sibling_placement_commits_the_whole_group_once(self):
        workload = GroupedWorkload.from_groups(
            [RequestGroup(group_id=0, prompt_len=32, output_lens=(16, 16))]
        )
        cfg = SimConfig(
            batch_size=1,
            group_size=2,
            max_staleness=0,
            n_versions=1,
            max_concurrent_seqs_per_instance=1,
            workload=workload,
        )
        sim = _sim(cfg, policy=ArealPolicy(max_inflight=1))

        sim._stage()
        assert sim.n_admitted == 2
        assert len(sim.inflight) == 2
        assert len(sim.router) == 1
        assert len(sim._admitted_group_ids) == 1
        assert sim.policy.running == 1

    def test_unadmitted_groups_rebase_when_rollout_weights_advance(self):
        """A queued, uncommitted group adopts the new weights rather than aging.

        As above, the router cap is what keeps the groups queued: staging now
        follows `admit_quota`, so a policy admitting nothing would leave the
        queue empty and there would be nothing to rebase.
        """
        cfg = SimConfig(
            batch_size=2,
            max_staleness=1,
            n_versions=1,
            n_instances=1,
            max_concurrent_seqs_per_instance=1,
        )
        sim = _sim(cfg, policy=ArealPolicy())
        sim._stage()
        queued = [
            req
            for group_id, group in sim._staged_groups.items()
            if group_id not in sim._admitted_group_ids
            for req in group
        ]
        assert sim.router.queue_depths == {0: 3}
        assert queued, "No uncommitted group was left to rebase."

        sim._publish_weights()

        assert sim.rollout_version == 1
        assert all(req.v_traj == 1 for req in queued)
        assert set(sim.router.queue_depths) == {1}


class TestPreemptOverflowInAFullRun:
    def test_global_waiting_gate_keeps_the_rest_in_the_router(self):
        cfg = SimConfig(
            batch_size=2,
            max_staleness=1,
            n_versions=1,
            n_instances=2,
            router_reject_on_waiting=True,
        )
        sim = _sim(cfg, uniform(100, 100, 200, seed=17))

        staged = sim._stage()

        assert staged > len(sim.instances)
        assert [len(inst.waiting) for inst in sim.instances] == [1, 1]
        assert len(sim.router) == staged - len(sim.instances)

    def test_preempt_only_waiting_gate_keeps_initial_admission_permissive(self):
        cfg = SimConfig(
            batch_size=2,
            max_staleness=1,
            n_versions=1,
            n_instances=2,
            router_reject_preempted_on_waiting=True,
        )
        sim = _sim(cfg, uniform(100, 100, 200, seed=17))

        staged = sim._stage()

        assert sum(inst.n_active for inst in sim.instances) == staged
        assert len(sim.router) == 0

    def test_global_waiting_gate_collects_simultaneous_completions(self):
        cfg = SimConfig(
            batch_size=1,
            max_staleness=1,
            n_versions=1,
            n_instances=2,
            router_reject_on_waiting=True,
        )
        result = _sim(cfg, uniform(100, 10, 10, seed=17)).run()

        assert not result.livelocked
        assert result.versions_done == 1

    def test_completion_with_preemption_loops_back_and_is_collected(self):
        cfg = SimConfig(
            batch_size=1,
            max_staleness=1,
            n_versions=1,
            max_waiting_after_preempt=1,
            router_reject_preempted_on_waiting=True,
        )
        policy = ArealPolicy()
        sim = _sim(cfg, policy=policy)
        inst = sim.instances[0]
        already_waiting = Request(
            rid=0, group_id=0, prompt_len=100, target_len=100, v_traj=0
        )
        victim = Request(
            rid=1, group_id=1, prompt_len=100, target_len=100, v_traj=0
        )
        assert inst.add_request(already_waiting, t=0.0)
        assert inst.add_request(victim, t=0.0)
        victim.num_preemptions = 1

        completed = Request(
            rid=2, group_id=2, prompt_len=100, target_len=100, v_traj=0
        )
        completed.complete_time = 0.0
        sim.inflight.append(completed)
        sim._admitted_group_ids.add(completed.group_id)
        policy.on_admit([completed], sim._state())
        event = EngineEvent(
            kind=EventKind.REQUEST_COMPLETE,
            instance_id=0,
            t=0.0,
            completed=[completed],
            preempted=[victim],
        )
        emitted = False

        def emit_once(_version):
            nonlocal emitted
            if emitted:
                return None
            emitted = True
            return event

        inst.advance_one_event = emit_once
        sim._run_engine(stop_on_batch=True)

        assert sim.router.queue.peek() is victim
        assert completed not in sim.inflight
        assert completed in sim.ready

    def test_crossing_the_threshold_moves_work_across_instances(self):
        """The path the aligned baseline cannot exercise: it sets the threshold
        to 1024 while its instance queues peak at 218, so every one of its 383
        loopbacks is a weight-update abort. Lowering the threshold to a depth
        this workload actually reaches is the only way to execute it."""
        cfg = SimConfig(
            batch_size=8,
            max_staleness=2,
            n_versions=5,
            n_instances=2,
            kv_blocks=700,  # tight enough that preemption happens
            block_size=16,
            max_waiting_after_preempt=1,
            router_reject_preempted_on_waiting=True,
        )
        sim = _sim(cfg, uniform(100, 400, 900, seed=17))
        result = sim.run()
        assert not result.livelocked
        preemptions = sum(inst.n_preemptions for inst in sim.instances)
        assert preemptions > 0, "The fixture must actually cause preemption."
        assert sim.router.n_reroutes > 0, "Overflow never reached the router."

    def test_the_default_leaves_every_preemption_local(self):
        """Without a threshold the router must never see a preempted request,
        which is the behaviour before this path existed."""
        cfg = SimConfig(
            batch_size=8,
            max_staleness=2,
            n_versions=5,
            n_instances=2,
            kv_blocks=700,
            block_size=16,
            partial_rollout=False,  # isolate preemption from weight-update aborts
        )
        sim = _sim(cfg, uniform(100, 400, 900, seed=17))
        sim.run()
        assert sum(inst.n_preemptions for inst in sim.instances) > 0
        assert sim.router.n_reroutes == 0


class TestCapacityReleaseIsWatched:
    def test_free_blocks_only_grow_at_events_the_router_watches(self):
        """The plan's load-bearing claim, asserted rather than trusted: if a
        fourth release path is ever added, this fails at that commit instead of
        silently leaving the router blind to it.

        `advance_to` is excluded by construction: it is the committed-alignment
        path and can retire requests, which is the same release as a
        REQUEST_COMPLETE and is followed by a dispatch anyway.
        """
        cfg = SimConfig(batch_size=4, max_staleness=2, n_versions=5, n_instances=2)
        sim = _sim(cfg, uniform(100, 100, 300, seed=11))

        releasing = {
            EventKind.REQUEST_COMPLETE,
            EventKind.PREEMPT,
        }
        violations: list[tuple[str, int, int]] = []
        originals = {}

        for inst in sim.instances:
            def wrapped(version, _inst=inst):
                before = _inst.free_blocks()
                event = originals[_inst.instance_id](version)
                after = _inst.free_blocks()
                if after > before and event is not None and event.kind not in releasing:
                    violations.append((event.kind.value, before, after))
                return event

            originals[inst.instance_id] = inst.advance_one_event
            inst.advance_one_event = wrapped

        sim.run()
        assert violations == [], f"Capacity was released by an unwatched event: {violations}"
