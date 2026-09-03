"""End-to-end simulation tests."""

import pytest

from dryrun.component.rollout.engine import Request
from dryrun.cost.rollout import UnifiedRoofline
from dryrun.cost.training import (
    FixedTrainingLatency,
    FixedTrainingMemory,
    TrainingCostModel,
)
from dryrun.policy.areal import ArealPolicy
from dryrun.policy.base import CompleteAction, SimState
from dryrun.policy.psrl import PSRLPolicy
from dryrun.policy.roll import RollPolicy
from dryrun.policy.slime import SlimePolicy
from dryrun.policy.verl import VerlPolicy
from dryrun.simulator import SimConfig, Simulator, clip_length
from dryrun.telemetry.store import JsonlStore
from dryrun.workload.distributions import bimodal, uniform
from dryrun.workload.grouped import GroupedWorkload


def _run_sim(policy, lengths, cfg):
    """
    Run a simulation, optionally re-pairing `cfg.workload` with `lengths`.

    `lengths` is a list of output lengths for the common case where a test only
    varies the response distribution. It is cycled against the prompt lengths
    already in `cfg.workload`. Pass None when the test set `cfg.workload` itself.
    """
    cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7)
    training_cost = TrainingCostModel(
        FixedTrainingLatency(0.5),
        FixedTrainingMemory(0),
    )
    if lengths is not None:
        cfg.workload = GroupedWorkload.from_lengths(
            [group.prompt_len for group in cfg.workload.groups], lengths
        )
    sim = Simulator(
        rollout_cost=cost,
        training_cost=training_cost,
        policy=policy,
        cfg=cfg,
    )
    return sim.run()


class TestPSRLBoundHolds:
    def test_uniform_lengths(self):
        cfg = SimConfig(batch_size=4, max_staleness=2, n_versions=10)
        lengths = uniform(100, 50, 200, seed=42)
        result = _run_sim(PSRLPolicy(), lengths, cfg)
        assert not result.livelocked
        assert result.versions_done == 10
        for r in result.consumed:
            assert r.staleness_true() <= cfg.max_staleness

    def test_bimodal_lengths(self):
        cfg = SimConfig(batch_size=4, max_staleness=2, n_versions=8)
        lengths = bimodal(100, short_len=50, long_len=500, long_frac=0.25, seed=42)
        result = _run_sim(PSRLPolicy(), lengths, cfg)
        assert not result.livelocked
        for r in result.consumed:
            assert r.staleness_true() <= cfg.max_staleness


class TestPSRLRequestLifecycle:
    def test_interrupt_preserves_reservation_and_discard_releases_it(self):
        policy = PSRLPolicy()
        st = SimState(batch_size=4, max_staleness=2, train_version=0, rollout_version=0)
        req = Request(rid=0, prompt_len=10, target_len=100, v_traj=0)

        assert policy.admit_quota(st) == 12
        policy.on_admit(req, st)
        inventory = policy._inventory
        assert inventory is not None
        original_location = inventory.data_tracker[req.rid]
        buffer_id, entry_id = original_location
        original_version = inventory.buffers[buffer_id].entries[entry_id].entry_info.model_version

        interrupted = SimState(
            batch_size=4,
            max_staleness=2,
            train_version=1,
            rollout_version=1,
            inflight=[req],
        )
        policy.on_interrupt([req], interrupted)
        policy.on_interrupt([req], interrupted)

        assert inventory.data_tracker[req.rid] == original_location
        assert inventory.buffers[buffer_id].entries[entry_id].entry_info.model_version == original_version
        assert policy.admit_quota(st) == 11

        policy.on_discard([req], st)
        assert req.rid not in inventory.data_tracker
        assert policy.admit_quota(st) == 12

    def test_discard_releases_occupied_entry(self):
        policy = PSRLPolicy()
        st = SimState(batch_size=4, max_staleness=2, train_version=0, rollout_version=0)
        req = Request(rid=0, prompt_len=10, target_len=100, v_traj=0)

        policy.admit_quota(st)
        policy.on_admit(req, st)
        assert policy.on_complete(req, st) is CompleteAction.KEEP

        inventory = policy._inventory
        assert inventory is not None
        assert req.rid in inventory.data_tracker
        policy.on_discard([req], st)
        assert req.rid not in inventory.data_tracker


class TestArealTokenBucket:
    def test_bucket_invariant(self):
        cfg = SimConfig(batch_size=4, max_staleness=2, n_versions=10)
        lengths = uniform(100, 50, 100, seed=0)
        result = _run_sim(ArealPolicy(), lengths, cfg)
        assert not result.livelocked
        assert result.versions_done == 10


class TestRollBound:
    def test_bound_holds_uniform(self):
        cfg = SimConfig(batch_size=4, max_staleness=2, n_versions=10)
        lengths = uniform(100, 50, 100, seed=0)
        result = _run_sim(RollPolicy(), lengths, cfg)
        assert not result.livelocked
        for r in result.consumed:
            assert r.staleness_true() <= cfg.max_staleness


class TestSlimeBarrier:
    def test_staleness_equals_interval(self):
        cfg = SimConfig(
            batch_size=4,
            max_staleness=1,
            n_versions=10,
            partial_rollout=False,
        )
        lengths = uniform(100, 50, 100, seed=0)
        policy = SlimePolicy(update_weights_interval=2)
        result = _run_sim(policy, lengths, cfg)
        assert not result.livelocked
        for r in result.consumed:
            assert r.staleness_true() <= 2


class TestVerlAsync:
    def test_runs_without_crash(self):
        cfg = SimConfig(batch_size=4, max_staleness=1, n_versions=8)
        lengths = uniform(100, 50, 100, seed=0)
        policy = VerlPolicy(staleness_threshold=0.1)
        result = _run_sim(policy, lengths, cfg)
        assert not result.livelocked
        assert result.versions_done == 8

    def test_window_budget_and_sync_carryover(self):
        policy = VerlPolicy(staleness_threshold=2.0)
        st = SimState(batch_size=4, n_instances=2)
        assert policy.admit_quota(st) == 12

        reqs = [Request(rid=i, prompt_len=10, target_len=100, v_traj=0) for i in range(6)]
        for req in reqs:
            st.inflight.append(req)
            policy.on_admit(req, st)
        assert policy.staleness_samples == 6
        assert policy.admit_quota(st) == 6

        for req in reqs[:4]:
            st.inflight.remove(req)
            assert policy.on_complete(req, st) is CompleteAction.KEEP
        assert policy.take_batch(st) == reqs[:4]
        assert policy.staleness_samples == 6
        assert policy.admit_quota(st) == 6

        policy.on_train_version_advance(1, st)
        assert policy.staleness_samples == len(st.inflight) + len(policy.queue) == 2


class TestDeterminism:
    def test_same_seed_same_result(self):
        cfg = SimConfig(batch_size=4, max_staleness=2, n_versions=5)
        lengths = uniform(50, 50, 100, seed=42)
        r1 = _run_sim(PSRLPolicy(), lengths, cfg)
        r2 = _run_sim(PSRLPolicy(), lengths, cfg)
        assert r1.sim_time == r2.sim_time
        assert r1.versions_done == r2.versions_done
        assert len(r1.consumed) == len(r2.consumed)


class TestMultiInstance:
    def test_two_instances(self):
        cfg = SimConfig(
            batch_size=4,
            max_staleness=2,
            n_versions=8,
            n_instances=2,
        )
        lengths = uniform(100, 50, 100, seed=0)
        result = _run_sim(PSRLPolicy(), lengths, cfg)
        assert not result.livelocked
        assert result.versions_done == 8
        for r in result.consumed:
            assert r.staleness_true() <= cfg.max_staleness


class TestVariablePromptLengths:
    def test_variable_prompt_lengths(self):
        prompt_lengths = uniform(100, 100, 1000, seed=7)
        cfg = SimConfig(
            batch_size=4,
            max_staleness=2,
            n_versions=8,
            workload=GroupedWorkload.from_lengths(prompt_lengths, uniform(100, 50, 100, seed=0)),
        )
        result = _run_sim(PSRLPolicy(), None, cfg)
        assert not result.livelocked
        assert result.versions_done == 8
        observed_prompt_lens = {r.prompt_len for r in result.consumed}
        assert len(observed_prompt_lens) > 1


def _is_partial_abort(row: dict) -> bool:
    """True when running is empty and interrupted waiters still need a re-prefill."""
    if row["num_running_reqs"] != 0:
        return False
    waiting = row.get("waiting_len") or {}
    if not waiting:
        return False
    infos = list(waiting.values())
    return all(info["prefilled"] == 0 for info in infos) and any(
        info["generated"] > 0 for info in infos
    )


class TestPartialRolloutPublish:
    def test_abort_after_train_and_sync_not_at_train_start(self, tmp_path):
        store = JsonlStore(tmp_path / "logs")
        cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7)
        training_cost = TrainingCostModel(
            FixedTrainingLatency(1.0),
            FixedTrainingMemory(0),
        )
        cfg = SimConfig(
            batch_size=4,
            max_staleness=2,
            n_versions=4,
            n_instances=1,
            partial_rollout=True,
            sync_time=0.25,
            workload=GroupedWorkload.from_lengths([512], uniform(80, 200, 400, seed=0)),
        )
        sim = Simulator(
            rollout_cost=cost,
            training_cost=training_cost,
            policy=ArealPolicy(),
            cfg=cfg,
            store=store,
        )
        result = sim.run()
        store.close()

        assert not result.livelocked
        trains = list(store.iter("train"))
        inst_rows = list(store.iter("instance/0"))
        assert trains
        assert inst_rows

        train_ends = [train["end"] for train in trains]
        for row in inst_rows:
            if _is_partial_abort(row):
                assert any(abs(row["t"] - end) < 1e-9 for end in train_ends), (
                    "Partial abort should only occur after train+sync."
                )

        saw_abort = False
        for train in trains:
            start, end = train["start"], train["end"]
            assert end > start, "Train+sync window should have positive duration."
            during = [r for r in inst_rows if start <= r["t"] < end]
            assert any(r["num_running_reqs"] > 0 for r in during), (
                "Rollout should keep generating during training and sync."
            )
            at_end = [r for r in inst_rows if abs(r["t"] - end) < 1e-9]
            if any(_is_partial_abort(r) for r in at_end):
                saw_abort = True
        assert saw_abort, "Expected at least one partial abort after train+sync."

    def test_areal_admits_at_publish_not_mid_train(self, tmp_path):
        store = JsonlStore(tmp_path / "logs")
        cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7)
        training_cost = TrainingCostModel(
            FixedTrainingLatency(1.0),
            FixedTrainingMemory(0),
        )
        cfg = SimConfig(
            batch_size=4,
            max_staleness=2,
            n_versions=4,
            n_instances=1,
            partial_rollout=True,
            sync_time=0.25,
            workload=GroupedWorkload.from_lengths([512], uniform(80, 200, 400, seed=0)),
        )
        sim = Simulator(
            rollout_cost=cost,
            training_cost=training_cost,
            policy=ArealPolicy(),
            cfg=cfg,
            store=store,
        )
        result = sim.run()
        store.close()

        assert not result.livelocked
        trains = list(store.iter("train"))
        admit_ts = [r["t"] for r in store.iter("request") if r["event"] == "admit"]
        for train in trains:
            start, end = train["start"], train["end"]
            during = [t for t in admit_ts if start < t < end]
            assert not during, "AReaL quota should not open mid-training."
            at_end = [t for t in admit_ts if abs(t - end) < 1e-9]
            assert at_end, "AReaL quota should open when train_version is published."

    def test_psrl_interrupt_keeps_one_batch_per_published_version(self, tmp_path):
        store = JsonlStore(tmp_path / "logs")
        cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7)
        training_cost = TrainingCostModel(
            FixedTrainingLatency(1.0),
            FixedTrainingMemory(0),
        )
        cfg = SimConfig(
            batch_size=4,
            max_staleness=2,
            n_versions=4,
            n_instances=1,
            partial_rollout=True,
            sync_time=0.25,
            workload=GroupedWorkload.from_lengths([512], uniform(80, 200, 400, seed=0)),
        )
        sim = Simulator(
            rollout_cost=cost,
            training_cost=training_cost,
            policy=PSRLPolicy(),
            cfg=cfg,
            store=store,
        )
        result = sim.run()
        store.close()

        assert not result.livelocked
        assert not result.dropped
        assert all(r.staleness_true() <= cfg.max_staleness for r in result.consumed)
        assert sim.n_admitted == (cfg.max_staleness + 1 + cfg.n_versions) * cfg.batch_size
        assert sim.n_admitted - sim.n_consumed == (cfg.max_staleness + 1) * cfg.batch_size

        admit_counts: dict[float, int] = {}
        for row in store.iter("request"):
            if row["event"] == "admit":
                admit_counts[row["t"]] = admit_counts.get(row["t"], 0) + 1
        trains = list(store.iter("train"))
        assert admit_counts.pop(0.0) == (cfg.max_staleness + 1) * cfg.batch_size
        for train in trains:
            assert admit_counts.pop(train["end"]) == cfg.batch_size
        assert not admit_counts, "PSRL should admit only initially and when a version is published."


class TestLengthCaps:
    def test_clip_length_passthrough(self):
        assert clip_length(900, None) == 900

    def test_clip_length_truncates(self):
        assert clip_length(900, 128) == 128
        assert clip_length(64, 128) == 64

    def test_clip_length_rejects_non_positive_cap(self):
        with pytest.raises(ValueError, match="Expected max_length >= 1"):
            clip_length(10, 0)

    def test_simulator_truncates_prompt_and_output(self):
        cfg = SimConfig(
            batch_size=4,
            max_staleness=2,
            n_versions=4,
            workload=GroupedWorkload.from_lengths([100, 4000, 50], [500, 400, 10]),
            max_prompt_length=256,
            max_output_length=80,
            log_telemetry=False,
        )
        result = _run_sim(PSRLPolicy(), None, cfg)
        requests = result.consumed + result.dropped
        assert requests, "Expected at least one admitted request."
        assert all(r.prompt_len <= 256 for r in requests)
        assert all(r.target_len <= 80 for r in requests)
        assert any(r.target_len == 10 for r in requests)


class TestGroupIdPropagation:
    def test_group_id_reaches_requests(self):
        """Group ids from the workload must appear on the generated requests."""
        from collections import Counter

        from dryrun.workload.trace import RequestGroup

        groups = [
            RequestGroup(group_id=i, prompt_len=128, output_lens=(50, 60, 70, 80))
            for i in range(6)
        ]
        cfg = SimConfig(
            batch_size=8,
            max_staleness=2,
            n_versions=6,
            workload=GroupedWorkload.from_groups(groups),
            log_telemetry=False,
        )
        result = _run_sim(PSRLPolicy(), None, cfg)
        all_requests = result.consumed + result.dropped
        assert all_requests, "Expected at least one admitted request."

        workload_group_ids = {g.group_id for g in groups}
        observed_group_ids = {r.group_id for r in all_requests}

        # Every observed group id must come from the workload.
        assert observed_group_ids <= workload_group_ids, (
            f"Unexpected group ids {observed_group_ids - workload_group_ids!r} found on requests."
        )

        # More than one distinct group id must be seen. If group_id is not
        # threaded from the workload into Request, every request would land
        # on the default (group 0) and len(observed_group_ids) would be 1.
        assert len(observed_group_ids) > 1, (
            f"Expected more than one distinct group id, got {observed_group_ids!r}."
        )

        # At least one group id must appear on more than one request, proving
        # that multi-response groups are actually used (not just group 0 for all).
        counts = Counter(r.group_id for r in all_requests)
        assert max(counts.values()) > 1, (
            "Expected at least one group id to appear on more than one request."
        )
