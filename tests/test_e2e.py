"""End-to-end simulation tests."""

import pytest

from srl.component.rollout.engine import Request
from srl.cost.rollout import UnifiedRoofline
from srl.cost.training import (
    FixedTrainingLatency,
    FixedTrainingMemory,
    TrainingCostModel,
)
from srl.policy.areal import ArealPolicy
from srl.policy.base import CancelAction, CompleteAction, SimState
from srl.policy.psrl import PSRLPolicy
from srl.policy.roll import RollPolicy
from srl.policy.slime import SlimePolicy
from srl.policy.verl import VerlPolicy
from srl.simulator import SimConfig, Simulator, clip_length, clip_total_length
from srl.telemetry.store import JsonlStore
from srl.workload.distributions import bimodal, uniform
from srl.workload.grouped import GroupedWorkload
from srl.workload.trace import RequestGroup


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
        cfg = SimConfig(batch_size=4, staleness_horizon=2, n_versions=10)
        lengths = uniform(100, 50, 200, seed=42)
        result = _run_sim(PSRLPolicy(), lengths, cfg)
        assert not result.livelocked
        assert result.versions_done == 10
        for r in result.consumed:
            assert r.staleness_true() <= cfg.staleness_horizon

    def test_bimodal_lengths(self):
        cfg = SimConfig(batch_size=4, staleness_horizon=2, n_versions=8)
        lengths = bimodal(100, short_len=50, long_len=500, long_frac=0.25, seed=42)
        result = _run_sim(PSRLPolicy(), lengths, cfg)
        assert not result.livelocked
        for r in result.consumed:
            assert r.staleness_true() <= cfg.staleness_horizon


class TestPSRLRequestLifecycle:
    def test_interrupt_preserves_reservation_and_discard_releases_it(self):
        policy = PSRLPolicy()
        st = SimState(batch_size=4, group_size=1, max_staleness=2, train_version=0, rollout_version=0)
        req = Request(rid=0, group_id=0, prompt_len=10, target_len=100, v_traj=0)

        assert policy.admit_quota(st) == 12
        policy.on_admit([req], st)
        inventory = policy._inventory
        assert inventory is not None
        original_location = inventory.data_tracker[req.group_id]
        buffer_id, entry_id = original_location
        original_version = inventory.buffers[buffer_id].entries[entry_id].entry_info.model_version

        interrupted = SimState(
            batch_size=4,
            group_size=1,
            max_staleness=2,
            train_version=1,
            rollout_version=1,
            inflight=[req],
        )
        policy.on_interrupt([req], interrupted)
        policy.on_interrupt([req], interrupted)

        assert inventory.data_tracker[req.group_id] == original_location
        assert inventory.buffers[buffer_id].entries[entry_id].entry_info.model_version == original_version
        assert policy.admit_quota(st) == 11

        policy.on_discard([req], st)
        assert req.group_id not in inventory.data_tracker
        assert policy.admit_quota(st) == 12

    def test_discard_releases_occupied_entry(self):
        policy = PSRLPolicy()
        st = SimState(batch_size=4, group_size=1, max_staleness=2, train_version=0, rollout_version=0)
        req = Request(rid=0, group_id=0, prompt_len=10, target_len=100, v_traj=0)

        policy.admit_quota(st)
        policy.on_admit([req], st)
        assert policy.on_group_complete([req], st) is CompleteAction.KEEP

        inventory = policy._inventory
        assert inventory is not None
        assert req.group_id in inventory.data_tracker
        policy.on_discard([req], st)
        assert req.group_id not in inventory.data_tracker


def _psrl_group(group_id: int, v_traj: int, size: int = 2) -> list[Request]:
    return [
        Request(
            rid=group_id * size + i, group_id=group_id, group_index=i, prompt_len=10, target_len=100, v_traj=v_traj
        )
        for i in range(size)
    ]


class TestPSRLInventorySnapshot:
    @pytest.mark.parametrize("staleness", [0, 1, 2])
    def test_a_reserved_group_sits_in_the_last_buffer_and_its_deadline_is_that_buffer(self, staleness):
        policy = PSRLPolicy()
        st = SimState(batch_size=4, group_size=2, max_staleness=staleness, train_version=0, rollout_version=0)
        policy.admit_quota(st)
        policy.on_admit(_psrl_group(7, v_traj=0), st)

        snapshot = policy.inventory_snapshot(st)

        assert snapshot.staleness == staleness
        assert snapshot.num_entries == 4
        assert [view.buffer_id for view in snapshot.buffers] == list(range(staleness + 1))
        assert [view.reserved for view in snapshot.buffers] == [()] * staleness + [((7, 0),)]
        assert snapshot.deadline(7) == staleness

    def test_each_group_keeps_the_deadline_of_the_version_it_was_admitted_at(self):
        policy = PSRLPolicy()
        st0 = SimState(batch_size=4, group_size=2, max_staleness=2, train_version=0, rollout_version=0)
        st1 = SimState(batch_size=4, group_size=2, max_staleness=2, train_version=1, rollout_version=1)
        policy.admit_quota(st0)
        policy.on_admit(_psrl_group(7, v_traj=0), st0)
        policy.admit_quota(st1)
        policy.on_admit(_psrl_group(8, v_traj=1), st1)

        snapshot = policy.inventory_snapshot(st1)

        reserved = {view.buffer_id: view.reserved for view in snapshot.buffers}
        assert reserved == {0: (), 1: (), 2: ((7, 0),), 3: ((8, 1),)}
        assert snapshot.deadline(7) == 2
        assert snapshot.deadline(8) == 3

    def test_a_completed_group_is_occupied_rather_than_reserved(self):
        policy = PSRLPolicy()
        st = SimState(batch_size=4, group_size=2, max_staleness=1, train_version=0, rollout_version=0)
        group = _psrl_group(7, v_traj=0)
        policy.admit_quota(st)
        policy.on_admit(group, st)
        assert policy.on_group_complete(group, st) is CompleteAction.KEEP

        snapshot = policy.inventory_snapshot(st)

        assert [(view.n_occupied, view.reserved) for view in snapshot.buffers] == [(1, ()), (0, ())]
        with pytest.raises(KeyError, match="no reservation"):
            snapshot.deadline(7)


class TestArealTokenBucket:
    def test_bucket_invariant(self):
        cfg = SimConfig(batch_size=4, staleness_horizon=2, n_versions=10)
        lengths = uniform(100, 50, 100, seed=0)
        result = _run_sim(ArealPolicy(), lengths, cfg)
        assert not result.livelocked
        assert result.versions_done == 10

    def test_a_straggler_is_consumed_unbounded_and_the_report_is_not_strict(self):
        """AReaL's bucket counts running prompts, not their age: a straggler holds

        one `running` slot for as long as it takes while the bucket keeps growing
        with `train_version`, so nothing stops its eventual staleness from
        exceeding the declared horizon.
        """
        workload = GroupedWorkload.from_lengths([150], [10000] + [50] * 11)
        # n_versions well past the brief's 8: a 10000-token straggler needs that many
        # training steps to actually finish and register its staleness.
        cfg = SimConfig(batch_size=2, staleness_horizon=2, n_versions=40, workload=workload)
        result = _run_sim(ArealPolicy(), None, cfg)

        assert not result.livelocked
        report = result.staleness_report()
        assert report.max > cfg.staleness_horizon, "the straggler must outrun the declared horizon."
        assert report.strict is False


class TestRollBound:
    def test_bound_holds_uniform(self):
        cfg = SimConfig(batch_size=4, staleness_horizon=2, n_versions=10)
        lengths = uniform(100, 50, 100, seed=0)
        result = _run_sim(RollPolicy(), lengths, cfg)
        assert not result.livelocked
        for r in result.consumed:
            assert r.staleness_true() <= cfg.staleness_horizon


def _roll_sim(group_size: int, store=None, n_versions: int = 10):
    lengths = uniform(64, 50, 900, seed=7)
    if group_size == 1:
        workload = GroupedWorkload.from_lengths([150], lengths)
    else:
        workload = GroupedWorkload.from_groups(
            [
                RequestGroup(
                    group_id=i,
                    prompt_len=150,
                    output_lens=tuple(lengths[(i * group_size + j) % 64] for j in range(group_size)),
                )
                for i in range(16)
            ]
        )
    cfg = SimConfig(
        batch_size=4,
        group_size=group_size,
        staleness_horizon=2,
        ingress_window_groups=3 * 4,
        n_versions=n_versions,
        workload=workload,
    )
    policy = RollPolicy()
    sim = Simulator(
        rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
        training_cost=TrainingCostModel(FixedTrainingLatency(0.5), FixedTrainingMemory(0)),
        policy=policy,
        cfg=cfg,
        store=store,
    )
    return sim, policy


class TestRollGarbageCollection:
    @pytest.mark.parametrize("group_size", [1, 4])
    def test_every_drop_is_a_gc_of_a_whole_group(self, group_size):
        sim, policy = _roll_sim(group_size)
        result = sim.run()
        assert not result.livelocked
        assert policy.gc_dropped > 0, "the fixture must make GC fire."
        assert len(result.dropped) == policy.gc_dropped * group_size
        assert result.staleness_report().strict is True, "GC is what keeps the bound strict here."

    @pytest.mark.parametrize("group_size", [1, 4])
    def test_policy_counters_agree_with_the_ledger(self, group_size):
        sim, policy = _roll_sim(group_size)
        sim.run()
        views = sim._ledger.views()
        running = {gid for cohort in policy.cohorts.values() for gid in cohort.running}
        finished = {group[0].group_id for cohort in policy.cohorts.values() for group in cohort.finished}
        assert running == {v.group_id for v in views if v.status != "done"}
        assert finished == {v.group_id for v in views if v.status == "done"}
        assert policy.running_prompts == len(running)
        consumed_groups = sim.n_consumed // group_size
        assert policy.completed_prompts == consumed_groups + len(finished)

    def test_gc_fires_at_publish_not_at_batch_commit(self, tmp_path):
        store = JsonlStore(tmp_path / "logs")
        sim, policy = _roll_sim(1, store=store)
        sim.run()
        store.close()
        assert policy.gc_dropped > 0
        trains = list(store.iter("train"))
        drops = [row["t"] for row in store.iter("request") if row["event"] == "drop"]
        for t in drops:
            assert any(abs(t - train["end"]) < 1e-9 for train in trains), (
                f"drop at t={t} is not at a weight publication."
            )
        assert sum(train["n_dropped_step"] for train in trains) == len(drops)

    def test_expired_groups_are_replaced(self):
        sim, policy = _roll_sim(1)
        sim.run()
        # Base admission is capped at what training will ever consume
        # (n_versions * batch_size, see `Simulator._grant_ingress_credits`); GC
        # replacement credits are the one exception, since each one backfills a
        # group already counted once before it was dropped permanently.
        base_cap = sim.cfg.n_versions * sim.cfg.batch_size
        assert sim.n_admitted == base_cap + policy.gc_dropped, (
            "each GC'd group must be refilled with exactly one fresh prompt, "
            "on top of the capped base admission."
        )

    def test_on_publish_cancels_only_cohorts_below_the_window(self):
        policy = RollPolicy()
        st = SimState(batch_size=2, group_size=1, max_staleness=1, train_version=0)
        policy.admit_quota(st)
        reqs = {
            gid: Request(rid=gid, prompt_len=1, target_len=1, v_traj=version, group_id=gid)
            for gid, version in ((0, 0), (1, 0), (2, 2))
        }
        for req in reqs.values():
            policy.on_admit([req], st)
        policy.on_group_complete([reqs[1]], st)

        late = SimState(batch_size=2, group_size=1, max_staleness=1, train_version=1)
        assert policy.on_publish(1, late) == []

        late.train_version = 3
        cancels = policy.on_publish(3, late)
        assert {(c.group_id, c.action) for c in cancels} == {
            (0, CancelAction.REPLACE),
            (1, CancelAction.REPLACE),
        }
        assert policy.gc_dropped == 2
        assert policy.running_prompts == 1
        assert policy.completed_prompts == 0
        assert 2 in {gid for cohort in policy.cohorts.values() for gid in cohort.running}


class TestSlimeBarrier:
    def test_staleness_equals_interval(self):
        cfg = SimConfig(
            batch_size=4,
            staleness_horizon=1,
            n_versions=10,
        )
        lengths = uniform(100, 50, 100, seed=0)
        policy = SlimePolicy(update_weights_interval=2, pause_mode="in_place")
        result = _run_sim(policy, lengths, cfg)
        assert not result.livelocked
        for r in result.consumed:
            assert r.staleness_true() <= 2

    def test_a_straggler_under_admission_alone_is_not_bounded(self):
        """A wide ingress window lets several cohorts stage ahead, so one can be

        admitted at an earlier version and still be mid-generation once a
        later cohort's own consumption has pushed the version forward.
        Nothing here checks a group's generation version at training time,
        so admission concurrency alone does not bound its eventual staleness.
        """
        workload = GroupedWorkload.from_lengths([150], [10000] + [50] * 11)
        # n_versions well past the brief's 8: a 10000-token straggler needs that many
        # training steps to actually finish and register its staleness.
        cfg = SimConfig(
            batch_size=2, staleness_horizon=0, n_versions=40, ingress_window_groups=24, workload=workload
        )
        policy = SlimePolicy(update_weights_interval=1, over_sampling_ratio=4.0, pause_mode="in_place")
        result = _run_sim(policy, None, cfg)

        assert not result.livelocked
        report = result.staleness_report()
        assert report.max > cfg.staleness_horizon, "some cohort must train with a nonzero version gap."
        assert report.strict is False


class TestVerlAsync:
    def test_runs_without_crash(self):
        cfg = SimConfig(batch_size=4, staleness_horizon=1, n_versions=8)
        lengths = uniform(100, 50, 100, seed=0)
        policy = VerlPolicy(staleness_threshold=0.1)
        result = _run_sim(policy, lengths, cfg)
        assert not result.livelocked
        assert result.versions_done == 8

    def test_window_budget_and_sync_carryover(self):
        policy = VerlPolicy(staleness_threshold=2.0)
        st = SimState(batch_size=4, group_size=1, n_instances=2)
        assert policy.admit_quota(st) == 12

        reqs = [
            Request(rid=i, group_id=i, prompt_len=10, target_len=100, v_traj=0)
            for i in range(6)
        ]
        for req in reqs:
            st.inflight.append(req)
            policy.on_admit([req], st)
        assert policy.staleness_samples == 6
        assert policy.admit_quota(st) == 6

        for req in reqs[:4]:
            st.inflight.remove(req)
            assert policy.on_group_complete([req], st) is CompleteAction.KEEP
        assert policy.take_batch(st) == reqs[:4]
        assert policy.staleness_samples == 6
        assert policy.admit_quota(st) == 6

        policy.on_train_version_advance(1, st)
        assert policy.staleness_samples == len(st.inflight) + len(policy.queue) == 2


class TestDeterminism:
    def test_same_seed_same_result(self):
        cfg = SimConfig(batch_size=4, staleness_horizon=2, n_versions=5)
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
            staleness_horizon=2,
            n_versions=8,
            n_instances=2,
        )
        lengths = uniform(100, 50, 100, seed=0)
        result = _run_sim(PSRLPolicy(), lengths, cfg)
        assert not result.livelocked
        assert result.versions_done == 8
        for r in result.consumed:
            assert r.staleness_true() <= cfg.staleness_horizon


class TestVariablePromptLengths:
    def test_variable_prompt_lengths(self):
        prompt_lengths = uniform(100, 100, 1000, seed=7)
        cfg = SimConfig(
            batch_size=4,
            staleness_horizon=2,
            n_versions=8,
            workload=GroupedWorkload.from_lengths(prompt_lengths, uniform(100, 50, 100, seed=0)),
        )
        result = _run_sim(PSRLPolicy(), None, cfg)
        assert not result.livelocked
        assert result.versions_done == 8
        observed_prompt_lens = {r.prompt_len for r in result.consumed}
        assert len(observed_prompt_lens) > 1


def _is_partial_abort(row: dict) -> bool:
    """True when a weight update has just emptied the running queue.

    Keyed on `num_aborts > 0` rather than on `generated > 0`. A weight update
    discards the generated tokens (smg re-dispatches from the original prompt
    instead of continuing the sequence), so a just-aborted waiter is otherwise
    indistinguishable from a freshly admitted one.
    """
    if row["num_running_reqs"] != 0:
        return False
    waiting = row.get("waiting_len") or {}
    if not waiting:
        return False
    return all(info["prefilled"] == 0 for info in waiting.values()) and any(
        info.get("num_aborts", 0) > 0 for info in waiting.values()
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
            staleness_horizon=2,
            ingress_window_groups=3 * 4,
            n_versions=4,
            n_instances=1,
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
        for index, train in enumerate(trains):
            start, end = train["start"], train["end"]
            assert end > start, "Train+sync window should have positive duration."
            during = [r for r in inst_rows if start <= r["t"] < end]
            # Total admission is capped at what training will ever consume (see
            # `Simulator._grant_ingress_credits`), so by the last training step
            # rollout can legitimately have already generated and finished
            # everything it will ever be given -- nothing is left to run during
            # that final window. Checked for every window except the last.
            if index < len(trains) - 1:
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
            staleness_horizon=2,
            ingress_window_groups=3 * 4,
            n_versions=4,
            n_instances=1,
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
        # Total ever admitted is capped at what training will ever consume, so
        # the per-publish top-up stops once the cumulative cap is reached,
        # which can be before the last publish -- track the same
        # remaining-budget bookkeeping the simulator does rather than assuming
        # every non-final publish opens a fresh batch.
        cap = cfg.n_versions * cfg.batch_size
        seed = min((cfg.staleness_horizon + 1) * cfg.batch_size, cap)
        remaining = cap - seed
        for index, train in enumerate(trains):
            start, end = train["start"], train["end"]
            during = [t for t in admit_ts if start < t < end]
            assert not during, "AReaL quota should not open mid-training."
            at_end = [t for t in admit_ts if abs(t - end) < 1e-9]
            is_last = index == len(trains) - 1
            grant = 0 if is_last else min(cfg.batch_size, remaining)
            if not is_last:
                remaining -= grant
            if grant:
                assert at_end, "AReaL quota should open when train_version is published."
            else:
                assert not at_end, (
                    "AReaL quota should not open past the last published version, "
                    "or once the cumulative admission cap is reached."
                )

    def test_psrl_interrupt_keeps_one_batch_per_published_version_but_not_past_the_last(self, tmp_path):
        store = JsonlStore(tmp_path / "logs")
        cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7)
        training_cost = TrainingCostModel(
            FixedTrainingLatency(1.0),
            FixedTrainingMemory(0),
        )
        cfg = SimConfig(
            batch_size=4,
            staleness_horizon=2,
            ingress_window_groups=3 * 4,
            n_versions=4,
            n_instances=1,
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
        assert all(r.staleness_true() <= cfg.staleness_horizon for r in result.consumed)
        # Total ever admitted is capped at what training will ever consume
        # (see `Simulator._grant_ingress_credits`), so nothing is stranded.
        cap = cfg.n_versions * cfg.batch_size
        assert sim.n_admitted == cap
        assert sim.n_admitted == sim.n_consumed

        admit_counts: dict[float, int] = {}
        for row in store.iter("request"):
            if row["event"] == "admit":
                admit_counts[row["t"]] = admit_counts.get(row["t"], 0) + 1
        trains = list(store.iter("train"))
        # The window is seeded with `staleness_horizon + 1` batches up front,
        # clamped to the cap, then tops up by one batch per publish -- except
        # the last publish (nothing left to overlap it with), and except once
        # the cumulative cap is reached, which can happen before the last
        # publish. Track the same remaining-budget bookkeeping the simulator
        # does, rather than assuming every non-final publish gets a full
        # batch's worth.
        seed = min((cfg.staleness_horizon + 1) * cfg.batch_size, cap)
        assert admit_counts.pop(0.0) == seed
        remaining = cap - seed
        for train in trains[:-1]:
            grant = min(cfg.batch_size, remaining)
            remaining -= grant
            if grant:
                assert admit_counts.pop(train["end"]) == grant
            else:
                assert train["end"] not in admit_counts
        assert trains[-1]["end"] not in admit_counts, (
            "The last published version must not admit a fresh batch: nothing "
            "will ever train on it."
        )
        assert not admit_counts, "PSRL should admit only initially and when a version is published."


class TestIngressAdmissionCap:
    """Regression guard for `Simulator._grant_ingress_credits`.

    Before the cap existed, total admission was `(staleness_horizon +
    n_versions) * batch_size`, permanently stranding `staleness_horizon *
    batch_size` requests at the end of every run, independent of run length.
    A deliberately large `staleness_horizon` relative to a small `n_versions`
    makes that gap easy to catch if it ever comes back.
    """

    @pytest.mark.parametrize(
        "policy_factory",
        [PSRLPolicy, ArealPolicy, VerlPolicy, lambda: SlimePolicy(update_weights_interval=1)],
        ids=["psrl", "areal", "verl", "slime"],
    )
    def test_total_admission_never_exceeds_what_training_will_consume(self, policy_factory):
        cfg = SimConfig(
            batch_size=4,
            staleness_horizon=3,
            n_versions=3,
            workload=GroupedWorkload.from_lengths([512], uniform(80, 200, 400, seed=0)),
        )
        sim = Simulator(
            rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
            training_cost=TrainingCostModel(FixedTrainingLatency(0.5), FixedTrainingMemory(0)),
            policy=policy_factory(),
            cfg=cfg,
        )
        result = sim.run()
        assert not result.livelocked
        assert sim._ingress_granted <= cfg.n_versions * cfg.batch_size

    def test_gc_replacement_credit_is_allowed_to_exceed_the_base_cap(self):
        # ROLL's GC replaces a dropped group's credit (unlike the base seed
        # and per-publish top-up), so its total admission is the capped base
        # plus exactly one credit per dropped group -- not just the base cap.
        sim, policy = _roll_sim(1, n_versions=3)
        sim.run()
        base_cap = sim.cfg.n_versions * sim.cfg.batch_size
        assert sim.n_admitted <= base_cap + policy.gc_dropped


class TestLengthCaps:
    def test_clip_length_passthrough(self):
        assert clip_length(900, None) == 900

    def test_clip_length_truncates(self):
        assert clip_length(900, 128) == 128
        assert clip_length(64, 128) == 64

    def test_clip_length_rejects_non_positive_cap(self):
        with pytest.raises(ValueError, match="Expected max_length >= 1"):
            clip_length(10, 0)

    def test_clip_total_length_gives_short_prompts_extra_budget(self):
        """The cap is on the total, so a short prompt can generate past the output cap.

        This is what the serving stack does: PSRL's `max_tokens` fallback is
        `min(response_length, response_length + prompt_length - len(prompt_ids))`.
        The measured run confirms it: a 90-token prompt generated 11174
        tokens against a 10240 "output cap", landing exactly on the 11264 total.
        """
        prompt, output = clip_total_length(90, 20_000, 1024, 10_240)
        assert (prompt, output) == (90, 11_174)
        assert prompt + output == 1024 + 10_240

    def test_clip_total_length_prompt_at_cap_gets_only_the_output_cap(self):
        prompt, output = clip_total_length(2_000, 20_000, 1024, 10_240)
        assert (prompt, output) == (1024, 10_240)

    def test_clip_total_length_leaves_short_samples_alone(self):
        assert clip_total_length(90, 5_000, 1024, 10_240) == (90, 5_000)

    def test_clip_total_length_uncapped_dimensions_pass_through(self):
        assert clip_total_length(9_000, 20_000, None, None) == (9_000, 20_000)
        # Without a prompt cap there is no total to budget against, so the
        # output cap applies on its own.
        assert clip_total_length(90, 20_000, None, 10_240) == (90, 10_240)

    def test_clip_total_length_always_leaves_one_token(self):
        """A prompt filling the whole budget must still generate something."""
        _, output = clip_total_length(1024, 20_000, 1024, 0 + 1)
        assert output >= 1

    def test_simulator_truncates_prompt_and_output(self):
        cfg = SimConfig(
            batch_size=4,
            staleness_horizon=2,
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
        # Capped on the total, so the output allowance is 256 + 80 - prompt_len.
        assert all(r.prompt_len + r.target_len <= 256 + 80 for r in requests)
        assert any(r.target_len == 10 for r in requests)


class TestGrouping:
    def _grouped_cfg(self, **overrides):
        from srl.workload.trace import RequestGroup

        groups = [
            RequestGroup(group_id=i, prompt_len=128, output_lens=(50, 60, 70, 80))
            for i in range(6)
        ]
        kwargs = dict(
            batch_size=2,
            group_size=4,
            staleness_horizon=2,
            n_versions=6,
            workload=GroupedWorkload.from_groups(groups),
            log_telemetry=False,
        )
        kwargs.update(overrides)
        return kwargs["workload"], SimConfig(**kwargs)

    def test_group_identity_reaches_requests(self):
        """Provenance and the simulator's own group key both land on requests."""
        workload, cfg = self._grouped_cfg()
        result = _run_sim(PSRLPolicy(), None, cfg)
        all_requests = result.consumed + result.dropped
        assert all_requests, "Expected at least one admitted request."

        source_ids = {g.group_id for g in workload.groups}
        assert {r.source_group_id for r in all_requests} <= source_ids

        # Each group key must carry exactly one response per group_index.
        by_group: dict[int, set[int]] = {}
        for req in all_requests:
            by_group.setdefault(req.group_id, set()).add(req.group_index)
        for group_id, indices in by_group.items():
            assert indices <= {0, 1, 2, 3}, f"Group {group_id} has indices {indices!r}."

    def test_consumed_batches_hold_only_complete_groups(self):
        """A training batch is whole groups, never a slice across them."""
        from collections import Counter

        _, cfg = self._grouped_cfg()
        result = _run_sim(PSRLPolicy(), None, cfg)
        assert not result.livelocked

        by_version: dict[int, list] = {}
        for req in result.consumed:
            by_version.setdefault(req.consume_version, []).append(req)

        assert by_version, "Expected at least one consumed batch."
        for version, batch in by_version.items():
            counts = Counter(r.group_id for r in batch)
            assert len(counts) == cfg.batch_size, (
                f"Version {version} consumed {len(counts)} groups, expected {cfg.batch_size}."
            )
            assert set(counts.values()) == {cfg.group_size}, (
                f"Version {version} consumed partial groups: {dict(counts)!r}."
            )

    def test_group_keys_are_unique_across_workload_wraparound(self):
        """A workload shorter than the run must not reuse a group key.

        The workload repeats, so its own ids recur. Reusing them would merge two
        distinct groups into one oversized staleness entry.
        """
        _, cfg = self._grouped_cfg(n_versions=6, batch_size=2)
        result = _run_sim(PSRLPolicy(), None, cfg)
        consumed_groups = {r.group_id for r in result.consumed}
        assert len(consumed_groups) == cfg.n_versions * cfg.batch_size, (
            f"Expected every consumed group to be distinct, got {len(consumed_groups)} "
            f"for {cfg.n_versions * cfg.batch_size} consumed groups."
        )

    def test_config_rejects_group_size_disagreeing_with_workload(self):
        from srl.workload.trace import RequestGroup

        workload = GroupedWorkload.from_groups(
            [RequestGroup(group_id=0, prompt_len=8, output_lens=(10, 20))]
        )
        with pytest.raises(ValueError, match="does not match the workload"):
            SimConfig(batch_size=2, group_size=4, workload=workload)
