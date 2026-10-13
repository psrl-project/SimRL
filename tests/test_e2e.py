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
from dryrun.simulator import SimConfig, Simulator, clip_length, clip_total_length
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
        # Capped on the total, so the output allowance is 256 + 80 - prompt_len.
        assert all(r.prompt_len + r.target_len <= 256 + 80 for r in requests)
        assert any(r.target_len == 10 for r in requests)


class TestGrouping:
    def _grouped_cfg(self, **overrides):
        from dryrun.workload.trace import RequestGroup

        groups = [
            RequestGroup(group_id=i, prompt_len=128, output_lens=(50, 60, 70, 80))
            for i in range(6)
        ]
        kwargs = dict(
            batch_size=2,
            group_size=4,
            max_staleness=2,
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
        from dryrun.workload.trace import RequestGroup

        workload = GroupedWorkload.from_groups(
            [RequestGroup(group_id=0, prompt_len=8, output_lens=(10, 20))]
        )
        with pytest.raises(ValueError, match="does not match the workload"):
            SimConfig(batch_size=2, group_size=4, workload=workload)
