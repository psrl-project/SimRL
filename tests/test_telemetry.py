"""Tests that the simulator's JsonlStore telemetry is rich, committed-only, and consistent."""

from __future__ import annotations

from srl.component.rollout.engine import Request
from srl.cost.rollout import UnifiedRoofline
from srl.policy.base import Cancel, CancelAction, CompleteAction, SimState, StalenessPolicy
from srl.policy.psrl import PSRLPolicy
from srl.policy.verl import VerlPolicy
from srl.simulator import SimConfig, Simulator
from srl.telemetry.store import JsonlStore
from srl.workload.distributions import uniform
from srl.workload.grouped import GroupedWorkload


def _run_with_store(tmp_path, n_instances=2, n_versions=6):
    store = JsonlStore(tmp_path / "logs")
    cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7)
    cfg = SimConfig(
        batch_size=4,
        staleness_horizon=2,
        n_versions=n_versions,
        n_instances=n_instances,
        workload=GroupedWorkload.from_lengths([512], uniform(100, 50, 150, seed=0)),
    )
    sim = Simulator(
        rollout_cost=cost,
        policy=PSRLPolicy(),
        cfg=cfg,
        store=store,
    )
    result = sim.run()
    store.close()
    return store, result, cfg


class TestNoStoreIsNoOp:
    def test_runs_identically_without_a_store(self):
        cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7)
        cfg = SimConfig(
            batch_size=4,
            staleness_horizon=2,
            n_versions=5,
            workload=GroupedWorkload.from_lengths([512], uniform(50, 50, 100, seed=0)),
        )
        sim = Simulator(rollout_cost=cost, policy=PSRLPolicy(), cfg=cfg)
        result = sim.run()
        assert not result.livelocked
        assert result.versions_done == 5


class TestLogTelemetrySwitch:
    def test_log_telemetry_false_disables_writes_even_with_a_store(self, tmp_path):
        """`cfg.log_telemetry=False` must win over passing a real store in."""
        store = JsonlStore(tmp_path / "logs")
        cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7)
        cfg = SimConfig(
            batch_size=4,
            staleness_horizon=2,
            n_versions=5,
            log_telemetry=False,
            workload=GroupedWorkload.from_lengths([512], uniform(50, 50, 100, seed=0)),
        )
        sim = Simulator(
            rollout_cost=cost,
            policy=PSRLPolicy(),
            cfg=cfg,
            store=store,
        )
        result = sim.run()
        store.close()

        assert not sim.telemetry.enabled
        assert not result.livelocked
        assert store.list_streams() == []

    def test_log_telemetry_true_is_the_default_and_writes_files(self, tmp_path):
        store = JsonlStore(tmp_path / "logs")
        cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7)
        cfg = SimConfig(
            batch_size=4,
            staleness_horizon=2,
            n_versions=5,
            workload=GroupedWorkload.from_lengths([512], uniform(50, 50, 100, seed=0)),
        )
        assert cfg.log_telemetry is True
        sim = Simulator(
            rollout_cost=cost,
            policy=PSRLPolicy(),
            cfg=cfg,
            store=store,
        )
        sim.run()
        store.close()

        assert sim.telemetry.enabled
        assert store.list_streams() != []
        assert list(store.iter("train"))


class TestInstanceStream:
    def test_nonempty_per_instance_and_monotonic_time(self, tmp_path):
        store, result, cfg = _run_with_store(tmp_path)
        assert not result.livelocked
        for iid in range(cfg.n_instances):
            rows = list(store.iter(f"instance/{iid}"))
            assert rows, f"instance {iid} has no telemetry rows"
            ts = [r["t"] for r in rows]
            assert ts == sorted(ts), "instance samples must be in non-decreasing time order"
            for r in rows:
                assert r["num_running_reqs"] >= 0
                assert r["num_waiting_reqs"] >= 0
                assert 0.0 <= r["kv_cache_usage"] <= 1.0 + 1e-9
                assert r["kv_blocks_used"] >= 0
                assert r["total_ctx"] >= 0
                assert isinstance(r["running_len"], dict)
                assert isinstance(r["waiting_len"], dict)
                assert len(r["running_len"]) == r["num_running_reqs"]
                assert len(r["waiting_len"]) == r["num_waiting_reqs"]

    def test_no_exact_duplicate_consecutive_samples(self, tmp_path):
        store, _, cfg = _run_with_store(tmp_path)
        for iid in range(cfg.n_instances):
            rows = list(store.iter(f"instance/{iid}"))
            keys = [(r["t"], r["num_running_reqs"], r["num_waiting_reqs"], r["kv_blocks_used"]) for r in rows]
            for a, b in zip(keys, keys[1:]):
                assert a != b, "consecutive instance samples must differ"


class TestGatewayStream:
    def test_queue_depth_is_reported_per_version(self, tmp_path):
        store, _, _ = _run_with_store(tmp_path)
        rows = list(store.iter("gateway"))
        assert rows
        for row in rows:
            assert isinstance(row["queue_by_version"], dict)
            assert sum(row["queue_by_version"].values()) == row["queue_len"]


class TestRolloutStepStream:
    def test_phases_and_positive_dt(self, tmp_path):
        store, _, cfg = _run_with_store(tmp_path)
        rows = list(store.iter("rollout_step"))
        assert rows
        seen_instances = set()
        for r in rows:
            assert r["phase"] in ("mixed", "decode")
            assert r["dt"] > 0
            assert r["steps"] >= 1
            assert r["n_tokens"] == r["n_prefill"] + r["n_decode"]
            seen_instances.add(r["instance_id"])
        assert seen_instances == set(range(cfg.n_instances))

    def test_times_non_decreasing_per_instance(self, tmp_path):
        store, _, cfg = _run_with_store(tmp_path)
        rows = list(store.iter("rollout_step"))
        by_instance: dict[int, list[float]] = {}
        for r in rows:
            by_instance.setdefault(r["instance_id"], []).append(r["t"])
        for iid, ts in by_instance.items():
            assert ts == sorted(ts), f"rollout_step times for instance {iid} are out of order"


class TestTrainStream:
    def test_one_row_per_version_with_expected_fields(self, tmp_path):
        n_versions = 6
        store, result, cfg = _run_with_store(tmp_path, n_versions=n_versions)
        rows = list(store.iter("train"))
        assert len(rows) == n_versions
        for i, r in enumerate(rows, start=1):
            assert r["version"] == i
            assert r["end"] >= r["start"]
            assert r["dt"] == r["end"] - r["start"]
            assert r["n_groups"] == cfg.batch_size
            assert r["group_size"] == cfg.group_size
            assert r["n_reqs"] == cfg.responses_per_step
            assert r["total_tokens"] > 0
            assert r["train_time"] > 0
            assert r["peak_memory_bytes"] == 0
            assert r["training_tp"] == 1
            assert r["training_pp"] == 1
            assert r["training_dp"] == 1
            assert r["micro_batch_size"] == 1
            assert r["schedule"] == "1f1b"
            assert r["n_inflight"] >= 0
            assert r["n_ready"] >= 0
        assert result.train_peak_memory_bytes == [0] * n_versions

    def test_train_end_matches_final_sim_time(self, tmp_path):
        store, result, _ = _run_with_store(tmp_path)
        rows = list(store.iter("train"))
        assert abs(rows[-1]["end"] - result.sim_time) < 1e-9


class TestRequestStream:
    def test_admit_complete_consume_events_present(self, tmp_path):
        store, result, _ = _run_with_store(tmp_path)
        rows = list(store.iter("request"))
        events = {r["event"] for r in rows}
        assert "admit" in events
        assert "complete" in events
        assert "consume" in events

    def test_consume_count_matches_result(self, tmp_path):
        store, result, _ = _run_with_store(tmp_path)
        consume_rows = [r for r in store.iter("request") if r["event"] == "consume"]
        assert len(consume_rows) == result.n_consumed

    def test_consume_rows_carry_staleness_inputs(self, tmp_path):
        store, _, _ = _run_with_store(tmp_path)
        consume_rows = [r for r in store.iter("request") if r["event"] == "consume"]
        for r in consume_rows:
            assert r["consume_version"] is not None
            assert r["v_min"] is not None
            assert r["v_max"] is not None
            assert r["v_min"] <= r["v_max"]
            assert r["staleness_true"] == r["consume_version"] - r["v_min"]
            assert r["staleness_reported"] == r["consume_version"] - r["v_max"]


def _bare_sim(tmp_path, policy):
    store = JsonlStore(tmp_path / "logs")
    cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7)
    cfg = SimConfig(batch_size=1, n_versions=1, workload=GroupedWorkload.from_lengths([10], [10]))
    sim = Simulator(rollout_cost=cost, policy=policy, cfg=cfg, store=store)
    return sim, store


def _drop_rows(store):
    return [r for r in store.iter("request") if r["event"] == "drop"]


class TestDropReasonTelemetry:
    def test_l2_cancel_for_a_group_still_generating(self, tmp_path):
        sim, store = _bare_sim(tmp_path, PSRLPolicy())
        req = Request(rid=0, prompt_len=1, target_len=10, v_traj=0, group_id=0)
        sim.inflight.append(req)

        sim._apply_cancels([Cancel(0, CancelAction.DROP)])
        store.close()

        rows = _drop_rows(store)
        assert len(rows) == 1
        assert rows[0]["drop_reason"] == "l2_cancel"
        assert rows[0]["retry"] is False

    def test_l3_stale_for_a_group_already_in_the_ready_pool(self, tmp_path):
        sim, store = _bare_sim(tmp_path, PSRLPolicy())
        req = Request(rid=0, prompt_len=1, target_len=10, v_traj=0, group_id=0)
        sim.ready.append(req)

        sim._apply_cancels([Cancel(0, CancelAction.REPLACE)])
        store.close()

        rows = _drop_rows(store)
        assert len(rows) == 1
        assert rows[0]["drop_reason"] == "l3_stale"
        assert rows[0]["retry"] is False

    def test_span_cap_and_retry_for_a_retry_from_scratch_cancel_with_an_explicit_reason(self, tmp_path):
        sim, store = _bare_sim(tmp_path, PSRLPolicy())
        req = Request(rid=0, prompt_len=1, target_len=10, v_traj=0, group_id=0)
        sim.inflight.append(req)

        sim._apply_cancels([Cancel(0, CancelAction.RETRY_FROM_SCRATCH, "span_cap")])
        store.close()

        rows = _drop_rows(store)
        assert len(rows) == 1
        assert rows[0]["drop_reason"] == "span_cap"
        assert rows[0]["retry"] is True

    def test_retry_from_scratch_without_an_explicit_reason_is_not_mislabeled_span_cap(self, tmp_path):
        """A Cancel's action alone must not imply span_cap.

        A future (or already-configurable: prime-rl's own `action` knob)
        RETRY_FROM_SCRATCH user that is not VersionSpanCap still gets the
        same running/ready inference as any other cancel, only `retry`
        distinguishes it.
        """
        sim, store = _bare_sim(tmp_path, PSRLPolicy())
        req = Request(rid=0, prompt_len=1, target_len=10, v_traj=0, group_id=0)
        sim.inflight.append(req)

        sim._apply_cancels([Cancel(0, CancelAction.RETRY_FROM_SCRATCH)])
        store.close()

        rows = _drop_rows(store)
        assert len(rows) == 1
        assert rows[0]["drop_reason"] == "l2_cancel"
        assert rows[0]["retry"] is True

    def test_l3_stale_is_the_default_for_an_undistinguished_on_group_complete_drop(self, tmp_path):
        class _AlwaysDropPolicy(StalenessPolicy):
            name = "always_drop"

            def admit_quota(self, st: SimState) -> int:
                return 0

            def on_group_complete(self, group, st) -> CompleteAction:
                return CompleteAction.DROP

            def peek_batch(self, st: SimState):
                return None

        sim, store = _bare_sim(tmp_path, _AlwaysDropPolicy())
        req = Request(rid=0, prompt_len=1, target_len=10, v_traj=0, group_id=0)
        req.complete_time = 0.0

        sim._collect([req])
        store.close()

        rows = _drop_rows(store)
        assert len(rows) == 1
        assert rows[0]["drop_reason"] == "l3_stale"

    def test_queue_overflow_for_a_verl_legacy_handoff_queue_drop(self, tmp_path):
        policy = VerlPolicy(staleness_threshold=0.0, require_batches=1)
        sim, store = _bare_sim(tmp_path, policy)
        stale_occupant = [Request(rid=99, prompt_len=1, target_len=10, v_traj=0, group_id=99)]
        policy._overage.queue.append(stale_occupant)

        req = Request(rid=0, prompt_len=1, target_len=10, v_traj=0, group_id=0)
        req.complete_time = 0.0
        sim._collect([req])
        store.close()

        rows = _drop_rows(store)
        assert len(rows) == 1
        assert rows[0]["drop_reason"] == "queue_overflow"
