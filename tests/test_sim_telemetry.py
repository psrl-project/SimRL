"""Unit tests for the `SimTelemetry` class itself, isolated from `Simulator`."""

from __future__ import annotations

from dryrun.component.rollout.engine import NativeInstance, Request
from dryrun.cost.rollout import UnifiedRoofline
from dryrun.telemetry.store import JsonlStore
from dryrun.telemetry.writer import SimTelemetry


def _make_instance(kv_blocks: int = 1000) -> NativeInstance:
    cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7)
    return NativeInstance(cost, token_budget=8192, kv_blocks=kv_blocks, block_size=16, instance_id=0)


def _make_request(rid: int = 0) -> Request:
    return Request(rid=rid, prompt_len=512, target_len=50, v_traj=0, admit_time=0.0)


class TestDisabledIsNoOp:
    def test_no_store_writes_nothing(self, tmp_path):
        store = JsonlStore(tmp_path / "logs")
        telemetry = SimTelemetry(None)
        inst = _make_instance()
        telemetry.emit_instance(inst)
        telemetry.flush_rollout_step(inst)
        telemetry.write_request("admit", 0.0, _make_request())
        telemetry.write_train(
            t=1.0, start=0.0, end=1.0, version=1, batch_size=1, n_reqs=1,
            total_tokens=10, train_time=1.0, peak_memory_bytes=123,
            training_tp=1, training_pp=1, training_dp=1, training_cp=1,
            micro_batch_size=1, schedule="1f1b", activation_recompute="none",
            sync_time=0.0, recompute_time=0.0,
            n_inflight=0, n_ready=0, n_dropped_step=0,
        )
        assert not telemetry.enabled
        assert store.list_streams() == []  # store was never touched.
        assert not (tmp_path / "logs").exists() or list((tmp_path / "logs").iterdir()) == []

    def test_enabled_false_disables_even_with_a_real_store(self, tmp_path):
        store = JsonlStore(tmp_path / "logs")
        telemetry = SimTelemetry(store, enabled=False)
        assert not telemetry.enabled
        telemetry.write_request("admit", 0.0, _make_request())
        telemetry.write_train(
            t=1.0, start=0.0, end=1.0, version=1, batch_size=1, n_reqs=1,
            total_tokens=10, train_time=1.0, peak_memory_bytes=123,
            training_tp=1, training_pp=1, training_dp=1, training_cp=1,
            micro_batch_size=1, schedule="1f1b", activation_recompute="none",
            sync_time=0.0, recompute_time=0.0,
            n_inflight=0, n_ready=0, n_dropped_step=0,
        )
        assert list(store.iter("request")) == []
        assert list(store.iter("train")) == []


class TestEnabledWritesRichRecords:
    def test_emit_instance_writes_expected_schema(self, tmp_path):
        store = JsonlStore(tmp_path / "logs")
        telemetry = SimTelemetry(store)
        inst = _make_instance()
        req = _make_request()
        inst.add_request(req, 0.0)
        telemetry.emit_instance(inst)
        rows = list(store.iter("instance/0"))
        assert len(rows) == 1
        row = rows[0]
        assert row["instance_id"] == 0
        assert row["num_waiting_reqs"] == 1
        assert row["num_running_reqs"] == 0
        assert row["kv_cache_usage"] == 0.0
        assert row["total_ctx"] == 0
        assert row["running_len"] == {}
        # `num_aborts` is part of the snapshot because a weight update discards
        # generated tokens, leaving an interrupted waiter otherwise identical to
        # a freshly admitted one.
        assert row["waiting_len"] == {
            "0": {"generated": 0, "target_len": 50, "ctx": 512, "prefilled": 0, "num_aborts": 0}
        }

    def test_emit_instance_deduplicates_identical_samples(self, tmp_path):
        store = JsonlStore(tmp_path / "logs")
        telemetry = SimTelemetry(store)
        inst = _make_instance()
        telemetry.emit_instance(inst)
        telemetry.emit_instance(inst)  # nothing changed -> should be skipped.
        telemetry.emit_instance(inst)
        assert len(list(store.iter("instance/0"))) == 1

    def test_emit_instance_all_covers_every_instance(self, tmp_path):
        store = JsonlStore(tmp_path / "logs")
        telemetry = SimTelemetry(store)
        instances = [_make_instance(), _make_instance()]
        instances[1].instance_id = 1
        telemetry.emit_instance_all(instances)
        assert len(list(store.iter("instance/0"))) == 1
        assert len(list(store.iter("instance/1"))) == 1

    def test_flush_rollout_step_only_emits_new_stats_since_last_flush(self, tmp_path):
        store = JsonlStore(tmp_path / "logs")
        telemetry = SimTelemetry(store)
        inst = _make_instance()
        req = _make_request()
        inst.add_request(req, 0.0)
        inst.advance_one_event(0)  # prefill step -> appends one StepStat.

        telemetry.flush_rollout_step(inst)
        assert len(list(store.iter("rollout_step"))) == 1

        telemetry.flush_rollout_step(inst)  # nothing new -> still 1.
        assert len(list(store.iter("rollout_step"))) == 1

        inst.advance_one_event(0)  # a decode step -> appends another StepStat.
        telemetry.flush_rollout_step(inst)
        assert len(list(store.iter("rollout_step"))) == 2

    def test_write_request_round_trips_request_fields(self, tmp_path):
        store = JsonlStore(tmp_path / "logs")
        telemetry = SimTelemetry(store)
        req = _make_request(rid=7)
        req.instance_id = 0
        telemetry.write_request("admit", 1.5, req)
        rows = list(store.iter("request"))
        assert len(rows) == 1
        row = rows[0]
        assert row["t"] == 1.5
        assert row["event"] == "admit"
        assert row["rid"] == 7
        assert row["group_id"] == 0
        assert row["prompt_len"] == 512
        assert row["target_len"] == 50

    def test_write_train_computes_dt_from_start_and_end(self, tmp_path):
        store = JsonlStore(tmp_path / "logs")
        telemetry = SimTelemetry(store)
        telemetry.write_train(
            t=3.0, start=2.0, end=3.0, version=1, batch_size=4, n_reqs=4,
            total_tokens=100, train_time=1.0, peak_memory_bytes=456,
            training_tp=2, training_pp=1, training_dp=2, training_cp=1,
            micro_batch_size=1, schedule="1f1b", activation_recompute="none",
            sync_time=0.0, recompute_time=0.0,
            n_inflight=5, n_ready=0, n_dropped_step=0,
        )
        row = next(iter(store.iter("train")))
        assert row["dt"] == 1.0
        assert row["version"] == 1
        assert row["total_tokens"] == 100
        assert row["peak_memory_bytes"] == 456
        assert row["training_tp"] == 2
        assert row["training_dp"] == 2

    def test_close_delegates_to_store(self, tmp_path):
        store = JsonlStore(tmp_path / "logs")
        telemetry = SimTelemetry(store)
        telemetry.write_request("admit", 0.0, _make_request())
        telemetry.close()
        # A closed store's write handles are gone, but iter() re-opens for reads.
        assert len(list(store.iter("request"))) == 1
