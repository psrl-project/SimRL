"""Tests that a PSRL log directory converts into simulator-shaped telemetry streams."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from dryrun.telemetry.psrl_log import PsrlLogLayout, convert_psrl_logs
from dryrun.telemetry.store import JsonlStore
from dryrun.visualizer.plots import SimPlotter

# The fixture writes UTC into the JSON sinks and this offset into the text logs,
# exactly as PSRL does, so the converter has a real offset to recover.
_OFFSET_HOURS = 8
_ORIGIN = datetime(2026, 9, 3, 17, 0, 0, tzinfo=timezone.utc)

# One block per step: recompute, then the optimizer step, then the weight push.
_RECOMPUTE_S = 30.0
_TRAIN_S = 110.0
_SYNC_S = 0.5
_STEP_PERIOD_S = 300.0
_N_STEPS = 3
_N_SAMPLES = 1000


def _utc(offset_s: float) -> str:
    return (_ORIGIN + timedelta(seconds=offset_s)).isoformat().replace("+00:00", "+00:00")


def _local_line(offset_s: float, message: str) -> str:
    """A `ray_logger`-style line stamped in local time rather than UTC."""
    stamp = _ORIGIN + timedelta(seconds=offset_s, hours=_OFFSET_HOURS)
    return f"{stamp.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - ray_logger.py - 68 - {message}\n"


def _write_stats(path, instance_id: int, busy: bool) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i in range(_N_SAMPLES):
            f.write(
                json.dumps(
                    {
                        "ts": _utc(i),
                        "total_elapsed_time": 400.0 + i,
                        "model_version": 0,
                        "num_running_reqs": (i % 64) if busy else 0,
                        "num_waiting_reqs": 0,
                        "kv_cache_usage": (i % 64) / 64.0 if busy else 0.0,
                        "generation_throughput": 0.0,
                        "avg_ttft": None,
                        "avg_itl": None,
                    }
                )
                + "\n"
            )


def _step_offsets(step: int) -> tuple[float, float, float, float]:
    """`(recompute_start, train_begin, train_end, sync_end)` for one step."""
    start = 100.0 + step * _STEP_PERIOD_S
    train_begin = start + _RECOMPUTE_S
    train_end = train_begin + _TRAIN_S
    return start, train_begin, train_end, train_end + _SYNC_S


def _build_log_dir(tmp_path, n_ranks: int = 2, rank_skew_s: float = 0.2):
    """A miniature PSRL log directory: 2 rollout replicas, 1 paused validate one."""
    log_dir = tmp_path / "psrl_log"
    log_dir.mkdir()

    for instance_id, busy in ((0, True), (2, True), (1, False)):
        _write_stats(log_dir / f"stats_r{instance_id}_dp0.jsonl", instance_id, busy)
        role = "rollout" if busy else "validate"
        (log_dir / f"DPLBStatCollector_{role}_I{instance_id}.log").write_text(
            "2026-09-04 01:06:57,197 - stats_collector.py - 360 - Engine ready\n", encoding="utf-8"
        )

    with (log_dir / "smg_routing_status.jsonl").open("w", encoding="utf-8") as f:
        for i in range(_N_SAMPLES):
            f.write(
                json.dumps(
                    {
                        "ts": _utc(i),
                        "queue_len": i % 17,
                        "pending_request_num": i % 17,
                        "running_request_num": 128,
                    }
                )
                + "\n"
            )

    phase_lines = []
    for step in range(_N_STEPS):
        start, _, _, _ = _step_offsets(step)
        phase_lines.append(_local_line(start, "[SWITCH] switch_to_trainer_mode called: foo=True"))
    (log_dir / "MainRayTrainer.log").write_text("".join(phase_lines), encoding="utf-8")

    # Ranks are deliberately skewed, so the converter has to union them rather
    # than trust one rank.
    for rank in range(n_ranks):
        lines = []
        for step in range(_N_STEPS):
            _, train_begin, train_end, sync_end = _step_offsets(step)
            skew = rank * rank_skew_s
            lines.append(_local_line(train_begin + skew, "[Begin Event] TRAIN - Train actor"))
            lines.append(_local_line(train_end + skew, "[End Event] TRAIN - Train actor - Time taken: 1.0 seconds"))
            lines.append(_local_line(sync_end + skew, "[End Event] PUSH - Push model - Time taken: 0.5 seconds"))
        (log_dir / f"TrainWorker_R{rank}.log").write_text("".join(lines), encoding="utf-8")

    return log_dir


def _convert(tmp_path, log_dir, **kwargs):
    store = JsonlStore(tmp_path / "logs", overwrite=True)
    summary = convert_psrl_logs(log_dir, store, **kwargs)
    store.close()
    return summary, JsonlStore(tmp_path / "logs")


class TestRoleSelection:
    def test_validate_replicas_are_dropped(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        summary, store = _convert(tmp_path, log_dir)
        assert summary.instance_ids == [0, 2]
        assert summary.dropped_instance_ids == [1]
        assert "instance/1" not in store.list_streams()

    def test_explicit_instances_override_the_role(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        summary, _ = _convert(tmp_path, log_dir, instance_ids=[0])
        assert summary.instance_ids == [0]
        assert summary.dropped_instance_ids == [1, 2]

    def test_selecting_the_validate_role_picks_the_other_replicas(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        summary, _ = _convert(tmp_path, log_dir, layout=PsrlLogLayout(role="validate"))
        assert summary.instance_ids == [1]

    def test_missing_role_logs_raise_rather_than_guess(self, tmp_path):
        """An idle rollout replica must not be mistaken for a paused validate one."""
        log_dir = _build_log_dir(tmp_path)
        for path in log_dir.glob("DPLBStatCollector_*.log"):
            path.unlink()
        with pytest.raises(ValueError, match="cannot be told apart"):
            _convert(tmp_path, log_dir)

    def test_naming_an_absent_replica_raises(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        with pytest.raises(ValueError, match=r"Replicas \[7\]"):
            _convert(tmp_path, log_dir, instance_ids=[7])


class TestClockAlignment:
    def test_offset_is_recovered_from_the_two_clocks(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        summary, _ = _convert(tmp_path, log_dir)
        assert summary.utc_offset_hours == pytest.approx(_OFFSET_HOURS)

    def test_an_explicit_offset_is_used_verbatim(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        summary, _ = _convert(tmp_path, log_dir, utc_offset_hours=0.0)
        assert summary.utc_offset_hours == 0.0

    def test_a_wrong_offset_is_reported_as_out_of_span(self, tmp_path, caplog):
        log_dir = _build_log_dir(tmp_path)
        with caplog.at_level("WARNING", logger="dryrun"):
            _convert(tmp_path, log_dir, utc_offset_hours=0.0)
        assert "offset is probably wrong" in caplog.text

    def test_instance_time_starts_at_zero(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        _, store = _convert(tmp_path, log_dir)
        assert next(iter(store.iter("instance/0")))["t"] == pytest.approx(0.0)


class TestTrainPhases:
    def test_phases_are_contiguous_and_sum_to_the_block(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        _, store = _convert(tmp_path, log_dir)
        rows = list(store.iter("train"))
        assert len(rows) == _N_STEPS
        for row in rows:
            parts = row["recompute_time"] + row["train_time"] + row["sync_time"]
            assert parts == pytest.approx(row["end"] - row["start"])

    def test_ranks_are_unioned_not_sampled(self, tmp_path):
        """The block must cover every rank's busy time, not just rank 0's."""
        skew = 5.0
        log_dir = _build_log_dir(tmp_path, n_ranks=3, rank_skew_s=skew)
        _, store = _convert(tmp_path, log_dir)
        row = next(iter(store.iter("train")))
        # Earliest begin, latest end: recompute shrinks by nothing, the block
        # grows by the skew of the last rank.
        assert row["recompute_time"] == pytest.approx(_RECOMPUTE_S)
        assert row["train_time"] == pytest.approx(_TRAIN_S + 2 * skew)

    def test_phase_durations_match_the_fixture(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        summary, _ = _convert(tmp_path, log_dir)
        assert summary.mean_recompute_s == pytest.approx(_RECOMPUTE_S)
        assert summary.mean_sync_s == pytest.approx(_SYNC_S)

    def test_disagreeing_ranks_raise(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        path = log_dir / "TrainWorker_R1.log"
        path.write_text(
            "".join(path.read_text(encoding="utf-8").splitlines(keepends=True)[:-1]), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="disagree on how many training steps"):
            _convert(tmp_path, log_dir)

    def test_a_trailing_incomplete_block_is_dropped(self, tmp_path):
        """A run killed mid-step leaves a recompute marker with no step behind it."""
        log_dir = _build_log_dir(tmp_path)
        phase_path = log_dir / "MainRayTrainer.log"
        start, _, _, _ = _step_offsets(_N_STEPS)
        with phase_path.open("a", encoding="utf-8") as f:
            f.write(_local_line(start, "[SWITCH] switch_to_trainer_mode called: foo=True"))
        summary, _ = _convert(tmp_path, log_dir)
        assert summary.n_train_steps == _N_STEPS

    def test_missing_train_worker_logs_raise(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        for path in log_dir.glob("TrainWorker_R*.log"):
            path.unlink()
        with pytest.raises(ValueError, match="train worker logs"):
            _convert(tmp_path, log_dir)


class TestGatewayStream:
    def test_gateway_rows_are_converted(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        summary, store = _convert(tmp_path, log_dir)
        assert summary.n_gateway_rows == _N_SAMPLES
        assert next(iter(store.iter("gateway")))["queue_len"] == 0

    def test_a_missing_gateway_file_is_not_fatal(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        (log_dir / "smg_routing_status.jsonl").unlink()
        summary, store = _convert(tmp_path, log_dir)
        assert summary.n_gateway_rows == 0
        assert list(store.iter("gateway")) == []


class TestRendersWithTheSharedPlotter:
    def test_converted_logs_render_the_instance_load_figure(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        _, store = _convert(tmp_path, log_dir)
        plotter = SimPlotter(store, tmp_path / "plots", time_label="Wall-clock Time (s)")
        path = plotter.plot_instance_load()
        assert path.exists() and path.stat().st_size > 0

    def test_the_field_names_are_the_ones_the_plotter_reads(self, tmp_path):
        """The whole point of the conversion: no plotter branching on provenance."""
        log_dir = _build_log_dir(tmp_path)
        _, store = _convert(tmp_path, log_dir)
        row = next(iter(store.iter("instance/0")))
        assert {"t", "instance_id", "num_running_reqs", "num_waiting_reqs", "kv_cache_usage"} <= set(row)


class TestBadInput:
    def test_a_missing_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No PSRL log directory"):
            _convert(tmp_path, tmp_path / "nope")

    def test_a_directory_without_stats_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No stats files matching"):
            _convert(tmp_path, empty)
