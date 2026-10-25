"""Tests that a verl separate_async run's telemetry directory converts into
simulator-shaped telemetry streams and a workload trace."""

from __future__ import annotations

import json

import pytest

from srl.telemetry.store import JsonlStore
from srl.telemetry.verl_log import convert_verl_logs, extract_workload_trace
from srl.workload.trace import read_group_trace

_N_SAMPLES = 50


def _write_stats(path, busy: bool) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i in range(_N_SAMPLES):
            f.write(
                json.dumps(
                    {
                        "ts": 1000.0 + i,
                        "num_running_reqs": (i % 16) if busy else 0,
                        "num_waiting_reqs": 0,
                        "kv_cache_usage": (i % 16) / 16.0 if busy else 0.0,
                    }
                )
                + "\n"
            )


def _write_requests(path, start_id: int, n: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "request_id": f"req-{start_id + i:04d}",
                        "prompt_tokens": 100,
                        "response_tokens": 200 + i,
                        "stop_reason": "stop",
                        "num_preempted": 0,
                        "ts": 1000.0 + i,
                    }
                )
                + "\n"
            )


def _build_log_dir(tmp_path):
    log_dir = tmp_path / "verl_telemetry"
    log_dir.mkdir()
    _write_stats(log_dir / "stats_r0.jsonl", busy=True)
    _write_stats(log_dir / "stats_r1.jsonl", busy=True)
    _write_requests(log_dir / "request_trace_pid111.jsonl", start_id=0, n=8)
    _write_requests(log_dir / "request_trace_pid222.jsonl", start_id=8, n=8)
    return log_dir


def _convert(tmp_path, log_dir, **kwargs):
    store = JsonlStore(tmp_path / "logs", overwrite=True)
    summary = convert_verl_logs(log_dir, store, **kwargs)
    store.close()
    return summary, JsonlStore(tmp_path / "logs")


class TestInstanceStreams:
    def test_all_replicas_are_converted(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        summary, _ = _convert(tmp_path, log_dir)
        assert summary.instance_ids == [0, 1]
        assert summary.n_instance_rows == 2 * _N_SAMPLES

    def test_instance_time_starts_at_zero(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        _, store = _convert(tmp_path, log_dir)
        assert next(iter(store.iter("instance/0")))["t"] == pytest.approx(0.0)

    def test_the_field_names_are_the_ones_the_plotter_reads(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        _, store = _convert(tmp_path, log_dir)
        row = next(iter(store.iter("instance/0")))
        assert {"t", "instance_id", "num_running_reqs", "num_waiting_reqs", "kv_cache_usage"} <= set(row)


class TestRequestCounting:
    def test_requests_across_processes_are_summed(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        summary, _ = _convert(tmp_path, log_dir)
        assert summary.n_request_rows == 16


class TestBadInput:
    def test_a_missing_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No verl telemetry directory"):
            _convert(tmp_path, tmp_path / "nope")

    def test_a_directory_without_stats_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No stats files matching"):
            _convert(tmp_path, empty)


class TestWorkloadTraceExtraction:
    def test_groups_are_chunked_by_group_size(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        out = tmp_path / "trace.jsonl"
        extract_workload_trace(log_dir, out, group_size=4)
        groups = read_group_trace(out)
        assert len(groups) == 4
        assert len(groups[0].output_lens) == 4

    def test_a_count_not_divisible_by_group_size_raises(self, tmp_path):
        log_dir = _build_log_dir(tmp_path)
        with pytest.raises(ValueError, match="not a whole number of groups"):
            extract_workload_trace(log_dir, tmp_path / "trace.jsonl", group_size=5)

    def test_missing_trace_files_raise(self, tmp_path):
        log_dir = tmp_path / "no_trace"
        log_dir.mkdir()
        (log_dir / "stats_r0.jsonl").write_text(
            '{"ts": 0, "num_running_reqs": 0, "num_waiting_reqs": 0, "kv_cache_usage": 0.0}\n'
        )
        with pytest.raises(ValueError, match="No request trace files"):
            extract_workload_trace(log_dir, tmp_path / "trace.jsonl", group_size=4)
