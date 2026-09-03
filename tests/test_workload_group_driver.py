"""Tests for the group collection driver's prompt selection and CLI wiring."""

from __future__ import annotations

import pytest

from dryrun.cli.workload import build_parser, cmd_check_shard, cmd_verify_collection
from dryrun.workload.inference import _partition_group_results
from dryrun.workload.sharding import shard_path
from dryrun.workload.trace import RequestGroup, write_group_trace


class TestCollectGroupsCLI:
    def test_parses_worker_and_group_options(self):
        args = build_parser().parse_args(
            [
                "collect-groups",
                "--dataset", "d.parquet",
                "--tokenizer", "/tok",
                "--model", "/model",
                "--output", "shard.jsonl",
                "--worker-index", "3",
                "--n-workers", "16",
                "--sample-size", "1024",
                "--n", "8",
                "--max-tokens", "10240",
                "--gpu", "3",
            ]
        )
        assert args.worker_index == 3
        assert args.n_workers == 16
        assert args.n == 8
        assert args.max_tokens == 10240
        assert args.gpu == 3

    def test_defaults_match_the_dapo_recipe(self):
        args = build_parser().parse_args(
            [
                "collect-groups",
                "--dataset", "d.parquet",
                "--tokenizer", "/tok",
                "--model", "/model",
                "--output", "shard.jsonl",
            ]
        )
        assert args.n == 8
        assert args.temperature == 1.0
        assert args.top_p == 1.0
        assert args.worker_index == 0
        assert args.n_workers == 1


class TestMergeShardsCLI:
    def test_parses_merge_options(self):
        args = build_parser().parse_args(
            [
                "merge-shards",
                "--shard-dir", "shards",
                "--n-workers", "16",
                "--output", "joint.jsonl",
                "--max-output-tokens", "10240",
            ]
        )
        assert args.n_workers == 16
        assert args.max_output_tokens == 10240


class TestCollectionUtilityCLI:
    def test_parses_list_workers(self):
        args = build_parser().parse_args(
            ["list-workers", "--hosts-file", "hosts.txt", "--base-port", "9000"]
        )
        assert args.hosts_file == "hosts.txt"
        assert args.base_port == 9000

    def test_check_shard_exit_status(self, tmp_path):
        shard = write_group_trace(
            [RequestGroup(0, 10, (5, 6))],
            shard_path(tmp_path, 0),
            metadata={"max_output_tokens": 100, "n_groups_requested": 1},
        )
        args = build_parser().parse_args(["check-shard", "--path", str(shard)])
        assert cmd_check_shard(args) == 0

    def test_check_shard_returns_failure_for_short_body(self, tmp_path):
        shard = write_group_trace(
            [RequestGroup(0, 10, (5, 6))],
            shard_path(tmp_path, 0),
            metadata={"max_output_tokens": 100, "n_groups_requested": 2},
        )
        args = build_parser().parse_args(["check-shard", "--path", str(shard)])
        assert cmd_check_shard(args) == 1

    def test_verify_collection_reports_success(self, tmp_path, capsys):
        group = RequestGroup(0, 10, (5, 6))
        write_group_trace(
            [group],
            shard_path(tmp_path, 0),
            metadata={"max_output_tokens": 100, "n_groups_requested": 1},
        )
        trace = write_group_trace(
            [group],
            tmp_path / "joint.jsonl",
            metadata={"max_output_tokens": 100},
        )
        args = build_parser().parse_args(
            [
                "verify-collection",
                "--trace",
                str(trace),
                "--expected-groups",
                "1",
                "--shard-dir",
                str(tmp_path),
                "--n-workers",
                "1",
            ]
        )
        assert cmd_verify_collection(args) == 0
        assert "1/1 groups collected" in capsys.readouterr().out

    def test_verify_collection_reports_shortfall(self, tmp_path, capsys):
        group = RequestGroup(0, 10, (5, 6))
        write_group_trace(
            [group],
            shard_path(tmp_path, 0),
            metadata={"max_output_tokens": 100, "n_groups_requested": 2},
        )
        trace = write_group_trace(
            [group],
            tmp_path / "joint.jsonl",
            metadata={"max_output_tokens": 100},
        )
        args = build_parser().parse_args(
            [
                "verify-collection",
                "--trace",
                str(trace),
                "--expected-groups",
                "2",
                "--shard-dir",
                str(tmp_path),
                "--n-workers",
                "1",
            ]
        )
        assert cmd_verify_collection(args) == 1
        assert "worker-00: 1/2 groups" in capsys.readouterr().err


class TestCollectGroupTraceValidation:
    """Validation guards must fire before loading any dataset or tokenizer."""

    def test_rejects_n_workers_zero(self):
        from dryrun.workload.inference import collect_group_trace

        with pytest.raises(ValueError, match="n_workers must be at least 1"):
            collect_group_trace(
                dataset_path="/nonexistent/data.parquet",
                tokenizer_path="/nonexistent/tokenizer",
                model_path="/nonexistent/model",
                output_path="/tmp/out.jsonl",
                worker_index=0,
                n_workers=0,
            )

    def test_rejects_negative_worker_index(self):
        from dryrun.workload.inference import collect_group_trace

        with pytest.raises(ValueError, match="worker_index must be in"):
            collect_group_trace(
                dataset_path="/nonexistent/data.parquet",
                tokenizer_path="/nonexistent/tokenizer",
                model_path="/nonexistent/model",
                output_path="/tmp/out.jsonl",
                worker_index=-1,
                n_workers=5,
            )

    def test_rejects_worker_index_out_of_range(self):
        from dryrun.workload.inference import collect_group_trace

        with pytest.raises(ValueError, match="worker_index must be in"):
            collect_group_trace(
                dataset_path="/nonexistent/data.parquet",
                tokenizer_path="/nonexistent/tokenizer",
                model_path="/nonexistent/model",
                output_path="/tmp/out.jsonl",
                worker_index=5,
                n_workers=5,
            )


def _make_group(group_id: int) -> RequestGroup:
    return RequestGroup(group_id=group_id, prompt_len=10, output_lens=(5, 6, 7))


class TestPartitionGroupResults:
    def test_all_success_returns_all(self):
        groups = [_make_group(i) for i in range(3)]
        result = _partition_group_results(groups, [0, 1, 2])
        assert result == groups

    def test_mixed_returns_only_successes(self):
        g0 = _make_group(0)
        g2 = _make_group(2)
        results: list = [g0, RuntimeError("oops"), g2]
        result = _partition_group_results(results, [0, 1, 2])
        assert result == [g0, g2]

    def test_all_failed_raises_runtime_error(self):
        results: list = [ValueError("a"), OSError("b")]
        with pytest.raises(RuntimeError, match="All 2 group requests failed"):
            _partition_group_results(results, [0, 1])
