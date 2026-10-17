"""Tests for distributed collection sharding and shard merging."""

from __future__ import annotations

import json

import pytest

from srl.workload.sharding import (
    Worker,
    build_workers,
    inspect_shard,
    merge_shards,
    parse_hosts_file,
    read_trace_header,
    shard_indices,
    shard_is_complete,
    shard_path,
    verify_collection,
)
from srl.workload.trace import RequestGroup, read_group_trace, write_group_trace


class TestParseHostsFile:
    def test_parses_host_and_gpu_count(self, tmp_path):
        path = tmp_path / "hosts.txt"
        path.write_text("# host [gpus]\n29.162.247.148 8\n28.49.16.220 4\n")
        assert parse_hosts_file(path) == [("29.162.247.148", 8), ("28.49.16.220", 4)]

    def test_gpu_count_defaults_to_one(self, tmp_path):
        path = tmp_path / "hosts.txt"
        path.write_text("10.0.0.1\n")
        assert parse_hosts_file(path) == [("10.0.0.1", 1)]

    def test_blank_lines_and_comments_are_skipped(self, tmp_path):
        path = tmp_path / "hosts.txt"
        path.write_text("\n# comment\n10.0.0.1 2\n\n")
        assert parse_hosts_file(path) == [("10.0.0.1", 2)]

    def test_empty_file_is_an_error(self, tmp_path):
        path = tmp_path / "hosts.txt"
        path.write_text("# only a comment\n")
        with pytest.raises(ValueError, match="no hosts"):
            parse_hosts_file(path)

    def test_host_with_zero_gpus_is_rejected(self, tmp_path):
        path = tmp_path / "hosts.txt"
        path.write_text("10.0.0.1 0\n")
        with pytest.raises(ValueError, match="invalid GPU count"):
            parse_hosts_file(path)

    def test_host_with_negative_gpus_is_rejected(self, tmp_path):
        path = tmp_path / "hosts.txt"
        path.write_text("10.0.0.1 -2\n")
        with pytest.raises(ValueError, match="invalid GPU count"):
            parse_hosts_file(path)


class TestBuildWorkers:
    def test_one_worker_per_gpu_with_distinct_ports_per_host(self):
        workers = build_workers([("a", 2), ("b", 2)], base_port=8700)
        assert workers == [
            Worker(index=0, host="a", gpu=0, port=8700),
            Worker(index=1, host="a", gpu=1, port=8701),
            Worker(index=2, host="b", gpu=0, port=8700),
            Worker(index=3, host="b", gpu=1, port=8701),
        ]


class TestShardIndices:
    def test_shards_partition_every_item_exactly_once(self):
        n_items, n_workers = 10, 4
        seen = [i for w in range(n_workers) for i in shard_indices(n_items, w, n_workers)]
        assert sorted(seen) == list(range(n_items))

    def test_strided_split_yields_expected_indices(self):
        assert shard_indices(10, 1, 4) == [1, 5, 9]

    def test_more_workers_than_items_yields_empty_shards(self):
        assert shard_indices(2, 3, 4) == []


class TestMergeShards:
    def _write_shard(self, shard_dir, worker_index, groups, cap=100, extra_meta=None):
        meta = {"max_output_tokens": cap}
        if extra_meta:
            meta.update(extra_meta)
        return write_group_trace(
            groups, shard_path(shard_dir, worker_index), metadata=meta
        )

    def test_merges_in_group_id_order(self, tmp_path):
        self._write_shard(tmp_path, 0, [RequestGroup(0, 10, (1, 2)), RequestGroup(2, 30, (5, 6))])
        self._write_shard(tmp_path, 1, [RequestGroup(1, 20, (3, 4))])
        out = merge_shards(tmp_path, 2, tmp_path / "joint.jsonl", {})
        assert [g.group_id for g in read_group_trace(out)] == [0, 1, 2]

    def test_missing_shard_is_named(self, tmp_path):
        self._write_shard(tmp_path, 0, [RequestGroup(0, 10, (1, 2))])
        with pytest.raises(FileNotFoundError, match="worker-01"):
            merge_shards(tmp_path, 2, tmp_path / "joint.jsonl", {})

    def test_empty_shard_file_is_rejected(self, tmp_path):
        self._write_shard(tmp_path, 0, [RequestGroup(0, 10, (1, 2))])
        empty_shard = shard_path(tmp_path, 1)
        empty_shard.write_text("")
        with pytest.raises(ValueError, match="worker-01"):
            merge_shards(tmp_path, 2, tmp_path / "joint.jsonl", {})

    def test_header_only_shard_file_is_rejected(self, tmp_path):
        self._write_shard(tmp_path, 0, [RequestGroup(0, 10, (1, 2))])
        shard_1_path = shard_path(tmp_path, 1)
        write_group_trace(
            [RequestGroup(1, 20, (3, 4))],
            shard_1_path,
            metadata={"max_output_tokens": 100},
        )
        lines = shard_1_path.read_text(encoding="utf-8").splitlines()
        shard_1_path.write_text(lines[0] + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="worker-01"):
            merge_shards(tmp_path, 2, tmp_path / "joint.jsonl", {})

    # --- FIX 1: cap extraction from shards ---

    def test_shards_disagreeing_on_cap_are_rejected(self, tmp_path):
        """Shards with different max_output_tokens raise, naming the differing values."""
        self._write_shard(tmp_path, 0, [RequestGroup(0, 10, (1, 2))], cap=1024)
        self._write_shard(tmp_path, 1, [RequestGroup(1, 20, (3, 4))], cap=512)
        with pytest.raises(ValueError, match="disagree on max_output_tokens"):
            merge_shards(tmp_path, 2, tmp_path / "joint.jsonl", {})

    def test_merged_cap_comes_from_shards_not_caller(self, tmp_path):
        """
        Shards collected at cap 1024 produce a merged trace with cap 1024
        and truncated_fraction 1.0 regardless of caller metadata.

        This is the exact bug that was fixed: collecting at cap 1024
        (every response truncated) then merging with the default cap 10240
        previously produced truncated_fraction 0.0.
        """
        # Every output_len equals the cap, so all responses are truncated.
        cap = 1024
        self._write_shard(tmp_path, 0, [RequestGroup(0, 10, (cap, cap))], cap=cap)
        self._write_shard(tmp_path, 1, [RequestGroup(1, 20, (cap, cap))], cap=cap)
        # Pass no caller cap: shards are authoritative.
        out = merge_shards(tmp_path, 2, tmp_path / "joint.jsonl", {})
        with out.open(encoding="utf-8") as fh:
            header = json.loads(fh.readline())
        assert header["max_output_tokens"] == cap
        assert header["stats"]["output"]["truncated_fraction"] == 1.0

    def test_caller_cap_as_assertion_passes_when_matching(self, tmp_path):
        """A caller-supplied cap that matches the shards is accepted."""
        self._write_shard(tmp_path, 0, [RequestGroup(0, 10, (50, 60))], cap=100)
        self._write_shard(tmp_path, 1, [RequestGroup(1, 20, (30, 40))], cap=100)
        # Passing the correct cap as an assertion should not raise.
        out = merge_shards(tmp_path, 2, tmp_path / "joint.jsonl", {"max_output_tokens": 100})
        with out.open(encoding="utf-8") as fh:
            header = json.loads(fh.readline())
        assert header["max_output_tokens"] == 100

    def test_caller_cap_assertion_fails_when_mismatched(self, tmp_path):
        """A caller-supplied cap that contradicts the shards raises, naming both."""
        self._write_shard(tmp_path, 0, [RequestGroup(0, 10, (50, 60))], cap=1024)
        self._write_shard(tmp_path, 1, [RequestGroup(1, 20, (30, 40))], cap=1024)
        with pytest.raises(ValueError, match="10240"):
            merge_shards(tmp_path, 2, tmp_path / "joint.jsonl", {"max_output_tokens": 10240})

    def test_provenance_fields_common_to_all_shards_survive(self, tmp_path):
        """Fields identical across every shard appear in the merged header."""
        common_meta = {"model": "qwen3-8b", "dataset": "dapo", "seed": 42}
        self._write_shard(
            tmp_path, 0, [RequestGroup(0, 10, (50, 60))], cap=100, extra_meta=common_meta
        )
        self._write_shard(
            tmp_path, 1, [RequestGroup(1, 20, (30, 40))], cap=100, extra_meta=common_meta
        )
        out = merge_shards(tmp_path, 2, tmp_path / "joint.jsonl", {})
        with out.open(encoding="utf-8") as fh:
            header = json.loads(fh.readline())
        assert header["model"] == "qwen3-8b"
        assert header["dataset"] == "dapo"
        assert header["seed"] == 42
        assert header["n_shards"] == 2

    def test_provenance_fields_differing_across_shards_are_omitted(self, tmp_path):
        """Fields that differ between shards are not copied to the merged header."""
        self._write_shard(
            tmp_path, 0, [RequestGroup(0, 10, (50, 60))], cap=100, extra_meta={"seed": 1}
        )
        self._write_shard(
            tmp_path, 1, [RequestGroup(1, 20, (30, 40))], cap=100, extra_meta={"seed": 2}
        )
        out = merge_shards(tmp_path, 2, tmp_path / "joint.jsonl", {})
        with out.open(encoding="utf-8") as fh:
            header = json.loads(fh.readline())
        # Differing seed should not appear in the merged header.
        assert "seed" not in header

    # --- FIX 2: body-count completeness check ---

    def test_header_over_reporting_n_groups_is_detected(self, tmp_path):
        """
        A shard whose header claims more groups than the body actually holds
        is treated as incomplete by read_group_trace.

        This validates the detection mechanism used by _shard_is_complete in
        collect_joint_trace.sh: counting actual body groups via read_group_trace
        against n_groups_requested, rather than trusting header.n_groups.
        """
        # Write a valid shard for worker-0.
        self._write_shard(tmp_path, 0, [RequestGroup(0, 10, (50, 60))], cap=100)

        # Fabricate a worker-1 shard whose header says 2 groups but body has only 1.
        shard_1_path = shard_path(tmp_path, 1)
        write_group_trace(
            [RequestGroup(1, 20, (30, 40)), RequestGroup(3, 30, (10, 20))],
            shard_1_path,
            metadata={"max_output_tokens": 100, "n_groups_requested": 2},
        )
        # Truncate the file to remove the second body line.
        lines = shard_1_path.read_text(encoding="utf-8").splitlines(keepends=True)
        # Keep header + first group line only (drop the second group line).
        shard_1_path.write_text("".join(lines[:2]), encoding="utf-8")

        # Verify that read_group_trace only sees 1 group (header said 2).
        actual = read_group_trace(shard_1_path)
        assert len(actual) == 1, (
            "read_group_trace should see 1 body group even though header claimed 2."
        )

        # The header's n_groups_requested is still 2, so the body is short.
        with shard_1_path.open(encoding="utf-8") as fh:
            header = json.loads(fh.readline())
        assert header.get("n_groups_requested") == 2
        assert len(actual) < header["n_groups_requested"]


class TestShardCompleteness:
    def test_complete_shard_counts_body_groups(self, tmp_path):
        path = write_group_trace(
            [RequestGroup(0, 10, (5, 6)), RequestGroup(1, 20, (7, 8))],
            shard_path(tmp_path, 0),
            metadata={"max_output_tokens": 100, "n_groups_requested": 2},
        )
        status = inspect_shard(path)
        assert status.complete
        assert status.actual == 2
        assert status.requested == 2
        assert shard_is_complete(path)

    def test_truncated_shard_is_incomplete(self, tmp_path):
        path = write_group_trace(
            [RequestGroup(0, 10, (5, 6)), RequestGroup(1, 20, (7, 8))],
            shard_path(tmp_path, 0),
            metadata={"max_output_tokens": 100, "n_groups_requested": 2},
        )
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(lines[:2]), encoding="utf-8")
        status = inspect_shard(path)
        assert not status.complete
        assert status.actual == 1
        assert status.requested == 2

    def test_unreadable_header_is_incomplete(self, tmp_path):
        path = shard_path(tmp_path, 0)
        path.write_text("not-json\n", encoding="utf-8")
        status = inspect_shard(path)
        assert not status.complete
        assert status.error

    def test_read_trace_header_requires_meta_record(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        path.write_text('{"group_id": 0}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="_meta"):
            read_trace_header(path)


class TestVerifyCollection:
    def test_returns_actual_count_when_complete(self, tmp_path):
        groups = [RequestGroup(0, 10, (5, 6)), RequestGroup(1, 20, (7, 8))]
        write_group_trace(
            groups,
            shard_path(tmp_path, 0),
            metadata={"max_output_tokens": 100, "n_groups_requested": 2},
        )
        trace = write_group_trace(
            groups,
            tmp_path / "joint.jsonl",
            metadata={"max_output_tokens": 100},
        )
        assert verify_collection(trace, 2, shard_dir=tmp_path, n_workers=1) == 2

    def test_short_collection_names_responsible_shard(self, tmp_path):
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
        with pytest.raises(ValueError, match=r"worker-00: 1/2 groups"):
            verify_collection(trace, 2, shard_dir=tmp_path, n_workers=1)

