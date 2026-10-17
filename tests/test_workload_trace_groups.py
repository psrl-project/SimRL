"""Tests for joint group trace artifacts."""

from __future__ import annotations

import json

import pytest

from srl.workload.trace import (
    RequestGroup,
    compute_group_stats,
    compute_length_stats,
    read_group_trace,
    write_group_trace,
)


def _groups() -> list[RequestGroup]:
    return [
        RequestGroup(group_id=0, prompt_len=411, output_lens=(6300, 10240)),
        RequestGroup(group_id=1, prompt_len=90, output_lens=(9179, 10240)),
    ]


def test_compute_length_stats():
    stats = compute_length_stats([10, 20, 30, 40, 50], n_bins=5)
    assert stats["min"] == 10
    assert stats["max"] == 50
    assert stats["mean"] == 30.0
    assert sum(stats["histogram"]["counts"]) == 5


class TestWriteGroupTrace:
    def test_roundtrip_preserves_groups(self, tmp_path):
        path = write_group_trace(
            _groups(), tmp_path / "joint.jsonl", metadata={"tp": 1, "max_output_tokens": 10240}
        )
        assert read_group_trace(path) == _groups()

    def test_header_records_group_size_and_metadata(self, tmp_path):
        path = write_group_trace(
            _groups(), tmp_path / "joint.jsonl", metadata={"tp": 1, "max_output_tokens": 10240}
        )
        with path.open() as fh:
            header = json.loads(fh.readline())
        assert header["_meta"] is True
        assert header["n_groups"] == 2
        assert header["group_size"] == 2
        assert header["tp"] == 1
        assert "stats" in header

    def test_ragged_group_sizes_are_rejected(self, tmp_path):
        ragged = [
            RequestGroup(group_id=0, prompt_len=10, output_lens=(1, 2)),
            RequestGroup(group_id=1, prompt_len=20, output_lens=(3,)),
        ]
        with pytest.raises(ValueError, match="same number of responses"):
            write_group_trace(ragged, tmp_path / "joint.jsonl", metadata={})

    def test_empty_groups_are_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="at least one group"):
            write_group_trace([], tmp_path / "joint.jsonl", metadata={})

    def test_missing_max_output_tokens_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="max_output_tokens"):
            write_group_trace(_groups(), tmp_path / "joint.jsonl", metadata={})


class TestReadGroupTrace:
    def test_n_groups_caps_the_read(self, tmp_path):
        path = write_group_trace(
            _groups(), tmp_path / "joint.jsonl", metadata={"max_output_tokens": 10240}
        )
        assert read_group_trace(path, n_groups=1) == _groups()[:1]

    def test_zero_n_groups_reads_all(self, tmp_path):
        path = write_group_trace(
            _groups(), tmp_path / "joint.jsonl", metadata={"max_output_tokens": 10240}
        )
        assert read_group_trace(path, n_groups=0) == _groups()


class TestComputeGroupStats:
    def test_reports_prompt_output_and_truncation(self):
        stats = compute_group_stats(_groups(), max_output_tokens=10240, n_bins=4)
        assert stats["n_groups"] == 2
        assert stats["group_size"] == 2
        assert stats["prompt"]["min"] == 90
        assert stats["output"]["max"] == 10240
        assert stats["output"]["truncated_fraction"] == pytest.approx(0.5)
