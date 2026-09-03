"""Tests for the grouped workload source."""

from __future__ import annotations

import pytest

from dryrun.workload.grouped import GroupedWorkload
from dryrun.workload.trace import RequestGroup, write_group_trace


def _groups() -> list[RequestGroup]:
    return [
        RequestGroup(group_id=0, prompt_len=100, output_lens=(11, 12)),
        RequestGroup(group_id=1, prompt_len=200, output_lens=(21, 22)),
    ]


class TestFromGroups:
    def test_counts(self):
        workload = GroupedWorkload.from_groups(_groups())
        assert workload.n_groups == 2
        assert workload.n_requests == 4

    def test_requests_are_group_ordered(self):
        workload = GroupedWorkload.from_groups(_groups())
        assert [workload.request(i) for i in range(4)] == [
            (0, 100, 11),
            (0, 100, 12),
            (1, 200, 21),
            (1, 200, 22),
        ]

    def test_prompt_length_is_shared_within_a_group(self):
        workload = GroupedWorkload.from_groups(_groups())
        assert workload.request(0)[1] == workload.request(1)[1]

    def test_wraparound_restarts_at_a_group_boundary(self):
        workload = GroupedWorkload.from_groups(_groups())
        assert workload.request(4) == workload.request(0)
        assert workload.request(5) == workload.request(1)

    def test_mean_prompt_len_weights_every_request(self):
        workload = GroupedWorkload.from_groups(_groups())
        assert workload.mean_prompt_len == pytest.approx(150.0)

    def test_group_size_reports_responses_per_group(self):
        workload = GroupedWorkload.from_groups(_groups())
        assert workload.group_size == 2

    def test_empty_workload_is_rejected(self):
        with pytest.raises(ValueError, match="at least one group"):
            GroupedWorkload.from_groups([])

    def test_group_with_zero_responses_is_rejected(self):
        with pytest.raises(ValueError, match="same number of responses"):
            GroupedWorkload.from_groups(
                [
                    RequestGroup(group_id=0, prompt_len=100, output_lens=(11, 12)),
                    RequestGroup(group_id=1, prompt_len=200, output_lens=()),
                ]
            )

    def test_zero_response_first_group_is_rejected(self):
        with pytest.raises(ValueError, match="same number of responses"):
            GroupedWorkload.from_groups([RequestGroup(group_id=0, prompt_len=100, output_lens=())])

    def test_ragged_group_sizes_are_rejected(self):
        with pytest.raises(ValueError, match="same number of responses"):
            GroupedWorkload.from_groups(
                [
                    RequestGroup(group_id=0, prompt_len=100, output_lens=(11,)),
                    RequestGroup(group_id=1, prompt_len=200, output_lens=(21, 22, 23)),
                ]
            )


class TestFromLengths:
    def test_each_response_is_its_own_group(self):
        workload = GroupedWorkload.from_lengths([10, 20], [1, 2])
        assert workload.n_groups == 2
        assert workload.n_requests == 2
        assert [workload.request(i) for i in range(2)] == [(0, 10, 1), (1, 20, 2)]

    def test_uneven_lists_cycle_independently(self):
        workload = GroupedWorkload.from_lengths([10], [1, 2, 3])
        assert workload.n_requests == 3
        assert [workload.request(i)[1] for i in range(3)] == [10, 10, 10]
        assert [workload.request(i)[2] for i in range(3)] == [1, 2, 3]

    def test_empty_lists_are_rejected(self):
        with pytest.raises(ValueError, match="must be non-empty"):
            GroupedWorkload.from_lengths([], [1])
        with pytest.raises(ValueError, match="must be non-empty"):
            GroupedWorkload.from_lengths([10], [])


class TestFromTrace:
    def test_reads_a_written_trace(self, tmp_path):
        path = write_group_trace(
            _groups(),
            tmp_path / "joint.jsonl",
            metadata={"max_output_tokens": 10240},
        )
        workload = GroupedWorkload.from_trace(path)
        assert workload.n_requests == 4
        assert workload.request(2) == (1, 200, 21)

    def test_n_groups_caps_the_read(self, tmp_path):
        path = write_group_trace(
            _groups(),
            tmp_path / "joint.jsonl",
            metadata={"max_output_tokens": 10240},
        )
        assert GroupedWorkload.from_trace(path, n_groups=1).n_requests == 2
