"""The simulator's workload source: groups of requests sharing one prompt.

DAPO samples `n` responses per prompt and computes group-relative advantages,
so a group is a structural unit of the workload. Holding the workload as an
ordered list of groups keeps group boundaries in the data: request wraparound
lands on a boundary instead of splitting a group, which is what lets whole-group
batch consumption and shared-prefix prefill build on this later.
"""

from __future__ import annotations

from pathlib import Path

from .trace import RequestGroup, read_group_trace


class GroupedWorkload:
    """An ordered, cycling sequence of requests grouped by shared prompt."""

    def __init__(self, groups: list[RequestGroup]):
        if not groups:
            raise ValueError("A grouped workload needs at least one group.")
        group_size = len(groups[0].output_lens)
        if group_size == 0:
            raise ValueError(
                f"Every group must hold the same number of responses, expected {group_size}."
            )
        if any(len(group.output_lens) != group_size for group in groups):
            offending = next(len(g.output_lens) for g in groups if len(g.output_lens) != group_size)
            raise ValueError(
                f"Every group must hold the same number of responses, expected {group_size}, "
                f"got {offending}."
            )
        self.groups = groups
        # Flatten once so `request()` is a list index rather than a scan, and so
        # wraparound is modulo the whole group-ordered sequence.
        self._requests: list[tuple[int, int, int]] = [
            (group.group_id, group.prompt_len, output_len)
            for group in groups
            for output_len in group.output_lens
        ]

    @classmethod
    def from_groups(cls, groups: list[RequestGroup]) -> "GroupedWorkload":
        """
        Build from explicit groups.

        Args:
            groups (list[RequestGroup]): Groups in the order requests are issued.

        Returns:
            GroupedWorkload: The workload over those groups.
        """
        return cls(list(groups))

    @classmethod
    def from_lengths(
        cls,
        prompt_lengths: list[int],
        output_lengths: list[int],
    ) -> "GroupedWorkload":
        """
        Build a workload of single-response groups from two length lists.

        An ungrouped workload is a workload whose groups hold one response each.
        The two lists are sampled into `max(len(prompt_lengths), len(output_lengths))`
        groups, with each list cycling modulo its own length. The workload then repeats
        with that same period.

        Args:
            prompt_lengths (list[int]): Prompt lengths to cycle.
            output_lengths (list[int]): Response lengths to cycle.

        Returns:
            GroupedWorkload: One group per request.
        """
        if not prompt_lengths or not output_lengths:
            raise ValueError("Both prompt_lengths and output_lengths must be non-empty.")
        n = max(len(prompt_lengths), len(output_lengths))
        return cls(
            [
                RequestGroup(
                    group_id=i,
                    prompt_len=int(prompt_lengths[i % len(prompt_lengths)]),
                    output_lens=(int(output_lengths[i % len(output_lengths)]),),
                )
                for i in range(n)
            ]
        )

    @classmethod
    def from_trace(cls, path: str | Path, n_groups: int = 0) -> "GroupedWorkload":
        """
        Build from a joint group trace artifact.

        Args:
            path (str | Path): JSONL trace written by `write_group_trace`.
            n_groups (int): Cap on groups read. 0 reads every group.

        Returns:
            GroupedWorkload: The workload over the trace's groups.
        """
        return cls(read_group_trace(path, n_groups=n_groups))

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def n_requests(self) -> int:
        return len(self._requests)

    @property
    def group_size(self) -> int:
        return len(self.groups[0].output_lens)

    @property
    def mean_prompt_len(self) -> float:
        """Mean prompt length per request, so larger groups weigh more."""
        return sum(prompt_len for _, prompt_len, _ in self._requests) / len(self._requests)

    def request(self, index: int) -> tuple[int, int, int]:
        """
        Return `(group_id, prompt_len, output_len)` for a request index.

        Indices beyond the workload wrap around. Because the sequence is
        group-ordered and complete, wraparound restarts at a group boundary.

        Args:
            index (int): Zero-based request index, unbounded above.

        Returns:
            tuple[int, int, int]: The group id and the paired lengths.
        """
        return self._requests[index % len(self._requests)]
