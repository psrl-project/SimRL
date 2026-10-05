"""Turn one global batch's real trajectory lengths into a microbatch plan.

Profiling measures uniform shapes, so this is the only place that knows how a real
RL batch becomes microbatches. Keeping it here rather than in the launcher means the
profile grid stays clean and there is exactly one implementation to keep honest.

The three modes span a measured 1.17x to 4.34x latency range on DAPO-like bimodal
lengths, so the mode is a first-class input to the cost model rather than a detail.
"""

from __future__ import annotations

from enum import Enum

from dryrun.profiler.schema import SequenceStats

# verl's balancing proxy: attention grows with the square of the length while the rest
# of the layer grows linearly, and 24576 is the linear weight it uses. Reproduced
# exactly, because a different proxy would partition differently and every predicted
# microbatch count would then be systematically wrong.
_VERL_LINEAR_WEIGHT = 24576


def verl_workload(length: int) -> int:
    """Return verl's per-sequence workload proxy."""
    return _VERL_LINEAR_WEIGHT * length + length**2


class BatchingMode(Enum):
    """How a rank's sequences are grouped into microbatches."""

    # Every microbatch is padded to its longest sequence. Pad tokens cost full compute.
    PAD_TO_MAX = "pad_to_max"
    # Grouped to a token budget, still padded within a microbatch. verl's default.
    TOKEN_BUDGET = "token_budget"
    # Concatenated with no pad tokens, priced on true token counts.
    PACKED_NO_PAD = "packed_no_pad"


def _balanced_partitions(lengths: list[int], count: int) -> list[list[int]]:
    """
    Partition indices into `count` groups of similar workload.

    Greedy longest-processing-time (LPT): sort by descending workload, then assign
    each item to the currently lightest group via a min-heap.  This approximates
    verl's partitioner rather than reproducing it exactly; the two can disagree on
    the same workload proxy.
    """
    import heapq

    if count <= 1:
        return [list(range(len(lengths)))]
    # Largest first, so the heaviest sequences are placed while groups are still empty.
    order = sorted(range(len(lengths)), key=lambda index: -verl_workload(lengths[index]))
    heap = [(0, group, []) for group in range(count)]
    heapq.heapify(heap)
    for index in order:
        total, group, members = heapq.heappop(heap)
        members.append(index)
        heapq.heappush(heap, (total + verl_workload(lengths[index]), group, members))
    return [sorted(members) for _, _, members in sorted(heap, key=lambda item: item[1])]


class MicrobatchPlanner:
    """Build the microbatch plan for the data-parallel rank that ends last."""

    def __init__(self, mode: BatchingMode, *, max_tokens_per_rank: int | None = None):
        if mode is BatchingMode.TOKEN_BUDGET and (max_tokens_per_rank is None or max_tokens_per_rank <= 0):
            raise ValueError("max_tokens_per_rank is required for the token_budget mode.")
        self.mode = mode
        self.max_tokens_per_rank = max_tokens_per_rank

    def plan(
        self,
        sequence_lengths: tuple[int, ...],
        dp: int,
        *,
        micro_batch_size: int,
    ) -> tuple[SequenceStats, ...]:
        """
        Return the critical rank's microbatches.

        A training step ends when its slowest rank ends, so the plan describes the rank
        carrying the most work rather than an average one.

        `micro_batch_size` is keyword-only so callers cannot silently pass the wrong
        positional argument. It must be positive, and no default is provided because a
        silent default would hide the same bug this parameter was added to fix.
        """
        if not sequence_lengths:
            raise ValueError("A microbatch plan needs at least one sequence.")
        if dp <= 0:
            raise ValueError("dp must be positive.")
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive.")
        if len(sequence_lengths) < dp:
            raise ValueError(
                f"A batch of {len(sequence_lengths)} sequences cannot fill dp={dp} ranks."
            )
        ranks = _balanced_partitions(list(sequence_lengths), dp)
        rank_lengths = [
            [sequence_lengths[index] for index in members] for members in ranks
        ]
        critical = max(
            rank_lengths,
            key=lambda lengths: sum(verl_workload(length) for length in lengths),
        )
        return self._microbatches(critical, micro_batch_size)

    def _microbatches(self, lengths: list[int], micro_batch_size: int) -> tuple[SequenceStats, ...]:
        if self.mode is BatchingMode.PAD_TO_MAX:
            # Split into groups of micro_batch_size. Padding is per-microbatch because
            # that is the actual tensor shape: a rank does not pad its whole epoch at once.
            return self._fixed_size_microbatches_padded(lengths, micro_batch_size)
        if self.mode is BatchingMode.PACKED_NO_PAD:
            # Each microbatch is a contiguous slice of micro_batch_size sequences,
            # concatenated end-to-end with no padding tokens wasted.
            return self._fixed_size_microbatches_packed(lengths, micro_batch_size)
        return self._token_budget_microbatches(lengths, micro_batch_size)

    @staticmethod
    def _fixed_size_microbatches_padded(
        lengths: list[int], micro_batch_size: int
    ) -> tuple[SequenceStats, ...]:
        """Split the rank's sequences into groups and pad within each group."""
        microbatches = []
        for start in range(0, len(lengths), micro_batch_size):
            group = lengths[start : start + micro_batch_size]
            padded = max(group)
            microbatches.append(SequenceStats.from_lengths([padded] * len(group)))
        return tuple(microbatches)

    @staticmethod
    def _fixed_size_microbatches_packed(
        lengths: list[int], micro_batch_size: int
    ) -> tuple[SequenceStats, ...]:
        """Split the rank's sequences into groups, charging only true token counts."""
        microbatches = []
        for start in range(0, len(lengths), micro_batch_size):
            group = lengths[start : start + micro_batch_size]
            microbatches.append(SequenceStats.from_lengths(group))
        return tuple(microbatches)

    def _token_budget_microbatches(
        self, lengths: list[int], micro_batch_size: int
    ) -> tuple[SequenceStats, ...]:
        """
        Group to the token budget, then pad within each group.

        The count is `ceil(total_tokens / budget)`, capped at one sequence per
        microbatch. This reproduces verl exactly: `rearrange_micro_batches` computes
        `min(num_groups, ceildiv(total_seqlen, max_token_len))` and then balances
        that many groups by workload. Validated against both matched PSRL runs --
        15 of 15 steps agree with the microbatch count the engine's own progress
        bar reports (44, 51, 46, 52, 51, 45, 52, 43, 48, 47 at batch 8, and
        189, 187, 174, 184, 171 at batch 32).

        `micro_batch_size` is deliberately ignored. verl reads it only on the
        static path (`use_dynamic_bsz=False`); under dynamic batching the budget
        alone decides, so raising the count to `ceil(len / micro_batch_size)`
        invents microbatches the engine never runs. With the usual
        `ppo_micro_batch_size_per_gpu=1` that meant one sequence per microbatch --
        64 instead of 48 on the batch-8 run -- and the step is charged per
        pipeline slot, so the error went straight into the predicted latency.

        Token budget is a target, not a guarantee: a single sequence longer than
        the budget yields one oversized microbatch, as it does in verl.
        """
        budget = int(self.max_tokens_per_rank or 0)
        total = sum(lengths)
        count = max(1, -(-total // budget))
        count = min(count, len(lengths))
        groups = _balanced_partitions(lengths, count)
        microbatches = []
        for members in groups:
            group = [lengths[index] for index in members]
            if not group:
                continue
            padded = max(group)
            microbatches.append(SequenceStats.from_lengths([padded] * len(group)))
        return tuple(microbatches)
