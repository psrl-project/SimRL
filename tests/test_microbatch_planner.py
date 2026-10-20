"""Turning real trajectory lengths into a microbatch plan."""

from __future__ import annotations

import pytest

from srl.cost.training.microbatch import BatchingMode, MicrobatchPlanner, verl_workload


def test_verl_workload_matches_the_reference_proxy():
    """verl balances on `24576 * L + L^2`. Diverging would skew every predicted count."""
    assert verl_workload(1000) == 24576 * 1000 + 1000**2
    assert verl_workload(2000) == 24576 * 2000 + 2000**2


def test_pad_to_max_charges_for_the_padding():
    planner = MicrobatchPlanner(BatchingMode.PAD_TO_MAX)
    plan = planner.plan((100, 200, 300, 400), dp=2, micro_batch_size=2)
    # Two ranks, each holds two sequences. micro_batch_size=2 fits both in one microbatch.
    assert len(plan) == 1
    # The longest sequence on the critical rank sets every slot's cost.
    assert plan[0].count == 2
    assert plan[0].sum_tokens == plan[0].max_tokens * plan[0].count


def test_packed_no_pad_charges_only_real_tokens():
    planner = MicrobatchPlanner(BatchingMode.PACKED_NO_PAD)
    plan = planner.plan((100, 200, 300, 400), dp=2, micro_batch_size=2)
    assert sum(item.sum_tokens for item in plan) < 400 * 2
    # LPT pairs the longest with the shortest, giving 400 + 100 on the critical rank
    # against 300 + 200 on the other. Both total 500. The critical rank is the one
    # with the greater quadratic workload.
    assert plan[0].sum_tokens == 500
    assert plan[0].count == 2


def test_packed_beats_padded_on_a_skewed_distribution():
    """This is the 1.17x-to-4.34x measured gap. The plan must express it."""
    lengths = (10240, 10240, 900, 1100, 950, 1050, 800, 1200)
    padded = MicrobatchPlanner(BatchingMode.PAD_TO_MAX).plan(lengths, dp=2, micro_batch_size=4)
    packed = MicrobatchPlanner(BatchingMode.PACKED_NO_PAD).plan(lengths, dp=2, micro_batch_size=4)
    padded_tokens = sum(item.sum_tokens for item in padded)
    packed_tokens = sum(item.sum_tokens for item in packed)
    assert packed_tokens < padded_tokens
    assert padded_tokens / packed_tokens > 1.5


def test_token_budget_splits_when_the_cap_is_exceeded():
    planner = MicrobatchPlanner(BatchingMode.TOKEN_BUDGET, max_tokens_per_rank=1000)
    plan = planner.plan((400, 400, 400, 400), dp=1, micro_batch_size=4)
    assert len(plan) >= 2
    assert all(item.sum_tokens <= 1000 for item in plan)


def test_token_budget_keeps_one_microbatch_when_it_fits():
    planner = MicrobatchPlanner(BatchingMode.TOKEN_BUDGET, max_tokens_per_rank=10_000)
    plan = planner.plan((400, 400), dp=1, micro_batch_size=2)
    assert len(plan) == 1


def test_token_budget_is_deterministic():
    """A nondeterministic plan would make a prediction irreproducible."""
    planner = MicrobatchPlanner(BatchingMode.TOKEN_BUDGET, max_tokens_per_rank=2000)
    lengths = (900, 1100, 700, 1300, 500, 1500)
    first = planner.plan(lengths, dp=2, micro_batch_size=3)
    second = planner.plan(lengths, dp=2, micro_batch_size=3)
    assert [(i.count, i.sum_tokens) for i in first] == [(i.count, i.sum_tokens) for i in second]


def test_token_budget_balances_rather_than_filling_greedily():
    """LPT balances workload across microbatches. A greedy input-order fill does not."""
    planner = MicrobatchPlanner(BatchingMode.TOKEN_BUDGET, max_tokens_per_rank=2100)
    plan = planner.plan((2000, 100, 2000, 100), dp=1, micro_batch_size=4)
    # LPT pairs each long sequence with a short one, so both groups hold two.
    # An input-order fill would put the two 2000s together and leave a nearly empty group.
    assert all(item.count == 2 for item in plan)
    # Padding to the group maximum is why this costs 2 * 2000 rather than 2100: the
    # budget selects the microbatch *count*, and padding then charges for the short
    # sequence as if it were long. Only packed_no_pad avoids that.
    assert sorted(item.sum_tokens for item in plan) == [4000, 4000]


def test_token_budget_requires_a_cap():
    with pytest.raises(ValueError, match="max_tokens_per_rank"):
        MicrobatchPlanner(BatchingMode.TOKEN_BUDGET)


def test_a_single_sequence_longer_than_the_cap_still_yields_one_microbatch():
    """A sequence cannot be split, so the cap is a target rather than a guarantee."""
    planner = MicrobatchPlanner(BatchingMode.TOKEN_BUDGET, max_tokens_per_rank=100)
    plan = planner.plan((5000,), dp=1, micro_batch_size=1)
    assert len(plan) == 1
    assert plan[0].sum_tokens == 5000


def test_plan_uses_the_busiest_rank():
    """The step ends when the slowest rank ends, so the plan describes that rank."""
    planner = MicrobatchPlanner(BatchingMode.PACKED_NO_PAD)
    plan = planner.plan((1000, 10, 10, 10), dp=2, micro_batch_size=2)
    assert max(item.max_tokens for item in plan) == 1000


def test_plan_rejects_a_batch_smaller_than_the_dp_degree():
    planner = MicrobatchPlanner(BatchingMode.PACKED_NO_PAD)
    with pytest.raises(ValueError, match="dp"):
        planner.plan((100,), dp=4, micro_batch_size=1)


def test_plan_rejects_empty_lengths():
    planner = MicrobatchPlanner(BatchingMode.PACKED_NO_PAD)
    with pytest.raises(ValueError):
        planner.plan((), dp=1, micro_batch_size=1)


def test_critical_rank_selected_by_workload_not_token_count():
    """Workload, not token count, picks the critical rank.

    Rank A: 505 sequences of length 1: token count 505, workload 505*(24577) = 12,411,385.
    Rank B: 1 sequence of length 500: token count 500, workload 24576*500+500^2 = 12,538,000.
    Token count picks rank A. Workload picks rank B. The plan must reflect rank B.
    """
    planner = MicrobatchPlanner(BatchingMode.PACKED_NO_PAD)
    lengths = tuple([1] * 505 + [500])
    plan = planner.plan(lengths, dp=2, micro_batch_size=253)
    assert max(item.max_tokens for item in plan) == 500


def test_token_budget_rejects_zero_cap():
    with pytest.raises(ValueError, match="max_tokens_per_rank"):
        MicrobatchPlanner(BatchingMode.TOKEN_BUDGET, max_tokens_per_rank=0)


def test_token_budget_rejects_negative_cap():
    with pytest.raises(ValueError, match="max_tokens_per_rank"):
        MicrobatchPlanner(BatchingMode.TOKEN_BUDGET, max_tokens_per_rank=-5)


def test_plan_rejects_zero_micro_batch_size():
    planner = MicrobatchPlanner(BatchingMode.PACKED_NO_PAD)
    with pytest.raises(ValueError, match="micro_batch_size"):
        planner.plan((100, 200), dp=1, micro_batch_size=0)


def test_plan_rejects_negative_micro_batch_size():
    planner = MicrobatchPlanner(BatchingMode.PACKED_NO_PAD)
    with pytest.raises(ValueError, match="micro_batch_size"):
        planner.plan((100, 200), dp=1, micro_batch_size=-1)


# ---- regression tests for the single-oversized-microbatch bug ----

def test_pad_to_max_micro_batch_size_1_yields_one_microbatch_per_sequence():
    """With micro_batch_size=1 every sequence is its own microbatch, not crammed together."""
    planner = MicrobatchPlanner(BatchingMode.PAD_TO_MAX)
    # 8 sequences per rank (batch=8, dp=1)
    lengths = tuple([9176] * 8)
    plan = planner.plan(lengths, dp=1, micro_batch_size=1)
    assert len(plan) == 8
    for microbatch in plan:
        assert microbatch.count == 1
        assert microbatch.sum_tokens == 9176


def test_packed_no_pad_micro_batch_size_1_yields_one_microbatch_per_sequence():
    """Same regression for packed_no_pad mode."""
    planner = MicrobatchPlanner(BatchingMode.PACKED_NO_PAD)
    lengths = tuple([9176] * 8)
    plan = planner.plan(lengths, dp=1, micro_batch_size=1)
    assert len(plan) == 8
    for microbatch in plan:
        assert microbatch.count == 1
        assert microbatch.sum_tokens == 9176


def test_pad_to_max_micro_batch_size_2_yields_four_microbatches():
    """micro_batch_size=2 with 8 sequences per rank gives 4 microbatches."""
    planner = MicrobatchPlanner(BatchingMode.PAD_TO_MAX)
    lengths = tuple([9176] * 8)
    plan = planner.plan(lengths, dp=1, micro_batch_size=2)
    assert len(plan) == 4
    for microbatch in plan:
        assert microbatch.count == 2


def test_packed_no_pad_micro_batch_size_2_yields_four_microbatches():
    """Same for packed_no_pad."""
    planner = MicrobatchPlanner(BatchingMode.PACKED_NO_PAD)
    lengths = tuple([9176] * 8)
    plan = planner.plan(lengths, dp=1, micro_batch_size=2)
    assert len(plan) == 4
    for microbatch in plan:
        assert microbatch.count == 2


def test_pad_to_max_padding_is_per_microbatch_not_rank_wide():
    """A short sequence in one group must not be charged the rank-wide maximum."""
    planner = MicrobatchPlanner(BatchingMode.PAD_TO_MAX)
    # Two groups: first has [9176, 9176], second has [100, 100].
    # With rank-wide padding both groups would charge 9176 tokens. With per-microbatch
    # padding the second group charges only 100.
    lengths = (9176, 9176, 100, 100)
    plan = planner.plan(lengths, dp=1, micro_batch_size=2)
    assert len(plan) == 2
    token_counts = sorted(item.sum_tokens for item in plan)
    # The short group should cost 200, not 9176*2=18352.
    assert token_counts[0] == 200
    assert token_counts[1] == 9176 * 2


def test_token_budget_ignores_micro_batch_size():
    """Under dynamic batching the budget alone sets the count.

    verl reads `micro_batch_size_per_gpu` only on the static path. With
    `use_dynamic_bsz=True` `rearrange_micro_batches` computes
    `min(num_groups, ceildiv(total_seqlen, max_token_len))` and never raises the
    count to one sequence per microbatch. Raising it invents pipeline slots the
    engine does not run, and the step is charged per slot: with the usual
    `ppo_micro_batch_size_per_gpu=1` that made a 48-microbatch step cost 64,
    which is where a 33% training overprediction came from.

    Checked against ground truth on two PSRL runs: the microbatch count the
    engine's own progress bar reports agrees with `ceildiv` on 15 of 15 steps.
    """
    planner = MicrobatchPlanner(BatchingMode.TOKEN_BUDGET, max_tokens_per_rank=100_000)
    lengths = tuple([100] * 8)
    plan = planner.plan(lengths, dp=1, micro_batch_size=1)
    # 800 tokens against a 100k budget is one pass, whatever micro_batch_size says.
    assert len(plan) == 1
    assert plan[0].count == 8


def test_token_budget_seq_cap_does_not_affect_when_budget_already_tighter():
    """When the token budget already demands more splits than the seq cap, seq cap is a no-op."""
    # 4 sequences of 300 tokens, budget of 500 → ceil(1200/500)=3 microbatches.
    # micro_batch_size=4 → seq_cap = ceil(4/4)=1 microbatch.
    # The budget (3) wins over the seq cap (1).
    planner = MicrobatchPlanner(BatchingMode.TOKEN_BUDGET, max_tokens_per_rank=500)
    lengths = (300, 300, 300, 300)
    plan = planner.plan(lengths, dp=1, micro_batch_size=4)
    assert len(plan) >= 3
