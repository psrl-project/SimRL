"""FSDP candidate enumeration and its topology constraints."""

from __future__ import annotations

import numpy as np
import pytest

from dryrun.profiler.fsdp_search_space import FsdpCandidate, enumerate_fsdp_candidates
from dryrun.profiler.schema import ProfileValidationError


def _candidate(**overrides) -> FsdpCandidate:
    defaults = {
        "shard_size": 8,
        "replicate_size": 1,
        "n_seqs_per_rank": 2,
        "sequence_length": 1024,
        "gradient_checkpointing": True,
        "optimizer_offload": False,
        "reshard_after_forward": True,
    }
    return FsdpCandidate(**{**defaults, **overrides})


def test_world_size_is_the_mesh_product():
    assert _candidate(shard_size=4, replicate_size=2).world_size == 8


def test_every_rank_receives_distinct_data_so_dp_equals_world_size():
    """HSDP replicates parameters, not data, so the profile topology is pure DP."""
    parallelism = _candidate(shard_size=4, replicate_size=2).parallelism
    assert (parallelism.tp, parallelism.pp, parallelism.cp) == (1, 1, 1)
    assert parallelism.dp == 8
    assert parallelism.world_size == 8


def test_a_rank_runs_identically_shaped_microbatches():
    """Accumulation depth is swept; every microbatch a rank runs is the same shape."""
    candidate = _candidate(n_seqs_per_rank=4, num_microbatches=3)
    assert candidate.num_microbatches == 3
    assert candidate.micro_batch_size == 4
    assert candidate.sequence_shape == (1024, 1024, 1024, 1024)


def test_num_microbatches_defaults_to_one():
    assert _candidate().num_microbatches == 1


def test_rejects_a_non_positive_microbatch_count():
    with pytest.raises(ProfileValidationError, match="num_microbatches"):
        _candidate(num_microbatches=0)


def test_global_batch_size_matches_the_payload_invariant():
    candidate = _candidate(
        shard_size=8, replicate_size=1, n_seqs_per_rank=2, num_microbatches=3
    )
    expected = candidate.micro_batch_size * candidate.parallelism.dp * candidate.num_microbatches
    assert candidate.global_batch_size == expected


def test_candidate_id_is_stable_and_sensitive():
    assert _candidate().candidate_id == _candidate().candidate_id
    assert _candidate().candidate_id != _candidate(sequence_length=2048).candidate_id
    # The accumulation depth changes what actually ran, so it must change the identity.
    assert _candidate().candidate_id != _candidate(num_microbatches=2).candidate_id
    assert len(_candidate().candidate_id) == 16


def test_round_trips_through_a_dict():
    candidate = _candidate(
        replicate_size=2, shard_size=4, optimizer_offload=True, num_microbatches=4
    )
    assert FsdpCandidate.from_dict(candidate.to_dict()) == candidate


def test_rejects_sequence_parallelism():
    """Ulysses needs an all-to-all attention wrapper that does not exist yet."""
    with pytest.raises(ProfileValidationError, match="sequence_parallel"):
        _candidate(sequence_parallel=2)


def test_rejects_non_positive_dimensions():
    with pytest.raises(ProfileValidationError):
        _candidate(shard_size=0)
    with pytest.raises(ProfileValidationError):
        _candidate(n_seqs_per_rank=0)
    with pytest.raises(ProfileValidationError):
        _candidate(sequence_length=0)


def test_enumerate_derives_replicate_from_world_size():
    candidates = enumerate_fsdp_candidates(
        world_size=8,
        shard_sizes=[2, 4, 8],
        n_seqs_per_rank=[1],
        sequence_lengths=[512],
        gradient_checkpointing=[True],
        optimizer_offload=[False],
        reshard_after_forward=[True],
    )
    assert sorted((c.replicate_size, c.shard_size) for c in candidates) == [
        (1, 8),
        (2, 4),
        (4, 2),
    ]


def test_enumerate_skips_shard_sizes_that_do_not_divide_the_world():
    candidates = enumerate_fsdp_candidates(
        world_size=8,
        shard_sizes=[3, 8],
        n_seqs_per_rank=[1],
        sequence_lengths=[512],
        gradient_checkpointing=[True],
        optimizer_offload=[False],
        reshard_after_forward=[True],
    )
    assert [c.shard_size for c in candidates] == [8]


def test_enumerate_takes_the_product_of_the_shape_axes():
    """
    `n_seqs_per_rank` and `sequence_length` both move only the token count.

    Their product is a wider token range, not a third design column: the name this test
    used to carry claimed it reached rank three, which it cannot. See
    `test_varying_both_axes_reaches_a_rank_three_latency_design` for the axis that does.
    """
    candidates = enumerate_fsdp_candidates(
        world_size=8,
        shard_sizes=[8],
        n_seqs_per_rank=[1, 2, 4],
        sequence_lengths=[512, 1024],
        gradient_checkpointing=[True],
        optimizer_offload=[False],
        reshard_after_forward=[True],
    )
    assert len({c.sequence_length for c in candidates}) == 2
    assert len({c.n_seqs_per_rank for c in candidates}) == 3
    assert len(candidates) == 6
    assert _latency_design_rank(candidates) == 2


def _latency_design_rank(candidates: list[FsdpCandidate]) -> int:
    """
    Return the rank of the latency design the fit builds from a sweep.

    Mirrors `LatencyFeatures.design_row` for `pp=1`, where slots collapse to the
    microbatch count: `[1, slots * tokens, slots]`. Built here rather than imported so
    the property is pinned against the algebra the cost model documents, and a change to
    either side has to be made deliberately on both.
    """
    rows = [
        (
            1.0,
            float(c.num_microbatches * c.sequence_length * c.n_seqs_per_rank),
            float(c.num_microbatches),
        )
        for c in candidates
    ]
    return int(np.linalg.matrix_rank(np.asarray(rows, dtype=np.float64)))


def test_sequence_length_alone_cannot_identify_the_two_overheads():
    """
    At a fixed microbatch count the slots column is constant, hence collinear with the
    intercept, so the design caps at rank two while the model needs three.

    Left unfixed the fit either raises on every real profile or, under
    `require_full_rank=False`, splits the two overheads arbitrarily: the measured
    `0.125/0.125` against a `0.20/0.05` ground truth is a 53.5% latency error at nmb=4
    while the in-sample coefficient of determination reads 1.0.
    """
    candidates = enumerate_fsdp_candidates(
        world_size=8,
        shard_sizes=[8],
        n_seqs_per_rank=[1],
        sequence_lengths=[512, 1024, 2048],
        gradient_checkpointing=[True],
        optimizer_offload=[False],
        reshard_after_forward=[True],
    )
    assert {c.num_microbatches for c in candidates} == {1}
    assert _latency_design_rank(candidates) == 2


def test_varying_both_axes_reaches_a_rank_three_latency_design():
    """Sweeping the accumulation depth is what makes all three coefficients identifiable."""
    candidates = enumerate_fsdp_candidates(
        world_size=8,
        shard_sizes=[8],
        n_seqs_per_rank=[1],
        sequence_lengths=[512, 1024, 2048],
        num_microbatches=[1, 2, 4],
        gradient_checkpointing=[True],
        optimizer_offload=[False],
        reshard_after_forward=[True],
    )
    assert {c.num_microbatches for c in candidates} == {1, 2, 4}
    assert len(candidates) == 9
    assert _latency_design_rank(candidates) == 3


def test_enumerate_rejects_a_non_positive_microbatch_count():
    with pytest.raises(ProfileValidationError, match="num_microbatches"):
        enumerate_fsdp_candidates(
            world_size=8,
            shard_sizes=[8],
            n_seqs_per_rank=[1],
            sequence_lengths=[512],
            num_microbatches=[0, 2],
            gradient_checkpointing=[True],
            optimizer_offload=[False],
            reshard_after_forward=[True],
        )


def test_enumerate_repeats_produce_distinct_candidates():
    candidates = enumerate_fsdp_candidates(
        world_size=8,
        shard_sizes=[8],
        n_seqs_per_rank=[1],
        sequence_lengths=[512],
        gradient_checkpointing=[True],
        optimizer_offload=[False],
        reshard_after_forward=[True],
        repeats=3,
    )
    assert [c.repeat for c in candidates] == [0, 1, 2]
    assert len({c.candidate_id for c in candidates}) == 3


def test_enumerate_rejects_non_positive_repeats():
    with pytest.raises(ProfileValidationError):
        enumerate_fsdp_candidates(
            world_size=8,
            shard_sizes=[8],
            n_seqs_per_rank=[1],
            sequence_lengths=[512],
            gradient_checkpointing=[True],
            optimizer_offload=[False],
            reshard_after_forward=[True],
            repeats=0,
        )


def test_enumerate_rejects_sequence_parallelism_before_the_product_loop():
    """Returning an empty sweep for sp>1 looks like a config typo; failing fast makes the cause obvious."""
    with pytest.raises(ProfileValidationError, match="sequence_parallel"):
        enumerate_fsdp_candidates(
            world_size=8,
            shard_sizes=[8],
            n_seqs_per_rank=[1],
            sequence_lengths=[512],
            gradient_checkpointing=[True],
            optimizer_offload=[False],
            reshard_after_forward=[True],
            sequence_parallel_sizes=[2],
        )


def test_enumerate_rejects_non_positive_shard_size():
    """A zero or negative shard_size is never valid; it must raise rather than silently disappear."""
    with pytest.raises(ProfileValidationError, match="shard_size"):
        enumerate_fsdp_candidates(
            world_size=8,
            shard_sizes=[0, 8],
            n_seqs_per_rank=[1],
            sequence_lengths=[512],
            gradient_checkpointing=[True],
            optimizer_offload=[False],
            reshard_after_forward=[True],
        )
