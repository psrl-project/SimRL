"""Tests for the physically structured training latency model.

The model separates a training step into three costs that scale differently: work
proportional to a microbatch's tokens, a fixed cost per microbatch, and a fixed cost per
step. These tests pin that separation and the robust aggregation that makes it fittable
against real measurements.
"""

from __future__ import annotations

import pytest

from dryrun.cost.training.base import TrainingParallelism
from dryrun.cost.training.latency import (
    latency_features,
    pipeline_slots,
    robust_step_time,
    step_time_dispersion,
)
from dryrun.profiler.schema import SequenceStats


def _microbatches(seq_length: int, count: int) -> tuple[SequenceStats, ...]:
    return tuple(SequenceStats.from_lengths([seq_length]) for _ in range(count))


# -- pipeline slots ---------------------------------------------------------


def test_slots_reduce_to_the_microbatch_count_without_pipelining():
    assert pipeline_slots(1, 1) == 1
    assert pipeline_slots(4, 1) == 4


def test_slots_add_the_pipeline_bubble():
    """A stage cannot fill the `pp - 1` slots before its first backward arrives."""
    assert pipeline_slots(1, 4) == 4
    assert pipeline_slots(4, 4) == 7
    assert pipeline_slots(8, 2) == 9


def test_slots_reject_nonpositive_input():
    with pytest.raises(ValueError):
        pipeline_slots(0, 1)
    with pytest.raises(ValueError):
        pipeline_slots(1, 0)


# -- design row -------------------------------------------------------------


def test_design_row_separates_the_three_costs():
    """
    The row is `[1, slots * tokens, slots]`.

    The intercept is the per-step cost, the product term is token-proportional work
    repeated per slot, and the bare slot count is the per-microbatch fixed cost. Keeping
    them distinct is what the old single-linear form could not do.
    """
    features = latency_features(
        _microbatches(512, 4), TrainingParallelism(tp=1, pp=1, dp=8, cp=1)
    )
    assert features.tokens == 512
    assert features.slots == 4
    # Four columns: intercept, slots*tokens, slots, slots*sum_squared_tokens. The last is
    # slots * 512^2 here because each microbatch holds a single 512-token sequence.
    assert features.design_row() == (1.0, 4 * 512.0, 4.0, 4 * 512.0**2)


def test_design_row_is_rank_deficient_when_only_the_sequence_varies():
    """
    Sequence length alone cannot separate the per-microbatch from the per-step cost.

    At a fixed microbatch count the slot column is constant and therefore collinear with
    the intercept, so the design cannot reach its full rank of four however many sequence
    lengths are measured. This is why the sweep must vary both axes.
    """
    import numpy as np

    parallelism = TrainingParallelism(tp=1, pp=1, dp=8, cp=1)
    design = np.array(
        [
            latency_features(_microbatches(seq, 4), parallelism).design_row()
            for seq in (512, 1024, 2048, 4096)
        ]
    )
    # Three of four: varying the sequence moves the linear and quadratic token columns
    # independently, but the intercept and slots columns stay collinear.
    assert np.linalg.matrix_rank(design) == 3
    assert design.shape[1] == 4

    with_both = np.array(
        [
            latency_features(_microbatches(seq, nmb), parallelism).design_row()
            for seq in (512, 1024, 2048)
            for nmb in (1, 4)
        ]
    )
    # Both axes varied, three sequence lengths: full rank four. Two lengths would leave the
    # linear and quadratic token columns collinear and cap it at three.
    assert np.linalg.matrix_rank(with_both) == 4


def test_features_use_the_largest_microbatch():
    """The pipeline advances at the speed of its slowest occupant."""
    parallelism = TrainingParallelism(tp=1, pp=1, dp=1, cp=1)
    ragged = (
        SequenceStats.from_lengths([256]),
        SequenceStats.from_lengths([2048]),
    )
    assert latency_features(ragged, parallelism).tokens == 2048


def test_features_reject_an_empty_batch():
    with pytest.raises(ValueError, match="at least one microbatch"):
        latency_features((), TrainingParallelism())


# -- robust aggregation -----------------------------------------------------


def test_step_time_uses_the_median_not_the_mean():
    """
    Step times are right-skewed, so the mean is biased upward.

    Interference and straggling collectives inflate individual steps, but nothing makes a
    step faster than the hardware allows. One profiled run showed a 107.7% spread across
    three retained steps from a single slow outlier.
    """
    clean = (0.500, 0.501, 0.499, 0.500)
    with_outlier = clean + (1.400,)
    assert robust_step_time(with_outlier) == pytest.approx(0.5005, abs=1e-3)
    mean = sum(with_outlier) / len(with_outlier)
    assert mean > 0.6
    assert robust_step_time(with_outlier) < mean


def test_step_time_requires_a_sample():
    with pytest.raises(ValueError, match="at least one"):
        robust_step_time(())


def test_dispersion_flags_a_noisy_candidate():
    """Reported so a measurement problem is visible instead of widening the fit."""
    assert step_time_dispersion((0.500, 0.501, 0.499)) < 0.01
    assert step_time_dispersion((0.500, 0.750, 1.000)) > 0.05


def test_dispersion_is_zero_without_repeats():
    assert step_time_dispersion((0.5,)) == 0.0
    assert step_time_dispersion(()) == 0.0
