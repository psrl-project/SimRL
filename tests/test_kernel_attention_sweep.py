"""Attention sweep enumeration, including the memory-budget truncation."""

from __future__ import annotations

from pathlib import Path

import pytest

from dryrun.profiler.hf import model_descriptor_from_yaml
from dryrun.profiler.kernels.attention import (
    AttentionPoint,
    context_sweep_points,
    generation_sweep_points,
    rank_head_counts,
)
from dryrun.profiler.kernels.config import AttentionSweep
from dryrun.profiler.schema import ProfileValidationError

_FIXTURE_MODEL = Path("tests/fixtures/hf/qwen3_8b").resolve()


@pytest.fixture
def model():
    return model_descriptor_from_yaml(str(_FIXTURE_MODEL), base_dir=Path.cwd())


def test_rank_head_counts_shard_by_tensor_parallel(model):
    assert rank_head_counts(model, tensor_parallel=1) == (32, 8)
    assert rank_head_counts(model, tensor_parallel=2) == (16, 4)


def test_indivisible_tensor_parallel_is_rejected(model):
    with pytest.raises(ProfileValidationError):
        rank_head_counts(model, tensor_parallel=3)


def test_context_budget_truncates_the_expensive_corner():
    sweep = AttentionSweep(
        batch_sizes=(1, 2, 4),
        context_seq_lens=(1024, 8192),
        generation_steps=(0,),
        max_context_tokens=16384,
    )
    points = context_sweep_points(sweep)
    pairs = {(point.batch_size, point.seq_len) for point in points}
    assert (4, 8192) not in pairs, "batch 4 x seq 8192 is 32768 tokens, over the budget."
    assert (2, 8192) in pairs
    assert (4, 1024) in pairs
    assert len(pairs) == 5


def test_context_points_carry_step_zero():
    sweep = AttentionSweep(
        batch_sizes=(1,), context_seq_lens=(1024,), generation_steps=(0,)
    )
    assert context_sweep_points(sweep) == (AttentionPoint(batch_size=1, seq_len=1024, step=0),)


def test_generation_points_carry_unit_seq_len():
    sweep = AttentionSweep(
        batch_sizes=(4,), context_seq_lens=(1024,), generation_steps=(0, 2048)
    )
    points = generation_sweep_points(sweep, num_kv_heads=8, head_dim=128)
    assert all(point.seq_len == 1 for point in points)
    assert {point.step for point in points} == {0, 2048}


def test_generation_budget_truncates_by_kv_element_count():
    sweep = AttentionSweep(
        batch_sizes=(1, 64),
        context_seq_lens=(1024,),
        generation_steps=(1024, 65536),
        max_generation_kv_tokens=1 << 22,  # 4_194_304 elements
    )
    points = generation_sweep_points(sweep, num_kv_heads=8, head_dim=128)
    pairs = {(point.batch_size, point.step) for point in points}
    # Footprints: batch * max(step, 1) * 8 * 128
    #   (1,  1024):  1 *  1024 * 8 * 128 =   1_048_576  -> under budget, survives
    #   (1, 65536):  1 * 65536 * 8 * 128 =  67_108_864  -> over budget, dropped
    #  (64,  1024): 64 *  1024 * 8 * 128 =  67_108_864  -> over budget, dropped
    #  (64, 65536): 64 * 65536 * 8 * 128 = 4_294_967_296 -> over budget, dropped
    assert (1, 1024) in pairs
    assert (1, 65536) not in pairs
    assert (64, 1024) not in pairs
    assert (64, 65536) not in pairs
    assert len(pairs) == 1


def test_points_are_ordered_largest_first():
    sweep = AttentionSweep(
        batch_sizes=(1, 4), context_seq_lens=(1024, 2048), generation_steps=(0,)
    )
    points = context_sweep_points(sweep)
    footprints = [point.batch_size * point.seq_len for point in points]
    assert footprints == sorted(footprints, reverse=True)


def test_generation_points_are_ordered_largest_first():
    """Generation points are ordered by batch_size * max(step, 1), largest first.

    Including step=0 pins that zero-step points sort last (footprint 1) rather than
    first, which is the expected behavior per the module docstring.
    """
    sweep = AttentionSweep(
        batch_sizes=(1, 4),
        context_seq_lens=(1024,),
        generation_steps=(0, 512, 2048),
    )
    points = generation_sweep_points(sweep, num_kv_heads=1, head_dim=1)
    footprints = [point.batch_size * max(point.step, 1) for point in points]
    assert footprints == sorted(footprints, reverse=True)


def test_default_budgets_cover_the_whole_h20_qwen3_8b_sweep():
    """
    Guard the raggedness budgets against being too small to measure.

    A budget below the largest configured point drops sweep rows silently, and a sparse
    grid starves the holdout gate of interior points. The grid below is the one the H20
    Qwen3-8B configs sweep, and every point of it must survive the defaults: the largest
    decode point is 8 GiB of KV against the roughly 75 GiB an H20 has free.
    """
    batch_sizes = (1, 2, 4, 8, 16, 32, 48, 64)
    generation_steps = (0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
    sweep = AttentionSweep(
        batch_sizes=batch_sizes,
        context_seq_lens=(128, 256, 512, 1024, 2048, 4096, 8192, 16384),
        generation_steps=generation_steps,
    )
    points = generation_sweep_points(sweep, num_kv_heads=8, head_dim=128)
    assert len(points) == len(batch_sizes) * len(generation_steps)
