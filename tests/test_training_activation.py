"""Tests for the architecture-derived training activation model.

These lock in the facts the model was built on, each measured on 8x H20 with
Qwen3-8B under Megatron-Bridge 0.4.2 and full activation recompute. They are the
guard against the previous model's failure mode, where an opaque token-count feature
with a fitted per-model slope hid the fact that it was tracking allocator noise.
"""

from __future__ import annotations

import pytest

from dryrun.cost.training.activation import (
    LOGIT_OVERHEAD,
    ActivationGeometry,
    activation_bytes,
    in_flight_microbatches,
)
from dryrun.cost.training.base import TrainingParallelism
from dryrun.cost.training.inventory import (
    heaviest_stage_parameters,
    parameter_inventory,
)
from dryrun.profiler.schema import ModelDescriptor, SequenceStats

GiB = 2**30
QWEN3_8B_PARAMS = 8_190_735_360


def _qwen3_8b_geometry(num_layers: int = 36) -> ActivationGeometry:
    return ActivationGeometry(
        num_layers=num_layers,
        hidden_size=4096,
        vocab_size=151936,
    )


def _microbatches(seq_length: int, count: int) -> tuple[SequenceStats, ...]:
    return tuple(SequenceStats.from_lengths([seq_length]) for _ in range(count))


# -- static inventory -------------------------------------------------------


def test_gradient_buffer_is_not_sharded_by_data_parallelism():
    """
    The gradient buffer is the reduce-scatter source, so it is full size on every rank.

    Probing the live tensors at TP1/PP1/DP8 reported `param_data` at 15.256 GiB and
    `grad_data` at 30.513 GiB -- exactly 2 and 4 bytes per parameter with no division
    -- while the fp32 master shard was 3.814 GiB, exactly 4 bytes per parameter over
    eight ranks. Dividing gradients by the data-parallel degree understated the true
    footprint roughly twofold.
    """
    inventory = parameter_inventory(
        parameter_count=QWEN3_8B_PARAMS,
        parallelism=TrainingParallelism(tp=1, pp=1, dp=8, cp=1),
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        distributed_optimizer=True,
    )
    assert inventory.model_parameters == pytest.approx(QWEN3_8B_PARAMS * 2)
    assert inventory.gradients == pytest.approx(QWEN3_8B_PARAMS * 4)
    assert inventory.master_parameters == pytest.approx(QWEN3_8B_PARAMS * 4 / 8)
    assert inventory.optimizer_moments == pytest.approx(QWEN3_8B_PARAMS * 2 * 4 / 8)


def test_static_inventory_matches_the_measured_allocator_total():
    """
    The steady-state total reproduces the measured allocator reading.

    At TP1/PP1/DP8 the allocator reported 49.584 GiB live right after construction,
    before Adam's moments existed. Excluding those moments, the inventory must land on
    the same number.
    """
    inventory = parameter_inventory(
        parameter_count=QWEN3_8B_PARAMS,
        parallelism=TrainingParallelism(tp=1, pp=1, dp=8, cp=1),
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        distributed_optimizer=True,
    )
    before_first_step = inventory.total_bytes - inventory.optimizer_moments
    assert before_first_step / GiB == pytest.approx(49.584, abs=0.01)


def test_optimizer_sharding_only_affects_optimizer_state():
    """Turning off the distributed optimizer must not change params or gradients."""
    common = {
        "parameter_count": QWEN3_8B_PARAMS,
        "parallelism": TrainingParallelism(tp=1, pp=1, dp=8, cp=1),
        "parameter_dtype": "bf16",
        "gradient_dtype": "fp32",
        "optimizer_state_dtype": "fp32",
    }
    sharded = parameter_inventory(**common, distributed_optimizer=True)
    replicated = parameter_inventory(**common, distributed_optimizer=False)
    assert sharded.model_parameters == replicated.model_parameters
    assert sharded.gradients == replicated.gradients
    assert replicated.optimizer_moments == pytest.approx(sharded.optimizer_moments * 8)
    assert replicated.total_bytes > sharded.total_bytes


# -- activation geometry ----------------------------------------------------


def test_per_token_slope_matches_the_measured_thirty_six_layer_slope():
    """
    Measured peak allocated bytes fit 624403 bytes per token at 36 layers.

    Solving the 2-layer and 36-layer slopes jointly put the layer coefficient at
    1.0006, so the layer term is analytic; the remainder is the calibrated loss-path
    overhead on the logits.
    """
    geometry = _qwen3_8b_geometry()
    per_token = geometry.checkpoint_bytes_per_token(1) + geometry.logit_bytes_per_token()
    assert per_token == pytest.approx(624403, rel=0.01)


def test_per_token_slope_matches_the_measured_two_layer_slope():
    """The same geometry reproduces the 2-layer smoke slope of 345331 bytes per token."""
    geometry = _qwen3_8b_geometry(num_layers=2)
    per_token = geometry.checkpoint_bytes_per_token(1) + geometry.logit_bytes_per_token()
    assert per_token == pytest.approx(345331, rel=0.01)


def test_layer_checkpoint_term_is_one_hidden_tensor_per_layer():
    """Full uniform recompute at one-layer granularity retains only layer inputs."""
    geometry = _qwen3_8b_geometry()
    expected = (36 + 1) * 4096 * 2
    assert geometry.checkpoint_bytes_per_token(1) == pytest.approx(expected)


def test_logit_term_carries_the_calibrated_loss_overhead():
    """Logits are not recomputed, so they survive the forward pass."""
    geometry = _qwen3_8b_geometry()
    assert geometry.logit_bytes_per_token() == pytest.approx(
        151936 * 2 * LOGIT_OVERHEAD
    )


def test_tensor_parallelism_divides_checkpoints_but_not_logits():
    """
    Only the layer checkpoints shrink with tensor parallelism.

    Measured per-token slopes fall from 624202 bytes at TP=1 to 398009 at TP=4 and
    354868 at TP=8. Dividing both terms would predict 156059 and 78029, so the logit
    footprint is replicated across tensor-parallel ranks.
    """
    geometry = _qwen3_8b_geometry()
    assert geometry.checkpoint_bytes_per_token(4) == pytest.approx(
        geometry.checkpoint_bytes_per_token(1) / 4
    )
    assert geometry.logit_bytes_per_token() == pytest.approx(151936 * 2 * LOGIT_OVERHEAD)


def test_geometry_round_trips_through_coefficients():
    geometry = _qwen3_8b_geometry()
    assert ActivationGeometry.from_dict(geometry.to_dict()) == geometry


def test_geometry_requires_vocab_size():
    """A missing vocabulary would silently drop the dominant activation term."""
    model = ModelDescriptor(name="m", num_layers=8, hidden_size=1024)
    with pytest.raises(ValueError, match="vocab_size"):
        ActivationGeometry.from_model(model)


# -- in-flight microbatches -------------------------------------------------


def test_microbatch_count_does_not_change_activation_without_pipelining():
    """
    At PP=1 the microbatch count is irrelevant to peak memory.

    1F1B runs one microbatch's forward and backward to completion before the next
    begins, so only one is ever live. The measurement agrees exactly: at PP=1 the peak
    allocated bytes for one and four microbatches were identical to the byte.
    """
    geometry = _qwen3_8b_geometry()
    parallelism = TrainingParallelism(tp=1, pp=1, dp=8, cp=1)
    single = activation_bytes(geometry, _microbatches(512, 1), parallelism)
    many = activation_bytes(geometry, _microbatches(512, 4), parallelism)
    assert single == pytest.approx(many)


def test_pipelining_keeps_up_to_pipeline_depth_microbatches_live():
    assert in_flight_microbatches(1, 4) == 1
    assert in_flight_microbatches(4, 4) == 4
    # Deeper local queues do not add concurrency beyond the pipeline depth.
    assert in_flight_microbatches(16, 4) == 4


def test_activation_scales_with_sequence_length():
    geometry = _qwen3_8b_geometry()
    parallelism = TrainingParallelism(tp=1, pp=1, dp=8, cp=1)
    short = activation_bytes(geometry, _microbatches(512, 1), parallelism)
    long = activation_bytes(geometry, _microbatches(1024, 1), parallelism)
    assert long == pytest.approx(short * 2)


def test_activation_uses_the_largest_microbatch_for_ragged_batches():
    """Peak memory is set by the worst microbatch, not the average."""
    geometry = _qwen3_8b_geometry()
    parallelism = TrainingParallelism(tp=1, pp=1, dp=1, cp=1)
    ragged = (
        SequenceStats.from_lengths([256]),
        SequenceStats.from_lengths([2048]),
    )
    uniform = _microbatches(2048, 2)
    assert activation_bytes(geometry, ragged, parallelism) == pytest.approx(
        activation_bytes(geometry, uniform, parallelism)
    )


def test_activation_rejects_an_empty_batch():
    geometry = _qwen3_8b_geometry()
    with pytest.raises(Exception, match="at least one microbatch"):
        activation_bytes(geometry, (), TrainingParallelism())


def test_full_prediction_reproduces_the_measured_peak():
    """
    Static state plus activation reproduces the measured peak within the fitted offset.

    The measured peak allocated at TP1/PP1/DP8, sequence 512, was 58.766 GiB. The two
    analytic terms should account for all but the small constant the fit absorbs, which
    came out near 1.25 GiB on both an independent one-GPU and eight-GPU dataset.
    """
    parallelism = TrainingParallelism(tp=1, pp=1, dp=8, cp=1)
    static = parameter_inventory(
        parameter_count=QWEN3_8B_PARAMS,
        parallelism=parallelism,
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        distributed_optimizer=True,
    ).total_bytes
    activation = activation_bytes(
        _qwen3_8b_geometry(), _microbatches(512, 1), parallelism
    )
    predicted = (static + activation) / GiB
    assert predicted == pytest.approx(58.766, abs=1.5)


# -- pipeline stage imbalance ----------------------------------------------


def test_pipeline_stages_are_not_uniform():
    """
    The heaviest pipeline stage carries an embedding on top of its layers.

    Megatron splits transformer layers evenly, but the embedding sits on the first
    stage and the output projection on the last, and neither is a transformer layer.
    Peak memory is set by the worst stage, so a uniform share understates it. Measured
    at PP=4 on Qwen3-8B, assuming uniform predicted 24.33 GiB where the heaviest rank
    held 27.81 GiB; accounting for the embedding brings it to 27.81 GiB.
    """
    parallelism = TrainingParallelism(tp=1, pp=4, dp=2, cp=1)
    uniform = QWEN3_8B_PARAMS / 4
    heaviest = heaviest_stage_parameters(
        parameter_count=QWEN3_8B_PARAMS,
        num_layers=36,
        hidden_size=4096,
        vocab_size=151936,
        parallelism=parallelism,
    )
    assert heaviest > uniform
    # Nine of thirty-six layers plus one vocabulary-sized tensor.
    embedding = 4096 * 151936
    expected = (QWEN3_8B_PARAMS - 2 * embedding) / 4 + embedding
    assert heaviest == pytest.approx(expected)


def test_heaviest_stage_reproduces_the_measured_pipeline_peak():
    """Static plus activation must reproduce the PP=4 measurement within the offset."""
    parallelism = TrainingParallelism(tp=1, pp=4, dp=2, cp=1)
    static = parameter_inventory(
        parameter_count=QWEN3_8B_PARAMS,
        parallelism=parallelism,
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        distributed_optimizer=True,
        local_parameters=heaviest_stage_parameters(
            parameter_count=QWEN3_8B_PARAMS,
            num_layers=36,
            hidden_size=4096,
            vocab_size=151936,
            parallelism=parallelism,
        ),
    ).total_bytes
    activation = activation_bytes(
        _qwen3_8b_geometry(), _microbatches(512, 1), parallelism
    )
    # 1.257 GiB is the constant the fit recovered on two independent datasets.
    predicted = (static + activation) / GiB + 1.257
    assert predicted == pytest.approx(27.81, abs=0.2)


def test_without_pipelining_the_share_stays_uniform():
    """Tensor parallelism splits every tensor, so there is no imbalance to model."""
    parallelism = TrainingParallelism(tp=4, pp=1, dp=2, cp=1)
    assert heaviest_stage_parameters(
        parameter_count=QWEN3_8B_PARAMS,
        num_layers=36,
        hidden_size=4096,
        vocab_size=151936,
        parallelism=parallelism,
    ) == pytest.approx(QWEN3_8B_PARAMS / 4)


def test_heaviest_stage_rejects_an_inconsistent_architecture():
    """Embeddings larger than the whole model mean the descriptor is wrong."""
    with pytest.raises(ValueError, match="Embeddings cannot exceed"):
        heaviest_stage_parameters(
            parameter_count=10_000_000,
            num_layers=8,
            hidden_size=1024,
            vocab_size=32_000,
            parallelism=TrainingParallelism(tp=1, pp=2, dp=1, cp=1),
        )


def test_heaviest_stage_requires_divisible_layers():
    with pytest.raises(ValueError, match="not divisible"):
        heaviest_stage_parameters(
            parameter_count=QWEN3_8B_PARAMS,
            num_layers=36,
            hidden_size=4096,
            vocab_size=151936,
            parallelism=TrainingParallelism(tp=1, pp=8, dp=1, cp=1),
        )


def test_peak_pipeline_stage_is_insensitive_to_microbatch_count():
    """
    Under 1F1B the warmup depth and the logits sit on opposite pipeline ends.

    The first stage keeps up to `pp` microbatches in flight but holds no logits; the
    last stage holds the logits but backpropagates each microbatch immediately, so its
    warmup depth is one. The worst stage is therefore the larger of the two, and for a
    large vocabulary that is the last one regardless of microbatch count. Measured at
    PP=4, the peak rank read 27.81 GiB for both one and four microbatches; scaling the
    peak stage's checkpoints by the in-flight count would have predicted 0.56 GiB of
    difference.
    """
    geometry = _qwen3_8b_geometry()
    parallelism = TrainingParallelism(tp=1, pp=4, dp=2, cp=1)
    single = activation_bytes(geometry, _microbatches(512, 1), parallelism)
    many = activation_bytes(geometry, _microbatches(512, 4), parallelism)
    assert single == pytest.approx(many)


def test_deep_pipeline_warmup_can_dominate_a_small_vocabulary():
    """
    With a small vocabulary the warmup end becomes the worst stage.

    This is the branch the Qwen3-8B measurements never exercise, so it is asserted
    directly: the model must compare both ends rather than assume the logits win.
    """
    geometry = ActivationGeometry(num_layers=64, hidden_size=8192, vocab_size=256)
    parallelism = TrainingParallelism(tp=1, pp=2, dp=1, cp=1)
    single = activation_bytes(geometry, _microbatches(1024, 1), parallelism)
    many = activation_bytes(geometry, _microbatches(1024, 2), parallelism)
    assert many > single
