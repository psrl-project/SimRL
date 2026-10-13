"""The optimizer-step op: its roofline, its query, and the guard on an unmeasurable shard.

The op is structurally the simplest in the tree (one axis, no token dependence), so the
tests that matter are about what it refuses rather than what it computes. In particular a
shard too large for the card produces a *fast* row rather than an error, and nothing
downstream rejects a fast row, so the refusal has to happen at measurement time.
"""

from __future__ import annotations

import pytest

from dryrun.cost.kernel.database import OptimizerQuery
from dryrun.cost.kernel.hardware import load_gpu_spec
from dryrun.cost.kernel.interp import KernelCoverageError
from dryrun.cost.kernel.ops.base import op_for
from dryrun.cost.kernel.ops.optimizer import BYTES_PER_PARAMETER, OPTIMIZER_STEP_OP
from dryrun.profiler.kernel.sweep import OptimizerSweep
from dryrun.profiler.schema import ProfileValidationError

_SLICE = ("adam", "master_fp32")


def test_the_op_is_registered_under_its_own_family():
    """`floor.py` and `holdout.py` discover it through the registry with no edit of their own."""
    assert op_for("optimizer_step") is OPTIMIZER_STEP_OP
    assert OPTIMIZER_STEP_OP.family == "optimizer"


def test_the_only_axis_is_the_parameter_count():
    """An Adam step's cost does not depend on how many tokens produced the gradients.

    This is the one op in the tree with no token or shape axis, and declaring a token axis
    would invite a sweep that varies something the kernel cannot see.
    """
    assert OPTIMIZER_STEP_OP.axis_columns == ("param_count",)
    assert OPTIMIZER_STEP_OP.slice_columns == ("optimizer", "state_kind")


def test_the_roofline_is_pure_bandwidth_with_no_compute_term():
    """Adam does a handful of FLOPs per parameter against 28 bytes of traffic.

    That is four orders of magnitude from the compute roof, so a math term would always be
    discarded by the `max` and declaring it zero says so explicitly.
    """
    spec = load_gpu_spec("h20")
    components = OPTIMIZER_STEP_OP.sol(
        spec, slice_key=_SLICE, coords=(1_000_000.0,), floor_s=0.0
    )
    assert components.math_s == 0.0
    assert components.memory_s > 0.0


def test_the_roofline_charges_the_mixed_precision_state_traffic():
    """Read bf16 grad + fp32 master + two fp32 moments, write all four back: 28 bytes.

    Derived from the state layout rather than from the parameter's dtype, because which
    tensors are fp32 is a property of the optimizer's configuration.
    """
    spec = load_gpu_spec("h20")
    param_count = 1_000_000_000
    components = OPTIMIZER_STEP_OP.sol(
        spec, slice_key=_SLICE, coords=(float(param_count),), floor_s=0.0
    )
    expected = param_count * BYTES_PER_PARAMETER / spec.memory_bandwidth_bytes_per_s
    assert components.memory_s == pytest.approx(expected)


def test_the_roofline_is_linear_in_the_parameter_count():
    """A bandwidth-bound kernel's SOL is a straight line through the origin."""
    spec = load_gpu_spec("h20")

    def sol(count: float) -> float:
        return OPTIMIZER_STEP_OP.sol(
            spec, slice_key=_SLICE, coords=(count,), floor_s=0.0
        ).memory_s

    assert sol(2_000_000.0) == pytest.approx(2.0 * sol(1_000_000.0))


def test_the_floor_is_applied_inside_the_max_rather_than_added():
    spec = load_gpu_spec("h20")
    floor_s = 12e-6
    tiny = OPTIMIZER_STEP_OP.sol(spec, slice_key=_SLICE, coords=(1.0,), floor_s=floor_s)
    assert tiny.latency_s == pytest.approx(floor_s)


def test_an_unknown_state_layout_is_refused_rather_than_priced():
    """bf16 moments move different traffic, so they are a separate slice, not an interpolation."""
    spec = load_gpu_spec("h20")
    with pytest.raises(ValueError, match="Unknown optimizer state_kind"):
        OPTIMIZER_STEP_OP.sol(
            spec, slice_key=("adam", "bf16_moments"), coords=(1_000_000.0,), floor_s=0.0
        )


def test_a_non_positive_parameter_count_is_refused():
    spec = load_gpu_spec("h20")
    with pytest.raises(ValueError, match="param_count must be positive"):
        OPTIMIZER_STEP_OP.sol(spec, slice_key=_SLICE, coords=(0.0,), floor_s=0.0)


def test_the_query_defaults_to_the_training_framework_not_the_inference_server():
    """vLLM runs no optimizer, so the database's `vllm` default would name no directory."""
    assert OptimizerQuery(param_count=1_000_000).backend == "megatron"


def test_the_query_refuses_a_non_positive_shard():
    with pytest.raises(KernelCoverageError, match="param_count must be positive"):
        OptimizerQuery(param_count=0)


def test_the_query_refuses_an_unknown_state_layout():
    with pytest.raises(KernelCoverageError, match="Unknown optimizer state_kind"):
        OptimizerQuery(param_count=1_000_000, state_kind="bf16_moments")


def test_the_sweep_parses_parameter_counts_and_refuses_a_non_positive_one():
    assert OptimizerSweep.from_dict({"param_counts": [1_000, 2_000]}).param_counts == (
        1_000,
        2_000,
    )
    with pytest.raises(ProfileValidationError, match="optimizer.param_counts"):
        OptimizerSweep.from_dict({"param_counts": [1_000, 0]})


class _FakeTensor:
    def __init__(self, numel: int) -> None:
        self._numel = numel

    def numel(self) -> int:
        return self._numel


class _FakeOptimizer:
    """Stands in for `FusedAdam` to exercise the state check without allocating 30 GiB."""

    def __init__(self, parameter: object, state: dict[str, object]) -> None:
        self.state = {parameter: state}


def test_an_unallocated_optimizer_state_is_refused_rather_than_timed():
    """The failure mode this exists for records a very *fast* row, not an error.

    A 4.096B-parameter shard needs ~115 GiB against H20's 95 GiB. The allocating `step()`
    OOM'd inside TE, the exception was swallowed, and the timed `step()` then iterated an
    empty state list and returned in 0.60 ms where 2.048B parameters took 21.56 ms. That
    row implied 190 TB/s against a 4.0 TB/s card. Nothing downstream rejects a fast row: a
    floor fit accepts it and the holdout MRE *improves*, because the point is easy to
    interpolate.
    """
    from dryrun.profiler.kernel.runner.optimizer import _require_allocated_state

    parameter = object()
    with pytest.raises(ProfileValidationError, match="was not allocated"):
        _require_allocated_state(_FakeOptimizer(parameter, {}), parameter, 4_096_000_000)


def test_a_partially_allocated_optimizer_state_is_refused():
    """All three of master, exp_avg and exp_avg_sq must be present."""
    from dryrun.profiler.kernel.runner.optimizer import _require_allocated_state

    parameter = object()
    state = {"exp_avg": _FakeTensor(1_000), "master_param": _FakeTensor(1_000)}
    with pytest.raises(ProfileValidationError, match="exp_avg_sq"):
        _require_allocated_state(_FakeOptimizer(parameter, state), parameter, 1_000)


def test_a_state_tensor_of_the_wrong_length_is_refused():
    """A short state tensor means the timed step measures a different size than the grid names."""
    from dryrun.profiler.kernel.runner.optimizer import _require_allocated_state

    parameter = object()
    state = {name: _FakeTensor(500) for name in ("exp_avg", "exp_avg_sq", "master_param")}
    with pytest.raises(ProfileValidationError, match="holds 500 elements"):
        _require_allocated_state(_FakeOptimizer(parameter, state), parameter, 1_000)


def test_a_fully_allocated_state_passes():
    from dryrun.profiler.kernel.runner.optimizer import _require_allocated_state

    parameter = object()
    state = {name: _FakeTensor(1_000) for name in ("exp_avg", "exp_avg_sq", "master_param")}
    _require_allocated_state(_FakeOptimizer(parameter, state), parameter, 1_000)


def test_the_shard_divides_by_every_parallelism_degree_that_splits_parameters():
    """tp, pp and (under the distributed optimizer) dp each reduce what one rank updates."""
    from dryrun.cost.training.training_ops import TrainingOps
    from dryrun.profiler.schema import ModelDescriptor

    model = ModelDescriptor(
        name="test",
        architecture="Qwen3ForCausalLM",
        num_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        parameter_count=8_192_000_000,
        extra={"intermediate_size": 12288, "head_dim": 128, "qk_layernorm": True,
               "vocab_size": 151936},
    )
    ops = TrainingOps(model, tensor_parallel=2, dp=4, distributed_optimizer=True)
    query = ops.optimizer_query(pipeline_parallel=2)
    assert query is not None
    assert query.param_count == 8_192_000_000 // (2 * 2 * 4)

    # Without the distributed optimizer every DP rank updates the same parameters.
    replicated = TrainingOps(model, tensor_parallel=2, dp=4, distributed_optimizer=False)
    replicated_query = replicated.optimizer_query(pipeline_parallel=2)
    assert replicated_query is not None
    assert replicated_query.param_count == 8_192_000_000 // (2 * 2)


def test_an_unknown_parameter_count_omits_the_term_rather_than_refusing_to_predict():
    """A descriptor built without a weight index may carry no count.

    Refusing the whole prediction over a term measured at 0.3% would trade a small known
    error for a total one. The decomposition reports the zero.
    """
    from dryrun.cost.training.training_ops import TrainingOps
    from dryrun.profiler.schema import ModelDescriptor

    model = ModelDescriptor(
        name="test",
        architecture="Qwen3ForCausalLM",
        num_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        extra={"intermediate_size": 12288, "head_dim": 128, "vocab_size": 151936},
    )
    assert TrainingOps(model, tensor_parallel=2).optimizer_query(pipeline_parallel=1) is None
