"""The elementwise op: bandwidth-only SOL, a fitted floor, and no wiring into predict()."""

from __future__ import annotations

from collections import Counter

import pytest

from dryrun.cost.kernel.hardware import load_gpu_spec
from dryrun.cost.kernel.ops.base import op_for
from dryrun.cost.kernel.ops.elementwise import ELEMENTWISE_KINDS, ELEMENTWISE_OP


class _RecordingDatabase:
    """Stand in for `KernelDatabase`, recording the `op_name` of every query it prices.

    Returns a constant rather than a shape-dependent latency: this instrument answers which ops
    `predict()` asks for, not what they cost, and any op family added to `predict()` shows up
    here whatever it is named.
    """

    def __init__(self) -> None:
        self.op_names: list[str] = []

    def _record(self, query) -> float:
        self.op_names.append(query.op_name)
        return 1.0e-5

    query_gemm = _record
    query_attention = _record
    query_comm = _record

    def query_elementwise(self, query) -> float:
        """Present so a `predict()` that called it would record rather than raise."""
        return self._record(query)


class _FakeGemmOp:
    """One projection shape, matching `model_ops.GemmOp`'s read surface."""

    def __init__(self, n: int, k: int) -> None:
        self.n = n
        self.k = k


class _FakeModelOps:
    """A two-layer, two-projection op list at tp=1, so the expected multiset is small."""

    tensor_parallel = 1
    num_layers = 2

    def __init__(self) -> None:
        self.projection_ops = (_FakeGemmOp(4096, 4096), _FakeGemmOp(4096, 11008))
        self.logits_op = _FakeGemmOp(151936, 4096)

    def gemm_queries(self, token_count: int, request_count: int):
        """Return `(query, multiplier)` pairs for the projections plus the logits GEMM."""
        from dryrun.cost.kernel.database import GemmQuery

        pairs = [
            (GemmQuery(m=token_count, n=op.n, k=op.k), self.num_layers)
            for op in self.projection_ops
        ]
        pairs.append((GemmQuery(m=request_count, n=self.logits_op.n, k=self.logits_op.k), 1))
        return tuple(pairs)

    def comm_queries(self, token_count: int):
        """Return no collectives; this fixture is tp=1."""
        return ()

    def attention_query(self, phase: str, *, batch_size: int, seq_len: int = 1, step: int = 0, backend=None):
        """Return the attention query of one step at this fixture's head counts."""
        from dryrun.cost.kernel.database import AttentionQuery

        return AttentionQuery(
            phase=phase,
            batch_size=batch_size,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            seq_len=seq_len,
            step=step,
            backend=backend,
        )



@pytest.fixture(scope="module")
def spec():
    return load_gpu_spec("h20")


def test_the_op_is_registered_in_its_own_family():
    assert op_for("elementwise") is ELEMENTWISE_OP
    assert ELEMENTWISE_OP.family == "elementwise"


def test_the_kinds_share_one_schema_because_they_share_one_sol_shape():
    """All four are pure bandwidth; only the bytes touched differ, which n_elements carries."""
    assert ELEMENTWISE_OP.slice_columns == ("dtype", "op_kind")
    assert ELEMENTWISE_OP.axis_columns == ("n_elements",)
    assert set(ELEMENTWISE_KINDS) == {"rms_norm", "rotary_embedding", "silu_and_mul", "qk_norm"}


def test_the_op_takes_a_fitted_floor():
    """Measured: all four kinds sit at 4-5 us flat below 128 tokens, so dispatch dominates."""
    assert ELEMENTWISE_OP.fits_floor is True


def test_sol_has_no_compute_term(spec):
    """An elementwise kernel does no matmul, so a math-bound roofline would be meaningless."""
    got = ELEMENTWISE_OP.sol(spec, slice_key=("bfloat16", "rms_norm"), coords=(4096.0 * 128,))
    assert got.math_s == 0.0
    assert got.memory_s > 0.0


def test_sol_scales_linearly_with_element_count(spec):
    """Bandwidth-bound means doubling the elements doubles the time, exactly."""
    one = ELEMENTWISE_OP.sol(spec, slice_key=("bfloat16", "rms_norm"), coords=(1.0e6,)).memory_s
    two = ELEMENTWISE_OP.sol(spec, slice_key=("bfloat16", "rms_norm"), coords=(2.0e6,)).memory_s
    assert two == pytest.approx(2.0 * one, rel=1e-12)


def test_a_narrower_dtype_moves_fewer_bytes(spec):
    wide = ELEMENTWISE_OP.sol(spec, slice_key=("float32", "rms_norm"), coords=(1.0e6,)).memory_s
    narrow = ELEMENTWISE_OP.sol(spec, slice_key=("bfloat16", "rms_norm"), coords=(1.0e6,)).memory_s
    assert wide == pytest.approx(2.0 * narrow, rel=1e-12)


def test_the_floor_binds_at_a_tiny_element_count(spec):
    got = ELEMENTWISE_OP.sol(spec, slice_key=("bfloat16", "rms_norm"), coords=(64.0,), floor_s=4.5e-6)
    assert got.latency_s == pytest.approx(4.5e-6, rel=1e-12)


def test_the_floor_does_not_bind_at_a_large_element_count(spec):
    got = ELEMENTWISE_OP.sol(spec, slice_key=("bfloat16", "rms_norm"), coords=(1.0e9,), floor_s=4.5e-6)
    assert got.latency_s == pytest.approx(got.memory_s, rel=1e-12)
    assert got.latency_s > 4.5e-6


def test_a_non_positive_element_count_is_rejected(spec):
    with pytest.raises(ValueError, match="n_elements"):
        ELEMENTWISE_OP.sol(spec, slice_key=("bfloat16", "rms_norm"), coords=(0.0,))


def test_an_unknown_kind_is_rejected(spec):
    """A typo'd kind would silently create a new slice with one point in it."""
    with pytest.raises(ValueError, match="rmsnorm"):
        ELEMENTWISE_OP.sol(spec, slice_key=("bfloat16", "rmsnorm"), coords=(1.0e6,))


def test_latency_is_the_last_table_column():
    assert ELEMENTWISE_OP.table_columns() == ("dtype", "op_kind", "n_elements", "latency_s")


def test_parse_sweep_builds_the_token_grid():
    sweep = ELEMENTWISE_OP.parse_sweep({"token_counts": [1, 128, 4096]})
    assert sweep.token_counts == (1, 128, 4096)


def test_parse_sweep_rejects_an_empty_grid():
    from dryrun.profiler.schema import ProfileValidationError

    with pytest.raises(ProfileValidationError, match="token_counts"):
        ELEMENTWISE_OP.parse_sweep({"token_counts": []})


def test_the_composed_model_does_not_add_an_elementwise_term():
    """Measured: adding it takes decode 4.84% -> 11.18% because utility already absorbs it.

    Utility is SOL/measured, and the measured GEMM and attention latencies came from a server
    already running these kernels between them, so a separate term double-counts. If this test
    fails, someone wired an extra term into predict(); read the spec's elementwise section
    before changing this assertion.

    The assertion is behavioural: a spy records the `op_name` of every query one `predict()`
    call issues, and the multiset must be exactly the GEMM and attention ops. Checking the
    source for the word "elementwise" guards a spelling, not the behaviour -- a real
    `_pointwise_seconds` term passed the whole file.
    """
    from dryrun.cost.kernel.database import GemmQuery
    from dryrun.cost.rollout import RolloutWorkload
    from dryrun.cost.rollout.composed import ComposedRollout

    ops = _FakeModelOps()
    spy = _RecordingDatabase()
    model = ComposedRollout(spy, ops, tensor_parallel=1)
    workload = RolloutWorkload(
        phase="decode", n_tokens=8.0, sum_l2=0.0, ctxsum=8192.0, request_count=8
    )

    model.predict(workload)

    gemm_count = len(ops.gemm_queries(8, 8))
    # One attention query, not `num_layers` of them: `predict()` prices one layer and multiplies.
    expected = Counter({"gemm": gemm_count, "generation_attention": 1})
    assert Counter(spy.op_names) == expected, (
        f"predict() priced {sorted(set(spy.op_names))} but must price only the GEMM and "
        f"attention ops. Any additional op double-counts work already inside the fitted "
        f"GEMM and attention utilities."
    )
    # Guard the spy: a database that recorded nothing would make the assertion above vacuous.
    assert gemm_count > 0
    assert isinstance(ops.gemm_queries(8, 8)[0][0], GemmQuery)


def test_the_composed_model_source_never_mentions_elementwise():
    """A cheap second line of defence, kept because it reads at a glance."""
    import inspect

    from dryrun.cost.rollout.composed import ComposedRollout

    source = inspect.getsource(ComposedRollout)
    assert "elementwise" not in source.lower(), (
        "ComposedRollout appears to price elementwise ops; that double-counts work already "
        "inside the fitted GEMM and attention utilities."
    )
    assert not hasattr(ComposedRollout, "_elementwise_seconds")

