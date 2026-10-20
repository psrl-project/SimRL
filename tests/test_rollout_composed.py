"""ComposedRollout: step composition, segment integration, and its refusals."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from srl.cost.kernel.database import AttentionQuery, GemmQuery, KernelDatabase
from srl.cost.kernel.model_ops import ModelOps
from srl.cost.rollout import ROLLOUT_MODEL_REGISTRY, RolloutWorkload
from srl.cost.rollout.composed import (
    _MAX_CACHE_ENTRIES,
    _SEGMENT_CAP,
    CLOSED_FORM_DISABLED,
    ComposedRollout,
)
from srl.cost.rollout.segment import DEFAULT_STRIDE

# Anchored to the repo root, not the working directory: `outputs/` is gitignored, so
# these are real local artifacts. Absent them the tests skip rather than fail, matching
# tests/test_hf_model.py's REAL_QWEN convention.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CHECKPOINT = Path("/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B")
_TABLES = _REPO_ROOT / "outputs/profile/qwen3_8b/kernel_tp1/kernel"

pytestmark = pytest.mark.skipif(
    not (_CHECKPOINT / "config.json").exists() or not _TABLES.is_dir(),
    reason="Needs the real Qwen3-8B checkpoint and the profiled kernel tables.",
)


@pytest.fixture(scope="module")
def ops():
    return ModelOps.from_checkpoint(str(_CHECKPOINT), tensor_parallel=1)


@pytest.fixture(scope="module")
def database():
    return KernelDatabase.from_root(str(_TABLES), gpu="h20", utility_surface="pchip")


@pytest.fixture
def model(database, ops):
    return ComposedRollout(database, ops, tensor_parallel=1)


def test_the_registry_exposes_the_model():
    assert ROLLOUT_MODEL_REGISTRY["composed"] is ComposedRollout


def test_a_decode_step_equals_the_hand_composed_sum(model, database, ops):
    """Nothing may be dropped or double-counted relative to an explicit sum."""
    n_d, ctx = 16, 2048
    expected = ops.num_layers * (
        sum(database.query_gemm(GemmQuery(m=n_d, n=op.n, k=op.k)) for op in ops.projection_ops)
        + database.query_attention(
            AttentionQuery(
                phase="generation",
                batch_size=n_d,
                num_heads=ops.num_heads,
                num_kv_heads=ops.num_kv_heads,
                head_dim=ops.head_dim,
                step=ctx,
            )
        )
    ) + database.query_gemm(
        GemmQuery(m=n_d, n=ops.logits_op.n, k=ops.logits_op.k)
    )
    workload = RolloutWorkload(
        phase="decode", n_tokens=n_d, sum_l2=0.0, ctxsum=float(n_d * ctx), request_count=n_d
    )
    assert model.predict(workload) == pytest.approx(expected, rel=1e-12)


def test_the_logits_projection_is_counted_once_not_once_per_layer(model, database, ops):
    """A 36x error here would still look like a plausible latency."""
    n_d = 8
    workload = RolloutWorkload(
        phase="decode", n_tokens=n_d, sum_l2=0.0, ctxsum=float(n_d * 1024), request_count=n_d
    )
    logits = database.query_gemm(GemmQuery(m=n_d, n=ops.logits_op.n, k=ops.logits_op.k))
    assert model.predict(workload) < ops.num_layers * logits


def test_a_prefill_step_runs_its_gemms_over_all_prompt_tokens(model):
    """Prefill's m is the token count. Decode's is the request count."""
    reqs, tokens = 4, 4096
    prefill = model.predict(
        RolloutWorkload(
            phase="prefill",
            n_tokens=float(tokens),
            sum_l2=float(reqs * (tokens // reqs) ** 2),
            ctxsum=0.0,
            request_count=reqs,
        )
    )
    decode = model.predict(
        RolloutWorkload(phase="decode", n_tokens=reqs, sum_l2=0.0, ctxsum=4096.0, request_count=reqs)
    )
    assert prefill > 10 * decode


def test_a_mixed_step_costs_more_than_either_pure_phase(model):
    """The documented approximation: context attention plus generation attention."""
    reqs = 8
    mixed = model.predict(
        RolloutWorkload(
            phase="mixed",
            n_tokens=2048.0,
            sum_l2=float(reqs * 256**2),
            ctxsum=float(reqs * 4096),
            request_count=reqs,
        )
    )
    prefill = model.predict(
        RolloutWorkload(
            phase="prefill", n_tokens=2048.0, sum_l2=float(reqs * 256**2), ctxsum=0.0, request_count=reqs
        )
    )
    decode_only = model.predict(
        RolloutWorkload(
            phase="decode", n_tokens=float(reqs), sum_l2=0.0, ctxsum=float(reqs * 4096), request_count=reqs
        )
    )
    assert mixed > prefill
    assert mixed > decode_only


def test_every_prediction_is_strictly_positive(model):
    for n_d in (1, 4, 16, 64):
        for ctx in (128, 4096, 32768):
            value = model.predict(
                RolloutWorkload(
                    phase="decode",
                    n_tokens=n_d,
                    sum_l2=0.0,
                    ctxsum=float(n_d * ctx),
                    request_count=n_d,
                )
            )
            assert value > 0.0


def test_a_short_segment_costs_only_a_couple_of_predictions(model):
    """instance.py builds an integrator per decode event, so short segments must be cheap."""
    calls = []
    original = model.predict

    def counting(workload):
        calls.append(workload)
        return original(workload)

    model.predict = counting
    try:
        model.decode_segment_time(16, 32768.0, 300)
    finally:
        model.predict = original
    assert len(calls) == 2, f"a 300-step segment made {len(calls)} predictions"


def test_a_segment_is_integrated_not_evaluated_per_step(model):
    """A 4096-step segment must cost a handful of table lookups, not 4096 predictions."""
    calls = []
    original = model.predict

    def counting(workload):
        calls.append(workload)
        return original(workload)

    model.predict = counting
    try:
        model.decode_segment_time(16, 32768.0, 4096)
    finally:
        model.predict = original
    assert len(calls) <= 8, f"Integrating 4096 steps made {len(calls)} predictions."


def test_step_cost_memo_deduplicates_across_integrators(model):
    """Asking for the same segment twice must not double the predict() count.

    `advance_to` builds two integrators with the same (n_d, ctxsum): one for
    `decode_steps_within` and one for `decode_segment_time`. Without the memo,
    identical (n_d, ctxsum, offset) triples are evaluated twice: a 1.75x
    measured redundancy. The bounded dict cache on `_step_seconds` must collapse
    that to 1.00x.
    """
    calls = []
    original = model.predict

    def counting(workload):
        calls.append(workload.ctxsum)
        return original(workload)

    model.predict = counting
    try:
        # First call populates the cache for these (n_d, ctxsum, offset) triples.
        model.decode_segment_time(16, 32768.0, 300)
        first_count = len(calls)
        # Second call with identical arguments must hit the cache. predict() must not
        # be called again.
        model.decode_segment_time(16, 32768.0, 300)
        second_count = len(calls)
    finally:
        model.predict = original

    assert first_count > 0, "First call made no predictions."
    assert second_count == first_count, (
        f"Second identical segment call added {second_count - first_count} extra "
        f"predict() calls. Expected 0 (all results already cached)."
    )


def test_the_segment_time_grows_strictly_with_the_step_count(model):
    previous = model.decode_segment_time(16, 32768.0, 0)
    assert previous == 0.0
    for k in (1, 100, 1024, 1025, 2048, 4096):
        current = model.decode_segment_time(16, 32768.0, k)
        assert current > previous
        previous = current


def test_steps_within_exactly_inverts_the_segment_time(model):
    """instance.py compares the returned count to its budget by integer identity."""
    for k in (1, 7, 1023, 1024, 1025, 2048, 4096):
        budget = model.decode_segment_time(16, 32768.0, k)
        assert model.decode_steps_within(16, 32768.0, budget) == k


def test_steps_within_returns_zero_for_an_exhausted_budget(model):
    assert model.decode_steps_within(16, 32768.0, 0.0) == 0
    assert model.decode_steps_within(16, 32768.0, -1.0) == 0


def test_the_segment_matches_a_step_by_step_sum_within_one_percent(model):
    """The lattice is an approximation. This bounds it against the exact reference."""
    n_d, ctx0, k = 16, 2048.0, 2048
    exact = sum(
        model.predict(
            RolloutWorkload(
                phase="decode",
                n_tokens=n_d,
                sum_l2=0.0,
                ctxsum=float(n_d) * (ctx0 + i),
                request_count=n_d,
            )
        )
        for i in range(k)
    )
    integrated = model.decode_segment_time(n_d, float(n_d) * ctx0, k)
    assert integrated == pytest.approx(exact, rel=0.01)


def test_every_step_of_a_composed_segment_is_saturated(model):
    """Unlike PSRL, a composed model has no fitted low-load floor to sit on."""
    assert model.decode_saturated_steps(16, 32768.0, 512) == 512
    assert model.decode_saturated_steps(16, 32768.0, 0) == 0


def test_decode_coeffs_reports_a_secant_over_the_segment(model):
    """The closed form does not apply, but the interface must still return usable terms."""
    n_d, ctx = 16, 32768.0
    floor, alpha, beta = model.decode_coeffs(n_d, ctx)
    assert floor >= 0.0
    assert alpha > 0.0
    # `alpha` must equal the model's own prediction for the first step of the segment.
    first_step = model.predict(
        RolloutWorkload(phase="decode", n_tokens=n_d, sum_l2=0.0, ctxsum=ctx, request_count=n_d)
    )
    assert alpha == pytest.approx(first_step, rel=1e-12)
    # The slope is the forward difference (later - first) / stride, clamped to zero.
    # On a real cost table decode cost rises with context, so the slope must be positive.
    later_step = model.predict(
        RolloutWorkload(
            phase="decode",
            n_tokens=n_d,
            sum_l2=0.0,
            ctxsum=ctx + n_d * DEFAULT_STRIDE,
            request_count=n_d,
        )
    )
    expected_slope = (later_step - first_step) / DEFAULT_STRIDE
    assert beta == pytest.approx(expected_slope, rel=1e-12)
    assert beta > 0.0, f"Expected decode cost to rise with context, got slope {beta!r}."


def test_tensor_parallel_above_one_constructs(database):
    """Replaces a test asserting tp>1 was rejected. The all_reduce op removed that refusal.

    Construction only. The prediction it would make is end-to-end unvalidated: no vLLM tp=2
    baseline exists, and this database holds tp=1 tables, so a `predict` call would raise on
    the missing collective table rather than return a wrong number.
    """
    tp2_ops = ModelOps.from_checkpoint(str(_CHECKPOINT), tensor_parallel=2)
    model = ComposedRollout(database, tp2_ops, tensor_parallel=2)
    assert model.step_time is not None
    assert tp2_ops.comm_queries(16), "tp=2 ops must emit a collective term."


def test_a_tensor_parallel_argument_disagreeing_with_the_ops_is_rejected(database, ops):
    """`ops` decides whether a comm term is emitted, so a mismatch would silently omit it."""
    with pytest.raises(ValueError, match="sharded for"):
        ComposedRollout(database, ops, tensor_parallel=2)


def test_importing_the_registry_does_not_load_the_parquet_stack():
    """A run using a fitted model must not pay for the composed model's dependencies."""
    probe = """
import sys
import srl.cost.rollout  # noqa: F401
kernel = sorted(name for name in sys.modules if name.startswith("srl.cost.kernel"))
assert not kernel, f"eagerly imported {kernel}"
assert "pyarrow" not in sys.modules, "eagerly imported pyarrow"
print("clean")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_the_two_saturation_reports_agree(model):
    """Telemetry reads both fields, so a step cannot be saturated and unsaturated at once."""
    assert model.decode_saturated_steps(16, 32768.0, 100) == 100
    assert model.saturated(16, 0, 32768.0, 16) is True


def test_a_non_positive_stride_is_rejected_at_construction(database, ops):
    """`stride` comes from config, so a typo must not surface mid-simulation."""
    with pytest.raises(ValueError, match="stride"):
        ComposedRollout(database, ops, tensor_parallel=1, stride=0)


def test_the_memo_is_sized_to_hold_one_segments_lattice(database, ops):
    """Arbitrary eviction would thrash and lose the deduplication if a segment overflowed it."""
    for stride in (DEFAULT_STRIDE, 256, 64):
        model = ComposedRollout(database, ops, tensor_parallel=1, stride=stride)
        assert model._predict_cache_cap >= _SEGMENT_CAP // stride + 2


def test_the_memo_is_clamped_so_a_tiny_stride_cannot_exhaust_memory(database, ops):
    """stride=1 wants 2^20 entries (~180 MB). The clamp trades hit rate for a bounded footprint."""
    model = ComposedRollout(database, ops, tensor_parallel=1, stride=1)
    assert model._predict_cache_cap == _MAX_CACHE_ENTRIES
    assert model._predict_cache_cap < _SEGMENT_CAP // 1 + 2


# --- Hybrid closed-form / lattice routing ---


@pytest.fixture
def deep_model(database, ops):
    """A model whose default threshold routes deep KV to the closed form."""
    return ComposedRollout(database, ops, tensor_parallel=1, saturation_step_threshold=8192)


def test_shallow_kv_uses_the_lattice_path(deep_model):
    """At depth < threshold the routing helper must return False."""
    n_d, ctxsum = 16, 16 * 512.0  # depth = 512 < 8192
    assert deep_model._use_closed_form(n_d, ctxsum) is False


def test_deep_kv_uses_the_closed_form_path(deep_model):
    """At depth >= threshold the routing helper must return True."""
    n_d, ctxsum = 16, 16 * 8192.0  # depth = 8192 >= 8192
    assert deep_model._use_closed_form(n_d, ctxsum) is True


def test_threshold_boundary_is_inclusive(deep_model):
    """Depth exactly equal to the threshold must use the closed form."""
    n_d, ctxsum = 8, 8 * 8192.0
    assert deep_model._use_closed_form(n_d, ctxsum) is True
    n_d2, ctxsum2 = 8, 8 * 8191.0
    assert deep_model._use_closed_form(n_d2, ctxsum2) is False


def test_steps_within_inverts_segment_time_shallow_kv(deep_model):
    """Lattice path: integer identity must hold at nodes and near-nodes."""
    n_d, ctxsum = 16, 32768.0  # depth = 2048 < 8192 -> lattice
    assert deep_model._use_closed_form(n_d, ctxsum) is False
    for k in (1, 7, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096):
        budget = deep_model.decode_segment_time(n_d, ctxsum, k)
        result = deep_model.decode_steps_within(n_d, ctxsum, budget)
        assert result == k, f"Lattice inversion failed at k={k}: got {result}."


def test_steps_within_inverts_segment_time_deep_kv(deep_model):
    """Closed-form path: integer identity must hold at and around typical values."""
    n_d, ctxsum = 16, 16 * 16384.0  # depth = 16384 >= 8192 -> closed form
    assert deep_model._use_closed_form(n_d, ctxsum) is True
    for k in (1, 7, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096):
        budget = deep_model.decode_segment_time(n_d, ctxsum, k)
        result = deep_model.decode_steps_within(n_d, ctxsum, budget)
        assert result == k, f"Closed-form inversion failed at k={k}: got {result}."


def test_closed_form_path_costs_fewer_predictions_than_lattice(database, ops):
    """Closed form uses 2 predict() calls (decode_coeffs). Lattice uses stride-sampled nodes."""
    n_d, ctxsum = 16, 16 * 16384.0  # depth 16384 >= 8192 -> closed form
    k = 4096

    def _count_calls(model, n_d, ctxsum, k):
        calls = []
        original = model.predict

        def counting(workload):
            calls.append(1)
            return original(workload)

        model.predict = counting
        model._predict_cache.clear()
        try:
            model.decode_segment_time(n_d, ctxsum, k)
        finally:
            model.predict = original
        return len(calls)

    closed_model = ComposedRollout(database, ops, tensor_parallel=1, saturation_step_threshold=8192)
    lattice_model = ComposedRollout(
        database, ops, tensor_parallel=1, saturation_step_threshold=1_000_000
    )
    assert closed_model._use_closed_form(n_d, ctxsum) is True
    assert lattice_model._use_closed_form(n_d, ctxsum) is False

    closed_calls = _count_calls(closed_model, n_d, ctxsum, k)
    lattice_calls = _count_calls(lattice_model, n_d, ctxsum, k)
    # Closed form: decode_coeffs evaluates offset=0 and offset=stride -> 2 predict calls.
    assert closed_calls == 2, f"Closed form used {closed_calls} predict calls, expected 2."
    assert lattice_calls > closed_calls, (
        f"Lattice ({lattice_calls}) was not more expensive than closed form ({closed_calls})."
    )


def test_both_paths_report_segment_monotonically_growing(deep_model):
    """Total time must strictly increase with k for both routing paths."""
    for n_d, ctxsum, label in (
        (16, 32768.0, "shallow"),  # depth 2048 < 8192 -> lattice
        (16, 16 * 16384.0, "deep"),  # depth 16384 >= 8192 -> closed form
    ):
        previous = 0.0
        for k in (1, 512, 1024, 2048, 4096):
            current = deep_model.decode_segment_time(n_d, ctxsum, k)
            assert current > previous, f"{label} segment not monotone at k={k}."
            previous = current


def test_the_closed_form_fast_path_is_off_by_default(database, ops):
    """The default must match the spec: 8192."""
    model = ComposedRollout(database, ops, tensor_parallel=1)
    # Measured on the real DAPO trace, the fast path bought 1.20x while deviating 4.62% --
    # the same magnitude as this model's own error against a real server. So it is disabled
    # by default and every decode segment integrates on the lattice.
    assert model._saturation_step_threshold == CLOSED_FORM_DISABLED
    assert not model._use_closed_form(16, 16 * 1_000_000.0)
