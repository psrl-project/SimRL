"""
Check that the C++ trampoline reaches every method a Python cost model overrides.

`PyWrappedCostModel` makes the two backends interchangeable only if each override is
dispatched virtually. A model such as `ComposedRollout` integrates decode segments
exactly, while the C++ defaults rebuild them from the `decode_coeffs` secant, so a
missed override silently prices a run differently.
"""

from __future__ import annotations

import pytest

from srl.cost.rollout import UnifiedRoofline

pytest.importorskip("srl._engine_cpp")

from srl._engine_cpp import PyWrappedCostModel  # noqa: E402


class _SegmentAwareCost(UnifiedRoofline):
    """
    Model a cost whose segment methods disagree with its own `decode_coeffs`.

    The constants are far apart so a failure cannot be mistaken for float noise.
    """

    SEGMENT_SECONDS = 123.0
    STEPS_WITHIN = 77
    SATURATED_STEPS = 11
    STEP_SECONDS = 0.5

    def step_time(self, n_tokens, t2, ctxsum, n_reqs=0, prefill_ctx=0.0) -> float:
        return self.STEP_SECONDS

    def decode_segment_time(self, n_d: int, ctxsum: float, k: int) -> float:
        return self.SEGMENT_SECONDS

    def decode_steps_within(self, n_d: int, ctxsum: float, budget: float) -> int:
        return self.STEPS_WITHIN

    def decode_saturated_steps(self, n_d: int, ctxsum: float, k: int) -> int:
        return self.SATURATED_STEPS


@pytest.fixture
def wrapped() -> tuple[_SegmentAwareCost, PyWrappedCostModel]:
    py_cost = _SegmentAwareCost(F=0.005, W=0.001, G=0.0001, A_p=0.0, A_d=1e-7)
    return py_cost, PyWrappedCostModel(py_cost)


class TestTrampolineReachesEveryOverride:
    """Check that each method the Python model overrides survives the round trip."""

    def test_step_time(self, wrapped):
        py_cost, cpp_cost = wrapped
        assert cpp_cost.step_time(8.0, 0.0, 8000.0, 8, 0.0) == pytest.approx(
            py_cost.STEP_SECONDS
        )

    def test_decode_segment_time(self, wrapped):
        py_cost, cpp_cost = wrapped
        assert cpp_cost.decode_segment_time(4, 4096.0, 500) == pytest.approx(
            py_cost.SEGMENT_SECONDS
        )

    def test_decode_steps_within(self, wrapped):
        py_cost, cpp_cost = wrapped
        assert cpp_cost.decode_steps_within(4, 4096.0, 10.0) == py_cost.STEPS_WITHIN

    def test_decode_saturated_steps(self, wrapped):
        py_cost, cpp_cost = wrapped
        assert cpp_cost.decode_saturated_steps(4, 4096.0, 500) == py_cost.SATURATED_STEPS


class TestDefaultsStillDeriveFromCoeffs:
    """Check that a model overriding nothing keeps the base-class arithmetic on both sides."""

    def test_plain_roofline_agrees(self):
        py_cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0.0, A_d=1e-7)
        cpp_cost = PyWrappedCostModel(py_cost)
        for n_d, ctxsum, k in ((1, 200.0, 100), (8, 8000.0, 500), (48, 1e6, 50)):
            assert cpp_cost.decode_segment_time(n_d, ctxsum, k) == pytest.approx(
                py_cost.decode_segment_time(n_d, ctxsum, k), rel=1e-12
            )
            assert cpp_cost.decode_steps_within(
                n_d, ctxsum, 10.0
            ) == py_cost.decode_steps_within(n_d, ctxsum, 10.0)
