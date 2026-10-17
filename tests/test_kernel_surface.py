"""The pluggable utility surfaces, in one dimension."""

from __future__ import annotations

import math

import pytest

from srl.cost.kernel.interp import KernelCoverageError, UtilitySample
from srl.cost.kernel.surface import SURFACE_REGISTRY, PchipSurface, ShepardSurface

_RISING = (
    UtilitySample(coords=(1.0,), utility=0.5),
    UtilitySample(coords=(32.0,), utility=0.6),
    UtilitySample(coords=(64.0,), utility=0.8),
    UtilitySample(coords=(1024.0,), utility=0.9),
)


@pytest.mark.parametrize("surface_type", [ShepardSurface, PchipSurface])
def test_a_measured_point_is_returned_verbatim(surface_type):
    """Both surfaces must reproduce a measurement, not smooth it away."""
    surface = surface_type(_RISING)
    for sample in _RISING:
        assert surface.utility(sample.coords) == pytest.approx(sample.utility, rel=1e-12)


def test_the_spline_interpolates_between_measured_points():
    surface = PchipSurface(_RISING)
    # Value at a query near a knot should be close to that knot's measurement.
    value_near_left = surface.utility((2.0,))
    assert value_near_left == pytest.approx(0.5, rel=0.05)
    # Spline is increasing across a rising stretch.
    value_mid = surface.utility((100.0,))
    assert 0.8 < value_mid < 0.9
    assert value_mid > value_near_left


def test_the_spline_is_shape_preserving_across_a_trough():
    """The real GEMM utility curve dips before climbing; the spline must not overshoot it."""
    samples = (
        UtilitySample(coords=(1.0,), utility=0.6294),
        UtilitySample(coords=(32.0,), utility=0.4727),
        UtilitySample(coords=(64.0,), utility=0.7486),
        UtilitySample(coords=(1024.0,), utility=0.9246),
        UtilitySample(coords=(16384.0,), utility=0.9343),
    )
    surface = PchipSurface(samples)
    for query in (2.0, 8.0, 16.0, 24.0, 48.0, 96.0, 512.0, 4096.0):
        value = surface.utility((query,))
        assert 0.4727 <= value <= 0.9343, f"overshoot at {query}: {value}"


def test_the_spline_clamps_above_the_measured_range():
    """PCHIP extrapolation falls back below the last measurement; clamping must prevent it."""
    surface = PchipSurface(_RISING)
    for query in (2048.0, 65536.0, 1.0e9):
        assert surface.utility((query,)) == pytest.approx(0.9, rel=1e-12)


def test_the_spline_clamps_below_the_measured_range():
    surface = PchipSurface(_RISING)
    for query in (0.5, 0.001):
        assert surface.utility((query,)) == pytest.approx(0.5, rel=1e-12)


def test_a_single_sample_holds_its_value_everywhere():
    """A slice with one measured point cannot be splined; it must hold, not raise."""
    surface = PchipSurface((UtilitySample(coords=(4096.0,), utility=0.72),))
    assert surface.utility((1.0,)) == pytest.approx(0.72)
    assert surface.utility((1.0e6,)) == pytest.approx(0.72)


def test_two_samples_spline_linearly_in_log2_space():
    """With two points PCHIP degenerates to a straight line over log2 x."""
    surface = PchipSurface(
        (UtilitySample(coords=(1.0,), utility=0.5), UtilitySample(coords=(4.0,), utility=0.7))
    )
    assert surface.utility((2.0,)) == pytest.approx(0.6, rel=1e-12)


def test_the_hermite_evaluation_matches_scipy():
    """Guard the hand-rolled Fritsch-Carlson slopes against the reference implementation."""
    scipy_interpolate = pytest.importorskip("scipy.interpolate")
    xs = [1.0, 32.0, 64.0, 1024.0, 16384.0]
    ys = [0.6294, 0.4727, 0.7486, 0.9246, 0.9343]
    reference = scipy_interpolate.PchipInterpolator(
        [math.log2(x) for x in xs], ys, extrapolate=False
    )
    surface = PchipSurface(tuple(
        UtilitySample(coords=(x,), utility=y) for x, y in zip(xs, ys)
    ))
    for query in (3.0, 11.0, 48.0, 300.0, 5000.0):
        assert surface.utility((query,)) == pytest.approx(
            float(reference(math.log2(query))), rel=1e-9
        )


@pytest.mark.parametrize("surface_type", [ShepardSurface, PchipSurface])
def test_the_surfaces_report_the_measured_range(surface_type):
    assert surface_type(_RISING).bounds() == ((1.0, 1024.0),)
    assert surface_type(_RISING).sample_count == 4


@pytest.mark.parametrize("surface_type", [ShepardSurface, PchipSurface])
def test_an_empty_sample_set_is_rejected(surface_type):
    with pytest.raises(KernelCoverageError):
        surface_type(())


@pytest.mark.parametrize("surface_type", [ShepardSurface, PchipSurface])
def test_a_non_positive_query_is_rejected(surface_type):
    """Coordinates are placed in log2 space, so zero and negatives cannot be answered."""
    surface = surface_type(_RISING)
    with pytest.raises(KernelCoverageError):
        surface.utility((0.0,))


@pytest.mark.parametrize("surface_type", [ShepardSurface, PchipSurface])
def test_duplicate_sample_coordinates_are_rejected(surface_type):
    """Both surfaces reject duplicate coordinates during construction."""
    duplicates = (
        UtilitySample(coords=(1.0,), utility=0.5),
        UtilitySample(coords=(32.0,), utility=0.6),
        UtilitySample(coords=(32.0,), utility=0.7),
    )
    with pytest.raises(KernelCoverageError, match="Duplicate sample coordinates"):
        surface_type(duplicates)



@pytest.mark.parametrize("surface_type", [ShepardSurface, PchipSurface])
def test_a_wrong_dimensionality_query_is_rejected(surface_type):
    surface = surface_type(_RISING)
    with pytest.raises(KernelCoverageError):
        surface.utility((16.0, 4.0))


def test_the_registry_names_both_surfaces():
    assert SURFACE_REGISTRY == {"shepard": ShepardSurface, "pchip": PchipSurface}
