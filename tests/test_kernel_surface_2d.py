"""The 2-D PCHIP path, its order symmetry, and the fallback above two axes."""

from __future__ import annotations

import pytest

from srl.cost.kernel.interp import UtilitySample
from srl.cost.kernel.surface import PchipSurface, ShepardSurface

# A 3x3 lattice over (batch_size, step), shaped like the real generation-attention surface:
# utility rises with both batch size and KV depth, and saturates.
_LATTICE = tuple(
    UtilitySample(coords=(32.0, batch, step), utility=utility)
    for batch, row in zip((1.0, 8.0, 64.0), ((0.02, 0.05, 0.07), (0.10, 0.22, 0.28), (0.20, 0.38, 0.44)))
    for step, utility in zip((128.0, 2048.0, 32768.0), row)
)


def test_every_measured_lattice_point_is_returned_verbatim():
    surface = PchipSurface(_LATTICE)
    for sample in _LATTICE:
        assert surface.utility(sample.coords) == pytest.approx(sample.utility, rel=1e-9)


def test_an_interior_query_lands_between_its_neighbours():
    surface = PchipSurface(_LATTICE)
    value = surface.utility((32.0, 8.0, 8192.0))
    assert 0.22 < value < 0.28


def test_the_result_does_not_depend_on_which_axis_is_splined_first():
    """The geometric mean of both orders is what makes this true; a single order is not."""
    surface = PchipSurface(_LATTICE)
    transposed = PchipSurface(tuple(
        UtilitySample(coords=(s.coords[0], s.coords[2], s.coords[1]), utility=s.utility)
        for s in _LATTICE
    ))
    for batch, step in ((2.0, 512.0), (8.0, 8192.0), (40.0, 1024.0)):
        assert surface.utility((32.0, batch, step)) == pytest.approx(
            transposed.utility((32.0, step, batch)), rel=1e-12
        )


def test_a_query_past_both_frontiers_clamps_to_the_corner():
    surface = PchipSurface(_LATTICE)
    assert surface.utility((32.0, 4096.0, 1.0e6)) == pytest.approx(0.44, rel=1e-9)
    assert surface.utility((32.0, 0.25, 1.0)) == pytest.approx(0.02, rel=1e-9)


def test_a_ragged_lattice_is_supported():
    """The real context-attention grid is ragged: large batches measure fewer sequence lengths."""
    ragged = (
        UtilitySample(coords=(32.0, 1.0, 128.0), utility=0.10),
        UtilitySample(coords=(32.0, 1.0, 1024.0), utility=0.30),
        UtilitySample(coords=(32.0, 1.0, 16384.0), utility=0.90),
        UtilitySample(coords=(32.0, 64.0, 128.0), utility=0.20),
        UtilitySample(coords=(32.0, 64.0, 1024.0), utility=0.50),
    )
    surface = PchipSurface(ragged)
    for sample in ragged:
        assert surface.utility(sample.coords) == pytest.approx(sample.utility, rel=1e-9)
    # Past batch 64's measured sequence frontier, that line clamps rather than inventing data.
    assert 0.5 <= surface.utility((32.0, 64.0, 16384.0)) <= 0.9


def test_three_varying_axes_fall_back_to_shepard():
    """Above two axes the tensor construction is not defined, so defer rather than mis-spline."""
    samples = tuple(
        UtilitySample(coords=(heads, batch, step), utility=0.1 * i + 0.05)
        for i, (heads, batch, step) in enumerate(
            (
                (8.0, 1.0, 128.0),
                (8.0, 1.0, 1024.0),
                (8.0, 8.0, 128.0),
                (8.0, 8.0, 1024.0),
                (32.0, 1.0, 128.0),
                (32.0, 1.0, 1024.0),
                (32.0, 8.0, 128.0),
                (32.0, 8.0, 1024.0),
            )
        )
    )
    surface = PchipSurface(samples)
    reference = ShepardSurface(samples)
    query = (16.0, 4.0, 512.0)
    assert surface.utility(query) == pytest.approx(reference.utility(query), rel=1e-12)


def test_a_degenerate_axis_does_not_count_toward_the_fallback():
    """num_heads is constant inside a slice, so a 3-axis attention slice is really 2-D."""
    samples = tuple(
        UtilitySample(coords=(32.0, batch, step), utility=utility)
        for batch, step, utility in (
            (1.0, 128.0, 0.02),
            (1.0, 2048.0, 0.05),
            (64.0, 128.0, 0.20),
            (64.0, 2048.0, 0.38),
        )
    )
    surface = PchipSurface(samples)
    reference = ShepardSurface(samples)
    query = (32.0, 8.0, 512.0)
    # Splining, not falling back: the two must disagree, and both must be in range.
    assert surface.utility(query) != pytest.approx(reference.utility(query), rel=1e-6)
    assert 0.02 < surface.utility(query) < 0.38
