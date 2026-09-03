"""Utility interpolation: exact hits, interior blends, frontier holds, ragged grids."""

from __future__ import annotations

import pytest

from dryrun.cost.kernel.interp import (
    KernelCoverageError,
    UtilityGrid,
    UtilitySample,
)


def _grid(points: dict[tuple[float, ...], float]) -> UtilityGrid:
    return UtilityGrid([UtilitySample(coords=key, utility=value) for key, value in points.items()])


def test_exact_hit_returns_the_measured_utility():
    grid = _grid({(1024.0, 4096.0): 0.5, (2048.0, 4096.0): 0.6, (4096.0, 4096.0): 0.7})
    assert grid.utility((2048.0, 4096.0)) == pytest.approx(0.6)


def test_uniform_utility_reproduces_exactly_anywhere():
    grid = _grid({(1024.0, 4096.0): 0.42, (2048.0, 4096.0): 0.42, (4096.0, 4096.0): 0.42})
    for query in ((1500.0, 4096.0), (3000.0, 4096.0), (99999.0, 4096.0), (1.0, 4096.0)):
        assert grid.utility(query) == pytest.approx(0.42), query


def test_interior_query_stays_bracketed_by_its_neighbors():
    grid = _grid({(1024.0, 4096.0): 0.5, (2048.0, 4096.0): 0.6, (4096.0, 4096.0): 0.7})
    value = grid.utility((1448.0, 4096.0))
    assert 0.5 <= value <= 0.6


def test_query_past_the_frontier_holds_near_the_boundary_utility():
    grid = _grid({(1024.0, 4096.0): 0.5, (2048.0, 4096.0): 0.6, (4096.0, 4096.0): 0.7})
    held = grid.utility((16384.0, 4096.0))
    assert 0.6 <= held <= 0.7, "Utility is transferred, never extrapolated past the data."


def test_query_below_the_frontier_holds_near_the_boundary_utility():
    grid = _grid({(1024.0, 4096.0): 0.5, (2048.0, 4096.0): 0.6, (4096.0, 4096.0): 0.7})
    held = grid.utility((16.0, 4096.0))
    assert 0.5 <= held <= 0.6


def test_ragged_grid_answers_the_truncated_corner():
    # A corner-truncated grid: (4096, 8192) was never measured because it would OOM.
    grid = _grid(
        {
            (1024.0, 4096.0): 0.50,
            (2048.0, 4096.0): 0.55,
            (4096.0, 4096.0): 0.60,
            (1024.0, 8192.0): 0.45,
            (2048.0, 8192.0): 0.48,
        }
    )
    value = grid.utility((4096.0, 8192.0))
    assert 0.4 <= value <= 0.65


def test_single_sample_grid_holds_that_utility_everywhere():
    grid = _grid({(1024.0, 4096.0): 0.33})
    assert grid.utility((1024.0, 4096.0)) == pytest.approx(0.33)
    assert grid.utility((8192.0, 4096.0)) == pytest.approx(0.33)


def test_bounds_reports_per_axis_min_and_max():
    grid = _grid({(1024.0, 4096.0): 0.5, (4096.0, 8192.0): 0.6})
    assert grid.bounds() == ((1024.0, 4096.0), (4096.0, 8192.0))


def test_sample_count_is_exposed():
    grid = _grid({(1024.0, 4096.0): 0.5, (4096.0, 8192.0): 0.6})
    assert grid.sample_count == 2


def test_empty_grid_is_rejected():
    with pytest.raises(KernelCoverageError):
        UtilityGrid([])


def test_wrong_dimensionality_is_rejected():
    grid = _grid({(1024.0, 4096.0): 0.5, (2048.0, 4096.0): 0.6})
    with pytest.raises(KernelCoverageError):
        grid.utility((1024.0,))


def test_mismatched_sample_dimensionality_is_rejected():
    with pytest.raises(KernelCoverageError):
        UtilityGrid(
            [
                UtilitySample(coords=(1024.0, 4096.0), utility=0.5),
                UtilitySample(coords=(2048.0,), utility=0.6),
            ]
        )


def test_nonpositive_coordinates_are_rejected():
    with pytest.raises(KernelCoverageError):
        UtilityGrid([UtilitySample(coords=(0.0, 4096.0), utility=0.5)])


def test_nonpositive_utility_is_rejected():
    with pytest.raises(KernelCoverageError):
        UtilityGrid([UtilitySample(coords=(1024.0, 4096.0), utility=0.0)])


def test_duplicate_sample_coordinates_are_rejected():
    with pytest.raises(KernelCoverageError):
        UtilityGrid(
            [
                UtilitySample(coords=(1024.0, 4096.0), utility=0.5),
                UtilitySample(coords=(1024.0, 4096.0), utility=0.6),
            ]
        )

