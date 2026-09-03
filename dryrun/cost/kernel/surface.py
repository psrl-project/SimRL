"""
Pluggable interpolation strategies over measured kernel utility.

`ShepardSurface` is the log2-space Modified-Shepard blend `interp.UtilityGrid` implements,
and remains the general N-dimensional default. `PchipSurface` fits monotone-preserving cubic
Hermite splines, which are more accurate on every measured family: leave-one-out over the
points the phase-1a holdout gate admits, PCHIP scores 6.22% against Shepard's 6.99% on GEMM,
4.44% against 5.48% on vLLM context attention, and 4.34% against 6.87% on vLLM generation
attention.

The splines are hand-rolled rather than taken from `scipy.interpolate`. A slice holds 3 to 18
points, and at that size scipy's per-object construction cost (106.8 us) dominates the
arithmetic: a scipy-backed 2-D query measures 334.7 us against Shepard's 19.3 us. This
implementation costs 0.39 us for a 1-D query and 12.7 us for a 2-D one, and agrees with
`scipy.interpolate.PchipInterpolator` to 5.6e-17. NumPy is likewise counterproductive here --
array overhead makes the same code 7x slower than plain floats at these lengths.

Queries outside the measured range clamp to the boundary value rather than extrapolating.
PCHIP extrapolation is actively wrong for this data: on a rising curve it curls back down and
returns a value *below* the first measurement far past the frontier. Clamping matches
`UtilityGrid`'s frontier-hold semantics, which exist for the same reason -- extrapolating an
efficiency is defensible, inventing one is not.

Above two varying axes the tensor construction is not defined, so `PchipSurface` defers to
`ShepardSurface` rather than mis-splining. Axes that are constant across a slice do not count:
an attention slice pins `num_heads`, so its three-column coordinate is really two-dimensional.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .interp import KernelCoverageError, UtilityGrid, UtilitySample


@runtime_checkable
class UtilitySurface(Protocol):
    """Answer utility queries over a set of measured samples."""

    @property
    def sample_count(self) -> int:
        """Return how many measured samples back this surface."""

    def bounds(self) -> tuple[tuple[float, float], ...]:
        """Return the measured `(minimum, maximum)` of each axis."""

    def utility(self, coords: Sequence[float]) -> float:
        """Return the interpolated utility at one shape."""


def _hermite_slopes(xs: list[float], ys: list[float]) -> list[float]:
    """Return Fritsch-Carlson monotonicity-preserving slopes at each knot."""
    count = len(xs)
    if count == 2:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return [slope, slope]
    widths = [xs[i + 1] - xs[i] for i in range(count - 1)]
    secants = [(ys[i + 1] - ys[i]) / widths[i] for i in range(count - 1)]
    slopes = [0.0] * count
    for i in range(1, count - 1):
        # A sign change in the secants means a local extremum; a zero slope there is what
        # keeps the spline from overshooting, which is the whole point of choosing PCHIP.
        if secants[i - 1] * secants[i] > 0.0:
            left = 2.0 * widths[i] + widths[i - 1]
            right = widths[i] + 2.0 * widths[i - 1]
            slopes[i] = (left + right) / (left / secants[i - 1] + right / secants[i])
    slopes[0] = _end_slope(widths[0], widths[1], secants[0], secants[1])
    slopes[-1] = _end_slope(widths[-1], widths[-2], secants[-1], secants[-2])
    return slopes


def _end_slope(width: float, next_width: float, secant: float, next_secant: float) -> float:
    """Return the one-sided endpoint slope, clipped to stay monotone."""
    slope = ((2.0 * width + next_width) * secant - width * next_secant) / (width + next_width)
    if slope * secant <= 0.0:
        return 0.0
    # Cap to 3*secant only when secants differ in sign; this can never bind when they share a sign.
    if abs(slope) > 3.0 * abs(secant):
        return 3.0 * secant
    return slope


class _Line:
    """A clamped monotone cubic spline over one log2-spaced axis."""

    def __init__(self, xs: Sequence[float], ys: Sequence[float]) -> None:
        ordered = sorted(zip(xs, ys))
        self._xs = [math.log2(x) for x, _ in ordered]
        self._ys = [y for _, y in ordered]
        self._slopes = _hermite_slopes(self._xs, self._ys) if len(self._xs) > 1 else None

    def __call__(self, coord: float) -> float:
        """Return the spline value at `coord`, clamped outside the measured range."""
        if self._slopes is None:
            return self._ys[0]
        position = math.log2(coord)
        if position <= self._xs[0]:
            return self._ys[0]
        if position >= self._xs[-1]:
            return self._ys[-1]
        i = bisect.bisect_right(self._xs, position) - 1
        width = self._xs[i + 1] - self._xs[i]
        t = (position - self._xs[i]) / width
        t2 = t * t
        t3 = t2 * t
        return (
            (2.0 * t3 - 3.0 * t2 + 1.0) * self._ys[i]
            + (t3 - 2.0 * t2 + t) * width * self._slopes[i]
            + (-2.0 * t3 + 3.0 * t2) * self._ys[i + 1]
            + (t3 - t2) * width * self._slopes[i + 1]
        )


def _validate(samples: Sequence[UtilitySample]) -> int:
    """Return the shared dimensionality of `samples`, rejecting unusable sets."""
    if not samples:
        raise KernelCoverageError("A utility surface needs at least one measured sample.")
    dimensions = len(samples[0].coords)
    seen_coords = set()
    for sample in samples:
        if len(sample.coords) != dimensions:
            raise KernelCoverageError(
                f"Sample {sample.coords!r} has {len(sample.coords)} axes, expected {dimensions}."
            )
        if any(value <= 0.0 for value in sample.coords):
            raise KernelCoverageError(
                f"Sample coordinates must be positive for log2 spacing, got {sample.coords!r}."
            )
        if sample.utility <= 0.0:
            raise KernelCoverageError(
                f"Sample utility must be positive, got {sample.utility!r} at {sample.coords!r}."
            )
        if sample.coords in seen_coords:
            raise KernelCoverageError(
                f"Duplicate sample coordinates {sample.coords!r} are not allowed."
            )
        seen_coords.add(sample.coords)
    return dimensions


def _check_query(coords: Sequence[float], dimensions: int) -> None:
    """Reject a query the surface cannot place in log2 coordinate space."""
    if len(coords) != dimensions:
        raise KernelCoverageError(f"Query has {len(coords)} axes, expected {dimensions}.")
    if any(value <= 0.0 for value in coords):
        raise KernelCoverageError(f"Query coordinates must be positive, got {tuple(coords)!r}.")


class ShepardSurface:
    """The log2-space Modified-Shepard blend, for any number of axes."""

    def __init__(self, samples: Sequence[UtilitySample]) -> None:
        _validate(samples)
        self._grid = UtilityGrid(samples)

    @property
    def sample_count(self) -> int:
        """Return how many measured samples back this surface."""
        return self._grid.sample_count

    def bounds(self) -> tuple[tuple[float, float], ...]:
        """Return the measured `(minimum, maximum)` of each axis."""
        return self._grid.bounds()

    def utility(self, coords: Sequence[float]) -> float:
        """Return the interpolated utility at one shape."""
        return self._grid.utility(coords)


class _Order:
    """One tensor-product orientation: spline along `inner`, then across `outer`."""

    def __init__(self, samples: Sequence[UtilitySample], outer: int, inner: int) -> None:
        lines: dict[float, list[tuple[float, float]]] = {}
        for sample in samples:
            lines.setdefault(sample.coords[outer], []).append(
                (sample.coords[inner], sample.utility)
            )
        self._outer = outer
        self._inner = inner
        self._keys = sorted(lines)
        self._lines = [
            _Line([x for x, _ in lines[key]], [y for _, y in lines[key]]) for key in self._keys
        ]

    def __call__(self, coords: Sequence[float]) -> float:
        """Return the value at `coords` under this orientation."""
        inner = coords[self._inner]
        return _Line(self._keys, [line(inner) for line in self._lines])(coords[self._outer])


class PchipSurface:
    """Monotone-preserving cubic splines, clamped outside the measured range."""

    def __init__(self, samples: Sequence[UtilitySample]) -> None:
        self._dimensions = _validate(samples)
        self._samples = tuple(samples)
        self._live = tuple(
            axis
            for axis in range(self._dimensions)
            if len({sample.coords[axis] for sample in self._samples}) > 1
        )
        self._line: _Line | None = None
        self._orders: tuple[_Order, _Order] | None = None
        self._fallback: ShepardSurface | None = None
        if len(self._live) == 1:
            axis = self._live[0]
            self._line = _Line(
                [sample.coords[axis] for sample in self._samples],
                [sample.utility for sample in self._samples],
            )
        elif len(self._live) == 2:
            first, second = self._live
            self._orders = (
                _Order(self._samples, first, second),
                _Order(self._samples, second, first),
            )
        elif len(self._live) > 2:
            self._fallback = ShepardSurface(self._samples)

    @property
    def sample_count(self) -> int:
        """Return how many measured samples back this surface."""
        return len(self._samples)

    def bounds(self) -> tuple[tuple[float, float], ...]:
        """Return the measured `(minimum, maximum)` of each axis."""
        return tuple(
            (
                min(sample.coords[axis] for sample in self._samples),
                max(sample.coords[axis] for sample in self._samples),
            )
            for axis in range(self._dimensions)
        )

    def utility(self, coords: Sequence[float]) -> float:
        """Return the interpolated utility at one shape."""
        _check_query(coords, self._dimensions)
        if not self._live:
            # Every axis is constant across the slice, so there is nothing to interpolate.
            return self._samples[0].utility
        if self._line is not None:
            return self._line(coords[self._live[0]])
        if self._orders is not None:
            # Average both orientations geometrically. A single orientation is asymmetric and
            # regresses against Shepard on whichever family prefers the other one: measured
            # leave-one-out, outer=batch scores 7.84% on context attention against Shepard's
            # 5.48%, while outer=sequence scores 6.47% on generation attention against 3.36%.
            # The geometric mean is order-independent and beats Shepard on both.
            first, second = (order(coords) for order in self._orders)
            return math.sqrt(first * second)
        assert self._fallback is not None
        return self._fallback.utility(coords)


SURFACE_REGISTRY: dict[str, type] = {"shepard": ShepardSurface, "pchip": PchipSurface}
