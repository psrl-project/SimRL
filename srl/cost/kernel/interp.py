"""
Interpolate kernel utility over measured shapes.

Utility (`SOL / measured_latency`) is interpolated rather than latency, because
utility varies slowly while latency spans orders of magnitude across a sweep. All
latency growth in a prediction comes from the SOL formula. This module only decides
*how efficient* the kernel is at a shape. That is why a query past the measured
frontier holds a transferred utility instead of extrapolating: extrapolating an
efficiency is defensible, extrapolating a latency is not.

Distances are Euclidean in log2 coordinate space, so a factor-of-two step counts the
same on every axis and at every magnitude. Weights are Modified-Shepard tapered,
`((R - d) / (R * d))**2`, with the support radius `R` taken from the first site
outside the neighbor set. The taper sends the weight of a site at `d == R` to zero,
so the blend stays continuous when the neighbor ranking changes and the surface has
no cliffs at rank swaps.
"""


from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_NEIGHBOR_COUNT = 2
_EPSILON = 1e-12


class KernelCoverageError(RuntimeError):
    """Indicate that measured kernel data cannot answer a query."""


@dataclass(frozen=True)
class UtilitySample:
    """One measured shape and its achieved efficiency."""

    coords: tuple[float, ...]
    utility: float


class UtilityGrid:
    """Scattered measured utilities answering queries by log2-space blending."""

    def __init__(self, samples: Sequence[UtilitySample]) -> None:
        """
        Build a grid from measured samples.

        Args:
            samples (Sequence[UtilitySample]): At least one sample. Every sample must
                share a dimensionality, carry strictly positive coordinates (log2 is
                taken of them), and a strictly positive utility.

        Raises:
            KernelCoverageError: If the samples are empty or inconsistent.
        """
        if not samples:
            raise KernelCoverageError("A utility grid needs at least one measured sample.")
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
        self._dimensions = dimensions
        self._coords = np.asarray([sample.coords for sample in samples], dtype=np.float64)
        self._log_coords = np.log2(self._coords)
        self._utilities = np.asarray([sample.utility for sample in samples], dtype=np.float64)

    @property
    def sample_count(self) -> int:
        """Return how many measured samples back this grid."""
        return int(self._utilities.size)

    def bounds(self) -> tuple[tuple[float, float], ...]:
        """
        Return the measured range of each axis.

        Returns:
            tuple[tuple[float, float], ...]: One `(minimum, maximum)` pair per axis.
        """
        return tuple(
            (float(self._coords[:, axis].min()), float(self._coords[:, axis].max()))
            for axis in range(self._dimensions)
        )

    def utility(self, coords: Sequence[float]) -> float:
        """
        Return the interpolated utility at one shape.

        Args:
            coords (Sequence[float]): Query coordinates in the grid's axis order, all
                strictly positive.

        Returns:
            float: Blended utility. An exact match returns its measured value verbatim.

        Raises:
            KernelCoverageError: If the dimensionality or coordinates are invalid.
        """
        if len(coords) != self._dimensions:
            raise KernelCoverageError(
                f"Query has {len(coords)} axes, expected {self._dimensions}."
            )
        query = np.asarray(coords, dtype=np.float64)
        if np.any(query <= 0.0):
            raise KernelCoverageError(f"Query coordinates must be positive, got {tuple(coords)!r}.")

        distances = np.sqrt(np.sum((self._log_coords - np.log2(query)) ** 2, axis=1))
        exact = np.flatnonzero(distances <= _EPSILON)
        if exact.size:
            return float(self._utilities[exact[0]])

        order = np.argsort(distances, kind="stable")
        take = order[:_NEIGHBOR_COUNT]
        neighbor_distances = distances[take]
        if order.size > _NEIGHBOR_COUNT:
            radius = float(distances[order[_NEIGHBOR_COUNT]])
        else:
            radius = float(neighbor_distances.max()) * 2.0
        # Guarantee a strictly positive taper for every neighbor, including the
        # degenerate case where the support site is equidistant with the last neighbor.
        radius = max(radius, float(neighbor_distances.max()) * (1.0 + _EPSILON))

        weights = ((radius - neighbor_distances) / (radius * neighbor_distances)) ** 2
        total = float(weights.sum())
        if total <= 0.0:
            return float(self._utilities[take[0]])
        return float(np.dot(weights, self._utilities[take]) / total)
