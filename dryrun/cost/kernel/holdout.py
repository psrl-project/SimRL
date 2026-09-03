"""Held-out interpolation accuracy for the phase 1a exit criterion.

Only bracketed points are scored. A point is bracketed if, after removing it, the
remaining points still span it on every axis, so interpolation genuinely applies.
Removing a point that is no longer bracketed turns the query into a frontier hold,
which measures a different mechanism entirely: the hold is designed to be
conservative, not accurate. Scoring holds would flatter or damn the interpolator
for the wrong reason.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hardware import GpuSpec, gpu_slug, load_gpu_spec
from .interp import KernelCoverageError, UtilityGrid, UtilitySample
from .ops.base import op_for, ops_in
from .sol import FALLBACK_FLOORS_S
from .table import KernelTable

_MIN_SLICE_POINTS: int = 4


def _log2_compatible(coords: tuple[float, ...]) -> bool:
    """Return True if every axis value is strictly positive (required for log2 interpolation)."""
    return all(v > 0.0 for v in coords)


@dataclass(frozen=True)
class HoldoutResult:
    """Interpolation accuracy over one `(op_name, backend)` table."""

    op_name: str
    backend: str
    point_count: int
    mre_percent: float
    p90_percent: float
    max_percent: float
    worst_coords: tuple[float, ...]
    skipped_slices: int
    utility_min: float
    utility_max: float

    def passes(self, gate_percent: float) -> bool:
        """Return True if mean relative error is below the gate threshold and at least one point was scored."""
        return self.point_count > 0 and self.mre_percent < gate_percent


def bracketing_indices(coords: Sequence[tuple[float, ...]]) -> tuple[int, ...]:
    """
    Return indices of points that are still bracketed after removal.

    A point is bracketed if, after removing it from the set, every axis value
    still lies within the per-axis `[min, max]` range of the remaining points
    (inclusive). This is the correct criterion for the phase 1a holdout gate:
    the property needed is that interpolation genuinely applies, not that the
    point is a strict geometric interior point. The strict-interior criterion
    wrongly excludes every point in a real unified-GEMM slice, because the four
    Qwen3-style projections share one `(gemm_dtype,)` slice and pin the `n` and
    `k` axes to their extremes, leaving no geometrically interior point at all.

    On axes where all remaining points share the same value, bracketing is
    vacuously satisfied (no variation, so no frontier can be defined).

    Args:
        coords (Sequence[tuple[float, ...]]): Axis-aligned coordinates.

    Returns:
        tuple[int, ...]: Sorted indices of bracketed points.
    """
    if not coords:
        return ()
    n_axes = len(coords[0])
    if n_axes == 0:
        return ()
    result = []
    for idx, c in enumerate(coords):
        remaining = [coords[j] for j in range(len(coords)) if j != idx]
        if not remaining:
            continue
        bracketed = True
        for axis in range(n_axes):
            values = [r[axis] for r in remaining]
            lo, hi = min(values), max(values)
            if lo == hi:
                # Degenerate axis after removal — vacuously satisfied.
                continue
            if c[axis] < lo or c[axis] > hi:
                bracketed = False
                break
        if bracketed:
            result.append(idx)
    return tuple(result)


def _make_sol_fn(op_name: str, spec: GpuSpec, slice_key: tuple[Any, ...], floor_s: float = 0.0) -> Any:
    """Return the SOL callable for one op and slice."""
    op = op_for(op_name)

    def sol_fn(coords: tuple[float, ...]) -> float:
        return op.sol(spec, slice_key=slice_key, coords=coords, floor_s=floor_s).latency_s

    return sol_fn


def evaluate_holdout(
    table: KernelTable,
    *,
    op_name: str,
    backend: str,
    gpu_spec: GpuSpec,
    fraction: float = 1.0,
    seed: int = 0,
    floor_override: float | None = None,
) -> HoldoutResult:
    """
    Evaluate held-out interpolation accuracy over one `(op_name, backend)` table.

    For each slice with at least `_MIN_SLICE_POINTS` rows, bracketed points are
    identified and a random subset (size `ceil(fraction * len(bracketed))`) is
    held out one at a time. Each held-out point is predicted from a grid built
    on all other rows, and the relative error is recorded. Only bracketed points
    are scored: a point is bracketed when, after removal, the remaining points
    still span it on every axis, so interpolation genuinely applies. Removing a
    point that loses bracketing turns the query into a frontier hold, which
    measures a different mechanism than interpolation accuracy.

    Rows with a zero or negative coordinate on any axis are excluded from the
    interpolation scoring set (they cannot be placed in log2 coordinate space),
    but their utilities are included in the utility range diagnostic. Generation
    attention at step=0 is the canonical example.

    The `backend` name is used to look up the per-backend dispatch floor in
    `FALLBACK_FLOORS_S`. Both the held-out point's SOL and the grid's SOL use the
    same floor, so utility is computed consistently on both sides of the round trip.

    Args:
        table (KernelTable): Loaded perf table for one `(op_name, backend)`.
        op_name (str): The op identifier matching the table's schema.
        backend (str): Backend label; used in the result and to select the dispatch floor.
        gpu_spec (GpuSpec): Hardware spec for the SOL computation.
        fraction (float): Fraction of bracketed points to hold out, in (0, 1]. Defaults to
            the full set: a subset can report a far better surface than exists.
        seed (int): Random seed for reproducible sampling.
        floor_override (float | None): When set, use this value as the dispatch floor
            instead of reading `FALLBACK_FLOORS_S`. Ignored by ops declaring `fits_floor`
            False. Pass a candidate floor during fitting to avoid mutating any global state.

    Returns:
        HoldoutResult: Aggregated accuracy metrics over all held-out points.
    """
    rng = random.Random(seed)
    errors: list[float] = []
    worst_error = 0.0
    worst_coords: tuple[float, ...] = ()
    skipped_slices = 0
    all_utilities: list[float] = []

    # Resolve the floor once for this (op, backend) pair so grid construction and the
    # predicted-SOL call use identical values -- the round trip must be consistent.
    op = op_for(op_name)
    if not op.fits_floor:
        floor_s = 0.0
    elif floor_override is not None:
        floor_s = floor_override
    else:
        floor_s = FALLBACK_FLOORS_S.get(backend, {}).get(op_name, 0.0)

    for slice_key in table.slice_keys():
        rows = table.rows(slice_key)
        if len(rows) < _MIN_SLICE_POINTS:
            skipped_slices += 1
            continue

        all_coords = [c for c, _ in rows]
        measured_map: dict[tuple[float, ...], float] = {c: lat for c, lat in rows}
        sol_fn = _make_sol_fn(op_name, gpu_spec, slice_key, floor_s=floor_s)

        # Collect utilities over every row for the utility range diagnostic.
        for coords, latency in rows:
            all_utilities.append(sol_fn(coords) / latency)

        # Only interpolatable rows (all-positive coords) participate in the holdout.
        # Rows with a zero coordinate (e.g. generation attention at step=0) cannot be
        # placed in log2 coordinate space, so they are excluded from scoring.
        interp_coords = [c for c in all_coords if _log2_compatible(c)]
        if len(interp_coords) < _MIN_SLICE_POINTS:
            skipped_slices += 1
            continue
        interp_map = {c: measured_map[c] for c in interp_coords}

        bracketed = list(bracketing_indices(interp_coords))
        n_holdout = math.ceil(fraction * len(bracketed))
        rng.shuffle(bracketed)
        candidates = bracketed[:n_holdout]

        for held_idx in candidates:
            held_coords = interp_coords[held_idx]
            remaining_coords = [c for i, c in enumerate(interp_coords) if i != held_idx]
            remaining_latencies = {c: interp_map[c] for c in remaining_coords}

            samples = [
                UtilitySample(coords=c, utility=sol_fn(c) / lat)
                for c, lat in remaining_latencies.items()
            ]
            try:
                grid = UtilityGrid(samples)
            except KernelCoverageError:
                continue

            predicted = sol_fn(held_coords) / grid.utility(held_coords)
            measured = measured_map[held_coords]
            rel_error = abs(predicted - measured) / measured * 100.0
            errors.append(rel_error)
            if rel_error > worst_error:
                worst_error = rel_error
                worst_coords = held_coords

    if not errors:
        mre = 0.0
        p90 = 0.0
        max_err = 0.0
    else:
        errors_sorted = sorted(errors)
        mre = sum(errors) / len(errors)
        p90_idx = math.ceil(0.9 * len(errors_sorted)) - 1
        p90 = errors_sorted[max(p90_idx, 0)]
        max_err = max(errors)

    if all_utilities:
        utility_min = min(all_utilities)
        utility_max = max(all_utilities)
    else:
        utility_min = 0.0
        utility_max = 0.0

    return HoldoutResult(
        op_name=op_name,
        backend=backend,
        point_count=len(errors),
        mre_percent=mre,
        p90_percent=p90,
        max_percent=max_err,
        worst_coords=worst_coords,
        skipped_slices=skipped_slices,
        utility_min=utility_min,
        utility_max=utility_max,
    )


def holdout_report(
    root: str | Path,
    *,
    gpu: str,
    gpu_spec: GpuSpec | None = None,
    fraction: float = 1.0,
    seed: int = 0,
) -> tuple[HoldoutResult, ...]:
    """
    Walk a perf-table tree and return one `HoldoutResult` per `(op_name, backend)`.

    Walks `{root}/{gpu}/{family}/{backend}/{version}/*_perf.parquet` and evaluates
    the held-out accuracy for every parquet file found. Results are sorted
    deterministically by `(op_name, backend)`.

    Args:
        root (str | Path): Directory holding `{gpu}/{family}/{backend}/{version}/`.
        gpu (str): GPU slug or device name; also selects the bundled spec.
        gpu_spec (GpuSpec | None): Hardware spec for SOL. Defaults to the bundled
            spec for `gpu`.
        fraction (float): Fraction of bracketed points to hold out.
        seed (int): Random seed for reproducible sampling.

    Returns:
        tuple[HoldoutResult, ...]: Sorted results, empty if the tree has no parquet files.
    """
    root = Path(root)
    slug = gpu_slug(gpu)
    spec = gpu_spec or load_gpu_spec(slug)
    gpu_dir = root / slug
    if not gpu_dir.is_dir():
        return ()

    # Resolve load_floors lazily to avoid the circular import:
    # floor.py imports evaluate_holdout from this module.
    from .floor import load_floors

    results: list[HoldoutResult] = []
    for family_dir in sorted(gpu_dir.iterdir()):
        if not family_dir.is_dir():
            continue
        for backend_dir in sorted(family_dir.iterdir()):
            if not backend_dir.is_dir():
                continue
            backend = backend_dir.name
            for version_dir in sorted(backend_dir.iterdir()):
                if not version_dir.is_dir():
                    continue
                # Load floors.json once per version_dir, and only when some op here wants a
                # floor, so the fallback warning is not emitted for a GEMM-only directory.
                version_floors: dict[str, float] | None = None
                if any(op.fits_floor for op in ops_in(version_dir)):
                    version_floors = load_floors(version_dir, backend=backend, warn_missing=True)
                for op in ops_in(version_dir):
                    try:
                        table = KernelTable.load(version_dir / f"{op.name}_perf.parquet", op)
                    except KernelCoverageError:
                        continue
                    floor_override = (
                        version_floors.get(op.name) if version_floors is not None else None
                    )
                    result = evaluate_holdout(
                        table,
                        op_name=op.name,
                        backend=backend,
                        gpu_spec=spec,
                        fraction=fraction,
                        seed=seed,
                        floor_override=floor_override,
                    )
                    results.append(result)

    results.sort(key=lambda r: (r.op_name, r.backend))
    return tuple(results)
