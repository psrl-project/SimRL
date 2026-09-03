"""Fit and persist per-backend, per-op dispatch floors.

The floor is a fitted hyperparameter: it is the value of `floor_s` in
`max(floor_s, roofline_work)` that minimises the held-out mean relative error
(MRE) over the full bracketed set.  A pure roofline sets `floor_s = 0`, which
causes large errors in the dispatch-limited regime where the GPU finishes the
kernel work faster than the Python/dispatch path can schedule the next call.

The fitting procedure is a coarse scan over `CANDIDATE_FLOORS_US` followed by
reading the minimiser.  The MRE-vs-floor curve is unimodal in practice, so a
scan finds the global minimum.  Running 2 ops × 12 candidates costs roughly
a second at the grid sizes used in practice (H20 + Qwen3-8B TP=1), so this is
cheap enough to run as part of the profiling workflow.

Which ops get a floor is the op's own decision, read from `op.fits_floor`, so a
new floor-taking op is fitted here with no edit to this module.  GEMM declares
`fits_floor = False` and is skipped.

Results are written to `floors.json` **inside the version directory** —
`{root}/{gpu}/{family}/{backend}/{version}/floors.json` — which is already
scoped to the exact (backend, version) pair the floor is valid for, and keyed by
op name so no site has to translate between a phase and an op.

`KernelDatabase` and `holdout_report` load that file and use it in preference
to the hardcoded fallback in `FALLBACK_FLOORS_S`, with a warning when the file
is absent so the fallback is never silent. A floor-taking op the fallback has no
entry for gets a second, per-op warning: it takes a zero floor, and reporting only
"falling back to FALLBACK_FLOORS_S" would name a fallback that holds nothing for it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .hardware import GpuSpec, gpu_slug, load_gpu_spec
from .holdout import evaluate_holdout
from .interp import KernelCoverageError
from .ops.base import op_for, ops_in
from .sol import FALLBACK_FLOORS_S
from .table import KernelTable

dryrun_logger = logging.getLogger(__file__)

# Candidate floors (in microseconds), converted to seconds during the scan.
# The list spans zero through values well above any observed minimum so the
# scan brackets the optimum on both sides.
CANDIDATE_FLOORS_US: tuple[float, ...] = (0, 2, 4, 6, 8, 12, 16, 22, 32, 45, 60, 80)

_LEGACY_PHASE_KEYS = frozenset({"context", "generation"})
_METADATA_KEYS = frozenset({"fitted_mre_percent", "candidates_us", "gpu", "backend"})


def fit_floors(
    version_dir: Path,
    *,
    backend: str,
    gpu: str,
    gpu_spec: GpuSpec | None = None,
    candidates_us: tuple[float, ...] = CANDIDATE_FLOORS_US,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Scan candidate floors for every floor-taking op in one version directory.

    For each parquet file in `version_dir` whose op declares `fits_floor`, sweeps
    `candidates_us` and picks the floor that minimises the held-out MRE (full set,
    fraction=1.0). Ops declaring `fits_floor = False` are skipped.

    Args:
        version_dir (Path): The `{root}/{gpu}/{family}/{backend}/{version}/` directory.
        backend (str): Backend label (e.g. `vllm`, `flashinfer`).
        gpu (str): GPU slug or device name; also selects the bundled spec when
            `gpu_spec` is None.
        gpu_spec (GpuSpec | None): Hardware spec for SOL. Defaults to the bundled
            spec for `gpu`.
        candidates_us (tuple[float, ...]): Floor candidates in microseconds.
        seed (int): Random seed for reproducible sampling inside `evaluate_holdout`.

    Returns:
        dict[str, Any]: One key per fitted op name, plus `fitted_mre_percent`,
            `candidates_us`, `gpu`, `backend`. Only ops whose parquet file exists
            appear in the fitted values.
    """
    slug = gpu_slug(gpu)
    spec = gpu_spec or load_gpu_spec(slug)
    fitted_floors: dict[str, float] = {}
    fitted_mre: dict[str, float] = {}

    for op in ops_in(version_dir):
        if not op.fits_floor:
            # GEMM declares fits_floor = False: its regime is compute- or bandwidth-limited,
            # so there is no dispatch cost for a floor to represent.
            continue
        parquet_path = version_dir / f"{op.name}_perf.parquet"
        try:
            table = KernelTable.load(parquet_path, op)
        except Exception as exc:
            dryrun_logger.warning(
                f"Skipping {op.name} floor fit for {backend!r}: could not load table: {exc}."
            )
            continue

        best_floor_s = 0.0
        best_mre = float("inf")
        for candidate_us in candidates_us:
            candidate_s = candidate_us * 1e-6
            result = evaluate_holdout(
                table,
                op_name=op.name,
                backend=backend,
                gpu_spec=spec,
                fraction=1.0,
                seed=seed,
                floor_override=candidate_s,
            )
            if result.point_count > 0 and result.mre_percent < best_mre:
                best_mre = result.mre_percent
                best_floor_s = candidate_s
            dryrun_logger.debug(
                f"  {backend}/{op.name} floor {candidate_us:.0f} us -> MRE {result.mre_percent:.2f}%."
            )

        fitted_floors[op.name] = best_floor_s
        fitted_mre[op.name] = round(best_mre, 4)
        dryrun_logger.info(
            f"Fitted {backend}/{op.name}: floor {best_floor_s * 1e6:.0f} us, MRE {best_mre:.2f}%."
        )

    return {
        **fitted_floors,
        "fitted_mre_percent": fitted_mre,
        "candidates_us": list(candidates_us),
        "gpu": slug,
        "backend": backend,
    }


def write_floors(version_dir: Path, data: dict[str, Any]) -> Path:
    """
    Persist fitted floors to `{version_dir}/floors.json`.

    Args:
        version_dir (Path): Directory to write into.
        data (dict[str, Any]): Output of `fit_floors`.

    Returns:
        Path: The written file path.
    """
    output = version_dir / "floors.json"
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output


def _warn_unfloored_ops(version_dir: Path, *, backend: str, available: dict[str, float]) -> None:
    """Warn once per floor-taking op in `version_dir` that the fallback does not cover.

    Falling back to `FALLBACK_FLOORS_S` is only reassuring when the fallback holds an entry for
    the op being priced. When it does not, the op silently takes a zero floor, and for
    `all_reduce` at 4096 bytes that is a 220x SOL error. This makes the loudness proportional to
    what is actually missing rather than reporting a fallback that falls back to nothing.
    """
    for op in ops_in(version_dir):
        if op.fits_floor and op.name not in available:
            dryrun_logger.warning(
                f"Op {op.name!r} in {version_dir} declares a fitted floor but backend "
                f"{backend!r} has no entry for it in FALLBACK_FLOORS_S, so it will be priced "
                f"with a zero floor. Run dryrun-profile kernels fit-floor before trusting any "
                f"latency for this op."
            )


def load_floors(
    version_dir: Path,
    *,
    backend: str,
    warn_missing: bool = True,
) -> dict[str, float] | None:
    """
    Load fitted floors from `{version_dir}/floors.json`, keyed by op name.

    Args:
        version_dir (Path): The version directory that should contain `floors.json`.
        backend (str): Backend label; used in messages and to select the fallback.
        warn_missing (bool): When True, a missing file logs a warning and returns the
            hardcoded fallback. A second warning is logged for each floor-taking op in
            `version_dir` the fallback has no entry for, because such an op takes a zero
            floor and "falling back to FALLBACK_FLOORS_S" would otherwise describe a
            fallback to nothing. When False it returns None.

    Returns:
        dict[str, float] | None: `{op_name: floor_s}`, or None when the file is absent and
            `warn_missing` is False.

    Raises:
        KernelCoverageError: If the file carries the retired phase keys. Reading it leniently
            would apply a zero floor, which measures 34.89% MRE against 6.63% at the fitted
            22 us -- a silent five-fold accuracy loss.
    """
    floors_path = version_dir / "floors.json"
    if not floors_path.is_file():
        if warn_missing:
            fallback = dict(FALLBACK_FLOORS_S.get(backend, {}))
            dryrun_logger.warning(
                f"No floors.json in {version_dir}; falling back to hardcoded FALLBACK_FLOORS_S "
                f"for backend {backend!r} (fitted for H20, may be wrong for other hardware or "
                f"backend versions). Run dryrun-profile kernels fit-floor to generate it."
            )
            _warn_unfloored_ops(version_dir, backend=backend, available=fallback)
            return fallback
        return None
    try:
        raw = json.loads(floors_path.read_text(encoding="utf-8"))
    except Exception as exc:
        fallback = dict(FALLBACK_FLOORS_S.get(backend, {}))
        dryrun_logger.warning(
            f"Could not read {floors_path}: {exc}. "
            f"Falling back to hardcoded FALLBACK_FLOORS_S for backend {backend!r}."
        )
        _warn_unfloored_ops(version_dir, backend=backend, available=fallback)
        return fallback
    if _LEGACY_PHASE_KEYS & set(raw):
        raise KernelCoverageError(
            f"{floors_path} uses the retired phase keys {sorted(_LEGACY_PHASE_KEYS & set(raw))}. "
            f"Floors are now keyed by op name. Regenerate it with "
            f"dryrun-profile kernels fit-floor."
        )
    result: dict[str, float] = {}
    for key, value in raw.items():
        if key in _METADATA_KEYS:
            continue
        try:
            op_for(key)
        except KernelCoverageError:
            # A stale key from a removed op: ignore it rather than failing a whole report.
            continue
        result[key] = float(value)
    return result


def fit_floors_report(
    root: str | Path,
    *,
    gpu: str,
    gpu_spec: GpuSpec | None = None,
    candidates_us: tuple[float, ...] = CANDIDATE_FLOORS_US,
    seed: int = 0,
) -> list[Path]:
    """
    Walk the perf-table tree and fit floors for every `(family, backend, version)`.

    Walks `{root}/{gpu}/{family}/{backend}/{version}/` and calls `fit_floors` for each
    version directory holding at least one op that declares `fits_floor`. Writes
    `floors.json` into each such version directory.

    Args:
        root (str | Path): Directory holding `{gpu}/{family}/{backend}/{version}/`.
        gpu (str): GPU slug or device name; also selects the bundled spec.
        gpu_spec (GpuSpec | None): Hardware spec for SOL. Defaults to the bundled spec.
        candidates_us (tuple[float, ...]): Floor candidates in microseconds.
        seed (int): Random seed for reproducible sampling.

    Returns:
        list[Path]: Paths of every `floors.json` written.
    """
    root = Path(root)
    slug = gpu_slug(gpu)
    spec = gpu_spec or load_gpu_spec(slug)
    gpu_dir = root / slug
    if not gpu_dir.is_dir():
        dryrun_logger.warning(f"No perf tables found at {gpu_dir}.")
        return []

    written: list[Path] = []
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
                if not any(op.fits_floor for op in ops_in(version_dir)):
                    continue
                dryrun_logger.info(f"Fitting floors for {backend!r} version {version_dir.name!r}...")
                data = fit_floors(
                    version_dir,
                    backend=backend,
                    gpu=slug,
                    gpu_spec=spec,
                    candidates_us=candidates_us,
                    seed=seed,
                )
                path = write_floors(version_dir, data)
                written.append(path)
                dryrun_logger.info(f"Wrote fitted floors to {path}.")
    return written
