"""Shared numerical helpers for rollout cost model fitting."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from dryrun.profiler.io import profile_digest
from dryrun.profiler.schema import Parallelism, ProfileRecord, RolloutPayload


@dataclass(frozen=True)
class RolloutSample:
    """Keep one payload with its launch group for leakage-free holdout."""

    run_id: str
    payload: RolloutPayload


def rollout_groups(
    records: Iterable[ProfileRecord],
) -> tuple[
    dict[Parallelism, list[RolloutSample]],
    list[ProfileRecord],
]:
    """Validate rollout records and group successful observations by topology."""
    materialized = list(records)
    if not materialized:
        raise ValueError("At least one profiling record is required.")
    groups: dict[Parallelism, list[RolloutSample]] = defaultdict(list)
    for record in materialized:
        if record.record_type != "rollout":
            continue
        if record.run.status != "success":
            continue
        payload = record.payload
        if not isinstance(payload, RolloutPayload):
            continue
        if payload.latency_s <= 0:
            continue
        groups[record.parallelism].append(RolloutSample(record.run.run_id, payload))
    if not groups:
        raise ValueError("No successful rollout observations are available.")
    return dict(groups), materialized


def artifact_context(records: list[ProfileRecord]) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Return stable model, hardware, and profile identity values."""
    first = records[0]
    model_names = {record.model.name for record in records}
    gpu_names = {record.hardware.gpu_name for record in records}
    if len(model_names) != 1 or len(gpu_names) != 1:
        raise ValueError("A fitted artifact must target one model and one GPU type.")
    hardware = asdict(first.hardware)
    hardware["gpu_count"] = sorted({record.hardware.gpu_count for record in records})
    return asdict(first.model), hardware, profile_digest(records)


def split_by_run(
    samples: list[RolloutSample],
    *,
    validation_fraction: float,
) -> tuple[list[RolloutSample], list[RolloutSample]]:
    """Split whole benchmark launches so adjacent token steps cannot leak."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    run_ids = sorted({sample.run_id for sample in samples})
    if validation_fraction == 0:
        return samples, []
    if len(run_ids) < 2:
        return samples, samples
    ranked = sorted(
        run_ids,
        key=lambda run_id: hashlib.sha256(run_id.encode("utf-8")).digest(),
    )
    validation_count = max(1, round(len(ranked) * validation_fraction))
    validation_ids = set(ranked[:validation_count])
    train = [sample for sample in samples if sample.run_id not in validation_ids]
    validation = [sample for sample in samples if sample.run_id in validation_ids]
    if not train:
        return samples, samples
    return train, validation


def robust_nonnegative_fit(
    predict: Callable[[np.ndarray], np.ndarray],
    *,
    initial: np.ndarray,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit non-negative parameters with soft-L1 loss and MAD refitting."""
    first = least_squares(
        lambda parameters: predict(parameters),
        x0=np.maximum(initial, np.finfo(np.float64).eps),
        bounds=(0, np.inf),
        loss="soft_l1",
        max_nfev=50_000,
    )
    residual = predict(first.x)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    if mad <= np.finfo(np.float64).eps:
        inlier_mask = np.ones(n_samples, dtype=bool)
    else:
        inlier_mask = np.abs(residual - median) <= 4.5 * 1.4826 * mad
    if int(np.sum(inlier_mask)) <= len(initial):
        return first.x, np.ones(n_samples, dtype=bool)

    second = least_squares(
        lambda parameters: predict(parameters)[inlier_mask],
        x0=first.x,
        bounds=(0, np.inf),
        loss="soft_l1",
        max_nfev=50_000,
    )
    return second.x, inlier_mask


def feature_ranges(samples: list[RolloutSample]) -> dict[str, tuple[float, float]]:
    """Record the numerical domain represented by a fitted cell."""
    values = {
        "n_tokens": [float(sample.payload.n_tokens) for sample in samples],
        "sum_l2": [float(sample.payload.sum_l2) for sample in samples],
        "ctxsum": [float(sample.payload.ctxsum) for sample in samples],
        "request_count": [float(sample.payload.request_count) for sample in samples],
    }
    return {
        name: (min(feature_values), max(feature_values))
        for name, feature_values in values.items()
    }
