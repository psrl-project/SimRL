"""Robust fitting primitives shared by the training cost models.

Every training backend faces the same statistical problem: a handful of profiled
candidates, right-skewed measurement noise, and coefficients that are physically
non-negative. Keeping the estimator, the run-level split, and the domain check here
rather than once per backend means a correction to the outlier rule cannot apply to one
backend and silently miss another.
"""

from __future__ import annotations

import hashlib

import numpy as np

from srl.cost.artifact import ArtifactEntry
from srl.profiler.schema import ProfileRecord

from .base import TrainingOutOfDomainError


def split_records_by_run(
    records: list[ProfileRecord],
    *,
    validation_fraction: float,
) -> tuple[list[ProfileRecord], list[ProfileRecord]]:
    """
    Split candidates into train and validation sets by run identity.

    The split is on `run_id` rather than on rows so repeats of one configuration cannot
    straddle the boundary and leak. Ordering is a hash of the identifier, so the same
    profile always produces the same split without depending on file order.

    At `validation_fraction=0`, or with fewer than two distinct runs, the same list is
    returned twice, so a caller's validation metrics are its training metrics. Callers
    that store the result as `validation_metrics` must record that in their diagnostics or
    a reader of the artifact cannot tell an in-sample score from a held-out one.
    """
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    run_ids = sorted({record.run.run_id for record in records})
    if validation_fraction == 0 or len(run_ids) < 2:
        return records, records
    ranked = sorted(
        run_ids,
        key=lambda run_id: hashlib.sha256(run_id.encode("utf-8")).digest(),
    )
    count = max(1, round(len(ranked) * validation_fraction))
    validation_ids = set(ranked[:count])
    train = [record for record in records if record.run.run_id not in validation_ids]
    validation = [record for record in records if record.run.run_id in validation_ids]
    return (train, validation) if train else (records, records)


def validation_is_in_sample(
    records: list[ProfileRecord],
    *,
    validation_fraction: float,
) -> bool:
    """
    Report whether `split_records_by_run` would hand back the same list twice.

    Defined once here rather than restated per backend, so the condition and the split it
    describes cannot drift apart. An artifact needs this recorded because an in-sample
    coefficient of determination of 1.0 is indistinguishable from a held-out one on
    inspection, and every entry stores the score under the name `validation_metrics`.
    """
    if validation_fraction == 0:
        return True
    run_ids = {record.run.run_id for record in records}
    if len(run_ids) < 2:
        return True
    # The split also falls back to the full list when the held-out share leaves no
    # training records at all, which one run per hash bucket can do at a large fraction.
    train, _ = split_records_by_run(records, validation_fraction=validation_fraction)
    return len(train) == len(records)


def scaled_nonnegative_fit(
    design: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Solve a non-negative least-squares problem with one outlier-rejection pass.

    Columns are scaled to comparable magnitudes first, because a token count and an
    intercept differ by several orders of magnitude and an unscaled solve spends its
    conditioning budget on that rather than on the fit. The soft-L1 loss keeps a single
    straggling step from dominating, and the second pass drops rows beyond 4.5 robust
    standard deviations only when enough rows remain to keep the design determined.

    Residuals are **relative**, not absolute. A training sweep spans orders of
    magnitude in step time, so minimising absolute error optimises the largest
    candidates while letting the smallest absorb almost any relative error. An
    absolute fit has been seen to drive `token_cost` to near zero, absorbing the
    linear term into the quadratic one, and still leave large error on mid-sized
    candidates. What the simulator needs is a percentage, since a one-second error
    matters on a two-second step and not on a two-hundred-second one.

    Returns the coefficients in unscaled units, the inlier mask, and the rank of the
    original design so the caller can reject an unidentifiable cell.
    """
    scales = np.maximum(np.max(np.abs(design), axis=0), 1.0)
    scaled = design / scales
    # Guard against a zero or negative target, which a latency measurement
    # should never be but which would invert the weighting if it occurred.
    weights = 1.0 / np.maximum(np.abs(target), np.finfo(np.float64).eps)
    initial = np.maximum(
        np.linalg.lstsq(scaled * weights[:, None], target * weights, rcond=None)[0], 0
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        return (scaled @ parameters - target) * weights

    # Deferred because only fitting needs scipy, and simulation should not pay its import.
    from scipy.optimize import least_squares

    first = least_squares(
        residual,
        x0=np.maximum(initial, np.finfo(np.float64).eps),
        bounds=(0, np.inf),
        loss="soft_l1",
        max_nfev=50_000,
    )
    raw_residual = residual(first.x)
    median = float(np.median(raw_residual))
    mad = float(np.median(np.abs(raw_residual - median)))
    if mad <= np.finfo(np.float64).eps:
        inliers = np.ones(len(target), dtype=bool)
    else:
        inliers = np.abs(raw_residual - median) <= 4.5 * 1.4826 * mad
    if int(np.sum(inliers)) > design.shape[1]:
        second = least_squares(
            lambda parameters: residual(parameters)[inliers],
            x0=first.x,
            bounds=(0, np.inf),
            loss="soft_l1",
            max_nfev=50_000,
        )
        fitted = second.x
    else:
        fitted = first.x
        inliers = np.ones(len(target), dtype=bool)
    return fitted / scales, inliers, int(np.linalg.matrix_rank(design))


def check_feature_range(
    entry: ArtifactEntry,
    name: str,
    value: float,
    *,
    extrapolate: frozenset[str],
) -> str | None:
    """Check whether a query is inside the profiled range for one feature.

    When the value is in range the call returns None. When it is out of range and the
    feature name is in `extrapolate`, the call succeeds and returns a diagnostic string
    describing the exceedance: the caller must record it so extrapolations are never
    silent. When it is out of range and the feature name is not in `extrapolate`, the call
    raises `TrainingOutOfDomainError` naming the feature, its value, and the range.

    `extrapolate` is validated at model construction time (see
    `validate_extrapolate_names`), so an unrecognised name here is an implementation
    error, not a user error.

    A feature the artifact does not record is not checked. Ranges are written at
    fit time, so an artifact produced before a gate was added simply has no entry
    for it, and refusing to predict would make every older artifact unusable for a
    check it was never able to record. Refitting restores the gate.
    """
    if name not in entry.feature_ranges:
        return None
    lower, upper = entry.feature_ranges[name]
    if lower <= value <= upper:
        return None
    if name in extrapolate:
        return (
            f"extrapolated {name!r}={value} outside profiled range [{lower}, {upper}]"
        )
    raise TrainingOutOfDomainError(
        f"Feature {name!r}={value} is outside the profiled range [{lower}, {upper}]."
    )


def validate_extrapolate_names(
    names: frozenset[str],
    valid: frozenset[str],
    *,
    context: str = "artifact",
) -> None:
    """Raise ValueError if any name is not a valid feature for this artifact.

    `valid` comes from an entry's `feature_ranges` keys so the check is automatically
    correct as the feature set evolves. An unknown name like `num_microbatch` (a typo)
    silently enables nothing and defeats the whole point of the explicit list.

    Args:
        names: The caller-supplied set of feature names to permit.
        valid: The complete set of known feature names for this artifact component.
        context: A short description of the artifact or component, used in the message.
    """
    unknown = names - valid
    if unknown:
        raise ValueError(
            f"Unknown extrapolation feature(s) {sorted(unknown)} for {context}. "
            f"Valid names are {sorted(valid)}."
        )
