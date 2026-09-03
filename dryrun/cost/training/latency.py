"""Physically structured training step latency.

The previous form was a single linear combination,
`a_q*Q_eff + b_t*T_eff + c_pipeline*(nmb + pp - 1) + update_overhead`, where `Q_eff` and
`T_eff` folded the pipeline bubble into the workload features themselves. Three problems
followed from that.

The features were collinear by construction. With micro-batch size one and a uniform
shape, the design row is `[k*S^2, k*S, k, 1]` with `k = nmb + pp - 1`, so the rank-four
requirement could only be met by varying two axes at once, and the three non-constant
columns stayed proportional to one another within a cell. The quadratic term therefore
absorbed whatever the linear term could not explain rather than representing attention.

Costs that scale differently were summed with one coefficient each. A training step is
three physically distinct pieces: work proportional to the tokens in a microbatch, a
fixed cost paid once per microbatch, and a fixed cost paid once per step. Measured
per-microbatch times grow by factors of 1.0 to 1.9 when the sequence length doubles.
Pure token-proportional work would give 2.0 and attention-dominated work would give 4.0,
so the sub-linear growth is a fixed per-microbatch cost that a token-proportional term
cannot express.

And the pipeline bubble was inseparable from the workload. Modelling it as its own
multiplier on the per-microbatch cost keeps it interpretable: `nmb + pp - 1` slots for
1F1B, which reduces to `nmb` without pipelining.

The resulting form, per data-parallel rank:

    step = per_step_overhead + slots * (token_cost * tokens + microbatch_overhead)

with `slots = nmb + pp - 1` and `tokens` the largest microbatch's token count. All three
coefficients are non-negative and separately identifiable: sequence length at fixed
microbatch count moves only `token_cost`, microbatch count at fixed sequence length
separates `microbatch_overhead` from `per_step_overhead`.

A quadratic token term was tried and rejected. Adding `slots * tokens^2` improves the
in-sample error on clean data (6.60% to 4.65% at TP1/PP1) but not leave-one-out error
(10.24% to 10.38%), so it buys no generalization. It is also not physically attention:
the fitted coefficient implies roughly 18000% of the H20's bf16 throughput, and directly
measured attention kernels account for only 10 to 45 ms of a 515 to 1297 ms step, about
2 to 3%. Attention is not the bottleneck at these sequence lengths under full recompute,
so a quadratic term absorbs residual rather than representing a mechanism. Do not re-add
it without sequence lengths long enough for attention to dominate, and check the implied
utilization when doing so.

The residual that remains is structured: the shortest sequence is over-predicted by
about 6 to 7% in every cell, the regime where small GEMMs underutilize the device so the
effective per-token rate is worse than at longer sequences. Capturing that needs a
throughput curve rather than a constant `token_cost` -- a kernel-level concern that the
rollout cost model already solves by interpolating measured utility, so feeding training
GEMMs through the same kernel tables is the principled next step rather than another
fitted polynomial term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dryrun.profiler.schema import SequenceStats

from .base import TrainingParallelism


@dataclass(frozen=True)
class LatencyFeatures:
    """Design row for one candidate: the pipeline-slot and token structure."""

    tokens: float
    slots: float
    sum_squared_tokens: float

    def design_row(self) -> tuple[float, float, float, float]:
        """
        Return the row for
        `[per_step_overhead, token_cost, microbatch_overhead, token_quadratic_cost]`.

        The intercept is first so a rank check reads naturally against the four physical
        terms.

        The quadratic term is `slots * sum_squared_tokens`, not `slots * tokens ** 2`.
        Attention does not cross sequence boundaries, so a packed microbatch costs the sum
        of each sequence's square; squaring the packed total would overcharge it by the
        sequence count, an 8x error for eight packed sequences.

        Both backends carry the term because both need it once the profiled sequence length
        is long enough for attention to matter. Measured by leave-one-out on real Megatron
        candidates: at seq <= 2048 the linear form wins (14.17% against 16.29% at TP1/PP1),
        but at seq <= 8192 the quadratic form wins decisively (60.48% against 5.52%). The
        earlier decision to reject this term was correct for the domain it was tested in and
        wrong for the domain the simulator actually needs.
        """
        return (
            1.0,
            self.slots * self.tokens,
            self.slots,
            self.slots * self.sum_squared_tokens,
        )


def pipeline_slots(num_microbatches: int, pipeline_parallel: int) -> float:
    """
    Return the number of serialized microbatch slots on the critical path.

    Under 1F1B a stage runs `nmb` microbatches plus a `pp - 1` bubble it cannot fill,
    which collapses to `nmb` when there is no pipeline.
    """
    if num_microbatches <= 0 or pipeline_parallel <= 0:
        raise ValueError("num_microbatches and pipeline_parallel must be positive.")
    return float(num_microbatches + pipeline_parallel - 1)


def latency_features(
    microbatches: tuple[SequenceStats, ...],
    parallelism: TrainingParallelism,
) -> LatencyFeatures:
    """
    Build the design row for one rank's microbatch sequence.

    The critical path is set by the largest microbatch, since the pipeline advances at
    the speed of its slowest stage occupant.
    """
    if not microbatches:
        raise ValueError("Latency features need at least one microbatch.")
    return LatencyFeatures(
        tokens=float(max(item.sum_tokens for item in microbatches)),
        slots=pipeline_slots(len(microbatches), parallelism.pp),
        sum_squared_tokens=float(max(item.sum_squared_tokens for item in microbatches)),
    )


def robust_step_time(step_times: tuple[float, ...]) -> float:
    """
    Reduce one candidate's retained step times to a single robust estimate.

    The median, not the mean. Step times are right-skewed: an interfering process or a
    straggling collective inflates individual steps but nothing makes a step faster than
    the hardware allows, so the mean is biased upward while the median is not. One run
    showed a 107.7% spread across three retained steps from a single outlier.

    Every step also used to enter the fit as its own row, which let a candidate with an
    outlier outvote a clean one. Collapsing to one row per candidate weights each
    profiled configuration equally, which is what the model is meant to be fitted over.
    """
    if not step_times:
        raise ValueError("A candidate needs at least one retained step time.")
    return float(np.median(np.asarray(step_times, dtype=np.float64)))


def step_time_dispersion(step_times: tuple[float, ...]) -> float:
    """
    Return the relative median absolute deviation of a candidate's step times.

    Reported as a diagnostic so a noisy candidate is visible in the artifact instead of
    silently widening the fit. Values above roughly 0.05 mean the measurement, not the
    model, is the limiting factor.
    """
    if len(step_times) < 2:
        return 0.0
    values = np.asarray(step_times, dtype=np.float64)
    median = float(np.median(values))
    if median <= 0:
        return 0.0
    return float(np.median(np.abs(values - median)) / median)


# Number of fitted latency coefficients, and therefore the rank the design must reach. Both
# backends share it because they share the design row.
LATENCY_TERM_COUNT = 4

LATENCY_COEFFICIENT_NAMES = (
    "per_step_overhead",
    "token_cost",
    "microbatch_overhead",
    "token_quadratic_cost",
)


def latency_design_arrays(
    features_and_targets: list[tuple[LatencyFeatures, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stack design rows and targets for a least-squares latency fit.

    Args:
        features_and_targets (list[tuple[LatencyFeatures, float]]): One pair per candidate,
            its step times already reduced to a single robust value.

    Returns:
        tuple[np.ndarray, np.ndarray]: The design matrix and the target vector.
    """
    rows = [features.design_row() for features, _ in features_and_targets]
    targets = [target for _, target in features_and_targets]
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def check_latency_rank(
    rank: int,
    *,
    backend: str,
    key: object,
    slots_feature: str,
) -> None:
    """
    Raise unless the design reached the rank the four-term form needs.

    Args:
        rank (int): Rank reported by the fit.
        backend (str): Backend name, for the message only.
        key (object): Cell key, for the message only.
        slots_feature (str): What the backend calls its slots axis, so the message names the
            knob the caller actually sets.

    Raises:
        ValueError: If `rank` is below `LATENCY_TERM_COUNT`.
    """
    if rank >= LATENCY_TERM_COUNT:
        return
    raise ValueError(
        f"{backend} latency design matrix is rank {rank}, expected {LATENCY_TERM_COUNT} for "
        f"{key!r}. Vary both the sequence length and {slots_feature}, and use at least three "
        f"sequence lengths: at a fixed {slots_feature} the slots column is constant and "
        f"collinear with the intercept, and two sequence lengths leave the linear and "
        f"quadratic token columns collinear."
    )


def latency_coefficients(parameters: np.ndarray) -> dict[str, float]:
    """Name the fitted latency parameters in design-row order."""
    if len(parameters) != LATENCY_TERM_COUNT:
        raise ValueError(
            f"Expected {LATENCY_TERM_COUNT} latency parameters, got {len(parameters)}."
        )
    return {
        name: float(value)
        for name, value in zip(LATENCY_COEFFICIENT_NAMES, parameters, strict=True)
    }


def latency_from_coefficients(
    coefficients: dict[str, float],
    features: LatencyFeatures,
) -> float:
    """
    Evaluate the fitted latency form at one candidate's features.

    Both backends call this rather than inlining the arithmetic. The two copies were
    byte-identical once the quadratic term landed, and a formula written twice is a formula
    that can drift: this codebase has already had a fit and its predictor disagree on the
    same input.

    Args:
        coefficients (dict[str, float]): The four fitted terms.
        features (LatencyFeatures): The same features the fit was built from.

    Returns:
        float: Predicted step latency in seconds, strictly positive.

    Raises:
        ValueError: If the prediction is not positive.
    """
    latency = coefficients["per_step_overhead"] + features.slots * (
        coefficients["token_cost"] * features.tokens
        + coefficients["token_quadratic_cost"] * features.sum_squared_tokens
        + coefficients["microbatch_overhead"]
    )
    if latency <= 0:
        raise ValueError("Training latency prediction must be positive.")
    return latency


def latency_feature_ranges(
    all_features: list[LatencyFeatures],
    *,
    slots_feature: str,
    extra: dict[str, tuple[float, float]] | None = None,
) -> dict[str, tuple[float, float]]:
    """
    Return the profiled span of each latency feature, which becomes a prediction-time gate.

    Args:
        all_features (list[LatencyFeatures]): Features of every successful candidate.
        slots_feature (str): The backend's name for its slots axis, so the recorded range
            matches the name `predict` checks against.
        extra (dict[str, tuple[float, float]] | None): Backend-specific ranges to merge in.

    Returns:
        dict[str, tuple[float, float]]: Feature name to `(minimum, maximum)`.
    """
    ranges = {
        "tokens": (
            min(item.tokens for item in all_features),
            max(item.tokens for item in all_features),
        ),
        slots_feature: (
            min(item.slots for item in all_features),
            max(item.slots for item in all_features),
        ),
    }
    ranges.update(extra or {})
    return ranges
