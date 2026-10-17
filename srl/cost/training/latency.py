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

A quadratic token term was tried, rejected, and later restored. On short sequences it
improves in-sample error without improving leave-one-out error, and attention accounts
for only a few percent of a step under full recompute, so there it absorbs residual
rather than representing a mechanism. Past roughly 2048 tokens attention does dominate
and the term earns its place; `design_row` carries the reasoning. If it is ever dropped
again, check the implied utilization against the device's throughput before trusting it.

The residual that remains is structured: the shortest sequence is over-predicted by
about 6 to 7% in every cell, the regime where small GEMMs underutilize the device so the
effective per-token rate is worse than at longer sequences. Capturing that needs a
throughput curve rather than a constant `token_cost`: a kernel-level concern that the
rollout cost model already solves by interpolating measured utility, so feeding training
GEMMs through the same kernel tables is the principled next step rather than another
fitted polynomial term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from srl.profiler.schema import SequenceStats

from .base import TrainingParallelism


@dataclass(frozen=True)
class LatencyFeatures:
    """Design row for one candidate: the pipeline-slot and token structure."""

    tokens: float
    slots: float
    sum_squared_tokens: float

    # Totals over the real microbatch sequence, plus the bubble charged at the
    # largest microbatch. On a uniform sequence these equal `slots * tokens` and
    # `slots * sum_squared_tokens` exactly, so a fit built before they existed
    # stays valid; they only differ once the microbatches are ragged.
    weighted_tokens: float = 0.0
    weighted_sum_squared_tokens: float = 0.0
    weighted_slots: float = 0.0

    def design_row(self) -> tuple[float, float, float, float]:
        """
        Return the row for
        `[per_step_overhead, token_cost, microbatch_overhead, token_quadratic_cost]`.

        The intercept is first so a rank check reads naturally against the four physical
        terms.

        The token terms are totals over the actual microbatch sequence rather than the
        largest microbatch times the slot count. A pipeline stage runs every microbatch
        it is given, so a step costs the sum of their costs; only the `pp - 1` bubble
        slots it cannot fill are charged at the largest, since that is what sets the
        drain rate. Charging every slot the maximum treats a ragged batch as though
        every sequence were the longest, which both overpredicts it and flattens the
        prediction to near-constant, because the longest sequence barely moves between
        steps while the token total varies widely.

        Profiling measures uniform microbatches, where the two forms coincide, which is
        why every fitted artifact remains valid without a refit.

        The quadratic term sums each sequence's square rather than squaring a total.
        Attention does not cross sequence boundaries, so a packed microbatch costs the
        sum of each sequence's square; squaring the packed total would overcharge it by
        the sequence count.

        Both backends carry the term because both need it once the profiled sequence
        length is long enough for attention to matter. Leave-one-out on measured
        candidates favours the linear form below roughly 2048 tokens and the quadratic
        form decisively above it, so the earlier decision to drop this term was right
        for the domain it was tested in and wrong for the one the simulator needs.
        """
        return (
            1.0,
            self.weighted_tokens,
            self.weighted_slots,
            self.weighted_sum_squared_tokens,
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

    A stage runs every microbatch it is handed, so the step costs their total. The
    `pp - 1` bubble slots it cannot fill are charged at the largest microbatch, which
    is what sets the rate at which the pipeline fills and drains.

    `tokens` and `sum_squared_tokens` stay as the per-microbatch maxima, because they
    are what the fitted `feature_ranges` gate on: exceeding the profiled range is a
    property of one microbatch, not of the batch total.

    All three workload terms are divided by `pp`, which is what makes pipeline
    parallelism derivable rather than something to profile. A 1F1B stage owns
    `num_layers / pp` of the layers, so its cost per slot falls by `pp` while the
    slot count rises to `nmb + pp - 1`; both factors are known from the topology
    alone. On a sweep holding TP and DP fixed, predicting an unprofiled PP costs a
    few percent, against an order of magnitude more when the terms are left
    unscaled. Point-to-point activation sends between stages are folded into the
    fitted per-microbatch cost rather than modelled, which the same sweep shows is
    good enough at these shapes.

    Establishing this needs a sweep that pins DP. Letting DP absorb whatever TP and
    PP leave over makes every PP comparison a DP comparison as well, which is why
    `enumerate_training_candidates` accepts an explicit `data_parallel_sizes`.
    """
    if not microbatches:
        raise ValueError("Latency features need at least one microbatch.")
    max_tokens = float(max(item.sum_tokens for item in microbatches))
    max_squared = float(max(item.sum_squared_tokens for item in microbatches))
    bubble = float(parallelism.pp - 1)
    # Per-layer normalisation: one stage's share of the model.
    stage_share = 1.0 / float(parallelism.pp)
    return LatencyFeatures(
        tokens=max_tokens,
        slots=pipeline_slots(len(microbatches), parallelism.pp),
        sum_squared_tokens=max_squared,
        weighted_tokens=(
            float(sum(item.sum_tokens for item in microbatches)) + bubble * max_tokens
        )
        * stage_share,
        weighted_sum_squared_tokens=(
            float(sum(item.sum_squared_tokens for item in microbatches)) + bubble * max_squared
        )
        * stage_share,
        weighted_slots=(float(len(microbatches)) + bubble) * stage_share,
    )


def robust_step_time(step_times: tuple[float, ...]) -> float:
    """
    Reduce one candidate's retained step times to a single robust estimate.

    The median, not the mean. Step times are right-skewed: an interfering process or a
    straggling collective inflates individual steps but nothing makes a step faster than
    the hardware allows, so the mean is biased upward while the median is not. A single
    outlier has been seen to double the spread across a candidate's retained steps.

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
    # Kept in lockstep with `LatencyFeatures.design_row`, so the fit and the
    # predictor cannot disagree about what the coefficients multiply.
    latency = (
        coefficients["per_step_overhead"]
        + coefficients["token_cost"] * features.weighted_tokens
        + coefficients["token_quadratic_cost"] * features.weighted_sum_squared_tokens
        + coefficients["microbatch_overhead"] * features.weighted_slots
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
        # The quadratic feature needs its own gate: `tokens` is a per-microbatch
        # maximum, so it passes whatever the batch size, while this is a batch total
        # that grows with both. Ungated, a grid that never co-varies the two lets the
        # fit park long-sequence cost here, and the real batch then multiplies it by
        # the sequence count.
        "sum_squared_tokens": (
            min(item.weighted_sum_squared_tokens for item in all_features),
            max(item.weighted_sum_squared_tokens for item in all_features),
        ),
    }
    ranges.update(extra or {})
    return ranges
