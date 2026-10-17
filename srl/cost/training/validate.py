"""
Score a training latency model against real profiled step times.

The rollout side has `cost/kernel/validate.py` for exactly this purpose, and the reason
training needs its own module rather than reusing it is that the two have different units
of comparison. A rollout baseline holds tens of thousands of engine iterations that must be
bucketed before scoring, because the shapes nearly repeat and the report would otherwise be
dominated by noise. A training profile holds one record per *swept candidate*, each already
reduced to a handful of step times, so there is nothing to bucket: the record is the bucket.

What this module exists to prevent is scoring a kernel composition against a workload it
was not asked to predict. A `ProfileRecord` carries the microbatch plan the measured run
actually executed (`payload.microbatches`, with per-microbatch `sum_tokens` under THD
packing). Reconstructing that plan from `sequence_lengths` through `MicrobatchPlanner` would
fold the planner's own disagreement with Megatron's batching into the latency score, and a
composition error and a planning error would become indistinguishable. So the measured plan
is passed through verbatim, and `predict_microbatches` is the entry point rather than
`predict`.

The report is per-term, not a single MRE. A kernel sum's whole value over a fitted
polynomial is that a systematic miss points to a named op, and a pooled number throws that
away. `TermBreakdown` carries the mean share of each op family so the residual can be
compared against the terms that are modelled.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

from ...profiler.io import read_records
from ...profiler.schema import ProfileRecord, TrainingPayload
from .base import TrainingParallelism
from .latency import robust_step_time


@dataclass(frozen=True)
class StepComparison:
    """One profiled candidate against its prediction."""

    label: str
    """Human-readable identity: tokens, microbatch count, and parallelism."""

    sum_tokens: int
    """Total tokens in the step, summed over microbatches."""

    num_microbatches: int
    measured_s: float
    predicted_s: float

    @property
    def relative_error(self) -> float:
        """Return the signed relative error, negative when the model under-predicts."""
        return (self.predicted_s - self.measured_s) / self.measured_s

    @property
    def ratio(self) -> float:
        """Return predicted / measured."""
        return self.predicted_s / self.measured_s


@dataclass(frozen=True)
class TrainingValidationResult:
    """Error of one training latency model over a set of profiled candidates."""

    comparisons: tuple[StepComparison, ...]
    term_shares: dict[str, float]
    """Mean fraction of the prediction contributed by each op family."""

    @property
    def record_count(self) -> int:
        """Return how many candidates were scored."""
        return len(self.comparisons)

    @property
    def mre_percent(self) -> float:
        """Return the mean absolute relative error, in percent."""
        if not self.comparisons:
            return float("nan")
        return 100.0 * statistics.fmean(abs(c.relative_error) for c in self.comparisons)

    @property
    def signed_mean_percent(self) -> float:
        """Return the mean *signed* relative error, in percent.

        Reported alongside the absolute figure because the sign is diagnostic in a way the
        magnitude is not. A kernel sum that is uniformly negative is missing an additive
        term; one that scatters around zero has a noisy term; one that is uniformly positive
        is double-counting or ignoring overlap.
        """
        if not self.comparisons:
            return float("nan")
        return 100.0 * statistics.fmean(c.relative_error for c in self.comparisons)

    @property
    def max_percent(self) -> float:
        """Return the worst absolute relative error, in percent."""
        if not self.comparisons:
            return float("nan")
        return 100.0 * max(abs(c.relative_error) for c in self.comparisons)

    @property
    def median_ratio(self) -> float:
        """Return the median predicted/measured ratio."""
        if not self.comparisons:
            return float("nan")
        return statistics.median(c.ratio for c in self.comparisons)

    @property
    def worst(self) -> StepComparison | None:
        """Return the candidate with the largest absolute error."""
        if not self.comparisons:
            return None
        return max(self.comparisons, key=lambda c: abs(c.relative_error))


def load_training_records(
    path: str | Path,
    *,
    packing: str | None = None,
) -> tuple[ProfileRecord, ...]:
    """
    Read successful training records from a profile JSONL.

    Args:
        path (str | Path): Profile JSONL written by a training adapter.
        packing (str | None): Keep only records measured under this sequence layout.
            THD and BSHD price attention differently and must not be pooled, which is why
            the filter exists rather than being left to the caller.

    Returns:
        tuple[ProfileRecord, ...]: Records with a non-empty `step_times_s`.
    """
    records = read_records(Path(path), record_type="training")
    kept = []
    for record in records:
        payload = record.payload
        if record.run.status != "success" or not isinstance(payload, TrainingPayload):
            continue
        if not payload.step_times_s:
            continue
        if packing is not None and payload.packing != packing:
            continue
        kept.append(record)
    return tuple(kept)


def score_composed_training(
    model: object,
    records: tuple[ProfileRecord, ...],
) -> TrainingValidationResult:
    """
    Score a `ComposedTraining` against profiled step times, using each record's own plan.

    The model is typed `object` rather than `TrainingLatencyModel` because this function
    calls `predict_microbatches` and `decompose_microbatches`, which are `ComposedTraining`'s
    own interface rather than the abstract one. A fitted model has no per-term decomposition
    to report, so it is scored through its own harness.

    Args:
        model (object): A `ComposedTraining`.
        records (tuple[ProfileRecord, ...]): Successful training records, one per candidate.

    Returns:
        TrainingValidationResult: Per-candidate comparisons plus mean per-term shares.

    Raises:
        KernelCoverageError: If any kernel query falls outside the measured tables. Not
            caught: a coverage gap is a missing measurement, and silently dropping the
            record would report an MRE over an unstated subset.
    """
    comparisons: list[StepComparison] = []
    term_totals: dict[str, float] = {}
    term_denominator = 0.0

    for record in records:
        payload = record.payload
        assert isinstance(payload, TrainingPayload)
        parallelism = TrainingParallelism(
            tp=record.parallelism.tp,
            pp=record.parallelism.pp,
            dp=record.parallelism.dp,
        )
        measured = robust_step_time(payload.step_times_s)
        predicted = model.predict_microbatches(payload.microbatches, parallelism)
        sum_tokens = sum(item.sum_tokens for item in payload.microbatches)
        comparisons.append(
            StepComparison(
                label=(
                    f"{sum_tokens}tok x{payload.num_microbatches}mb "
                    f"tp{parallelism.tp}pp{parallelism.pp}dp{parallelism.dp}"
                ),
                sum_tokens=sum_tokens,
                num_microbatches=payload.num_microbatches,
                measured_s=measured,
                predicted_s=predicted,
            )
        )
        terms = model.decompose_microbatches(payload.microbatches, parallelism)
        total = sum(terms.values())
        if total > 0.0:
            term_denominator += 1.0
            for name, value in terms.items():
                term_totals[name] = term_totals.get(name, 0.0) + value / total

    shares = (
        {name: value / term_denominator for name, value in sorted(term_totals.items())}
        if term_denominator
        else {}
    )
    return TrainingValidationResult(comparisons=tuple(comparisons), term_shares=shares)


def format_training_report(result: TrainingValidationResult) -> str:
    """
    Render a validation result as a fixed-width table.

    Args:
        result (TrainingValidationResult): Output of `score_composed_training`.

    Returns:
        str: Multi-line report: per-candidate rows, summary statistics, term shares.
    """
    lines = [
        f"{'candidate':34s} {'measured':>10s} {'predicted':>10s} {'error':>9s}",
        "-" * 68,
    ]
    for comparison in sorted(result.comparisons, key=lambda c: c.sum_tokens):
        lines.append(
            f"{comparison.label:34s} "
            f"{comparison.measured_s:9.4f}s "
            f"{comparison.predicted_s:9.4f}s "
            f"{100.0 * comparison.relative_error:+8.2f}%"
        )
    lines.append("-" * 68)
    lines.append(
        f"records {result.record_count}  "
        f"MRE {result.mre_percent:.2f}%  "
        f"signed {result.signed_mean_percent:+.2f}%  "
        f"max {result.max_percent:.2f}%  "
        f"median ratio {result.median_ratio:.4f}"
    )
    if result.term_shares:
        lines.append("")
        lines.append("mean share of prediction by term:")
        for name, share in sorted(result.term_shares.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name:18s} {100.0 * share:5.1f}%")
    return "\n".join(lines)
