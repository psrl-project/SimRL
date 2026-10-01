"""
Score a rollout cost model against a real vLLM profiling baseline.

A holdout gate measures whether interpolation reproduces a *single measured kernel*. It says
nothing about whether summing kernels reproduces an *engine iteration*, which is a different
claim and the one that matters. This module makes that comparison reproducible.

Bucketing is the part worth being careful about. A baseline holds tens of thousands of
iterations at nearly-repeating shapes, so they must be grouped before scoring or the report is
dominated by measurement noise. An earlier version of this grouping snapped each bucket's mean
context to a power of two, which floored a true mean of 7300 tokens to 4096 and produced a
19.1% error that did not exist: the real figure was 4.4%. Buckets here therefore round the
mean context to a linear multiple, never to a power of two, and the bucket's latency is the
median so one slow iteration cannot move it.

Phases are reported separately because they are different regimes: prefill attention is
quadratic in sequence length and its GEMMs run over every prompt token, while decode attention
is linear in KV history and its GEMMs run over one token per request. A single blended number
would hide a systematic error in either.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..rollout.base import RolloutCostModel, RolloutWorkload

DEFAULT_CONTEXT_BUCKET = 256


@dataclass(frozen=True)
class BaselineBucket:
    """One group of near-identical measured iterations."""

    phase: str
    request_count: int
    n_tokens: float
    sum_l2: float
    ctxsum: float
    latency_s: float
    sample_count: int

    def workload(self) -> RolloutWorkload:
        """Return the workload a cost model should be asked to predict."""
        return RolloutWorkload(
            phase=self.phase,
            n_tokens=self.n_tokens,
            sum_l2=self.sum_l2,
            ctxsum=self.ctxsum,
            request_count=self.request_count,
        )


@dataclass(frozen=True)
class ValidationResult:
    """Error of one model against one phase's buckets."""

    phase: str
    bucket_count: int
    mre_percent: float
    p90_percent: float
    max_percent: float
    median_ratio: float
    worst_bucket: tuple[int, float]

    def passes(self, gate: float) -> bool:
        """Return whether the mean relative error is below `gate` percent."""
        return self.bucket_count > 0 and self.mre_percent < gate


def load_baseline(
    path: str | Path,
    *,
    min_samples: int = 1,
    context_bucket: int = DEFAULT_CONTEXT_BUCKET,
) -> tuple[BaselineBucket, ...]:
    """
    Group a vLLM JSONL baseline into buckets of near-identical iterations.

    Args:
        path (str | Path): JSONL file whose records carry a `payload` object with `phase`,
            `request_count`, `n_tokens`, `sum_l2`, `ctxsum`, and `latency_s`.
        min_samples (int): Drop buckets holding fewer iterations than this.
        context_bucket (int): Round each bucket's mean context to this multiple. Linear, never
            a power of two.

    Returns:
        tuple[BaselineBucket, ...]: Sorted by `(phase, request_count, ctxsum)`.
    """
    groups: dict[tuple[str, int, int, int], list[dict]] = defaultdict(list)
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)["payload"]
            request_count = int(payload["request_count"])
            latency = float(payload["latency_s"])
            if request_count <= 0 or latency <= 0.0:
                continue
            n_tokens = float(payload["n_tokens"])
            sum_l2 = float(payload["sum_l2"])
            ctxsum = float(payload["ctxsum"])
            phase = RolloutWorkload.from_step(n_tokens, sum_l2, ctxsum, request_count).phase
            mean_context = round(ctxsum / request_count / context_bucket) * context_bucket
            token_key = round(n_tokens / context_bucket) * context_bucket
            groups[(phase, request_count, int(mean_context), int(token_key))].append(payload)

    buckets = []
    for (phase, request_count, _, _), payloads in groups.items():
        if len(payloads) < min_samples:
            continue
        buckets.append(
            BaselineBucket(
                phase=phase,
                request_count=request_count,
                n_tokens=statistics.median(float(p["n_tokens"]) for p in payloads),
                sum_l2=statistics.median(float(p["sum_l2"]) for p in payloads),
                ctxsum=statistics.median(float(p["ctxsum"]) for p in payloads),
                latency_s=statistics.median(float(p["latency_s"]) for p in payloads),
                sample_count=len(payloads),
            )
        )
    return tuple(sorted(buckets, key=lambda b: (b.phase, b.request_count, b.ctxsum)))


def score_model(
    model: RolloutCostModel, buckets: tuple[BaselineBucket, ...]
) -> tuple[ValidationResult, ...]:
    """
    Score `model` against `buckets`, one result per phase present.

    Args:
        model (RolloutCostModel): Any model that can predict a `RolloutWorkload`.
        buckets (tuple[BaselineBucket, ...]): Measured groups from `load_baseline`.

    Returns:
        tuple[ValidationResult, ...]: Sorted by phase, empty when `buckets` is empty.
    """
    by_phase: dict[str, list[tuple[BaselineBucket, float]]] = defaultdict(list)
    for bucket in buckets:
        by_phase[bucket.phase].append((bucket, model.predict(bucket.workload())))

    results = []
    for phase, scored in sorted(by_phase.items()):
        errors = [
            abs(predicted - bucket.latency_s) / bucket.latency_s * 100.0
            for bucket, predicted in scored
        ]
        ratios = sorted(predicted / bucket.latency_s for bucket, predicted in scored)
        worst_index = max(range(len(errors)), key=errors.__getitem__)
        worst = scored[worst_index][0]
        results.append(
            ValidationResult(
                phase=phase,
                bucket_count=len(scored),
                mre_percent=statistics.fmean(errors),
                p90_percent=sorted(errors)[min(int(0.9 * len(errors)), len(errors) - 1)],
                max_percent=max(errors),
                median_ratio=statistics.median(ratios),
                worst_bucket=(worst.request_count, worst.ctxsum),
            )
        )
    return tuple(results)
