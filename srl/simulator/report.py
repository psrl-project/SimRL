"""Staleness distribution report: token-weighted percentiles and wasted tokens."""

from __future__ import annotations

from dataclasses import dataclass

from ..component.rollout.engine import Request


@dataclass(frozen=True)
class StalenessReport:
    n_requests: int
    n_tokens: int
    mean: float
    p50: float
    p95: float
    p99: float
    max: int
    mean_reported: float
    max_reported: int
    wasted_tokens: dict[str, int]
    """Drop reason to generated tokens, summed over every dropped request."""
    configured_horizon: int | None
    strict: bool | None
    """Whether every consumed request's true staleness is within configured_horizon.

    None when configured_horizon is None, since there is no bound to check against.
    """


def _weighted_percentile(pairs: list[tuple[int, int]], pct: float) -> float:
    """Nearest-rank percentile over (value, token weight) pairs."""
    total = sum(weight for _, weight in pairs)
    if total == 0:
        return 0.0
    target = pct / 100.0 * total
    cumulative = 0
    for value, weight in sorted(pairs):
        cumulative += weight
        if cumulative >= target:
            return float(value)
    return float(pairs[-1][0])


def summarize(
    reqs: list[Request],
    dropped: list[tuple[Request, str]],
    horizon: int | None,
) -> StalenessReport:
    """Token-weighted staleness distribution over `reqs`, plus wasted tokens from `dropped`."""
    n_tokens = sum(r.generated for r in reqs)
    true_pairs = [(r.staleness_true(), r.generated) for r in reqs]
    reported_pairs = [(r.staleness_reported(), r.generated) for r in reqs]

    wasted_tokens: dict[str, int] = {}
    for req, reason in dropped:
        wasted_tokens[reason] = wasted_tokens.get(reason, 0) + req.generated

    return StalenessReport(
        n_requests=len(reqs),
        n_tokens=n_tokens,
        mean=sum(v * w for v, w in true_pairs) / n_tokens if n_tokens else 0.0,
        p50=_weighted_percentile(true_pairs, 50),
        p95=_weighted_percentile(true_pairs, 95),
        p99=_weighted_percentile(true_pairs, 99),
        max=max((r.staleness_true() for r in reqs), default=0),
        mean_reported=sum(v * w for v, w in reported_pairs) / n_tokens if n_tokens else 0.0,
        max_reported=max((r.staleness_reported() for r in reqs), default=0),
        wasted_tokens=wasted_tokens,
        configured_horizon=horizon,
        strict=None if horizon is None else all(v <= horizon for v, _ in true_pairs),
    )
