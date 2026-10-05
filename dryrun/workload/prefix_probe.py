"""Read vLLM's prefix-cache counters, for validating the simulated block pool.

`vllm:prefix_cache_queries` and `vllm:prefix_cache_hits`
(`third_party/vllm/vllm/v1/metrics/loggers.py`) count **tokens** in the vLLM
version vendored here: `PrefixCacheStats.record(num_tokens, num_hits, ...)` in
`vllm/v1/metrics/stats.py` is called per request with token counts, and the
metric documentation says "in terms of number of queried tokens". Older releases
counted blocks, so do not assume the unit from the name alone.

The simulator's `KvCacheManager` counts blocks, because that is the unit its
pool allocates in. So the two sides are compared as **hit rates**, which are
unit-free, rather than as raw counts. Every expectation in
`examples/profile/vllm/prefix_cache_probe.yaml` is therefore a rate.

Both counters are cumulative since server start, so a scenario reads them
before and after and takes the difference.
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass

_SAMPLE_LINE = re.compile(
    r"^vllm:(?P<name>prefix_cache_queries|prefix_cache_hits)"
    # Prometheus appends `_total` to a counter's exposed sample name, so the
    # registered name and the scraped name differ. `_created` is the companion
    # timestamp sample and must not be summed into the count.
    r"(?:_total)?"
    r"\{[^}]*\}\s+(?P<value>[0-9.eE+-]+)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class PrefixCacheCounters:
    """Blocks queried and blocks found, summed across engines.

    Args:
        queries: Blocks looked up.
        hits: Blocks found.

    Raises:
        ValueError: If either is negative, or if hits exceed queries, which
            would mean the reading is not self-consistent.
    """

    queries: int
    hits: int

    def __post_init__(self) -> None:
        if self.queries < 0 or self.hits < 0:
            raise ValueError(f"Counters must be non-negative, got {self!r}.")
        if self.hits > self.queries:
            raise ValueError(
                f"More hits than queries ({self.hits} > {self.queries}), so this "
                f"reading is not self-consistent."
            )

    @property
    def hit_rate(self) -> float:
        """Fraction of queried blocks that were found, 0.0 when nothing was asked."""
        if self.queries == 0:
            return 0.0
        return self.hits / self.queries

    def __sub__(self, other: "PrefixCacheCounters") -> "PrefixCacheCounters":
        """The delta between two cumulative readings."""
        return PrefixCacheCounters(
            queries=self.queries - other.queries,
            hits=self.hits - other.hits,
        )


def parse_prefix_cache_counters(metrics_text: str) -> PrefixCacheCounters:
    """Extract the GPU prefix-cache counters from a Prometheus exposition.

    Args:
        metrics_text: The body of a `GET /metrics` response.

    Returns:
        The two counters, summed over every engine label.

    Raises:
        ValueError: If `vllm:prefix_cache_queries` is absent. That means the
            server never had prefix caching enabled, and reporting 0.0 would
            read as "measured no hits" rather than "measured nothing".
    """
    totals = {"prefix_cache_queries": 0, "prefix_cache_hits": 0}
    seen = set()
    for match in _SAMPLE_LINE.finditer(metrics_text):
        name = match.group("name")
        totals[name] += int(float(match.group("value")))
        seen.add(name)

    if "prefix_cache_queries" not in seen:
        raise ValueError(
            "No vllm:prefix_cache_queries sample in /metrics, so this server was "
            "not running with prefix caching enabled."
        )
    return PrefixCacheCounters(
        queries=totals["prefix_cache_queries"],
        hits=totals["prefix_cache_hits"],
    )


def read_prefix_cache_counters(
    base_url: str, timeout_s: float = 10.0
) -> PrefixCacheCounters:
    """Fetch and parse `{base_url}/metrics`.

    Args:
        base_url: Server root, e.g. `http://127.0.0.1:8000`.
        timeout_s: Socket timeout.

    Returns:
        The counters at the moment of the call.
    """
    url = f"{base_url.rstrip('/')}/metrics"
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return parse_prefix_cache_counters(response.read().decode("utf-8"))
