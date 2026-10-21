"""Turn raw simulator results into the per-arm record the gates expect.

The in-process harness returns a `SimResult`. This module measures it the same way the existing
sweeps do, so a point run here is comparable with one run through `headroom_probe`.
"""

from __future__ import annotations

import statistics
from typing import Any, Sequence


def concurrency(sim, n_instances: int) -> tuple[float, int]:
    """
    Return mean and peak concurrent requests per instance.

    Args:
        sim: A finished `Simulator`.
        n_instances (int): Instances to sweep over.

    Returns:
        tuple[float, int]: Mean of the per-instance averages, and the fleet peak.
    """
    spans: dict[int, list[tuple[float, float]]] = {}
    for req in sim.consumed:
        if req.complete_time is None:
            continue
        spans.setdefault(req.instance_id, []).append((req.admit_time, req.complete_time))

    means: list[float] = []
    peak = 0
    for instance_id in range(n_instances):
        events: list[tuple[float, int]] = []
        for start, end in spans.get(instance_id, ()):
            events.append((start, 1))
            events.append((end, -1))
        if not events:
            means.append(0.0)
            continue
        events.sort()
        current = area = 0
        previous = events[0][0]
        for time, delta in events:
            area += current * (time - previous)
            current += delta
            peak = max(peak, current)
            previous = time
        span = events[-1][0] - events[0][0]
        means.append(area / span if span else 0.0)
    return statistics.mean(means), peak


def measure(sim, result, n_instances: int) -> dict[str, Any]:
    """
    Measure one finished run into a gate-ready record.

    Args:
        sim: A finished `Simulator`.
        result (SimResult): The result `sim.run()` returned.
        n_instances (int): Instances the run used.

    Returns:
        dict: Span, consumed counts, tokens per group, concurrency, preemptions, and outcome flags.
    """
    tokens = sum(req.generated for req in sim.consumed)
    groups = len({req.group_id for req in sim.consumed})
    mean_conc, peak_conc = concurrency(sim, n_instances)
    return {
        "span": result.sim_time,
        "consumed_tokens": tokens,
        "consumed_groups": groups,
        "tokens_per_group": tokens / groups if groups else float("nan"),
        "mean_concurrency": mean_conc,
        "peak_concurrency": peak_conc,
        "preemptions": sum(req.num_preemptions for req in sim.consumed),
        "versions_done": result.versions_done,
        "livelocked": result.livelocked,
        "dropped": len(result.dropped),
    }


def compare_rows(rows: Sequence[dict[str, Any]], control: str) -> list[dict[str, Any]]:
    """Attach each row's speedup against `control` at the same point."""
    by_point: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["arm"] == control:
            by_point[row["point"]] = row
    out = []
    for row in rows:
        base = by_point.get(row["point"])
        out.append({**row, "speedup": base["span"] / row["span"] if base else float("nan")})
    return out
