"""Which staleness level produced each part of a run's generation throughput.

Reduces the `request` stream into one stacked bar's worth of numbers: tokens
consumed at each observed `staleness_true`, tokens generated and then discarded
keyed by the reason, and the span they were produced in. Every segment over the
span sums to the fleet's full generation throughput, so tightening a staleness
bound shows up as tokens moving from a consumed level into a discard bucket
rather than disappearing from the accounting.

`span` is the later of `max(t)` over the request stream and the last training
step's `end`, matching what `SimResult.sim_time` reaches once the clock is
force-aligned to a training deadline after rollout has drained, so a rate
computed here is comparable with `sim_time`-based rates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..telemetry.store import JsonlStore

# A discard row whose path supplies no reason still carries tokens that were
# really spent.
_UNKNOWN_REASON = "unknown"


@dataclass(frozen=True)
class StalenessBreakdown:
    """One finished run, reduced to its staleness-by-throughput composition.

    Args:
        span (float): Latest wall-clock instant the run reached, combining the
            request and train streams. The denominator of every rate below.
        n_consumed (int): Trajectories that reached a training batch.
        consumed_tokens (dict[int, int]): `staleness_true` to generated tokens.
        wasted_tokens (dict[str, int]): `drop_reason` to generated tokens that
            never reached a training batch.
        reprefill_tokens (int): Tokens re-computed after a preemption or abort,
            over the consumed trajectories. Reported rather than stacked,
            because this is recomputation of tokens already counted once in
            `consumed_tokens` and adding it to the stack would double count.
    """

    span: float
    n_consumed: int
    consumed_tokens: dict[int, int]
    wasted_tokens: dict[str, int]
    reprefill_tokens: int

    @property
    def max_staleness(self) -> int:
        """Highest staleness any consumed trajectory reached."""
        return max(self.consumed_tokens) if self.consumed_tokens else 0

    @property
    def total_consumed_tokens(self) -> int:
        return sum(self.consumed_tokens.values())

    @property
    def total_wasted_tokens(self) -> int:
        return sum(self.wasted_tokens.values())

    @property
    def throughput(self) -> float:
        """Full generation throughput in tokens per second, trained on or not."""
        return (self.total_consumed_tokens + self.total_wasted_tokens) / self.span

    def scg(self, s: int) -> float:
        """Staleness-constrained goodput: consumed tokens at staleness <= `s`, per second.

        Defined for every `s`. Saturates at the consumed rate once nothing is
        excluded, so an `s` above this run's own maximum equals the value below it.
        """
        kept = sum(n for staleness, n in self.consumed_tokens.items() if staleness <= s)
        return kept / self.span


def staleness_breakdown(store: JsonlStore) -> StalenessBreakdown:
    """Reduce one run's `request` stream.

    Args:
        store (JsonlStore): A finished run's log store.

    Returns:
        StalenessBreakdown: The run's staleness composition.

    Raises:
        ValueError: If the stream holds no consumed trajectory with a recorded
            staleness, or if its span is not positive. Both make every rate
            meaningless, so they are refused rather than reported as zero.
    """
    span = 0.0
    consumed: dict[int, int] = {}
    wasted: dict[str, int] = {}
    n_consumed = 0
    reprefill = 0

    for row in store.iter("request"):
        span = max(span, float(row.get("t", 0.0) or 0.0))
        event = row.get("event")
        if event == "consume":
            staleness = row.get("staleness_true")
            if staleness is None:
                continue
            n_consumed += 1
            level = int(staleness)
            consumed[level] = consumed.get(level, 0) + int(row.get("generated", 0) or 0)
            reprefill += int(row.get("reprefill_tokens", 0) or 0)
        elif event == "drop":
            reason = row.get("drop_reason") or _UNKNOWN_REASON
            wasted[reason] = wasted.get(reason, 0) + int(row.get("generated", 0) or 0)

    # The last training step force-advances the clock to its own end even when
    # rollout has already drained and writes no further request row, so the
    # request stream alone can understate the run's true span.
    for row in store.iter("train"):
        span = max(span, float(row.get("end", 0.0) or 0.0))

    if not n_consumed:
        raise ValueError(f"Run at {store.root!s} has no consumed trajectories with a recorded staleness.")
    if span <= 0.0:
        raise ValueError(f"Run at {store.root!s} has a non-positive span of {span!r}.")

    return StalenessBreakdown(
        span=span,
        n_consumed=n_consumed,
        consumed_tokens=consumed,
        wasted_tokens=wasted,
        reprefill_tokens=reprefill,
    )


def check_same_consumption(points: Sequence[tuple[str, StalenessBreakdown]]) -> None:
    """Refuse a comparison whose points did not finish the same training work.

    Throughput is a rate, so a point that ended early scores arbitrarily well.
    Consumed *tokens* may still differ between points, because a tighter bound
    resamples a different set of trajectories. That difference is real and is
    reported on the figure, so only the trajectory count is gated here.

    Args:
        points (Sequence[tuple[str, StalenessBreakdown]]): Labelled points.

    Raises:
        ValueError: If the consumed trajectory counts disagree.
    """
    counts = {bd.n_consumed for _, bd in points}
    if len(counts) > 1:
        detail = ", ".join(f"{label}: {bd.n_consumed}" for label, bd in points)
        raise ValueError(f"Points consumed different trajectory counts: {detail}.")
