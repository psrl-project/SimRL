"""
Numeric integration of a decode segment whose per-step cost has no closed form.

`advance.py` sums a segment analytically, which requires step cost to be exactly
`max(F, alpha + beta*k)`. A cost composed from measured kernel latencies is not that shape, so
it needs a numeric integrator, and the obvious one is wrong in a way that matters.

Sampling the cost curve at a stride and trapezoid-summing *per query* is not monotone in `k`:
on a spiked cost curve it produces 429 values of `k` in [1, 4096] where `total(k) <
total(k - 1)`. A non-monotone total has no unique inverse, and `instance.py` compares the
returned step count against its budget by integer identity, so an inexact inverse silently
changes which branch the simulator takes.

The construction here is monotone by design. Lattice nodes at `0, stride, 2*stride, ...` are
sampled once at construction. Node `i` carries a fixed rate: the trapezoid average of the
per-step cost at nodes `i` and `i+1`, and a cumulative total. A query interpolates linearly
inside one node interval at that fixed rate:

    total(k) = cumulative[i] + rates[i] * (k - node[i]),   i = k // stride

Because `rates[i]` does not depend on `k`, the total grows at a fixed positive slope inside
every interval and is continuous across them, so it is strictly increasing and bisection
inverts it exactly. Measured against a step-by-step reference over 4096 steps on the real
Qwen3-8B tables, the error depends strongly on how deep the batch's KV already is, because
that sets how curved the cost is across a lattice interval:

    starting context   batch 16   batch 64
    0 (cold segment)      0.46%      0.42%
    2048                  0.07%      0.01%
    16384 and beyond    <=0.01%    <=0.01%

Over 2048 steps the cold case is the worst at 1.31% (batch 16) and 1.11% (batch 64); a cold
segment is the common case right after prefill, so treat ~1.3% as the bound rather than the
sub-0.1% figures a warm segment shows. A 4096-step segment costs 6 cost evaluations instead of
4096, against 7.08% and 13.68% for a single tangent.
"""

from __future__ import annotations

from collections.abc import Callable

DEFAULT_STRIDE = 1024


class LatticeIntegrator:
    """Integrate a decode segment on a fixed lattice, monotonically and invertibly."""

    def __init__(
        self,
        step_cost: Callable[[int], float],
        *,
        k_max: int,
        stride: int = DEFAULT_STRIDE,
    ) -> None:
        """
        Prepare to integrate; no cost evaluation happens here.

        Args:
            step_cost (Callable[[int], float]): Seconds of the step `offset` steps into the
                segment. Called at most once per lattice node, never per query. It must
                answer for any `offset` that is a multiple of `stride`, including the node
                just past the segment end: integrating `k` steps needs the node bracketing
                them from above, so a 300-step segment at stride 1024 evaluates offsets 0
                and 1024. It must therefore extrapolate rather than raise past `k_max`.
            k_max (int): Hard ceiling on segment length. `total` clamps there.
            stride (int): Steps between lattice nodes. At `stride=1` each interval is one
                step and the sum is exact.

        Raises:
            ValueError: If `stride` is not positive or `k_max` is negative.
        """
        if stride <= 0:
            raise ValueError(f"Segment stride must be positive, got {stride!r}.")
        if k_max < 0:
            raise ValueError(f"Segment k_max must not be negative, got {k_max!r}.")
        self._step_cost = step_cost
        self._stride = stride
        self._k_max = k_max
        self._costs: list[float] = []
        self._rates: list[float] = []
        self._cumulative: list[float] = [0.0]

    @property
    def sample_count(self) -> int:
        """Return how many times `step_cost` has been evaluated so far."""
        return len(self._costs)

    def total(self, k: int) -> float:
        """
        Return the seconds of the first `k` steps of this segment.

        Args:
            k (int): Step count. Zero or negative returns exactly `0.0`; above `k_max` clamps.

        Returns:
            float: Strictly increasing in `k` up to `k_max`.
        """
        if k <= 0:
            return 0.0
        k = min(k, self._k_max)
        index = k // self._stride
        # `_extend_to` guarantees `len(self._rates) > index` on return, so the rate and the
        # cumulative total below are always present.
        self._extend_to(index)
        return self._cumulative[index] + self._rates[index] * (k - index * self._stride)

    def steps_within(self, budget: float) -> int:
        """
        Return the largest step count whose total fits inside `budget`.

        Args:
            budget (float): Seconds available. Zero or negative returns 0.

        Returns:
            int: In `[0, k_max]`. Exactly inverts `total` because `total` is strictly
                increasing.
        """
        if budget <= 0.0:
            return 0
        # Walk lattice intervals until the cumulative total passes the budget, growing the
        # lattice only as far as the budget actually reaches.
        index = 0
        while True:
            self._extend_to(index)
            node_end = (index + 1) * self._stride
            if self._cumulative[index + 1] > budget or node_end >= self._k_max:
                break
            index += 1
        low = index * self._stride
        high = min((index + 1) * self._stride, self._k_max)
        if self.total(high) <= budget:
            return high
        while low < high:
            middle = (low + high + 1) // 2
            if self.total(middle) <= budget:
                low = middle
            else:
                high = middle - 1
        return low

    def _extend_to(self, index: int) -> None:
        """Sample lattice nodes until interval `index` has a rate and a cumulative total."""
        while len(self._costs) <= index + 1:
            offset = len(self._costs) * self._stride
            cost = self._step_cost(offset)
            if not cost > 0.0:
                raise ValueError(
                    f"Segment step cost must be positive, got {cost!r} at step offset {offset}."
                )
            self._costs.append(cost)
        while len(self._rates) <= index:
            i = len(self._rates)
            # At stride 1 each interval is one step, so the left endpoint IS that step's cost
            # and the sum is exact. Above stride 1 the trapezoid average is what earns the
            # accuracy: over 4096 real decode steps from a 2048-token context it scores 0.07%
            # (batch 16) and 0.01% (batch 64) against a step-by-step reference, where the left
            # endpoint scores 6.29% and 8.47% because it ignores growth across the interval.
            if self._stride == 1:
                rate = self._costs[i]
            else:
                rate = 0.5 * (self._costs[i] + self._costs[i + 1])
            self._rates.append(rate)
            self._cumulative.append(self._cumulative[-1] + rate * self._stride)
