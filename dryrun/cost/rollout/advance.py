"""Closed-form advancement of a decode segment.

A decode segment is a run of steps over a fixed batch, with no prefill mixed in
and no request entering or leaving. It is unrelated to `Segment` in `request`,
which tracks weight versions.

Within a decode segment of n_d requests each gaining one token per step,
the k-th step costs

    tau(k) = max(F, alpha + beta * k),   k = 0, 1, 2, ...

The cumulative sum splits at the crossing step k0 = ceil((F - alpha) / beta):

    Delta(k) = k0 * F + (k - k0) * (alpha + beta * k0)
               + beta * (k - k0) * (k - k0 - 1) / 2

All functions are exact under Fraction inputs.
"""

from __future__ import annotations

import math
from fractions import Fraction


def crossing_step(F, alpha, beta, k_cap: int | None = None) -> int:
    """First step index at which the linear branch overtakes the floor F."""
    if beta <= 0:
        k0 = 0 if alpha >= F else (k_cap if k_cap is not None else 0)
    elif alpha >= F:
        k0 = 0
    else:
        need = F - alpha
        if isinstance(need, Fraction) or isinstance(beta, Fraction):
            q = Fraction(need) / Fraction(beta)
            k0 = -((-q.numerator) // q.denominator)
        else:
            k0 = math.ceil(need / beta)
        k0 = max(0, k0)
    if k_cap is not None:
        k0 = min(k0, k_cap)
    return k0


def elapsed(F, alpha, beta, k: int):
    """Total time for k steps of tau(i) = max(F, alpha + beta*i)."""
    if k <= 0:
        return 0 * F
    k0 = crossing_step(F, alpha, beta, k_cap=k)
    total = k0 * F
    n = k - k0
    if n > 0:
        tri = (n * (n - 1)) // 2
        total = total + n * (alpha + beta * k0) + beta * tri
    return total


def steps_within(F, alpha, beta, budget) -> int:
    """Largest k with elapsed(k) <= budget."""
    if budget <= 0:
        return 0
    if beta <= 0:
        tau = max(F, alpha)
        if tau <= 0:
            return 0
        k = int(budget // tau)
        # Float division can land one step short when budget is an exact multiple of tau.
        # Step forward while elapsed fits, then step back if it overshoots.
        while elapsed(F, alpha, beta, k + 1) <= budget:
            k += 1
        while k > 0 and elapsed(F, alpha, beta, k) > budget:
            k -= 1
        return k

    k0 = crossing_step(F, alpha, beta)
    t_floor = k0 * F
    if budget <= t_floor:
        if F <= 0:
            return k0
        # `budget // F` can land one step short of an exact quotient (10.0 // 0.005 is
        # 1999.0), so correct it the same way the branches below do.
        k = int(budget // F)
        while elapsed(F, alpha, beta, k + 1) <= budget:
            k += 1
        while k > 0 and elapsed(F, alpha, beta, k) > budget:
            k -= 1
        return k

    rest = budget - t_floor
    a = float(beta) / 2.0
    b_coeff = float(alpha) + float(beta) * k0 - float(beta) / 2.0
    c = -float(rest)
    disc = b_coeff * b_coeff - 4 * a * c
    n = 0 if disc < 0 else int((-b_coeff + math.sqrt(disc)) / (2 * a))
    n = max(0, n)
    while elapsed(F, alpha, beta, k0 + n + 1) <= budget:
        n += 1
    while n > 0 and elapsed(F, alpha, beta, k0 + n) > budget:
        n -= 1
    return k0 + n


def next_completion(remaining: list[int]) -> tuple[int, int]:
    """Steps until the earliest completion, and how many finish simultaneously."""
    if not remaining:
        return 0, 0
    k = min(remaining)
    return k, sum(1 for r in remaining if r == k)
