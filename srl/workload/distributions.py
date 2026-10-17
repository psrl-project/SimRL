"""Workload generation: response and prompt length distributions."""

from __future__ import annotations

import random


def bimodal(n: int, short_len: int, long_len: int, long_frac: float, seed: int = 0) -> list[int]:
    """Mostly short requests with a fixed fraction of very long ones."""
    rng = random.Random(seed)
    n_long = int(round(n * long_frac))
    out = [long_len] * n_long + [short_len] * (n - n_long)
    rng.shuffle(out)
    return out


def lognormal(n: int, mu: float, sigma: float, lo: int = 1, hi: int | None = None, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        v = int(rng.lognormvariate(mu, sigma))
        v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        out.append(v)
    return out


def normal(n: int, mu: float, sigma: float, lo: int = 1, hi: int | None = None, seed: int = 0) -> list[int]:
    """Round samples from `N(mu, sigma)`. Draws below `lo` are raised to `lo`."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        v = int(round(rng.gauss(mu, sigma)))
        v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        out.append(v)
    return out


def powerlaw(n: int, alpha: float, lo: int, hi: int, seed: int = 0) -> list[int]:
    """Pareto-ish lengths truncated to [lo, hi]."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        u = rng.random()
        v = int(lo * (1 - u) ** (-1.0 / alpha))
        out.append(min(hi, max(lo, v)))
    return out


def uniform(n: int, lo: int, hi: int, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(lo, hi) for _ in range(n)]


def fixed(n: int, value: int, seed: int = 0) -> list[int]:
    """Constant length for every request."""
    return [value] * n
