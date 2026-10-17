"""The length model that cheats, which is what makes it useful.

`OracleLength` reads `target_len` directly. It is not deployable and is not meant to
be. It bounds what any predictor could buy: run it against an observable model and
the gap between them is the predictor's fault, while a shared loss against the
baseline is the mechanism's fault. Without it a null result cannot be attributed.
"""

from __future__ import annotations

import math

from .length import TrajView


class OracleLength:
    """Answers from the trace's own target length, softened by a tolerance band.

    A true step at `remaining` makes `V` correct and `G` useless. The central
    difference in `marginal_value` would then be nonzero only for a budget within
    one epsilon of the exact remaining work, so nearly every request prices at
    exactly zero and a comparison against any positive cost refuses it.

    `band` replaces the step with a ramp of that fractional width, so `G` falls
    off smoothly either side of the boundary while `V` keeps the shape the oracle
    is for. It is a readout smoother, not a claim about the distribution: the
    model still knows the exact answer and `V` still crosses 0.5 at `remaining`.
    """

    name = "oracle"

    def __init__(self, band: float = 0.25) -> None:
        """
        Args:
            band (float): Half-width of the ramp, as a fraction of the remaining
                length. Zero restores the bare step, which is degenerate for
                ranking but is what an exactness test wants.
        """
        if band < 0.0:
            raise ValueError(f"band must be >= 0, got {band!r}.")
        self.band = float(band)

    @property
    def ready(self) -> bool:
        """Always, since the trace is complete before the run starts."""
        return True

    def hit_prob(self, traj: TrajView, budget: float) -> float:
        remaining = traj.remaining
        width = self.band * max(1.0, float(remaining))
        if width <= 0.0:
            return 1.0 if remaining <= budget else 0.0
        if budget <= remaining - width:
            return 0.0
        if budget >= remaining + width:
            return 1.0
        return (budget - (remaining - width)) / (2.0 * width)

    def expected_final_length(self, traj: TrajView) -> float:
        """The exact answer, which is the point of this model."""
        return float(traj.target_len)

    def observe(self, traj: TrajView, final_length: int) -> None:
        """Nothing to learn."""

    def decay(self, factor: float) -> None:
        """Nothing to discount."""


_MASK = (1 << 64) - 1


def _mix(x: int) -> int:
    """splitmix64: a stateless integer hash, so a request's noise never changes between queries."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK
    return x ^ (x >> 31)


def _gauss(key: int) -> float:
    """A standard normal drawn deterministically from `key` by Box-Muller."""
    u1 = (_mix(key) + 1) / (_MASK + 2)
    u2 = (_mix(key ^ 0xD1B54A32D192ED03) + 1) / (_MASK + 2)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


class NoisyOracleLength(OracleLength):
    """The exact length times lognormal noise, for asking how good a predictor must be.

    Every number a scheduler gets from a predictor is a noisy version of the true
    length, so the question a design needs answered is how much noise the gain
    survives. This holds the model exact except for a per-request multiplicative
    error of log-standard-deviation `sigma`, drawn from a hash of the request's
    identity rather than a stream, so the same request is mispredicted the same way
    every time it is asked about, as a deployed model would.

    `sigma=0` reproduces `OracleLength`. At `sigma=0.5` the prediction is typically
    off by a factor of 1.65 in either direction.
    """

    name = "noisy_oracle"

    def __init__(self, sigma: float = 0.25, seed: int = 0, band: float = 0.25) -> None:
        """
        Args:
            sigma (float): Log-space standard deviation of the multiplicative error.
            seed (int): Selects the noise realisation.
            band (float): As for `OracleLength`.
        """
        super().__init__(band=band)
        if sigma < 0.0:
            raise ValueError(f"sigma must be >= 0, got {sigma!r}.")
        self.sigma = float(sigma)
        self.seed = int(seed)

    def expected_final_length(self, traj: TrajView) -> float:
        if self.sigma == 0.0:
            return float(traj.target_len)
        key = _mix(self.seed) ^ _mix(traj.prompt_key) ^ _mix(traj.target_len + (traj.prompt_len << 20))
        return float(traj.target_len) * math.exp(self.sigma * _gauss(key))
