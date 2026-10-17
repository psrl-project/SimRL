"""Total-length predictors a scheduler can be run against.

`OracleLength` reads the trace's `target_len` and bounds what any predictor could buy.
`NoisyOracleLength` perturbs it by a deterministic per-request lognormal error to ask how
much prediction error a policy survives.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..component.rollout.engine import TrajView

_MASK = (1 << 64) - 1


class LengthPredictor(Protocol):
    """Predicts the total length a request will reach."""

    name: str

    def predict_total(self, traj: TrajView) -> float:
        """Return the predicted total number of generated tokens."""
        ...

    def remaining(self, traj: TrajView) -> float:
        """Return the predicted tokens still to generate, floored at 1."""
        ...


class _LengthPredictor:
    name: str

    def predict_total(self, traj: TrajView) -> float:
        raise NotImplementedError

    def remaining(self, traj: TrajView) -> float:
        return max(1.0, self.predict_total(traj) - traj.generated)


class OracleLength(_LengthPredictor):
    """The exact length from the trace."""

    name = "oracle"

    def predict_total(self, traj: TrajView) -> float:
        return float(traj.target_len)


def _mix(x: int) -> int:
    """splitmix64, a stateless hash so a request's noise never changes between queries."""
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
    """The exact length times a per-request lognormal error of log-std `sigma`."""

    name = "noisy_oracle"

    def __init__(self, sigma: float = 0.25, seed: int = 0) -> None:
        """
        Args:
            sigma (float): Log-space standard deviation of the multiplicative error.
            seed (int): Selects the noise realisation.
        """
        if sigma < 0.0:
            raise ValueError(f"sigma must be >= 0, got {sigma!r}.")
        self.sigma = float(sigma)
        self.seed = int(seed)

    def predict_total(self, traj: TrajView) -> float:
        if self.sigma == 0.0:
            return float(traj.target_len)
        key = _mix(self.seed) ^ _mix(traj.prompt_key) ^ _mix(traj.target_len + (traj.prompt_len << 20))
        return float(traj.target_len) * math.exp(self.sigma * _gauss(key))


LENGTH_PREDICTORS: dict[str, type[_LengthPredictor]] = {
    OracleLength.name: OracleLength,
    NoisyOracleLength.name: NoisyOracleLength,
}


def build_length_predictor(name: str, **args) -> LengthPredictor:
    """Construct the registered predictor called `name`.

    Raises:
        KeyError: If no predictor is registered under that name.
    """
    if name not in LENGTH_PREDICTORS:
        raise KeyError(f"Unknown length predictor {name!r}. Available: {sorted(LENGTH_PREDICTORS)}.")
    return LENGTH_PREDICTORS[name](**args)
