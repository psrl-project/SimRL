"""Predicting trajectories, the counterpart to `cost/` predicting time.

`cost/` answers "how long will this step take". This package answers "how many
trajectories will be consumed in time", and prices scheduling decisions in that unit.

The only modelled quantity is remaining length (`LengthModel`). Everything else is
arithmetic: a deadline in training cycles, a token budget, and a marginal value
obtained by differencing the length model twice.
"""

from .deadline import TrainCycle, budget_tokens
from .empirical import EmpiricalLength
from .length import LengthModel, TrajView, traj_view
from .marginal import BatchState, Valuator, decode_rates
from .oracle import NoisyOracleLength, OracleLength
from .tool import NoToolCalls, ToolTimeModel

LENGTH_MODEL_REGISTRY: dict[str, type] = {
    "oracle": OracleLength,
    "noisy_oracle": NoisyOracleLength,
    "empirical": EmpiricalLength,
}
"""Length model name -> constructor, the set a router strategy may name."""


def build_length_model(name: str, **kwargs) -> LengthModel:
    """Construct the registered length model called `name`.

    Raises:
        KeyError: If no model is registered under that name, listing the ones
            that are.
    """
    if name not in LENGTH_MODEL_REGISTRY:
        raise KeyError(
            f"Unknown length model {name!r}. Available: {sorted(LENGTH_MODEL_REGISTRY)}."
        )
    return LENGTH_MODEL_REGISTRY[name](**kwargs)


__all__ = [
    "LENGTH_MODEL_REGISTRY",
    "BatchState",
    "EmpiricalLength",
    "LengthModel",
    "NoisyOracleLength",
    "NoToolCalls",
    "OracleLength",
    "ToolTimeModel",
    "TrainCycle",
    "TrajView",
    "Valuator",
    "budget_tokens",
    "build_length_model",
    "decode_rates",
    "traj_view",
]
