"""What a routing strategy is, and how one is looked up by name.

A strategy selects among instances that passed the admission gate. It may also
narrow candidates for one version queue and decide whether dispatch may bypass
that queue when its head cannot be placed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from ..engine import InstanceView, Request


class RoutingStrategy(Protocol):
    """Picks one instance for one request."""

    name: str

    def select(self, req: Request, candidates: list[InstanceView]) -> int | None:
        """Choose among `candidates`.

        Args:
            req: The request to place.
            candidates: Instances that passed the admission gate, in instance
                id order. Never empty: the caller skips the strategy entirely
                when nothing is eligible.

        Returns:
            An index into `candidates`, or None to leave the request queued.
        """
        ...


@dataclass(frozen=True)
class RoutingContext:
    """Queue state visible to a strategy during one placement attempt."""

    blocked_versions: tuple[int, ...]
    fleet_min_tp: int
    fleet_max_tp: int
    now: float = 0.0
    """Simulation time of this attempt, for a strategy that prices a deadline."""


def narrow_candidates(
    strategy: RoutingStrategy,
    req: Request,
    candidates: list[InstanceView],
    context: RoutingContext,
) -> list[int]:
    """Apply an optional strategy-owned admission rule."""
    narrow = getattr(strategy, "narrow_candidates", None)
    if not callable(narrow):
        return list(range(len(candidates)))
    return narrow(req, candidates, context)


def allows_version_bypass(
    strategy: RoutingStrategy,
    req: Request,
    context: RoutingContext,
) -> bool:
    """Return whether dispatch may continue after this version blocks."""
    allow = getattr(strategy, "allow_version_bypass", None)
    return bool(allow(req, context)) if callable(allow) else False


def select_with_context(
    strategy: RoutingStrategy,
    req: Request,
    candidates: list[InstanceView],
    context: RoutingContext,
) -> int | None:
    """Choose an instance, passing the context to a strategy that wants it.

    `select` is the whole protocol for a strategy that ranks instances on load or
    cache state. One that prices a deadline also needs the clock, so it declares
    `select_context` instead and receives both. Duck-typed like
    `narrow_candidates` and `allow_version_bypass`, so no strategy is forced to
    accept an argument it has no use for.
    """
    contextual = getattr(strategy, "select_context", None)
    if callable(contextual):
        return contextual(req, candidates, context)
    return strategy.select(req, candidates)


STRATEGY_REGISTRY: dict[str, Callable[..., RoutingStrategy]] = {}
"""Strategy name -> constructor. Populated by `register_strategy`."""


def register_strategy(name: str) -> Callable[[type], type]:
    """Class decorator adding a strategy to the registry under `name`.

    Raises:
        ValueError: If `name` is already registered, so two strategies cannot
            silently shadow each other.
    """

    def wrap(cls: type) -> type:
        if name in STRATEGY_REGISTRY:
            raise ValueError(f"Routing strategy {name!r} is already registered.")
        STRATEGY_REGISTRY[name] = cls
        cls.name = name
        return cls

    return wrap


def build_strategy(name: str, **kwargs) -> RoutingStrategy:
    """Construct the registered strategy called `name`.

    Raises:
        KeyError: If no strategy is registered under that name, listing the
            ones that are.
    """
    if name not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Unknown routing strategy {name!r}. Available: {sorted(STRATEGY_REGISTRY)}."
        )
    return STRATEGY_REGISTRY[name](**kwargs)
