"""What an eviction strategy is, and how one is looked up by name.

When a decode step cannot fund its next token the engine evicts a resident: it
frees that request's KV, keeps its generated tokens, and puts it at the head of
the instance's own queue to resume later. Which resident pays is a policy, and
vLLM's own answer (reverse arrival order) is arbitrary with respect to a
deadline: the newest arrival may be the one most likely to finish in time.

Deliberately a separate registry from routing. The two decisions are made by
different components at different moments, and keeping them apart is what lets
an experiment change one while holding the other fixed.
"""

from __future__ import annotations

from typing import Callable, Protocol

from ..request import Request


class EvictionStrategy(Protocol):
    """Picks which resident to evict when the KV pool cannot fund a step."""

    name: str

    def select(self, candidates: list[Request], now: float) -> Request | None:
        """Choose a victim from `candidates`.

        Args:
            candidates: Residents eligible for eviction, never empty. The
                instance has already excluded the one request it must protect
                to guarantee forward progress.
            now: The instance's local clock.

        Returns:
            The request to evict, or None to defer to the engine's default.
            Returning None is how a strategy declines to express a preference,
            for example before it has learned enough to rank anything.
        """
        ...


EVICTION_REGISTRY: dict[str, Callable[..., EvictionStrategy]] = {}
"""Strategy name -> constructor. Populated by `register_eviction`."""


def register_eviction(name: str) -> Callable[[type], type]:
    """Class decorator adding an eviction strategy to the registry under `name`.

    Raises:
        ValueError: If `name` is already registered, so two strategies cannot
            silently shadow each other.
    """

    def wrap(cls: type) -> type:
        if name in EVICTION_REGISTRY:
            raise ValueError(f"Eviction strategy {name!r} is already registered.")
        EVICTION_REGISTRY[name] = cls
        cls.name = name
        return cls

    return wrap


def build_eviction(name: str, **kwargs) -> EvictionStrategy:
    """Construct the registered eviction strategy called `name`.

    Raises:
        KeyError: If no strategy is registered under that name, listing the
            ones that are.
    """
    if name not in EVICTION_REGISTRY:
        raise KeyError(
            f"Unknown eviction strategy {name!r}. Available: {sorted(EVICTION_REGISTRY)}."
        )
    return EVICTION_REGISTRY[name](**kwargs)
