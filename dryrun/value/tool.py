"""Time a trajectory spends waiting on a tool, which is zero here.

A deadline is wall clock, but only generating time converts into tokens. An agentic
trajectory spends part of its slack blocked on a tool, holding KV and producing
nothing, so its budget is `rate * (slack - expected_tool_wait)`.

DryRun has no tool calls, so the wait is zero and the two coincide. The subtraction
is still written where it belongs, in `Valuator.budget` by way of this protocol, so
turning it on is an implementation swap rather than a change to the budget arithmetic.
"""

from __future__ import annotations

from typing import Protocol

from .length import TrajView


class ToolTimeModel(Protocol):
    """Predicts remaining non-generating time in a trajectory's life."""

    name: str

    def expected_wait(self, traj: TrajView, horizon: float) -> float:
        """Return the seconds `traj` will spend blocked on tools within `horizon`.

        Args:
            traj: The trajectory being priced.
            horizon: Seconds until its deadline, bounding the answer.

        Returns:
            Seconds of non-generating time, in `[0, horizon]`.
        """
        ...


class NoToolCalls:
    """Every trajectory generates continuously from admission to completion."""

    name = "none"

    def expected_wait(self, traj: TrajView, horizon: float) -> float:
        return 0.0
