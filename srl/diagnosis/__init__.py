"""Post-hoc diagnosis of a finished simulation.

Reads only the JSONL telemetry streams a run already wrote, so a diagnosis
never re-simulates and never changes what it measures. The entry points are
`request_phases` for per-request timelines, `steps` for which request bound
each training step, and the `bounds` module for optimistic counterfactuals.
"""

from __future__ import annotations

from .pricer import SoloPricer
from .timeline import Check, Phase, RequestTimeline, request_phases

__all__ = [
    "Check",
    "Phase",
    "RequestTimeline",
    "SoloPricer",
    "request_phases",
]
