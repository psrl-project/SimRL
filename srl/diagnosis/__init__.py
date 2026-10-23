"""Post-hoc diagnosis of a finished simulation.

Reads only the JSONL telemetry streams a run already wrote, so a diagnosis
never re-simulates and never changes what it measures. The entry points are
`request_phases` for per-request timelines, `steps` for which request bound
each training step, `staleness_breakdown` for the staleness composition, and
the `bounds` module for optimistic counterfactuals.
"""

from __future__ import annotations

from .bounds import Bound, span_solo_all, span_tail_only
from .critical_path import Decomposition, StepWait, decompose, steps
from .fleet import IdleWindow, InstanceImbalance, idle_windows, instance_imbalance
from .pricer import SoloPricer
from .staleness import StalenessBreakdown, check_same_consumption, staleness_breakdown
from .timeline import Check, Phase, RequestTimeline, request_phases

__all__ = [
    "Check",
    "Phase",
    "RequestTimeline",
    "SoloPricer",
    "request_phases",
    "Decomposition",
    "StepWait",
    "decompose",
    "steps",
    "Bound",
    "span_solo_all",
    "span_tail_only",
    "IdleWindow",
    "InstanceImbalance",
    "idle_windows",
    "instance_imbalance",
    "StalenessBreakdown",
    "check_same_consumption",
    "staleness_breakdown",
]
