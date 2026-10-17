"""Discrete events emitted by an engine instance."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .request import Request


class EventKind(Enum):
    """What an event did, not which code path produced it.

    There is one preemption kind because there is one preemption: a victim is
    evicted and its blocks are freed. The engine reaches that from a decode
    segment hitting the KV ceiling and from a step whose decode pass was
    refused a block, but the two are indistinguishable downstream and vLLM
    models them as one mechanism (`_preempt_request`).
    """

    ITERATION_DONE = "iteration_done"
    """One engine iteration advanced and nothing else happened.

    Not "a prefill step": the iteration decodes the running batch and prefills
    only what the token budget still allows, and on an oversubscribed pool
    98.7% of them admit nothing at all. It is emitted only when the iteration
    neither retired nor evicted anything, so it carries no payload and exists
    to tell the engine loop to seed the next one.

    A decode segment never emits this. A segment runs until something happens,
    so it always ends in `REQUEST_COMPLETE` or `PREEMPT`.
    """

    REQUEST_COMPLETE = "request_complete"
    PREEMPT = "preempt"


@dataclass
class EngineEvent:
    """
    A single discrete event produced by one `NativeInstance`.

    Attributes:
        kind: Category of state change this event represents. A step can both
            retire and evict, so `kind` reports the one the engine loop
            dispatches on and the two lists carry what actually happened.
        instance_id: Which instance produced this event.
        t: Value of `t_local` on that instance immediately after the event.
        completed: Requests that finished generation as part of this event.
            Non-empty only for `REQUEST_COMPLETE` events.
        preempted: Requests evicted as part of this event. A `REQUEST_COMPLETE`
            may carry these too: the step that retires a request can be the
            same one whose decode pass was refused a block.
    """

    kind: EventKind
    instance_id: int
    t: float
    completed: list[Request] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)
