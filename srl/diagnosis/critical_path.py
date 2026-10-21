"""Attribute each training step's wait to the request that bound it.

A step starts when its batch is complete, so the binding request of step `k`
is the last member of `k`'s batch to finish, and the step's idle time ends at
that completion. When it does not, the wait had another cause and is reported
as unexplained rather than attributed anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pricer import SoloPricer
from .timeline import RequestTimeline

_TIME_EPS = 1e-6


@dataclass(frozen=True)
class StepWait:
    """One training step, its idle time, and the request that bound it."""

    version: int
    start: float
    end: float
    idle: float
    train_seconds: float
    group_ids: tuple[int, ...]
    binding_group_id: int | None
    binding_rid: int | None
    binding_complete_time: float | None
    wait_cause: str
    sibling_spread: float


@dataclass(frozen=True)
class Decomposition:
    """Where one request's wall-clock lifetime went."""

    rid: int
    solo_seconds: float
    interference_seconds: float
    queue_seconds: float
    reprefill_seconds: float
    weight_load_seconds: float
    observed_seconds: float

    def shares(self) -> dict[str, float]:
        """Return each component as a fraction of the observed lifetime."""
        total = self.observed_seconds
        if total <= 0.0:
            return dict.fromkeys(
                ("solo", "interference", "queue", "reprefill", "weight_load"), 0.0
            )
        return {
            "solo": self.solo_seconds / total,
            "interference": self.interference_seconds / total,
            "queue": self.queue_seconds / total,
            "reprefill": self.reprefill_seconds / total,
            "weight_load": self.weight_load_seconds / total,
        }


def steps(store, timelines: dict[int, RequestTimeline]) -> list[StepWait]:
    """
    Return one `StepWait` per training step, in version order.

    Args:
        store (JsonlStore): A finished run's log store.
        timelines (dict[int, RequestTimeline]): Output of `request_phases`.

    Returns:
        list[StepWait]: Steps with their idle time and binding request. A step
            whose batch holds no timeline (every member dropped) carries
            `binding_rid=None` and `wait_cause="unexplained"`.
    """
    by_group: dict[int, list[RequestTimeline]] = {}
    for line in timelines.values():
        by_group.setdefault(line.group_id, []).append(line)

    consumed_at: dict[int, list[int]] = {}
    for rec in store.iter("request"):
        if rec["event"] == "consume":
            consumed_at.setdefault(rec["consume_version"], []).append(rec["group_id"])

    records = sorted(store.iter("train"), key=lambda rec: rec["version"])
    out: list[StepWait] = []
    previous_end = 0.0
    for rec in records:
        version = rec["version"]
        # A train record's version is the policy it produces, so the batch it
        # trained on is tagged with the version one below it.
        group_ids = tuple(sorted(set(consumed_at.get(version - 1, ()))))
        members = [line for gid in group_ids for line in by_group.get(gid, ())]
        idle = max(0.0, rec["start"] - previous_end)
        train_seconds = rec["train_time"] + rec["sync_time"] + rec["recompute_time"]
        binding = max(members, key=lambda line: line.complete_time) if members else None
        spread = 0.0
        cause = "no_wait" if idle <= _TIME_EPS else "unexplained"
        if binding is not None:
            siblings = [line for line in members if line.group_id == binding.group_id]
            spread = max(s.complete_time for s in siblings) - min(s.complete_time for s in siblings)
            if idle > _TIME_EPS and abs(binding.complete_time - rec["start"]) <= _TIME_EPS:
                cause = "completion"
        out.append(
            StepWait(
                version=version,
                start=rec["start"],
                end=rec["end"],
                idle=idle,
                train_seconds=train_seconds,
                group_ids=group_ids,
                binding_group_id=binding.group_id if binding else None,
                binding_rid=binding.rid if binding else None,
                binding_complete_time=binding.complete_time if binding else None,
                wait_cause=cause,
                sibling_spread=spread,
            )
        )
        previous_end = rec["end"]
    return out


def decompose(
    line: RequestTimeline,
    pricer: SoloPricer,
    tp_by_instance: dict[int, int],
) -> Decomposition:
    """
    Split one request's lifetime into solo work and the four things added to it.

    Solo work is the first prefill plus every decode segment priced at a batch
    of one. Interference is the observed time of those same phases minus their
    solo price, so it is exactly what co-residency cost. Re-prefill is charged
    whole, since a solo run would not have been interrupted.

    Args:
        line (RequestTimeline): One request's reconstructed timeline.
        pricer (SoloPricer): Prices a phase at a batch of one.
        tp_by_instance (dict[int, int]): Tensor-parallel degree per instance id.

    Returns:
        Decomposition: Components that sum to the observed lifetime.
    """
    solo = observed_work = 0.0
    for phase in line.phases:
        if phase.kind not in ("prefill", "decode"):
            continue
        tp = tp_by_instance.get(phase.instance_id, 1)
        if phase.kind == "prefill":
            solo += pricer.prefill_seconds(phase.tokens, tp)
        else:
            solo += pricer.decode_seconds(phase.ctx0, phase.tokens, tp)
        observed_work += phase.duration

    observed = line.complete_time - line.admit_time
    queue = line.seconds_in("router_queue") + line.seconds_in("engine_wait")
    reprefill = line.seconds_in("reprefill")
    weight_load = line.seconds_in("weight_load")
    # Solo price can exceed the observed time of a phase the engine batched
    # efficiently, so clamp rather than report negative interference.
    interference = max(0.0, observed_work - solo)
    solo = observed - interference - queue - reprefill - weight_load
    return Decomposition(
        rid=line.rid,
        solo_seconds=solo,
        interference_seconds=interference,
        queue_seconds=queue,
        reprefill_seconds=reprefill,
        weight_load_seconds=weight_load,
        observed_seconds=observed,
    )
