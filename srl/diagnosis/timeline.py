"""Reconstruct each request's phase timeline from the telemetry streams.

The `instance/N` stream records a resident's `generated`, `ctx` and `prefilled`
at every committed change on that instance, so consecutive records bracket an
interval whose kind is decided by the state at its start. `prefilled < ctx`
means context is still being computed, `prefilled >= ctx` with `generated`
rising means decoding, and a drop of `prefilled` back below `ctx` after tokens
already existed means an abort or a preemption discarded the cache.

Every reconstruction is cross-checked against `request`, a stream this module
does not use to build phases, so a silent mismatch becomes a failed `Check`
rather than a plausible-looking number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PHASE_KINDS = (
    "router_queue",
    "engine_wait",
    "prefill",
    "reprefill",
    "decode",
    "weight_load",
)


@dataclass(frozen=True)
class Phase:
    """One contiguous interval of a request's life, with a single kind."""

    kind: str
    t0: float
    t1: float
    instance_id: int | None
    tokens: int
    ctx0: int
    attempt: int

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


@dataclass(frozen=True)
class Check:
    """One cross-source agreement test and its outcome."""

    name: str
    ok: bool
    detail: str


@dataclass
class RequestTimeline:
    """A request's phases plus the `request`-stream facts they are checked against."""

    rid: int
    group_id: int
    prompt_len: int
    target_len: int
    generated: int
    admit_time: float
    complete_time: float
    num_aborts: int
    num_preemptions: int
    reprefill_tokens: int
    phases: list[Phase] = field(default_factory=list)

    def seconds_in(self, kind: str) -> float:
        """Return total seconds this request spent in phases of `kind`."""
        if kind not in PHASE_KINDS:
            raise ValueError(f"Unknown phase kind {kind!r}, expected one of {PHASE_KINDS}.")
        return sum(phase.duration for phase in self.phases if phase.kind == kind)

    def tokens_in(self, kind: str) -> int:
        """Return total tokens attributed to phases of `kind`."""
        return sum(phase.tokens for phase in self.phases if phase.kind == kind)

    @property
    def instance_ids(self) -> tuple[int, ...]:
        seen = {phase.instance_id for phase in self.phases if phase.instance_id is not None}
        return tuple(sorted(seen))


def _consume_requests(store) -> dict[int, RequestTimeline]:
    """Build one timeline shell per consumed request, from the `request` stream."""
    out: dict[int, RequestTimeline] = {}
    for rec in store.iter("request"):
        if rec["event"] != "consume":
            continue
        out[rec["rid"]] = RequestTimeline(
            rid=rec["rid"],
            group_id=rec["group_id"],
            prompt_len=rec["prompt_len"],
            target_len=rec["target_len"],
            generated=rec["generated"],
            admit_time=rec["admit_time"],
            complete_time=rec["complete_time"],
            num_aborts=rec["num_aborts"],
            num_preemptions=rec["num_preemptions"],
            reprefill_tokens=rec["reprefill_tokens"],
        )
    return out


def _instance_streams(store) -> list[str]:
    return [name for name in store.list_streams() if name.startswith("instance/")]


def _engine_phases(store, timelines: dict[int, RequestTimeline]) -> dict[int, list[Phase]]:
    """Walk each instance stream, turning consecutive record pairs into phases.

    A record is written only when committed state changed, so the interval
    between two consecutive records has one kind throughout, decided by the
    state at its start. Queue membership alone does not decide the kind: a
    queued resident whose `prefilled` still lags `ctx` is being (re)computed
    just as much as a running one, so only a resident already fully prefilled
    while queued is genuinely idle. A resident missing from the later record
    on this instance either completed (the `request` stream's `complete_time`
    lines up with this interval's end) or moved to another instance, which
    that instance's own stream accounts for separately.
    """
    wanted = set(timelines)
    finals = {rid: (line.generated, line.complete_time) for rid, line in timelines.items()}
    phases: dict[int, list[Phase]] = {rid: [] for rid in wanted}
    attempt: dict[int, int] = dict.fromkeys(wanted, 0)
    prefill_done: dict[int, bool] = dict.fromkeys(wanted, False)
    in_reprefill: dict[int, bool] = dict.fromkeys(wanted, False)

    for stream in _instance_streams(store):
        records = list(store.iter(stream))
        for earlier, later in zip(records, records[1:]):
            t0, t1 = earlier["t"], later["t"]
            if t1 <= t0:
                continue
            instance_id = earlier["instance_id"]
            for queue in ("waiting_len", "running_len"):
                for rid_text, info in earlier[queue].items():
                    rid = int(rid_text)
                    if rid not in phases:
                        continue
                    ctx0 = info["ctx"]
                    prefilled = info["prefilled"]
                    nxt = later["running_len"].get(rid_text) or later["waiting_len"].get(rid_text)
                    if prefilled < ctx0:
                        kind = "reprefill" if prefill_done[rid] else "prefill"
                        if kind == "reprefill" and not in_reprefill[rid]:
                            attempt[rid] += 1
                            in_reprefill[rid] = True
                        computed = (nxt["prefilled"] if nxt is not None else ctx0) - prefilled
                        phases[rid].append(
                            Phase(kind, t0, t1, instance_id, max(0, computed), ctx0, attempt[rid])
                        )
                        if nxt is not None and nxt["prefilled"] >= nxt["ctx"]:
                            prefill_done[rid] = True
                        continue
                    in_reprefill[rid] = False
                    if queue == "waiting_len":
                        phases[rid].append(
                            Phase("engine_wait", t0, t1, instance_id, 0, ctx0, attempt[rid])
                        )
                        continue
                    if nxt is not None:
                        decoded = nxt["generated"] - info["generated"]
                    else:
                        # Gone from this instance with no trace, so either this
                        # interval ends at the known completion time, or the
                        # resident moved on and generated nothing on the hop.
                        final_generated, complete_time = finals[rid]
                        at_completion = abs(t1 - complete_time) <= 1e-9
                        decoded = (final_generated - info["generated"]) if at_completion else 0
                    prefill_done[rid] = True
                    phases[rid].append(
                        Phase("decode", t0, t1, instance_id, max(0, decoded), ctx0, attempt[rid])
                    )
    for rid in phases:
        phases[rid].sort(key=lambda p: p.t0)
    return phases


def _weight_load_windows(store) -> list[tuple[float, float]]:
    """Return the `[start, end]` windows a publish spent loading weights.

    The `train` record carries `dt` and the three priced components, so the
    remainder is the rollout-side weight load the publish paused for.
    """
    windows = []
    for rec in store.iter("train"):
        charged = rec["train_time"] + rec["sync_time"] + rec["recompute_time"]
        load = rec["dt"] - charged
        if load > 1e-12:
            windows.append((rec["end"] - load, rec["end"]))
    return windows


def _fill_gaps(line: RequestTimeline, windows: list[tuple[float, float]]) -> list[Phase]:
    """Cover a request's lifetime completely, labelling uncovered stretches.

    An uncovered stretch is time the request was admitted and unfinished while
    on no instance, which is router queueing, except where it overlaps a
    weight-load window.
    """
    out = list(line.phases)
    cursor = line.admit_time
    for phase in sorted(line.phases, key=lambda p: p.t0):
        if phase.t0 > cursor + 1e-12:
            out.extend(_label_gap(cursor, phase.t0, windows, phase.attempt))
        cursor = max(cursor, phase.t1)
    if line.complete_time > cursor + 1e-12:
        last_attempt = max((p.attempt for p in line.phases), default=0)
        out.extend(_label_gap(cursor, line.complete_time, windows, last_attempt))
    out.sort(key=lambda p: p.t0)
    return out


def _label_gap(t0: float, t1: float, windows: list[tuple[float, float]], attempt: int) -> list[Phase]:
    """Split `[t0, t1)` into router-queue and weight-load pieces."""
    cuts = sorted({t0, t1} | {bound for window in windows for bound in window if t0 < bound < t1})
    pieces = []
    for left, right in zip(cuts, cuts[1:]):
        middle = (left + right) / 2.0
        paused = any(start <= middle <= end for start, end in windows)
        pieces.append(
            Phase("weight_load" if paused else "router_queue", left, right, None, 0, 0, attempt)
        )
    return pieces


def _checks(timelines: dict[int, RequestTimeline]) -> list[Check]:
    """Compare the reconstruction against facts it did not read."""
    clean = [
        line
        for line in timelines.values()
        if line.num_aborts == 0 and line.num_preemptions == 0
    ]
    token_bad = [
        (line.rid, line.tokens_in("decode"), line.generated)
        for line in clean
        if line.tokens_in("decode") != line.generated
    ]
    time_bad = [
        (line.rid, max((p.t1 for p in line.phases), default=line.admit_time), line.complete_time)
        for line in timelines.values()
        if abs(max((p.t1 for p in line.phases), default=line.admit_time) - line.complete_time) > 1e-6
    ]
    abort_bad = [
        (line.rid, max((p.attempt for p in line.phases), default=0), line.num_aborts + line.num_preemptions)
        for line in timelines.values()
        if max((p.attempt for p in line.phases), default=0) > line.num_aborts + line.num_preemptions
    ]
    reprefill_bad = [
        line.rid
        for line in timelines.values()
        if line.reprefill_tokens > 0 and line.seconds_in("reprefill") <= 0.0
    ]
    return [
        Check(
            "decode_tokens_match",
            not token_bad,
            f"{len(token_bad)} of {len(clean)} uninterrupted requests disagree: {token_bad[:5]}",
        ),
        Check(
            "completion_time_match",
            not time_bad,
            f"{len(time_bad)} requests end at a time other than complete_time: {time_bad[:5]}",
        ),
        Check(
            "abort_count_match",
            not abort_bad,
            f"{len(abort_bad)} requests show more attempts than aborts plus preemptions: {abort_bad[:5]}",
        ),
        Check(
            "reprefill_present",
            not reprefill_bad,
            f"{len(reprefill_bad)} requests charged reprefill tokens but show no reprefill phase: {reprefill_bad[:5]}",
        ),
    ]


def request_phases(store) -> tuple[dict[int, RequestTimeline], list[Check]]:
    """
    Reconstruct every consumed request's phase timeline, with cross-source checks.

    Args:
        store (JsonlStore): A finished run's log store, opened for reading.

    Returns:
        tuple[dict[int, RequestTimeline], list[Check]]: Timelines by request id,
            and the agreement checks. A failed check means the reconstruction
            and the `request` stream disagree, which is a defect to fix rather
            than a tolerance to widen.
    """
    timelines = _consume_requests(store)
    if not timelines:
        return {}, [Check("has_consumed_requests", False, "The request stream holds no consume events.")]
    engine = _engine_phases(store, timelines)
    windows = _weight_load_windows(store)
    for rid, line in timelines.items():
        line.phases = engine.get(rid, [])
        line.phases = _fill_gaps(line, windows)
    return timelines, _checks(timelines)
