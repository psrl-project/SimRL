"""Convert a PSRL run's log directory into the simulator's own telemetry streams.

A real PSRL run and a DryRun-RL simulation describe the same thing: rollout
instances filling up and draining while training blocks the loop, but they
record it in different places and different formats. This module bridges the
two by rewriting a PSRL log directory into the exact `JsonlStore` streams that
`dryrun.telemetry.writer.SimTelemetry` emits, so that
`dryrun.visualizer.plots.SimPlotter` renders both with one implementation
instead of two that drift apart.

The mapping is close to a rename, because the PSRL engine sink already uses the
field names the simulator uses (`num_running_reqs`, `num_waiting_reqs`,
`kv_cache_usage`). Three things do need real work:

- **Role filtering.** Validation replicas write the same stats files as rollout
  replicas, and the rows carry no role field. The role lives only in the
  `DPLBStatCollector_{role}_I{n}.log` filenames, so that is what selects them.
- **Two clocks.** The JSONL sinks stamp UTC, the Python text logs stamp naive
  local time. `infer_utc_offset_hours` recovers the offset.
- **Training phases.** The simulator logs `recompute_time`, `train_time` and
  `sync_time` as one record; PSRL scatters the same three across a controller
  log and the per-rank train worker logs, as begin/end markers.

Nothing here writes to the PSRL directory: it is opened read-only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .._logging import dryrun_logger
from ..workload.trace import RequestGroup, write_group_trace
from .store import JsonlStore

# Text-log lines start with the `ray_logger` timestamp, e.g.
# "2026-09-04 01:17:42,684 - ray_logger.py - 73 - [End Event] TRAIN ...".
_TEXT_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
_TEXT_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"

# Instance index as it appears in `DPLBStatCollector_rollout_I3.log` and in
# `stats_r3_dp0.jsonl`. Both spellings identify the same replica.
_ROLE_LOG_INSTANCE = re.compile(r"_I(\d+)\.log$")
_STATS_INSTANCE = re.compile(r"^stats_r(\d+)_dp(\d+)\.jsonl$")

# Real-world UTC offsets are whole quarter hours, so rounding the measured
# difference to this grid removes the few seconds of skew between the last
# text-log line and the last stats sample.
_OFFSET_QUANTUM_S = 15 * 60


@dataclass(frozen=True)
class PsrlLogLayout:
    """Where each fact lives inside a PSRL log directory.

    Every filename and marker is a field rather than a literal in the parsing
    code, so a run that names things differently is handled by overriding the
    field instead of editing this module.

    Args:
        stats_glob: Per-engine vLLM scheduler stats, one JSON object per line.
        role_log_glob: Per-replica collector log. `{role}` is substituted.
        role: Which replica role to plot. Validation replicas write the same
            stats files, so this is what separates rollout from eval.
        gateway_stats: Router queue depth samples, one JSON object per line.
        phase_log: Controller log holding the recompute-phase begin marker.
        train_worker_glob: Per-rank train worker logs holding train and sync.
        recompute_begin_marker: Regex marking the start of the log-prob
            recomputation forward pass, which is where the training block
            begins.
        train_begin_marker: Regex marking the start of the optimizer step.
        train_end_marker: Regex marking the end of the optimizer step.
        sync_end_marker: Regex marking the end of the weight push, which is
            where the training block ends.
    """

    stats_glob: str = "stats_r*_dp*.jsonl"
    role_log_glob: str = "DPLBStatCollector_{role}_I*.log"
    role: str = "rollout"
    gateway_stats: str = "smg_routing_status.jsonl"
    phase_log: str = "MainRayTrainer.log"
    train_worker_glob: str = "TrainWorker_R*.log"
    recompute_begin_marker: str = r"\[SWITCH\] switch_to_trainer_mode"
    train_begin_marker: str = r"\[Begin Event\] TRAIN"
    train_end_marker: str = r"\[End Event\] TRAIN"
    sync_end_marker: str = r"\[End Event\] PUSH"
    trajectory_glob: str = "trajectories/v*/*.txt"
    route_trace: str = "route_trace.log"


@dataclass
class ConversionSummary:
    """What `convert_psrl_logs` found, for the CLI to report back.

    Args:
        instance_ids: Replica indices written as `instance/*` streams.
        dropped_instance_ids: Replica indices skipped because their role did
            not match.
        n_instance_rows: Occupancy samples written across all instances.
        n_train_steps: Training blocks written to the `train` stream.
        n_gateway_rows: Router queue samples written to the `gateway` stream.
        duration_s: Wall-clock span covered by the instance streams.
        utc_offset_hours: Offset applied to the naive text-log timestamps.
        mean_recompute_s: Mean log-prob recomputation time per step.
        mean_train_s: Mean optimizer step time per step.
        mean_sync_s: Mean weight push time per step.
    """

    instance_ids: list[int] = field(default_factory=list)
    dropped_instance_ids: list[int] = field(default_factory=list)
    n_instance_rows: int = 0
    n_train_steps: int = 0
    n_gateway_rows: int = 0
    duration_s: float = 0.0
    utc_offset_hours: float = 0.0
    mean_recompute_s: float = 0.0
    mean_train_s: float = 0.0
    mean_sync_s: float = 0.0


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _utc_seconds(stamp: str) -> float:
    """Epoch seconds from an ISO-8601 stamp that carries its own offset."""
    return datetime.fromisoformat(stamp).timestamp()


def _naive_seconds(stamp: str) -> float:
    """Epoch seconds from a naive text-log stamp, read as if it were UTC.

    The result is not a real instant. It is off by exactly the writer's UTC
    offset, which `infer_utc_offset_hours` measures and removes.
    """
    parsed = datetime.strptime(stamp, _TEXT_TIMESTAMP_FORMAT)
    return parsed.replace(tzinfo=timezone.utc).timestamp()


def _marker_times(path: Path, marker: str) -> list[float]:
    """Naive epoch seconds of every line in `path` matching `marker`."""
    pattern = re.compile(marker)
    times = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stamp = _TEXT_TIMESTAMP.match(line)
            if stamp is not None and pattern.search(line):
                times.append(_naive_seconds(stamp.group(1)))
    return times


def _last_text_log_time(log_dir: Path, layout: PsrlLogLayout) -> float:
    """Naive epoch seconds of the newest timestamped line in the text logs.

    Only the logs this module already reads are scanned, so the anchor cannot
    be dragged by an unrelated file that happened to outlive the run.
    """
    paths = [log_dir / layout.phase_log, *sorted(log_dir.glob(layout.train_worker_glob))]
    times = [t for path in paths if path.is_file() for t in _marker_times(path, r"")]
    if not times:
        raise ValueError(
            f"No timestamped lines found in {layout.phase_log} or {layout.train_worker_glob} under "
            f"{log_dir!s}, so the local-time offset cannot be inferred. Pass an explicit offset."
        )
    return max(times)


def infer_utc_offset_hours(log_dir: Path, layout: PsrlLogLayout, stats_rows: dict[int, list[dict]]) -> float:
    """Recover the UTC offset the text logs were written in.

    The JSON sinks stamp UTC while the Python text logs stamp naive local time,
    so the two cannot be placed on one axis until the offset is known. Both
    clocks are anchored at the *end* of the run, where the last text-log line
    and the last stats sample are within seconds of each other, and the
    difference is rounded onto the quarter-hour grid that real UTC offsets live
    on.

    Anchoring at the start would not work: the controller begins logging
    minutes before the engines come up, and that gap is genuine rather than
    clock skew.

    Args:
        log_dir: Directory holding the run's logs.
        layout: Which files to read.
        stats_rows: Already-parsed stats rows, keyed by replica index.

    Returns:
        The offset in hours, positive east of UTC.
    """
    last_stats = max(_utc_seconds(rows[-1]["ts"]) for rows in stats_rows.values() if rows)
    raw = _last_text_log_time(log_dir, layout) - last_stats
    offset_s = round(raw / _OFFSET_QUANTUM_S) * _OFFSET_QUANTUM_S
    dryrun_logger.info(
        "Inferred a UTC offset of %+.2f h for the text logs (raw gap %.1f s, residual %+.1f s).",
        offset_s / 3600.0,
        raw,
        raw - offset_s,
    )
    return offset_s / 3600.0


def _instance_ids_for_role(log_dir: Path, layout: PsrlLogLayout) -> list[int]:
    """Replica indices whose collector log matches `layout.role`.

    The stats rows themselves carry no role, and an idle rollout replica is
    indistinguishable from a paused validation one by content alone, so the
    filenames are the only trustworthy source.
    """
    pattern = layout.role_log_glob.format(role=layout.role)
    ids = []
    for path in sorted(log_dir.glob(pattern)):
        matched = _ROLE_LOG_INSTANCE.search(path.name)
        if matched is not None:
            ids.append(int(matched.group(1)))
    return sorted(set(ids))


def _stats_paths_by_instance(log_dir: Path, layout: PsrlLogLayout) -> dict[int, Path]:
    paths = {}
    for path in sorted(log_dir.glob(layout.stats_glob)):
        matched = _STATS_INSTANCE.match(path.name)
        if matched is not None:
            paths[int(matched.group(1))] = path
    return paths


def _select_instances(
    log_dir: Path,
    layout: PsrlLogLayout,
    instance_ids: list[int] | None,
) -> tuple[dict[int, Path], list[int]]:
    """Resolve which stats files to convert, and which were left behind."""
    available = _stats_paths_by_instance(log_dir, layout)
    if not available:
        raise ValueError(f"No stats files matching {layout.stats_glob!r} found under {log_dir!s}.")

    if instance_ids is None:
        selected_ids = _instance_ids_for_role(log_dir, layout)
        if not selected_ids:
            pattern = layout.role_log_glob.format(role=layout.role)
            raise ValueError(
                f"No collector logs matching {pattern!r} found under {log_dir!s}, so rollout replicas "
                f"cannot be told apart from validation ones. Name the replicas explicitly instead."
            )
    else:
        selected_ids = sorted(set(instance_ids))

    missing = [iid for iid in selected_ids if iid not in available]
    if missing:
        raise ValueError(
            f"Replicas {missing} have no stats file under {log_dir!s}. "
            f"Stats files exist for replicas {sorted(available)}."
        )

    selected = {iid: available[iid] for iid in selected_ids}
    dropped = sorted(set(available) - set(selected_ids))
    return selected, dropped


def _train_phases(
    log_dir: Path,
    layout: PsrlLogLayout,
    utc_offset_hours: float,
) -> list[tuple[float, float, float, float]]:
    """Rebuild each training block as `(start, recompute, train, sync)`.

    PSRL records the block as four markers rather than one record: the
    controller logs the switch into trainer mode (where the log-prob
    recomputation forward pass begins), and every train worker rank logs the
    begin and end of the optimizer step and the end of the weight push.

    Ranks are unioned rather than sampled: the earliest begin and the latest
    end across ranks is the interval during which the trainer was busy, which
    is what the plot is claiming. Picking one rank would understate it.

    Returns:
        One tuple per training step, in chronological order. The three
        durations are contiguous and sum to the block's total length.
    """
    offset_s = utc_offset_hours * 3600.0
    phase_path = log_dir / layout.phase_log
    if not phase_path.is_file():
        raise ValueError(f"No phase log at {phase_path!s}, so training blocks cannot be located.")

    recompute_starts = _marker_times(phase_path, layout.recompute_begin_marker)
    if not recompute_starts:
        raise ValueError(
            f"No lines matching {layout.recompute_begin_marker!r} in {phase_path!s}, "
            f"so training blocks cannot be located."
        )

    worker_paths = sorted(log_dir.glob(layout.train_worker_glob))
    if not worker_paths:
        raise ValueError(
            f"No train worker logs matching {layout.train_worker_glob!r} under {log_dir!s}, "
            f"so training and sync times cannot be recovered."
        )

    per_rank = {}
    for path in worker_paths:
        per_rank[path.name] = (
            _marker_times(path, layout.train_begin_marker),
            _marker_times(path, layout.train_end_marker),
            _marker_times(path, layout.sync_end_marker),
        )

    counts = {name: (len(b), len(e), len(s)) for name, (b, e, s) in per_rank.items()}
    distinct = set(counts.values())
    if len(distinct) != 1:
        raise ValueError(
            f"Train worker logs disagree on how many training steps they saw: {counts}. "
            f"The run is probably incomplete or the markers are mismatched."
        )

    n_begin, n_end, n_sync = distinct.pop()
    if not n_begin or n_begin != n_end or n_begin != n_sync:
        raise ValueError(
            f"Train worker logs hold {n_begin} train-begin, {n_end} train-end and {n_sync} sync-end "
            f"markers. All three must be present and equal to pair them into steps."
        )

    # A run interrupted mid-step leaves a trailing recompute start with no
    # optimizer step behind it, so pair from the front and drop the remainder.
    n_steps = min(len(recompute_starts), n_begin)
    if len(recompute_starts) != n_begin:
        dryrun_logger.warning(
            "Found %d recompute markers but %d training steps, so the last %d block(s) are incomplete "
            "and were dropped.",
            len(recompute_starts),
            n_begin,
            abs(len(recompute_starts) - n_begin),
        )

    train_begins = [min(per_rank[name][0][i] for name in per_rank) for i in range(n_steps)]
    train_ends = [max(per_rank[name][1][i] for name in per_rank) for i in range(n_steps)]
    sync_ends = [max(per_rank[name][2][i] for name in per_rank) for i in range(n_steps)]

    phases = []
    for i in range(n_steps):
        start = recompute_starts[i] - offset_s
        recompute_s = train_begins[i] - recompute_starts[i]
        train_s = train_ends[i] - train_begins[i]
        sync_s = sync_ends[i] - train_ends[i]
        if min(recompute_s, train_s, sync_s) < 0.0:
            raise ValueError(
                f"Training block {i} has markers out of order (recompute {recompute_s:.2f}s, "
                f"train {train_s:.2f}s, sync {sync_s:.2f}s). The marker patterns are probably "
                f"matching the wrong lines."
            )
        phases.append((start, recompute_s, train_s, sync_s))
    return phases


def _write_instances(
    store: JsonlStore,
    stats_rows: dict[int, list[dict]],
    origin: float,
) -> int:
    n_rows = 0
    for instance_id, rows in stats_rows.items():
        for row in rows:
            store.write(
                f"instance/{instance_id}",
                t=_utc_seconds(row["ts"]) - origin,
                instance_id=instance_id,
                num_running_reqs=row["num_running_reqs"],
                num_waiting_reqs=row["num_waiting_reqs"],
                kv_cache_usage=row["kv_cache_usage"],
            )
            n_rows += 1
    return n_rows


def _write_gateway(store: JsonlStore, path: Path, origin: float) -> int:
    if not path.is_file():
        dryrun_logger.info("No gateway stats at %s, so the queue-depth row is omitted.", str(path))
        return 0
    n_rows = 0
    for row in _read_jsonl(path):
        store.write(
            "gateway",
            t=_utc_seconds(row["ts"]) - origin,
            queue_len=row["queue_len"],
            pending_request_num=row["pending_request_num"],
            running_request_num=row["running_request_num"],
        )
        n_rows += 1
    return n_rows


def convert_psrl_logs(
    log_dir: str | Path,
    store: JsonlStore,
    layout: PsrlLogLayout | None = None,
    instance_ids: list[int] | None = None,
    utc_offset_hours: float | None = None,
) -> ConversionSummary:
    """Rewrite a PSRL log directory into simulator-shaped telemetry streams.

    Writes `instance/{id}`, `train` and `gateway` into `store`, using the same
    field names `SimTelemetry` writes, so `SimPlotter` needs no knowledge of
    where the records came from. Times are rebased so that zero is the first
    engine stats sample, matching the simulator's time-since-start axis.

    Args:
        log_dir: A PSRL run's log directory. Opened read-only.
        store: Destination store, typically rooted at `<output_dir>/logs`.
        layout: Which files and markers to read. Defaults to the layout PSRL
            writes today.
        instance_ids: Replica indices to convert. When None, they are taken
            from the collector logs matching `layout.role`.
        utc_offset_hours: Offset the text logs were written in. When None, it
            is inferred from the two clocks.

    Returns:
        A summary of what was converted, for the caller to report.

    Raises:
        ValueError: If the directory is missing files this conversion needs, or
            if the training markers cannot be paired into steps.
    """
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        raise ValueError(f"No PSRL log directory at {log_dir!s}.")
    layout = layout or PsrlLogLayout()

    stats_paths, dropped = _select_instances(log_dir, layout, instance_ids)
    stats_rows = {iid: _read_jsonl(path) for iid, path in stats_paths.items()}
    empty = sorted(iid for iid, rows in stats_rows.items() if not rows)
    if empty:
        raise ValueError(f"Replicas {empty} have empty stats files under {log_dir!s}, so there is nothing to plot.")

    if utc_offset_hours is None:
        utc_offset_hours = infer_utc_offset_hours(log_dir, layout, stats_rows)

    origin = min(_utc_seconds(rows[0]["ts"]) for rows in stats_rows.values())
    last = max(_utc_seconds(rows[-1]["ts"]) for rows in stats_rows.values())
    duration_s = last - origin

    n_instance_rows = _write_instances(store, stats_rows, origin)
    n_gateway_rows = _write_gateway(store, log_dir / layout.gateway_stats, origin)

    phases = _train_phases(log_dir, layout, utc_offset_hours)
    recompute_total = train_total = sync_total = 0.0
    for version, (start, recompute_s, train_s, sync_s) in enumerate(phases, start=1):
        start -= origin
        end = start + recompute_s + train_s + sync_s
        store.write(
            "train",
            t=end,
            start=start,
            end=end,
            dt=end - start,
            version=version,
            recompute_time=recompute_s,
            train_time=train_s,
            sync_time=sync_s,
        )
        recompute_total += recompute_s
        train_total += train_s
        sync_total += sync_s

    _check_phases_within_span(phases, origin, duration_s)

    n_steps = len(phases)
    return ConversionSummary(
        instance_ids=sorted(stats_paths),
        dropped_instance_ids=dropped,
        n_instance_rows=n_instance_rows,
        n_train_steps=n_steps,
        n_gateway_rows=n_gateway_rows,
        duration_s=duration_s,
        utc_offset_hours=utc_offset_hours,
        mean_recompute_s=recompute_total / n_steps if n_steps else 0.0,
        mean_train_s=train_total / n_steps if n_steps else 0.0,
        mean_sync_s=sync_total / n_steps if n_steps else 0.0,
    )


def extract_workload_trace(
    log_dir: str | Path,
    output_path: str | Path,
    group_size: int,
    layout: PsrlLogLayout | None = None,
    max_output_tokens: int = 0,
) -> Path:
    """Write the lengths the engine actually generated as a joint group trace.

    Simulating a synthetic length distribution leaves workload mismatch
    entangled with cost-model error: a divergent KV curve could be either.
    Replaying the run's own lengths removes that ambiguity and leaves the cost
    models as the only thing under test.

    The lengths come from `route_trace.log`'s `pop` events, **not** from the
    trajectory dumps. The dumps record the final training sample, which is
    truncated to the response cap, so their longest entry is exactly 10240 on
    the DAPO baseline. The engine generated more than that: 242 of 398
    never-interrupted requests emitted over 10240 tokens, each landing exactly
    on `max_prompt_length + max_output_length - prompt_len` (a 90-token prompt
    produced 11174). Sourcing from the dumps therefore understated generation
    by roughly 900 tokens per request and made the simulated first batch finish
    at 342.6 s against a measured 386.6 s.

    A request interrupted by a weight update pops twice, and its
    `response_tokens` is cumulative across iterations, so the final pop already
    carries the whole generation. Only `status="completed"` pops are read.

    Grouping is not recorded in the trace, so requests are chunked in id order
    into groups of `group_size`. That reproduces the length *distribution* and
    the group *arity* exactly, which is what the simulator consumes; it does not
    reproduce which prompt each response belonged to.

    Args:
        log_dir: A PSRL run's log directory. Opened read-only.
        output_path: Destination JSONL trace.
        group_size: Responses per prompt (the recipe's `rollout_n`).
        layout: Which files to read.
        max_output_tokens: Generation cap the run used, for truncation stats.
            Defaults to the longest observed response.

    Returns:
        The path written.

    Raises:
        ValueError: If no completed pop events are found, or if the count is
            not a whole number of groups.
    """
    log_dir = Path(log_dir)
    layout = layout or PsrlLogLayout()
    route_trace = log_dir / layout.route_trace
    if not route_trace.is_file():
        raise ValueError(f"No route trace at {route_trace!s}, so real generated lengths are unavailable.")

    fields = re.compile(r'(\w+)=("[^"]*"|\S+)')
    parsed: list[tuple[int, int, int]] = []
    for line in route_trace.read_text(encoding="utf-8", errors="replace").splitlines():
        if 'event="pop"' not in line or 'status="completed"' not in line:
            continue
        row = dict(fields.findall(line))
        try:
            request_id = int(row["request_id"])
        except (KeyError, ValueError):
            continue
        parsed.append((request_id, int(row["prompt_tokens"]), int(row["response_tokens"])))

    if not parsed:
        raise ValueError(
            f"No completed 'pop' events in {route_trace!s}, so no generated lengths could be read. "
            f"Check that the gateway's route_trace target was enabled for this run."
        )
    if len(parsed) % group_size != 0:
        raise ValueError(
            f"Found {len(parsed)} completed requests, which is not a whole number of groups of "
            f"{group_size}. Pass the group_size this run was launched with."
        )

    parsed.sort()
    groups = [
        RequestGroup(
            group_id=index,
            prompt_len=parsed[index * group_size][1],
            output_lens=tuple(row[2] for row in parsed[index * group_size : (index + 1) * group_size]),
        )
        for index in range(len(parsed) // group_size)
    ]
    cap = max_output_tokens or max(row[2] for row in parsed)
    dryrun_logger.info(
        "Extracted %d trajectories into %d groups of %d from %s.",
        len(parsed),
        len(groups),
        group_size,
        str(log_dir),
    )
    return write_group_trace(
        groups,
        output_path,
        metadata={
            "source": str(log_dir),
            "max_output_tokens": cap,
            "group_size": group_size,
            "provenance": "psrl-trajectory-dump",
        },
    )


def _check_phases_within_span(
    phases: list[tuple[float, float, float, float]],
    origin: float,
    duration_s: float,
    tolerance_s: float = 60.0,
) -> None:
    """Warn when the offset-corrected training blocks fall outside the run.

    A wrong UTC offset moves the training blocks by whole hours while leaving
    the instance streams untouched, which would otherwise show up only as an
    empty-looking figure. The blocks may legitimately start slightly before the
    first stats sample, hence the tolerance.
    """
    if not phases:
        return
    first = phases[0][0] - origin
    last = phases[-1][0] - origin + sum(phases[-1][1:])
    if first < -tolerance_s or last > duration_s + tolerance_s:
        dryrun_logger.warning(
            "Training blocks span %.1f s to %.1f s but the engine stats only cover 0.0 s to %.1f s. "
            "The inferred local-time offset is probably wrong.",
            first,
            last,
            duration_s,
        )
