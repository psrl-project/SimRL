"""Convert a real verl separate_async run's telemetry directory into the
simulator's own `JsonlStore` streams.

Verl's telemetry (`verl/utils/telemetry_sink.py` in verl-new's
`psrl-compare` branch) already writes structured JSON with proper
wall-clock timestamps, so unlike `psrl_log.py` this module does no text-log
parsing and no UTC-offset inference: every source file already carries a
`ts` field in the same clock.

Nothing here writes to the verl run's telemetry directory: it is opened
read-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .._logging import srl_logger
from ..workload.trace import RequestGroup, write_group_trace
from .store import JsonlStore


@dataclass(frozen=True)
class VerlLogLayout:
    """Where each fact lives inside a verl telemetry directory.

    Args:
        request_trace_glob: Per-process completion trace, one JSON object per
            line. One file per writer process (`telemetry_sink.py` names each
            by its own pid), so every match is read and merged.
        stats_glob: Per-replica engine occupancy, one JSON object per line.
            `{r}` in the filename is the replica rank.
        train_metrics_filename: One JSON object per training step, written by
            `write_step_metrics` straight from `trainer_base.py`'s own
            `fit()` loop. Optional: an older telemetry directory predating
            this file is still convertible, just without a `train` stream.
    """

    request_trace_glob: str = "request_trace_pid*.jsonl"
    stats_glob: str = "stats_r*.jsonl"
    train_metrics_filename: str = "train_metrics.jsonl"


@dataclass
class ConversionSummary:
    """What `convert_verl_logs` found, for the CLI to report back.

    Args:
        instance_ids: Replica ranks written as `instance/*` streams.
        n_instance_rows: Occupancy samples written across all instances.
        n_request_rows: Completed requests read from the trace files.
        n_train_rows: Training steps written as `train` stream rows. Zero
            when `train_metrics.jsonl` is absent, not an error.
        duration_s: Wall-clock span covered by the instance streams.
    """

    instance_ids: list[int] = field(default_factory=list)
    n_instance_rows: int = 0
    n_request_rows: int = 0
    n_train_rows: int = 0
    duration_s: float = 0.0


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _instance_id_from_stats_path(path: Path) -> int:
    # "stats_r3.jsonl" -> 3
    return int(path.stem.rsplit("_r", 1)[1])


def convert_verl_logs(
    log_dir: str | Path,
    store: JsonlStore,
    layout: VerlLogLayout | None = None,
) -> ConversionSummary:
    """Rewrite a verl telemetry directory into simulator-shaped streams.

    Writes `instance/{id}` into `store`, using the same field names
    `SimTelemetry` writes, so `SimPlotter` needs no knowledge of where the
    records came from. Times are rebased so that zero is the first engine
    stats sample, matching the simulator's time-since-start axis.

    Also writes a `train` stream when `train_metrics.jsonl` is present,
    mapping `trainer_base.py`'s own step-timing fields onto the simulator's
    three-phase breakdown: `timing_s/old_log_prob` is verl's own recompute
    (the forward pass that recomputes log probs under the current policy
    before the step trains on them) as `recompute_time`, `timing_s/
    update_actor` (the gradient step itself) as `train_time`, and `timing_s/
    update_weights` as `sync_time`. `timing_s/gen` (time blocked inside
    `replay_buffer.sample()`, i.e. waiting on rollout) and `timing_s/adv`
    (advantage estimation, sub-millisecond here) are deliberately excluded
    from the block: the first is rollout, not training, and the second is
    folded into `train_time` automatically by the plotter's own
    unattributed-time rule rather than drawn as its own band.

    Args:
        log_dir: A verl run's telemetry directory (`VERL_TELEMETRY_LOG_DIR`).
            Opened read-only.
        store: Destination store, typically rooted at `<output_dir>/logs`.
        layout: Which files to read. Defaults to the layout
            `telemetry_sink.py` writes today.

    Returns:
        A summary of what was converted, for the caller to report.

    Raises:
        ValueError: If the directory has no stats files to convert.
    """
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        raise ValueError(f"No verl telemetry directory at {log_dir!s}.")
    layout = layout or VerlLogLayout()

    stats_paths = sorted(log_dir.glob(layout.stats_glob))
    if not stats_paths:
        raise ValueError(f"No stats files matching {layout.stats_glob!r} found under {log_dir!s}.")

    stats_rows = {_instance_id_from_stats_path(p): _read_jsonl(p) for p in stats_paths}
    empty = sorted(iid for iid, rows in stats_rows.items() if not rows)
    if empty:
        raise ValueError(f"Replicas {empty} have empty stats files under {log_dir!s}, so there is nothing to plot.")

    origin = min(rows[0]["ts"] for rows in stats_rows.values())
    last = max(rows[-1]["ts"] for rows in stats_rows.values())
    duration_s = last - origin

    n_instance_rows = 0
    for instance_id, rows in stats_rows.items():
        for row in rows:
            store.write(
                f"instance/{instance_id}",
                t=row["ts"] - origin,
                instance_id=instance_id,
                num_running_reqs=row["num_running_reqs"],
                num_waiting_reqs=row["num_waiting_reqs"],
                kv_cache_usage=row["kv_cache_usage"],
            )
            n_instance_rows += 1

    request_paths = sorted(log_dir.glob(layout.request_trace_glob))
    n_request_rows = sum(len(_read_jsonl(p)) for p in request_paths)

    n_train_rows = 0
    train_metrics_path = log_dir / layout.train_metrics_filename
    if train_metrics_path.is_file():
        for row in _read_jsonl(train_metrics_path):
            recompute_time = float(row.get("timing_s/old_log_prob", 0.0) or 0.0)
            train_time = float(row.get("timing_s/update_actor", 0.0) or 0.0)
            sync_time = float(row.get("timing_s/update_weights", 0.0) or 0.0)
            end = row["ts"] - origin
            start = end - (recompute_time + train_time + sync_time)
            store.write(
                "train",
                start=start,
                end=end,
                version=row.get("step"),
                recompute_time=recompute_time,
                train_time=train_time,
                sync_time=sync_time,
            )
            n_train_rows += 1

    srl_logger.info(
        "Converted %d instance rows across replicas %s, counted %d completed requests, "
        "and wrote %d train rows from %s.",
        n_instance_rows,
        sorted(stats_rows),
        n_request_rows,
        n_train_rows,
        str(log_dir),
    )

    return ConversionSummary(
        instance_ids=sorted(stats_rows),
        n_instance_rows=n_instance_rows,
        n_request_rows=n_request_rows,
        n_train_rows=n_train_rows,
        duration_s=duration_s,
    )


def extract_workload_trace(
    log_dir: str | Path,
    output_path: str | Path,
    group_size: int,
    layout: VerlLogLayout | None = None,
    max_output_tokens: int = 0,
) -> Path:
    """Write the lengths the engine actually generated as a joint group trace.

    Reads every completed request from the `request_trace_pid*.jsonl` files
    `telemetry_sink.py` writes (one per writer process). Unlike PSRL's
    `route_trace.log`, each line is already the final, cumulative record for
    that request (`FullyAsyncLLMServerClient.generate()`'s own resume loop
    has already finished by the time it writes), so there is no "only read
    the last pop" step to do.

    Grouping is not recorded in the trace, so requests are chunked in
    request_id order into groups of `group_size`, exactly as `psrl_log.py`
    does. That reproduces the length *distribution* and the group *arity*,
    not which prompt each response belonged to.

    Under `max_off_policy_strategy=drop`, a few requests still in flight
    when training stops never get a surviving sibling to complete alongside
    them, so the total is not always a whole number of groups. Those
    trailing stragglers (sorted-order, not time-order, so no real group is
    identifiable anyway) are dropped and logged rather than failing the
    whole conversion.

    Args:
        log_dir: A verl run's telemetry directory. Opened read-only.
        output_path: Destination JSONL trace.
        group_size: Responses per prompt (`actor_rollout_ref.rollout.n`).
        layout: Which files to read.
        max_output_tokens: Generation cap the run used, for truncation stats.
            Defaults to the longest observed response.

    Returns:
        The path written.

    Raises:
        ValueError: If no completed requests are found at all.
    """
    log_dir = Path(log_dir)
    layout = layout or VerlLogLayout()
    request_paths = sorted(log_dir.glob(layout.request_trace_glob))
    if not request_paths:
        raise ValueError(
            f"No request trace files matching {layout.request_trace_glob!r} under {log_dir!s}, "
            f"so real generated lengths are unavailable."
        )

    parsed: list[tuple[str, int, int]] = []
    for path in request_paths:
        for row in _read_jsonl(path):
            parsed.append((row["request_id"], row["prompt_tokens"], row["response_tokens"]))

    if not parsed:
        raise ValueError(f"No completed requests in {request_paths!r}, so no generated lengths could be read.")

    parsed.sort(key=lambda row: row[0])
    straggler_count = len(parsed) % group_size
    if straggler_count:
        srl_logger.warning(
            "Found %d completed requests, not a whole number of groups of %d. Dropping %d trailing "
            "stragglers (likely left in flight by max_off_policy_strategy=drop when training stopped).",
            len(parsed),
            group_size,
            straggler_count,
        )
        parsed = parsed[: len(parsed) - straggler_count]
    groups = [
        RequestGroup(
            group_id=index,
            prompt_len=parsed[index * group_size][1],
            output_lens=tuple(row[2] for row in parsed[index * group_size : (index + 1) * group_size]),
        )
        for index in range(len(parsed) // group_size)
    ]
    cap = max_output_tokens or max(row[2] for row in parsed)
    srl_logger.info(
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
            "provenance": "verl-separate-async-request-trace",
        },
    )
