"""Sharding for distributed workload collection.

A worker is one `(host, gpu)` pair addressed by a flat index. Prompts are split
as `prompts[index::n_workers]`, so a worker's shard is a function of its index
alone: rerunning worker 7 reproduces exactly worker 7's prompts, and one failed
shard retries without disturbing the others.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._logging import srl_logger
from .trace import RequestGroup, read_group_trace, write_group_trace

# Provenance fields that are copied from shard headers into the merged header
# when they are identical across every shard. Per-worker fields that are
# meaningless after merging are excluded.
_PROVENANCE_FIELDS = (
    "model",
    "dataset",
    "tokenizer",
    "n",
    "tp",
    "seed",
    "temperature",
    "top_p",
    "kind",
)


@dataclass(frozen=True)
class Worker:
    """One collection worker: a GPU on a host, serving on its own port."""

    index: int
    host: str
    gpu: int
    port: int


@dataclass(frozen=True)
class ShardStatus:
    """Completeness information for one collection shard."""

    path: Path
    requested: int
    actual: int
    error: str | None = None

    @property
    def complete(self) -> bool:
        return self.error is None and self.requested > 0 and self.actual >= self.requested


def parse_hosts_file(path: str | Path) -> list[tuple[str, int]]:
    """
    Read a hosts file of `host [gpus]` lines.

    Blank lines and `#` comments are ignored. A line without a GPU count is
    treated as a single-GPU host. GPU count must be at least 1.

    Args:
        path (str | Path): Hosts file to read.

    Returns:
        list[tuple[str, int]]: `(host, gpu_count)` in file order.
    """
    hosts: list[tuple[str, int]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        gpu_count = int(parts[1]) if len(parts) > 1 else 1
        if gpu_count < 1:
            raise ValueError(
                f"Hosts file {str(path)!r} gives host {parts[0]!r} an invalid GPU count "
                f"of {gpu_count!r}, expected at least 1."
            )
        hosts.append((parts[0], gpu_count))
    if not hosts:
        raise ValueError(f"Hosts file {str(path)!r} lists no hosts.")
    return hosts


def build_workers(hosts: list[tuple[str, int]], base_port: int = 8700) -> list[Worker]:
    """
    Expand `(host, gpu_count)` pairs into one worker per GPU.

    Ports restart at `base_port` on each host, since each host has its own
    port space.

    Args:
        hosts (list[tuple[str, int]]): `(host, gpu_count)` pairs.
        base_port (int): First port used on each host.

    Returns:
        list[Worker]: Workers in host-then-GPU order, indexed from zero.
    """
    workers: list[Worker] = []
    for host, gpu_count in hosts:
        for gpu in range(gpu_count):
            workers.append(
                Worker(index=len(workers), host=host, gpu=gpu, port=base_port + gpu)
            )
    return workers


def shard_indices(n_items: int, worker_index: int, n_workers: int) -> list[int]:
    """
    Return the item indices assigned to `worker_index` by strided split.

    Args:
        n_items (int): Total number of items to shard.
        worker_index (int): Index of this worker (0 to n_workers - 1).
        n_workers (int): Total number of workers.

    Returns:
        list[int]: Indices assigned to this worker in strided order.
    """
    return list(range(worker_index, n_items, n_workers))


def shard_path(shard_dir: str | Path, worker_index: int) -> Path:
    """
    Return the shard artifact path for one worker.

    Args:
        shard_dir (str | Path): Directory holding shard files.
        worker_index (int): Index of the worker (zero-padded to 2 digits).

    Returns:
        Path: The shard file path (`worker-NN.jsonl`).
    """
    return Path(shard_dir) / f"worker-{worker_index:02d}.jsonl"


def read_trace_header(path: str | Path) -> dict[str, Any]:
    """Read and validate the JSON object on the first line of a trace."""
    trace_path = Path(path)
    try:
        with trace_path.open(encoding="utf-8") as fh:
            first_line = fh.readline()
    except OSError as exc:
        raise ValueError(f"Cannot read trace {str(trace_path)!r}: {exc}.") from exc
    if not first_line:
        raise ValueError(f"Trace {str(trace_path)!r} is empty.")
    try:
        header = json.loads(first_line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Trace {str(trace_path)!r} has an unreadable header: {exc}.") from exc
    if not isinstance(header, dict) or not header.get("_meta"):
        raise ValueError(f"Trace {str(trace_path)!r} does not start with a _meta header.")
    return header


def inspect_shard(path: str | Path) -> ShardStatus:
    """Return body-based completeness information without raising on bad shards."""
    shard = Path(path)
    try:
        header = read_trace_header(shard)
        requested = int(header.get("n_groups_requested", 0))
        if requested < 1:
            return ShardStatus(
                path=shard,
                requested=requested,
                actual=0,
                error="missing positive n_groups_requested",
            )
        actual = len(read_group_trace(shard))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return ShardStatus(path=shard, requested=0, actual=0, error=str(exc))
    return ShardStatus(path=shard, requested=requested, actual=actual)


def shard_is_complete(path: str | Path) -> bool:
    """Return whether a shard body contains every group requested by its header."""
    return inspect_shard(path).complete


def inspect_shards(shard_dir: str | Path, n_workers: int) -> list[ShardStatus]:
    """Inspect all expected worker shards in worker-index order."""
    if n_workers < 1:
        raise ValueError(f"n_workers must be at least 1, got {n_workers!r}.")
    return [inspect_shard(shard_path(shard_dir, index)) for index in range(n_workers)]


def verify_collection(
    trace_path: str | Path,
    expected_groups: int,
    *,
    shard_dir: str | Path,
    n_workers: int,
) -> int:
    """Verify the merged body count and diagnose incomplete shards on shortfall."""
    if expected_groups < 1:
        raise ValueError(f"expected_groups must be at least 1, got {expected_groups!r}.")
    try:
        actual = len(read_group_trace(trace_path))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read merged trace {str(trace_path)!r}: {exc}.") from exc
    if actual >= expected_groups:
        return actual

    lines = [
        f"Merged trace is short by {expected_groups - actual} groups "
        f"({actual}/{expected_groups}).",
        "Short shards:",
    ]
    for index, status in enumerate(inspect_shards(shard_dir, n_workers)):
        if status.complete:
            continue
        if status.error:
            lines.append(f"  worker-{index:02d}: {status.error}")
        else:
            lines.append(f"  worker-{index:02d}: {status.actual}/{status.requested} groups")
    raise ValueError("\n".join(lines))


def merge_shards(
    shard_dir: str | Path,
    n_workers: int,
    output_path: str | Path,
    metadata: dict[str, Any],
) -> Path:
    """
    Merge every worker shard into one joint trace ordered by `group_id`.

    Every expected shard must exist and hold groups: a missing or empty shard
    is named rather than silently producing a short trace.

    The generation cap (`max_output_tokens`) is taken from the shard headers,
    not from `metadata`. All shards must agree on the cap; if they disagree a
    `ValueError` is raised naming the differing values and their workers.

    If `metadata` contains `max_output_tokens`, it is treated as an assertion:
    a mismatch with the shards' agreed cap raises a `ValueError`, naming both
    the caller-supplied value and the shards' cap.

    Provenance fields common to all shards (`model`, `dataset`, `tokenizer`,
    `n`, `tp`, `seed`, `temperature`, `top_p`, `kind`) are copied into the
    merged header. Fields that differ across shards are omitted silently.
    Per-worker fields (`worker_index`, `n_workers`, `n_groups_requested`) are
    never copied.

    Args:
        shard_dir (str | Path): Directory holding `worker-NN.jsonl` shards.
        n_workers (int): Number of shards expected.
        output_path (str | Path): Destination joint trace path.
        metadata (dict[str, Any]): Extra provenance for the merged `_meta`
            header. Must NOT contain `max_output_tokens` unless it is being
            used as a consistency assertion against the shards.

    Returns:
        Path: The merged trace path.
    """
    groups: list[RequestGroup] = []
    # worker_index -> (cap, header_dict) for provenance and cap consistency.
    shard_caps: dict[int, int] = {}
    shard_headers: list[dict[str, Any]] = []

    for worker_index in range(n_workers):
        path = shard_path(shard_dir, worker_index)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing shard for worker-{worker_index:02d} at {str(path)!r}. "
                f"Rerun that worker before merging."
            )
        header = read_trace_header(path)
        cap = header.get("max_output_tokens")
        if cap is None:
            raise ValueError(
                f"Shard for worker-{worker_index:02d} at {str(path)!r} is missing "
                f"max_output_tokens in its header."
            )
        shard_caps[worker_index] = int(cap)
        shard_headers.append(header)
        shard_groups = read_group_trace(path)
        if not shard_groups:
            raise ValueError(
                f"Shard for worker-{worker_index:02d} at {str(path)!r} holds no groups. "
                f"The write was probably interrupted. Rerun that worker before merging."
            )
        groups.extend(shard_groups)

    # Require every shard to agree on the generation cap.
    distinct_caps = {cap for cap in shard_caps.values()}
    if len(distinct_caps) > 1:
        by_cap: dict[int, list[int]] = {}
        for idx, cap in shard_caps.items():
            by_cap.setdefault(cap, []).append(idx)
        cap_summary = ", ".join(
            f"cap {cap!r} on workers {idxs}" for cap, idxs in sorted(by_cap.items())
        )
        raise ValueError(
            f"Shards disagree on max_output_tokens: {cap_summary}. "
            f"A mixed-cap merge is meaningless; re-collect with a consistent cap."
        )
    shards_cap = next(iter(distinct_caps))

    # If the caller passed max_output_tokens, treat it as a consistency assertion.
    caller_cap = metadata.get("max_output_tokens")
    if caller_cap is not None and int(caller_cap) != shards_cap:
        raise ValueError(
            f"Caller-supplied max_output_tokens {int(caller_cap)!r} disagrees with "
            f"the shards' cap {shards_cap!r}. Either correct the flag or omit it."
        )

    # Build provenance from fields that are identical across every shard.
    provenance: dict[str, Any] = {}
    for field in _PROVENANCE_FIELDS:
        if field in metadata:
            # Caller-supplied value takes precedence for non-cap provenance.
            provenance[field] = metadata[field]
            continue
        values = {header.get(field) for header in shard_headers}
        values.discard(None)
        if len(values) == 1:
            provenance[field] = next(iter(values))
        # Fields present in some shards but with differing values are omitted.

    # Copy any extra caller metadata that is not a per-worker field.
    _skip = {"max_output_tokens", "worker_index", "n_workers", "n_groups_requested"}
    for key, val in metadata.items():
        if key not in _skip and key not in provenance:
            provenance[key] = val

    provenance["n_shards"] = n_workers
    provenance["max_output_tokens"] = shards_cap

    groups.sort(key=lambda group: group.group_id)
    srl_logger.info(f"Merged {len(groups)} groups from {n_workers} shards.")
    return write_group_trace(groups, output_path, metadata=provenance)
