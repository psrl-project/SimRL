"""Validated JSONL input and output for profiling records."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .schema import ProfileRecord, ProfileValidationError


def canonical_record_json(record: ProfileRecord) -> str:
    """Serialize a record deterministically for storage and hashing."""
    return json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def iter_records(
    path: str | Path,
    *,
    record_type: str | None = None,
) -> Iterator[ProfileRecord]:
    """Yield validated records from a JSONL file."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
                record = ProfileRecord.from_dict(value)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ProfileValidationError(
                    f"Invalid profiling record at {source}:{line_number}: {exc}"
                ) from exc
            if record_type is None or record.record_type == record_type:
                yield record


def read_records(
    path: str | Path,
    *,
    record_type: str | None = None,
) -> list[ProfileRecord]:
    """Read all matching records from a JSONL file."""
    return list(iter_records(path, record_type=record_type))


def write_records(
    path: str | Path,
    records: Iterable[ProfileRecord],
    *,
    append: bool = False,
) -> Path:
    """Validate and write records, using atomic replacement for a new file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(records)
    lines = [canonical_record_json(record) for record in materialized]

    if append:
        with destination.open("a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line)
                handle.write("\n")
        return destination

    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line)
                handle.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def profile_digest(records_or_path: Iterable[ProfileRecord] | str | Path) -> str:
    """Compute a stable SHA-256 digest over canonical records."""
    if isinstance(records_or_path, (str, Path)):
        records: Iterable[ProfileRecord] = iter_records(records_or_path)
    else:
        records = records_or_path
    digest = hashlib.sha256()
    for record in records:
        digest.update(canonical_record_json(record).encode("utf-8"))
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def load_json(path: str | Path) -> dict[str, Any] | list[Any]:
    """Load one raw third-party JSON artifact without modifying it."""
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)
