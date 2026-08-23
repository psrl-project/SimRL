"""General-purpose JSONL stream store for simulation telemetry.

A `JsonlStore` maps a stream name (e.g. "train", "instance/0", "rollout_step")
to a `.jsonl` file under a root directory. Within one store lifetime writers
append one JSON object per call and the store never holds history in memory;
readers stream records back with `iter`. Pass `overwrite=True` to wipe that
root first, so a new simulation does not concatenate onto a previous run.
This lets the simulator log richly (per-step, per-instance telemetry) without
accumulating large lists in `SimResult`, and lets visualization run standalone
against a previous run's logs without re-simulating.

Records have no fixed schema: callers pass whatever keyword fields matter for
that stream. Adding a field or a new stream never requires touching this
module.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from dataclasses import asdict, is_dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, TextIO


def _to_jsonable(value: Any) -> Any:
    """Recursively convert a value into something `json.dumps` can handle."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


class JsonlStore:
    """Append/iterate named JSONL streams rooted at a directory.

    Stream names may contain `/` to group related streams into
    subdirectories, e.g. `instance/0` maps to `root/instance/0.jsonl`.

    Args:
        root: Directory that holds the `.jsonl` stream files.
        overwrite: If True, delete `root` (when it already exists) before
            creating it, so this run's logs replace any previous run. Plotting
            and other read-only reopenings should leave this False.
    """

    def __init__(self, root: str | Path, *, overwrite: bool = False):
        self.root = Path(root)
        if overwrite and self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, TextIO] = {}

    def _path(self, stream: str) -> Path:
        return self.root / f"{stream}.jsonl"

    def _file(self, stream: str) -> TextIO:
        fp = self._files.get(stream)
        if fp is None:
            path = self._path(stream)
            path.parent.mkdir(parents=True, exist_ok=True)
            fp = path.open("a", encoding="utf-8")
            self._files[stream] = fp
        return fp

    def write(self, stream: str, **record: Any) -> None:
        """Append one JSON record to `stream`, flushing immediately.

        No schema is enforced: `record` is whatever keyword fields the
        caller wants persisted for this stream.
        """
        fp = self._file(stream)
        fp.write(json.dumps(_to_jsonable(record)) + "\n")
        fp.flush()

    def iter(self, stream: str) -> Iterator[dict[str, Any]]:
        """Yield records from `stream` in append order.

        A stream that was never written (missing file) yields nothing.
        """
        path = self._path(stream)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def list_streams(self) -> list[str]:
        """Return all stream names discovered on disk, relative to root."""
        return sorted(
            p.relative_to(self.root).as_posix()[: -len(".jsonl")] for p in self.root.rglob("*.jsonl")
        )

    def close(self) -> None:
        """Close all open write handles. Safe to call multiple times."""
        for fp in self._files.values():
            fp.close()
        self._files.clear()
