"""Record, on every summary row, enough to re-run exactly that row.

A sweep keeps telemetry off because the per-instance streams dominate disk and
most rows are never inspected. The analysis unit therefore re-runs the few rows
it wants with logging on, which is exact because the simulator is
deterministic, and gates the replay by comparing it against the original row.

Every experiment driver exposes:

    build(spec: dict, output_dir: Path | None) -> tuple[Simulator, int]

returning an unstarted `Simulator` and the instance count `collect.measure`
needs. `output_dir` of None means the driver leaves `log_telemetry` as the
sweep had it, and a path means it writes full telemetry there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def row_id(point: str, arm: str) -> str:
    """
    Return the stable identifier of one row.

    Args:
        point (str): Workload point name.
        arm (str): Arm name.

    Returns:
        str: `"<point>/<arm>"`, the key the analysis unit selects rows by.
    """
    return f"{point}/{arm}"


def replay_field(script: Path, spec: dict[str, Any], root: Path) -> dict[str, Any]:
    """
    Return the `replay` field for a row.

    Args:
        script (Path): The driver that can rebuild this row.
        spec (dict): Keyword state `build` needs to reproduce the row exactly.
        root (Path): Repository root, used to store a relative path.

    Returns:
        dict: `{"script": <repo-relative posix path>, "spec": spec}`.

    Raises:
        ValueError: When `script` is not inside `root`, since a row must stay
            replayable from a fresh checkout.
    """
    try:
        relative = Path(script).resolve().relative_to(Path(root).resolve())
    except ValueError as error:
        raise ValueError(f"Replay script {script} is outside the repository at {root}.") from error
    return {"script": relative.as_posix(), "spec": dict(spec)}


def attach(
    rows: list[dict[str, Any]],
    script: Path,
    specs: dict[str, dict[str, Any]],
    root: Path,
) -> list[dict[str, Any]]:
    """
    Add `row_id` and `replay` to every row.

    Args:
        rows (list[dict]): Rows carrying at least `point` and `arm`.
        script (Path): The driver that built them.
        specs (dict): `build` spec per row id.
        root (Path): Repository root.

    Returns:
        list[dict]: New rows, each with `row_id` and `replay` added.

    Raises:
        KeyError: When a row has no spec, since an unreplayable row cannot be
            diagnosed later and silently dropping it would hide that.
    """
    out = []
    for row in rows:
        key = row_id(row["point"], row["arm"])
        if key not in specs:
            raise KeyError(f"No replay spec for row {key}.")
        out.append({**row, "row_id": key, "replay": replay_field(script, specs[key], root)})
    return out
