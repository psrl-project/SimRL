"""Choose which sweep rows are worth a full-telemetry replay.

A sweep has many rows and a replay costs a full re-run, so the selection is
fixed rather than ad hoc: the best control row at every point, plus the
treatment rows that are clearly bad, clearly good, and typical. Those four
answer different questions, and picking them by rule keeps one run's choice
comparable with another's.
"""

from __future__ import annotations

import statistics
from typing import Any


def _gated(summary: dict[str, Any]) -> list[dict[str, Any]]:
    failed = set(summary.get("summary", {}).get("failed", {}))
    rows = summary["rows"]
    for row in rows:
        if "replay" not in row:
            raise ValueError(f"Row {row.get('row_id', row)} has no replay field, so it cannot be replayed.")
    return [row for row in rows if row.get("row_id") not in failed]


def pick(summary: dict[str, Any], control_arm: str, limit: int = 6) -> list[dict[str, Any]]:
    """
    Return the rows to replay, each with the reason it was chosen.

    Args:
        summary (dict): A `validate` `summary.json`, already loaded.
        control_arm (str): Name of the control arm.
        limit (int): Hard cap on replays, since each one is a full re-run.

    Returns:
        list[dict]: `{"row_id": str, "reason": str}` entries, deduplicated and
            truncated to `limit`.

    Raises:
        ValueError: When a row carries no `replay` field.
    """
    rows = _gated(summary)
    treatments = [row for row in rows if row["arm"] != control_arm]
    controls = [row for row in rows if row["arm"] == control_arm]

    chosen: dict[str, str] = {}
    best_control: dict[str, dict[str, Any]] = {}
    for row in controls:
        current = best_control.get(row["point"])
        if current is None or row["span"] < current["span"]:
            best_control[row["point"]] = row
    for row in best_control.values():
        chosen.setdefault(row["row_id"], f"best control span at point {row['point']}")

    if treatments:
        ranked = sorted(treatments, key=lambda row: row["speedup"])
        chosen.setdefault(ranked[0]["row_id"], "lowest gated speedup, the clearest loss")
        chosen.setdefault(ranked[-1]["row_id"], "highest gated speedup, the clearest win")
        median = statistics.median_low(row["speedup"] for row in ranked)
        typical = next(row for row in ranked if row["speedup"] == median)
        chosen.setdefault(typical["row_id"], "median gated speedup, the representative case")

    return [{"row_id": key, "reason": reason} for key, reason in list(chosen.items())[:limit]]
