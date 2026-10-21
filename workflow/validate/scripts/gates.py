"""Evaluate the reporting gates on simulator results.

This module only judges results that were produced by a real simulation. It never derives a
performance number of its own, apart from the ratios the report needs.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

PASS, FAIL = "pass", "fail"


@dataclass(frozen=True)
class Gate:
    """One named pass/fail check on a single arm result."""

    name: str
    ok: bool
    detail: str


def _get(result: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return result.get(key, default)


def gate_completed(result: Mapping[str, Any], n_versions: int) -> Gate:
    """Check the run consumed every version and never livelocked."""
    done = _get(result, "versions_done")
    live = bool(_get(result, "livelocked", False))
    ok = done == n_versions and not live
    return Gate("run_completed", ok, f"versions_done={done} expected={n_versions} livelocked={live}")


def gate_consumed_groups(result: Mapping[str, Any], expected: int) -> Gate:
    """Check the scheduler consumed the full group budget."""
    groups = _get(result, "consumed_groups")
    ok = groups == expected
    return Gate("consumed_groups", ok, f"consumed_groups={groups} expected={expected}")


def gate_tokens_per_group(
    result: Mapping[str, Any],
    control: Mapping[str, Any],
    tolerance: float = 0.01,
) -> Gate:
    """Check the arm trained on the same length mix as the control at the same point."""
    arm_tpg = _get(result, "tokens_per_group")
    base_tpg = _get(control, "tokens_per_group")
    if not arm_tpg or not base_tpg:
        return Gate("tokens_per_group", False, "missing tokens_per_group")
    ratio = arm_tpg / base_tpg
    ok = abs(ratio - 1.0) <= tolerance
    return Gate("tokens_per_group", ok, f"ratio={ratio:.4f} tolerance={tolerance}")


def gate_differs(result: Mapping[str, Any], control: Mapping[str, Any], keys: Sequence[str]) -> Gate:
    """Check the treatment is not byte-identical to the control on any reported field."""
    same = [key for key in keys if _get(result, key) == _get(control, key)]
    ok = len(same) < len(keys)
    return Gate("treatment_differs", ok, f"identical_on={same or 'none'}")


def gate_same_binary(before: Mapping[str, str], after: Mapping[str, str]) -> Gate:
    """Check no simulator extension was rebuilt while the sweep ran."""
    if before == after:
        return Gate("same_binary", True, f"{len(before)} extensions unchanged")
    changed = sorted(set(before) | set(after))
    changed = [name for name in changed if before.get(name) != after.get(name)]
    return Gate("same_binary", False, f"changed={changed}")


SOFT_KEYS = ("span", "consumed_tokens", "mean_concurrency", "peak_concurrency")
"""Fields where a treatment matching the control signals an unwired flag rather than a result."""


def evaluate_point(
    result: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    n_versions: int,
    batch_size: int,
    tolerance: float = 0.01,
    soft_keys: Sequence[str] = SOFT_KEYS,
) -> list[Gate]:
    """
    Run every per-point gate.

    Args:
        result (Mapping[str, Any]): Treatment arm result for this point.
        control (Mapping[str, Any]): Control arm result for the same point.
        n_versions (int): Training versions the run was asked for.
        batch_size (int): Prompt groups per version.
        tolerance (float): Relative band for the consumption-set gate.
        soft_keys (Sequence[str]): Fields checked by `gate_differs`.

    Returns:
        list[Gate]: One entry per gate, in a fixed order.
    """
    return [
        gate_completed(result, n_versions),
        gate_consumed_groups(result, batch_size * n_versions),
        gate_tokens_per_group(result, control, tolerance),
        gate_differs(result, control, soft_keys),
    ]


def geometric_mean(values: Sequence[float]) -> float:
    """Return the geometric mean of positive `values`.

    Raises:
        ValueError: When `values` is empty or holds a non-positive entry.
    """
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"geometric_mean needs positive values, got {values!r}.")
    return math.exp(statistics.mean(math.log(value) for value in values))


def speedup(control: Mapping[str, Any], result: Mapping[str, Any]) -> float:
    """Return the control's span over the treatment's span."""
    return float(_get(control, "span")) / float(_get(result, "span"))


def summarize(points: Sequence[Mapping[str, Any]]) -> dict:
    """
    Aggregate gated point comparisons.

    Args:
        points (Sequence[Mapping[str, Any]]): Rows with `name`, `speedup`, and `gates`, where a
            gate row is `{"name": str, "ok": bool}`.

    Returns:
        dict: Gated geometric mean, the per-point speedups that contributed, and the failures.
    """
    gated = [row for row in points if all(gate["ok"] for gate in row["gates"])]
    failed = [row for row in points if row not in gated]
    return {
        "n_points": len(points),
        "n_gated": len(gated),
        "geomean_speedup": geometric_mean([row["speedup"] for row in gated]) if gated else float("nan"),
        "per_point": {row["name"]: row["speedup"] for row in gated},
        "failed": {row["name"]: [g["name"] for g in row["gates"] if not g["ok"]] for row in failed},
    }
