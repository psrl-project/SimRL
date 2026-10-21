"""Re-run one sweep row with full telemetry, and refuse to proceed if it differs.

The simulator is deterministic, so a replay of the same spec reproduces the
row exactly. That makes the comparison a real gate rather than a formality: a
mismatch means the spec does not capture everything the original run depended
on, and any diagnosis built on it would describe a different run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

GATED_KEYS = (
    "span",
    "consumed_tokens",
    "consumed_groups",
    "tokens_per_group",
    "mean_concurrency",
    "peak_concurrency",
    "preemptions",
    "versions_done",
    "dropped",
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(row: dict[str, Any], root: Path, trace_dir: Path):
    """
    Replay `row` with every telemetry stream written to `trace_dir`.

    Args:
        row (dict): A summary row carrying `replay`.
        root (Path): Repository root.
        trace_dir (Path): Where the replay writes its streams. Use a local
            disk: the shared filesystem runs near its quota.

    Returns:
        tuple: The finished simulator, its `SimResult`, and the instance count.
    """
    script = root / row["replay"]["script"]
    driver = _load(f"replay_driver_{abs(hash(script))}", script)
    trace_dir.mkdir(parents=True, exist_ok=True)
    sim, n_instances = driver.build(row["replay"]["spec"], trace_dir)
    result = sim.run()
    sim.telemetry.close()
    return sim, result, n_instances


def fidelity(row: dict[str, Any], measured: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """
    Compare a replay against the row it came from.

    Args:
        row (dict): The original summary row.
        measured (dict): `collect.measure` output for the replay.

    Returns:
        list[tuple[str, bool, str]]: One `(key, ok, detail)` per gated field.
            Floats must agree to 1e-9 relative, counts exactly. Any False
            means stop: the diagnosis would describe a different run.
    """
    out = []
    for key in GATED_KEYS:
        if key not in row or key not in measured:
            out.append((key, False, f"missing from {'row' if key not in row else 'replay'}"))
            continue
        want, got = row[key], measured[key]
        if isinstance(want, float) or isinstance(got, float):
            ok = abs(got - want) <= 1e-9 * max(1.0, abs(want))
        else:
            ok = got == want
        out.append((key, ok, f"row={want!r} replay={got!r}"))
    return out
