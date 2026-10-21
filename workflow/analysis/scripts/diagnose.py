"""Turn a replayed run into a diagnosis: numbers, figures, and a report.

Runs in the same process as the replay so it can price bounds with the very
cost models the engine used, rather than rebuilding an approximation of them.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from srl.diagnosis.bounds import span_solo_all, span_tail_only
from srl.diagnosis.critical_path import decompose, steps
from srl.diagnosis.fleet import idle_windows, instance_imbalance
from srl.diagnosis.pricer import SoloPricer
from srl.diagnosis.timeline import request_phases
from srl.telemetry.store import JsonlStore
from srl.visualizer.diagnosis import DiagnosisPlotter


def _tp_by_instance(sim) -> dict[int, int]:
    return {inst.instance_id: inst.tp for inst in sim.instances}


def build(sim, trace_dir: Path) -> dict[str, Any]:
    """
    Compute the whole diagnosis for one replayed run.

    Args:
        sim: The finished `Simulator`, still holding its cost models.
        trace_dir (Path): Directory holding the replay's telemetry streams.

    Returns:
        dict: Checks, per-step attribution, bounds, fleet state, and the
            aggregate idle breakdown.

    Raises:
        RuntimeError: When a cross-source check fails, since every downstream
            number would then be built on a reconstruction known to disagree
            with the streams.
    """
    store = JsonlStore(trace_dir)
    timelines, checks = request_phases(store)
    bad = [check for check in checks if not check.ok]
    if bad:
        raise RuntimeError(
            "Cross-source checks failed, so the reconstruction disagrees with the streams: "
            + "; ".join(f"{check.name}: {check.detail}" for check in bad)
        )
    found = steps(store, timelines)
    tp_by_instance = _tp_by_instance(sim)
    pricer = SoloPricer({tp: sim.rollout_cost for tp in set(tp_by_instance.values())}, sim.cfg.block_size)
    horizon = sim.cfg.staleness_horizon or 0
    windows = idle_windows(store, found, timelines, horizon)

    parts = {rid: decompose(line, pricer, tp_by_instance) for rid, line in timelines.items()}
    idle_total = sum(step.idle for step in found)
    binding_shares: dict[str, float] = {}
    for step in found:
        piece = parts.get(step.binding_rid)
        if piece is None or idle_total <= 0:
            continue
        for kind, share in piece.shares().items():
            binding_shares[kind] = binding_shares.get(kind, 0.0) + share * step.idle / idle_total

    return {
        "checks": [dataclasses.asdict(check) for check in checks],
        "span": max((step.end for step in found), default=0.0),
        "train_seconds_total": sum(step.train_seconds for step in found),
        "idle_seconds_total": idle_total,
        "steps": [dataclasses.asdict(step) for step in found],
        "bounds": [
            dataclasses.asdict(span_solo_all(timelines, found, pricer, tp_by_instance, charge_reprefill=True)),
            dataclasses.asdict(span_solo_all(timelines, found, pricer, tp_by_instance, charge_reprefill=False)),
            dataclasses.asdict(span_tail_only(timelines, found, pricer, tp_by_instance)),
        ],
        "idle_breakdown_by_binding_phase": binding_shares,
        "idle_windows": [dataclasses.asdict(window) for window in windows],
        "instance_imbalance": [dataclasses.asdict(row) for row in instance_imbalance(store)],
        "binding_decompositions": {
            str(step.binding_rid): dataclasses.asdict(parts[step.binding_rid])
            for step in found
            if step.binding_rid in parts
        },
    }


def write(sim, trace_dir: Path, out_dir: Path, row: dict[str, Any], reason: str) -> dict[str, Any]:
    """
    Write `diagnosis.json` and the figures for one row, returning the diagnosis.

    Args:
        sim: The finished `Simulator`.
        trace_dir (Path): Replay telemetry directory.
        out_dir (Path): Where `diagnosis.json` and `figures/` go.
        row (dict): The original summary row.
        reason (str): Why this row was selected.

    Returns:
        dict: The diagnosis, with the row and reason attached.
    """
    diagnosis = build(sim, trace_dir)
    diagnosis["row_id"] = row["row_id"]
    diagnosis["selected_because"] = reason
    diagnosis["row"] = row
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnosis.json").write_text(json.dumps(diagnosis, indent=1, default=str))

    store = JsonlStore(trace_dir)
    timelines, _ = request_phases(store)
    found = steps(store, timelines)
    DiagnosisPlotter(
        store=store,
        timelines=timelines,
        steps_=found,
        windows=idle_windows(store, found, timelines, sim.cfg.staleness_horizon or 0),
        imbalance=instance_imbalance(store),
        out_dir=out_dir / "figures",
    ).plot_all()
    return diagnosis
