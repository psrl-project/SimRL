#!/usr/bin/env python
"""Run every staleness-policy preset on the DAPO trace, on both engine backends.

For each registered preset, builds the preset's own default policy, runs the
simulator once with `backend="native"` and once with `backend="cpp"`, and
checks the two backends agree on consumed count and the full staleness
report before recording either one. No store is attached, so no per-run
telemetry is written anywhere, only this summary.

Usage:
    python examples/analysis/staleness_presets.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from srl.config.policies import PRESETS  # noqa: E402
from srl.cost.rollout import UnifiedRoofline  # noqa: E402
from srl.cost.training import FixedTrainingLatency, FixedTrainingMemory, TrainingCostModel  # noqa: E402
from srl.simulator import SimConfig, Simulator  # noqa: E402
from srl.workload.grouped import GroupedWorkload  # noqa: E402

TRACE_PATH = REPO / "outputs" / "workload" / "psrl_aligned" / "joint.jsonl"
TRACE_N_GROUPS = 0  # 0 reads every group in the trace (160, as of this trace file).
BATCH_SIZE = 8
# 160 groups / batch_size 8 = 20 training steps for exactly one pass with no
# wraparound, so every preset's report reflects the trace's full diversity
# rather than two cycles of a narrow slice.
N_VERSIONS = 20
OUT_PATH = REPO / "outputs" / "analysis" / "staleness_presets.json"


def _run(preset_name: str, backend: str):
    workload = GroupedWorkload.from_trace(TRACE_PATH, n_groups=TRACE_N_GROUPS)
    preset = PRESETS[preset_name]()
    policy = preset.build()
    cfg = SimConfig(
        batch_size=BATCH_SIZE,
        group_size=workload.group_size,
        staleness_horizon=preset.horizon,
        ingress_window_groups=preset.ingress_window_groups(BATCH_SIZE),
        n_versions=N_VERSIONS,
        workload=workload,
    )
    sim = Simulator(
        rollout_cost=UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7),
        training_cost=TrainingCostModel(FixedTrainingLatency(0.5), FixedTrainingMemory(0)),
        policy=policy,
        cfg=cfg,
        backend=backend,
    )
    result = sim.run()
    return result, policy


def main() -> None:
    summary: dict[str, dict] = {}
    for name in sorted(PRESETS):
        native_result, native_policy = _run(name, "native")
        cpp_result, _ = _run(name, "cpp")

        native_report = asdict(native_result.staleness_report())
        cpp_report = asdict(cpp_result.staleness_report())
        if native_result.n_consumed != cpp_result.n_consumed:
            raise AssertionError(
                f"{name}: consumed count disagrees between backends "
                f"(native={native_result.n_consumed}, cpp={cpp_result.n_consumed})."
            )
        if native_report != cpp_report:
            raise AssertionError(f"{name}: staleness report disagrees between backends.\n{native_report}\n{cpp_report}")

        summary[name] = {
            "versions_done": native_result.versions_done,
            "livelocked": native_result.livelocked,
            "n_consumed": native_result.n_consumed,
            "n_dropped": len(native_result.dropped),
            "staleness_report": native_report,
            "backends_agree": True,
        }
        print(f"{name}: consumed={native_result.n_consumed} strict={native_report['strict']} "
              f"max={native_report['max']} horizon={native_report['configured_horizon']}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
