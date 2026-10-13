"""
Summarise a `py-spy record --native --format raw` profile of one simulation.

Each raw line is one distinct stack, root first and joined by `;`, followed by its
sample count. Two views are printed. The inclusive view gives the share of samples
whose stack contains a frame matching each pattern. The exclusive view charges each
sample once, to the innermost frame matching a subsystem rule, so its rows sum to 100%.

Usage:
    python examples/analysis/pyspy_breakdown.py pyspy_cpp_exact.txt
"""

from __future__ import annotations

import re
import sys
from collections import Counter


def load(path: str) -> list[tuple[list[str], int]]:
    """Return each stack in the raw profile with its sample count."""
    stacks = []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if not line:
            continue
        head, _, count = line.rpartition(" ")
        if not count.isdigit():
            continue
        stacks.append((head.split(";"), int(count)))
    return stacks


# inclusive probes as (label, regex over a single frame)
INCLUSIVE = [
    ("whole run: run_simulate", r"run_simulate \("),
    ("  setup: build_rollout_cost (kernel tables)", r"build_rollout_cost \("),
    ("  setup: build_workload", r"build_workload \("),
    ("  setup: resolve_kv_blocks", r"resolve_kv_blocks \("),
    ("  Simulator.run", r"\brun \(dryrun/simulator/simulator\.py"),
    ("    _run_engine (event loop)", r"_run_engine \("),
    ("    _train (training step pricing)", r"_train \("),
    ("    _publish_weights", r"_publish_weights \("),
    ("  plot_all (end-of-run figures)", r"plot_all \("),
    ("engine C++: advance_one_event*", r"NativeInstance::advance_one_event"),
    ("  C++ save_snapshot (undo journal)", r"NativeInstance::save_snapshot"),
    ("  C++ recycle_log/clear (journal teardown)", r"recycle_log|UndoLog::clear"),
    ("  C++ engine_step", r"NativeInstance::engine_step"),
    ("  C++ decode_running_once", r"NativeInstance::decode_running_once"),
    ("  C++ KvCacheManager::allocate", r"KvCacheManager::allocate"),
    ("cost model: ComposedRollout.step_time", r"step_time \(dryrun/cost/rollout/composed\.py"),
    ("cost model: any dryrun/cost/ frame", r"\(dryrun/cost/"),
    ("pybind vector->list conversion", r"list_caster|cast_impl|type_caster<std::vector"),
    ("telemetry: SimTelemetry", r"\(dryrun/telemetry/"),
    ("router", r"\(dryrun/component/rollout/router/"),
    ("policy", r"\(dryrun/policy/"),
    ("import machinery", r"_find_and_load|exec_module|_call_with_frames_removed"),
]

# exclusive subsystems as (label, regex), where the innermost match wins
EXCLUSIVE = [
    ("C++ undo journal", r"save_snapshot|UndoLog|recycle_log|RequestSnapshot|FlatKeySet|HeldBlocks|capture_held"),
    ("C++ KV manager / pool", r"KvCacheManager|BlockPool|FreeBlockList|block_hashes"),
    ("C++ engine (scheduler/steps)", r"dryrun::engine::"),
    ("pybind conversion/glue", r"pybind11::"),
    ("Python cost model", r"\(dryrun/cost/"),
    ("Python telemetry", r"\(dryrun/telemetry/"),
    ("Python router", r"\(dryrun/component/rollout/router/"),
    ("Python policy", r"\(dryrun/policy/"),
    ("Python eviction", r"\(dryrun/component/rollout/engine/eviction/"),
    ("Python simulator loop", r"\(dryrun/simulator/"),
    ("plotting", r"\(dryrun/visualizer/|matplotlib"),
    ("startup: imports/hydra/config", r"_find_and_load|hydra|omegaconf|\(dryrun/cli/"),
    ("numpy/scipy", r"numpy|scipy"),
]


def main() -> None:
    stacks = load(sys.argv[1])
    total = sum(c for _, c in stacks)
    if not total:
        sys.exit("No samples.")
    print(f"samples: {total}")

    print("\n=== inclusive (share of all samples) ===")
    for label, pat in INCLUSIVE:
        rx = re.compile(pat)
        n = sum(c for frames, c in stacks if any(rx.search(f) for f in frames))
        print(f"  {n / total * 100:6.1f}%  {label}")

    print("\n=== exclusive by subsystem (innermost matching frame) ===")
    rules = [(label, re.compile(pat)) for label, pat in EXCLUSIVE]
    agg: Counter = Counter()
    for frames, c in stacks:
        owner = "other"
        for frame in reversed(frames):
            hit = next((label for label, rx in rules if rx.search(frame)), None)
            if hit:
                owner = hit
                break
        agg[owner] += c
    for label, n in agg.most_common():
        print(f"  {n / total * 100:6.1f}%  {label}")

    print("\n=== hottest leaf frames ===")
    leaf: Counter = Counter()
    for frames, c in stacks:
        leaf[frames[-1][:110]] += c
    for frame, n in leaf.most_common(15):
        print(f"  {n / total * 100:6.1f}%  {frame}")


if __name__ == "__main__":
    main()
