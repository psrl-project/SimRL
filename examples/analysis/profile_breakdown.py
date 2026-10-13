"""
Break a cProfile dump of one simulation into phases and engine internals.

Prints the CLI phases, the self time of C++ builtins called from Python, and the
Python cost-model entry points the engine reaches through the trampoline.

Usage:
    python examples/analysis/profile_breakdown.py profile_cpp_cap160.prof
"""

import pstats
import sys

st = pstats.Stats(sys.argv[1])
S = st.stats


def find(fsub: str, fn: str):
    """Return the profile entry for function `fn` in a file matching `fsub`."""
    for (f, _ln, name), v in S.items():
        if fsub in f and name == fn:
            return v
    return None


print(f"profiled self-time total: {st.total_tt:.1f}s")
print("\n=== phases (cumulative) ===")
for fsub, fn in [
    ("cli/main.py", "run_simulate"),
    ("cli/main.py", "build_rollout_cost"),
    ("cli/main.py", "resolve_kv_blocks"),
    ("cli/main.py", "build_workload"),
    ("cli/main.py", "build_training_cost"),
    ("simulator/simulator.py", "run"),
    ("simulator/simulator.py", "_run_engine"),
    ("simulator/simulator.py", "_train"),
    ("visualizer/plots.py", "plot_all"),
]:
    v = find(fsub, fn)
    if v:
        print(f"  cum={v[3]:7.1f}s calls={v[1]:<9} {fn}")

print("\n=== builtins: C++ and pybind (self time) ===")
rows = [(v[2], v[3], v[1], name) for (f, _ln, name), v in S.items() if f == "~"]
for tt, ct, nc, name in sorted(rows, reverse=True)[:24]:
    print(f"  self={tt:7.2f}s cum={ct:7.2f}s calls={nc:>10}  {name[:95]}")

print("\n=== Python called from the engine ===")
for fsub, fn in [
    ("cost/rollout/composed.py", "step_time"),
    ("cost/rollout/composed.py", "decode_segment_time"),
    ("cost/rollout/composed.py", "decode_steps_within"),
    ("cost/rollout/composed.py", "saturated"),
    ("eviction/vllm_lru.py", "select"),
]:
    v = find(fsub, fn)
    if v:
        print(f"  cum={v[3]:7.1f}s calls={v[1]:<9} {fn}")
