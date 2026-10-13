"""
Measure how much of a run steps one iteration at a time through pure decode.

The engine folds decode steps into one segment only when an instance's waiting
queue is empty, so under KV pressure every step becomes its own event. This reads
the `rollout_step` stream and counts, per instance, segment events, single-step
iterations that prefilled something, and blocked ones that prefilled nothing. It
also reports run lengths of consecutive blocked iterations, which bound how many
events folding them into segments would remove.

Usage:
    python examples/analysis/event_granularity.py <run_output_dir>
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

run = Path(sys.argv[1])
rows = [json.loads(line) for line in (run / "logs" / "rollout_step.jsonl").open() if line.strip()]

by_inst = defaultdict(list)
for r in rows:
    by_inst[r["instance_id"]].append(r)

total = Counter()
all_runs = []
for iid, steps in sorted(by_inst.items()):
    c = Counter()
    runs, cur = [], 0
    for s in steps:
        if s["phase"] == "decode":
            c["segment"] += 1
            c["segment_steps"] += s["steps"]
            if cur:
                runs.append(cur)
                cur = 0
        elif s.get("n_prefill", 0) == 0:
            c["blocked_iter"] += 1
            cur += 1
        else:
            c["prefill_iter"] += 1
            if cur:
                runs.append(cur)
                cur = 0
    if cur:
        runs.append(cur)
    total.update(c)
    all_runs += runs
    n_events = c["segment"] + c["blocked_iter"] + c["prefill_iter"]
    print(f"instance {iid}: events={n_events:,}  segment={c['segment']:,} "
          f"(covering {c['segment_steps']:,} steps)  "
          f"prefill_iter={c['prefill_iter']:,}  blocked_iter={c['blocked_iter']:,} "
          f"({c['blocked_iter'] / max(1, n_events) * 100:.1f}%)")

n_events = total["segment"] + total["blocked_iter"] + total["prefill_iter"]
print(f"\nall: events={n_events:,}  blocked single-step iterations="
      f"{total['blocked_iter']:,} ({total['blocked_iter'] / max(1, n_events) * 100:.1f}%)")
if all_runs:
    all_runs.sort()
    q = lambda p: all_runs[min(len(all_runs) - 1, int(p * len(all_runs)))]  # noqa: E731
    print(f"blocked runs: {len(all_runs):,}  median length={q(0.5)}  p90={q(0.9)}  "
          f"max={all_runs[-1]}")
    # Collapsing each blocked run into one segment leaves one event per run.
    after = n_events - total["blocked_iter"] + len(all_runs)
    print(f"events if each blocked run became one segment: {after:,} "
          f"({n_events / max(1, after):.1f}x fewer)")
