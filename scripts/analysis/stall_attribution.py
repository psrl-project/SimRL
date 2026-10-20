#!/usr/bin/env python3
"""Where does a psrl training step's wait go, and is it a scheduling problem?

Under `policy=psrl` a prompt group is reserved into buffer `v_traj +
max_staleness` the moment it is admitted (`policy/psrl.py:155`), and a buffer
only becomes consumable once every one of its entries is occupied
(`peek_batch:205`). The staleness invariant (`check_invariants:281`) then
guarantees nothing is ever dropped for being late. So a late trajectory does
not pay a staleness penalty the way it does under verl. It stalls the trainer,
in wall clock, one second for one second. That is a different objective, and it
is only worth building a scheduler for if the stall is both large and caused by
something a scheduler can move.

Three numbers decide that, and they have to be read together:

  stall / span        fraction of the run with the trainer holding no batch.
                      Small means there is nothing to win regardless of cause.
  stall attribution   at the instant the trainer is unblocked, was the deciding
                      request *decoding* or *waiting*? Decoding means a long
                      trajectory genuinely took that long, which length-aware
                      scheduling can attack. Waiting means it sat behind KV,
                      which is an admission problem. Fixing the wrong one of
                      these buys nothing, so the split comes before any design.
  makespan ratio      last completion over median completion inside one batch.
                      Near 1 says the step is bulk-bound and every request
                      matters equally. Large says one straggler sets the pace
                      and there is a specific thing to rescue.

The trainer timeline comes from the `train` stream, whose records carry `start`
and `end` per step, so an inter-step gap is a stall by construction rather than
by inference. Residency comes from the per-instance streams, which snapshot
`running_len` and `waiting_len` keyed by rid at every engine event. A request in
neither is still in the router queue and is reported as a third state, because
conflating "queued at the fleet" with "queued at an instance" would hide the
case where admission, not the engine, is holding the batch up.
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys
from collections import defaultdict


def load_stream(run: str, name: str) -> list[dict]:
    path = os.path.join(run, "logs", f"{name}.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def consumed_batches(run: str) -> dict[float, list[dict]]:
    """`train_start -> the requests that step consumed`.

    Keyed on the consume event's own timestamp rather than on `consume_version`.
    `_train` stamps `consume_version` with the version being trained *from* and
    then writes the train record after `_publish_weights` has incremented it, so
    the two numbering schemes are off by one and matching on them silently pairs
    each step with the next step's batch. The consume event is written at
    `self.t == train_start` (`simulator.py:884`), which is the same instant the
    train record reports as `start`, so this mapping cannot drift.
    """
    batches: dict[float, list[dict]] = defaultdict(list)
    for rec in load_stream(run, "request"):
        if rec["event"] == "consume":
            batches[rec["t"]].append(rec)
    return batches


def residency(run: str) -> dict[int, list[tuple[float, str]]]:
    """`rid -> [(t, state)]` with state in {running, waiting}, time-ordered.

    Samples are emitted per instance, and a request moves between instances on
    reroute, so the series are merged on time rather than kept per instance.
    """
    series: dict[int, list[tuple[float, str]]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(run, "logs", "instance", "*.jsonl"))):
        with open(path) as handle:
            for line in handle:
                rec = json.loads(line)
                t = rec["t"]
                for rid in rec["running_len"]:
                    series[int(rid)].append((t, "running"))
                for rid in rec["waiting_len"]:
                    series[int(rid)].append((t, "waiting"))
    for rid in series:
        series[rid].sort()
    return series


def state_split(
    samples: list[tuple[float, str]],
    lo: float,
    hi: float,
) -> tuple[float, float, float]:
    """Seconds spent running / waiting / absent over `[lo, hi)`.

    A sample is taken to describe the interval up to the next sample, which is
    how the engine emits them: one record per state change. Absent time is what
    neither queue accounted for, meaning the request was held in the router.
    """
    if hi <= lo:
        return 0.0, 0.0, 0.0
    inside = [(t, s) for t, s in samples if lo <= t < hi]
    before = [(t, s) for t, s in samples if t < lo]
    marks: list[tuple[float, str]] = []
    if before:
        marks.append((lo, before[-1][1]))
    marks.extend(inside)
    if not marks:
        return 0.0, 0.0, hi - lo
    run_s = wait_s = 0.0
    for i, (t, state) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else hi
        dt = min(end, hi) - max(t, lo)
        if dt <= 0:
            continue
        if state == "running":
            run_s += dt
        else:
            wait_s += dt
    return run_s, wait_s, max(0.0, (hi - lo) - run_s - wait_s)


def analyse(run: str) -> dict | None:
    steps = load_stream(run, "train")
    if not steps:
        return None
    batches = consumed_batches(run)
    res = residency(run)
    span = max(rec["end"] for rec in steps)

    # Gaps between steps are stalls. The lead-in before the first step is the
    # initial fill, which is not a steady-state stall and is kept separate so it
    # cannot flatter or inflate the fraction.
    fill = steps[0]["start"]
    gaps = [
        (steps[i - 1]["end"], steps[i]["start"])
        for i in range(1, len(steps))
        if steps[i]["start"] > steps[i - 1]["end"]
    ]
    stall = sum(hi - lo for lo, hi in gaps)
    train_s = sum(rec["dt"] for rec in steps)

    per_step = []
    for i, rec in enumerate(steps):
        done = [r for r in batches.get(rec["start"], []) if r.get("complete_time") is not None]
        if not done:
            continue
        times = sorted(r["complete_time"] for r in done)
        last = max(done, key=lambda r: r["complete_time"])
        # The window the trainer actually waited through before this step.
        lo = steps[i - 1]["end"] if i else 0.0
        run_s, wait_s, absent_s = state_split(res.get(last["rid"], []), lo, rec["start"])
        per_step.append(
            {
                "version": rec["version"],
                "n": len(done),
                "makespan": times[-1],
                "median": times[len(times) // 2],
                "span_of_batch": times[-1] - min(r["admit_time"] for r in done),
                "ratio": times[-1] / times[len(times) // 2] if times[len(times) // 2] else 0.0,
                # How close the batch's last completion is to the trainer
                # starting. Near zero means that completion is what unblocked
                # it. A large negative means the trainer was held by something
                # other than this batch finishing.
                "slack_to_start": rec["start"] - times[-1],
                "stall_before": rec["start"] - lo,
                "last_rid": last["rid"],
                "last_generated": last["generated"],
                "last_preemptions": last["num_preemptions"],
                "run_s": run_s,
                "wait_s": wait_s,
                "absent_s": absent_s,
            }
        )

    return {
        "span": span,
        "fill": fill,
        "stall": stall,
        "train_s": train_s,
        "steps": per_step,
    }


def main(runs: list[str]) -> None:
    print("Trainer timeline (seconds, and share of span):")
    print(
        f"{'arm':<16}{'span':>9}{'fill':>9}{'train':>9}{'stall':>9}"
        f"{'stall%':>8}{'train%':>8}{'fill%':>7}"
    )
    results = []
    for run in runs:
        arm = run.rstrip("/").split("/")[-1]
        r = analyse(run)
        if r is None:
            print(f"{arm:<16}  no train stream")
            continue
        results.append((arm, r))
        print(
            f"{arm:<16}{r['span']:>9.0f}{r['fill']:>9.0f}{r['train_s']:>9.0f}"
            f"{r['stall']:>9.0f}{100 * r['stall'] / r['span']:>7.1f}%"
            f"{100 * r['train_s'] / r['span']:>7.1f}%"
            f"{100 * r['fill'] / r['span']:>6.1f}%"
        )

    for arm, r in results:
        print(f"\n{arm}: per step. 'slack' is trainer start minus the batch's last completion.")
        print(
            f"{'ver':>4}{'n':>5}{'median':>9}{'makespan':>10}{'ratio':>7}"
            f"{'slack':>8}{'stall':>8}"
            f"{'  deciding request: rid':<24}{'gen':>7}{'preempt':>8}"
            f"{'running':>9}{'waiting':>9}{'router':>8}"
        )
        for s in r["steps"]:
            print(
                f"{s['version']:>4}{s['n']:>5}{s['median']:>9.0f}{s['makespan']:>10.0f}"
                f"{s['ratio']:>7.2f}{s['slack_to_start']:>8.0f}{s['stall_before']:>8.0f}"
                f"{s['last_rid']:>18}{'':<6}{s['last_generated']:>7}"
                f"{s['last_preemptions']:>8}"
                f"{s['run_s']:>9.0f}{s['wait_s']:>9.0f}{s['absent_s']:>8.0f}"
            )
        if r["steps"]:
            ratios = [s["ratio"] for s in r["steps"]]
            tot_run = sum(s["run_s"] for s in r["steps"])
            tot_wait = sum(s["wait_s"] for s in r["steps"])
            tot_absent = sum(s["absent_s"] for s in r["steps"])
            tot = tot_run + tot_wait + tot_absent
            print(
                f"  makespan/median: mean {statistics.mean(ratios):.2f}, "
                f"max {max(ratios):.2f}"
            )
            if tot > 0:
                print(
                    f"  deciding request spent {100 * tot_run / tot:.0f}% decoding, "
                    f"{100 * tot_wait / tot:.0f}% in an instance queue, "
                    f"{100 * tot_absent / tot:.0f}% in the router"
                )


if __name__ == "__main__":
    main(sys.argv[1:])
