"""Check that a training artifact covers the requested topology.

Called by conf/training.sh before the simulation starts, so mismatches fail
with a clear message rather than a bare KeyError mid-run.

Usage:
    python check_training_artifact.py <artifact> <backend> <tp> <pp> [<shard_size>]

Exit codes: 0 = OK, 1 = validation error.
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    path, backend, tp, pp = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    shard = sys.argv[5] if len(sys.argv) > 5 else ""

    entries = [
        e for e in json.load(open(path))["entries"]
        if e["selector"].get("component") == "latency"
    ]

    if backend == "fsdp":
        available = sorted({
            int(e["selector"]["shard_size"])
            for e in entries
            if "shard_size" in e["selector"]
        })
        if shard and int(shard) not in available:
            print(
                f"SHARD_SIZE={shard} is not fitted. Available shard sizes: {available}.",
                file=sys.stderr,
            )
            return 1
    else:
        # Only TP is a measured cell; PP is derived, since the latency features are
        # per-layer and the slot count is analytic. Predicting an unprofiled PP costs
        # 2.2-4.6%.
        available = sorted({int(e["parallelism"]["tp"]) for e in entries})
        if int(tp) not in available:
            degrees = ", ".join(f"tp={t}" for t in available)
            print(
                f"TRAINING_TP={tp} is not a fitted cell. Available: {degrees}.\n"
                f"TP changes how each layer is sharded, so it has to be measured; "
                f"PP is derived and needs no cell of its own.",
                file=sys.stderr,
            )
            return 1
        profiled = sorted({
            int(d)
            for e in entries
            for d in e.get("diagnostics", {}).get("profiled_pipeline_degrees", [])
        })
        if profiled and int(pp) not in profiled:
            print(
                f"note: PP={pp} was not profiled (measured: {profiled}); "
                f"its latency is derived from the per-layer form.",
                file=sys.stderr,
            )

    # Report the fitted token domain up front. `tokens` is the LARGEST microbatch,
    # not the mean, so one long sequence sets it. A step exceeding the domain raises
    # unless EXTRAPOLATE=[tokens] is set; the fit is linear while attention is
    # quadratic, so a stretched tail is understated.
    ranges = [e.get("feature_ranges", {}).get("tokens") for e in entries]
    spans = [r for r in ranges if r]
    if spans:
        lo = min(r[0] for r in spans)
        hi = max(r[1] for r in spans)
        print(
            f"fitted per-microbatch token domain: [{lo:.0f}, {hi:.0f}]\n"
            f"  (exceeding {hi:.0f} tokens in the longest microbatch raises "
            f"unless EXTRAPOLATE=[tokens] is set)",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
