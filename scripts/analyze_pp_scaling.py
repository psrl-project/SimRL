"""Test whether a pipeline-parallel cell can be derived instead of profiled.

The production Megatron sweep profiles every `(TP, PP)` pair, which is why the
PSRL baseline's TP2/PP3 mesh needed its own run before it could be simulated.
The hypothesis under test is that only TP needs measuring:

- a 1F1B stage holds `num_layers / pp` of the layers, so its per-microbatch cost
  should fall by `pp`.
- the critical path is `num_microbatches + pp - 1` slots, which is already
  analytic in `srl.cost.training.latency.pipeline_slots`.
- point-to-point activation sends between stages are small next to the layer
  compute, so ignoring them should cost little accuracy.

If all three hold, then dividing the token and per-microbatch terms by `pp`
should collapse every PP cell in a TP group onto one set of coefficients, and a
single fit should predict all of them.

Reads `outputs/profile/.../raw/megatron_bridge/*/` directly rather than the
collected JSONL so it can run while the sweep is still in progress.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Megatron prints one of these per iteration. The profiler's own JSONL is only
# written at collect time, which is too late to watch a running sweep.
_ITER_TIME = re.compile(r"elapsed time per iteration \(ms\): ([\d.]+)")

NUM_LAYERS = 36


def read_candidates(root: Path, warmup: int = 4) -> list[dict]:
    """Collect one record per successful candidate directory."""
    records = []
    for directory in sorted(root.glob("*/")):
        status_path = directory / "status.json"
        manifest_path = directory / "manifest.json"
        if not status_path.is_file() or not manifest_path.is_file():
            continue
        if json.loads(status_path.read_text()).get("status") != "success":
            continue
        manifest = json.loads(manifest_path.read_text())
        candidate = manifest["candidate"]
        times = [
            float(value) / 1000.0
            for value in _ITER_TIME.findall((directory / "run.log").read_text(errors="replace"))
        ]
        if len(times) <= warmup:
            continue
        parallelism = candidate["parallelism"]
        shape = candidate["sequence_shape"]
        records.append(
            {
                "tp": parallelism["tp"],
                "pp": parallelism["pp"],
                "dp": parallelism["dp"],
                "nmb": candidate["num_microbatches"],
                "seq": shape[0],
                # The median of the post-warmup tail, matching the fitter's own
                # `robust_step_time`, so the numbers here are comparable to it.
                "step": float(np.median(times[warmup:])),
                "sum_tokens": sum(shape) * candidate["num_microbatches"],
                "sum_sq": sum(value * value for value in shape) * candidate["num_microbatches"],
            }
        )
    return records


def design_row(record: dict, *, per_layer: bool) -> list[float]:
    """Build `[1, tokens, slots, sum_sq]`, optionally divided into per-layer work.

    `per_layer=True` divides the three workload terms by `pp`, which is the
    hypothesis: a stage owns `num_layers / pp` layers, so its cost per slot
    should scale down by exactly that factor while the slot count scales up.
    """
    pp, nmb = record["pp"], record["nmb"]
    slots = nmb + pp - 1
    bubble = pp - 1
    tokens = record["sum_tokens"] + bubble * record["seq"]
    sum_sq = record["sum_sq"] + bubble * record["seq"] ** 2
    scale = 1.0 / pp if per_layer else 1.0
    return [1.0, tokens * scale, slots * scale, sum_sq * scale]


def fit(records: list[dict], *, per_layer: bool) -> tuple[np.ndarray, float]:
    """Fit with relative residuals and report mean absolute percentage error."""
    matrix = np.array([design_row(record, per_layer=per_layer) for record in records])
    target = np.array([record["step"] for record in records])
    # Relative weighting, as in `fitting.scaled_nonnegative_fit`: this data spans
    # 0.48 s to 14 s, and an absolute least-squares fit lets the longest cells
    # dominate and drives the small coefficients negative.
    weights = 1.0 / np.maximum(np.abs(target), 1e-9)
    solution, *_ = np.linalg.lstsq(matrix * weights[:, None], target * weights, rcond=None)
    solution = np.maximum(solution, 0.0)
    predicted = matrix @ solution
    return solution, float(np.mean(np.abs(predicted - target) / target) * 100)


def main() -> int:
    root = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "outputs/profile/qwen3_8b/megatron_pp_scaling/raw/megatron_bridge"
    )
    records = read_candidates(root)
    if not records:
        print(f"No successful candidates under {root}.")
        return 1

    by_cell: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for record in records:
        by_cell[(record["tp"], record["pp"])].append(record)

    print(f"{len(records)} successful candidates\n")
    print(f"{'cell':>10}{'n':>4}{'nmb':>6}{'seq':>7}{'step (s)':>11}")
    for cell in sorted(by_cell):
        for record in sorted(by_cell[cell], key=lambda item: (item["nmb"], item["seq"])):
            print(
                f"{'tp%d/pp%d' % cell:>10}{len(by_cell[cell]):>4}"
                f"{record['nmb']:>6}{record['seq']:>7}{record['step']:>11.4f}"
            )

    present = sorted({record["pp"] for record in records})
    if len(present) < 2:
        print(f"\nOnly pp={present} measured so far. The comparison needs at least two.")
        return 0

    print("\n=== Per-cell fits (status quo: one fit per PP) ===")
    for cell in sorted(by_cell):
        if len(by_cell[cell]) < 4:
            print(f"  tp{cell[0]}/pp{cell[1]}: {len(by_cell[cell])} candidates, need 4")
            continue
        _, error = fit(by_cell[cell], per_layer=False)
        print(f"  tp{cell[0]}/pp{cell[1]}: in-sample MAPE {error:5.2f}%")

    print("\n=== Pooled fit across PP (the hypothesis: PP needs no profiling) ===")
    for scaling, label in ((True, "per-layer (cost / pp)"), (False, "unscaled")):
        solution, error = fit(records, per_layer=scaling)
        print(f"  {label:<24} MAPE {error:5.2f}%   coefficients {np.round(solution, 8).tolist()}")

    print("\n=== Leave-one-PP-out: predict a PP never measured ===")
    for held in present:
        train = [record for record in records if record["pp"] != held]
        test = [record for record in records if record["pp"] == held]
        if len(train) < 4 or not test:
            continue
        for scaling, label in ((True, "per-layer"), (False, "unscaled")):
            solution, _ = fit(train, per_layer=scaling)
            matrix = np.array([design_row(record, per_layer=scaling) for record in test])
            target = np.array([record["step"] for record in test])
            error = float(np.mean(np.abs(matrix @ solution - target) / target) * 100)
            print(f"  hold out pp={held}, fit on pp={sorted({r['pp'] for r in train})}, "
                  f"{label:<10} MAPE {error:6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
