#!/usr/bin/env python
"""Synthesise a heavy-tailed agentic-style trace for the SCG experiments.

**Why the aligned DAPO trace cannot test heterogeneous TP.** Its measured
length distribution has p99/p50 = 1.01 and CV = 0.26: 60.4% of responses sit
exactly on the 10240-token cap, so every trajectory is within 1% of every other
one. Dynamic heterogeneous parallelism concentrates *stragglers* onto scarce
low-batch capacity, and a distribution with no stragglers has nothing for it to
act on. Measuring C4 against that trace answers a question about the trace, not
about the mechanism.

**What this generates.** A LogNormal body with a Pareto tail, which is the shape
the paper's §6.2 predicts when the per-turn continuation hazard is decreasing
(the agentic regime: each solved sub-step reveals more work). The tail fraction
and heaviness are both explicit knobs so an ablation can sweep them and report
where the mechanism starts to pay, rather than asserting a single favourable
point.

Usage:
    python examples/workload/make_heavy_tail_trace.py \
        --out outputs/workload/heavy_tail/joint.jsonl \
        --n-groups 160 --group-size 8 --tail-frac 0.05 --tail-alpha 1.3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_lengths(
    n: int,
    body_median: int,
    body_sigma: float,
    tail_frac: float,
    tail_alpha: float,
    tail_scale_mult: float,
    cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw `n` response lengths from a LogNormal body plus a Pareto tail.

    Args:
        n: How many lengths to draw.
        body_median: Median of the LogNormal body, in tokens.
        body_sigma: Log-space sigma of the body. Controls bulk spread.
        tail_frac: Fraction of responses drawn from the Pareto tail instead.
        tail_alpha: Pareto shape. Smaller is heavier; below 2 the variance is
            infinite, which is the regime the paper calls anti-memoryful.
        tail_scale_mult: Pareto scale as a multiple of `body_median`, so the
            tail starts above the body rather than overlapping it.
        cap: Hard truncation, the serving stack's total length budget.
        rng: Seeded generator, so a run is reproducible.

    Returns:
        Integer lengths, every one in ``[1, cap]``.
    """
    n_tail = int(round(n * tail_frac))
    n_body = n - n_tail

    mu = np.log(body_median)
    body = rng.lognormal(mean=mu, sigma=body_sigma, size=n_body)

    # Pareto: scale * (1 + X) with X ~ pareto(alpha). Starts at `scale` and decays
    # with exponent alpha, so the mean diverges for alpha <= 1.
    scale = body_median * tail_scale_mult
    tail = scale * (1.0 + rng.pareto(tail_alpha, size=n_tail))

    lengths = np.concatenate([body, tail])
    rng.shuffle(lengths)
    return np.clip(np.rint(lengths), 1, cap).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-groups", type=int, default=160)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--body-median", type=int, default=1500)
    parser.add_argument("--body-sigma", type=float, default=0.5)
    parser.add_argument("--tail-frac", type=float, default=0.05)
    parser.add_argument("--tail-alpha", type=float, default=1.3)
    parser.add_argument("--tail-scale-mult", type=float, default=4.0)
    parser.add_argument("--cap", type=int, default=32768)
    parser.add_argument("--prompt-median", type=int, default=140)
    parser.add_argument("--prompt-sigma", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n_total = args.n_groups * args.group_size

    outputs = build_lengths(
        n_total,
        args.body_median,
        args.body_sigma,
        args.tail_frac,
        args.tail_alpha,
        args.tail_scale_mult,
        args.cap,
        rng,
    )
    prompts = np.clip(
        np.rint(rng.lognormal(np.log(args.prompt_median), args.prompt_sigma, args.n_groups)),
        16,
        4096,
    ).astype(np.int64)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    srt = np.sort(outputs)
    n = len(srt)
    stats = {
        "n": int(n),
        "min": int(srt[0]),
        "p50": int(srt[n // 2]),
        "p90": int(srt[9 * n // 10]),
        "p99": int(srt[99 * n // 100]),
        "max": int(srt[-1]),
        "mean": float(outputs.mean()),
        "cv": float(outputs.std() / outputs.mean()),
        "p99_over_p50": float(srt[99 * n // 100] / srt[n // 2]),
    }

    with args.out.open("w") as handle:
        meta = {
            "_meta": True,
            "n_groups": args.n_groups,
            "group_size": args.group_size,
            "max_output_tokens": args.cap,
            "provenance": "synthetic-heavy-tail",
            "generator": {
                "body_median": args.body_median,
                "body_sigma": args.body_sigma,
                "tail_frac": args.tail_frac,
                "tail_alpha": args.tail_alpha,
                "tail_scale_mult": args.tail_scale_mult,
                "cap": args.cap,
                "seed": args.seed,
            },
            "stats": stats,
        }
        handle.write(json.dumps(meta) + "\n")
        for gid in range(args.n_groups):
            lo = gid * args.group_size
            record = {
                "group_id": gid,
                "prompt_len": int(prompts[gid]),
                "output_lens": [int(x) for x in outputs[lo : lo + args.group_size]],
            }
            handle.write(json.dumps(record) + "\n")

    print(f"Wrote {args.out}")
    for key, value in stats.items():
        print(f"  {key:>14}: {value}")
    # The number that decides whether the experiment can show anything at all.
    print(
        f"\n  tail heaviness p99/p50 = {stats['p99_over_p50']:.2f} "
        f"(aligned DAPO trace: 1.01, i.e. no tail)"
    )


if __name__ == "__main__":
    main()
