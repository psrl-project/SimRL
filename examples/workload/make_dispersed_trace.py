#!/usr/bin/env python
"""Synthesise a trace whose length dispersion is a controlled variable.

A scheduling study needs to move the spread of response lengths without moving
their mean, because the mean sets the work a step contains and a regime change
that also changes the work cannot be read as a change in scheduling headroom.
This draws log-lengths from a normal, solves the location so the mean survives
truncation at `cap`, and exposes the two quantities that decide whether length
knowledge is worth anything:

* `--cv` is the coefficient of variation before truncation. Larger means a
  heavier right tail.
* `--rho` is the share of log-length variance that belongs to the prompt rather
  than to the sample. A prompt that is hard makes all of its responses long, and
  a group completes only when its slowest member does, so the group is the unit
  that is early or late.

Usage:
    python examples/workload/make_dispersed_trace.py \
        --out outputs/workload/dispersed/cv1.0.jsonl \
        --n-groups 400 --mean 8000 --cv 1.0 --rho 0.6 --cap 32768
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MIN_LEN = 64


def draw(
    n_groups: int,
    group_size: int,
    mean: float,
    cv: float,
    rho: float,
    cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw `(n_groups, group_size)` lengths with the requested truncated mean."""
    sigma2 = np.log1p(cv * cv)
    shared = rng.standard_normal((n_groups, 1))
    own = rng.standard_normal((n_groups, group_size))
    z = np.sqrt(rho * sigma2) * shared + np.sqrt((1.0 - rho) * sigma2) * own

    def truncated_mean(mu: float) -> float:
        return float(np.clip(np.exp(mu + z), MIN_LEN, cap).mean())

    lo, hi = np.log(MIN_LEN), np.log(cap)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if truncated_mean(mid) < mean:
            lo = mid
        else:
            hi = mid
    lengths = np.clip(np.exp(0.5 * (lo + hi) + z), MIN_LEN, cap)
    return np.rint(lengths).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-groups", type=int, default=400)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--mean", type=float, default=8000.0)
    parser.add_argument("--cv", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--cap", type=int, default=32768)
    parser.add_argument("--prompt-median", type=int, default=140)
    parser.add_argument("--prompt-sigma", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    lengths = draw(
        args.n_groups, args.group_size, args.mean, args.cv, args.rho, args.cap, rng
    )
    prompts = np.clip(
        np.rint(rng.lognormal(np.log(args.prompt_median), args.prompt_sigma, args.n_groups)),
        16,
        4096,
    ).astype(np.int64)

    flat = np.sort(lengths.ravel())
    n = flat.size
    stats = {
        "n": int(n),
        "mean": float(lengths.mean()),
        "p50": int(flat[n // 2]),
        "p90": int(flat[9 * n // 10]),
        "p99": int(flat[99 * n // 100]),
        "max": int(flat[-1]),
        "cv": float(lengths.std() / lengths.mean()),
        "group_max_over_mean": float(lengths.max(axis=1).mean() / lengths.mean()),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        handle.write(
            json.dumps(
                {
                    "_meta": True,
                    "n_groups": args.n_groups,
                    "group_size": args.group_size,
                    "max_output_tokens": args.cap,
                    "provenance": "synthetic-dispersed",
                    "generator": {
                        "mean": args.mean,
                        "cv": args.cv,
                        "rho": args.rho,
                        "cap": args.cap,
                        "seed": args.seed,
                    },
                    "stats": stats,
                }
            )
            + "\n"
        )
        for gid in range(args.n_groups):
            handle.write(
                json.dumps(
                    {
                        "group_id": gid,
                        "prompt_len": int(prompts[gid]),
                        "output_lens": [int(x) for x in lengths[gid]],
                    }
                )
                + "\n"
            )
    print(f"Wrote {args.out}")
    for key, value in stats.items():
        print(f"  {key:>20}: {value:.3f}" if isinstance(value, float) else f"  {key:>20}: {value}")


if __name__ == "__main__":
    main()
