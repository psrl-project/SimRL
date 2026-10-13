#!/usr/bin/env python
"""Build a joint trace on which the router active cap moves the span the most.

The cap can only change the span through the decode rate it allows, so the
trace removes every other thing that sets the span:

- Equal response lengths, so no group is held open by one straggler whose
  latency grows with batch size rather than shrinking.
- A short prompt, which adds KV depth and training tokens but no decode work.
- A length short enough that the best cap fits in the KV cache without
  preemption, and long enough that rollout still outlasts training at that
  cap. Below that point the span is training-bound and the cap stops mattering.

`--cv` adds lognormal spread around the mean to test how quickly that breaks.

Usage:
    python examples/analysis/make_cap_sensitive_trace.py \
        --out outputs/workload/cap_sensitive/joint.jsonl
"""

from __future__ import annotations

import argparse

import numpy as np

from dryrun.workload.trace import RequestGroup, write_group_trace


def draw_lengths(n: int, mean: int, cv: float, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Return `n` response lengths with mean `mean` and coefficient of variation `cv`.

    Raises:
        ValueError: If `cv` is negative or `mean` exceeds `cap`.
    """
    if cv < 0:
        raise ValueError(f"--cv must be non-negative, got {cv!r}.")
    if mean > cap:
        raise ValueError(f"--output-len {mean!r} exceeds --cap {cap!r}.")
    if cv == 0:
        return np.full(n, mean, dtype=np.int64)
    sigma2 = np.log1p(cv**2)
    draws = rng.lognormal(np.log(mean) - sigma2 / 2, np.sqrt(sigma2), n)
    return np.clip(np.round(draws), 1, cap).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--output-len", type=int, default=3072)
    parser.add_argument("--cv", type=float, default=0.0)
    parser.add_argument("--prompt-len", type=int, default=32)
    # Five times the 128 groups a 4-version, 32-group run consumes, matching
    # the oversupply of the tail-restored DAPO trace.
    parser.add_argument("--groups", type=int, default=640)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--cap", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    lengths = draw_lengths(args.groups * args.group_size, args.output_len, args.cv, args.cap, rng)
    groups = [
        RequestGroup(
            group_id=g,
            prompt_len=args.prompt_len,
            output_lens=tuple(int(x) for x in lengths[g * args.group_size : (g + 1) * args.group_size]),
        )
        for g in range(args.groups)
    ]
    path = write_group_trace(
        groups,
        args.out,
        {
            "max_output_tokens": args.cap,
            "provenance": "synthetic-cap-sensitive",
            "generator": {k: v for k, v in vars(args).items() if k != "out"},
        },
    )
    print(f"Wrote {len(groups)} groups to {path}: mean={lengths.mean():.0f}, max={lengths.max()}.")


if __name__ == "__main__":
    main()
