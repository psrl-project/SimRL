#!/usr/bin/env python
"""
Extend the measured DAPO trace past its truncation cap.

The aligned DAPO trace is right-censored at its generation cap, so it has almost no
tail, and the slowest trajectory in a batch is what separates a strict staleness
bound from a loose one. This keeps every uncensored observation as measured and
redraws each censored one from a Pareto tail conditioned on exceeding the old cap.
The shape is fitted with Hill's estimator to the top `--fit-quantile` of the
uncensored body, or set with `--tail-alpha`. Draws are clipped at `--cap`, and the
clipped fraction is reported because a large one means the tail is still censored.

Usage:
    python examples/analysis/extend_dapo_trace.py \
        --source outputs/workload/psrl_aligned/joint.jsonl \
        --out outputs/workload/dapo_32k/joint.jsonl \
        --cap 32768 --repeat 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fit_pareto_tail(lengths: np.ndarray, threshold: float) -> float:
    """
    Return the Pareto shape `alpha` fitted to `lengths` above `threshold`.

    Uses Hill's estimator, `alpha = n / sum(log(x_i / threshold))` over `x_i > threshold`.

    Args:
        lengths (np.ndarray): Uncensored observations.
        threshold (float): Lower bound of the region treated as the tail.

    Returns:
        float: The shape parameter. Smaller means heavier.

    Raises:
        ValueError: If fewer than 10 observations exceed the threshold.
    """
    tail = lengths[lengths > threshold]
    if tail.size < 10:
        raise ValueError(
            f"Only {tail.size} uncensored observations above {threshold:.0f}, "
            f"too few to fit a tail. Lower --fit-quantile."
        )
    return float(tail.size / np.sum(np.log(tail / threshold)))


def draw_beyond_cap(
    n: int,
    cap: int,
    alpha: float,
    new_cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw `n` lengths from a Pareto(alpha) conditioned on exceeding `cap`.

    A Pareto conditioned on exceeding a threshold is again Pareto with that threshold
    as its scale, so `cap * U^(-1/alpha)` with `U ~ Uniform(0, 1)` is exact.

    Args:
        n (int): How many draws.
        cap (int): The old generation cap, i.e. the censoring threshold.
        alpha (float): Pareto shape.
        new_cap (int): Draws are clipped here.
        rng (np.random.Generator): Seeded generator.

    Returns:
        np.ndarray: Integer lengths in `[cap, new_cap]`.
    """
    draws = cap * rng.uniform(size=n) ** (-1.0 / alpha)
    return np.clip(draws, cap, new_cap).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--old-cap",
        type=int,
        default=0,
        help="Censoring threshold. 0 reads it from the source's own metadata.",
    )
    parser.add_argument("--cap", type=int, default=32768, help="New generation cap.")
    parser.add_argument(
        "--fit-quantile",
        type=float,
        default=0.70,
        help="Quantile of the uncensored body above which the tail is fitted.",
    )
    parser.add_argument(
        "--tail-alpha",
        type=float,
        default=0.0,
        help=(
            "Override the fitted Pareto shape. 0 uses the fit. Smaller is heavier. "
            "The fit is reported either way. DAPO's own tail decays faster than a "
            "power law, so a heavy tail has to be entered as an explicit assumption."
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Tile the prompt set this many times, so policies have more groups than "
            "a run consumes to choose from."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
    meta = next((r for r in rows if r.get("_meta")), None)
    data = [r for r in rows if not r.get("_meta")]
    if not data:
        raise SystemExit(f"{args.source} holds no trace rows.")

    old_cap = args.old_cap or int((meta or {}).get("max_output_tokens", 0))
    if not old_cap:
        raise SystemExit("Could not determine the old cap. Pass --old-cap.")
    if args.cap <= old_cap:
        raise SystemExit(f"--cap {args.cap} must exceed the old cap {old_cap}.")

    measured = np.array([o for r in data for o in r["output_lens"]], dtype=np.float64)
    uncensored = measured[measured < old_cap]
    threshold = float(np.quantile(uncensored, args.fit_quantile))
    fitted_alpha = fit_pareto_tail(uncensored, threshold)
    alpha = args.tail_alpha if args.tail_alpha > 0 else fitted_alpha

    rng = np.random.default_rng(args.seed)
    group_size = len(data[0]["output_lens"])
    out_rows: list[dict] = []
    n_extended = 0
    n_clipped = 0

    for copy in range(args.repeat):
        for row in data:
            lengths: list[int] = []
            for value in row["output_lens"]:
                if value < old_cap:
                    lengths.append(int(value))
                    continue
                # Censored: its true length is somewhere beyond the cap.
                drawn = int(draw_beyond_cap(1, old_cap, alpha, args.cap, rng)[0])
                lengths.append(drawn)
                n_extended += 1
                if drawn >= args.cap:
                    n_clipped += 1
            out_rows.append(
                {
                    "group_id": copy * len(data) + row["group_id"],
                    "prompt_len": int(row["prompt_len"]),
                    "output_lens": lengths,
                }
            )

    final = np.array([o for r in out_rows for o in r["output_lens"]], dtype=np.float64)
    q = lambda x: float(np.quantile(final, x))  # noqa: E731
    stats = {
        "n": int(final.size),
        "n_groups": len(out_rows),
        "group_size": group_size,
        "p50": q(0.50),
        "p90": q(0.90),
        "p99": q(0.99),
        "max": float(final.max()),
        "mean": float(final.mean()),
        "cv": float(final.std() / final.mean()),
        "p99_over_p50": q(0.99) / q(0.50),
        "extended": n_extended,
        "clipped_at_new_cap": n_clipped,
        "alpha_used": alpha,
        "fitted_alpha": fitted_alpha,
        "alpha_overridden": args.tail_alpha > 0,
        "fit_threshold": threshold,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "_meta": True,
                    "n_groups": len(out_rows),
                    "group_size": group_size,
                    "max_output_tokens": args.cap,
                    "provenance": "dapo-measured-with-pareto-extended-tail",
                    "source": str(args.source),
                    "generator": {
                        "old_cap": old_cap,
                        "cap": args.cap,
                        "fit_quantile": args.fit_quantile,
                        "alpha_used": alpha,
                        "fitted_alpha": fitted_alpha,
                        "repeat": args.repeat,
                        "seed": args.seed,
                    },
                    "stats": stats,
                }
            )
            + "\n"
        )
        for row in out_rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Wrote {args.out}")
    print(f"  fitted Pareto alpha = {fitted_alpha:.3f} above {threshold:.0f} tokens")
    if args.tail_alpha > 0:
        print(f"  USING OVERRIDE alpha = {alpha:.3f} (assumption, not measurement)")
    print(f"  extended {n_extended} censored responses ({n_extended / final.size * 100:.1f}%)")
    pct = n_clipped / max(n_extended, 1) * 100
    print(f"  clipped at the new cap: {n_clipped} ({pct:.1f}% of the extended ones)")
    if pct > 25.0:
        print("  WARNING: a quarter of the tail is still censored, raise --cap.")
    for key in ("n_groups", "p50", "p90", "p99", "max", "cv", "p99_over_p50"):
        print(f"  {key:>14}: {stats[key]}")


if __name__ == "__main__":
    main()
