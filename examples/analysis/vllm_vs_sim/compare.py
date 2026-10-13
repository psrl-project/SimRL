"""
Compare a simulated instance run against a real vLLM engine run of the same workload spec.

Reports the span, the per-request finish-time error, preemption counts, and how closely the
running count, waiting count and KV occupancy track each other over time. Writes one figure
per workload so the shapes can be judged by eye.

Usage:
    python examples/analysis/vllm_vs_sim/compare.py --real outputs/vllm_vs_sim/real/grpo_small \
        --sim outputs/vllm_vs_sim/sim/grpo_small --out outputs/vllm_vs_sim/plots/grpo_small.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _resample(t: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Hold each sample until the next one."""
    idx = np.searchsorted(t, grid, side="right") - 1
    return values[np.clip(idx, 0, None)]


def _interpolate_used(series: dict[str, np.ndarray], grid: np.ndarray) -> np.ndarray:
    """Occupancy grows linearly through a step, so interpolate from one event's end state to the next's."""
    t = series["t"]
    out = np.empty(len(grid))
    idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 2)
    t0, t1 = t[idx], t[idx + 1]
    span = np.maximum(t1 - t0, 1e-9)
    frac = np.clip((grid - t0) / span, 0.0, 1.0)
    out[:] = series["used"][idx] + frac * (series["used_pre"][idx + 1] - series["used"][idx])
    return out


def _series(run: Path, blocks: int, real: bool) -> dict[str, np.ndarray]:
    rows = _rows(run / "steps.jsonl")
    t = np.array([r["t"] for r in rows])
    if real:
        used = (blocks - 1) - np.array([r["free_blocks"] for r in rows])
        used_pre = used
    else:
        used = np.array([r["blocks_used"] for r in rows])
        used_pre = np.array([r["blocks_used_pre"] for r in rows])
    return {
        "t": t,
        "running": np.array([r["running"] for r in rows]),
        "waiting": np.array([r["waiting"] for r in rows]),
        "used": used / blocks,
        "used_pre": used_pre / blocks,
    }


def _finish(run: Path) -> dict[int, float]:
    return {r["rid"]: r["finish"] for r in _rows(run / "requests.jsonl") if r["finish"] is not None}


def _preempts(run: Path) -> int:
    return sum(r["preemptions"] for r in _rows(run / "requests.jsonl"))


def compare(real: Path, sim: Path, out: Path | None) -> dict:
    meta = json.loads((real / "meta.json").read_text())
    blocks = meta["num_gpu_blocks"]
    a, b = _series(real, blocks, True), _series(sim, blocks, False)
    fa, fb = _finish(real), _finish(sim)
    common = sorted(set(fa) & set(fb))
    x = np.array([fa[r] for r in common])
    y = np.array([fb[r] for r in common])
    span_real, span_sim = float(x.max()), float(y.max())
    rank = float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
    rel = np.abs(y - x) / x

    horizon = min(a["t"][-1], b["t"][-1])
    grid = np.arange(0.0, horizon, 1.0)
    cols = {k: (_resample(a["t"], a[k], grid), _resample(b["t"], b[k], grid)) for k in ("running", "waiting")}
    cols["used"] = (_resample(a["t"], a["used"], grid), _interpolate_used(b, grid))
    summary = {
        "spec": meta["spec"],
        "blocks": blocks,
        "async": meta["async_scheduling"],
        "span_real": span_real,
        "span_sim": span_sim,
        "span_err_pct": (span_sim / span_real - 1) * 100,
        "finish_rel_err_median_pct": float(np.median(rel) * 100),
        "finish_rel_err_p90_pct": float(np.percentile(rel, 90) * 100),
        "finish_rank_corr": rank,
        "preempt_real": _preempts(real),
        "preempt_sim": _preempts(sim),
        "running_mean_real": float(cols["running"][0].mean()),
        "running_mean_sim": float(cols["running"][1].mean()),
        "running_mae": float(np.abs(cols["running"][0] - cols["running"][1]).mean()),
        "waiting_mean_real": float(cols["waiting"][0].mean()),
        "waiting_mean_sim": float(cols["waiting"][1].mean()),
        "kv_mean_real": float(cols["used"][0].mean()),
        "kv_mean_sim": float(cols["used"][1].mean()),
        "kv_peak_real": float(a["used"].max()),
        "kv_peak_sim": float(b["used"].max()),
    }

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
        for ax, key, label in zip(axes, ("running", "waiting", "used"), ("running", "waiting", "KV used")):
            ax.plot(grid, cols[key][0], color="#C0392B", lw=1.0, label="real vLLM")
            ax.plot(grid, cols[key][1], color="#2874A6", lw=1.0, label="simulated")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
        axes[0].legend(loc="upper right")
        axes[0].set_title(
            f"{meta['spec']}: span {span_real:.0f}s real vs {span_sim:.0f}s sim "
            f"({summary['span_err_pct']:+.1f}%), preemptions {summary['preempt_real']} vs {summary['preempt_sim']}"
        )
        axes[3].scatter(x, y, s=8, color="#555555")
        lim = max(x.max(), y.max())
        axes[3].plot([0, lim], [0, lim], color="black", lw=0.8)
        axes[3].set_xlabel("time (s)  /  scatter panel: real finish time")
        axes[3].set_ylabel("sim finish time")
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(compare(args.real, args.sim, args.out), indent=1))


if __name__ == "__main__":
    main()
