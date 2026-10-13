"""
Workload specs shared by the real vLLM engine runner and the simulated instance runner.

A spec is plain JSON, so both sides replay exactly the same requests. Prompts are
random token ids shared within a group, which is what makes siblings hit the prefix
cache the way a GRPO group does. Output lengths are forced with `ignore_eos`, so the
only freedom the engine has is scheduling.

Usage:
    python examples/analysis/vllm_vs_sim/workloads.py --out outputs/vllm_vs_sim/specs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TRACE = Path("outputs/workload/psrl_aligned/joint.jsonl")
MODEL_PATH = "/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B"

# The engine settings of the matched PSRL run, except where a workload shrinks the KV cache.
ENGINE = {
    "gpu_memory_utilization": 0.9,
    "max_model_len": 11264,
    "max_num_batched_tokens": 12800,
    "max_num_seqs": 1024,
    "enable_prefix_caching": True,
    "async_scheduling": None,
}


def _spec(
    name: str,
    requests: list[dict],
    engine: dict | None = None,
    abort_at: list[float] | None = None,
    pause_s: float = 0.0,
) -> dict:
    return {
        "name": name,
        "engine": {**ENGINE, **(engine or {})},
        "requests": requests,
        "abort_at": abort_at or [],
        "pause_s": pause_s,
    }


def _grouped(
    rng: np.random.Generator,
    n_groups: int,
    group_size: int,
    prompt_lens: np.ndarray,
    output_lens: np.ndarray,
    arrivals: np.ndarray | None = None,
) -> list[dict]:
    """Build `n_groups * group_size` requests, one prompt per group."""
    requests = []
    for g in range(n_groups):
        for i in range(group_size):
            rid = g * group_size + i
            requests.append(
                {
                    "rid": rid,
                    "group": g,
                    "prompt_len": int(prompt_lens[g]),
                    "output_len": int(output_lens[rid]),
                    "arrival": 0.0 if arrivals is None else float(arrivals[rid]),
                }
            )
    return requests


def _capped_mix(rng: np.random.Generator, n: int, cap: int, frac_capped: float) -> np.ndarray:
    """Outputs that pile up at `cap`, with the rest spread below it."""
    lens = rng.uniform(0.05 * cap, 0.95 * cap, size=n)
    lens[rng.random(n) < frac_capped] = cap
    return lens.astype(np.int64)


def _trace_groups(n_groups: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in TRACE.read_text().splitlines() if line.strip()]
    data = [r for r in rows if not r.get("_meta")]
    picked = np.random.default_rng(seed).choice(len(data), size=n_groups, replace=False)
    prompts = np.array([data[i]["prompt_len"] for i in picked])
    outputs = np.array([o for i in picked for o in data[i]["output_lens"]])
    return prompts, outputs


def build_all() -> dict[str, dict]:
    specs: dict[str, dict] = {}

    rng = np.random.default_rng(1)
    specs["nopressure"] = _spec(
        "nopressure",
        _grouped(rng, 32, 1, np.full(32, 256), rng.integers(512, 2049, size=32)),
    )

    rng = np.random.default_rng(2)
    outputs = np.clip(rng.lognormal(7.0, 0.8, size=160), 256, 4000).astype(np.int64)
    specs["pressure_short"] = _spec(
        "pressure_short",
        _grouped(rng, 160, 1, rng.integers(128, 513, size=160), outputs),
        engine={"gpu_memory_utilization": 0.30, "max_model_len": 5120},
    )

    rng = np.random.default_rng(3)
    prompts = rng.integers(128, 301, size=16)
    outputs = _capped_mix(rng, 128, 2560, 0.6)
    small = _grouped(rng, 16, 8, prompts, outputs)
    small_engine = {"gpu_memory_utilization": 0.30, "max_model_len": 3072}
    specs["grpo_small"] = _spec("grpo_small", small, engine=small_engine)
    specs["grpo_small_abort"] = _spec(
        "grpo_small_abort", small, engine=small_engine, abort_at=[40.0], pause_s=3.0
    )

    prompts, outputs = _trace_groups(16, seed=4)
    full = _grouped(np.random.default_rng(4), 16, 8, prompts, outputs)
    specs["grpo_full"] = _spec("grpo_full", full)
    specs["grpo_full_abort"] = _spec("grpo_full_abort", full, abort_at=[350.0], pause_s=20.0)

    prompts, outputs = _trace_groups(32, seed=7)
    full256 = _grouped(np.random.default_rng(7), 32, 8, prompts, outputs)
    specs["grpo_256"] = _spec("grpo_256", full256)
    specs["grpo_256_abort"] = _spec("grpo_256_abort", full256, abort_at=[700.0], pause_s=20.0)

    rng = np.random.default_rng(5)
    n = 300
    arrivals = np.cumsum(rng.exponential(0.35, size=n))
    outputs = np.clip(rng.lognormal(6.6, 0.7, size=n), 200, 2500).astype(np.int64)
    specs["stream"] = _spec(
        "stream",
        _grouped(rng, n, 1, rng.integers(128, 1025, size=n), outputs, arrivals),
        engine={"gpu_memory_utilization": 0.30, "max_model_len": 4096},
    )

    rng = np.random.default_rng(6)
    specs["prefill_heavy"] = _spec(
        "prefill_heavy",
        _grouped(rng, 64, 1, rng.integers(4000, 9001, size=64), rng.integers(64, 257, size=64)),
        engine={"gpu_memory_utilization": 0.45},
    )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, spec in build_all().items():
        path = args.out / f"{name}.json"
        path.write_text(json.dumps(spec))
        total = sum(r["prompt_len"] + r["output_len"] for r in spec["requests"])
        print(f"{path}: {len(spec['requests'])} requests, {total:,} tokens in total")


if __name__ == "__main__":
    main()
