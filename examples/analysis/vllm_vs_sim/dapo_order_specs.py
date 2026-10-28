"""Build real-vLLM workload specs for the LPT-admission-order question, on the real DAPO batch.

Reuses `run_vllm.py`/`run_sim.py`/`compare.py` unmodified. Neither script has any concept of
admission order beyond `(arrival, rid)` sort order, and every request here has `arrival=0`, so
"LPT" needs no change to either script: it is just `rid` assigned in output-length-descending
order instead of natural (trace) order. `group` stays at its original value either way, so
`run_vllm.py`'s per-group prompt-token generation (and prefix-cache sharing) is unaffected by the
`rid` reassignment.

Reads the first `--n-groups` groups of `outputs/workload/qwen3_8b_dapo_38912_tp1/joint.jsonl`
directly, in file order (`srl/workload/trace.py::read_group_trace` reads file order, no shuffling).
`--n-groups 32` (256 requests) matches `workflow/validate/experiments/lpt_single_instance.py`'s
`cap40`/`cap40_kv0.6` points exactly.

Usage:
    python examples/analysis/vllm_vs_sim/dapo_order_specs.py calibrate --util 0.9 --n-groups 8
    python examples/analysis/vllm_vs_sim/dapo_order_specs.py build --kv06-util 0.6108 \\
        --kv05-util 0.40 --n-groups 8 --tag n64
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REAL_TRACE_PATH = ROOT / "outputs" / "workload" / "qwen3_8b_dapo_38912_tp1" / "joint.jsonl"
OUT_SPECS = ROOT / "outputs" / "vllm_vs_sim" / "specs"
MODEL_PATH = "/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B"

ENGINE_BASE = {
    "gpu_memory_utilization": 0.9,
    "max_model_len": 40000,
    "max_num_batched_tokens": 12800,
    "max_num_seqs": 1024,
    "enable_prefix_caching": True,
    "async_scheduling": None,
}
CAP = 40


def _load_run_vllm():
    spec = importlib.util.spec_from_file_location(
        "run_vllm", Path(__file__).resolve().parent / "run_vllm.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _first_n_groups(n_groups: int) -> list[dict]:
    rows = [json.loads(line) for line in REAL_TRACE_PATH.read_text().splitlines() if line.strip()]
    data = [r for r in rows if not r.get("_meta")]
    return data[:n_groups]


def _natural_requests(groups: list[dict]) -> list[dict]:
    out = []
    for g in groups:
        for output_len in g["output_lens"]:
            out.append({"group": g["group_id"], "prompt_len": g["prompt_len"], "output_len": output_len})
    return out


def _with_rid(requests: list[dict], order: str) -> list[dict]:
    """Assign `rid` 0..N-1 in natural order, or in output-length-descending (LPT) order."""
    if order == "normal":
        ordered = requests
    elif order == "lpt":
        ordered = sorted(enumerate(requests), key=lambda pair: (-pair[1]["output_len"], pair[0]))
        ordered = [r for _, r in ordered]
    else:
        raise ValueError(f"order must be 'normal' or 'lpt', got {order!r}.")
    return [{"rid": i, "arrival": 0.0, **r} for i, r in enumerate(ordered)]


def _spec(name: str, requests: list[dict], engine: dict) -> dict:
    return {"name": name, "engine": engine, "requests": requests, "abort_at": [], "pause_s": 0.0}


def calibrate(utils: list[float], n_groups: int) -> None:
    """Print `num_gpu_blocks` at each utilization, engine-init only, no requests. Run each
    utilization as its own process invocation (the caller launches one per `CUDA_VISIBLE_DEVICES`)
    so vLLM's GPU memory is never reused across two engine constructions in one process.
    `num_gpu_blocks` is a property of the engine config and the GPU, not the workload, so a
    calibration done at one `n_groups` is valid for any other -- `n_groups` only matters here
    because `_build_engine` needs *a* spec to read `engine.max_num_seqs` from, which does not
    affect KV sizing."""
    run_vllm = _load_run_vllm()
    groups = _first_n_groups(n_groups)
    requests = _with_rid(_natural_requests(groups), "normal")
    for u in utils:
        engine = {**ENGINE_BASE, "gpu_memory_utilization": u, "max_num_seqs": CAP}
        spec = _spec("calibrate", requests, engine)
        llm = run_vllm._build_engine(spec, MODEL_PATH)
        cache = llm.llm_engine.vllm_config.cache_config
        print(
            json.dumps(
                {"util": u, "num_gpu_blocks": int(cache.num_gpu_blocks), "block_size": int(cache.block_size)}
            ),
            flush=True,
        )


def build(kv06_util: float, kv05_util: float, n_groups: int, tag: str) -> None:
    groups = _first_n_groups(n_groups)
    natural = _natural_requests(groups)
    n_requests = len(natural)
    OUT_SPECS.mkdir(parents=True, exist_ok=True)
    points = {
        "cap40_kv1": {"max_num_seqs": CAP, "gpu_memory_utilization": 0.9},
        "cap40_kv0.6": {"max_num_seqs": CAP, "gpu_memory_utilization": kv06_util},
        "cap40_kv0.5": {"max_num_seqs": CAP, "gpu_memory_utilization": kv05_util},
        "uncapped_kv1": {"max_num_seqs": n_requests, "gpu_memory_utilization": 0.9},
    }
    for point_name, overrides in points.items():
        engine = {**ENGINE_BASE, **overrides}
        for order in ("normal", "lpt"):
            requests = _with_rid(natural, order)
            spec = _spec(f"dapo_{tag}_{point_name}_{order}", requests, engine)
            path = OUT_SPECS / f"dapo_{tag}_{point_name}_{order}.json"
            path.write_text(json.dumps(spec))
            total = sum(r["prompt_len"] + r["output_len"] for r in requests)
            print(f"{path}: {len(requests)} requests, {total:,} tokens, engine={engine}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_cal = sub.add_parser("calibrate")
    p_cal.add_argument("--util", type=float, nargs="+", required=True)
    p_cal.add_argument("--n-groups", type=int, default=32)

    p_build = sub.add_parser("build")
    p_build.add_argument("--kv06-util", type=float, required=True)
    p_build.add_argument("--kv05-util", type=float, required=True)
    p_build.add_argument("--n-groups", type=int, default=32)
    p_build.add_argument("--tag", default="n256", help="Spec filename tag, e.g. n64.")

    args = parser.parse_args()
    if args.command == "calibrate":
        calibrate(args.util, args.n_groups)
    else:
        build(args.kv06_util, args.kv05_util, args.n_groups, args.tag)


if __name__ == "__main__":
    main()
