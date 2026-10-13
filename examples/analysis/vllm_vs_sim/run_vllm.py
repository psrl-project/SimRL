"""
Replay a workload spec on a real vLLM engine core and record its scheduling decisions.

The engine core runs in this process with no output processing, so every step is the
scheduler plus the model runner and nothing else, as inside the engine process of a
server. Each `schedule()` call, each preemption and each finished request is logged.
A weight update is modelled as the rollout side sees it: abort everything, reset the
prefix cache, pause, then resubmit each request with its generated tokens appended to
the prompt.

Usage:
    CUDA_VISIBLE_DEVICES=0 python examples/analysis/vllm_vs_sim/run_vllm.py \
        --spec outputs/vllm_vs_sim/specs/grpo_small.json --out outputs/vllm_vs_sim/real/grpo_small
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL_PATH = "/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B"


def _prompt_tokens(group: int, length: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed * 1_000_003 + group)
    return rng.integers(1000, 100_000, size=length).tolist()


class Recorder:
    """Wraps the scheduler so every decision lands in a log."""

    def __init__(self, core, t0: float) -> None:
        self.core = core
        self.sched = core.scheduler
        self.t0 = t0
        self.steps: list[dict] = []
        self.preempts: list[dict] = []
        self.preempt_count: dict[str, int] = {}
        self._schedule = self.sched.schedule
        self._preempt = self.sched._preempt_request
        self.sched.schedule = self._wrapped_schedule
        self.sched._preempt_request = self._wrapped_preempt
        self._n_preempt = 0

    def _wrapped_preempt(self, request, timestamp):
        self._n_preempt += 1
        rid = request.request_id
        self.preempt_count[rid] = self.preempt_count.get(rid, 0) + 1
        self.preempts.append(
            {
                "t": time.perf_counter() - self.t0,
                "rid": rid,
                "computed": int(request.num_computed_tokens),
                "running": len(self.sched.running),
            }
        )
        return self._preempt(request, timestamp)

    def _wrapped_schedule(self, *args, **kwargs):
        before = self._n_preempt
        out = self._schedule(*args, **kwargs)
        pool = self.sched.kv_cache_manager.block_pool
        self.steps.append(
            {
                "t": time.perf_counter() - self.t0,
                "running": len(self.sched.running),
                "waiting": len(self.sched.waiting),
                "free_blocks": int(pool.get_num_free_blocks()),
                "tokens": int(out.total_num_scheduled_tokens),
                "new": len(out.scheduled_new_reqs),
                "cached": len(out.scheduled_cached_reqs.req_ids),
                "preempted": self._n_preempt - before,
            }
        )
        return out


def _build_engine(spec: dict, model: str):
    from vllm import LLM

    engine = spec["engine"]
    kwargs = {
        "model": model,
        "dtype": "bfloat16",
        "load_format": "dummy",
        "trust_remote_code": True,
        "gpu_memory_utilization": engine["gpu_memory_utilization"],
        "max_model_len": engine["max_model_len"],
        "max_num_batched_tokens": engine["max_num_batched_tokens"],
        "max_num_seqs": engine["max_num_seqs"],
        "enable_prefix_caching": engine["enable_prefix_caching"],
        "enable_chunked_prefill": True,
    }
    if engine.get("async_scheduling") is not None:
        kwargs["async_scheduling"] = engine["async_scheduling"]
    return LLM(**kwargs)


def _make_request(rid: str, prompt: list[int], max_tokens: int):
    from vllm import SamplingParams
    from vllm.v1.engine import EngineCoreRequest

    params = SamplingParams(
        max_tokens=max_tokens,
        min_tokens=max_tokens,
        ignore_eos=True,
        temperature=1.0,
        detokenize=False,
    )
    return EngineCoreRequest(
        request_id=rid,
        prompt_token_ids=prompt,
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def run(spec: dict, out: Path, model: str, seed: int) -> None:
    llm = _build_engine(spec, model)
    core = llm.llm_engine.engine_core.engine_core
    vllm_config = llm.llm_engine.vllm_config
    cache = vllm_config.cache_config
    meta = {
        "spec": spec["name"],
        "num_gpu_blocks": int(cache.num_gpu_blocks),
        "block_size": int(cache.block_size),
        "async_scheduling": bool(vllm_config.scheduler_config.async_scheduling),
        "scheduler": type(core.scheduler).__name__,
    }
    print(json.dumps(meta), flush=True)

    reqs = {r["rid"]: r for r in spec["requests"]}
    prompts = {rid: _prompt_tokens(r["group"], r["prompt_len"], seed) for rid, r in reqs.items()}
    generated: dict[int, list[int]] = {rid: [] for rid in reqs}
    live: dict[str, int] = {}
    finish: dict[int, dict] = {}
    first_token: dict[int, float] = {}
    generation = 0

    # Warm up so the first measured step does not pay one-time costs.
    warm_id = "warmup"
    core.add_request(*core.preprocess_add_request(_make_request(warm_id, prompts[0][:64], 8)))
    while core.scheduler.has_requests() or core.batch_queue:
        outputs, executed = core.step_fn()
        core.post_step(executed)
    core.reset_prefix_cache()

    t0 = time.perf_counter()
    rec = Recorder(core, t0)
    pending = sorted(reqs.values(), key=lambda r: (r["arrival"], r["rid"]))
    aborts = list(spec["abort_at"])
    events: list[dict] = []

    def submit(rid: int) -> None:
        r = reqs[rid]
        prompt = prompts[rid] + generated[rid]
        left = r["output_len"] - len(generated[rid])
        name = f"{rid}_g{generation}"
        live[name] = rid
        core.add_request(*core.preprocess_add_request(_make_request(name, prompt, left)))

    def drain_outputs(outputs) -> None:
        if not outputs:
            return
        now = time.perf_counter() - t0
        for client_outputs in outputs.values():
            for item in client_outputs.outputs:
                rid = live.get(item.request_id)
                if rid is None:
                    continue
                if item.new_token_ids:
                    first_token.setdefault(rid, now)
                    generated[rid].extend(item.new_token_ids)
                if item.finish_reason is not None and len(generated[rid]) >= reqs[rid]["output_len"]:
                    finish[rid] = {"t": now, "n_out": len(generated[rid])}
                    live.pop(item.request_id, None)

    while pending or live or core.batch_queue:
        now = time.perf_counter() - t0
        while pending and pending[0]["arrival"] <= now:
            submit(pending.pop(0)["rid"])
        if aborts and now >= aborts[0]:
            aborts.pop(0)
            snapshot = {str(rid): len(generated[rid]) for rid in live.values()}
            events.append({"kind": "abort", "t": now, "generated": snapshot})
            core.abort_requests(list(live))
            while core.scheduler.has_requests() or core.batch_queue:
                outputs, executed = core.step_fn()
                core.post_step(executed)
            resumed = sorted(live.values())
            live.clear()
            core.reset_prefix_cache()
            time.sleep(spec["pause_s"])
            generation += 1
            for rid in resumed:
                submit(rid)
            events.append({"kind": "resume", "t": time.perf_counter() - t0})
            continue
        if not (core.scheduler.has_requests() or core.batch_queue):
            time.sleep(0.001)
            continue
        outputs, executed = core.step_fn()
        core.post_step(executed)
        drain_outputs(outputs)

    meta["wall_s"] = time.perf_counter() - t0
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(meta))
    (out / "events.json").write_text(json.dumps(events))
    with (out / "steps.jsonl").open("w") as handle:
        for row in rec.steps:
            handle.write(json.dumps(row) + "\n")
    with (out / "preempts.jsonl").open("w") as handle:
        for row in rec.preempts:
            handle.write(json.dumps(row) + "\n")
    with (out / "requests.jsonl").open("w") as handle:
        for rid, r in reqs.items():
            row = {
                "rid": rid,
                "output_len": r["output_len"],
                "first_token": first_token.get(rid),
                "finish": finish.get(rid, {}).get("t"),
                "n_out": len(generated[rid]),
                "preemptions": sum(
                    n for name, n in rec.preempt_count.items() if name.split("_g")[0] == str(rid)
                ),
            }
            handle.write(json.dumps(row) + "\n")
    print(f"Done in {meta['wall_s']:.1f}s with {len(rec.steps)} schedule calls and {len(rec.preempts)} preemptions.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--async-scheduling",
        choices=["default", "on", "off"],
        default="default",
        help="Override the engine's async scheduling. default leaves vLLM to decide.",
    )
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.async_scheduling != "default":
        spec["engine"]["async_scheduling"] = args.async_scheduling == "on"
    run(spec, args.out, args.model, args.seed)


if __name__ == "__main__":
    main()
