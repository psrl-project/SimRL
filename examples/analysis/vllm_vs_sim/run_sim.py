"""
Replay a workload spec on one simulated instance and record the same logs as `run_vllm.py`.

The instance is `NativeInstance` driven directly, with the composed rollout cost model the
aligned runs use, so what differs from the real engine is the scheduler model and the cost
model and nothing else. A weight update is `abort_all`, a pause, then the same requests
re-added in id order. A boundary (arrival or abort) cuts a pure decode segment with `advance_to`.
While requests wait, whole iterations are stepped instead, since `advance_to` skips a step that
would cross the boundary.

Usage:
    python examples/analysis/vllm_vs_sim/run_sim.py \
        --spec outputs/vllm_vs_sim/specs/grpo_small.json \
        --real outputs/vllm_vs_sim/real/grpo_small --out outputs/vllm_vs_sim/sim/grpo_small
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from srl.cli.main import build_rollout_cost
from srl.component.rollout.engine import NativeInstance, Request
from srl.component.rollout.engine.eviction import build_eviction

MODEL_PATH = "/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B"
KERNEL_ROOT = "outputs/profile/qwen3_8b/kernel_tp1/kernel"


def _cost_model(model: str, kernel_root: str):
    return build_rollout_cost(
        OmegaConf.create(
            {
                "name": "composed",
                "kernel_table_root": kernel_root,
                "gpu": "h20",
                "model_path": model,
                "parallelism": {"tp": 1},
            }
        )
    )


def _state(inst: NativeInstance, pre: dict | None) -> dict:
    row = {
        "t": inst.t_local,
        "running": len(inst.running),
        "waiting": len(inst.waiting),
        "blocks_used": inst.blocks_used(),
        "preemptions": inst.n_preemptions,
    }
    row["blocks_used_pre"] = pre["blocks_used"] if pre else row["blocks_used"]
    return row


def _watch_retire(inst: NativeInstance) -> dict:
    """Capture occupancy at the moment a step ends, before completions free their blocks."""
    seen: dict = {}
    retire = inst._retire

    def wrapped(reqs):
        seen["blocks_used"] = inst.blocks_used()
        return retire(reqs)

    inst._retire = wrapped
    return seen


def run(spec: dict, out: Path, blocks: int, model: str, kernel_root: str, version: int = 0) -> None:
    engine = spec["engine"]
    inst = NativeInstance(
        _cost_model(model, kernel_root),
        token_budget=engine["max_num_batched_tokens"],
        kv_blocks=blocks,
        block_size=16,
        max_concurrency=engine["max_num_seqs"],
        enable_prefix_caching=engine["enable_prefix_caching"],
        eviction=build_eviction("vllm_lru"),
    )
    reqs = {
        r["rid"]: Request(
            rid=r["rid"],
            prompt_len=r["prompt_len"],
            target_len=r["output_len"],
            v_traj=version,
            group_id=r["group"],
            group_index=r["rid"],
            source_group_id=r["group"],
        )
        for r in spec["requests"]
    }
    pre = _watch_retire(inst)
    arrival = {r["rid"]: r["arrival"] for r in spec["requests"]}
    pending = sorted(reqs, key=lambda rid: (arrival[rid], rid))
    aborts = list(spec["abort_at"])
    rows: list[dict] = []
    events: list[dict] = []

    while pending or inst.n_active or aborts:
        for rid in [p for p in pending if arrival[p] <= inst.t_local]:
            pending.remove(rid)
            inst.add_request(reqs[rid], inst.t_local)
        boundary = min(
            [arrival[pending[0]]] if pending else [],
            default=float("inf"),
        )
        if aborts:
            boundary = min(boundary, aborts[0])
        if not inst.n_active and boundary == float("inf"):
            break
        if not inst.n_active:
            inst.t_local = max(inst.t_local, boundary)
        elif boundary < float("inf") and not inst.waiting:
            inst.advance_to(boundary, version)
        elif inst.advance_one_event_committed(version) is None:
            break
        rows.append(_state(inst, pre))
        pre.clear()
        if aborts and inst.t_local >= aborts[0]:
            aborts.pop(0)
            events.append(
                {
                    "kind": "abort",
                    "t": inst.t_local,
                    "generated": {str(r.rid): r.generated for r in inst.running + inst.waiting},
                }
            )
            resumed = sorted(inst.abort_all(requeue=False), key=lambda r: r.rid)
            inst.t_local += spec["pause_s"]
            for req in resumed:
                inst.add_request(req, inst.t_local)
            events.append({"kind": "resume", "t": inst.t_local})
            rows.append(_state(inst, None))

    out.mkdir(parents=True, exist_ok=True)
    (out / "events.json").write_text(json.dumps(events))
    (out / "meta.json").write_text(json.dumps({"spec": spec["name"], "blocks": blocks, "wall_s": inst.t_local}))
    with (out / "steps.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with (out / "requests.jsonl").open("w") as handle:
        for rid, req in reqs.items():
            handle.write(
                json.dumps(
                    {
                        "rid": rid,
                        "output_len": req.target_len,
                        "finish": req.complete_time,
                        "n_out": req.generated,
                        "preemptions": req.num_preemptions,
                    }
                )
                + "\n"
            )
    with (out / "step_stats.jsonl").open("w") as handle:
        for s in inst.step_stats:
            handle.write(
                json.dumps(
                    {
                        "t": s.t,
                        "dt": s.dt,
                        "steps": s.steps,
                        "n_decode": s.n_decode,
                        "n_prefill": s.n_prefill,
                        "phase": s.phase,
                    }
                )
                + "\n"
            )
    print(f"Done: span {inst.t_local:.1f}s, {inst.n_preemptions} preemptions, {len(rows)} events.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--real", type=Path, required=True, help="A run_vllm.py output dir, for the block count.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--kernel-root", default=KERNEL_ROOT)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    blocks = json.loads((args.real / "meta.json").read_text())["num_gpu_blocks"]
    run(spec, args.out, blocks, args.model, args.kernel_root)


if __name__ == "__main__":
    main()
