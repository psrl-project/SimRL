"""
Decompose a composed tp>1 prediction into GEMM, attention and collective, and price the
collective three ways.

The composed model sums measured kernels, so the share each term contributes is recoverable
exactly rather than by difference. That matters here because the collective is the term whose
measurement changed: it used to be timed with `torch.distributed.all_reduce`, which no vLLM
server invokes, and is now timed through the dispatcher a server actually uses.

Three prices per bucket answer three separate questions:

- `vllm`, the dispatched kernel: what the model now predicts.
- `nccl`, the control: how much the old table biased the prediction.
- zero: how large the collective term is at all, i.e. what a comm-free model would have said.

Compared against the real per-bucket latency, the residual also says whether the remaining
error looks like unmodelled compute/communication overlap. The collective is timed in isolation,
so if the prediction runs high by roughly the collective term while tp=1 is accurate, overlap is
the cause and fitting the comm term downward would be curve-fitting over a known mechanism.

Usage:
    python examples/profile/kernel/decompose_tp.py \
        --config examples/profile/kernel/qwen3_8b_h20_tp2.yaml \
        --baseline outputs/profile/qwen3_8b/vllm_tp2/profiles/vllm.jsonl \
        --model-path $MODEL_PATH
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median

from srl.cli.profile import _kernel_config
from srl.cost.kernel.database import KernelDatabase
from srl.cost.kernel.model_ops import ModelOps
from srl.cost.kernel.validate import load_baseline
from srl.cost.rollout.base import RolloutWorkload
from srl.cost.rollout.composed import ComposedRollout


@dataclass(frozen=True)
class Decomposition:
    """One bucket's predicted terms, in seconds, plus the measured latency."""

    phase: str
    request_count: int
    n_tokens: int
    measured_s: float
    gemm_s: float
    attention_s: float
    comm_vllm_s: float
    comm_nccl_s: float

    @property
    def predicted_s(self) -> float:
        """Return what the model predicts, which prices the collective as dispatched."""
        return self.gemm_s + self.attention_s + self.comm_vllm_s

    def error_percent(self, comm_s: float) -> float:
        """Return the signed relative error of a prediction using `comm_s` for the collective."""
        predicted = self.gemm_s + self.attention_s + comm_s
        return 100.0 * (predicted - self.measured_s) / self.measured_s


def _decompose(
    model: ComposedRollout,
    ops: ModelOps,
    database: KernelDatabase,
    bucket: object,
) -> Decomposition:
    workload = RolloutWorkload(
        phase=bucket.phase,
        n_tokens=bucket.n_tokens,
        sum_l2=bucket.sum_l2,
        ctxsum=bucket.ctxsum,
        request_count=bucket.request_count,
    )
    request_count = max(int(bucket.request_count), 1)
    token_count = max(int(round(bucket.n_tokens)), 1)

    gemm_s = sum(
        multiplier * database.query_gemm(query)
        for query, multiplier in ops.gemm_queries(token_count, request_count)
    )
    # `predict` is the authority on the attention term, including the mixed-phase rule, so it is
    # recovered by subtraction from a comm-free total rather than reimplemented here.
    comm_vllm_s = sum(
        multiplier * database.query_comm(query)
        for query, multiplier in ops.comm_queries(token_count)
    )
    comm_nccl_s = sum(
        multiplier * database.query_comm(replace(query, backend="nccl"))
        for query, multiplier in ops.comm_queries(token_count)
    )
    attention_s = model.predict(workload) - gemm_s - comm_vllm_s
    return Decomposition(
        phase=bucket.phase,
        request_count=request_count,
        n_tokens=token_count,
        measured_s=bucket.latency_s,
        gemm_s=gemm_s,
        attention_s=attention_s,
        comm_vllm_s=comm_vllm_s,
        comm_nccl_s=comm_nccl_s,
    )


def _summarize(rows: list[Decomposition], phase: str) -> dict[str, float]:
    selected = [row for row in rows if row.phase == phase]
    if not selected:
        return {}
    comm_share = [100.0 * r.comm_vllm_s / r.predicted_s for r in selected]
    return {
        "buckets": len(selected),
        "comm_share_percent_median": median(comm_share),
        "comm_share_percent_max": max(comm_share),
        # MRE under each pricing of the collective. The gap between them is the cost of having
        # measured the wrong kernel; the gap to `comm_zero` is the size of the term itself.
        "mre_percent_vllm": median(abs(r.error_percent(r.comm_vllm_s)) for r in selected),
        "mre_percent_nccl": median(abs(r.error_percent(r.comm_nccl_s)) for r in selected),
        "mre_percent_comm_zero": median(abs(r.error_percent(0.0)) for r in selected),
        # Signed, to separate over-prediction (the overlap signature) from noise.
        "signed_error_percent_median": median(r.error_percent(r.comm_vllm_s) for r in selected),
        "nccl_over_vllm_comm": median(
            r.comm_nccl_s / r.comm_vllm_s for r in selected if r.comm_vllm_s > 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--surface", default="pchip", choices=("pchip", "shepard"))
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = _kernel_config(args.config)
    if config.tensor_parallel <= 1:
        raise SystemExit(
            f"This tool decomposes a collective term, but tensor_parallel is "
            f"{config.tensor_parallel}, where no collective runs."
        )
    buckets = load_baseline(args.baseline, min_samples=args.min_samples)
    if not buckets:
        raise SystemExit(f"No usable iterations in {args.baseline}.")
    database = KernelDatabase.from_root(
        config.table_root(), gpu=config.gpu(), utility_surface=args.surface
    )
    ops = ModelOps.from_checkpoint(
        args.model_path or config.model.name, tensor_parallel=config.tensor_parallel
    )
    model = ComposedRollout(database, ops, tensor_parallel=config.tensor_parallel)

    rows = [_decompose(model, ops, database, bucket) for bucket in buckets]
    summary = {phase: _summarize(rows, phase) for phase in sorted({r.phase for r in rows})}
    summary = {phase: value for phase, value in summary.items() if value}

    for phase, value in summary.items():
        print(
            f"{phase}: {value['buckets']} buckets | "
            f"comm {value['comm_share_percent_median']:.1f}% of step "
            f"(max {value['comm_share_percent_max']:.1f}%) | "
            f"MRE vllm {value['mre_percent_vllm']:.2f}% "
            f"nccl {value['mre_percent_nccl']:.2f}% "
            f"comm-free {value['mre_percent_comm_zero']:.2f}% | "
            f"signed {value['signed_error_percent_median']:+.2f}% | "
            f"nccl/vllm comm {value['nccl_over_vllm_comm']:.2f}x"
        )

    if args.output:
        payload = {
            "tensor_parallel": config.tensor_parallel,
            "baseline": str(args.baseline),
            "surface": args.surface,
            "summary": summary,
        }
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote {args.output}.")


if __name__ == "__main__":
    main()
