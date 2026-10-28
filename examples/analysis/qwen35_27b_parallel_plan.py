#!/usr/bin/env python
"""
Plan inference parallelism, training parallelism, and a GPU split for Qwen3.8-27B.

Training numbers come from `HydraulisTrainingCost`, fitted from five real Megatron steps
at `tp4/pp2/cp1` (`outputs/profile/qwen35_27b/hydraulis_training.json`): three sequence
lengths that are not all multiples of one another (16384, 24576, 32768), needed because two
lengths related by an exact factor leave the linear and quadratic token columns collinear,
and two microbatch counts (1, 4) to separate the per-microbatch cost from the per-step one.
Both fitted entries were checked against the five measured points (latency MRE 1.51%, memory
MRE 1.02%) and against the one measured OOM boundary (32768 tokens on 8 GPUs, predicted
95.6 GiB on a 95 GiB device).

Context parallelism was never measured: the vendored Megatron asserts
`context_parallel_size == 1` whenever a gated-delta-net layer is present, so no `cp > 1`
step could be run. `HydraulisTrainingCost` therefore only ever fits `cp == 1` cells, and a
`cp > 1` query is an explicit, opted-in analytical projection (`allow_context_parallel=True`
on `from_artifact`): the per-token activation and the token-proportional and quadratic
latency terms divide by `cp`, the optimizer shard widens by `cp`, and the GDN head count
divides by `cp` on top of `tp` (the way the vendored Megatron itself would shard it, per
`GatedDeltaNet.cp_size`, once CP support lands). Every plan this script proposes that uses
`cp > 1` is flagged accordingly: it is a model-side extrapolation, not a measurement.

Inference uses TP4 at every context, the only tensor-parallel degree this script considers:
the checkpoint has 4 KV heads, so TP cannot exceed 4, and a TP1/TP2 profiling sweep showed
per-GPU throughput no better than TP4's while holding far less KV cache. Rollout numbers are
taken directly from the measured vLLM benchmarks, not a fitted model: there is no comparable
fitting gap on the inference side, since TP4 was actually run end to end at every context.

Usage:
    python examples/analysis/qwen35_27b_parallel_plan.py
    python examples/analysis/qwen35_27b_parallel_plan.py --markdown
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from srl.cost.artifact import CostArtifact
from srl.cost.training import HydraulisTrainingCost, TrainingParallelism, TrainingWorkload
from srl.cost.training.base import TrainingOutOfDomainError

ROOT = Path(__file__).resolve().parents[2]
GIB = 2**30
ARTIFACT_PATH = ROOT / "outputs/profile/qwen35_27b/hydraulis_training.json"

# Allocated-byte budget. Reserved ran 1-5 GiB above allocated across the five measured
# points, so staying under 90 GiB of allocation keeps the reservation under a 95 GiB device
# with headroom for the allocator's own fragmentation.
MEMORY_BUDGET_BYTES = 90 * GIB

# The hard caps established by actually running candidates (see QWEN38_27B.md):
# TP <= 4 (4 KV heads), PP <= 2 (the VL provider's position_ids path breaks past two
# stages), CP <= 4 (CP must divide both 16 and 48 linear-attention heads at TP4, so
# gcd(16, 48) / 4 bounds it). CP is tried only when TP4/PP2/CP1 does not fit, in ascending
# order, so a plan prefers the measured cp=1 cell whenever it can.
TP = 4
PP = 2
CP_OPTIONS = (1, 2, 4)

CONTEXTS = (16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024)
WORLDS = (16, 32, 64, 128)

BATCH_PROMPTS = 32
GROUP_N = 16
TRAJECTORIES = BATCH_PROMPTS * GROUP_N
"""512 trajectories per RL step: 32 prompts, 16 responses (group size) each."""

# Measured decode throughput per TP4 rollout replica, output tokens/s, at the concurrency
# the KV cache admits at that context cap (outputs/profile/qwen35_27b/vllm_tp4). Prefill
# throughput is close to flat with context (7.2k-9.5k tok/s across the sweep), so one
# representative figure stands in for all four contexts.
DECODE_TOKENS_PER_REPLICA = {
    16 * 1024: 698.0,
    32 * 1024: 454.0,
    64 * 1024: 232.0,
    128 * 1024: 119.0,
}
PREFILL_TOKENS_PER_REPLICA = 9_000.0


@dataclass(frozen=True)
class TrainingPlan:
    tp: int
    pp: int
    cp: int
    dp: int
    peak_bytes: int
    extrapolated_cp: bool


def _workload(seq: int) -> TrainingWorkload:
    return TrainingWorkload(
        sequence_lengths=(seq,) * TRAJECTORIES,
        micro_batch_size=1,
        recompute="full",
    )


def best_training_plan(cost: HydraulisTrainingCost, world: int, seq: int) -> TrainingPlan | None:
    """Return the smallest-cp feasible plan at `world` GPUs, preferring the measured cp=1 cell."""
    for cp in CP_OPTIONS:
        model_parallel = TP * PP * cp
        if model_parallel > world or world % model_parallel:
            continue
        dp = world // model_parallel
        if TRAJECTORIES % dp:
            continue
        parallelism = TrainingParallelism(tp=TP, pp=PP, dp=dp, cp=cp)
        try:
            peak = cost.memory_model.predict(_workload(seq), parallelism)
        except TrainingOutOfDomainError:
            continue
        if peak <= MEMORY_BUDGET_BYTES:
            return TrainingPlan(
                tp=TP, pp=PP, cp=cp, dp=dp, peak_bytes=peak, extrapolated_cp=cp != 1
            )
    return None


def train_step_seconds(cost: HydraulisTrainingCost, plan: TrainingPlan, seq: int) -> float:
    parallelism = TrainingParallelism(tp=plan.tp, pp=plan.pp, dp=plan.dp, cp=plan.cp)
    return cost.latency_model.predict(_workload(seq), parallelism)


def rollout_seconds(rollout_gpus: int, seq: int) -> float:
    """Return the wall time to roll out `TRAJECTORIES` trajectories at one context cap."""
    if rollout_gpus <= 0:
        return float("inf")
    replicas = rollout_gpus // TP
    if replicas <= 0:
        return float("inf")
    # Prompt and response each take half the context cap.
    output_tokens = TRAJECTORIES * seq / 2
    prefill_tokens = TRAJECTORIES * seq / 2
    decode = output_tokens / (replicas * DECODE_TOKENS_PER_REPLICA[seq])
    prefill = prefill_tokens / (replicas * PREFILL_TOKENS_PER_REPLICA)
    return decode + prefill


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", action="store_true", help="emit a Markdown table")
    args = parser.parse_args()

    artifact = CostArtifact.load(ARTIFACT_PATH)
    # The fitted sweep used 1-4 microbatches per rank and 16384-32768 tokens per sequence.
    # A real RL step at world=8 puts all 512 trajectories on one DP rank (512 microbatches),
    # and 64k/128k context exceeds the sequence range outright. Both are linear
    # extrapolations of a latency form whose `slots` and `tokens` columns are linear by
    # construction (`slots = nmb + pp - 1` is 1F1B's own definition, not a fitted curve), so
    # this opts in rather than leaving the domain gate to silently clip the workload down to
    # what was profiled. Every such query still records an explicit note (see
    # `extrapolation_notes` on `TrainingCostEstimate`).
    extrapolate = [
        "tokens", "pipeline_slots", "sum_squared_tokens", "micro_batch_size",
        "activation_bytes", "profiled_dp",
    ]
    strict_cost = HydraulisTrainingCost.from_artifact(artifact, extrapolate=extrapolate)
    permissive_cost = HydraulisTrainingCost.from_artifact(
        artifact, allow_context_parallel=True, extrapolate=extrapolate
    )

    rows = []
    for seq in CONTEXTS:
        for world in WORLDS:
            # Staleness 1 means rollout(k+1) and train(k) run concurrently on separate GPU
            # pools: that overlap is the entire point, and it only exists if the machine is
            # actually partitioned. A "train on everything, then roll out on everything"
            # number is a staleness-0 wall clock wearing a staleness-1 label, so it is
            # reported only as a fallback when no partition exists, never preferred over a
            # real one just for being numerically smaller.
            best = None
            for train_gpus in range(TP * PP, world + 1, TP * PP):
                rollout_gpus = world - train_gpus
                if rollout_gpus < TP:
                    continue
                plan = best_training_plan(permissive_cost, train_gpus, seq)
                if plan is None:
                    continue
                cost = permissive_cost if plan.extrapolated_cp else strict_cost
                t_train = train_step_seconds(cost, plan, seq)
                t_roll = rollout_seconds(rollout_gpus, seq)
                step = max(t_train, t_roll)
                if best is None or step < best["step"]:
                    best = {
                        "plan": plan,
                        "train_gpus": train_gpus,
                        "rollout_gpus": rollout_gpus,
                        "t_train": t_train,
                        "t_roll": t_roll,
                        "step": step,
                        "serial": False,
                    }
            if best is None:
                # No partition leaves both phases a usable share: fall back to the whole
                # machine training, then the whole machine rolling out. This does not
                # realize staleness=1's overlap, so the reported time is a time-shared
                # total, not a steady-state per-step figure.
                plan_full = best_training_plan(permissive_cost, world, seq)
                if plan_full is not None:
                    cost = permissive_cost if plan_full.extrapolated_cp else strict_cost
                    t_train_full = train_step_seconds(cost, plan_full, seq)
                    t_roll_full = rollout_seconds(world, seq)
                    best = {
                        "plan": plan_full,
                        "train_gpus": world,
                        "rollout_gpus": world,
                        "t_train": t_train_full,
                        "t_roll": t_roll_full,
                        "step": t_train_full + t_roll_full,
                        "serial": True,
                    }
            rows.append({"seq": seq, "world": world, "best": best})

    if args.markdown:
        _print_markdown(rows)
    else:
        _print_table(rows)


def _print_table(rows: list[dict]) -> None:
    print(
        f"{'ctx':>6} {'gpus':>5} {'infer TP':>9} {'train(tp,pp,cp,dp)':>19} "
        f"{'train/roll':>11} {'peak GiB':>9} {'train s':>8} {'roll s':>8} {'step':>10}"
    )
    for row in rows:
        seq, world, best = row["seq"], row["world"], row["best"]
        label = f"{seq // 1024}k"
        if best is None:
            print(
                f"{label:>6} {world:>5} {'TP4':>9} {'-':>19} {'-':>11} {'-':>9} "
                f"{'-':>8} {'-':>8} {'infeasible':>10}"
            )
            continue
        plan = best["plan"]
        mesh = f"({plan.tp},{plan.pp},{plan.cp},{plan.dp})" + ("*" if plan.extrapolated_cp else "")
        split = f"{best['train_gpus']}/{best['rollout_gpus']}"
        step_label = f"{best['step']:.0f}s" + ("!" if best["serial"] else "")
        print(
            f"{label:>6} {world:>5} {'TP4':>9} {mesh:>19} {split:>11} "
            f"{plan.peak_bytes / GIB:>9.1f} {best['t_train']:>8.0f} {best['t_roll']:>8.0f} {step_label:>10}"
        )
    print("\n* = cp > 1, an analytical extrapolation, never a measured step.")
    print("! = no split leaves rollout a share of the machine; phases run serially.")


def _print_markdown(rows: list[dict]) -> None:
    print(
        "| context | GPUs | train (tp,pp,cp,dp) | train/rollout split | peak | "
        "train s | rollout s | step |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for row in rows:
        seq, world, best = row["seq"], row["world"], row["best"]
        label = f"{seq // 1024}k"
        if best is None:
            print(f"| {label} | {world} | - | - | - | - | - | infeasible |")
            continue
        plan = best["plan"]
        mesh = f"({plan.tp},{plan.pp},{plan.cp},{plan.dp})" + (" *" if plan.extrapolated_cp else "")
        split = f"{best['train_gpus']}/{best['rollout_gpus']}"
        step_label = f"{best['step']:.0f} s" + (" !" if best["serial"] else "")
        print(
            f"| {label} | {world} | {mesh} | {split} | {plan.peak_bytes / GIB:.1f} GiB | "
            f"{best['t_train']:.0f} | {best['t_roll']:.0f} | **{step_label}** |"
        )


if __name__ == "__main__":
    main()
