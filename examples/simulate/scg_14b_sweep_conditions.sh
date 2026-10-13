#!/usr/bin/env bash
# Search the configuration space for conditions under which heterogeneous TP beats
# a homogeneous fleet, on DeepSeek-R1-Distill-Qwen-14B.
#
# THE QUESTION THIS ANSWERS
#   Earlier 14B and 35B runs both found heterogeneity winning only at S=0 and losing
#   by S=2. The cause -- correct, as it turned out -- was that those runs were
#   training-bound: train plus recompute was 2338s of a 3104s span, i.e. **75%**.
#   Rollout was only 25%, so the long tail that heterogeneous TP exists to absorb
#   spent most of its life hidden behind a training block. A tail that is not on the
#   critical path cannot be optimized.
#
#   So this sweep varies the things that move the train/rollout balance and the
#   tail's weight within rollout. Each cell prints its own `rollout% of span`, which
#   is the explanatory variable.
#
# AXES
#   TRAIN_DP    1,2,4   data-parallel width of the trainer. The Hydraulis fit scales
#                       as "ideal-local-workload-scaling", so DP divides the per-rank
#                       token load: DP=4 is roughly a 4x faster step, i.e. a second
#                       or third machine. Measured: 557s -> 300s -> 175s.
#   CAP                 output-length cap, which sets how long the tail can get.
#   BATCH_SIZE          prompt groups per step. Smaller batch means more frequent,
#                       cheaper training steps, so rollout is a larger share of span.
#   trace               `long_ctx` is dispersed (p99/p50 = 4.5); `psrl_aligned` is
#                       near-uniform (1.01) and is the negative control -- if
#                       heterogeneity also wins there, the mechanism is not the tail.
#
#   Held fixed within a run: `N_GPUS` (default 8), group size 8, policy, and the
#   staleness target. Set `INCLUDE_UNCAPPED_BASELINES=true` to add homogeneous
#   TP1 and TP2 arms with the router active cap disabled.
#
# ⚠️ READING THE OUTPUT
#   `dryrun.cli.main` prints the SCG(S) curve, so this script only relays it. SCG is
#   a RATE in tokens/s: compare two arms ONLY when their `Consumed:` lines agree. An
#   arm that ended early divides a smaller token total by a smaller span and scores
#   arbitrarily well -- a cap=60000 cell was once read as a 2.54x win that way, when
#   its heterogeneous arm had been killed at 4 of 6 versions.
#
# USAGE
#   bash examples/simulate/scg_14b_sweep_conditions.sh                      # full grid
#   CELLS="dp4_cap30k_b8:4:30000:8:long_ctx" bash .../scg_14b_sweep_conditions.sh
#   POLICY=psrl N_GPUS=4 INCLUDE_UNCAPPED_BASELINES=true bash ...            # five arms
#   N_VERSIONS=20 is the default. Shorter runs are intentionally rejected because
#   startup transients dominate the version-capped routing comparison.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

DRYRUN_ENV="${DRYRUN_ENV:-/apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh}"
if [[ -f "${DRYRUN_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${DRYRUN_ENV}"
fi

MODEL_PATH="${MODEL_PATH:-/apdcephfs_zwfy/share_303760348/lhy/models/DeepSeek-R1-Distill-Qwen-14B}"
COST_DIR="${COST_DIR:-${ROOT_DIR}/outputs/cost/deepseek_r1_distill_qwen_14b}"
ROLLOUT_ARTIFACT="${ROLLOUT_ARTIFACT:-${COST_DIR}/rollout_psrl_both.json}"
TRAINING_ARTIFACT="${TRAINING_ARTIFACT:-${COST_DIR}/training_hydraulis.json}"
WORK_BASE="${WORK_BASE:-${ROOT_DIR}/outputs/simulate/scg_14b_sweep_v2}"
N_VERSIONS="${N_VERSIONS:-20}"
ACTIVE_CAP="${ACTIVE_CAP:-40}"
MAX_PARALLEL_CELLS="${MAX_PARALLEL_CELLS:-1}"
N_GPUS="${N_GPUS:-8}"
INCLUDE_UNCAPPED_BASELINES="${INCLUDE_UNCAPPED_BASELINES:-false}"

if (( N_VERSIONS < 20 )); then
    echo "N_VERSIONS must be at least 20 for this routing comparison." >&2
    exit 1
fi
if (( MAX_PARALLEL_CELLS < 1 )); then
    echo "MAX_PARALLEL_CELLS must be positive." >&2
    exit 1
fi
if (( N_GPUS < 4 || N_GPUS % 4 != 0 )); then
    echo "N_GPUS must be a positive multiple of 4." >&2
    exit 1
fi
if [[ "${INCLUDE_UNCAPPED_BASELINES}" != "true" \
    && "${INCLUDE_UNCAPPED_BASELINES}" != "false" ]]; then
    echo "INCLUDE_UNCAPPED_BASELINES must be true or false." >&2
    exit 1
fi

tp_array() {
    local degree=$1 count=$2 result="["
    for ((index = 0; index < count; index++)); do
        if (( index > 0 )); then
            result+=","
        fi
        result+="${degree}"
    done
    printf '%s]' "${result}"
}

heterogeneous_tp_array() {
    local tp1_count=$1 tp2_count=$2 result="["
    for ((index = 0; index < tp1_count; index++)); do
        if (( index > 0 )); then
            result+=","
        fi
        result+="1"
    done
    for ((index = 0; index < tp2_count; index++)); do
        if (( tp1_count + index > 0 )); then
            result+=","
        fi
        result+="2"
    done
    printf '%s]' "${result}"
}

ALL1_INSTANCES="${N_GPUS}"
ALL2_INSTANCES="$((N_GPUS / 2))"
HETERO_TP1_INSTANCES="$((N_GPUS / 2))"
HETERO_TP2_INSTANCES="$((N_GPUS / 4))"
HETERO_INSTANCES="$((HETERO_TP1_INSTANCES + HETERO_TP2_INSTANCES))"
ALL1_TP_ARRAY="$(tp_array 1 "${ALL1_INSTANCES}")"
ALL2_TP_ARRAY="$(tp_array 2 "${ALL2_INSTANCES}")"
HETERO_TP_ARRAY="$(
    heterogeneous_tp_array "${HETERO_TP1_INSTANCES}" "${HETERO_TP2_INSTANCES}"
)"

# "label:train_dp:cap:batch:trace_name"
# The grid concentrates around the earlier crossover near batch=8 and cap=30k.
# It varies both axes on either side, plus a near-uniform negative control.
ALL_CELLS=(
    "cap20k_b6:4:20000:6:long_ctx"
    "cap20k_b8:4:20000:8:long_ctx"
    "cap20k_b10:4:20000:10:long_ctx"
    "cap30k_b6:4:30000:6:long_ctx"
    "cap30k_b8:4:30000:8:long_ctx"
    "cap30k_b10:4:30000:10:long_ctx"
    "cap40k_b8:4:40000:8:long_ctx_60k"
    "cap40k_b10:4:40000:10:long_ctx_60k"
    "cap30k_b8_flat:4:30000:8:psrl_aligned"
)
read -r -a CELLS_ARR <<< "${CELLS:-${ALL_CELLS[*]}}"

for required in "${ROLLOUT_ARTIFACT}" "${TRAINING_ARTIFACT}"; do
    [[ -f "${required}" ]] || { echo "Missing: ${required}" >&2; exit 1; }
done

cd "${ROOT_DIR}"

# The extra-long-tail trace for the cap=60k cell, generated here so the cell is
# reproducible from this script alone.
if [[ ! -f "${ROOT_DIR}/outputs/workload/long_ctx_x/joint.jsonl" ]]; then
    echo "Generating the extra-long-tail trace..."
    python3 "${ROOT_DIR}/examples/workload/make_heavy_tail_trace.py" \
        --out "${ROOT_DIR}/outputs/workload/long_ctx_x/joint.jsonl" \
        --n-groups 160 --group-size 8 \
        --body-median 6000 --body-sigma 0.5 \
        --tail-frac 0.15 --tail-alpha 1.2 --tail-scale-mult 8.0 \
        --cap 60000 --prompt-median 400 --prompt-sigma 0.3 --seed 42 >/dev/null
fi

# Same random draws and distribution as `long_ctx`, but without clipping at 30k.
# This isolates the generation cap axis from a simultaneous distribution change.
if [[ ! -f "${ROOT_DIR}/outputs/workload/long_ctx_60k/joint.jsonl" ]]; then
    echo "Generating the cap-extended long-context trace..."
    python3 "${ROOT_DIR}/examples/workload/make_heavy_tail_trace.py" \
        --out "${ROOT_DIR}/outputs/workload/long_ctx_60k/joint.jsonl" \
        --n-groups 160 --group-size 8 \
        --body-median 6000 --body-sigma 0.45 \
        --tail-frac 0.12 --tail-alpha 1.4 --tail-scale-mult 4.0 \
        --cap 60000 --prompt-median 400 --prompt-sigma 0.3 --seed 42 >/dev/null
fi

export WORKLOAD=dapo_trace
export POLICY="${POLICY:-verl}"
export ROLLOUT_MODEL=psrl
export ROLLOUT_ARTIFACT
export GPU="${GPU:-h20}"
export MODEL_PATH
export TRAINING_BACKEND=hydraulis
export TRAINING_ARTIFACT
# pp=4 is the mesh the fit covers for BOTH latency and memory; pp=1 has no memory
# entry and raises on lookup.
export TRAINING_TP=2 TRAINING_PP=4 TRAINING_CP=1
export MICRO_BATCH_SIZE=1
export MAX_TOKENS_PER_MICROBATCH="${MAX_TOKENS_PER_MICROBATCH:-12800}"
export ACTIVATION_RECOMPUTE=full
# Geometry for the unembedding correction comes from TRAINING_ARTIFACT itself.
export RECOMPUTE_MODEL=training_forward
export EXTRAPOLATE="[tokens,activation_bytes,pipeline_slots,sum_squared_tokens,micro_batch_size,profiled_dp]"
export GROUP_SIZE=8
export MAX_STALENESS="${MAX_STALENESS:-2}"
export MAX_PROMPT_LENGTH=1024
export MAX_CONCURRENCY=1024
# 7.80 GiB was backed out of the profiled tp=1 server and reproduces its measured
# KV capacity exactly. Do not launch a real vLLM probe for every sweep arm:
# uncached max-length combinations are expensive, and parallel probes contend
# for the same GPUs without changing the routing comparison.
export KV_MODEL_PATH="${MODEL_PATH}"
export KV_GPU_MEMORY_UTILIZATION=0.9
export KV_PROBE=false
export KV_NON_KV_OVERHEAD_BYTES=8375186227
export N_VERSIONS

run_arm() {  # cell  label  n_inst  tp_arr  strategy  trace  router_cap
    local cell=$1 label=$2 n_inst=$3 tp_arr=$4 strategy=$5 trace=$6
    local router_cap=${7:-${ACTIVE_CAP}}
    echo ""
    echo "---- ${label} ----"
    export N_INSTANCES="${n_inst}"
    export OUTPUT_DIR="${WORK_BASE}/${cell}/${label}"
    export ROLLOUT_TP=1
    export EXTRA_OVERRIDES="\
dynamic_parallel.enabled=true \
dynamic_parallel.n_gpus=${N_GPUS} \
dynamic_parallel.tp_candidates=[1,2] \
dynamic_parallel.strategy=fixed \
dynamic_parallel.tp_per_instance=${tp_arr} \
rollout.max_concurrency=${MAX_CONCURRENCY} \
rollout.router.max_concurrent_seqs_per_instance=${router_cap} \
workload.trace.path=${trace} \
rollout.router.strategy=${strategy}"
    # Relay the simulator's own summary, which already carries the SCG(S) curve.
    bash "${ROOT_DIR}/examples/simulate/simulate.sh" 2>&1 \
        | sed -n '/^Simulation complete/,/^  Logs saved/p' \
        | grep -vE "^  (Logs|Plots) saved"
}

run_cell() (
    local spec=$1
    IFS=':' read -r label dp cap batch trace_name <<< "${spec}"
    trace="${ROOT_DIR}/outputs/workload/${trace_name}/joint.jsonl"
    [[ -f "${trace}" ]] || { echo "skip ${label}: no trace ${trace}"; exit 0; }
    echo ""
    echo "######## ${label}  (train_dp=${dp} cap=${cap} batch=${batch} trace=${trace_name}) ########"
    export TRAINING_DP="${dp}"
    export BATCH_SIZE="${batch}"
    export MAX_OUTPUT_LENGTH="${cap}"
    export TOKEN_BUDGET="${cap}"
    run_arm "${label}" all1 "${ALL1_INSTANCES}" "${ALL1_TP_ARRAY}" request_num_balance "${trace}"
    run_arm "${label}" all2 "${ALL2_INSTANCES}" "${ALL2_TP_ARRAY}" request_num_balance "${trace}"
    run_arm \
        "${label}" half_va "${HETERO_INSTANCES}" "${HETERO_TP_ARRAY}" \
        version_capped_tp "${trace}"
    if [[ "${INCLUDE_UNCAPPED_BASELINES}" == "true" ]]; then
        run_arm \
            "${label}" all1_uncapped "${ALL1_INSTANCES}" "${ALL1_TP_ARRAY}" \
            request_num_balance "${trace}" 0
        run_arm \
            "${label}" all2_uncapped "${ALL2_INSTANCES}" "${ALL2_TP_ARRAY}" \
            request_num_balance "${trace}" 0
    fi
)

COMPLETED_SPECS=()
PIDS=()
for spec in "${CELLS_ARR[@]}"; do
    IFS=':' read -r label _ <<< "${spec}"
    trace_name="${spec##*:}"
    trace="${ROOT_DIR}/outputs/workload/${trace_name}/joint.jsonl"
    if [[ ! -f "${trace}" ]]; then
        echo "skip ${label}: no trace ${trace}"
        continue
    fi
    COMPLETED_SPECS+=("${spec}")
    run_cell "${spec}" &
    PIDS+=("$!")
    if (( ${#PIDS[@]} >= MAX_PARALLEL_CELLS )); then
        wait "${PIDS[0]}"
        PIDS=("${PIDS[@]:1}")
    fi
done
for pid in "${PIDS[@]}"; do
    wait "${pid}"
done

echo ""
python3 - \
    "${WORK_BASE}" "${MAX_STALENESS}" "${N_VERSIONS}" "${N_GPUS}" "${POLICY}" \
    "${INCLUDE_UNCAPPED_BASELINES}" \
    "${COMPLETED_SPECS[@]}" <<'PYSUMMARY'
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
target_s = int(sys.argv[2])
expected_versions = int(sys.argv[3])
n_gpus = int(sys.argv[4])
policy = sys.argv[5]
include_uncapped = sys.argv[6] == "true"
specs = sys.argv[7:]
arms = (
    ("all1", "all2", "half_va", "all1_uncapped", "all2_uncapped")
    if include_uncapped
    else ("all1", "all2", "half_va")
)
homogeneous_arms = tuple(arm for arm in arms if arm != "half_va")


def load_arm(run_dir: Path) -> dict:
    train_path = run_dir / "logs" / "train.jsonl"
    request_path = run_dir / "logs" / "request.jsonl"
    train = [
        json.loads(line)
        for line in train_path.read_text().splitlines()
        if line.strip()
    ]
    requests = [
        json.loads(line)
        for line in request_path.read_text().splitlines()
        if line.strip()
    ]
    consumed = [row for row in requests if row.get("event") == "consume"]
    span = max(float(row["end"]) for row in train)
    staleness = [int(row["staleness_true"]) for row in consumed]
    generated = [int(row.get("generated", 0)) for row in consumed]
    max_s = max(staleness, default=0)
    curve = {
        str(s): sum(
            tokens
            for value, tokens in zip(staleness, generated)
            if value <= s
        )
        / span
        for s in range(max(max_s, target_s) + 1)
    }
    return {
        "versions": len(train),
        "consumed": len(consumed),
        "span_s": span,
        "throughput_tok_s": sum(generated) / span,
        "max_staleness": max_s,
        "staleness_counts": {
            str(s): staleness.count(s) for s in sorted(set(staleness))
        },
        "scg_tok_s": curve,
    }


cells = []
for spec in specs:
    label, train_dp, cap, batch, trace = spec.split(":")
    runs = {arm: load_arm(base / label / arm) for arm in arms}
    counts = {runs[arm]["consumed"] for arm in arms}
    versions = {runs[arm]["versions"] for arm in arms}
    comparable = len(counts) == 1 and versions == {expected_versions}
    homogeneous = max(
        float(runs[arm]["scg_tok_s"].get(str(target_s), 0.0))
        for arm in homogeneous_arms
    )
    heterogeneous = float(
        runs["half_va"]["scg_tok_s"].get(str(target_s), 0.0)
    )
    ratio = heterogeneous / homogeneous if comparable and homogeneous > 0 else None
    cells.append(
        {
            "label": label,
            "train_dp": int(train_dp),
            "generation_cap": int(cap),
            "batch_size": int(batch),
            "trace": trace,
            "comparable": comparable,
            "heterogeneous_vs_best_homogeneous_at_target": ratio,
            "runs": runs,
        }
    )

ranked = sorted(
    (cell for cell in cells if cell["comparable"]),
    key=lambda cell: cell["heterogeneous_vs_best_homogeneous_at_target"],
    reverse=True,
)
summary = {
    "selection_metric": (
        f"half_va SCG({target_s}) / "
        f"max({', '.join(homogeneous_arms)}) SCG({target_s})"
    ),
    "target_staleness": target_s,
    "n_versions": expected_versions,
    "n_gpus": n_gpus,
    "policy": policy,
    "arms": arms,
    "cells": cells,
    "best_cell": ranked[0]["label"] if ranked else None,
}
(base / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)

lines = [
    "# Version-capped heterogeneous TP condition sweep",
    "",
    f"- Versions: {expected_versions}",
    f"- GPU budget: {n_gpus}",
    f"- Policy: {policy}",
    f"- Selection: `{summary['selection_metric']}`",
    "",
    "## SCG target ranking",
    "",
    (
        "| condition | trace | batch | cap | "
        + " | ".join(arms)
        + " | heterogeneous / best homogeneous |"
    ),
    "|" + "---|" * 4 + "---:|" * (len(arms) + 1),
]
for cell in ranked:
    runs = cell["runs"]
    s = str(target_s)
    arm_values = " | ".join(
        f"{runs[arm]['scg_tok_s'][s]:.1f}" for arm in arms
    )
    lines.append(
        f"| {cell['label']} | {cell['trace']} | {cell['batch_size']} | "
        f"{cell['generation_cap']} | {arm_values} | "
        f"{cell['heterogeneous_vs_best_homogeneous_at_target']:.3f}x |"
    )

if ranked:
    best = ranked[0]
    max_s = max(run["max_staleness"] for run in best["runs"].values())
    lines.extend(
        [
            "",
            f"## Full SCG curve for selected condition: {best['label']}",
            "",
            "| S | " + " | ".join(arms) + " |",
            "|" + "---:|" * (len(arms) + 1),
        ]
    )
    for s in range(max_s + 1):
        values = []
        for arm in arms:
            curve = best["runs"][arm]["scg_tok_s"]
            value = float(
                curve.get(str(s), best["runs"][arm]["throughput_tok_s"])
            )
            values.append(value)
        rendered = " | ".join(f"{value:.1f}" for value in values)
        lines.append(f"| {s} | {rendered} |")
(base / "summary.md").write_text("\n".join(lines) + "\n")

print("========= condition ranking at SCG(" + str(target_s) + ") =========")
for cell in ranked:
    runs = cell["runs"]
    s = str(target_s)
    rendered = " ".join(
        f"{arm}={runs[arm]['scg_tok_s'][s]:8.1f}" for arm in arms
    )
    print(
        f"  {cell['label']:20s} "
        f"{rendered} "
        f"ratio={cell['heterogeneous_vs_best_homogeneous_at_target']:.3f}x"
    )
if ranked:
    print()
    print("Selected:", ranked[0]["label"])
    print("Full results:", base / "summary.md")
PYSUMMARY

echo ""
echo "Per-arm logs and plots: ${WORK_BASE}/<cell>/<arm>/"
