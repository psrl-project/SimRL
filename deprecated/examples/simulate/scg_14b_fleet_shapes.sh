#!/usr/bin/env bash
# SCG(S) by fleet shape on DeepSeek-R1-Distill-Qwen-14B: 8xTP1 vs 4xTP2 vs heterogeneous.
#
# WHY THIS MODEL
#   The Qwen3.5-35B-A3B comparison priced training as a fixed constant and had no
#   kernel tables, so only the rollout side was measured. This checkpoint has the
#   full stack profiled on H20 -- vLLM rollout at tp=1 and tp=2 (merged into one
#   psrl artifact), kernel tables at both degrees, and a Megatron THD training fit --
#   so the training block is a measured cost and the span, hence SCG, is trustworthy.
#
#   It is also dense (Qwen2ForCausalLM, 48 layers, 40 heads, 8 KV heads, head_dim 128),
#   so none of the hybrid attention/Mamba KV accounting the 35B needed applies.
#
# KV SIZING IS VALIDATED, NOT ASSUMED
#   Against the two real vLLM servers this repo profiled, the simulated capacity is
#   exact: 276,944 tokens at tp=1 (measured 276,944) and 725,360 at tp=2 (measured
#   725,344), bytes-per-token identical at 196608 / 98304. The weight footprint is
#   read from the checkpoint, so it cannot be inherited from another model -- the bug
#   that made the 35B's tp=1 KV 9.4x too large.
#
# ARMS (8 GPUs in every arm, which is what makes them comparable)
#   all1    8xTP1              homogeneous baseline
#   all2    4xTP2              homogeneous, concentrated
#   half    4xTP1 + 2xTP2      heterogeneous, request_num_balance (TP-blind)
#   half_va 4xTP1 + 2xTP2      heterogeneous, version_capped_tp
#
#   `half` is included on purpose: it is the TP-blind failure mode, and comparing
#   `half_va` only against it would flatter the routing strategy.
#
# ⚠️ READING THE OUTPUT
#   `dryrun.cli.main` prints the SCG(S) curve itself, so this script only relays it.
#   SCG is a RATE in tokens/s, so it is comparable across arms ONLY when every arm
#   consumed the same number of trajectories -- check the `Consumed:` lines agree
#   before comparing any two SCG rows. An arm that ended early divides a smaller
#   token total by a smaller span and scores arbitrarily well.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/scg_14b_fleet_shapes.log}"
mkdir -p "$(dirname -- "${LOG_FILE}")"
exec > >(tee "${LOG_FILE}") 2>&1
echo "Console log:   ${LOG_FILE}"

FIGURE_DIR="${FIGURE_DIR:-${ROOT_DIR}/examples/visualize}"

DRYRUN_ENV="${DRYRUN_ENV:-/apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh}"
if [[ -f "${DRYRUN_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${DRYRUN_ENV}"
fi

MODEL_PATH="${MODEL_PATH:-/apdcephfs_zwfy/share_303760348/lhy/models/DeepSeek-R1-Distill-Qwen-14B}"
COST_DIR="${COST_DIR:-${ROOT_DIR}/outputs/cost/deepseek_r1_distill_qwen_14b}"
ROLLOUT_ARTIFACT="${ROLLOUT_ARTIFACT:-${COST_DIR}/rollout_psrl_both.json}"
TRAINING_ARTIFACT="${TRAINING_ARTIFACT:-${COST_DIR}/training_hydraulis.json}"
TRACE="${TRACE:-${ROOT_DIR}/outputs/workload/long_ctx/joint.jsonl}"
WORK_BASE="${WORK_BASE:-${ROOT_DIR}/outputs/simulate/scg_14b_fleet}"

for required in "${ROLLOUT_ARTIFACT}" "${TRAINING_ARTIFACT}" "${TRACE}"; do
    if [[ ! -f "${required}" ]]; then
        echo "Missing required input: ${required}" >&2
        echo "" >&2
        echo "The rollout artifact must carry BOTH TP degrees. Build it with:" >&2
        echo "  cat outputs/profile/deepseek_r1_distill_qwen_14b/vllm_tp{1,2}/profiles/vllm.jsonl \\" >&2
        echo "    > ${COST_DIR}/vllm_merged.jsonl" >&2
        echo "  dryrun-regress rollout --model psrl --input ${COST_DIR}/vllm_merged.jsonl \\" >&2
        echo "    --output ${ROLLOUT_ARTIFACT}" >&2
        exit 1
    fi
done

# A single-degree artifact would price one arm with the other's coefficients. Checked
# with jq-free python because the repo ships no jq dependency.
python3 -c "
import json, sys
entries = json.load(open('${ROLLOUT_ARTIFACT}'))['entries']
degrees = sorted({int(e['parallelism'].get('tp', 1)) for e in entries})
if not {1, 2}.issubset(degrees):
    sys.exit(f'Artifact carries TP {degrees}; these arms need both 1 and 2.')
print(f'  rollout artifact OK: TP degrees {degrees}')
"

cd "${ROOT_DIR}"

export WORKLOAD=dapo_trace
export POLICY=verl
export ROLLOUT_MODEL=psrl
export ROLLOUT_ARTIFACT
export GPU="${GPU:-h20}"
export MODEL_PATH

# Training is a MEASURED cost here, not a constant.
#
# pp=4 rather than pp=1, because that is the mesh the artifact covers for BOTH
# components. The fit carries latency at tp2/pp1 and tp2/pp4 but memory only at
# tp2/pp4, and the memory lookup is exact on pp (the 1F1B peak depends on which
# stage is in flight, so it cannot be derived by division the way latency can).
# Asking for pp=1 raises on the memory entry. Every arm shares this configuration,
# so the choice does not affect the comparison.
export TRAINING_BACKEND=hydraulis
export TRAINING_ARTIFACT
export TRAINING_TP=2 TRAINING_PP=4 TRAINING_CP=1
# DP widens the trainer. At DP=1 training was 75% of span and the rollout side
# could not matter; DP=4 puts rollout on the critical path (72-78% of span).
export TRAINING_DP="${TRAINING_DP:-4}"
export MICRO_BATCH_SIZE=1
export MAX_TOKENS_PER_MICROBATCH="${MAX_TOKENS_PER_MICROBATCH:-12800}"
export ACTIVATION_RECOMPUTE=full
# Geometry for the unembedding correction comes from TRAINING_ARTIFACT's own
# model descriptor, so there is nothing per-model to set here.
export RECOMPUTE_MODEL=training_forward
# `num_microbatches` is NOT a valid Hydraulis feature name (the fit carries
# `micro_batch_size`); naming it raises before the first step.
export EXTRAPOLATE="${EXTRAPOLATE:-[tokens,activation_bytes,pipeline_slots,sum_squared_tokens,micro_batch_size,profiled_dp]}"

export N_VERSIONS="${N_VERSIONS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export GROUP_SIZE="${GROUP_SIZE:-8}"
export MAX_STALENESS="${MAX_STALENESS:-2}"
export MAX_PROMPT_LENGTH=1024
export MAX_OUTPUT_LENGTH="${CAP:-30000}"
export TOKEN_BUDGET="${CAP:-30000}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-1024}"

# Capacity comes from a real vLLM engine (KV_PROBE defaults to true), cached on
# disk per checkpoint/TP/budget. The non-KV overhead is part of what the engine
# measures, so there is no overhead constant to supply here.
export KV_MODEL_PATH="${MODEL_PATH}"
export KV_GPU_MEMORY_UTILIZATION=0.9

run_arm() {  # label  n_instances  tp_per_instance  router_strategy
    local label=$1 n_inst=$2 tp_arr=$3 strategy=$4
    echo ""
    echo "================ ${label} ================"
    export N_INSTANCES="${n_inst}"
    export OUTPUT_DIR="${WORK_BASE}/${label}"
    # Base degree only; `build_dynamic_parallel` overrides it per candidate.
    export ROLLOUT_TP=1
    export EXTRA_OVERRIDES="\
dynamic_parallel.enabled=true \
dynamic_parallel.n_gpus=8 \
dynamic_parallel.tp_candidates=[1,2] \
dynamic_parallel.strategy=fixed \
dynamic_parallel.tp_per_instance=${tp_arr} \
rollout.max_concurrency=${MAX_CONCURRENCY} \
rollout.router.max_concurrent_seqs_per_instance=40 \
workload.trace.path=${TRACE} \
rollout.router.strategy=${strategy}"
    # Stream the full simulator output so progress, warnings, and per-step logs
    # remain visible while each arm is running.
    bash "${ROOT_DIR}/examples/simulate/simulate.sh" 2>&1
}

run_arm all1    8 "[1,1,1,1,1,1,1,1]" request_num_balance
run_arm all2    4 "[2,2,2,2]"         request_num_balance
run_arm half    6 "[1,1,1,1,2,2]"     request_num_balance
run_arm half_va 6 "[1,1,1,1,2,2]"     version_capped_tp

mkdir -p "${FIGURE_DIR}"
cp "${WORK_BASE}/all1/plots/instance_load.png"    "${FIGURE_DIR}/instance_load_14b_homogeneous_tp1.png"
cp "${WORK_BASE}/all2/plots/instance_load.png"    "${FIGURE_DIR}/instance_load_14b_homogeneous_tp2.png"
cp "${WORK_BASE}/half/plots/instance_load.png"    "${FIGURE_DIR}/instance_load_14b_heterogeneous_blind.png"
cp "${WORK_BASE}/half_va/plots/instance_load.png" "${FIGURE_DIR}/instance_load_14b_heterogeneous_tp.png"

echo ""
echo "Figures:        ${FIGURE_DIR}/instance_load_14b_*.png"
echo "Per-arm plots:  ${WORK_BASE}/<arm>/plots/  (incl. scg_curve.png)"
echo ""
echo "⚠️  Compare the SCG(S) rows above ONLY across arms whose 'Consumed:' agree."
