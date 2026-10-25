#!/usr/bin/env bash
# Simulate the real verl separate_async run this script is aligned against:
#   verl-new/examples/psrl_compare/qwen3_8b_separate_async_megatron.sh
#
# Every number below is transcribed from that script, mirroring how
# psrl_aligned.sh is written against qwen3_8b_megatron_simulate.sh. The two
# files describe one system, so a knob moved there has to move here too.
#
# Unlike psrl_aligned.sh, the real run's workload trace and telemetry are
# already structured JSON (verl-new's telemetry_sink.py), so this script's
# trace extraction calls srl.telemetry.verl_log instead of srl.telemetry.psrl_log.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VERL_LOG_DIR="${VERL_LOG_DIR:-/apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new/examples/psrl_compare/output/telemetry}"
MODEL_PATH="${MODEL_PATH:-/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B}"
TRACE="${TRACE:-${ROOT_DIR}/outputs/workload/verl_aligned/joint.jsonl}"
# rollout.n: responses per prompt. Both the trace's group arity and the
# simulator's group_size come from here so they cannot disagree.
export GROUP_SIZE="${GROUP_SIZE:-8}"

if [[ ! -f "${TRACE}" ]]; then
    if [[ ! -d "${VERL_LOG_DIR}" ]]; then
        echo "Missing both the cached trace and the verl telemetry directory." >&2
        echo "  trace:    ${TRACE}" >&2
        echo "  log dir:  ${VERL_LOG_DIR}" >&2
        echo "Point VERL_LOG_DIR at the run's VERL_TELEMETRY_LOG_DIR, or TRACE at a cached extract." >&2
        exit 1
    fi
    echo "Extracting the workload trace from ${VERL_LOG_DIR}..."
    python - "${VERL_LOG_DIR}" "${TRACE}" "${GROUP_SIZE}" <<'PYTRACE'
import sys
from srl.telemetry.verl_log import extract_workload_trace

extract_workload_trace(
    sys.argv[1], sys.argv[2], group_size=int(sys.argv[3]), max_output_tokens=10240
)
PYTRACE
fi

# --- transcribed from qwen3_8b_separate_async_megatron.sh --------------------
export BATCH_SIZE="${BATCH_SIZE:-32}"
export N_INSTANCES="${N_INSTANCES:-2}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_OUTPUT_LENGTH="${MAX_OUTPUT_LENGTH:-10240}"
export TOKEN_BUDGET="${TOKEN_BUDGET:-12800}"
export N_VERSIONS="${N_VERSIONS:-5}"
export POLICY="${POLICY:-verl_v1}"
export WORKLOAD=dapo_trace

# verl_v1 policy knobs, 1:1 with trainer.v1.separate_async / trainer.v1.sampler
# in the real launch script.
export NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"
export MAX_OFF_POLICY_THRESHOLD="${MAX_OFF_POLICY_THRESHOLD:-1}"
export MAX_OFF_POLICY_STRATEGY="${MAX_OFF_POLICY_STRATEGY:-wait}"

# Training mesh: tensor_model_parallel_size=2, pipeline_model_parallel_size=3,
# context_parallel_size=1, on 6 training GPUs, so DP=1. Identical to PSRL's
# mesh, so the same fitted cost artifact applies unchanged.
export TRAINING_BACKEND=hydraulis
export TRAINING_TP="${TRAINING_TP:-2}"
export TRAINING_PP="${TRAINING_PP:-3}"
export TRAINING_DP="${TRAINING_DP:-1}"
export TRAINING_CP="${TRAINING_CP:-1}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export MAX_TOKENS_PER_MICROBATCH="${MAX_TOKENS_PER_MICROBATCH:-12800}"
export ACTIVATION_RECOMPUTE="${ACTIVATION_RECOMPUTE:-full}"
export TRAINING_ARTIFACT="${TRAINING_ARTIFACT:-${ROOT_DIR}/outputs/cost/qwen3_8b_tp2_pp3_thd/training_hydraulis.json}"

if [[ ! -f "${TRAINING_ARTIFACT}" ]]; then
    echo "Missing training cost artifact for the TP2/PP3 mesh: ${TRAINING_ARTIFACT}" >&2
    echo "This is the same artifact psrl_aligned.sh uses. Build it there first if missing:" >&2
    echo "  CONFIG=examples/profile/megatron/qwen3_8b_thd.yaml CANDIDATE_INDEX=all \\" >&2
    echo "    bash examples/profile/megatron/profile_qwen3_8b.sh run" >&2
    echo "  bash examples/regress/regress_training_hydraulis.sh" >&2
    exit 1
fi

export KV_MODEL_PATH="${KV_MODEL_PATH:-${MODEL_PATH}}"
export KV_GPU_MEMORY_UTILIZATION="${KV_GPU_MEMORY_UTILIZATION:-0.9}"
export RECOMPUTE_MODEL="${RECOMPUTE_MODEL:-training_forward}"
export ROLLOUT_MODEL="${ROLLOUT_MODEL:-composed}"
export KERNEL_ROOT="${KERNEL_ROOT:-${ROOT_DIR}/outputs/profile/qwen3_8b/kernel_tp1/kernel}"
export GPU="${GPU:-h20}"
export MODEL_PATH

export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/simulate/verl_aligned}"
export EXTRAPOLATE="${EXTRAPOLATE:-[tokens,activation_bytes,pipeline_slots,sum_squared_tokens]}"
export EXTRA_OVERRIDES="workload.trace.path=${TRACE} policy.num_warmup_batches=${NUM_WARMUP_BATCHES} policy.max_off_policy_threshold=${MAX_OFF_POLICY_THRESHOLD} policy.max_off_policy_strategy=${MAX_OFF_POLICY_STRATEGY} ${EXTRA_OVERRIDES:-}"

bash "${SCRIPT_DIR}/simulate.sh"
