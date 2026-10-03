#!/usr/bin/env bash
# Simulate the DAPO Qwen3-8B tp=1 workload from a collected joint group trace.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

TRACE="${TRACE:-${ROOT_DIR}/outputs/workload/qwen3_8b_dapo_tp1/joint.jsonl}"
if [[ ! -f "${TRACE}" ]]; then
    echo "Missing joint workload trace: ${TRACE}" >&2
    echo "Collect one first: bash examples/workload/collect_joint_trace.sh" >&2
    exit 1
fi

# PSRL DAPO recipe (psrl/examples/dapo_trainer/qwen3_8b_megatron.sh):
# 64 prompts x n=8 per step, prompt cap 1024, response cap 10240, staleness 2,
# gen tp=1, 16 rollout instances.
export WORKLOAD=dapo_trace
export N_VERSIONS="${N_VERSIONS:-16}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export GROUP_SIZE="${GROUP_SIZE:-8}"
export N_INSTANCES="${N_INSTANCES:-16}"
export MAX_STALENESS="${MAX_STALENESS:-2}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_OUTPUT_LENGTH="${MAX_OUTPUT_LENGTH:-10240}"
export TOKEN_BUDGET="${TOKEN_BUDGET:-40960}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/simulate/qwen3_8b_dapo_tp1}"
export EXTRA_OVERRIDES="workload.trace.path=${TRACE}"

bash "${SCRIPT_DIR}/simulate.sh"
