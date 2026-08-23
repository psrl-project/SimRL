#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIST_FILE="${DIST_FILE:-${SCRIPT_DIR}/distributions.yaml}"

ROLLOUT_MODEL="${ROLLOUT_MODEL:-psrl}"
POLICY="${POLICY:-psrl}"
COST_DIR="${COST_DIR:-${ROOT_DIR}/outputs/cost/qwen3_8b}"
ROLLOUT_ARTIFACT="${ROLLOUT_ARTIFACT:-${COST_DIR}/rollout_${ROLLOUT_MODEL}.json}"
TRAINING_ARTIFACT="${TRAINING_ARTIFACT:-}"
TRAINING_LATENCY_S="${TRAINING_LATENCY_S:-50}"

N_VERSIONS="${N_VERSIONS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
N_INSTANCES="${N_INSTANCES:-4}"
MAX_STALENESS="${MAX_STALENESS:-2}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_OUTPUT_LENGTH="${MAX_OUTPUT_LENGTH:-16384}"
TOKEN_BUDGET="${TOKEN_BUDGET:-65536}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/simulate/qwen3_8b_fitted_cost}"

if [[ ! -f "${ROLLOUT_ARTIFACT}" ]]; then
    echo "Missing rollout cost artifact: ${ROLLOUT_ARTIFACT}" >&2
    echo "Fit one first, for example: bash examples/regress/regress_rollout_${ROLLOUT_MODEL}.sh" >&2
    exit 1
fi

if [[ ! -f "${DIST_FILE}" ]]; then
    echo "Missing distribution config: ${DIST_FILE}" >&2
    exit 1
fi

DIST_OVERRIDES=()
while IFS= read -r line; do
    DIST_OVERRIDES+=("${line}")
done < <(python "${SCRIPT_DIR}/emit_workload_overrides.py" "${DIST_FILE}")

if [[ ${#DIST_OVERRIDES[@]} -eq 0 ]]; then
    echo "No workload overrides emitted from ${DIST_FILE}." >&2
    exit 1
fi

TRAINING_OVERRIDES=(
    training_cost=fixed
    training_cost.latency_s="${TRAINING_LATENCY_S}"
)
if [[ -n "${TRAINING_ARTIFACT}" ]]; then
    if [[ ! -f "${TRAINING_ARTIFACT}" ]]; then
        echo "Missing training cost artifact: ${TRAINING_ARTIFACT}" >&2
        exit 1
    fi
    TRAINING_OVERRIDES=(
        training_cost=hydraulis
        training_cost.artifact_path="${TRAINING_ARTIFACT}"
        training_cost.parallelism.tp=1
        training_cost.parallelism.pp=1
        training_cost.parallelism.dp=1
        training_cost.parallelism.cp=1
        training_cost.micro_batch_size=1
    )
fi

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
python -m dryrun.cli.main \
    workload=normal \
    policy="${POLICY}" \
    rollout_cost="${ROLLOUT_MODEL}" \
    rollout_cost.artifact_path="${ROLLOUT_ARTIFACT}" \
    rollout_cost.parallelism.tp=1 \
    rollout_cost.parallelism.pp=1 \
    rollout_cost.parallelism.dp=1 \
    rollout_cost.parallelism.cp=1 \
    "${DIST_OVERRIDES[@]}" \
    "${TRAINING_OVERRIDES[@]}" \
    job.n_versions="${N_VERSIONS}" \
    job.batch_size="${BATCH_SIZE}" \
    job.max_staleness="${MAX_STALENESS}" \
    job.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    job.max_output_length="${MAX_OUTPUT_LENGTH}" \
    rollout.n_instances="${N_INSTANCES}" \
    rollout.token_budget="${TOKEN_BUDGET}" \
    output_dir="${OUTPUT_DIR}"
