#!/usr/bin/env bash
# Run one simulation with a fully configured cost model.
#
# Every concern is handled in a dedicated conf/ file, which this script sources
# in order. Each conf/ file reads env vars and sets one array of Hydra overrides;
# this script concatenates them and passes them to srl.cli.main.
#
# QUICKSTART (14B with Megatron training, psrl rollout):
#   MODEL_PATH=... \
#   ROLLOUT_ARTIFACT=outputs/cost/deepseek_r1_distill_qwen_14b/rollout_psrl_both.json \
#   TRAINING_ARTIFACT=outputs/cost/deepseek_r1_distill_qwen_14b/training_hydraulis.json \
#   TRAINING_TP=2 TRAINING_PP=4 TRAINING_DP=4 \
#   RECOMPUTE_MODEL=training_forward \
#   KV_MODEL_PATH="$MODEL_PATH" \
#   bash examples/simulate/simulate.sh
#
# ENV VARS  (all optional; override any default)
# -----------------------------------------------
#   ROLLOUT_MODEL        psrl | distserve | roofline | composed  (default: psrl)
#   ROLLOUT_ARTIFACT     path to the fitted rollout artifact
#   MODEL_PATH           HuggingFace checkpoint dir (composed only)
#   KERNEL_ROOT          kernel perf table dir (composed only)
#   ROLLOUT_TP           tensor-parallel degree of the inference engine (default: 1)
#   GPU                  GPU label used in table lookups, e.g. h20 (default: h20)
#
#   WORKLOAD             normal | dapo_trace | uniform | ...  (default: normal)
#   DIST_FILE            distributions.yaml path (normal workload only)
#
#   TRAINING_ARTIFACT    path to the fitted training artifact (omit = fixed latency)
#   TRAINING_BACKEND     hydraulis | fsdp  (default: hydraulis)
#   TRAINING_TP / PP / DP / CP
#   TRAINING_LATENCY_S   fallback when no artifact (default: 50)
#   MICRO_BATCH_SIZE     (hydraulis, default: 1)
#   MAX_TOKENS_PER_MICROBATCH
#   ACTIVATION_RECOMPUTE full | none  (default: full)
#   EXTRAPOLATE          bracket list of feature names allowed outside fit domain
#   BATCHING_MODE        pad_to_max | token_budget | packed_no_pad  (fsdp, default: pad_to_max)
#   SHARD_SIZE / REPLICATE_SIZE  (fsdp)
#   MAX_TOKENS_PER_RANK  (fsdp)
#
#   KV_MODEL_PATH        path to the checkpoint for KV sizing (omit = use kv_blocks literal)
#   KV_GPU_MEMORY_UTILIZATION   (default: 0.9)
#   KV_GPU_MEMORY_MIB           (default: 97871 — one H20)
#   KV_PROBE             true | false  start vLLM to measure KV capacity (default: true)
#   KV_NON_KV_OVERHEAD_BYTES    KV_PROBE=false only: the non-KV footprint the
#                               arithmetic path cannot derive (default: 0). A
#                               probe measures it, so this is ignored then.
#
#   RECOMPUTE_MODEL      training_forward | analytical | (empty = fixed 0)
#                        training_forward reads the model's geometry from the
#                        training artifact; there is nothing per-model to set.
#   RECOMPUTE_TIME_PER_TOKEN                              (fixed)
#   RECOMPUTE_MODEL_PARAMS / PEAK_FLOPS / OVERHEAD_FACTOR (analytical)
#
#   N_VERSIONS           training steps to simulate (default: 20)
#   BATCH_SIZE           prompt groups per step (default: 256)
#   GROUP_SIZE           responses per prompt; empty reads from workload
#   N_INSTANCES          rollout engine count (default: 4)
#   MAX_STALENESS        (default: 2)
#   MAX_PROMPT_LENGTH    (default: 2048)
#   MAX_OUTPUT_LENGTH    (default: 16384)
#   TOKEN_BUDGET         per-instance token budget (default: 65536)
#   OUTPUT_DIR           where logs and plots go
#
#   EXTRA_OVERRIDES      additional Hydra overrides, space-separated
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONF_DIR="${SCRIPT_DIR}/conf"
export CONF_DIR  # conf/training.sh needs it to find check_training_artifact.py

# ── defaults ──────────────────────────────────────────────────────────────────
DIST_FILE="${DIST_FILE:-${SCRIPT_DIR}/distributions.yaml}"
WORKLOAD="${WORKLOAD:-normal}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

ROLLOUT_MODEL="${ROLLOUT_MODEL:-psrl}"
POLICY="${POLICY:-psrl}"
COST_DIR="${COST_DIR:-${ROOT_DIR}/outputs/cost/qwen3_8b}"
ROLLOUT_ARTIFACT="${ROLLOUT_ARTIFACT:-${COST_DIR}/rollout_${ROLLOUT_MODEL}.json}"
TRAINING_ARTIFACT="${TRAINING_ARTIFACT:-}"
TRAINING_LATENCY_S="${TRAINING_LATENCY_S:-50}"
TRAINING_BACKEND="${TRAINING_BACKEND:-hydraulis}"
BATCHING_MODE="${BATCHING_MODE:-pad_to_max}"
TRAINING_DP="${TRAINING_DP:-}"
EXTRAPOLATE="${EXTRAPOLATE:-}"

N_VERSIONS="${N_VERSIONS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GROUP_SIZE="${GROUP_SIZE:-}"
N_INSTANCES="${N_INSTANCES:-4}"
MAX_STALENESS="${MAX_STALENESS:-2}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_OUTPUT_LENGTH="${MAX_OUTPUT_LENGTH:-16384}"
TOKEN_BUDGET="${TOKEN_BUDGET:-65536}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/simulate/qwen3_8b_fitted_cost}"

KERNEL_ROOT="${KERNEL_ROOT:-${ROOT_DIR}/outputs/profile/qwen3_8b/kernel_tp1/kernel}"
MODEL_PATH="${MODEL_PATH:-}"
GPU="${GPU:-h20}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"

# ── source the five concern files ─────────────────────────────────────────────
# Each sets one array of Hydra overrides; we concatenate them below.
# shellcheck source=conf/rollout.sh
source "${CONF_DIR}/rollout.sh"
# shellcheck source=conf/workload.sh
source "${CONF_DIR}/workload.sh"
# shellcheck source=conf/training.sh
source "${CONF_DIR}/training.sh"
# shellcheck source=conf/kv.sh
source "${CONF_DIR}/kv.sh"
# shellcheck source=conf/recompute.sh
source "${CONF_DIR}/recompute.sh"

# ── invoke the simulator ───────────────────────────────────────────────────────
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
python -m srl.cli.main \
    workload="${WORKLOAD}" \
    policy="${POLICY}" \
    rollout_cost="${ROLLOUT_MODEL}" \
    "${COST_OVERRIDES[@]}" \
    rollout_cost.parallelism.tp="${ROLLOUT_TP}" \
    rollout_cost.parallelism.pp=1 \
    rollout_cost.parallelism.dp=1 \
    rollout_cost.parallelism.cp=1 \
    "${DIST_OVERRIDES[@]+"${DIST_OVERRIDES[@]}"}" \
    "${TRAINING_OVERRIDES[@]}" \
    job.n_versions="${N_VERSIONS}" \
    job.batch_size="${BATCH_SIZE}" \
    ${GROUP_SIZE:+job.group_size="${GROUP_SIZE}"} \
    job.max_staleness="${MAX_STALENESS}" \
    job.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    job.max_output_length="${MAX_OUTPUT_LENGTH}" \
    rollout.n_instances="${N_INSTANCES}" \
    rollout.token_budget="${TOKEN_BUDGET}" \
    "${KV_OVERRIDES[@]+"${KV_OVERRIDES[@]}"}" \
    "${RECOMPUTE_OVERRIDES[@]+"${RECOMPUTE_OVERRIDES[@]}"}" \
    output_dir="${OUTPUT_DIR}" \
    ${EXTRA_OVERRIDES:+${EXTRA_OVERRIDES}}
