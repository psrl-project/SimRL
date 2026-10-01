#!/usr/bin/env bash
# Simulate the PSRL Qwen3-8B Megatron baseline as closely as its own logs allow,
# so the instance-load figure can be compared against the measured one.
#
# Every number below is transcribed from the run being matched:
#   psrl/examples/dapo_trainer/qwen3_8b_megatron_baseline.sh
# and the three things that cannot be read off a script: the workload lengths,
# the KV capacity, and the training cost: come from the run's own logs and a
# profile of its exact mesh.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PSRL_LOG_DIR="${PSRL_LOG_DIR:-/apdcephfs_zwfy10_303541817/share_303541817/lhy/psrl/examples/dapo_trainer/megatron_psrl_log/qwen3_8b_megatron_baseline}"
MODEL_PATH="${MODEL_PATH:-/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B}"
TRACE="${TRACE:-${ROOT_DIR}/outputs/workload/psrl_baseline/joint.jsonl}"

if [[ ! -d "${PSRL_LOG_DIR}" ]]; then
    echo "Missing PSRL log directory: ${PSRL_LOG_DIR}" >&2
    echo "It supplies the workload trace and the measured KV capacity." >&2
    exit 1
fi

# The run's own trajectory lengths, rather than a synthetic distribution. Without
# this, workload mismatch and cost-model error are entangled and a divergent KV
# curve cannot be attributed to either.
if [[ ! -f "${TRACE}" ]]; then
    echo "Extracting the workload trace from ${PSRL_LOG_DIR}..."
    python - "${PSRL_LOG_DIR}" "${TRACE}" <<'PYTRACE'
import sys
from dryrun.telemetry.psrl_log import extract_workload_trace

extract_workload_trace(sys.argv[1], sys.argv[2], group_size=8, max_output_tokens=10240)
PYTRACE
fi

# --- transcribed from the baseline script -------------------------------------
# rollout_N=8 responses per prompt, train_batch_size=8 prompts -> 64 responses
# per training step. batch_size counts responses.
export BATCH_SIZE="${BATCH_SIZE:-64}"
# GEN_INSTANCES = (1 node * 2 GPUs) / (GEN_TP=1 * GEN_PP=1) = 2.
export N_INSTANCES="${N_INSTANCES:-2}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
# psrl.staleness=1.
export MAX_STALENESS="${MAX_STALENESS:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_OUTPUT_LENGTH="${MAX_OUTPUT_LENGTH:-10240}"
# gen_actor_rollout_ref.rollout.max_num_batched_tokens=12800.
export TOKEN_BUDGET="${TOKEN_BUDGET:-12800}"
# psrl.rollout_coordination.routing_strategy.max_concurrent_seqs_per_instance=1024.
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-1024}"
# trainer.total_training_steps=10.
export N_VERSIONS="${N_VERSIONS:-10}"
export POLICY="${POLICY:-psrl}"
export WORKLOAD=dapo_trace

# Training mesh: tensor_model_parallel_size=2, pipeline_model_parallel_size=3,
# context_parallel_size=1, on TRAIN_NGPUS_PER_NODE=6, so DP=1.
export TRAINING_BACKEND=hydraulis
export TRAINING_TP="${TRAINING_TP:-2}"
export TRAINING_PP="${TRAINING_PP:-3}"
export TRAINING_DP="${TRAINING_DP:-1}"
export TRAINING_CP="${TRAINING_CP:-1}"
# ppo_micro_batch_size_per_gpu=1.
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"

# The baseline sets use_dynamic_bsz=True with ppo_max_token_len_per_gpu=12800,
# so the trainer packs sequences to a token budget instead of taking one per
# microbatch. Instrumenting psrl's forward_backward_batch measured 44
# microbatches for this recipe, and ceil(563200/12800) = 44 reproduces it, so
# the budget is 12800 as configured: verl's sp_size here is 1, not TP.
#
# Worth a factor rather than a few percent: grouping by count gives 66 pipeline
# slots at PP=3 against the real 46, and the count form overpredicted the
# measured step by 1.84x.
export MAX_TOKENS_PER_MICROBATCH="${MAX_TOKENS_PER_MICROBATCH:-12800}"
# override_transformer_config.recompute_granularity=full.
export ACTIVATION_RECOMPUTE="${ACTIVATION_RECOMPUTE:-full}"

# A hydraulis cell is a measurement of one (TP, PP) pair and cannot be
# interpolated, so this needs the artifact that actually contains TP2/PP3.
export TRAINING_ARTIFACT="${TRAINING_ARTIFACT:-${ROOT_DIR}/outputs/cost/qwen3_8b_tp2_pp3/training_hydraulis.json}"
if [[ ! -f "${TRAINING_ARTIFACT}" ]]; then
    echo "Missing training cost artifact for the TP2/PP3 mesh: ${TRAINING_ARTIFACT}" >&2
    echo "The production sweep covers TP/PP in {1,4,8}x{1,4} and has no TP2/PP3 cell." >&2
    echo "Profile and fit it:" >&2
    echo "  CONFIG=examples/profile/megatron/qwen3_8b_tp2_pp3.yaml CANDIDATE_INDEX=all \\" >&2
    echo "    bash examples/profile/megatron/profile_qwen3_8b.sh run" >&2
    echo "  bash examples/regress/regress_training_hydraulis.sh" >&2
    exit 1
fi

# The KV denominator, computed from the model the way vLLM sizes it rather than
# read back from this run's log: the simulator has to work for configurations
# nobody has executed. gpu_memory_utilization=0.9 is the baseline's
# gen_actor_rollout_ref.rollout.gpu_memory_utilization. The bare formula gives
# 32205 against this engine's measured 31279 (+3.0%); the 2 GiB below is the
# CUDA context and activation peak that no config file exposes, measured once on
# this hardware and reused.
export KV_MODEL_PATH="${KV_MODEL_PATH:-${MODEL_PATH}}"
export KV_GPU_MEMORY_UTILIZATION="${KV_GPU_MEMORY_UTILIZATION:-0.9}"
export KV_NON_KV_OVERHEAD_BYTES="${KV_NON_KV_OVERHEAD_BYTES:-2147483648}"

# The log-prob forward pass, priced as a FLOP-derived fraction of the same fitted
# training step. Measured at 27-39 s per step in the real run, so omitting it
# understates every training block by about a quarter.
export RECOMPUTE_MODEL="${RECOMPUTE_MODEL:-training_forward}"

# Rollout cost. The composed model reads measured kernel tables rather than a
# single end-to-end regression, which is what makes a per-phase prefill/decode
# comparison against TimeSplit_I*.log meaningful.
export ROLLOUT_MODEL="${ROLLOUT_MODEL:-composed}"
export KERNEL_ROOT="${KERNEL_ROOT:-${ROOT_DIR}/outputs/profile/qwen3_8b/kernel_tp1/kernel}"
export GPU="${GPU:-h20}"
export MODEL_PATH

export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/simulate/psrl_baseline_aligned}"
# Strict by default: the TP2/PP3 artifact is now swept at microbatch counts
# {1, 4, 64}, so `pipeline_slots` covers [3, 66] and the baseline's 66 is
# in-domain, as is its 11264-token tail against a 12288 ceiling. Nothing needs
# extrapolating, and leaving this empty means a future config that drifts out of
# domain fails loudly instead of being silently stretched.
export EXTRAPOLATE="${EXTRAPOLATE:-}"
export EXTRA_OVERRIDES="workload.trace.path=${TRACE} rollout.max_concurrency=${MAX_CONCURRENCY}"

bash "${SCRIPT_DIR}/simulate.sh"
