#!/usr/bin/env bash
# Simulate the PSRL run that SimRL is aligned against.
#
# Every number below is transcribed from the run being matched:
#   psrl/examples/dapo_trainer/qwen3_8b_megatron_simulate.sh
# and the three things that cannot be read off a script (the workload lengths,
# the KV capacity, and the training cost) come from the run's own logs and a
# profile of its exact mesh. The two files describe one system, so a knob moved
# there has to move here too.
#
# Why `train_batch_size=32` and not a smaller batch: it is what puts the KV
# cache under real pressure. In the matched run 28.9% of routed requests observe
# `kv_cache_usage >= 0.9`, 20.7% observe `>= 0.99`, the maximum is exactly 1.0,
# and 383 of 1663 requests finish with `finish_reason="abort"` and come back
# with `iteration_count=2`. At batch 8 the cache never fills, so preemption,
# eviction and re-prefill go unexercised and a simulator could get all three
# wrong while still appearing to match.
#
# Current agreement (srl-compare against the run's own logs, measured at
# ACTIVE_CAP=50): train/step +1.9%, peak KV utilisation +0.0%, mean KV
# utilisation -2.5%, mean running requests +2.7%, span +5.0%. Not yet
# re-measured against the ACTIVE_CAP=0 default below.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PSRL_LOG_DIR="${PSRL_LOG_DIR:-/apdcephfs_zwfy10_303541817/share_303541817/lhy/psrl/examples/dapo_trainer/megatron_psrl_log/qwen3_8b_megatron_simulate}"
MODEL_PATH="${MODEL_PATH:-/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B}"
TRACE="${TRACE:-${ROOT_DIR}/outputs/workload/psrl_aligned/joint.jsonl}"
# rollout_N: responses per prompt. The trace's group arity and the simulator's
# group_size are both taken from here so they cannot disagree.
export GROUP_SIZE="${GROUP_SIZE:-8}"

# The run's own trajectory lengths, rather than a synthetic distribution. Without
# this, workload mismatch and cost-model error are entangled and a divergent KV
# curve cannot be attributed to either. 160 groups of 8, median output 11057
# tokens, 60.4% truncated at the cap.
if [[ ! -f "${TRACE}" ]]; then
    if [[ ! -d "${PSRL_LOG_DIR}" ]]; then
        echo "Missing both the cached trace and the PSRL log directory." >&2
        echo "  trace:    ${TRACE}" >&2
        echo "  log dir:  ${PSRL_LOG_DIR}" >&2
        echo "Point PSRL_LOG_DIR at the run's psrl.logging_path, or TRACE at a cached extract." >&2
        exit 1
    fi
    echo "Extracting the workload trace from ${PSRL_LOG_DIR}..."
    python - "${PSRL_LOG_DIR}" "${TRACE}" "${GROUP_SIZE}" <<'PYTRACE'
import sys
from srl.telemetry.psrl_log import extract_workload_trace

extract_workload_trace(
    sys.argv[1], sys.argv[2], group_size=int(sys.argv[3]), max_output_tokens=10240
)
PYTRACE
fi

# --- transcribed from qwen3_8b_megatron_simulate.sh ---------------------------
# train_batch_size=32 prompts, the setting that saturates the KV cache.
export BATCH_SIZE="${BATCH_SIZE:-32}"
# GEN_INSTANCES = (1 node * 2 GPUs) / (GEN_TP=1 * GEN_PP=1) = 2.
export N_INSTANCES="${N_INSTANCES:-2}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_OUTPUT_LENGTH="${MAX_OUTPUT_LENGTH:-10240}"
# gen_actor_rollout_ref.rollout.max_num_batched_tokens=12800.
export TOKEN_BUDGET="${TOKEN_BUDGET:-12800}"
# psrl.rollout_coordination.routing_strategy.max_concurrent_seqs_per_instance=1024.
# This knob lives in psrl's coordinator, not in the engine, so it belongs on the
# router: it decides eligibility before a request is offered, and a rejected
# request dwells in the router queue instead of the instance's own waiting queue.
# At 1024 it never binds here, since a step routes 32*8=256 requests across 2
# instances, so it is transcribed for fidelity rather than for its effect.
# Defaults to disabled (0), matching the real run's cap effectively never
# binding. Set it to a positive number to study a tighter per-instance
# residency limit instead; examples/visualize/instance_load_simulated.png was
# generated with ACTIVE_CAP=50 (each instance capped at 50 resident requests).
export ACTIVE_CAP="${ACTIVE_CAP:-0}"
# vLLM's own max_num_seqs. psrl leaves it at the engine default, which is 1024
# for this build, and it is a separate limit from the router cap above.
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-1024}"
# vLLM's preemption_notification_threshold: instance queue depth past which a
# preempted request is handed back to the router and may be re-placed elsewhere.
# Empty leaves every preemption local, which is what the matched run does. Its
# 2122 preemptions all requeue in place. This is the only path by which work
# crosses instances mid-generation, so setting it changes migration, not just
# queueing, and each migration pays a full re-prefill of the reclaimed context.
export MAX_WAITING_AFTER_PREEMPT="${MAX_WAITING_AFTER_PREEMPT:-1024}"
# Keep a preemption loopback out of an instance that already has a local queue.
# Fresh admissions and weight-update loopbacks still see the normal candidate
# set and are ranked by the configured routing strategy.
export ROUTER_REJECT_PREEMPTED_ON_WAITING="${ROUTER_REJECT_PREEMPTED_ON_WAITING:-true}"
# trainer.total_training_steps=5.
export N_VERSIONS="${N_VERSIONS:-5}"
export POLICY="${POLICY:-verl}"
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
# use_dynamic_bsz=True with ppo_max_token_len_per_gpu=12800, so the trainer packs
# sequences to a token budget instead of taking one per microbatch. verl's
# sp_size here is 1, not TP, so the budget is 12800 as configured.
export MAX_TOKENS_PER_MICROBATCH="${MAX_TOKENS_PER_MICROBATCH:-12800}"
# override_transformer_config.recompute_granularity=full.
export ACTIVATION_RECOMPUTE="${ACTIVATION_RECOMPUTE:-full}"

# A hydraulis cell is a measurement of one (TP, PP) pair and cannot be
# interpolated, so this needs the artifact that actually contains TP2/PP3.
# THD, not BSHD. PSRL trains with `use_remove_padding=True`, which concatenates
# a microbatch behind `cu_seqlens`, so attention costs O(sum s_i^2) rather than
# the padded O(max_len^2) a BSHD sweep measures. Those are different functions of
# the same batch: pointing this at the BSHD cell mispriced the training step by
# +60.1%, and the only change needed was this path.
export TRAINING_ARTIFACT="${TRAINING_ARTIFACT:-${ROOT_DIR}/outputs/cost/qwen3_8b_tp2_pp3_thd/training_hydraulis.json}"

if [[ ! -f "${TRAINING_ARTIFACT}" ]]; then
    echo "Missing training cost artifact for the TP2/PP3 mesh: ${TRAINING_ARTIFACT}" >&2
    echo "The production sweep covers TP/PP in {1,4,8}x{1,4} and has no TP2/PP3 cell." >&2
    echo "Profile and fit it:" >&2
    echo "  CONFIG=examples/profile/megatron/qwen3_8b_thd.yaml CANDIDATE_INDEX=all \\" >&2
    echo "    bash examples/profile/megatron/profile_qwen3_8b.sh run" >&2
    echo "  bash examples/regress/regress_training_hydraulis.sh" >&2
    exit 1
fi

# The KV denominator, measured by a real vLLM engine and cached per
# checkpoint/TP/budget rather than read back from this run's log: the simulator
# has to answer for configurations nobody has executed. The engine's own
# measurement covers the non-KV overhead. gpu_memory_utilization=0.9 is this
# run's gen_actor_rollout_ref.rollout.gpu_memory_utilization.
export KV_MODEL_PATH="${KV_MODEL_PATH:-${MODEL_PATH}}"
export KV_GPU_MEMORY_UTILIZATION="${KV_GPU_MEMORY_UTILIZATION:-0.9}"

# The log-prob forward pass, priced as a FLOP-derived fraction of the same fitted
# training step. The real run measures 82.31 s per step here. The simulation
# currently prices it 11.2% low, which is tracked separately from the step itself.
export RECOMPUTE_MODEL="${RECOMPUTE_MODEL:-training_forward}"

# Rollout cost. The composed model reads measured kernel tables rather than a
# single end-to-end regression, which is what makes a per-phase prefill/decode
# comparison against TimeSplit_I*.log meaningful.
export ROLLOUT_MODEL="${ROLLOUT_MODEL:-composed}"
export KERNEL_ROOT="${KERNEL_ROOT:-${ROOT_DIR}/outputs/profile/qwen3_8b/kernel_tp1/kernel}"
export GPU="${GPU:-h20}"
export MODEL_PATH

export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/simulate/psrl_aligned}"
# This batch packs more tokens into a training microbatch than the TP2/PP3 sweep
# covers, and the activation footprint follows. Every exceedance is recorded in
# the telemetry rather than hidden. See docs/simulation/domain-gates.
# Four features exceed the sweep here, all for the same reason: more prompts
# means more and longer microbatches than were profiled.
#   tokens              per-microbatch token count
#   activation_bytes    follows tokens
#   pipeline_slots      191 microbatches against a [3, 66] sweep
#   sum_squared_tokens  the quadratic attention term, 7.9e9 against 3.3e9
# `pipeline_slots` is the best behaved (the form is exactly linear in it, one
# 1F1B stage occupancy each). `sum_squared_tokens` is the worst: it is the
# attention term itself, so a 2.4x exceedance is 2.4x of the part of the fit
# that is genuinely quadratic. Treat this run's *training* time as indicative
# only. Its rollout numbers are in domain and are the point of the comparison.
export EXTRAPOLATE="${EXTRAPOLATE:-[tokens,activation_bytes,pipeline_slots,sum_squared_tokens]}"
# Appended, not assigned: a caller studying one knob passes it in EXTRA_OVERRIDES,
# and overwriting here discarded it silently. Every arm then ran the default
# router and matched the baseline exactly, which reads as a null result.
export EXTRA_OVERRIDES="workload.trace.path=${TRACE} rollout.max_concurrency=${MAX_CONCURRENCY} rollout.router.max_concurrent_seqs_per_instance=${ACTIVE_CAP} rollout.router.max_waiting_after_preempt=${MAX_WAITING_AFTER_PREEMPT} rollout.router.reject_preempted_on_waiting=${ROUTER_REJECT_PREEMPTED_ON_WAITING} ${EXTRA_OVERRIDES:-}"

bash "${SCRIPT_DIR}/simulate.sh"
