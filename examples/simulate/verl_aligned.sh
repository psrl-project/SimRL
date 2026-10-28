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
#
# Current agreement (srl-compare against output/telemetry_drop, the run's
# own converted logs), at the MAX_OFF_POLICY_STRATEGY=drop default below,
# which is what the launch script itself now runs by default: span +1.1%,
# train/step +7.9%, recompute/step -16.3%, sync/step -100.0% (simulated
# sync_time is always 0 -- weight sync is not modeled as its own phase, a
# small, separate gap from the one this comment used to describe here),
# peak running requests +0.0%, mean running (busy) -4.7%, peak KV util
# +0.0%, mean KV util (busy) -5.7%, idle fraction 0.032 vs 0.015. Train/step
# timing is not part of this comparison beyond what's above. convert_verl_logs
# writes no train stream for a real verl run (see srl/telemetry/verl_log.py's
# own docstring), so there is nothing on the real side for srl-compare's
# train/step column to compare against beyond the timing_s/* fields it does
# carry.
#
# This comment used to report a much larger gap here (span -18.3%, idle
# fraction simulated 0.261 vs real 0.015), diagnosed as SimRL's verl_v1
# policy granting a new version's full batch_size*group_size admission
# credit only once per train-version advance (admit_quota), identically for
# wait and drop -- under-modeling the inter-version overlap separate_async's
# GPU disaggregation is meant to buy under drop (rollout keeps admitting the
# NEXT version's prompts on its own GPU pool while the CURRENT version's
# recompute/train still runs on the training pool). That diagnosis held:
# `cdb049e "grant verl_v1 admission credit on drop eviction, not just
# version-advance"` now grants a fresh admission credit on every per-group
# drop eviction too (matching ReplayBufferAsync.wait_for_sampleable's
# refill_fn, called in the same loop iteration as the eviction), not only on
# a full version-advance. Separately -- and this is why re-measuring the old
# numbers directly would not have reproduced them even pre-fix -- the
# canonical output_dir this script defaults to had gone stale: it last held
# a run from 2026-10-25 09:13, before output/telemetry_drop existed
# (2026-10-26 01:35) and before the launch script's own default switched
# wait->drop, so it was silently comparing a leftover wait-strategy
# simulation against drop-strategy real telemetry. VERL_LOG_DIR below
# defaults to telemetry_drop, not the plain telemetry/ directory, for this
# reason -- telemetry/ still holds that same stale pre-switch wait capture.
#
# Both real and simulated drop runs independently confirm the system runs at
# staleness 0, not 1: the real run's own training/off_policy/trajectory_
# staleness/mean reports 0.0 for consumed samples (evicted_samples_staleness/
# mean is 2.0 -- that's the staleness at which drop evicts, not what gets
# trained on), and the simulated run's own staleness distribution is 100% at
# staleness 0. This is a genuinely different regime from the wait-strategy
# numbers further down: under drop, nothing stale ever survives to be
# consumed, so NUM_WARMUP_BATCHES needs no offset -- the literal
# trainer.v1.separate_async.num_warmup_batches=1 from the launch script
# reproduces this directly. The simulated run also confirmed drop's refill
# semantics against the real run's own telemetry: both dropped exactly one
# batch's worth of groups (32 groups x 8 = 256 requests simulated vs the real
# run's own reported training/off_policy/evicted_samples:32.0 at the step it
# happened), matching verl's ReplayBufferAsync.wait_for_sampleable(), which
# evicts stale terminal groups and calls refill_fn in the same loop
# iteration, and SimRL's OffPolicySpan rule, which cancels with
# CancelAction.REPLACE (grants one new ingress credit per cancellation, not a
# plain drop) -- same evict-and-immediately-replace semantics on both sides.
#
# Separately, and corrected from an earlier version of this comment: zero
# temporal overlap between generation and training is a WAIT-specific
# effect, not a universal one forced by parameter_sync_step=1. Under wait,
# the real run's own instance streams do go idle before every train step (see
# the idle-fraction paragraph above) -- wait's threshold check blocks new
# admission outright until the stale buffer clears, which serializes rollout
# and training regardless of parameter_sync_step. Under drop, the real run's
# own instance streams show no such idle stretches: eviction-and-immediate-
# refill never blocks admission, so the next version's rollout overlaps the
# current version's training on verl's separate GPU pools, exactly as
# separate_async's disaggregation is meant to allow. Data-version staleness
# (which weight version generated a sample) and wall-clock overlap (whether
# generation and training run concurrently) remain independent concepts --
# this config has off-policy staleness control active (threshold=1) under
# either strategy -- but whether that control manifests as idle time or as
# genuine overlap depends entirely on wait vs drop, which is also why this
# script's own simulated-vs-real idle-fraction gap under drop matters: the
# simulator is not yet capturing the overlap drop actually buys on real
# hardware.
#
# The historical wait-strategy numbers (and the NUM_WARMUP_BATCHES=2
# offset finding that applied only to wait) are preserved below the
# MAX_OFF_POLICY_STRATEGY export, for anyone who passes
# MAX_OFF_POLICY_STRATEGY=wait explicitly.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VERL_LOG_DIR="${VERL_LOG_DIR:-/apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new/examples/psrl_compare/output/telemetry_drop}"
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

# verl_v1 policy knobs. MAX_OFF_POLICY_THRESHOLD and MAX_OFF_POLICY_STRATEGY
# are 1:1 with trainer.v1.sampler in the real launch script. The launch
# script itself now defaults to drop (see its own header for why), so this
# defaults to drop too -- pass MAX_OFF_POLICY_STRATEGY=wait explicitly to
# reproduce the historical wait-strategy numbers below instead. That
# historical run's own trace came from output/telemetry (not telemetry_drop,
# this script's own default since the header fix above) -- pass
# VERL_LOG_DIR back to the plain telemetry/ directory too if reproducing it
# from scratch, so the workload trace and the strategy being simulated come
# from the same real run.
#
# NUM_WARMUP_BATCHES=1 is the literal
# trainer.v1.separate_async.num_warmup_batches=1 from the launch script, and
# needs no offset under drop (see the drop agreement note above).
#
# Historical note, applies only when MAX_OFF_POLICY_STRATEGY=wait: a literal
# NUM_WARMUP_BATCHES=1 collapsed rollout and training into lockstep under
# wait, because SimRL's admit_quota grants exactly num_warmup_batches *
# batch_size groups of credit up front and replenishes only batch_size more
# per train-version advance, leaving no spare credit for a second batch to
# admit while the first one trains -- the simulated run showed zero
# staleness on every one of its 1280 requests and a peak running-requests
# count of 128 against the real run's 256 (-50.0%). The real run's own
# telemetry reports training/off_policy/trajectory_staleness/mean = 1.0 for
# 4 of its 5 steps (0.0 only for the first step, which has no prior version
# to be stale against), so the two systems appear to count num_warmup_batches
# with a one-batch offset between them under wait (inferred from this
# evidence, not verified against verl's own sampler source). Passing
# NUM_WARMUP_BATCHES=2 with MAX_OFF_POLICY_STRATEGY=wait reproduces that same
# 20/80 split (256 requests at staleness 0, 1024 at staleness 1, mean 0.80)
# and brings peak running requests to an exact match. Raising
# MAX_OFF_POLICY_THRESHOLD instead of NUM_WARMUP_BATCHES was tried first and
# measured to have no effect at all here. admit_quota's credit, not the span
# check, is what was forcing lockstep, so threshold alone could not fix it.
export NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"
export MAX_OFF_POLICY_THRESHOLD="${MAX_OFF_POLICY_THRESHOLD:-1}"
export MAX_OFF_POLICY_STRATEGY="${MAX_OFF_POLICY_STRATEGY:-drop}"

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
