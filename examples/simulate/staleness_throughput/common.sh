#!/usr/bin/env bash
# The one regime every preset's sweep is measured in.
#
# Sourced by sweep.sh before the preset's grid.sh. Every value is overridable
# from the environment, so a caller can move the whole study to another trace or
# another GPU split without editing twelve grids.
#
# The split and mesh are the ones examples/simulate/psrl_aligned.sh transcribes
# from the real PSRL run, so the cost models are asked for predictions in the
# configuration they were matched against.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export ROOT_DIR

# Pinned rather than `auto`, so every point in a sweep comes from one engine.
export SIMRL_BACKEND="${SIMRL_BACKEND:-cpp}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Near-untruncated DAPO lengths for Qwen3-8B. The dispersion is what generates
# staleness here, so a trace capped lower changes what is being measured.
export TRACE="${TRACE:-${ROOT_DIR}/outputs/workload/qwen3_8b_dapo_38912_tp1/joint.jsonl}"
export MAX_OUTPUT_LENGTH="${MAX_OUTPUT_LENGTH:-38912}"

export BATCH_SIZE="${BATCH_SIZE:-32}"
export GROUP_SIZE="${GROUP_SIZE:-8}"
# Staleness cannot exceed the number of versions a run contains, so a short run
# reports its own length rather than the mechanism.
export N_VERSIONS="${N_VERSIONS:-12}"

export N_INSTANCES="${N_INSTANCES:-2}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export TRAINING_TP="${TRAINING_TP:-2}"
export TRAINING_PP="${TRAINING_PP:-3}"
export TRAINING_DP="${TRAINING_DP:-1}"

RUN_BASE="${RUN_BASE:-${ROOT_DIR}/outputs/simulate/staleness_throughput}"
MAX_PARALLEL="${MAX_PARALLEL:-16}"
REUSE="${REUSE:-true}"

# NOTE(claude): At cap 38912 an unsplittable sequence forms a microbatch wider
# than the training sweep's domain on four features at once (tokens,
# activation_bytes, pipeline_slots, sum_squared_tokens). Every figure carries this footnote.
FOOTNOTE="${FOOTNOTE:-Training cost extrapolated beyond its fit domain (tokens, activation_bytes, pipeline_slots, sum_squared_tokens). Rollout and staleness are in domain.}"
