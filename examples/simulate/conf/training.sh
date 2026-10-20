#!/usr/bin/env bash
# conf/training.sh: training cost model configuration
# Sourced by simulate.sh. Sets TRAINING_OVERRIDES.
# Inputs: TRAINING_ARTIFACT, TRAINING_BACKEND, TRAINING_TP, TRAINING_PP, TRAINING_DP,
#         TRAINING_CP, TRAINING_LATENCY_S, MICRO_BATCH_SIZE, ACTIVATION_RECOMPUTE,
#         MAX_TOKENS_PER_MICROBATCH, MAX_TOKENS_PER_RANK, BATCHING_MODE,
#         SHARD_SIZE, REPLICATE_SIZE, EXTRAPOLATE

TRAINING_OVERRIDES=(
    training_cost=fixed
    training_cost.latency_s="${TRAINING_LATENCY_S}"
)
if [[ -z "${TRAINING_ARTIFACT}" ]]; then
    return 0  # nothing more to do, caller keeps the fixed-latency default
fi

if [[ ! -f "${TRAINING_ARTIFACT}" ]]; then
    echo "Missing training cost artifact: ${TRAINING_ARTIFACT}" >&2
    exit 1
fi

# Validate that the requested topology is fitted before the simulator starts.
# conf/check_training_artifact.py prints the fitted token domain and reports a
# clear error rather than a bare KeyError mid-run.
if ! python3 "${CONF_DIR}/check_training_artifact.py" \
        "${TRAINING_ARTIFACT}" \
        "${TRAINING_BACKEND}" \
        "${TRAINING_TP:-1}" \
        "${TRAINING_PP:-1}" \
        "${SHARD_SIZE:-}"; then
    exit 1
fi

if [[ "${TRAINING_BACKEND}" == "fsdp" ]]; then
    # Read the mesh the artifact was fitted on. Guessing would silently pin to a
    # wrong topology or raise TrainingOutOfDomainError.
    if [[ -z "${TRAINING_DP:-}" ]]; then
        TRAINING_DP="$(python3 - "${TRAINING_ARTIFACT}" <<'PYDP'
import json, sys
entries = json.load(open(sys.argv[1]))["entries"]
meshes = {
    int(e["selector"]["shard_size"]) * int(e["selector"]["replicate_size"])
    for e in entries if "shard_size" in e["selector"]
}
print(max(meshes) if meshes else 1)
PYDP
)"
    fi
    SHARD_SIZE="${SHARD_SIZE:-${TRAINING_DP}}"
    REPLICATE_SIZE="${REPLICATE_SIZE:-$(( TRAINING_DP / SHARD_SIZE ))}"
    TRAINING_OVERRIDES=(
        training_cost=fsdp
        training_cost.shard_size="${SHARD_SIZE}"
        training_cost.replicate_size="${REPLICATE_SIZE}"
        training_cost.artifact_path="${TRAINING_ARTIFACT}"
        training_cost.schedule=fsdp
        training_cost.batching_mode="${BATCHING_MODE:-pad_to_max}"
        training_cost.parallelism.tp=1
        training_cost.parallelism.pp=1
        training_cost.parallelism.dp="${TRAINING_DP}"
        training_cost.parallelism.cp=1
        training_cost.micro_batch_size=1
        training_cost.activation_recompute="${ACTIVATION_RECOMPUTE:-full}"
    )
    if [[ -n "${MAX_TOKENS_PER_RANK:-}" ]]; then
        TRAINING_OVERRIDES+=( training_cost.max_tokens_per_rank="${MAX_TOKENS_PER_RANK}" )
    fi
else
    # Hydraulis: (TP, PP) is the measured cell. TP is required, and PP is derived.
    TRAINING_OVERRIDES=(
        training_cost=hydraulis
        training_cost.artifact_path="${TRAINING_ARTIFACT}"
        training_cost.parallelism.tp="${TRAINING_TP:-1}"
        training_cost.parallelism.pp="${TRAINING_PP:-1}"
        training_cost.parallelism.dp="${TRAINING_DP:-1}"
        training_cost.parallelism.cp="${TRAINING_CP:-1}"
        training_cost.micro_batch_size="${MICRO_BATCH_SIZE:-1}"
        training_cost.activation_recompute="${ACTIVATION_RECOMPUTE:-full}"
    )
    if [[ -n "${MAX_TOKENS_PER_MICROBATCH:-}" ]]; then
        TRAINING_OVERRIDES+=( "training_cost.max_tokens_per_microbatch=${MAX_TOKENS_PER_MICROBATCH}" )
    fi
fi

if [[ -n "${EXTRAPOLATE:-}" ]]; then
    TRAINING_OVERRIDES+=( "training_cost.extrapolate=${EXTRAPOLATE}" )
fi
