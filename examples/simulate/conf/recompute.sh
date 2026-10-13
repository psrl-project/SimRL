#!/usr/bin/env bash
# conf/recompute.sh — log-prob recomputation cost overrides
# Sourced by simulate.sh. Sets RECOMPUTE_OVERRIDES.
# Inputs: RECOMPUTE_MODEL, RECOMPUTE_TIME_PER_TOKEN,
#         RECOMPUTE_MODEL_PARAMS, RECOMPUTE_PEAK_FLOPS, RECOMPUTE_OVERHEAD_FACTOR

RECOMPUTE_OVERRIDES=()
if [[ "${RECOMPUTE_MODEL:-}" == "training_forward" ]]; then
    # The unembedding correction's geometry is read from the training artifact,
    # so there is nothing to configure. It used to default to Qwen3-8B's
    # constants, which silently mispriced every other model.
    RECOMPUTE_OVERRIDES=(
        recompute_cost.name=training_forward
    )
elif [[ "${RECOMPUTE_MODEL:-}" == "analytical" ]]; then
    RECOMPUTE_OVERRIDES=(
        recompute_cost.name=analytical
        "recompute_cost.model_params=${RECOMPUTE_MODEL_PARAMS:-8190735360}"
        "recompute_cost.peak_flops=${RECOMPUTE_PEAK_FLOPS:-148000000000000}"
        "recompute_cost.overhead_factor=${RECOMPUTE_OVERHEAD_FACTOR:-1.1}"
    )
elif [[ -n "${RECOMPUTE_TIME_PER_TOKEN:-}" ]]; then
    RECOMPUTE_OVERRIDES=(
        recompute_cost.name=fixed
        "recompute_cost.time_per_token=${RECOMPUTE_TIME_PER_TOKEN}"
    )
fi
