#!/usr/bin/env bash
# conf/rollout.sh: rollout cost model configuration
# Sourced by simulate.sh. Sets COST_OVERRIDES.
# Inputs: ROLLOUT_MODEL, ROLLOUT_ARTIFACT, KERNEL_ROOT, MODEL_PATH, GPU, ROLLOUT_TP

if [[ "${ROLLOUT_MODEL}" == "composed" ]]; then
    if [[ ! -d "${KERNEL_ROOT}" ]]; then
        echo "Missing kernel perf tables: ${KERNEL_ROOT}" >&2
        echo "Profile them first: bash examples/profile/kernel/profile_qwen3_8b.sh all" >&2
        exit 1
    fi
    if [[ -z "${MODEL_PATH}" ]]; then
        echo "ROLLOUT_MODEL=composed needs a checkpoint to read the architecture from." >&2
        echo "Set MODEL_PATH=/path/to/Qwen3-8B" >&2
        exit 1
    fi
    if [[ "${ROLLOUT_TP}" -gt 1 ]]; then
        COLLECTIVE_DIR="${KERNEL_ROOT}/${GPU}/collective"
        if [[ ! -d "${COLLECTIVE_DIR}" ]]; then
            echo "ROLLOUT_TP=${ROLLOUT_TP} needs a collective table, but ${COLLECTIVE_DIR} does not exist." >&2
            echo "Sweep it: TP=${ROLLOUT_TP} FAMILY=collective bash examples/profile/kernel/profile_qwen3_8b.sh all" >&2
            exit 1
        fi
        if [[ "${KERNEL_ROOT}" != *"tp${ROLLOUT_TP}"* ]]; then
            echo "ROLLOUT_TP=${ROLLOUT_TP} but KERNEL_ROOT does not name that degree:" >&2
            echo "  ${KERNEL_ROOT}" >&2
            echo "GEMM shapes are sharded at sweep time. Point KERNEL_ROOT at kernel_tp${ROLLOUT_TP}." >&2
            exit 1
        fi
    fi
    COST_OVERRIDES=(
        "rollout_cost.kernel_table_root=${KERNEL_ROOT}"
        "rollout_cost.gpu=${GPU}"
        "rollout_cost.model_path=${MODEL_PATH}"
    )
else
    if [[ ! -f "${ROLLOUT_ARTIFACT}" ]]; then
        echo "Missing rollout cost artifact: ${ROLLOUT_ARTIFACT}" >&2
        echo "Fit one first, e.g.: bash examples/regress/regress_rollout_${ROLLOUT_MODEL}.sh" >&2
        exit 1
    fi
    COST_OVERRIDES=( "rollout_cost.artifact_path=${ROLLOUT_ARTIFACT}" )
fi
