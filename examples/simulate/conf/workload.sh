#!/usr/bin/env bash
# conf/workload.sh: workload distribution overrides
# Sourced by simulate.sh. Sets DIST_OVERRIDES.
# Inputs: WORKLOAD, DIST_FILE, SCRIPT_DIR

DIST_OVERRIDES=()
if [[ "${WORKLOAD}" == "normal" ]]; then
    if [[ ! -f "${DIST_FILE}" ]]; then
        echo "Missing distribution config: ${DIST_FILE}" >&2
        exit 1
    fi
    while IFS= read -r line; do
        DIST_OVERRIDES+=("${line}")
    done < <(python "${SCRIPT_DIR}/emit_workload_overrides.py" "${DIST_FILE}")
    if [[ ${#DIST_OVERRIDES[@]} -eq 0 ]]; then
        echo "No workload overrides emitted from ${DIST_FILE}." >&2
        exit 1
    fi
fi
