#!/usr/bin/env bash
# Sweep one preset's staleness knob and draw its throughput composition.
#
# Usage:  bash sweep.sh <preset-dir>            e.g. bash sweep.sh psrl
#
# Reads the preset's grid.sh, runs every point in parallel against the regime in
# common.sh, refuses the comparison if the points did not finish the same
# training work, and writes one figure.
#
# ENV  RUN_BASE      where run directories go
#      MAX_PARALLEL  concurrent simulations
#      REUSE         true skips a point whose request stream already exists
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PRESET_DIR="${1:?Usage: sweep.sh <preset-dir>}"
GRID="${SCRIPT_DIR}/${PRESET_DIR%/}/grid.sh"
[[ -f "${GRID}" ]] || { echo "No grid at ${GRID}" >&2; exit 1; }

# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
SECONDARY_KNOB=""
SECONDARY_VALUES=()
PINS=()
# shellcheck source=/dev/null
source "${GRID}"

OUT_DIR="${SCRIPT_DIR}/${PRESET_DIR%/}/figures"
# Keyed on the directory, not $PRESET: a preset directory may sweep a
# secondary knob of a policy another directory already sweeps (e.g.
# `prime_rl_lag` sweeping `target_lag` against the same `policy=prime_rl`
# `prime_rl/` sweeps `max_off_policy_steps` against), and $PRESET has to stay
# the real Hydra policy name for `POLICY="${PRESET}"` below to resolve.
RUNS="${RUN_BASE}/${PRESET_DIR%/}"
mkdir -p "${OUT_DIR}" "${RUNS}"

# NOTE(claude): A rebuild of the engine mid-sweep kills running simulations with
# a native signal that reads like an engine bug, and silently mixes binaries
# when it does not, so the binary's identity is part of the result.
SO="$(echo "${ROOT_DIR}"/srl/_engine_cpp*.so)"
SO_MD5_BEFORE="$(md5sum "${SO}" | cut -d' ' -f1)"

# One panel per secondary value, or a single unnamed panel when there is none.
if ((${#SECONDARY_VALUES[@]} == 0)); then
    PANEL_KEYS=("")
else
    PANEL_KEYS=("${SECONDARY_VALUES[@]}")
fi

n_points=$(( ${#PRIMARY_VALUES[@]} * ${#PANEL_KEYS[@]} ))
echo "Sweeping ${PRESET_DIR%/} (policy=${PRESET}): ${PRIMARY_KNOB} = ${PRIMARY_VALUES[*]}"
if [[ -n "${SECONDARY_KNOB}" ]]; then
    echo "  panels: ${SECONDARY_KNOB} = ${SECONDARY_VALUES[*]}"
fi
((${#PINS[@]} == 0)) || echo "  pinned: ${PINS[*]}"
echo "  ${n_points} points, ${N_VERSIONS} versions, backend ${SIMRL_BACKEND}, up to ${MAX_PARALLEL} at once"

run_point () {  # $1=run_dir  $2=overrides
    local run_dir="$1" overrides="$2"
    mkdir -p "${run_dir}"
    (
        export OUTPUT_DIR="${run_dir}"
        export POLICY="${PRESET}"
        export EXTRA_OVERRIDES="${overrides} +log_exclude='instance,rollout_step'"
        bash "${ROOT_DIR}/examples/simulate/psrl_aligned.sh" > "${run_dir}/sweep.log" 2>&1
    )
    # Only request.jsonl is read. The rest is dead weight on a full quota.
    rm -rf "${run_dir}/logs/instance" "${run_dir}/logs/rollout_step.jsonl"
}

MANIFEST="${RUNS}/points.tsv"
: > "${MANIFEST}"
pids=()
labels=()

for panel in "${PANEL_KEYS[@]}"; do
    for value in "${PRIMARY_VALUES[@]}"; do
        if [[ -n "${panel}" ]]; then
            run_dir="${RUNS}/s${panel}_p${value}"
            overrides="${PRIMARY_KNOB}=${value} ${SECONDARY_KNOB}=${panel}"
            panel_name="${SECONDARY_KNOB##*.} = ${panel}"
        else
            run_dir="${RUNS}/p${value}"
            overrides="${PRIMARY_KNOB}=${value}"
            panel_name=""
        fi
        for pin in ${PINS[@]+"${PINS[@]}"}; do
            overrides="${overrides} ${pin}"
        done
        printf '%s\t%s\t%s\n' "${panel_name}" "${value}" "${run_dir}" >> "${MANIFEST}"

        if [[ "${REUSE}" == "true" && -f "${run_dir}/logs/request.jsonl" ]]; then
            echo "  [reuse] ${run_dir##*/}"
            continue
        fi
        while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do wait -n || true; done
        echo "  [run]   ${run_dir##*/}"
        run_point "${run_dir}" "${overrides}" &
        pids+=("$!")
        labels+=("${run_dir##*/}")
    done
done

failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[index]}"; then
        echo "  FAILED ${labels[index]}, see ${RUNS}/${labels[index]}/sweep.log" >&2
        failed=1
    fi
done

if [[ "$(md5sum "${SO}" | cut -d' ' -f1)" != "${SO_MD5_BEFORE}" ]]; then
    echo "  ${SO} changed during the sweep. Points may come from different builds." >&2
    failed=1
fi
(( failed == 0 )) || exit 1

echo ""
srl-staleness-plot "${MANIFEST}" \
    --primary-label "${PRIMARY_LABEL}" \
    --out "${OUT_DIR}/staleness_throughput.png"
