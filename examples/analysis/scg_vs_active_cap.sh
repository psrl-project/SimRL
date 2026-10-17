#!/usr/bin/env bash
# Sweep the router's per-instance active cap and plot SCG(S) against it.
#
# Every run is `examples/simulate/psrl_aligned.sh` with one knob moved,
# `rollout.router.max_concurrent_seqs_per_instance` from 8 to 128. The cap is
# applied by the router, so a request held back by it dwells in the router queue
# rather than in an instance's waiting queue.
#
# The top curve, S = `job.max_staleness`, equals throughput because nothing is
# excluded there. A run that never reaches some S still has an exact point at it,
# equal to the curve below, since SCG saturates once nothing is filtered out.
#
# NOTE(claude): SCG is a rate, so runs are comparable only when they consumed the
# same work. The plotter refuses runs whose consumed trajectory counts differ.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Activate the env first, or a child python resolves scipy against the wrong numpy.
SIMRL_ENV="${SIMRL_ENV:-/apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh}"
if [[ -f "${SIMRL_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${SIMRL_ENV}"
fi

CAP_START="${CAP_START:-8}"
CAP_STEP="${CAP_STEP:-8}"
CAP_END="${CAP_END:-128}"
# The staleness policy under test, part of every output name since it decides admission.
POLICY="${POLICY:-verl}"
RUN_BASE="${RUN_BASE:-${ROOT_DIR}/outputs/simulate/scg_active_cap_${POLICY}}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}}"
OUT_PREFIX="${OUT_PREFIX:-scg_vs_active_cap_${POLICY}}"
export POLICY
# Skip a cap whose telemetry already exists. Set false to re-simulate every point.
REUSE="${REUSE:-true}"

# The run's own `job.max_staleness`, which no trajectory can exceed.
MAX_STALENESS="${MAX_STALENESS:-2}"

# Pin the backend rather than `auto`, so every cap runs on one implementation.
export SIMRL_BACKEND="${SIMRL_BACKEND:-native}"

CAPS=()
for ((cap = CAP_START; cap <= CAP_END; cap += CAP_STEP)); do
    CAPS+=("${cap}")
done

echo "Sweeping ${#CAPS[@]} active caps: ${CAPS[*]}"
echo "  Policy:           ${POLICY}"
echo "  Per-run outputs:  ${RUN_BASE}/cap<N>"
echo "  Figure and table: ${OUT_DIR}/${OUT_PREFIX}.{csv,png}"

for cap in "${CAPS[@]}"; do
    run_dir="${RUN_BASE}/cap${cap}"
    if [[ "${REUSE}" == "true" && -f "${run_dir}/logs/request.jsonl" ]]; then
        echo ""
        echo "================ cap=${cap} (reusing ${run_dir}) ================"
        continue
    fi
    echo ""
    echo "================ cap=${cap} ================"
    # A subshell per run, so one cap's exported settings cannot leak into the next.
    (
        export ACTIVE_CAP="${cap}"
        export OUTPUT_DIR="${run_dir}"
        export MAX_STALENESS
        bash "${ROOT_DIR}/examples/simulate/psrl_aligned.sh"
    )
done

echo ""
echo "================ SCG(S) vs active cap ================"
python "${SCRIPT_DIR}/plot_scg_vs_active_cap.py" \
    --run-base "${RUN_BASE}" \
    --caps "${CAPS[@]}" \
    --max-staleness "${MAX_STALENESS}" \
    --out-dir "${OUT_DIR}" \
    --out-prefix "${OUT_PREFIX}" \
    --policy "${POLICY}"
