#!/usr/bin/env bash
# Sweep the router's active cap under several staleness policies, on a
# tail-restored DAPO workload, and plot SCG(S) for each.
#
# The aligned DAPO trace is censored at its output cap, so it has no straggler
# for a staleness bound to act on, and too few prompt groups for policies to
# select differently. `extend_dapo_trace.py` restores the censored tail and tiles
# the prompt set.
#
# NOTE(claude): SCG is a rate, so runs are comparable only when they consumed the
# same work. The plotter refuses runs whose consumed trajectory counts differ.
#
# `log_exclude` suppresses the large per-step streams at the source, since a run
# must fit on disk while it executes.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

SIMRL_ENV="${SIMRL_ENV:-/apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh}"
if [[ -f "${SIMRL_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${SIMRL_ENV}"
fi

# Pin the backend rather than `auto`, whose silent fallback could mix engines
# across points of one sweep.
export SIMRL_BACKEND="${SIMRL_BACKEND:-cpp}"

# A KV probe needs a free GPU. A busy one fails as an opaque engine startup error.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

TRACE="${TRACE:-${ROOT_DIR}/outputs/workload/dapo_32k/joint.jsonl}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/policy_cap_sweep}"
RUN_BASE="${RUN_BASE:-${ROOT_DIR}/outputs/simulate/policy_cap_sweep}"
REUSE="${REUSE:-true}"

CAP_START="${CAP_START:-16}"
CAP_STEP="${CAP_STEP:-16}"
CAP_END="${CAP_END:-160}"
POLICIES="${POLICIES:-psrl verl}"
# The no-cap reference. 1024 is vLLM's `max_num_seqs` here, so the router cap never binds.
BASELINE_CAP="${BASELINE_CAP:-1024}"

# Transcribed from psrl_aligned.sh, except the settings the new trace changes.
export MAX_OUTPUT_LENGTH="${MAX_OUTPUT_LENGTH:-32768}"
export N_VERSIONS="${N_VERSIONS:-4}"
export MAX_STALENESS="${MAX_STALENESS:-2}"

if [[ ! -f "${TRACE}" ]]; then
    echo "Missing the tail-restored trace: ${TRACE}" >&2
    echo "Build it with:" >&2
    echo "  python examples/analysis/extend_dapo_trace.py \\" >&2
    echo "    --source outputs/workload/psrl_aligned/joint.jsonl \\" >&2
    echo "    --out ${TRACE} --cap 32768 --repeat 4 --tail-alpha 2.0" >&2
    exit 1
fi

CAPS=()
for ((cap = CAP_START; cap <= CAP_END; cap += CAP_STEP)); do
    CAPS+=("${cap}")
done

so="${ROOT_DIR}/srl/_engine_cpp.cpython-312-x86_64-linux-gnu.so"
so_md5_before="$(md5sum "${so}" | cut -d' ' -f1)"

echo "Sweeping ${#CAPS[@]} caps x $(echo "${POLICIES}" | wc -w) policies: ${CAPS[*]} (+ baseline ${BASELINE_CAP})"
echo "  Trace:    ${TRACE}"
echo "  Runs:     ${RUN_BASE}/<policy>/cap<N>"
echo "  Figures:  ${OUT_DIR}"

# Every point is independent and single-threaded, so run them all at once.
pids=()
for pol in ${POLICIES}; do
    for cap in "${CAPS[@]}" "${BASELINE_CAP}"; do
        run_dir="${RUN_BASE}/${pol}/cap${cap}"
        if [[ "${REUSE}" == "true" && -f "${run_dir}/logs/request.jsonl" ]]; then
            echo "  [reuse] ${pol} cap=${cap}"
            continue
        fi
        echo "  [run]   ${pol} cap=${cap}"
        # The per-run log is written beside the run directory, which must exist first.
        mkdir -p "${run_dir}"
        # A subshell per run, so one cap's exported settings cannot leak into the next.
        (
            export POLICY="${pol}"
            export ACTIVE_CAP="${cap}"
            export OUTPUT_DIR="${run_dir}"
            export EXTRA_OVERRIDES="workload.trace.path=${TRACE} \
rollout.max_concurrency=1024 \
rollout.router.max_concurrent_seqs_per_instance=${cap} \
rollout.router.max_waiting_after_preempt=1024 \
rollout.router.reject_preempted_on_waiting=true \
+log_exclude='instance,rollout_step'"
            bash "${ROOT_DIR}/examples/simulate/psrl_aligned.sh" > "${run_dir%/*}/cap${cap}.log" 2>&1
            # Backstop for any per-step stream that slips past `log_exclude`.
            rm -rf "${run_dir}/logs/instance" "${run_dir}/logs/rollout_step.jsonl"
        ) &
        pids+=("$!:${pol}/cap${cap}")
    done
done

failed=0
for entry in "${pids[@]}"; do
    if ! wait "${entry%%:*}"; then
        echo "  FAILED ${entry#*:}, see ${RUN_BASE}/${entry#*:}.log" >&2
        failed=1
    fi
done

# A concurrent rebuild of the engine `.so` can crash running sims or mix builds across points.
if [[ "${SIMRL_BACKEND}" == "cpp" && "$(md5sum "${so}" | cut -d' ' -f1)" != "${so_md5_before}" ]]; then
    echo "  ${so} changed during the sweep. Points may come from different builds." >&2
    failed=1
fi
(( failed == 0 )) || exit 1

echo ""
echo "================ figures ================"
python "${SCRIPT_DIR}/plot_policy_cap_sweep.py" \
    --run-base "${RUN_BASE}" \
    --policies ${POLICIES} \
    --caps "${CAPS[@]}" \
    --baseline-cap "${BASELINE_CAP}" \
    --max-staleness "${MAX_STALENESS}" \
    --out-dir "${OUT_DIR}"
