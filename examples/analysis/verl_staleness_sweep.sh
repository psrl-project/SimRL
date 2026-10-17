#!/usr/bin/env bash
# Sweep verl's own staleness knob, `policy.staleness_threshold`, with the router
# cap out of the way, and plot SCG(S) for S = 0..3 against it.
#
# WHAT THE KNOB IS
#   verl's threshold is a sample-overage ratio, not a version count: it admits
#   at most `int(B * (1 + threshold))` prompt groups between weight syncs
#   (`VerlPolicy._configure`). So it is fractional, and its effect on observed
#   staleness is indirect -- which is what this sweep measures.
#
# WHY job.max_staleness IS PINNED, NOT SWEPT
#   The simulator stages `(job.max_staleness + 1) * B` groups ahead of the
#   router as a dataset flow limit (`Simulator._stage`). If that window were
#   narrower than verl's admission window, it would bind first and the
#   threshold would be masked. Pinning it at the top of the sweep keeps it
#   non-binding at every point, and holding it constant also keeps the router's
#   candidate set the same width everywhere, since that width alone is known to
#   change which request gets placed.
#
# ⚠️ SCG IS A RATE
#   Only comparable across runs that consumed the same work. The plotter refuses
#   to draw runs that disagree on the consumed trajectory count.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

SIMRL_ENV="${SIMRL_ENV:-/apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh}"
if [[ -f "${SIMRL_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${SIMRL_ENV}"
fi

# Pinned rather than `auto` so every point comes from one engine; see
# policy_cap_sweep.sh for the parity check behind choosing C++.
export SIMRL_BACKEND="${SIMRL_BACKEND:-cpp}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

TRACE="${TRACE:-${ROOT_DIR}/outputs/workload/dapo_32k/joint.jsonl}"
N_VERSIONS="${N_VERSIONS:-4}"
RUN_BASE="${RUN_BASE:-${ROOT_DIR}/outputs/simulate/verl_staleness_sweep/v${N_VERSIONS}}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/verl_staleness_sweep/v${N_VERSIONS}}"
REUSE="${REUSE:-true}"

# Thresholds in tenths, so the grid is exact integers until it is printed.
THR_START_TENTHS="${THR_START_TENTHS:-2}"
THR_STEP_TENTHS="${THR_STEP_TENTHS:-2}"
THR_END_TENTHS="${THR_END_TENTHS:-30}"
# 1024 is vLLM's own max_num_seqs here, so the router never binds before it.
ACTIVE_CAP="${ACTIVE_CAP:-1024}"

export MAX_OUTPUT_LENGTH="${MAX_OUTPUT_LENGTH:-32768}"
export N_VERSIONS
# Must be >= the largest threshold, see the header.
export MAX_STALENESS="${MAX_STALENESS:-$(( (THR_END_TENTHS + 9) / 10 ))}"
export POLICY=verl

THRESHOLDS=()
for ((t = THR_START_TENTHS; t <= THR_END_TENTHS; t += THR_STEP_TENTHS)); do
    THRESHOLDS+=("$(printf '%d.%d' $((t / 10)) $((t % 10)))")
done

so="${ROOT_DIR}/srl/_engine_cpp.cpython-312-x86_64-linux-gnu.so"
so_md5_before="$(md5sum "${so}" | cut -d' ' -f1)"

echo "Sweeping ${#THRESHOLDS[@]} verl thresholds at N_VERSIONS=${N_VERSIONS}: ${THRESHOLDS[*]}"
echo "  job.max_staleness=${MAX_STALENESS}  active cap=${ACTIVE_CAP}  backend=${SIMRL_BACKEND}"
echo "  Runs:     ${RUN_BASE}/thr<X>"
echo "  Figures:  ${OUT_DIR}"

# Every point is independent and single-threaded, so run them all at once.
pids=()
for thr in "${THRESHOLDS[@]}"; do
    run_dir="${RUN_BASE}/thr${thr}"
    if [[ "${REUSE}" == "true" && -f "${run_dir}/logs/request.jsonl" ]]; then
        echo "  [reuse] thr=${thr}"
        continue
    fi
    echo "  [run]   thr=${thr}"
    mkdir -p "${run_dir}"
    (
        export ACTIVE_CAP
        export OUTPUT_DIR="${run_dir}"
        export EXTRA_OVERRIDES="workload.trace.path=${TRACE} \
policy.staleness_threshold=${thr} \
rollout.max_concurrency=1024 \
rollout.router.max_concurrent_seqs_per_instance=${ACTIVE_CAP} \
rollout.router.max_waiting_after_preempt=1024 \
rollout.router.reject_preempted_on_waiting=true \
+log_exclude='instance,rollout_step'"
        bash "${ROOT_DIR}/examples/simulate/psrl_aligned.sh" > "${RUN_BASE}/thr${thr}.log" 2>&1
        # Only `request.jsonl` is read; the rest is dead weight on a full quota.
        rm -rf "${run_dir}/logs/instance" "${run_dir}/logs/rollout_step.jsonl"
    ) &
    pids+=("$!:${thr}")
done

failed=0
for entry in "${pids[@]}"; do
    if ! wait "${entry%%:*}"; then
        echo "  FAILED thr=${entry#*:}, see ${RUN_BASE}/thr${entry#*:}.log" >&2
        failed=1
    fi
done

# Another session rebuilding the engine in place kills running sims with a
# native signal that reads like an engine bug, and silently mixes binaries
# across points when it does not.
if [[ "${SIMRL_BACKEND}" == "cpp" && "$(md5sum "${so}" | cut -d' ' -f1)" != "${so_md5_before}" ]]; then
    echo "  ${so} changed during the sweep; points may come from different builds." >&2
    failed=1
fi
(( failed == 0 )) || exit 1

echo ""
echo "================ figures ================"
python "${SCRIPT_DIR}/plot_verl_staleness_sweep.py" \
    --run-base "${RUN_BASE}" \
    --thresholds "${THRESHOLDS[@]}" \
    --s-values 0 1 2 3 \
    --title-suffix "verl, ${N_VERSIONS} versions, cap ${ACTIVE_CAP}" \
    --out-dir "${OUT_DIR}"
