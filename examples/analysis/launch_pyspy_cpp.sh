#!/usr/bin/env bash
# Sampled profiles of the C++ backend at cap=160, on an existing snapshot.
#
# NOTE(claude): `py-spy record -- <cmd>` samples only <cmd> itself, which here is
# a bash wrapper. So the run starts normally and py-spy attaches to the
# simulator's own PID, returning when that process exits.
set -euo pipefail
SNAP="${SNAP:-/apdcephfs_zwfy/share_303760348/lhy/bench_snapshot_diag}"
cd "${SNAP}"
set +u
# shellcheck disable=SC1091
source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh >/dev/null 2>&1
set -u
RES="${SNAP}/outputs_diag"
mkdir -p "${RES}"
export SIMRL_KV_CACHE="${SNAP}/outputs/bench_backends/kv_capacity.json"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export MAX_OUTPUT_LENGTH=65536 N_VERSIONS=4 POLICY=psrl ACTIVE_CAP=160
TRACE="${SNAP}/outputs/workload/dapo_64k/joint.jsonl"
# `rollout_step` is kept so `event_granularity.py` can count single-step events.
OV="workload.trace.path=${TRACE} rollout.max_concurrency=1024 \
rollout.router.max_concurrent_seqs_per_instance=160 \
rollout.router.max_waiting_after_preempt=1024 \
rollout.router.reject_preempted_on_waiting=true \
+log_exclude='instance'"

profile_one() {
    local name="$1" undo="$2" out="/tmp/diag_cpp_$1" pid=""
    rm -rf "${out}"
    env SIMRL_BACKEND=cpp SIMRL_UNDOABLE_EVENTS="${undo}" \
        EXTRA_OVERRIDES="${OV}" OUTPUT_DIR="${out}" \
        bash examples/simulate/psrl_aligned.sh >"${RES}/stdout_cpp_${name}.log" 2>&1 &
    local job=$!
    # The simulator is the only Python process whose argv names this output dir.
    for _ in $(seq 1 600); do
        pid=$(pgrep -f "^python -m srl.cli.main.*output_dir=${out}( |$)" | head -1 || true)
        [[ -n "${pid}" ]] && break
        sleep 0.2
    done
    if [[ -z "${pid}" ]]; then
        echo "cpp_${name}: simulator process never appeared" >&2
        wait "${job}" || true
        return 1
    fi
    py-spy record --native --rate 97 --format raw --pid "${pid}" \
        -o "${RES}/pyspy_cpp_${name}.txt" >"${RES}/pyspy_cpp_${name}.log" 2>&1 || true
    wait "${job}" || true
    python examples/analysis/pyspy_breakdown.py "${RES}/pyspy_cpp_${name}.txt" \
        >"${RES}/breakdown_cpp_${name}.txt" 2>&1 || true
    python examples/analysis/event_granularity.py "${out}" \
        >"${RES}/granularity_cpp_${name}.txt" 2>&1 || true
    md5sum "${out}/logs/request.jsonl" | cut -c1-12 >"${RES}/md5_cpp_${name}.txt"
    rm -rf "${out}/plots"
    echo "cpp_${name} done $(date +%T) pid=${pid}"
}

(
    echo "=== py-spy $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) ==="
    profile_one exact "request_complete,preempt,iteration_done" &
    profile_one fast "request_complete" &
    wait
    echo "=== all done $(date +%T) ==="
) >"${RES}/pyspy.log" 2>&1 </dev/null &
echo "launched pid=$! log=${RES}/pyspy.log"
