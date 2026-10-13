#!/usr/bin/env bash
# Diagnose where the C++ backend spends its time at cap=160, detached.
#
# Freezes a code snapshot first so concurrent edits cannot change what is
# measured, then times all four arms and profiles the C++ arms with py-spy.
# Results land in <snapshot>/outputs_diag/ on the shared filesystem.
set -euo pipefail
SRC=/apdcephfs_zwfy/share_303760348/lhy/DryRun-RL
SNAP=/apdcephfs_zwfy/share_303760348/lhy/bench_snapshot_diag
rm -rf "${SNAP}"
mkdir -p "${SNAP}"
rsync -a --exclude '__pycache__' --exclude 'build/' --exclude '*.pyc' \
    "${SRC}/dryrun" "${SRC}/examples" "${SRC}/tests" "${SNAP}/"
ln -s "${SRC}/outputs" "${SNAP}/outputs"
ln -s "${SRC}/third_party" "${SNAP}/third_party"
(cd "${SRC}" && git rev-parse HEAD >"${SNAP}/GIT_HEAD" && git diff >"${SNAP}/WORKTREE.diff")

cd "${SNAP}"
set +u
# shellcheck disable=SC1091
source /apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh >/dev/null 2>&1
set -u
RES="${SNAP}/outputs_diag"
mkdir -p "${RES}"
export DRYRUN_KV_CACHE="${SNAP}/outputs/bench_backends/kv_capacity.json"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export MAX_OUTPUT_LENGTH=65536 N_VERSIONS=4 POLICY=psrl ACTIVE_CAP=160
TRACE="${SNAP}/outputs/workload/dapo_64k/joint.jsonl"
OV="workload.trace.path=${TRACE} rollout.max_concurrency=1024 \
rollout.router.max_concurrent_seqs_per_instance=160 \
rollout.router.max_waiting_after_preempt=1024 \
rollout.router.reject_preempted_on_waiting=true \
+log_exclude='instance'"

(
    echo "host=$(hostname -I | awk '{print $1}') load=$(cut -d' ' -f1-3 /proc/loadavg) py-spy=$(command -v py-spy || echo MISSING)"

    # Phase 1: timing with the per-step streams suppressed.
    echo "=== phase 1: timing $(date +%T) ==="
    CAP=160 N_VERSIONS=4 REPEATS=3 PARALLEL=1 TRACE="${TRACE}" \
        LOG_EXCLUDE="instance,rollout_step" OUT=/tmp/diag_timing \
        bash examples/analysis/bench_backends.sh >"${RES}/timing.log" 2>&1
    echo "=== phase 1 done $(date +%T) ==="

    # Phase 2: sampled profiles, after the timing so they never share the node.
    echo "=== phase 2: py-spy $(date +%T) ==="
    SNAP="${SNAP}" bash examples/analysis/launch_pyspy_cpp.sh
    echo "=== phase 2 launched, progress in ${RES}/pyspy.log ==="
) >"${RES}/diag.log" 2>&1 </dev/null &
echo "launched pid=$! log=${RES}/diag.log"
