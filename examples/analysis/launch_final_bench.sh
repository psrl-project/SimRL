#!/usr/bin/env bash
# Final cap=160 backend timing, from a frozen code snapshot, detached.
#
# Run it from the snapshot copy, never the live repo, so concurrent edits cannot
# change the code mid-sweep. Phase A keeps every telemetry stream and phase B
# suppresses the per-step streams.
set -euo pipefail
SNAP="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${SNAP}"
set +u
# shellcheck disable=SC1091
source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh >/dev/null 2>&1
set -u
RES="${SNAP}/outputs_bench"
mkdir -p "${RES}"
export SIMRL_KV_CACHE="${SNAP}/outputs/bench_backends/kv_capacity.json"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export CAP=160 N_VERSIONS=4 REPEATS=3 PARALLEL=1 MAX_OUTPUT_LENGTH=65536
export TRACE="${SNAP}/outputs/workload/dapo_64k/joint.jsonl"
(
    echo "=== phase A: all telemetry on $(date +%T) host=$(hostname -I | awk '{print $1}') ==="
    LOG_EXCLUDE="" OUT=/tmp/bench_final_teleon bash examples/analysis/bench_backends.sh
    echo "=== phase B: log_exclude=instance,rollout_step $(date +%T) ==="
    LOG_EXCLUDE="instance,rollout_step" OUT=/tmp/bench_final_teleoff \
        bash examples/analysis/bench_backends.sh
    echo "=== all done $(date +%T) ==="
) >"${RES}/final.log" 2>&1 </dev/null &
echo "launched pid=$! log=${RES}/final.log"
