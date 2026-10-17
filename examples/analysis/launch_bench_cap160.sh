#!/usr/bin/env bash
# Launch the cap=160 backend timing benchmark detached, on whichever node runs
# this, and return immediately. Progress goes to outputs/bench_backends/.
set -euo pipefail
cd /apdcephfs_zwfy/share_303760348/lhy/SimRL
mkdir -p outputs/bench_backends
# The KV probe cache defaults to $HOME, which is per node, so point at the shared copy.
export SIMRL_KV_CACHE="${PWD}/outputs/bench_backends/kv_capacity.json"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export CAP="${CAP:-160}" N_VERSIONS="${N_VERSIONS:-4}" REPEATS="${REPEATS:-3}" PARALLEL="${PARALLEL:-1}"
export OUT="${OUT:-/tmp/bench_backends_cap${CAP}}"
log="outputs/bench_backends/cap${CAP}_$(hostname -I | awk '{print $1}').log"
nohup bash examples/analysis/bench_backends.sh >"${log}" 2>&1 </dev/null &
echo "launched pid=$! log=${PWD}/${log}"
