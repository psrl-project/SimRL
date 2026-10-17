#!/usr/bin/env bash
# Collect a joint (prompt, n responses) workload trace for Qwen3-8B on
# dapo-math-17k across every GPU listed in hosts.txt.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

HOSTS_FILE="${HOSTS_FILE:-${SCRIPT_DIR}/hosts.txt}"
# Collection generates from a real model on a real dataset, so both are required and
# have no sensible default.
DATASET="${DATASET:?set DATASET to a parquet dataset, e.g. /path/to/dapo-math-17k.parquet}"
MODEL="${MODEL:?set MODEL to a HuggingFace checkpoint, e.g. /path/to/Qwen3-8B}"
TOKENIZER="${TOKENIZER:-${MODEL}}"
# Sourced on every remote worker before launching, so it must be a path that exists on
# the workers rather than only on the host that starts the collection.
ENV_SCRIPT="${ENV_SCRIPT:?set ENV_SCRIPT to a shell script that activates the environment on each worker}"

OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/workload/qwen3_8b_dapo_tp1}"
SHARD_DIR="${SHARD_DIR:-${OUT_DIR}/shards}"
TRACE="${TRACE:-${OUT_DIR}/joint.jsonl}"

SAMPLE_SIZE="${SAMPLE_SIZE:-1024}"
GROUP_N="${GROUP_N:-8}"
MAX_TOKENS="${MAX_TOKENS:-10240}"
TP="${TP:-1}"
CONCURRENCY="${CONCURRENCY:-16}"
BASE_PORT="${BASE_PORT:-8700}"
SEED="${SEED:-42}"
# Serialize server boots per host: concurrent loads of a 5-shard checkpoint from
# the shared filesystem are slow enough to trip the readiness timeout.
STARTUP_STAGGER_S="${STARTUP_STAGGER_S:-25}"

mkdir -p "${SHARD_DIR}"

WORKER_ROWS="$(
    cd "${ROOT_DIR}" && python -m srl.cli.workload list-workers \
        --hosts-file "${HOSTS_FILE}" \
        --base-port "${BASE_PORT}"
)"
mapfile -t WORKERS <<<"${WORKER_ROWS}"
N_WORKERS="${#WORKERS[@]}"
echo "Collecting ${SAMPLE_SIZE} groups of n=${GROUP_N} across ${N_WORKERS} workers."

# Kill vLLM servers launched by this script on every host.  Scoped narrowly to
# the model path and port range so unrelated vLLM processes are not disturbed.
_cleanup() {
    echo "Cleaning up: terminating vLLM servers (model=${MODEL}, ports ${BASE_PORT}..$((BASE_PORT + 7)))..." >&2
    # Kill tracked background workers on this host.
    for pid in "${PIDS[@]+"${PIDS[@]}"}"; do
        kill "${pid}" 2>/dev/null || true
    done
    # Kill vLLM servers on every host, scoped to this model and port range.
    local -a _HOSTS=()
    local -A _SEEN_HOSTS=()
    for _entry in "${WORKERS[@]+"${WORKERS[@]}"}"; do
        read -r _index _host _gpu _port <<<"${_entry}"
        if [[ -z "${_SEEN_HOSTS[${_host}]+x}" ]]; then
            _HOSTS+=("${_host}")
            _SEEN_HOSTS["${_host}"]=1
        fi
    done
    LOCAL_IPS_CLEANUP="$(hostname -I)"
    for _host in "${_HOSTS[@]+"${_HOSTS[@]}"}"; do
        # Kill one pattern per port this script owns, so a vLLM server serving the
        # same model on any other port (the user may run unrelated work on these
        # hosts) is left alone.  `_VLLMServer._command` always emits
        # `vllm serve <model> --port <N>`, so the port is reliably in argv.
        _kill_cmd=""
        for _port in $(seq "${BASE_PORT}" $((BASE_PORT + 7))); do
            _kill_cmd+="pkill -f 'vllm serve ${MODEL} --port ${_port}' || true; "
        done
        if [[ " ${LOCAL_IPS_CLEANUP} " == *" ${_host} "* ]]; then
            echo "  [local ] killing vLLM servers on ${_host}..." >&2
            eval "${_kill_cmd}"
        else
            echo "  [remote] killing vLLM servers on ${_host}..." >&2
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${_host}" "${_kill_cmd}" || true
        fi
    done
}

PIDS=()
# Track worker metadata alongside PIDs for failure reporting.
PID_INDICES=()
PID_HOSTS=()
PID_GPUS=()
PID_LOGS=()

trap '_cleanup; exit 130' INT TERM

# DRY_RUN: print each worker command and exit without launching anything.
if [[ "${DRY_RUN:-}" == "1" ]]; then
    LOCAL_IPS="$(hostname -I)"
    for entry in "${WORKERS[@]}"; do
        read -r INDEX HOST GPU PORT <<<"${entry}"
        SHARD="$(printf '%s/worker-%02d.jsonl' "${SHARD_DIR}" "${INDEX}")"
        CMD="source ${ENV_SCRIPT} >/dev/null 2>&1; cd ${ROOT_DIR} && python -m srl.cli.workload collect-groups"
        CMD+=" --dataset ${DATASET} --tokenizer ${TOKENIZER} --model ${MODEL}"
        CMD+=" --output ${SHARD} --worker-index ${INDEX} --n-workers ${N_WORKERS}"
        CMD+=" --sample-size ${SAMPLE_SIZE} --n ${GROUP_N} --max-tokens ${MAX_TOKENS}"
        CMD+=" --tp ${TP} --concurrency ${CONCURRENCY} --port ${PORT} --gpu ${GPU} --seed ${SEED}"
        CMD+=" --serve-arg gpu_memory_utilization=0.85"
        CMD+=" --serve-arg max_model_len=$((MAX_TOKENS + 2048))"
        if [[ " ${LOCAL_IPS} " == *" ${HOST} "* ]]; then
            echo "[local ] worker-${INDEX} host=${HOST} gpu=${GPU} port=${PORT}: ${CMD}"
        else
            echo "[remote] worker-${INDEX} host=${HOST} gpu=${GPU} port=${PORT}: ssh -o BatchMode=yes -o StrictHostKeyChecking=no \"${HOST}\" \"${CMD}\""
        fi
    done
    exit 0
fi

# Check whether an existing shard is complete by counting the actual groups in
# the body against n_groups_requested in the header.  Trusting only the header
# count would miss a shard whose write was interrupted after the header was
# already committed.
_shard_is_complete() {
    local shard_path="$1"
    (cd "${ROOT_DIR}" && python -m srl.cli.workload check-shard --path "${shard_path}")
}

LOCAL_IPS="$(hostname -I)"
for entry in "${WORKERS[@]}"; do
    read -r INDEX HOST GPU PORT <<<"${entry}"
    SHARD="$(printf '%s/worker-%02d.jsonl' "${SHARD_DIR}" "${INDEX}")"
    if [[ -f "${SHARD}" ]]; then
        if _shard_is_complete "${SHARD}"; then
            echo "worker-${INDEX} on ${HOST} gpu ${GPU}: shard complete, skipping."
            continue
        else
            echo "worker-${INDEX} on ${HOST} gpu ${GPU}: shard exists but is SHORT, re-collecting."
        fi
    fi
    CMD="source ${ENV_SCRIPT} >/dev/null 2>&1; cd ${ROOT_DIR} && python -m srl.cli.workload collect-groups"
    CMD+=" --dataset ${DATASET} --tokenizer ${TOKENIZER} --model ${MODEL}"
    CMD+=" --output ${SHARD} --worker-index ${INDEX} --n-workers ${N_WORKERS}"
    CMD+=" --sample-size ${SAMPLE_SIZE} --n ${GROUP_N} --max-tokens ${MAX_TOKENS}"
    CMD+=" --tp ${TP} --concurrency ${CONCURRENCY} --port ${PORT} --gpu ${GPU} --seed ${SEED}"
    CMD+=" --serve-arg gpu_memory_utilization=0.85"
    CMD+=" --serve-arg max_model_len=$((MAX_TOKENS + 2048))"

    LOG="${SHARD_DIR}/worker-$(printf '%02d' "${INDEX}").log"
    if [[ " ${LOCAL_IPS} " == *" ${HOST} "* ]]; then
        bash -lc "${CMD}" >"${LOG}" 2>&1 &
    else
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${HOST}" "${CMD}" >"${LOG}" 2>&1 &
    fi
    PIDS+=("$!")
    PID_INDICES+=("${INDEX}")
    PID_HOSTS+=("${HOST}")
    PID_GPUS+=("${GPU}")
    PID_LOGS+=("${LOG}")
    echo "launched worker-${INDEX} on ${HOST} gpu ${GPU} (log ${LOG})"
    sleep "${STARTUP_STAGGER_S}"
done

FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[${i}]}"
    if ! wait "${pid}"; then
        FAILED=$((FAILED + 1))
        echo "worker-${PID_INDICES[${i}]} on ${PID_HOSTS[${i}]} gpu ${PID_GPUS[${i}]} FAILED" \
             "(log: ${PID_LOGS[${i}]})" >&2
    fi
done
if (( FAILED > 0 )); then
    echo "${FAILED} worker(s) failed. Inspect ${SHARD_DIR}/worker-*.log, then rerun" >&2
    echo "this script: completed shards are skipped and only failures retry." >&2
    exit 1
fi

cd "${ROOT_DIR}"
python -m srl.cli.workload merge-shards \
    --shard-dir "${SHARD_DIR}" \
    --n-workers "${N_WORKERS}" \
    --output "${TRACE}" \
    --max-output-tokens "${MAX_TOKENS}"
echo "Joint trace: ${TRACE}"

# Verify that the merged trace contains the requested number of groups.
python -m srl.cli.workload verify-collection \
    --trace "${TRACE}" \
    --expected-groups "${SAMPLE_SIZE}" \
    --shard-dir "${SHARD_DIR}" \
    --n-workers "${N_WORKERS}"
