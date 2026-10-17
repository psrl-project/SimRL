#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
INPUT="${INPUT:-${ROOT_DIR}/outputs/profile/qwen3_8b/megatron_bridge/profiles/megatron_bridge.jsonl}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/outputs/cost/qwen3_8b/training_hydraulis.json}"

cd "${ROOT_DIR}"
srl-regress training \
    --latency-model hydraulis \
    --memory-model hydraulis \
    --input "${INPUT}" \
    --output "${OUTPUT}"
