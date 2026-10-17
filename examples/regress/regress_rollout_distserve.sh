#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
INPUT="${INPUT:-${ROOT_DIR}/outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/outputs/cost/qwen3_8b/rollout_distserve.json}"

cd "${ROOT_DIR}"
srl-regress rollout \
    --model distserve \
    --input "${INPUT}" \
    --output "${OUTPUT}" \
    --block-size 16
