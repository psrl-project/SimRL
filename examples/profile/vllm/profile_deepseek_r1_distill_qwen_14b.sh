#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
TP="${TP:-1}"
CONFIG="${CONFIG:-${ROOT_DIR}/examples/profile/vllm/deepseek_r1_distill_qwen_14b_tp${TP}.yaml}"
PHASE="${PHASE:-all}"
ACTION="${1:-all}"
RUN_ARGS=()
if [[ -n "${MAX_PARALLEL:-}" ]]; then
    RUN_ARGS+=(--max-parallel "${MAX_PARALLEL}")
fi

cd "${ROOT_DIR}"

case "${ACTION}" in
    generate)
        srl-profile vllm generate --config "${CONFIG}" --phase "${PHASE}"
        ;;
    run)
        srl-profile vllm run \
            --config "${CONFIG}" \
            --phase "${PHASE}" \
            "${RUN_ARGS[@]}" \
            "${@:2}"
        ;;
    collect)
        srl-profile vllm collect --config "${CONFIG}" --phase "${PHASE}"
        ;;
    all)
        srl-profile vllm generate --config "${CONFIG}" --phase "${PHASE}"
        srl-profile vllm run \
            --config "${CONFIG}" \
            --phase "${PHASE}" \
            "${RUN_ARGS[@]}"
        srl-profile vllm collect --config "${CONFIG}" --phase "${PHASE}"
        ;;
    *)
        echo "Usage: $0 [generate|run|collect|all]" >&2
        echo "  TP=1|2  selects examples/profile/vllm/deepseek_r1_distill_qwen_14b_tp\${TP}.yaml" >&2
        exit 2
        ;;
esac
