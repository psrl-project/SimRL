#!/usr/bin/env bash
# vLLM rollout profile for Qwen3.5-35B-A3B at tp=1, tp=2, and tp=4.
#
# WHY THIS MODEL
#   Qwen3-8B at TP concentration showed a null result (TP2 vs TP1 at fixed GPU
#   budget) because on H20 its attention is already bandwidth-bound and TP sharding
#   buys little per token. This 35B MoE has three properties that change that:
#     * Only 3B parameters active per token (256 experts, 8 active): small active
#       FLOP per token relative to the total weight footprint that TP must shard.
#     * Only 10 of 40 layers are full-attention (every 4th). The other 30 are
#       linear/SSM, so the context-growth term is carried by a quarter of the stack.
#     * num_kv_heads=2 with head_dim=256: KV cache is ~1/8 of Qwen3-8B's per token,
#       so higher in-flight counts are feasible before admission throttles.
#   These three together mean the decode throughput vs latency tradeoff under TP
#   sharding is genuinely different from the dense 8B case.
#
# USAGE
#   TP=1 bash examples/profile/vllm/profile_qwen35_35b_a3b.sh [generate|run|collect|all]
#   TP=2 ...
#   TP=4 ...
#   Default TP is 2. Default ACTION is all (generate + run + collect).
#
#   MAX_PARALLEL overrides the per-config concurrency (useful for debugging).
#   PHASE=prefill or PHASE=decode to run one phase only.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/apdcephfs_zwfy10/share_303541817/lhy/models/Qwen3.5-35B-A3B}"
TP="${TP:-2}"
CONFIG="${ROOT_DIR}/examples/profile/vllm/qwen35_35b_a3b_tp${TP}.yaml"
PHASE="${PHASE:-all}"
ACTION="${1:-all}"
RUN_ARGS=()
if [[ -n "${MAX_PARALLEL:-}" ]]; then
    RUN_ARGS+=(--max-parallel "${MAX_PARALLEL}")
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "Config not found: ${CONFIG}" >&2
    echo "Valid TP values: 1, 2, 4" >&2
    exit 1
fi

cd "${ROOT_DIR}"

case "${ACTION}" in
    generate)
        MODEL_PATH="${MODEL_PATH}" srl-profile vllm generate \
            --config "${CONFIG}" --phase "${PHASE}"
        ;;
    run)
        MODEL_PATH="${MODEL_PATH}" srl-profile vllm run \
            --config "${CONFIG}" --phase "${PHASE}" \
            "${RUN_ARGS[@]}" "${@:2}"
        ;;
    collect)
        MODEL_PATH="${MODEL_PATH}" srl-profile vllm collect \
            --config "${CONFIG}" --phase "${PHASE}"
        ;;
    all)
        MODEL_PATH="${MODEL_PATH}" srl-profile vllm generate \
            --config "${CONFIG}" --phase "${PHASE}"
        MODEL_PATH="${MODEL_PATH}" srl-profile vllm run \
            --config "${CONFIG}" --phase "${PHASE}" \
            "${RUN_ARGS[@]}"
        MODEL_PATH="${MODEL_PATH}" srl-profile vllm collect \
            --config "${CONFIG}" --phase "${PHASE}"
        ;;
    *)
        echo "Usage: TP=<1|2|4> $0 [generate|run|collect|all]" >&2
        exit 2
        ;;
esac
