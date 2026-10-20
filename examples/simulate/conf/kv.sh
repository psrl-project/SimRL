#!/usr/bin/env bash
# conf/kv.sh: KV cache capacity overrides
# Sourced by simulate.sh. Sets KV_OVERRIDES.
# Inputs: KV_MODEL_PATH, KV_PROBE, KV_GPU_MEMORY_UTILIZATION, KV_GPU_MEMORY_MIB,
#         ROLLOUT_TP, GPU, KV_NON_KV_OVERHEAD_BYTES

KV_OVERRIDES=()
if [[ -n "${KV_MODEL_PATH:-}" ]]; then
    KV_OVERRIDES=(
        "rollout.kv_cache_sizing.model_path=${KV_MODEL_PATH}"
        "rollout.kv_cache_sizing.probe=${KV_PROBE:-true}"
        "rollout.kv_cache_sizing.gpu_memory_utilization=${KV_GPU_MEMORY_UTILIZATION:-0.9}"
        "rollout.kv_cache_sizing.gpu_memory_mib=${KV_GPU_MEMORY_MIB:-97871}"
        "rollout.kv_cache_sizing.tp_size=${ROLLOUT_TP}"
        "rollout.kv_cache_sizing.gpu_name=${GPU}"
        "rollout.kv_cache_sizing.non_kv_overhead_bytes=${KV_NON_KV_OVERHEAD_BYTES:-0}"
    )
fi
