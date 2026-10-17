#!/usr/bin/env bash
# Profile the TP2/PP3 mesh with flash attention, fit a cost model, and simulate
# the PSRL baseline.
#
# The old megatron_tp2_pp3 tree was profiled 2026-09-28 before the
# attention_backend fix in srl/profiler/adapters/megatron/launch.py. Every
# candidate there used cuDNN SDPA (the recipe default of None on H20), while
# verl's Megatron engine explicitly passes AttnBackend.flash. That single
# difference was the dominant source of the 1.84x gap between profiled step
# times and live PSRL step times. This script profiles into megatron_tp2_pp3_flash
# so the old data is preserved and the new fit is clean.
#
# Configs profiled:
#   qwen3_8b_tp2_pp3_flash.yaml       mbs=1, nmb={1,4}, seq={1k..12k}   10 candidates
#   qwen3_8b_tp2_pp3_flash_nmb64.yaml mbs=1, nmb={64},  seq={1k..12k}    5 candidates
#   qwen3_8b_tp2_pp3_flash_packed.yaml mbs=2, nmb={1,4}, seq={2k..10k}  10 candidates
#
# All three write to the same output_dir (megatron_tp2_pp3_flash) so collect
# merges them into one JSONL, giving the fit {mbs=1/nmb=1,4,64} + {mbs=2/nmb=1,4}.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh

FLASH_OUT="${ROOT_DIR}/outputs/profile/qwen3_8b/megatron_tp2_pp3_flash"
COST_OUT="${ROOT_DIR}/outputs/cost/qwen3_8b_tp2_pp3_flash"
mkdir -p "${FLASH_OUT}/control/megatron_bridge" "${COST_OUT}"

CONFIGS=(
    "examples/profile/megatron/qwen3_8b_tp2_pp3_flash.yaml"
    "examples/profile/megatron/qwen3_8b_tp2_pp3_flash_nmb64.yaml"
    "examples/profile/megatron/qwen3_8b_tp2_pp3_flash_packed.yaml"
)
CANDIDATE_FILES=(
    "${FLASH_OUT}/control/megatron_bridge/candidates_base.json"
    "${FLASH_OUT}/control/megatron_bridge/candidates_nmb64.json"
    "${FLASH_OUT}/control/megatron_bridge/candidates_packed.json"
)

echo "=== PHASE 1: generate candidate lists ==="
for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    cands="${CANDIDATE_FILES[$i]}"
    echo "Generating: ${cfg}"
    srl-profile megatron generate \
        --config "${cfg}" \
        --candidates "${cands}"
done

echo ""
echo "=== PHASE 2: run candidates ==="
for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    cands="${CANDIDATE_FILES[$i]}"
    n=$(python3 -c "import json; d=json.load(open('${cands}')); print(len(d['candidates']))" 2>/dev/null || echo "?")
    echo ""
    echo "--- Running ${cfg} (${n} candidates) ---"
    # Omitting --index runs all candidates
    srl-profile megatron run \
        --config "${cfg}" \
        --candidates "${cands}"
done

echo ""
echo "=== PHASE 3: collect into JSONL ==="
for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    cands="${CANDIDATE_FILES[$i]}"
    echo "Collecting: ${cfg}"
    srl-profile megatron collect \
        --config "${cfg}" \
        --candidates "${cands}"
done

echo ""
echo "=== PHASE 4: fit cost model ==="
srl-regress training \
    --latency-model hydraulis \
    --memory-model hydraulis \
    --input "${FLASH_OUT}/profiles/megatron_bridge.jsonl" \
    --output "${COST_OUT}/training_hydraulis.json"

echo ""
echo "=== PHASE 5: simulate PSRL baseline ==="
TRAINING_ARTIFACT="${COST_OUT}/training_hydraulis.json" \
    bash examples/simulate/psrl_aligned.sh

echo ""
echo "Done. Cost artifact: ${COST_OUT}/training_hydraulis.json"
echo "Simulation output:   ${ROOT_DIR}/outputs/simulate/psrl_aligned/"
