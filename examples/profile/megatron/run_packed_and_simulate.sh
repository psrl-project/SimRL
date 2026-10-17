#!/usr/bin/env bash
# Run the mbs=2 packed candidates for the tp2_pp3 mesh, re-fit the cost model,
# and simulate the PSRL baseline.
#
# The existing 15 mbs=1 rows (in megatron_tp2_pp3) cannot distinguish the
# token-linear cost from the quadratic attention cost because every candidate has
# micro_batch_size=1, making sum_squared_tokens = sum_tokens^2 exactly (perfect
# collinearity). These 10 mbs=2 candidates break that collinearity so the fit can
# separate the two terms. The new rows write to the same output_dir and get merged
# into the existing JSONL on collect.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh

PROFILE_DIR="${ROOT_DIR}/outputs/profile/qwen3_8b/megatron_tp2_pp3"
CONFIG="examples/profile/megatron/qwen3_8b_tp2_pp3_flash_packed.yaml"
CANDIDATES="${PROFILE_DIR}/control/megatron_bridge/candidates_packed.json"
COST_OUT="${ROOT_DIR}/outputs/cost/qwen3_8b_tp2_pp3"

echo "=== PHASE 1: run packed mbs=2 candidates ==="
srl-profile megatron run \
    --config "${CONFIG}" \
    --candidates "${CANDIDATES}"

echo ""
echo "=== PHASE 2: collect (merges new rows with existing 15) ==="
srl-profile megatron collect \
    --config "${CONFIG}" \
    --candidates "${CANDIDATES}"

# Also collect the base and nmb64 sets in case any were missed
srl-profile megatron collect \
    --config examples/profile/megatron/qwen3_8b_tp2_pp3.yaml \
    --candidates "${PROFILE_DIR}/control/megatron_bridge/candidates.json" 2>/dev/null || true
srl-profile megatron collect \
    --config examples/profile/megatron/qwen3_8b_tp2_pp3_nmb64.yaml \
    --candidates "${PROFILE_DIR}/control/megatron_bridge/candidates_nmb64.json" 2>/dev/null || true

echo ""
echo "=== PHASE 3: fit cost model ==="
mkdir -p "${COST_OUT}"
srl-regress training \
    --latency-model hydraulis \
    --memory-model hydraulis \
    --input "${PROFILE_DIR}/profiles/megatron_bridge.jsonl" \
    --output "${COST_OUT}/training_hydraulis.json"

echo ""
echo "=== PHASE 4: simulate PSRL baseline ==="
TRAINING_ARTIFACT="${COST_OUT}/training_hydraulis.json" \
    bash examples/simulate/psrl_aligned.sh

echo ""
echo "Done. Artifact: ${COST_OUT}/training_hydraulis.json"
