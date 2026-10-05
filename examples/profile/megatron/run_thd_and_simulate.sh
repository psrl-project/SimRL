#!/usr/bin/env bash
# Profile the PSRL mesh with THD packing, fit a cost model, and simulate the baseline.
#
# This is the pipeline that closes the gap. Earlier BSHD-based fits overpredicted the
# measured PSRL training steps by 57-60% because BSHD charges attention at the padded
# sequence length while verl, running `use_remove_padding=True`, charges it per packed
# sequence. The quadratic term in the latency model is fitted against
# `sum_squared_tokens`, and those are different quantities under the two layouts:
#
#   BSHD:  sum_squared_tokens == sum_tokens^2        (always, for mbs=1)
#   THD:   sum_squared_tokens == sum(s_i^2)         (what flash-attn varlen costs)
#
# See `examples/profile/megatron/qwen3_8b_thd.yaml` for why raising micro_batch_size
# under BSHD does not fix this.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

source /apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh

CONFIG="examples/profile/megatron/qwen3_8b_thd.yaml"
PROFILE_DIR="${ROOT_DIR}/outputs/profile/qwen3_8b/megatron_tp2_pp3_thd"
CANDIDATES="${PROFILE_DIR}/control/megatron_bridge/candidates.json"
COST_OUT="${ROOT_DIR}/outputs/cost/qwen3_8b_tp2_pp3_thd"

mkdir -p "${PROFILE_DIR}/control/megatron_bridge" "${COST_OUT}"

echo "=== PHASE 1: generate candidates ==="
dryrun-profile megatron generate --config "${CONFIG}" --candidates "${CANDIDATES}"

echo ""
echo "=== PHASE 2: run every candidate ==="
# Omitting --index runs them all. `allow_incomplete: true` in the config means an
# OOM at the largest packs is recorded rather than aborting the sweep.
dryrun-profile megatron run --config "${CONFIG}" --candidates "${CANDIDATES}"

echo ""
echo "=== PHASE 3: collect ==="
dryrun-profile megatron collect --config "${CONFIG}" --candidates "${CANDIDATES}"

echo ""
echo "=== PHASE 4: fit the cost model ==="
dryrun-regress training \
    --latency-model hydraulis \
    --memory-model hydraulis \
    --input "${PROFILE_DIR}/profiles/megatron_bridge.jsonl" \
    --output "${COST_OUT}/training_hydraulis.json"

echo ""
echo "=== PHASE 5: simulate the PSRL baseline ==="
TRAINING_ARTIFACT="${COST_OUT}/training_hydraulis.json" \
    OUTPUT_DIR="${ROOT_DIR}/outputs/simulate/psrl_thd" \
    bash examples/simulate/psrl_aligned.sh

echo ""
echo "Artifact:   ${COST_OUT}/training_hydraulis.json"
echo "Simulation: ${ROOT_DIR}/outputs/simulate/psrl_thd/"
