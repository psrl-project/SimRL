#!/usr/bin/env bash
# Collect and regress vLLM timing profiles for Qwen3.5-35B-A3B at tp=1 and tp=2.
#
# PIPELINE
#   1. dryrun-profile vllm collect  (parses raw timing JSON → profiles/vllm.jsonl)
#   2. dryrun-regress rollout        (fits psrl coefficients → rollout_psrl.json)
#
# The collect step reads all raw JSON from raw/vllm_timing/**, converts each
# scheduler-iteration record to a ProfileRecord, and writes a single JSONL.
# The regress step fits one entry per (tp, pp, dp, cp) that appears in that JSONL,
# so running collect on both the tp=1 and tp=2 raw outputs before regressing gives
# a single artifact with both entries.
#
# USAGE
#   bash examples/profile/vllm/collect_regress_qwen35_35b_a3b.sh
#
# PREREQUISITES
#   Both profiles must have completed:
#     TP=2 bash examples/profile/vllm/profile_qwen35_35b_a3b.sh all
#     TP=1 bash examples/profile/vllm/profile_qwen35_35b_a3b.sh all
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/apdcephfs_zwfy10/share_303541817/lhy/models/Qwen3.5-35B-A3B}"
PROFILE_BASE="${PROFILE_BASE:-${ROOT_DIR}/outputs/profile/qwen35_35b_a3b}"
COST_DIR="${COST_DIR:-${ROOT_DIR}/outputs/cost/qwen35_35b_a3b}"

mkdir -p "${COST_DIR}"

cd "${ROOT_DIR}"

source "${ROOT_DIR}/env/dryrun-rl.sh" 2>/dev/null || true

# --------------------------------------------------------------------------
# Step 1: collect each TP profile into a profiles/vllm.jsonl
# --------------------------------------------------------------------------
for TP in 2 1; do
    PROF_DIR="${PROFILE_BASE}/vllm_tp${TP}"
    if [[ ! -d "${PROF_DIR}/raw" ]]; then
        echo "Skipping TP=${TP}: raw output missing at ${PROF_DIR}/raw"
        continue
    fi
    echo "=== Collecting TP=${TP} profile ==="
    MODEL_PATH="${MODEL_PATH}" dryrun-profile vllm collect \
        --config "${ROOT_DIR}/examples/profile/vllm/qwen35_35b_a3b_tp${TP}.yaml"
    echo "Collected: ${PROF_DIR}/profiles/vllm.jsonl"
done

# --------------------------------------------------------------------------
# Step 2: merge collected JSONL files and fit a single psrl artifact
# --------------------------------------------------------------------------
# dryrun-regress accepts a single --input path; combine both TP profiles into
# one file so the fit sees both parallelism degrees and produces entries for both.
MERGED="${COST_DIR}/vllm_merged.jsonl"
> "${MERGED}"
for TP in 1 2; do
    JSONL="${PROFILE_BASE}/vllm_tp${TP}/profiles/vllm.jsonl"
    if [[ -f "${JSONL}" ]]; then
        cat "${JSONL}" >> "${MERGED}"
        echo "Merged TP=${TP}: $(wc -l < "${JSONL}") records"
    fi
done

echo ""
echo "=== Fitting psrl model from ${MERGED} ==="
dryrun-regress rollout \
    --model psrl \
    --input "${MERGED}" \
    --output "${COST_DIR}/rollout_psrl.json"

echo ""
echo "Artifact written: ${COST_DIR}/rollout_psrl.json"
echo "Entries available:"
python -c "
import json
from pathlib import Path
art = json.loads(Path('${COST_DIR}/rollout_psrl.json').read_text())
for e in art.get('entries', []):
    p = e.get('parallelism', {})
    print(f\"  tp={p.get('tp',1)} pp={p.get('pp',1)} dp={p.get('dp',1)}\")
"
