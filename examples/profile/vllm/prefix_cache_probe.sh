#!/usr/bin/env bash
# Measure vLLM's prefix-cache hit rate in four controlled scenarios, to check
# the simulated block pool against the engine it models.
#
# Expected values live in prefix_cache_probe.yaml and were registered before
# anything was measured. Nothing here is fitted: a disagreement means the
# pool's semantics are wrong and the logic must change, not that a coefficient
# needs tuning.
#
# Each scenario starts its own server, because gpu_memory_utilization differs
# between them and is fixed at startup. Servers run one at a time: parallel
# vLLM servers contaminate each other's measurements.
set -euo pipefail

source /apdcephfs_zwfy/share_303760348/lhy/env/simrl.sh
cd "$(git rev-parse --show-toplevel)"

MODEL="${MODEL:-/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B}"
PORT="${PORT:-8000}"
OUT="${OUT:-outputs/profile/prefix_cache_probe}"
mkdir -p "$OUT"

python -m srl.workload.prefix_probe_run \
  --config examples/profile/vllm/prefix_cache_probe.yaml \
  --model "$MODEL" \
  --port "$PORT" \
  --output "$OUT/measured.json"

echo
echo "Measured counters: $OUT/measured.json"
echo "The simulator encodes the same expectations without needing a GPU:"
echo "  pytest tests/test_kv_pool.py tests/test_kv_instance.py -q"
