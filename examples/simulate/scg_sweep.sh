#!/usr/bin/env bash
# Measure whether heterogeneous TP cohorts raise the SCG(S) curve, and by how much.
#
# Three fleet configurations over an identical workload, training cost, policy, seed and
# version count. Only the rollout fleet differs:
#
#   tp1_x4   4 x TP=1, static            baseline: maximum replicas
#   tp2_x2   2 x TP=2, static            baseline: maximum concentration
#   search   re-solved at every update   treatment (contribution C4)
#
# All three run with heterogeneous.enabled=true and cohort_affinity routing, so the two
# baselines traverse the SAME fleet/rebuild/routing code as the treatment. A measured
# difference therefore cannot be an artefact of the new plumbing -- which is the whole
# reason the baselines are expressed as `strategy=fixed` rather than by turning the
# feature off.
#
# WHAT COUNTS AS A POSITIVE RESULT
#   `search` must beat BOTH baselines at the same S. Beating only tp1_x4 is consistent
#   with "TP=2 is simply the better static choice here", which is a configuration
#   finding, not evidence that *dynamic* heterogeneity pays. The summary below prints the
#   comparison against both and says so explicitly.
#
# WHY n_gpus=4
#   Two cohorts need at least 3 GPUs, and with tp_candidates={1,2} the smallest real
#   split is 1xTP=2 + 2xTP=1 = 4. The aligned 2-GPU PSRL config cannot express a split at
#   all.
#
# EXPECTED EFFECT SIZE
#   At a fixed 4-GPU budget the statically-correct TP choice is worth 1.08-1.19x on fleet
#   throughput (measured against the kernel tables; see the design doc's E4 table). The
#   dynamic gain is bounded above by that, so expect single-digit percent and keep the
#   comparison tight: identical trace and seed across the three runs, which this script
#   enforces by construction.
#
# PREREQUISITES
#   - The DAPO joint trace. Run examples/simulate/psrl_aligned.sh once to extract it.
#   - Per-TP kernel tables: kernel_tp1/kernel AND kernel_tp2/kernel. The composed model
#     reads GEMM and attention tables sharded per TP, and a tp=2 model pointed at tp=1
#     tables raises KernelCoverageError rather than mispricing the step.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Activate the project environment before anything imports numpy/scipy. Without this the
# child python resolves scipy from the base anaconda install while numpy comes from the
# env, and the mismatch surfaces as `module 'numpy' has no attribute 'long'` from deep
# inside scipy.sparse -- an import error that looks nothing like an environment problem.
DRYRUN_ENV="${DRYRUN_ENV:-/apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh}"
if [[ -f "${DRYRUN_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${DRYRUN_ENV}"
fi

MODEL_PATH="${MODEL_PATH:-/apdcephfs_zwfy/share_303760348/lhy/models/Qwen3-8B}"
TRACE="${TRACE:-${ROOT_DIR}/outputs/workload/psrl_aligned/joint.jsonl}"
KERNEL_BASE="${KERNEL_BASE:-${ROOT_DIR}/outputs/profile/qwen3_8b}"
TRAINING_ARTIFACT="${TRAINING_ARTIFACT:-${ROOT_DIR}/outputs/cost/qwen3_8b_tp2_pp3_thd/training_hydraulis.json}"
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT_DIR}/outputs/simulate/scg_sweep}"

N_GPUS="${N_GPUS:-4}"
N_VERSIONS="${N_VERSIONS:-16}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GROUP_SIZE="${GROUP_SIZE:-8}"
MAX_STALENESS="${MAX_STALENESS:-1}"

TP1_ROOT="${KERNEL_BASE}/kernel_tp1/kernel"
TP2_ROOT="${KERNEL_BASE}/kernel_tp2/kernel"

for path in "${TRACE}" "${TRAINING_ARTIFACT}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing required input: ${path}" >&2
        echo "Run examples/simulate/psrl_aligned.sh first to produce the trace and artifact." >&2
        exit 1
    fi
done
for root in "${TP1_ROOT}" "${TP2_ROOT}"; do
    if [[ ! -d "${root}" ]]; then
        echo "Missing kernel tables: ${root}" >&2
        echo "The composed model needs GEMM and attention tables per TP degree." >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_BASE}"

# Shared across all three runs. Transcribed from psrl_aligned.sh so the workload and the
# training cost are the ones already validated against a real run.
export WORKLOAD=dapo_trace
export POLICY="${POLICY:-psrl}"
export ROLLOUT_MODEL=composed
export GPU="${GPU:-h20}"
export MODEL_PATH
export BATCH_SIZE GROUP_SIZE MAX_STALENESS N_VERSIONS
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_OUTPUT_LENGTH="${MAX_OUTPUT_LENGTH:-10240}"
export TOKEN_BUDGET="${TOKEN_BUDGET:-12800}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-1024}"
export TRAINING_BACKEND=hydraulis
export TRAINING_TP=2 TRAINING_PP=3 TRAINING_DP=1 TRAINING_CP=1
export MICRO_BATCH_SIZE=1
export MAX_TOKENS_PER_MICROBATCH="${MAX_TOKENS_PER_MICROBATCH:-12800}"
export ACTIVATION_RECOMPUTE=full
export TRAINING_ARTIFACT
export RECOMPUTE_MODEL=training_forward
export KV_MODEL_PATH="${MODEL_PATH}"
export KV_GPU_MEMORY_UTILIZATION=0.9
export KV_NON_KV_OVERHEAD_BYTES=2147483648
export EXTRAPOLATE="${EXTRAPOLATE:-[tokens,activation_bytes,pipeline_slots,sum_squared_tokens]}"

# `rollout_cost.kernel_table_root` still has to satisfy the composed model's own
# required-field check; the per-cohort roots below are what the fleet actually reads.
export KERNEL_ROOT="${TP1_ROOT}"

HETERO_COMMON=(
    "heterogeneous.enabled=true"
    "heterogeneous.n_gpus=${N_GPUS}"
    "heterogeneous.tp_candidates=[1,2]"
    # `+` prefix: kernel_table_roots defaults to an empty dict, which OmegaConf treats as
    # a struct, so adding keys to it is an append rather than an override. Without the
    # prefix Hydra refuses with "Key '1' is not in struct".
    "+heterogeneous.kernel_table_roots={1:'${TP1_ROOT}',2:'${TP2_ROOT}'}"
    "rollout.router.strategy=cohort_affinity"
    "rollout.max_concurrency=${MAX_CONCURRENCY}"
    "workload.trace.path=${TRACE}"
)

run_config() {
    local label=$1 overrides=$2 instances=$3 tp=$4
    echo ""
    echo "================ ${label} ================"
    # `rollout_cost.*` describes the single homogeneous model Simulator builds before the
    # fleet replaces it, so ROLLOUT_TP and KERNEL_ROOT must agree with each other:
    # simulate.sh checks that a tp>1 root carries a collective table, and a tp=2 model
    # reading the tp=1 root would fail that check (correctly). The per-cohort roots in
    # HETERO_COMMON are what the fleet itself reads.
    export N_INSTANCES="${instances}"
    export ROLLOUT_TP="${tp}"
    if [[ "${tp}" -eq 2 ]]; then
        export KERNEL_ROOT="${TP2_ROOT}"
    else
        export KERNEL_ROOT="${TP1_ROOT}"
    fi
    export OUTPUT_DIR="${OUTPUT_BASE}/${label}"
    export EXTRA_OVERRIDES="${HETERO_COMMON[*]} ${overrides}"
    bash "${SCRIPT_DIR}/simulate.sh"
}

#            label      overrides                                                              inst  tp
run_config "tp1_x4" "heterogeneous.strategy=fixed heterogeneous.cohorts=[{tp:1,n_instances:${N_GPUS}}]" "${N_GPUS}" 1
run_config "tp2_x2" "heterogeneous.strategy=fixed heterogeneous.cohorts=[{tp:2,n_instances:$((N_GPUS / 2))}]" "$((N_GPUS / 2))" 2
# `search` may pick either degree, so its pre-fleet model is the tp=1 one; the solver
# builds a correctly-rooted model per cohort regardless.
run_config "search" "heterogeneous.strategy=search" "${N_GPUS}" 1

echo ""
echo "================ SCG(S) comparison ================"
# Computed from the request telemetry stream, which already carries `staleness_true` on
# every consume event (dryrun/telemetry/writer.py). SimResult is not persisted, so the
# JSONL is the only way to compare finished runs.
python - "${OUTPUT_BASE}" <<'PYPLOT'
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
LABELS = ("tp1_x4", "tp2_x2", "search")


def load(run_dir: Path) -> tuple[list[int], float]:
    """Return (staleness_true per consumed response, span seconds) for one run."""
    stream = run_dir / "logs" / "request.jsonl"
    if not stream.exists():
        raise FileNotFoundError(stream)
    staleness: list[int] = []
    span = 0.0
    for line in stream.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        span = max(span, float(row.get("t", 0.0)))
        if row.get("event") == "consume" and row.get("staleness_true") is not None:
            staleness.append(int(row["staleness_true"]))
    return staleness, span


runs = {}
for label in LABELS:
    run_dir = base / label
    if not run_dir.exists():
        print(f"  skipping {label}: not found")
        continue
    runs[label] = load(run_dir)

if not runs:
    sys.exit("No runs to compare.")

max_s = max((max(st) for st, _ in runs.values() if st), default=0)
s_values = list(range(max_s + 1))
curves = {
    label: {
        s: (sum(1 for v in st if v <= s) / span if span > 0 else 0.0) for s in s_values
    }
    for label, (st, span) in runs.items()
}

print()
print("  S   " + "".join(f"{label:>12}" for label in runs))
for s in s_values:
    print(f"  {s:<3}" + "".join(f"{curves[label][s]:>12.4f}" for label in runs))
print()
print("  units: trajectories/second")
for label, (st, span) in runs.items():
    print(f"  {label:>8}: {len(st):5d} consumed over {span:8.1f}s")

if len(curves) == 3:
    print()
    margins = []
    for s in s_values:
        best_baseline = max(curves["tp1_x4"][s], curves["tp2_x2"][s])
        if best_baseline > 0:
            margins.append((s, curves["search"][s] / best_baseline))
    beats_both = all(ratio >= 1.0 - 1e-9 for _, ratio in margins)
    print("  search vs the BETTER baseline at each S:")
    for s, ratio in margins:
        print(f"    S={s}: {ratio:6.3f}x")
    print()
    if beats_both and any(ratio > 1.0 + 1e-9 for _, ratio in margins):
        print("  POSITIVE: search >= both baselines everywhere, strictly better somewhere.")
    elif beats_both:
        print("  NEUTRAL: search ties the better baseline; dynamic split bought nothing.")
    else:
        print("  NEGATIVE for C4 at this fleet size: a static choice beats the solver.")
        print("  That is a publishable result -- it says the split is not the lever here.")
PYPLOT

echo ""
echo "Per-run SCG plots: ${OUTPUT_BASE}/<label>/plots/scg_curve.png"
