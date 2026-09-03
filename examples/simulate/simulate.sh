#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIST_FILE="${DIST_FILE:-${SCRIPT_DIR}/distributions.yaml}"
WORKLOAD="${WORKLOAD:-normal}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

ROLLOUT_MODEL="${ROLLOUT_MODEL:-psrl}"
POLICY="${POLICY:-psrl}"
COST_DIR="${COST_DIR:-${ROOT_DIR}/outputs/cost/qwen3_8b}"
ROLLOUT_ARTIFACT="${ROLLOUT_ARTIFACT:-${COST_DIR}/rollout_${ROLLOUT_MODEL}.json}"
TRAINING_ARTIFACT="${TRAINING_ARTIFACT:-}"
TRAINING_LATENCY_S="${TRAINING_LATENCY_S:-50}"
# Which fitted training model the artifact holds. The two are not interchangeable:
# hydraulis carries a pipeline schedule and shards optimizer state only, while fsdp has
# no pipeline and shards parameters, gradients, and optimizer state alike.
TRAINING_BACKEND="${TRAINING_BACKEND:-hydraulis}"
# How real trajectory lengths become microbatches. FSDP only; profiling measures uniform
# shapes, so this is a simulate-time choice. pad_to_max charges for pad tokens,
# token_budget reproduces verl's dynamic batching, packed_no_pad charges real tokens only.
BATCHING_MODE="${BATCHING_MODE:-pad_to_max}"
MAX_TOKENS_PER_RANK="${MAX_TOKENS_PER_RANK:-}"
# An FSDP cell is a measurement of one mesh, so dp must equal shard*replicate or the
# model refuses the query. Left empty, it is read from the artifact.
TRAINING_DP="${TRAINING_DP:-}"
# Feature names permitted to extrapolate beyond the profiled range, as a bracket list.
# Empty string (the default) keeps strict behaviour: any out-of-domain query raises.
# Example: EXTRAPOLATE=[num_microbatches]
EXTRAPOLATE="${EXTRAPOLATE:-}"

N_VERSIONS="${N_VERSIONS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
N_INSTANCES="${N_INSTANCES:-4}"
MAX_STALENESS="${MAX_STALENESS:-2}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_OUTPUT_LENGTH="${MAX_OUTPUT_LENGTH:-16384}"
TOKEN_BUDGET="${TOKEN_BUDGET:-65536}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/simulate/qwen3_8b_fitted_cost}"

KERNEL_ROOT="${KERNEL_ROOT:-${ROOT_DIR}/outputs/profile/qwen3_8b/kernel_tp1/kernel}"
# Only the composed rollout model reads a checkpoint, so this stays unset by default
# and is required in that branch alone rather than for every run.
MODEL_PATH="${MODEL_PATH:-}"
GPU="${GPU:-h20}"
# Tensor parallelism of the inference engine being simulated. Only the composed model reads it;
# the fitted models were regressed at one degree and carry no notion of a second. A tp>1 step
# prices two all-reduces per layer, so KERNEL_ROOT must point at a tree swept at the same degree
# -- the collective table is sliced on world_size and cannot be interpolated across degrees.
ROLLOUT_TP="${ROLLOUT_TP:-1}"

if [[ "${ROLLOUT_MODEL}" == "composed" ]]; then
    # No fitted artifact exists for this model by design: it reads measured kernel tables.
    if [[ ! -d "${KERNEL_ROOT}" ]]; then
        echo "Missing kernel perf tables: ${KERNEL_ROOT}" >&2
        echo "Profile them first: bash examples/profile/kernel/profile_qwen3_8b.sh all" >&2
        exit 1
    fi
    if [[ -z "${MODEL_PATH}" ]]; then
        echo "ROLLOUT_MODEL=composed needs a checkpoint to read the architecture from." >&2
        echo "Set MODEL_PATH=/path/to/Qwen3-8B" >&2
        exit 1
    fi
    # A tp>1 run needs a collective table at exactly that world size. Checking here names the
    # missing sweep; letting it through produces a KernelCoverageError mid-simulation, after the
    # workload has already been built.
    if [[ "${ROLLOUT_TP}" -gt 1 ]]; then
        COLLECTIVE_DIR="${KERNEL_ROOT}/${GPU}/collective"
        if [[ ! -d "${COLLECTIVE_DIR}" ]]; then
            echo "ROLLOUT_TP=${ROLLOUT_TP} needs a collective table, but ${COLLECTIVE_DIR} does not exist." >&2
            echo "A tp>1 step prices two all-reduces per layer, so the table is not optional." >&2
            echo "Sweep it: TP=${ROLLOUT_TP} FAMILY=collective bash examples/profile/kernel/profile_qwen3_8b.sh all" >&2
            exit 1
        fi
        # KERNEL_ROOT is per-degree by convention (kernel_tp1, kernel_tp2, ...). Pointing a tp=2
        # run at the tp=1 tree is the easy mistake: the GEMM shapes there are unsharded, so every
        # projection would be priced at twice its real width and the error would look like a
        # model defect rather than a wiring one.
        if [[ "${KERNEL_ROOT}" != *"tp${ROLLOUT_TP}"* ]]; then
            echo "ROLLOUT_TP=${ROLLOUT_TP} but KERNEL_ROOT does not name that degree:" >&2
            echo "  ${KERNEL_ROOT}" >&2
            echo "GEMM shapes and attention head counts are sharded at sweep time, so a tree" >&2
            echo "swept at another degree prices the wrong shapes. Set KERNEL_ROOT to the" >&2
            echo "kernel_tp${ROLLOUT_TP} tree, or override this check if the path is named differently." >&2
            exit 1
        fi
    fi
    COST_OVERRIDES=(
        "rollout_cost.kernel_table_root=${KERNEL_ROOT}"
        "rollout_cost.gpu=${GPU}"
        "rollout_cost.model_path=${MODEL_PATH}"
    )
else
    if [[ ! -f "${ROLLOUT_ARTIFACT}" ]]; then
        echo "Missing rollout cost artifact: ${ROLLOUT_ARTIFACT}" >&2
        echo "Fit one first, for example: bash examples/regress/regress_rollout_${ROLLOUT_MODEL}.sh" >&2
        exit 1
    fi
    COST_OVERRIDES=( "rollout_cost.artifact_path=${ROLLOUT_ARTIFACT}" )
fi

DIST_OVERRIDES=()
if [[ "${WORKLOAD}" == "normal" ]]; then
    if [[ ! -f "${DIST_FILE}" ]]; then
        echo "Missing distribution config: ${DIST_FILE}" >&2
        exit 1
    fi
    while IFS= read -r line; do
        DIST_OVERRIDES+=("${line}")
    done < <(python "${SCRIPT_DIR}/emit_workload_overrides.py" "${DIST_FILE}")
    if [[ ${#DIST_OVERRIDES[@]} -eq 0 ]]; then
        echo "No workload overrides emitted from ${DIST_FILE}." >&2
        exit 1
    fi
fi

TRAINING_OVERRIDES=(
    training_cost=fixed
    training_cost.latency_s="${TRAINING_LATENCY_S}"
)
if [[ -n "${TRAINING_ARTIFACT}" ]]; then
    if [[ ! -f "${TRAINING_ARTIFACT}" ]]; then
        echo "Missing training cost artifact: ${TRAINING_ARTIFACT}" >&2
        exit 1
    fi
    # Check the requested topology exists before the simulator does. The model itself raises a
    # bare KeyError naming only the cell it wanted, which leaves you guessing what was fitted.
    if ! python - "${TRAINING_ARTIFACT}" "${TRAINING_BACKEND}" \
        "${TRAINING_TP:-1}" "${TRAINING_PP:-1}" "${SHARD_SIZE:-}" <<'PYCELL'
import json
import sys

path, backend, tp, pp, shard = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
entries = [e for e in json.load(open(path))["entries"] if e["selector"].get("component") == "latency"]
if backend == "fsdp":
    available = sorted({int(e["selector"]["shard_size"]) for e in entries if "shard_size" in e["selector"]})
    if shard and int(shard) not in available:
        print(f"SHARD_SIZE={shard} is not fitted. Available shard sizes: {available}.", file=sys.stderr)
        raise SystemExit(1)
else:
    available = sorted({(int(e["parallelism"]["tp"]), int(e["parallelism"]["pp"])) for e in entries})
    if (int(tp), int(pp)) not in available:
        pairs = ", ".join(f"tp={t} pp={p}" for t, p in available)
        print(
            f"TRAINING_TP={tp} TRAINING_PP={pp} is not a fitted cell. Available: {pairs}. "
            f"A cell is a measurement of one topology, so it cannot be interpolated.",
            file=sys.stderr,
        )
        raise SystemExit(1)
# The fitted latency domain is a hard gate at prediction time, so report it up front.
ranges = [e.get("feature_ranges", {}).get("tokens") for e in entries]
spans = [r for r in ranges if r]
if spans:
    lo, hi = min(r[0] for r in spans), max(r[1] for r in spans)
    print(f"fitted per-microbatch token domain: [{lo:.0f}, {hi:.0f}]", file=sys.stderr)
    # `tokens` is the LARGEST microbatch, not the mean, so one long sequence in a batch sets
    # it. Naming that here saves rediscovering it from a mid-run TrainingOutOfDomainError.
    print(
        f"  a step exceeding {hi:.0f} tokens in its longest microbatch raises unless "
        f"EXTRAPOLATE=[tokens] is set; the fit is linear in tokens while attention is "
        f"quadratic, so a stretched tail is understated",
        file=sys.stderr,
    )
PYCELL
    then
        exit 1
    fi
    if [[ "${TRAINING_BACKEND}" == "fsdp" ]]; then
        # Read the mesh the artifact was fitted on. Guessing dp here would either raise
        # TrainingOutOfDomainError or, worse, be silently answered for the wrong topology.
        if [[ -z "${TRAINING_DP}" ]]; then
            TRAINING_DP="$(python - "${TRAINING_ARTIFACT}" <<'PYDP'
import json, sys
entries = json.load(open(sys.argv[1]))["entries"]
meshes = {
    int(e["selector"]["shard_size"]) * int(e["selector"]["replicate_size"])
    for e in entries
    if "shard_size" in e["selector"]
}
print(max(meshes) if meshes else 1)
PYDP
)"
        fi
        # A cell is a measurement of one mesh, so the mesh must be named explicitly when
        # the artifact spans several shard degrees.
        SHARD_SIZE="${SHARD_SIZE:-${TRAINING_DP}}"
        REPLICATE_SIZE="${REPLICATE_SIZE:-$(( TRAINING_DP / SHARD_SIZE ))}"
        TRAINING_OVERRIDES=(
            training_cost=fsdp
            training_cost.shard_size="${SHARD_SIZE}"
            training_cost.replicate_size="${REPLICATE_SIZE}"
            training_cost.artifact_path="${TRAINING_ARTIFACT}"
            training_cost.schedule=fsdp
            training_cost.batching_mode="${BATCHING_MODE}"
            training_cost.parallelism.tp=1
            training_cost.parallelism.pp=1
            training_cost.parallelism.dp="${TRAINING_DP}"
            training_cost.parallelism.cp=1
            training_cost.micro_batch_size=1
            # Must match a profiled discrete cell; the sweep ran gradient checkpointing on.
            training_cost.activation_recompute="${ACTIVATION_RECOMPUTE:-full}"
        )
        if [[ -n "${MAX_TOKENS_PER_RANK}" ]]; then
            TRAINING_OVERRIDES+=( training_cost.max_tokens_per_rank="${MAX_TOKENS_PER_RANK}" )
        fi
        if [[ -n "${EXTRAPOLATE}" ]]; then
            TRAINING_OVERRIDES+=( "training_cost.extrapolate=${EXTRAPOLATE}" )
        fi
    else
        # A hydraulis cell is a measurement of one (TP, PP) topology, so the topology must be
        # named. Defaulting these to 1 silently pinned every run to the tp=1 cell and made the
        # fitted tp=4, tp=8, and pp=4 cells unreachable.
        TRAINING_OVERRIDES=(
            training_cost=hydraulis
            training_cost.artifact_path="${TRAINING_ARTIFACT}"
            training_cost.parallelism.tp="${TRAINING_TP:-1}"
            training_cost.parallelism.pp="${TRAINING_PP:-1}"
            training_cost.parallelism.dp="${TRAINING_DP:-1}"
            training_cost.parallelism.cp="${TRAINING_CP:-1}"
            training_cost.micro_batch_size="${MICRO_BATCH_SIZE:-1}"
            training_cost.activation_recompute="${ACTIVATION_RECOMPUTE:-full}"
        )
        if [[ -n "${EXTRAPOLATE}" ]]; then
            TRAINING_OVERRIDES+=( "training_cost.extrapolate=${EXTRAPOLATE}" )
        fi
    fi
fi

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
python -m dryrun.cli.main \
    workload="${WORKLOAD}" \
    policy="${POLICY}" \
    rollout_cost="${ROLLOUT_MODEL}" \
    "${COST_OVERRIDES[@]}" \
    rollout_cost.parallelism.tp="${ROLLOUT_TP}" \
    rollout_cost.parallelism.pp=1 \
    rollout_cost.parallelism.dp=1 \
    rollout_cost.parallelism.cp=1 \
    "${DIST_OVERRIDES[@]+"${DIST_OVERRIDES[@]}"}" \
    "${TRAINING_OVERRIDES[@]}" \
    job.n_versions="${N_VERSIONS}" \
    job.batch_size="${BATCH_SIZE}" \
    job.max_staleness="${MAX_STALENESS}" \
    job.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    job.max_output_length="${MAX_OUTPUT_LENGTH}" \
    rollout.n_instances="${N_INSTANCES}" \
    rollout.token_budget="${TOKEN_BUDGET}" \
    output_dir="${OUTPUT_DIR}" \
    ${EXTRA_OVERRIDES:+${EXTRA_OVERRIDES}}
