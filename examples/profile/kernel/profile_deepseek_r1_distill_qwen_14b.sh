#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TP="${TP:-1}"
CONFIG="${CONFIG:-${ROOT_DIR}/examples/profile/kernel/deepseek_r1_distill_qwen_14b_h20_tp${TP}.yaml}"
# Restricts `run` to one kernel family, e.g. to re-sweep collectives without spending hours
# re-measuring gemm and attention. One collective sweep writes world sizes 2, 4 and 8 into a
# single table, because the `collective:` block declares its own `topologies` rather than
# reading `parallelism.tensor_parallel`, so there is no tp=4 or tp=8 config here.
FAMILY="${FAMILY:-all}"
ACTION="${1:-all}"

case "${ACTION}" in
    run)
        dryrun-profile kernels run --config "${CONFIG}" --family "${FAMILY}"
        ;;
    verify-shapes)
        dryrun-profile kernels verify-shapes --config "${CONFIG}"
        ;;
    fit-floor)
        dryrun-profile kernels fit-floor --config "${CONFIG}"
        ;;
    holdout)
        # --fraction 1.0 deliberately. The default 0.2 scores a random fifth of the
        # eligible points, and a small sample flatters the surface: on the first H20 sweep
        # GEMM measured 3.38% MRE over 13 sampled points but 8.48% over all 64.
        dryrun-profile kernels holdout --config "${CONFIG}" --fraction 1.0
        ;;
    all)
        # fit-floor must follow run and precede holdout: it reads the tables that run
        # writes, and holdout scores the surface using the floors it fits. Skipping it
        # leaves every query falling back to floors fitted for Qwen3-8B on H20, and only a
        # log warning says so.
        dryrun-profile kernels run --config "${CONFIG}" --family "${FAMILY}"
        dryrun-profile kernels fit-floor --config "${CONFIG}"
        dryrun-profile kernels holdout --config "${CONFIG}" --fraction 1.0
        ;;
    *)
        echo "Usage: $0 [run|verify-shapes|fit-floor|holdout|all]" >&2
        echo "  TP=1|2             selects deepseek_r1_distill_qwen_14b_h20_tp\${TP}.yaml" >&2
        echo "  FAMILY=collective  restricts run to one family (one sweep covers ws 2/4/8)" >&2
        exit 1
        ;;
esac
