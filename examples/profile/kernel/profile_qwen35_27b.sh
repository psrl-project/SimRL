#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TP="${TP:-1}"
CONFIG="${CONFIG:-${ROOT_DIR}/examples/profile/kernel/qwen35_27b_h20_tp${TP}.yaml}"
# Restricts `run` to one kernel family, e.g. to re-sweep collectives without spending hours
# re-measuring gemm and attention.
#
# TP is capped at 4 for this checkpoint: it has 4 KV heads. The `collective:` block declares
# its own `topologies`, so one sweep writes world sizes 2, 4 and 8 into a single table
# whatever `parallelism.tensor_parallel` says.
FAMILY="${FAMILY:-all}"
ACTION="${1:-all}"

case "${ACTION}" in
    run)
        srl-profile kernels run --config "${CONFIG}" --family "${FAMILY}"
        ;;
    fit-floor)
        srl-profile kernels fit-floor --config "${CONFIG}"
        ;;
    holdout)
        srl-profile kernels holdout --config "${CONFIG}"
        ;;
    all)
        # fit-floor must follow run and precede holdout: it reads the tables that run
        # writes, and holdout scores the surface using the floors it fits. Skipping it
        # leaves every query falling back to floors fitted for H20: correct here, wrong
        # on any other GPU or backend version, and only a log warning says so.
        srl-profile kernels run --config "${CONFIG}" --family "${FAMILY}"
        srl-profile kernels fit-floor --config "${CONFIG}"
        srl-profile kernels holdout --config "${CONFIG}"
        ;;
    *)
        echo "Usage: $0 [run|fit-floor|holdout|all]" >&2
        echo "  TP=1|2|4|8       selects examples/profile/kernel/qwen35_27b_h20_tp\${TP}.yaml" >&2
        echo "  FAMILY=collective  restricts run to one family (one sweep covers ws 2/4/8)" >&2
        exit 1
        ;;
esac
