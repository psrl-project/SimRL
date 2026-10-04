#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TP="${TP:-1}"
CONFIG="${CONFIG:-${ROOT_DIR}/examples/profile/kernel/qwen3_8b_h20_tp${TP}.yaml}"
# Restricts `run` to one kernel family, e.g. to re-sweep collectives without spending hours
# re-measuring gemm and attention.
#
# The tp=4 and tp=8 configs are superseded for collectives: one sweep now writes world sizes
# 2, 4 and 8 into a single table, because the `collective:` block declares its own
# `topologies` instead of reading `parallelism.tensor_parallel`. Those configs remain only
# for compute sweeps at those degrees. Cross-node coverage lives in
# collective_multinode.yaml, which needs hostnames and passwordless ssh.
FAMILY="${FAMILY:-all}"
ACTION="${1:-all}"

case "${ACTION}" in
    run)
        dryrun-profile kernels run --config "${CONFIG}" --family "${FAMILY}"
        ;;
    fit-floor)
        dryrun-profile kernels fit-floor --config "${CONFIG}"
        ;;
    holdout)
        dryrun-profile kernels holdout --config "${CONFIG}"
        ;;
    all)
        # fit-floor must follow run and precede holdout: it reads the tables that run
        # writes, and holdout scores the surface using the floors it fits. Skipping it
        # leaves every query falling back to floors fitted for H20: correct here, wrong
        # on any other GPU or backend version, and only a log warning says so.
        dryrun-profile kernels run --config "${CONFIG}" --family "${FAMILY}"
        dryrun-profile kernels fit-floor --config "${CONFIG}"
        dryrun-profile kernels holdout --config "${CONFIG}"
        ;;
    *)
        echo "Usage: $0 [run|fit-floor|holdout|all]" >&2
        echo "  TP=1|2|4|8       selects examples/profile/kernel/qwen3_8b_h20_tp\${TP}.yaml" >&2
        echo "  FAMILY=collective  restricts run to one family (one sweep covers ws 2/4/8)" >&2
        exit 1
        ;;
esac
