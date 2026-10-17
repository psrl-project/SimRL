#!/usr/bin/env bash
# Does pricing admission in trajectories beat capping concurrency with a constant?
#
# Four arms over one trace, one seed, one version count. Everything except the
# router is held fixed, so a difference in SCG(S) is attributable.
#
#   no_cap        request_num_balance, no residency cap     the floor
#   static_cap    request_num_balance, ACTIVE_CAP=50        what psrl_aligned ships
#   scg_oracle    scg_admission reading target_len          the ceiling a predictor could reach
#   scg_empirical scg_admission learning the length CDF     the deployable one
#
# `static_cap` is in here because psrl_aligned already sets ACTIVE_CAP=50, so
# beating the shipped default is the claim, not beating an uncapped straw man.
# The two scg arms separate "the mechanism is wrong" from "the predictor is not
# good enough": if the oracle wins and empirical does not, the length model is
# the gap, and both losing indicts the mechanism.
#
# SCG is a RATE, so only compare arms whose Consumed count agrees. The summary
# below prints it per arm and refuses to rank arms that disagree.
#
# Usage:
#   bash examples/simulate/scg_admission_sweep.sh
#   POLICIES="roll" ARMS="no_cap scg_oracle" bash examples/simulate/scg_admission_sweep.sh
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# verl does not drop on staleness, so its SCG reflects how fast fresh work lands.
# roll GC's aged cohorts, so it also exposes work recovered from doomed
# trajectories. The mechanism should help under both for different reasons.
POLICIES="${POLICIES:-verl roll}"
ARMS="${ARMS:-no_cap static_cap scg_oracle scg_empirical}"
SWEEP_DIR="${SWEEP_DIR:-${ROOT_DIR}/outputs/simulate/scg_admission_sweep}"
# Must match job.max_staleness, which psrl_aligned sets to 2. The strategy
# derives its deadline from this, so a mismatch prices every budget wrongly.
MAX_STALENESS="${MAX_STALENESS:-2}"

# psrl_aligned prices rollout with ComposedRollout, which has no C++ equivalent,
# so the whole sweep runs on the Python engine. Without this the first arm dies
# in `_make_cpp_cost`.
export SIMRL_BACKEND="${SIMRL_BACKEND:-native}"

mkdir -p "${SWEEP_DIR}"

run_arm() {
    local policy="$1" arm="$2"
    local out="${SWEEP_DIR}/${policy}_${arm}"
    local log="${out}.log"
    if [[ -f "${out}/logs/request.jsonl" && -z "${FORCE:-}" ]]; then
        echo "  ${policy}/${arm}: cached"
        return 0
    fi
    echo "  ${policy}/${arm}: running..."

    local strategy="request_num_balance"
    local active_cap=0
    local extra=""
    case "${arm}" in
        no_cap)     ;;
        static_cap) active_cap=50 ;;
        scg_oracle)
            strategy="scg_admission"
            extra="+rollout.router.strategy_args.max_staleness=${MAX_STALENESS}"
            extra="${extra} +rollout.router.strategy_args.length_model=oracle"
            ;;
        scg_empirical)
            strategy="scg_admission"
            extra="+rollout.router.strategy_args.max_staleness=${MAX_STALENESS}"
            extra="${extra} +rollout.router.strategy_args.length_model=empirical"
            # The trace tops out at 11179 tokens and the bins smooth the CDF that
            # G is differenced from. min_samples gates the strategy into load
            # balancing until a version's worth of trajectories has retired.
            extra="${extra} +rollout.router.strategy_args.length_model_args.max_length=16384"
            extra="${extra} +rollout.router.strategy_args.length_model_args.bin_width=256"
            extra="${extra} +rollout.router.strategy_args.length_model_args.min_samples=64"
            ;;
        *) echo "Unknown arm ${arm}" >&2; return 1 ;;
    esac

    POLICY="${policy}" \
    MAX_STALENESS="${MAX_STALENESS}" \
    ACTIVE_CAP="${active_cap}" \
    OUTPUT_DIR="${out}" \
    EXTRA_OVERRIDES="rollout.router.strategy=${strategy} ${extra}" \
        bash "${SCRIPT_DIR}/psrl_aligned.sh" > "${log}" 2>&1 || {
            echo "    FAILED, see ${log}" >&2
            tail -20 "${log}" >&2
            return 1
        }
}

for policy in ${POLICIES}; do
    echo "policy=${policy}"
    for arm in ${ARMS}; do
        run_arm "${policy}" "${arm}" || true
    done
done

echo
echo "── SCG(S) in tokens/second ────────────────────────────────────────────"
python - "${SWEEP_DIR}" "${POLICIES}" "${ARMS}" <<'PYSUMMARY'
import json
import sys
from pathlib import Path

sweep = Path(sys.argv[1])
policies = sys.argv[2].split()
arms = sys.argv[3].split()


def curve(run: Path):
    """Return (scg_by_s, consumed, span) read from a run's request stream."""
    path = run / "logs" / "request.jsonl"
    if not path.exists():
        return None
    consumed, span = [], 0.0
    for line in path.open():
        row = json.loads(line)
        span = max(span, float(row.get("t", 0.0)))
        if row.get("event") == "consume":
            consumed.append(row)
    if not consumed or span <= 0.0:
        return None
    scg = {}
    for s in range(0, 4):
        tokens = sum(
            r.get("generated", 0) for r in consumed if r.get("staleness_true", 99) <= s
        )
        scg[s] = tokens / span
    return scg, len(consumed), span


for policy in policies:
    print(f"\npolicy={policy}")
    rows = {}
    for arm in arms:
        got = curve(sweep / f"{policy}_{arm}")
        if got is None:
            print(f"  {arm:<14} no output")
            continue
        rows[arm] = got
        scg, n, span = got
        cells = "  ".join(f"S={s}:{v:>8.1f}" for s, v in sorted(scg.items()))
        print(f"  {arm:<14} {cells}   consumed={n:<5} span={span:.0f}s")

    # A rate is only comparable across runs that did the same amount of work.
    counts = {arm: rows[arm][1] for arm in rows}
    if len(set(counts.values())) > 1:
        print(f"  ! consumed counts differ {counts}, so these rates are not comparable")
        continue
    if "no_cap" not in rows:
        continue
    for arm in arms:
        if arm == "no_cap" or arm not in rows:
            continue
        ratios = "  ".join(
            f"S={s}:{rows[arm][0][s] / rows['no_cap'][0][s]:.3f}x"
            if rows["no_cap"][0][s] > 0 else f"S={s}:n/a"
            for s in sorted(rows[arm][0])
        )
        print(f"  {arm:<14} vs no_cap: {ratios}")
PYSUMMARY
