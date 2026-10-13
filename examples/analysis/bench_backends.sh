#!/usr/bin/env bash
# Time the native and C++ engine backends on one fixed configuration, and check
# that they agree before reporting any speed.
#
# Arms differ only in backend and undoable event types. An arm whose request
# telemetry is not byte-identical to native exact undo is reported as DIVERGED.
# Arms are interleaved across REPEATS so background load falls on all of them,
# and the median wall time is printed beside CPU time, which exposes contention.
#
# Usage:
#   bash examples/analysis/bench_backends.sh
#   CAP=160 N_VERSIONS=4 REPEATS=3 bash examples/analysis/bench_backends.sh
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DRYRUN_ENV="${DRYRUN_ENV:-/apdcephfs_zwfy/share_303760348/lhy/env/dryrun-rl.sh}"
# shellcheck disable=SC1090
[[ -f "${DRYRUN_ENV}" ]] && source "${DRYRUN_ENV}" >/dev/null 2>&1

CAP="${CAP:-160}"
N_VERSIONS="${N_VERSIONS:-4}"
REPEATS="${REPEATS:-3}"
TRACE="${TRACE:-${ROOT_DIR}/outputs/workload/dapo_64k/joint.jsonl}"
export MAX_OUTPUT_LENGTH="${MAX_OUTPUT_LENGTH:-65536}"
OUT="${OUT:-/tmp/bench_backends_$(hostname)_cap${CAP}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

EXACT="request_complete,preempt,iteration_done"
FAST="request_complete"
# name:backend:undoable_events
ARMS=(
    "native_exact:native:${EXACT}"
    "cpp_exact:cpp:${EXACT}"
    "native_fast:native:${FAST}"
    "cpp_fast:cpp:${FAST}"
)

mkdir -p "${OUT}"
echo "host=$(hostname) load=$(cut -d' ' -f1-3 /proc/loadavg) cap=${CAP} versions=${N_VERSIONS} repeats=${REPEATS}"
echo "out=${OUT}"

OV="workload.trace.path=${TRACE} rollout.max_concurrency=1024 \
rollout.router.max_concurrent_seqs_per_instance=${CAP} \
rollout.router.max_waiting_after_preempt=1024 \
rollout.router.reject_preempted_on_waiting=true"
# Telemetry streams to drop. Set it empty to time every stream a default run writes.
LOG_EXCLUDE="${LOG_EXCLUDE-instance,rollout_step}"
if [[ -n "${LOG_EXCLUDE}" ]]; then
    OV="${OV} +log_exclude='${LOG_EXCLUDE}'"
fi
echo "log_exclude='${LOG_EXCLUDE}'"
# Per-step telemetry is deleted once read unless KEEP_TELEMETRY=1, since it is large.
KEEP_TELEMETRY="${KEEP_TELEMETRY:-0}"

# One repeat runs every arm in sequence. PARALLEL=1 runs repeats concurrently,
# which is only sound on an idle node with far more cores than repeats.
run_repeat() {
    local rep="$1" arm name backend undo dir wall usr sys done_ok
    for arm in "${ARMS[@]}"; do
        IFS=: read -r name backend undo <<<"${arm}"
        dir="${OUT}/${name}/rep${rep}"
        mkdir -p "${dir}"
        /usr/bin/time -f "%e %U %S" -o "${dir}/time.txt" \
            env DRYRUN_BACKEND="${backend}" DRYRUN_UNDOABLE_EVENTS="${undo}" \
            POLICY=psrl ACTIVE_CAP="${CAP}" N_VERSIONS="${N_VERSIONS}" \
            EXTRA_OVERRIDES="${OV}" OUTPUT_DIR="${dir}" \
            bash "${ROOT_DIR}/examples/simulate/psrl_aligned.sh" >"${dir}/stdout.log" 2>&1 || true
        read -r wall usr sys <"${dir}/time.txt" || { wall=nan; usr=nan; sys=nan; }
        done_ok=$(grep -c "Simulation complete" "${dir}/stdout.log" || true)
        # The summary reads only `request.jsonl`.
        if [[ "${KEEP_TELEMETRY}" != "1" ]]; then
            rm -rf "${dir}/logs/instance" "${dir}/logs/rollout_step.jsonl" "${dir}/plots"
        fi
        printf "  rep%-2s %-13s wall=%8.1fs  cpu=%8.1fs  complete=%s\n" \
            "${rep}" "${name}" "${wall}" "$(echo "${usr}+${sys}" | bc)" "${done_ok}"
    done
}

if [[ "${PARALLEL:-0}" == "1" ]]; then
    for ((rep = 1; rep <= REPEATS; rep++)); do
        run_repeat "${rep}" &
    done
    wait
else
    for ((rep = 1; rep <= REPEATS; rep++)); do
        run_repeat "${rep}"
    done
fi

python - "${OUT}" "${REPEATS}" <<'PY'
import hashlib, json, statistics, sys
from pathlib import Path

out, repeats = Path(sys.argv[1]), int(sys.argv[2])
arms = ["native_exact", "cpp_exact", "native_fast", "cpp_fast"]

def digest(p):
    return hashlib.md5(p.read_bytes()).hexdigest()[:12] if p.is_file() else None

def summary(p):
    rows = [json.loads(l) for l in p.open() if l.strip()]
    con = [r for r in rows if r.get("event") == "consume"]
    span = max(float(r["t"]) for r in rows) if rows else 0.0
    return len(con), span

ref = digest(out / "native_exact" / "rep1" / "logs" / "request.jsonl")
print()
print(f"{'arm':14s} {'wall_med':>9s} {'cpu_med':>9s} {'speedup':>8s}  consumed  span(s)    md5          vs native_exact")
base = None
for arm in arms:
    walls, cpus, md5s = [], [], set()
    consumed = span = None
    for r in range(1, repeats + 1):
        d = out / arm / f"rep{r}"
        try:
            w, u, s = (float(x) for x in (d / "time.txt").read_text().split())
        except Exception:
            continue
        if "Simulation complete" not in (d / "stdout.log").read_text():
            continue
        walls.append(w); cpus.append(u + s)
        req = d / "logs" / "request.jsonl"
        md5s.add(digest(req))
        consumed, span = summary(req)
    if not walls:
        print(f"{arm:14s}  (no completed repeat)")
        continue
    wm, cm = statistics.median(walls), statistics.median(cpus)
    if arm == "native_exact":
        base = wm
    sp = f"{base / wm:.2f}x" if base else "-"
    md5 = ",".join(sorted(m for m in md5s if m))
    verdict = "IDENTICAL" if md5s == {ref} else "DIVERGED"
    print(f"{arm:14s} {wm:9.1f} {cm:9.1f} {sp:>8s}  {consumed:8d}  {span:9.2f}  {md5:12s} {verdict}")
PY
