#!/usr/bin/env bash
# Sweep every preset. psrl and verl_v1 lead, because they are the contrast the
# study is built around: a bound that waits against a bound that discards.
#
# Presets run in sequence while their own points run in parallel, so one
# preset's failure does not hide eleven others. Each preset's exit status is
# reported at the end.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PRESETS=(
    psrl
    verl_v1
    verl
    areal
    roll
    slime
    skyrl
    openrlhf
    dressage
    nemo_rl
    trl
    prime_rl
)

declare -A status
for preset in "${PRESETS[@]}"; do
    echo ""
    echo "================ ${preset} ================"
    if bash "${SCRIPT_DIR}/sweep.sh" "${preset}"; then
        status["${preset}"]=ok
    else
        status["${preset}"]=FAILED
    fi
done

echo ""
echo "================ summary ================"
failed=0
for preset in "${PRESETS[@]}"; do
    printf '  %-10s %s\n' "${preset}" "${status[${preset}]}"
    [[ "${status[${preset}]}" == "ok" ]] || failed=1
done
exit "${failed}"
