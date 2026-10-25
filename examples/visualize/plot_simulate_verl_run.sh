#!/usr/bin/env bash
# Render the instance-load plot from a finished verl_aligned.sh simulation,
# drawn exactly the way plot_verl_run.sh draws the real run, so the two can
# be compared.
#
# This does not re-simulate. Point it at the output_dir of a prior
# verl_aligned.sh run.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

SIM_RUN_DIR="${SIM_RUN_DIR:-${ROOT_DIR}/outputs/simulate/verl_aligned}"
TIME_LABEL="${TIME_LABEL:-Wall-clock Time (s)}"
FIGURE="${FIGURE:-${SCRIPT_DIR}/instance_load_verl_simulated.png}"

if [[ ! -d "${SIM_RUN_DIR}/logs" ]]; then
    echo "Missing simulation logs: ${SIM_RUN_DIR}/logs" >&2
    echo "Simulate first, for example: bash examples/simulate/verl_aligned.sh" >&2
    echo "Then point SIM_RUN_DIR at that run's output_dir." >&2
    exit 1
fi

python -m srl.cli.plot "${SIM_RUN_DIR}" \
    --only instance_load \
    --time-label "${TIME_LABEL}"

cp "${SIM_RUN_DIR}/plots/instance_load.png" "${FIGURE}"
echo "Figure copied to: ${FIGURE}"
