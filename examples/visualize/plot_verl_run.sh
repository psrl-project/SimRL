#!/usr/bin/env bash
# Render the instance-load plot from a real verl separate_async run's
# telemetry directory, using the same plotting code the simulator uses, so
# the two figures can be compared directly.
#
# The telemetry directory is read only and never modified.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VERL_LOG_DIR="${VERL_LOG_DIR:-/apdcephfs_zwfy10_303541817/share_303541817/lhy/verl-new/examples/psrl_compare/output/telemetry}"
RUN_NAME="${RUN_NAME:-verl_separate_async}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/visualize/${RUN_NAME}}"
FIGURE="${FIGURE:-${SCRIPT_DIR}/instance_load_${RUN_NAME}.png}"
TIME_LABEL="${TIME_LABEL:-Wall-clock Time (s)}"

if [[ ! -d "${VERL_LOG_DIR}" ]]; then
    echo "Missing verl telemetry directory: ${VERL_LOG_DIR}" >&2
    echo "Point VERL_LOG_DIR at the run's VERL_TELEMETRY_LOG_DIR, for example:" >&2
    echo "  VERL_LOG_DIR=/path/to/telemetry bash examples/visualize/plot_verl_run.sh" >&2
    exit 1
fi

python -m srl.cli.verl_plot "${VERL_LOG_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --time-label "${TIME_LABEL}"

cp "${OUTPUT_DIR}/plots/instance_load.png" "${FIGURE}"
echo "Figure copied to: ${FIGURE}"
