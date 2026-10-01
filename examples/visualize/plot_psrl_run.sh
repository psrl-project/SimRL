#!/usr/bin/env bash
# Render the instance-load plot from a real PSRL run's logs, using the same
# plotting code the simulator uses, so the two figures can be compared directly.
#
# The PSRL log directory is read only and never modified.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PSRL_LOG_DIR="${PSRL_LOG_DIR:-/apdcephfs_zwfy10_303541817/share_303541817/lhy/psrl/examples/dapo_trainer/megatron_psrl_log/qwen3_8b_megatron_baseline}"
RUN_NAME="${RUN_NAME:-$(basename "${PSRL_LOG_DIR}")}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/visualize/${RUN_NAME}}"

# The converted logs stay under outputs/, but the figure is also dropped beside
# this script so the real and simulated versions sit side by side. The filename
# carries RUN_NAME so plotting a second experiment cannot silently overwrite the
# first one's figure.
FIGURE="${FIGURE:-${SCRIPT_DIR}/instance_load_${RUN_NAME}.png}"

# Which replica role to plot. Validation replicas write the same stats files as
# rollout ones and are told apart only by their collector log's filename.
ROLE="${ROLE:-rollout}"
# Replica indices, e.g. "0,2". Empty means select them by ROLE.
INSTANCES="${INSTANCES:-}"
# UTC offset the text logs were written in. Empty means infer it from the logs,
# which works whenever the run shut down cleanly.
UTC_OFFSET_HOURS="${UTC_OFFSET_HOURS:-}"
TIME_LABEL="${TIME_LABEL:-Wall-clock Time (s)}"

if [[ ! -d "${PSRL_LOG_DIR}" ]]; then
    echo "Missing PSRL log directory: ${PSRL_LOG_DIR}" >&2
    echo "Point PSRL_LOG_DIR at the psrl.logging_path of a finished run, for example:" >&2
    echo "  PSRL_LOG_DIR=/path/to/megatron_psrl_log/<experiment> bash examples/visualize/plot_psrl_run.sh" >&2
    exit 1
fi

ARGS=(
    "${PSRL_LOG_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --role "${ROLE}"
    --time-label "${TIME_LABEL}"
)
if [[ -n "${INSTANCES}" ]]; then
    ARGS+=(--instances "${INSTANCES}")
fi
if [[ -n "${UTC_OFFSET_HOURS}" ]]; then
    ARGS+=(--utc-offset-hours "${UTC_OFFSET_HOURS}")
fi

python -m dryrun.cli.psrl_plot "${ARGS[@]}"

cp "${OUTPUT_DIR}/plots/instance_load.png" "${FIGURE}"
echo "Figure copied to: ${FIGURE}"
