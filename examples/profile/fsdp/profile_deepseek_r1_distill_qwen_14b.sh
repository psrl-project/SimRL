#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="${CONFIG:-${ROOT_DIR}/examples/profile/fsdp/deepseek_r1_distill_qwen_14b.yaml}"
ACTION="${1:-generate}"
CANDIDATE_INDEX="${CANDIDATE_INDEX:-0}"

cd "${ROOT_DIR}"

# The candidate manifest lives beside the config's own output_dir, so a smoke
# config cannot overwrite the full sweep's candidates.
if [[ -z "${CANDIDATES:-}" ]]; then
    OUTPUT_DIR="$(python -c '
import sys
from pathlib import Path
from omegaconf import OmegaConf
path = Path(sys.argv[1]).resolve()
value = OmegaConf.load(path)
output = Path(str(value.output_dir))
print(output if output.is_absolute() else (path.parent / output).resolve())
' "${CONFIG}")"
    CANDIDATES="${OUTPUT_DIR}/control/fsdp/candidates.json"
fi

selection_args=()
if [[ "${CANDIDATE_INDEX}" != "all" ]]; then
    selection_args=(--index "${CANDIDATE_INDEX}")
fi

case "${ACTION}" in
    generate)
        dryrun-profile fsdp generate \
            --config "${CONFIG}" \
            --candidates "${CANDIDATES}"
        ;;
    run)
        dryrun-profile fsdp run \
            --config "${CONFIG}" \
            --candidates "${CANDIDATES}" \
            "${selection_args[@]}" \
            "${@:2}"
        ;;
    collect)
        dryrun-profile fsdp collect \
            --config "${CONFIG}" \
            --candidates "${CANDIDATES}" \
            "${selection_args[@]}"
        ;;
    *)
        echo "Usage: $0 [generate|run|collect]" >&2
        exit 2
        ;;
esac
