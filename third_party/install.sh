#!/bin/bash
set -e
set -o pipefail
trap 'echo "[ERROR] Failed at line $LINENO: $BASH_COMMAND" >&2; exit 1' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIN_FILE="${SCRIPT_DIR}/pins.json"

pin_value() {
    python - "${PIN_FILE}" "$1" <<'PY'
import json
import sys

path = sys.argv[2].split(".")
with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)
for key in path:
    value = value[key]
print(value)
PY
}

VLLM_REPO="$(pin_value "sources.vllm.repository")"
VLLM_TAG="$(pin_value "sources.vllm.tag")"
VLLM_COMMIT="$(pin_value "sources.vllm.commit")"
VLLM_DIR="${SCRIPT_DIR}/$(pin_value "sources.vllm.directory")"

BRIDGE_REPO="$(pin_value "sources.megatron_bridge.repository")"
BRIDGE_TAG="$(pin_value "sources.megatron_bridge.tag")"
BRIDGE_COMMIT="$(pin_value "sources.megatron_bridge.commit")"
BRIDGE_DIR="${SCRIPT_DIR}/$(pin_value "sources.megatron_bridge.directory")"
MEGATRON_LM_COMMIT="$(pin_value "sources.megatron_bridge.submodules.3rdparty/Megatron-LM.commit")"

TORCH_VERSION="$(pin_value "pytorch.torch")"
TORCHVISION_VERSION="$(pin_value "pytorch.torchvision")"
TORCHAUDIO_VERSION="$(pin_value "pytorch.torchaudio")"
TORCH_INDEX_URL="$(pin_value "pytorch.index_url")"
TRANSFORMERS_VERSION="$(pin_value "huggingface.transformers")"
FLASHINFER_PYTHON_VERSION="$(pin_value "flashinfer.python")"
FLASHINFER_CUBIN_VERSION="$(pin_value "flashinfer.cubin")"
FLASHINFER_INDEX_URL="$(pin_value "flashinfer.index_url")"
FLASH_ATTENTION_VERSION="$(pin_value "flash_attention.version")"
FLASH_ATTENTION_CUDA_ARCHS="$(pin_value "flash_attention.cuda_archs")"

clone_pinned() {
    local repository="$1"
    local destination="$2"
    local commit="$3"
    local recurse_submodules="$4"

    if [[ -e "${destination}" && ! -d "${destination}/.git" ]]; then
        echo "Refusing to replace non-git path ${destination}." >&2
        exit 1
    fi

    if [[ ! -d "${destination}/.git" ]]; then
        local clone_args=(clone --filter=blob:none)
        if [[ "${recurse_submodules}" == "1" ]]; then
            clone_args+=(--recurse-submodules)
        fi
        git "${clone_args[@]}" "${repository}" "${destination}"
    fi

    local current
    current="$(git -C "${destination}" rev-parse HEAD)"
    if [[ "${current}" != "${commit}" ]]; then
        if ! git -C "${destination}" diff --quiet || ! git -C "${destination}" diff --cached --quiet; then
            echo "Refusing to change ${destination}: tracked source files are modified." >&2
            exit 1
        fi
        git -C "${destination}" fetch --depth 1 origin "${commit}"
        git -C "${destination}" checkout --detach "${commit}"
    fi

    if [[ "${recurse_submodules}" == "1" ]]; then
        git -C "${destination}" submodule sync --recursive
        git -C "${destination}" submodule update --init --recursive
    fi

    current="$(git -C "${destination}" rev-parse HEAD)"
    if [[ "${current}" != "${commit}" ]]; then
        echo "Commit verification failed for ${destination}: expected ${commit}, got ${current}." >&2
        exit 1
    fi
}

echo "Using python: $(command -v python)"
python -c 'import sys; assert sys.version_info >= (3, 12), sys.version'

echo "0. Install uv"
python -m pip install uv
python -m uv pip install --upgrade pip wheel

echo "1. Install pytorch ${TORCH_VERSION} (${TORCH_INDEX_URL})"
python -m uv pip install --no-cache-dir \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

echo "2. Clone vLLM ${VLLM_TAG} (${VLLM_COMMIT})"
clone_pinned "${VLLM_REPO}" "${VLLM_DIR}" "${VLLM_COMMIT}" 0

echo "3. Install vLLM (precompiled, keep existing torch)"
(
    cd "${VLLM_DIR}"
    python use_existing_torch.py
)
python -m uv pip install -r "${VLLM_DIR}/requirements/build/cuda.txt"
VLLM_USE_PRECOMPILED=1 python -m uv pip install --no-build-isolation --editable "${VLLM_DIR}[bench]"

echo "4. Clone Megatron-Bridge ${BRIDGE_TAG} (${BRIDGE_COMMIT})"
clone_pinned "${BRIDGE_REPO}" "${BRIDGE_DIR}" "${BRIDGE_COMMIT}" 1
ACTUAL_CORE="$(git -C "${BRIDGE_DIR}/3rdparty/Megatron-LM" rev-parse HEAD)"
if [[ "${ACTUAL_CORE}" != "${MEGATRON_LM_COMMIT}" ]]; then
    echo "Megatron-Core verification failed: expected ${MEGATRON_LM_COMMIT}, got ${ACTUAL_CORE}." >&2
    exit 1
fi

echo "5. Install Megatron-Bridge from uv.lock"
command -v uv >/dev/null 2>&1 || {
    echo "uv is required to install Megatron-Bridge from its lock file." >&2
    exit 1
}
BRIDGE_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
UV_SYNC_ARGS=(--active --locked --extra recipes --inexact)
# Megatron-Bridge 0.4.2 locks transformers==5.3.0 and flashinfer-python==0.5.3.
# vLLM 0.26 needs transformers>=5.5.3 and flashinfer 0.6.14 (autotune cache=).
# Skip those locked wheels and restore the vLLM-compatible pins after sync.
UV_SYNC_ARGS+=(--no-install-package transformers --no-install-package flashinfer-python)
NVRX_FROM_SOURCE=0
# nvidia-resiliency-ext 0.6.0 only publishes manylinux_2_39 wheels (glibc 2.39+).
# This host reports manylinux_2_28; NVIDIA documents a source build in that case.
if ! python - <<'PY'
import ctypes
import sys

lib = ctypes.CDLL("libc.so.6")
lib.gnu_get_libc_version.restype = ctypes.c_char_p
parts = [int(p) for p in lib.gnu_get_libc_version().decode().split(".")[:2]]
sys.exit(0 if parts >= [2, 39] else 1)
PY
then
    echo "Host glibc < 2.39; skip locked nvidia-resiliency-ext wheel and build v0.6.0 from source"
    UV_SYNC_ARGS+=(--no-install-package nvidia-resiliency-ext)
    NVRX_FROM_SOURCE=1
fi
(
    cd "${BRIDGE_DIR}"
    # transformer-engine / mamba-ssm / causal-conv1d are no-build-isolation and
    # import pybind11/ninja from the current env. Match Megatron-LM: install the
    # locked build group first, then the project.
    VIRTUAL_ENV="${BRIDGE_PREFIX}" uv sync \
        --active \
        --locked \
        --only-group build \
        --no-install-project \
        --inexact
    VIRTUAL_ENV="${BRIDGE_PREFIX}" uv sync "${UV_SYNC_ARGS[@]}"
)
if [[ "${NVRX_FROM_SOURCE}" == 1 ]]; then
    python -m uv pip install \
        defusedxml grpcio "grpcio-tools>=1.76.0,<1.77.0" httpx \
        "nvidia-ml-py>=12.570.86" packaging psutil pyyaml numpy "protobuf>=4.22.0"
    STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1 python -m uv pip install --no-deps \
        "git+https://github.com/NVIDIA/nvidia-resiliency-ext.git@v0.6.0"
fi

echo "6. Pin transformers ${TRANSFORMERS_VERSION} and flashinfer ${FLASHINFER_PYTHON_VERSION} for vLLM"
python -m uv pip install "transformers==${TRANSFORMERS_VERSION}"
python -m uv pip install --extra-index-url "${FLASHINFER_INDEX_URL}" \
    "flashinfer-python==${FLASHINFER_PYTHON_VERSION}" \
    "flashinfer-cubin==${FLASHINFER_CUBIN_VERSION}"

echo "7. Install FlashAttention ${FLASH_ATTENTION_VERSION} (source build, sm${FLASH_ATTENTION_CUDA_ARCHS})"
# The published wheels link against GLIBC_2.32 while this host provides 2.28, so
# the prebuilt artifact imports but fails to load its C extension. Building from
# source is the only option here, the same situation as nvidia-resiliency-ext above.
#
# MAX_JOBS is capped deliberately. The host reports hundreds of cores but only a
# few GiB of free RAM, and each nvcc invocation on these kernels peaks near a
# gigabyte, so an uncapped build is killed by the OOM reaper with no diagnostic.
# Restricting the architecture list to the target GPU keeps the build to one arch
# instead of the six the sdist compiles by default.
MAX_JOBS="${FLASH_ATTENTION_MAX_JOBS:-8}" \
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTENTION_CUDA_ARCHS}" \
TORCH_CUDA_ARCH_LIST="9.0" \
FLASH_ATTENTION_FORCE_BUILD="TRUE" \
FLASH_ATTENTION_SKIP_CUDA_BUILD="FALSE" \
    python -m uv pip install --no-build-isolation --no-deps \
    "flash-attn==${FLASH_ATTENTION_VERSION}"

echo "8. Verify imports"
python -c "import vllm; print('  vLLM', vllm.__version__)"
python -c "import megatron.bridge, megatron.core; print('  Megatron-Bridge / Megatron-Core OK')"
python -c "from nvidia_resiliency_ext.inprocess import CallWrapper; print('  nvidia-resiliency-ext OK')"
python -c "import torch; print('  PyTorch', torch.__version__, 'CUDA', torch.cuda.is_available())"
python -c "import transformers; from transformers.configuration_utils import ALLOWED_LAYER_TYPES; print('  transformers', transformers.__version__)"
python -c "import flashinfer, inspect; from flashinfer.autotuner import autotune; print('  flashinfer', flashinfer.__version__, inspect.signature(autotune))"
# Importing the C extension explicitly: the pure-python package imports fine even
# when the compiled object cannot load, which is exactly how the wheel failed.
python -c "import flash_attn, flash_attn_2_cuda; from flash_attn import flash_attn_varlen_func; print('  flash-attn', flash_attn.__version__)"

echo "Successfully installed vLLM and Megatron-Bridge into the current environment"
