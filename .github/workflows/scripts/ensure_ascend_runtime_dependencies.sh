#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_HUST_REPO="${VLLM_HUST_REPO:-$PWD}"
ASCEND_TORCH_VERSION="${ASCEND_TORCH_VERSION:-2.10.0}"
ASCEND_TORCH_NPU_VERSION="${ASCEND_TORCH_NPU_VERSION:-2.10.0}"
ASCEND_TORCHVISION_VERSION="${ASCEND_TORCHVISION_VERSION:-0.25.0}"
ASCEND_TORCHAUDIO_VERSION="${ASCEND_TORCHAUDIO_VERSION:-2.10.0}"
ASCEND_NUMPY_SPEC="${ASCEND_NUMPY_SPEC:-numpy<2.0.0}"
ASCEND_TRITON_VERSION="${ASCEND_TRITON_VERSION:-3.2.1}"
ASCEND_TRITON_SPEC="triton-ascend==$ASCEND_TRITON_VERSION"
ASCEND_TRITON_INDEX_URL="${ASCEND_TRITON_INDEX_URL:-https://mirrors.huaweicloud.com/ascend/repos/pypi}"

COMMON_REQUIREMENTS="$VLLM_HUST_REPO/requirements/common.txt"
if [[ ! -f "$COMMON_REQUIREMENTS" ]]; then
  echo "vLLM common requirements not found: $COMMON_REQUIREMENTS" >&2
  exit 1
fi

CONSTRAINTS_FILE="${RUNNER_TEMP:-/tmp}/ascend-runtime-constraints.txt"
mkdir -p "$(dirname "$CONSTRAINTS_FILE")"
{
  printf 'torch==%s\n' "$ASCEND_TORCH_VERSION"
  printf 'torch-npu==%s\n' "$ASCEND_TORCH_NPU_VERSION"
  printf 'torchvision==%s\n' "$ASCEND_TORCHVISION_VERSION"
  printf 'torchaudio==%s\n' "$ASCEND_TORCHAUDIO_VERSION"
  printf '%s\n' "$ASCEND_NUMPY_SPEC"
  printf '%s\n' "$ASCEND_TRITON_SPEC"
} > "$CONSTRAINTS_FILE"

echo "Installing Ascend runtime dependencies with constraints:"
cat "$CONSTRAINTS_FILE"
PIP_DISABLE_PIP_VERSION_CHECK=1 "$PYTHON_BIN" -m pip install \
  -c "$CONSTRAINTS_FILE" \
  "$ASCEND_NUMPY_SPEC" \
  attrs \
  "cmake>=3.26" \
  decorator \
  googleapis-common-protos \
  msgpack \
  numba \
  "packaging>=24.2" \
  pandas \
  pandas-stubs \
  psutil \
  pybind11 \
  quart \
  scipy \
  "setuptools-scm>=8" \
  "setuptools-rust>=1.9.0" \
  "xgrammar>=0.1.30" \
  "compressed-tensors>=0.11.0" \
  "arctic-inference==0.1.1" \
  "transformers==5.5.4" \
  "jsonschema>=4" \
  "huggingface_hub>=0.20"
PIP_DISABLE_PIP_VERSION_CHECK=1 "$PYTHON_BIN" -m pip install \
  -c "$CONSTRAINTS_FILE" \
  -r "$COMMON_REQUIREMENTS"
PIP_DISABLE_PIP_VERSION_CHECK=1 "$PYTHON_BIN" -m pip install \
  --no-deps \
  --index-url "$ASCEND_TRITON_INDEX_URL" \
  "$ASCEND_TRITON_SPEC"

TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" - \
  "$ASCEND_TORCH_VERSION" "$ASCEND_TORCH_NPU_VERSION" \
  "$ASCEND_TORCHVISION_VERSION" "$ASCEND_TORCHAUDIO_VERSION" \
  "$ASCEND_TRITON_VERSION" <<'PY'
import importlib.metadata as metadata
import sys

import attr
import decorator
import numpy
import psutil
import regex
import scipy
import torch
import torch_npu

(
    expected_torch,
    expected_torch_npu,
    expected_torchvision,
    expected_torchaudio,
    expected_triton,
) = sys.argv[1:6]


def normalize(version: str) -> str:
    return version.split("+", 1)[0]

for distribution in (
    "attrs",
    "decorator",
    "numpy",
    "psutil",
    "regex",
    "scipy",
    "torchvision",
    "torchaudio",
    "triton-ascend",
):
    print(f"{distribution}={metadata.version(distribution)}")

print(f"torch={torch.__version__}")
print(f"torch-npu={torch_npu.__version__}")
if normalize(torch.__version__) != expected_torch:
    raise SystemExit(f"torch changed from the pinned version: {torch.__version__}")
if normalize(torch_npu.__version__) != expected_torch_npu:
    raise SystemExit(
        f"torch-npu changed from the pinned version: {torch_npu.__version__}"
    )
if normalize(metadata.version("torchvision")) != expected_torchvision:
    raise SystemExit("torchvision changed from the pinned version")
if normalize(metadata.version("torchaudio")) != expected_torchaudio:
    raise SystemExit("torchaudio changed from the pinned version")
if normalize(metadata.version("triton-ascend")) != expected_triton:
    raise SystemExit("triton-ascend changed from the pinned version")
if int(numpy.__version__.split(".", 1)[0]) >= 2:
    raise SystemExit(f"NumPy is outside the supported Ascend range: {numpy.__version__}")
PY
