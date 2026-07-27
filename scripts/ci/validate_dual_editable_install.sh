#!/usr/bin/env bash
# Validate that vllm-hust and vllm-ascend-hust can be installed as editable
# packages into the same clean Python 3.12 environment without dependency
# conflicts.
#
# Required environment variables:
#   VLLM_ASCEND_HUST_REPO  - path to a vllm-ascend-hust checkout
#
# Optional environment variables:
#   VLLM_HUST_REPO         - path to the vllm-hust checkout (auto-detected)
#   VENV_DIR               - virtualenv location (auto-created in a temp dir)
#
# This script is designed for CI (GitHub Actions ubuntu-latest) and local
# developer verification.  It does NOT require Ascend NPU hardware.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VLLM_HUST_REPO="${VLLM_HUST_REPO:-$REPO_ROOT}"
VLLM_ASCEND_HUST_REPO="${VLLM_ASCEND_HUST_REPO:?VLLM_ASCEND_HUST_REPO is required}"
VENV_DIR="${VENV_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/dual-editable-XXXXXX")}"

# Validate that paths are local directories, not URLs.
if [[ "$VLLM_ASCEND_HUST_REPO" == http* ]]; then
  echo "::error::VLLM_ASCEND_HUST_REPO must be a local path, not a URL: $VLLM_ASCEND_HUST_REPO" >&2
  echo "Usage: VLLM_ASCEND_HUST_REPO=/path/to/local/vllm-ascend-hust bash $0" >&2
  exit 1
fi
if [[ ! -d "$VLLM_ASCEND_HUST_REPO" ]]; then
  echo "::error::VLLM_ASCEND_HUST_REPO directory does not exist: $VLLM_ASCEND_HUST_REPO" >&2
  exit 1
fi
if [[ ! -d "$VLLM_HUST_REPO" ]]; then
  echo "::error::VLLM_HUST_REPO directory does not exist: $VLLM_HUST_REPO" >&2
  exit 1
fi

cleanup() {
  echo "::endgroup::"
}
trap cleanup EXIT

echo "::group::Environment"
echo "vllm-hust repo:       $VLLM_HUST_REPO"
echo "vllm-ascend-hust repo: $VLLM_ASCEND_HUST_REPO"
echo "virtualenv:           $VENV_DIR"
echo "Python:               $(command -v python3 || true)"
echo "::endgroup::"

# --------------------------------------------------------------------------- #
# Step 1 – Create a clean Python 3.12 virtual environment
# --------------------------------------------------------------------------- #
echo "::group::Step 1 – Create virtual environment"
if command -v uv &>/dev/null; then
  uv venv --python 3.12 "$VENV_DIR"
else
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python --version
pip install --upgrade pip setuptools wheel
echo "::endgroup::"

# --------------------------------------------------------------------------- #
# Step 2 – Upgrade build tooling
# --------------------------------------------------------------------------- #
echo "::group::Step 2 – Upgrade build tooling"
pip install --upgrade \
  "setuptools>=77.0.3,<81.0.0" \
  "setuptools-scm>=8.0" \
  "setuptools-rust>=1.9.0" \
  "wheel" \
  "packaging>=24.2"
echo "::endgroup::"

# --------------------------------------------------------------------------- #
# Step 3 – Install vllm-ascend-hust (plugin first)
# --------------------------------------------------------------------------- #
echo "::group::Step 3 – Install vllm-ascend-hust (editable, --no-deps)"
COMPILE_CUSTOM_KERNELS=0 pip install -e "$VLLM_ASCEND_HUST_REPO" --no-build-isolation --no-deps
echo "::endgroup::"

# --------------------------------------------------------------------------- #
# Step 4 – Install vllm-hust (core)
# --------------------------------------------------------------------------- #
echo "::group::Step 4 – Install vllm-hust (editable, --no-deps)"
VLLM_TARGET_DEVICE=empty pip install -e "$VLLM_HUST_REPO" --no-build-isolation --no-deps
echo "::endgroup::"

# --------------------------------------------------------------------------- #
# Step 5 – Import smoke tests
# --------------------------------------------------------------------------- #
echo "::group::Step 5 – Import smoke tests"
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import vllm_ascend; print(f'vllm-ascend version: {vllm_ascend.__version__}')"
echo "All imports passed."
echo "::endgroup::"

# --------------------------------------------------------------------------- #
# Step 6 – Print key dependency versions
# --------------------------------------------------------------------------- #
echo "::group::Step 6 – Dependency versions"
python -c "
import importlib.metadata as md
pkgs = [
    'vllm',
    'vllm-ascend-hust',
    'fastapi',
    'starlette',
    'transformers',
    'torch',
    'torch-npu',
    'setuptools',
    'pip',
]
seen = set()
for pkg in pkgs:
    key = pkg.lower().replace('-', '_')
    if key in seen:
        continue
    seen.add(key)
    try:
        ver = md.version(pkg)
    except md.PackageNotFoundError:
        ver = '<not installed>'
    print(f'  {pkg:30s} {ver}')
"
echo "::endgroup::"

# --------------------------------------------------------------------------- #
# Step 7 – pip check (warn but do not fail on known optional gaps)
# --------------------------------------------------------------------------- #
echo "::group::Step 7 – pip check"
CHECK_OUTPUT=$(pip check 2>&1) || true
if [[ -z "$CHECK_OUTPUT" ]]; then
  echo "pip check passed – no dependency conflicts detected."
else
  echo "pip check reported the following issues:"
  echo "$CHECK_OUTPUT"
  # Only fail if vllm or vllm-ascend-hust itself has a conflict.
  if echo "$CHECK_OUTPUT" | grep -qE "^vllm |^vllm-ascend-hust "; then
    echo ""
    echo "::error::vllm or vllm-ascend-hust has a dependency conflict."
    exit 1
  fi
  echo ""
  echo "Non-vllm conflicts are pre-existing and ignored."
fi
echo "::endgroup::"

echo "============================================="
echo " Dual-editable install validation succeeded."
echo "============================================="
