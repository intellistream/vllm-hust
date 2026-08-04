#!/usr/bin/env bash
# Validate that vllm-hust and vllm-ascend-hust can be installed as editable
# packages into the same clean Python 3.12 environment (metadata/build smoke).
#
# NOTE: This script performs a metadata/build smoke test, not a full dependency
# compatibility check. Both editable installs use --no-deps, so pip check only
# verifies metadata consistency of the installed distributions, not that the
# union of runtime dependencies is jointly satisfiable. Full dependency
# resolution is validated separately in the vllm-ascend-hust CI.
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
# On some systems (e.g. Kylin) python3 -m venv does not bundle pip.
if ! python -m pip --version &>/dev/null; then
  echo "pip not found in venv – bootstrapping with ensurepip …"
  if ! python -m ensurepip --upgrade 2>/dev/null; then
    echo "ensurepip failed – falling back to get-pip.py …"
    python -c "$(curl -fsSL https://bootstrap.pypa.io/get-pip.py)"
  fi
fi
python -m pip install --upgrade pip setuptools wheel
echo "::endgroup::"

# --------------------------------------------------------------------------- #
# Step 2 – Upgrade build tooling
# --------------------------------------------------------------------------- #
echo "::group::Step 2 – Upgrade build tooling"
python -m pip install --upgrade \
  "setuptools>=77.0.3,<81.0.0" \
  "setuptools-scm>=8.0" \
  "setuptools-rust>=1.9.0" \
  "wheel" \
  "packaging>=24.2"
echo "::endgroup::"

# --------------------------------------------------------------------------- #
# Step 2b – Install torch (CPU) as build-time prerequisite
# --------------------------------------------------------------------------- #
echo "::group::Step 2b – Install torch (CPU) for metadata generation"
# setup.py unconditionally imports torch at the top level, so it must be
# present even for --no-deps editable installs.  Use the CPU-only wheel to
# keep the CI environment lightweight.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
echo "::endgroup::"

# --------------------------------------------------------------------------- #
# Step 3 – Install vllm-ascend-hust (plugin first)
# --------------------------------------------------------------------------- #
echo "::group::Step 3 – Install vllm-ascend-hust (editable, --no-deps)"
# SOC_VERSION is required by vllm-ascend-hust's setup.py for chip detection.
# On CPU-only hosts (no npu-smi), set a default so metadata generation succeeds.
SOC_VERSION="${SOC_VERSION:-ascend910b1}"
export SOC_VERSION
COMPILE_CUSTOM_KERNELS=0 python -m pip install -e "$VLLM_ASCEND_HUST_REPO" --no-build-isolation --no-deps
echo "::endgroup::"

# --------------------------------------------------------------------------- #
# Step 4 – Install vllm-hust (core)
# --------------------------------------------------------------------------- #
echo "::group::Step 4 – Install vllm-hust (editable, --no-deps)"
VLLM_TARGET_DEVICE=empty python -m pip install -e "$VLLM_HUST_REPO" --no-build-isolation --no-deps
echo "::endgroup::"

# --------------------------------------------------------------------------- #
# Step 5 – Installed package metadata verification
# --------------------------------------------------------------------------- #
# Verify both packages are registered in the environment's metadata. This is a
# metadata/build smoke check: since both editable installs used --no-deps, the
# runtime dependencies (numpy, transformers, etc.) are NOT installed, so an
# actual `import vllm` would fail with ModuleNotFoundError. Full import
# validation is delegated to the vllm-ascend-hust CI which resolves the full
# dependency set.
echo "::group::Step 5 – Installed package metadata verification"
python -c "
import importlib.metadata as md
for pkg_name in ['vllm', 'vllm-ascend-hust']:
    try:
        dist = md.distribution(pkg_name)
        print(f'  {pkg_name:30s} {dist.version}')
    except md.PackageNotFoundError:
        raise SystemExit(f'ERROR: {pkg_name} not found in environment metadata')
print('Both packages are registered in the environment.')
"
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
# Step 7 – Metadata/build smoke check (NOT full dependency compatibility)
# --------------------------------------------------------------------------- #
# Both editable installs used --no-deps, so pip check here only verifies the
# installed distributions' metadata is self-consistent (e.g. version pins in
# pyproject.toml vs. installed versions). It does NOT prove the union of
# runtime dependencies is jointly satisfiable. Full dependency resolution is
# validated separately in the vllm-ascend-hust CI.
echo "::group::Step 7 – Metadata/build smoke check"
# Both editable installs used --no-deps, so pip check will report many
# "requires X, which is not installed" entries. These are EXPECTED in a
# --no-deps environment and do not indicate a metadata conflict. We only
# fail on real version conflicts (e.g. "requires X>=1.0, but you have X 0.9").
CHECK_OUTPUT=$(python -m pip check 2>&1) || true
if [[ -z "$CHECK_OUTPUT" ]]; then
  echo "pip check passed – no issues detected."
else
  # Filter out "requires X, which is not installed" (expected with --no-deps)
  REAL_CONFLICTS=$(echo "$CHECK_OUTPUT" | grep -v "which is not installed" || true)
  if [[ -z "$REAL_CONFLICTS" ]]; then
    echo "pip check reports only missing-dependency entries (expected with --no-deps)."
    echo "These are not metadata conflicts – the packages metadata is self-consistent."
    echo "Full dependency resolution is validated separately in vllm-ascend-hust CI."
  else
    echo "pip check found real version conflicts:"
    echo "$REAL_CONFLICTS"
    echo ""
    echo "::error::Real dependency version conflicts detected."
    exit 1
  fi
fi
echo "::endgroup::"

echo "============================================="
echo " Dual-editable install metadata/build smoke succeeded."
echo " (Not a full dependency compatibility check.)"
echo "============================================="
