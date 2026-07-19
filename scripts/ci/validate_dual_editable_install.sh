#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_VLLM_HUST_REPO=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

VLLM_HUST_REPO=${VLLM_HUST_REPO:-${DEFAULT_VLLM_HUST_REPO}}
VLLM_ASCEND_HUST_REPO=${VLLM_ASCEND_HUST_REPO:-$(cd -- "${VLLM_HUST_REPO}/.." && pwd)/vllm-ascend-hust}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
WORK_DIR=${WORK_DIR:-}
KEEP_WORK_DIR=${KEEP_WORK_DIR:-0}
VLLM_HUST_REF=${VLLM_HUST_REF:-}
VLLM_ASCEND_HUST_REF=${VLLM_ASCEND_HUST_REF:-}

if [[ -z "${WORK_DIR}" ]]; then
  WORK_DIR=$(mktemp -d -t vllm-dual-editable-install.XXXXXX)
  CREATED_WORK_DIR=1
else
  mkdir -p "${WORK_DIR}"
  CREATED_WORK_DIR=0
fi

PYTHON_BIN="${WORK_DIR}/.venv/bin/python"

log() {
  echo "::group::$*"
}

end_log() {
  echo "::endgroup::"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

cleanup() {
  local status=$?

  if [[ "${status}" -ne 0 ]]; then
    echo "Dual editable install validation failed." >&2
    echo "Work directory retained for debugging: ${WORK_DIR}" >&2
    exit "${status}"
  fi

  if [[ "${KEEP_WORK_DIR}" == "1" || "${KEEP_WORK_DIR}" == "true" ]]; then
    echo "Work directory retained by request: ${WORK_DIR}"
  elif [[ "${CREATED_WORK_DIR}" == "1" ]]; then
    rm -rf "${WORK_DIR}"
  fi
}
trap cleanup EXIT

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  echo "uv is not on PATH; installing with python3 -m pip install --user uv"
  python3 -m pip install --user uv
  export PATH="${HOME}/.local/bin:${PATH}"
  command -v uv >/dev/null 2>&1 || die "uv is still unavailable after installation"
}

print_repo_state() {
  local name=$1
  local repo=$2

  [[ -d "${repo}" ]] || die "${name} repo does not exist: ${repo}"
  [[ -d "${repo}/.git" ]] || die "${name} repo is not a git checkout: ${repo}"

  echo "${name}_repo=${repo}"
  git -C "${repo}" rev-parse --show-toplevel
  git -C "${repo}" rev-parse HEAD
  git -C "${repo}" status --short
}

maybe_checkout_ref() {
  local repo=$1
  local ref=$2

  if [[ -z "${ref}" ]]; then
    return
  fi

  echo "Checking out ${ref} in ${repo}"
  git -C "${repo}" fetch --no-tags --depth 1 origin "${ref}"
  git -C "${repo}" checkout --detach FETCH_HEAD
}

log "Environment"
ensure_uv
uname -a
uv --version
echo "PYTHON_VERSION=${PYTHON_VERSION}"
echo "WORK_DIR=${WORK_DIR}"
echo "VLLM_HUST_REPO=${VLLM_HUST_REPO}"
echo "VLLM_ASCEND_HUST_REPO=${VLLM_ASCEND_HUST_REPO}"
echo "UV_INDEX_URL=${UV_INDEX_URL:-<unset>}"
echo "UV_EXTRA_INDEX_URL=${UV_EXTRA_INDEX_URL:-<unset>}"
echo "UV_INDEX_STRATEGY=${UV_INDEX_STRATEGY:-<unset>}"
echo "UV_NO_CACHE=${UV_NO_CACHE:-<unset>}"
end_log

log "Checkout refs"
maybe_checkout_ref "${VLLM_HUST_REPO}" "${VLLM_HUST_REF}"
maybe_checkout_ref "${VLLM_ASCEND_HUST_REPO}" "${VLLM_ASCEND_HUST_REF}"
print_repo_state "vllm_hust" "${VLLM_HUST_REPO}"
print_repo_state "vllm_ascend_hust" "${VLLM_ASCEND_HUST_REPO}"
end_log

log "Create clean Python environment"
uv venv --python "${PYTHON_VERSION}" "${WORK_DIR}/.venv"
"${PYTHON_BIN}" --version
end_log

log "Install vllm-hust editable"
(
  cd "${VLLM_HUST_REPO}"
  VLLM_USE_PRECOMPILED=1 uv pip install \
    --python "${PYTHON_BIN}" \
    -e . \
    --torch-backend=auto
)
end_log

log "Install vllm-ascend-hust editable"
(
  cd "${VLLM_ASCEND_HUST_REPO}"
  COMPILE_CUSTOM_KERNELS=0 uv pip install \
    --python "${PYTHON_BIN}" \
    -e . \
    --no-deps
)
end_log

log "Import smoke and version summary"
"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata
import sys

import vllm
import vllm_ascend

print("python", sys.version.replace("\n", " "))

for dist in [
    "vllm",
    "vllm-hust",
    "vllm-ascend-hust",
    "torch",
    "torch-npu",
    "fastapi",
    "transformers",
]:
    try:
        print(f"{dist}={metadata.version(dist)}")
    except metadata.PackageNotFoundError:
        print(f"{dist}=<not installed>")

print("vllm_module", vllm.__file__)
print("vllm_ascend_module", vllm_ascend.__file__)
PY
end_log

log "Dependency consistency"
uv pip check --python "${PYTHON_BIN}"
end_log

log "Installed package snapshot"
uv pip freeze --python "${PYTHON_BIN}"
end_log

echo "Dual editable install validation passed."
