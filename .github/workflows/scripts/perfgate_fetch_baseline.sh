#!/bin/bash
set -euo pipefail

COMMIT=${1:-${FORK_POINT:-${GITHUB_SHA:-}}}
CENTRAL_REPO_URL=${PERFGATE_CENTRAL_REPO_URL:-https://github.com/vLLM-HUST/vllm-hust-benchmark.git}
BASELINE_BRANCH=${PERFGATE_BASELINE_BRANCH:-benchmark-baselines}
OUTPUT_DIR=${PERFGATE_BASELINE_OUTPUT_DIR:-${RUNNER_TEMP:-/tmp}/perfgate-baselines}
MODE=${PERFGATE_MODE:-report}
GITHUB_ENV=${GITHUB_ENV:-/dev/null}
GIT_NETWORK_ATTEMPTS=${GIT_NETWORK_ATTEMPTS:-3}
GIT_NETWORK_TIMEOUT_SECONDS=${GIT_NETWORK_TIMEOUT_SECONDS:-90}
GIT_NETWORK_RETRY_DELAY_SECONDS=${GIT_NETWORK_RETRY_DELAY_SECONDS:-10}
TARGET_REPOSITORY=${PERFGATE_TARGET_REPOSITORY:-${GITHUB_REPOSITORY:-vLLM-HUST/vllm-hust}}
SCENARIO=${PERFGATE_SCENARIO:-${BENCH_SCENARIO:-random-online}}
BENCHMARK_REPO_DIR=${VLLM_HUST_BENCHMARK_REPO:-${GITHUB_WORKSPACE:-$PWD}/vllm-hust-benchmark}

write_env() {
  local name=$1
  local value=$2
  local delimiter="EOF_${name}_$$_${RANDOM}"
  {
    echo "${name}<<${delimiter}"
    printf '%s\n' "$value"
    echo "$delimiter"
  } >> "$GITHUB_ENV"
}

baseline_unavailable() {
  local reason=$1
  echo "$reason" >&2
  if [[ "$MODE" == "report" ]]; then
    write_env PERFGATE_BASELINE_AVAILABLE 0
    write_env PERFGATE_BASELINE_COMMIT "$COMMIT"
    write_env PERFGATE_BASELINE_SOURCE unavailable
    write_env PERFGATE_BASELINE_UNAVAILABLE_REASON "$reason"
    exit 0
  fi
  exit 2
}

run_git_network() {
  local attempt=1

  while [[ "$attempt" -le "$GIT_NETWORK_ATTEMPTS" ]]; do
    if command -v timeout >/dev/null 2>&1; then
      if timeout --foreground "${GIT_NETWORK_TIMEOUT_SECONDS}s" \
        git -c http.version=HTTP/1.1 \
        -c http.lowSpeedLimit=1024 \
        -c http.lowSpeedTime=30 "$@"; then
        return 0
      fi
    elif git -c http.version=HTTP/1.1 \
      -c http.lowSpeedLimit=1024 \
      -c http.lowSpeedTime=30 "$@"; then
      return 0
    fi

    if [[ "$attempt" -lt "$GIT_NETWORK_ATTEMPTS" ]]; then
      echo "Git network command failed ($attempt/$GIT_NETWORK_ATTEMPTS); retrying in ${GIT_NETWORK_RETRY_DELAY_SECONDS}s." >&2
      sleep "$GIT_NETWORK_RETRY_DELAY_SECONDS"
    fi
    attempt=$((attempt + 1))
  done

  return 1
}

if [[ -z "$COMMIT" ]]; then
  echo "Usage: $0 <commit-sha> or set FORK_POINT/GITHUB_SHA" >&2
  exit 2
fi
if [[ "${PERFGATE_ALLOW_BASELINE_FALLBACK:-0}" == "1" ]]; then
  echo "Central perfgate consumer does not allow latest-main fallback" >&2
  exit 2
fi
if [[ -z "${SAME_SPEC_SPEC_FILE:-}" || ! -f "$SAME_SPEC_SPEC_FILE" ]]; then
  baseline_unavailable "Resolved perfgate spec file is required before exact baseline fetch"
fi
if [[ ! -d "$BENCHMARK_REPO_DIR/src/vllm_hust_benchmark" ]]; then
  baseline_unavailable "Trusted benchmark runner checkout not found: $BENCHMARK_REPO_DIR"
fi

SPEC_ID=$(jq -er '.id' "$SAME_SPEC_SPEC_FILE")
owner=${TARGET_REPOSITORY%%/*}
repository=${TARGET_REPOSITORY#*/}
rm -rf "$OUTPUT_DIR/branch"
mkdir -p "$OUTPUT_DIR"

if ! run_git_network ls-remote --exit-code --heads "$CENTRAL_REPO_URL" "$BASELINE_BRANCH" >/dev/null 2>&1; then
  baseline_unavailable "Central perfgate baseline branch not found: $BASELINE_BRANCH"
fi
if ! run_git_network clone --depth 1 --single-branch --branch "$BASELINE_BRANCH" \
  "$CENTRAL_REPO_URL" "$OUTPUT_DIR/branch"; then
  baseline_unavailable "Unable to clone perfgate baseline branch: $BASELINE_BRANCH"
fi

identity_root="$OUTPUT_DIR/branch/baselines/$owner/$repository/$COMMIT/$SCENARIO/$SPEC_ID"
if [[ ! -d "$identity_root" ]]; then
  baseline_unavailable "No exact central perfgate baseline found under $identity_root"
fi

manifests=()
while IFS= read -r manifest; do
  manifests+=("$manifest")
done < <(find "$identity_root" -mindepth 2 -maxdepth 2 -name baseline-metadata.json -type f | sort)
if [[ "${#manifests[@]}" -ne 1 ]]; then
  baseline_unavailable "Expected exactly one exact spec hash under $identity_root, found ${#manifests[@]}"
fi

manifest=${manifests[0]}
baseline_dir=$(dirname "$manifest")
baseline_file="$baseline_dir/run_leaderboard.json"
if [[ ! -f "$baseline_file" ]]; then
  baseline_unavailable "Exact central baseline artifact is missing: $baseline_file"
fi

read_manifest() {
  jq -er "$1" "$manifest"
}

SPEC_HASH=$(read_manifest '.identity.spec_hash')
VLLM_HUST_SHA=$(read_manifest '.provenance.vllm_hust_sha')
VLLM_ASCEND_HUST_SHA=$(read_manifest '.provenance.vllm_ascend_hust_sha')
BENCHMARK_RUNNER_SHA=$(read_manifest '.provenance.benchmark_runner_sha')
RUNTIME_MANAGER_SHA=$(read_manifest '.provenance.runtime_manager_sha')
HARDWARE_CHIP_MODEL=$(read_manifest '.provenance.hardware_chip_model')
CANN_VERSION=$(read_manifest '.provenance.cann_version')
TORCH_VERSION=$(read_manifest '.provenance.torch_version')
TORCH_NPU_VERSION=$(read_manifest '.provenance.torch_npu_version')

PYTHONPATH="$BENCHMARK_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  "${PYTHON_BIN:-python3}" -m vllm_hust_benchmark.perfgate_baselines validate \
  --repository-root "$OUTPUT_DIR/branch" \
  --target-repository "$TARGET_REPOSITORY" \
  --target-sha "$COMMIT" \
  --scenario "$SCENARIO" \
  --spec-id "$SPEC_ID" \
  --spec-hash "$SPEC_HASH" \
  --vllm-hust-sha "$VLLM_HUST_SHA" \
  --vllm-ascend-hust-sha "$VLLM_ASCEND_HUST_SHA" \
  --benchmark-runner-sha "$BENCHMARK_RUNNER_SHA" \
  --runtime-manager-sha "$RUNTIME_MANAGER_SHA" \
  --hardware-chip-model "$HARDWARE_CHIP_MODEL" \
  --cann-version "$CANN_VERSION" \
  --torch-version "$TORCH_VERSION" \
  --torch-npu-version "$TORCH_NPU_VERSION"

resolved_file="$OUTPUT_DIR/baseline-${COMMIT:0:8}.json"
cp "$baseline_file" "$resolved_file"
write_env PERFGATE_BASELINE_FILE "$resolved_file"
write_env PERFGATE_BASELINE_AVAILABLE 1
write_env PERFGATE_BASELINE_COMMIT "$COMMIT"
write_env PERFGATE_BASELINE_SOURCE central-exact
write_env PERFGATE_BASELINE_SPEC_HASH "$SPEC_HASH"
write_env PERFGATE_BASELINE_METADATA_FILE "$manifest"
write_env PERFGATE_BASELINE_VLLM_HUST_SHA "$VLLM_HUST_SHA"
write_env PERFGATE_BASELINE_VLLM_ASCEND_HUST_SHA "$VLLM_ASCEND_HUST_SHA"
write_env PERFGATE_BASELINE_RUNNER_SHA "$BENCHMARK_RUNNER_SHA"

echo "Fetched exact central perfgate baseline: $TARGET_REPOSITORY@$COMMIT -> $resolved_file"
