#!/bin/bash
set -euo pipefail

RUN_ID=${RUN_ID:-ci-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}-${GITHUB_SHA:-local}}
RESULT_ROOT=${RESULT_ROOT:-${GITHUB_WORKSPACE:-$PWD}/.benchmarks/ci/$RUN_ID}
TARGET_REPOSITORY=${PERFGATE_TARGET_REPOSITORY:-${GITHUB_REPOSITORY:-vLLM-HUST/vllm-hust}}
TARGET_SHA=${PERFGATE_TARGET_SHA:-${GITHUB_SHA:-$(git rev-parse HEAD)}}
TARGET_GIT_REPOSITORY=${PERFGATE_TARGET_GIT_REPOSITORY:-${GITHUB_WORKSPACE:-$PWD}}
CENTRAL_REPO_URL=${PERFGATE_CENTRAL_REPO_URL:-https://github.com/vLLM-HUST/vllm-hust-benchmark.git}
CENTRAL_BRANCH=${PERFGATE_BASELINE_BRANCH:-benchmark-baselines}
BENCHMARK_REPO_DIR=${PERFGATE_BENCHMARK_REPO_DIR:-${GITHUB_WORKSPACE:-$PWD}/vllm-hust-benchmark}
BASELINE_FILE=${PERFGATE_BASELINE_SOURCE_FILE:-$RESULT_ROOT/submissions/$RUN_ID/run_leaderboard.json}
MEASUREMENT_FILE=${PERFGATE_MEASUREMENT_FILE:-$(dirname "$BASELINE_FILE")/measurement.json}
PROVENANCE_FILE=${PERFGATE_PROVENANCE_FILE:-$(dirname "$BASELINE_FILE")/perfgate-provenance.json}
EXPECTED_WARMUP_RUNS=${PERFGATE_EXPECTED_WARMUP_RUNS:-1}
EXPECTED_MEASURED_RUNS=${PERFGATE_EXPECTED_MEASURED_RUNS:-3}
EXPECTED_SCHEMA_VERSION=${PERFGATE_EXPECTED_SCHEMA_VERSION:-perfgate-measurement/v2}
EXPECTED_STRATEGY=${PERFGATE_EXPECTED_STRATEGY:-warmup+primary-median-run}
EXPECTED_AGGREGATION=${PERFGATE_EXPECTED_AGGREGATION:-primary-median-run}
WRITER_TOKEN=${PERFGATE_BASELINE_WRITER_TOKEN:-}

for tool in git jq awk mktemp sha256sum; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Required perfgate publication tool is unavailable: $tool" >&2
    exit 2
  fi
done

for file in "$BASELINE_FILE" "$MEASUREMENT_FILE" "$PROVENANCE_FILE"; do
  if [[ ! -f "$file" ]]; then
    echo "Required perfgate publication artifact not found: $file" >&2
    exit 2
  fi
done
if [[ ! -d "$BENCHMARK_REPO_DIR/src/vllm_hust_benchmark" ]]; then
  echo "Trusted benchmark runner checkout not found: $BENCHMARK_REPO_DIR" >&2
  exit 2
fi
if [[ -z "$WRITER_TOKEN" ]]; then
  echo "PERFGATE_BASELINE_WRITER_TOKEN is required for central publication" >&2
  exit 2
fi

jq -e \
  --argjson warmup "$EXPECTED_WARMUP_RUNS" \
  --argjson measured "$EXPECTED_MEASURED_RUNS" \
  --arg schema "$EXPECTED_SCHEMA_VERSION" \
  --arg strategy "$EXPECTED_STRATEGY" \
  --arg aggregation "$EXPECTED_AGGREGATION" \
  '.schema_version == $schema and
   .strategy == $strategy and
   .warmup_runs == $warmup and
   .measured_runs == $measured and
   .aggregation == $aggregation and
   (.warmup | type == "array" and length == $warmup) and
   (.per_run | type == "array" and length == $measured) and
   (.selection | type == "object") and
   .selection.primary_metric == "throughput_tps" and
   .selection.sort_direction == "ascending" and
   .selection.secondary_sort_key == "run_index" and
   (.selection.ordered_run_indices | type == "array" and length == $measured) and
   .selection.selected_position == (($measured + 1) / 2)' \
  "$MEASUREMENT_FILE" >/dev/null

verify_raw_result_evidence() {
  local evidence_key=$1
  local count=$2
  local index
  local raw_result
  local expected_sha
  local actual_sha

  for ((index = 1; index <= count; index++)); do
    if [[ "$evidence_key" == "warmup" ]]; then
      raw_result="$RESULT_ROOT/runs/warmup-$index/raw_benchmark_result.json"
    else
      raw_result="$RESULT_ROOT/runs/$index/raw_benchmark_result.json"
    fi
    if [[ ! -f "$raw_result" ]]; then
      echo "Measurement source evidence is missing: $raw_result" >&2
      return 2
    fi
    expected_sha=$(jq -er \
      --arg key "$evidence_key" \
      --argjson index "$index" \
      '.[$key][] | select(.run_index == $index) | .raw_result_sha256' \
      "$MEASUREMENT_FILE")
    actual_sha=$(sha256sum "$raw_result" | awk '{print $1}')
    if [[ "$actual_sha" != "$expected_sha" ]]; then
      echo "Measurement source checksum mismatch: $raw_result" >&2
      return 2
    fi
  done
}

verify_raw_result_evidence warmup "$EXPECTED_WARMUP_RUNS"
verify_raw_result_evidence per_run "$EXPECTED_MEASURED_RUNS"

read_json() {
  jq -er "$1" "$2"
}

SCENARIO=$(read_json '.same_spec.scenario' "$BASELINE_FILE")
SPEC_ID=$(read_json '.same_spec.spec_id' "$BASELINE_FILE")
SPEC_HASH=$(read_json '.same_spec.resolved_spec_hash' "$BASELINE_FILE")
ARTIFACT_REPOSITORY=$(read_json '.metadata.github_repository' "$BASELINE_FILE")
ARTIFACT_SHA=$(read_json '.metadata.git_commit' "$BASELINE_FILE")
VLLM_HUST_SHA=$(read_json '.vllm_hust_sha' "$PROVENANCE_FILE")
VLLM_ASCEND_HUST_SHA=$(read_json '.vllm_ascend_hust_sha' "$PROVENANCE_FILE")
BENCHMARK_RUNNER_SHA=$(read_json '.benchmark_runner_sha' "$PROVENANCE_FILE")
RUNTIME_MANAGER_SHA=$(read_json '.runtime_manager_sha' "$PROVENANCE_FILE")
HARDWARE_CHIP_MODEL=$(read_json '.hardware_chip_model' "$PROVENANCE_FILE")
CANN_VERSION=$(read_json '.cann_version' "$PROVENANCE_FILE")
TORCH_VERSION=$(read_json '.torch_version' "$PROVENANCE_FILE")
TORCH_NPU_VERSION=$(read_json '.torch_npu_version' "$PROVENANCE_FILE")

if [[ "${ARTIFACT_REPOSITORY,,}" != "${TARGET_REPOSITORY,,}" ]]; then
  echo "Artifact repository mismatch: expected $TARGET_REPOSITORY, got $ARTIFACT_REPOSITORY" >&2
  exit 2
fi
if [[ "${ARTIFACT_SHA,,}" != "${TARGET_SHA,,}" ]]; then
  echo "Artifact SHA mismatch: expected $TARGET_SHA, got $ARTIFACT_SHA" >&2
  exit 2
fi

fetch_target_main_with_retry() {
  local max_attempts=${PERFGATE_TARGET_FETCH_MAX_ATTEMPTS:-4}
  local delay_seconds=${PERFGATE_TARGET_FETCH_INITIAL_DELAY_SECONDS:-15}
  local attempt=1

  if [[ ! "$max_attempts" =~ ^[1-9][0-9]*$ ]]; then
    echo "PERFGATE_TARGET_FETCH_MAX_ATTEMPTS must be a positive integer" >&2
    return 2
  fi
  if [[ ! "$delay_seconds" =~ ^[0-9]+$ ]]; then
    echo "PERFGATE_TARGET_FETCH_INITIAL_DELAY_SECONDS must be a non-negative integer" >&2
    return 2
  fi

  while [[ "$attempt" -le "$max_attempts" ]]; do
    echo "[perfgate-publisher] target main fetch attempt ${attempt}/${max_attempts}"
    if git -C "$TARGET_GIT_REPOSITORY" fetch --quiet \
      origin main:refs/remotes/origin/main; then
      return 0
    fi
    if [[ "$attempt" -lt "$max_attempts" ]]; then
      sleep "$delay_seconds"
    fi
    attempt=$((attempt + 1))
    delay_seconds=$((delay_seconds * 2))
  done

  echo "Failed to fetch target main after ${max_attempts} attempts" >&2
  return 1
}

fetch_target_main_with_retry
UPDATE_POINTER=()
if [[ "$(git -C "$TARGET_GIT_REPOSITORY" rev-parse origin/main)" == "$TARGET_SHA" ]]; then
  UPDATE_POINTER=(--update-latest-pointer)
else
  echo "Target main advanced; publishing exact baseline without updating latest pointer."
fi

ASKPASS_DIR=$(mktemp -d "${RUNNER_TEMP:-/tmp}/perfgate-writer-askpass.XXXXXX")
ASKPASS_FILE="$ASKPASS_DIR/askpass.sh"
cleanup() {
  local status=$?
  unset PERFGATE_BASELINE_WRITER_TOKEN WRITER_TOKEN GIT_ASKPASS GIT_TERMINAL_PROMPT
  rm -rf "$ASKPASS_DIR"
  exit "$status"
}
trap cleanup EXIT

cat >"$ASKPASS_FILE" <<'EOF'
#!/bin/sh
case "$1" in
  *Username*) printf '%s\n' 'x-access-token' ;;
  *Password*) printf '%s\n' "$PERFGATE_BASELINE_WRITER_TOKEN" ;;
  *) printf '\n' ;;
esac
EOF
chmod 700 "$ASKPASS_FILE"
export PERFGATE_BASELINE_WRITER_TOKEN="$WRITER_TOKEN"
export GIT_ASKPASS="$ASKPASS_FILE"
export GIT_TERMINAL_PROMPT=0

PYTHONPATH="$BENCHMARK_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  "${PYTHON_BIN:-python3}" -m vllm_hust_benchmark.perfgate_baselines publish \
  --remote "$CENTRAL_REPO_URL" \
  --branch "$CENTRAL_BRANCH" \
  --source "$BASELINE_FILE" \
  --measurement-file "$MEASUREMENT_FILE" \
  --target-git-repository "$TARGET_GIT_REPOSITORY" \
  --main-ref origin/main \
  --target-repository "$TARGET_REPOSITORY" \
  --target-sha "$TARGET_SHA" \
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
  --torch-npu-version "$TORCH_NPU_VERSION" \
  "${UPDATE_POINTER[@]}"
