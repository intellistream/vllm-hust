#!/bin/bash
set -euo pipefail

COMMIT=${1:-${FORK_POINT:-${GITHUB_SHA:-}}}
OUTPUT_DIR=${PERFGATE_BASELINE_OUTPUT_DIR:-${RUNNER_TEMP:-/tmp}/perfgate-baselines}
ALLOW_BASELINE_FALLBACK=${PERFGATE_ALLOW_BASELINE_FALLBACK:-0}
MODE=${PERFGATE_MODE:-report}
GITHUB_ENV=${GITHUB_ENV:-/dev/null}
PYTHON_BIN=${PYTHON_BIN:-python3}
CENTRAL_BASELINE_REMOTE=https://github.com/vLLM-HUST/vllm-hust-benchmark.git
CENTRAL_BASELINE_BRANCH=benchmark-baselines
CENTRAL_CHECKOUT=${PERFGATE_CENTRAL_REPOSITORY_ROOT:-$OUTPUT_DIR/central}
TARGET_REPOSITORY=${GITHUB_REPOSITORY:-vLLM-HUST/vllm-hust}
BENCHMARK_REPOSITORY=${VLLM_HUST_BENCHMARK_REPO:-${GITHUB_WORKSPACE:-$PWD}/vllm-hust-benchmark}
PLUGIN_REPOSITORY=${VLLM_ASCEND_HUST_REPO:-${GITHUB_WORKSPACE:-$PWD}/vllm-ascend-hust}
SPEC_FILE=${SAME_SPEC_SPEC_FILE:-}
CANN_VERSION=${HUST_ASCEND_RUNTIME_VERSION:-}

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
  write_env PERFGATE_BASELINE_AVAILABLE 0
  write_env PERFGATE_BASELINE_COMMIT "$COMMIT"
  write_env PERFGATE_BASELINE_SOURCE unavailable
  write_env PERFGATE_BASELINE_UNAVAILABLE_REASON "$reason"
  if [[ "$MODE" == "report" ]]; then
    echo "Central perfgate baseline unavailable in report mode; continuing without baseline."
    exit 0
  fi
  echo "Central perfgate baseline unavailable in enforce mode; failing."
  exit 2
}

if [[ -z "$COMMIT" ]]; then
  echo "Usage: $0 <commit-sha> or set FORK_POINT/GITHUB_SHA" >&2
  exit 2
fi
if [[ "$ALLOW_BASELINE_FALLBACK" != "0" ]]; then
  baseline_unavailable "Central required perfgate supports exact baselines only; fallback was requested."
fi
if [[ -z "$SPEC_FILE" || ! -f "$SPEC_FILE" ]]; then
  baseline_unavailable "Resolved perfgate spec file is unavailable: ${SPEC_FILE:-unset}"
fi
if [[ ! -d "$BENCHMARK_REPOSITORY/src/vllm_hust_benchmark" ]]; then
  baseline_unavailable "Pinned benchmark checkout is unavailable: $BENCHMARK_REPOSITORY"
fi
if [[ ! -d "$PLUGIN_REPOSITORY/.git" ]]; then
  baseline_unavailable "Pinned plugin checkout is unavailable: $PLUGIN_REPOSITORY"
fi
if [[ -z "$CANN_VERSION" ]]; then
  baseline_unavailable "HUST_ASCEND_RUNTIME_VERSION is required for central baseline provenance validation."
fi

mkdir -p "$OUTPUT_DIR"
if [[ -z "${PERFGATE_CENTRAL_REPOSITORY_ROOT:-}" ]]; then
  rm -rf "$CENTRAL_CHECKOUT"
  if ! git clone \
    --depth 1 \
    --single-branch \
    --no-tags \
    --branch "$CENTRAL_BASELINE_BRANCH" \
    "$CENTRAL_BASELINE_REMOTE" \
    "$CENTRAL_CHECKOUT"; then
    baseline_unavailable "Unable to clone central baseline branch: $CENTRAL_BASELINE_BRANCH"
  fi
fi

resolved_file="$OUTPUT_DIR/baseline-${COMMIT:0:8}.json"
if ! PYTHONPATH="$BENCHMARK_REPOSITORY/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" \
    .github/workflows/scripts/fetch_central_perfgate_baseline.py \
    --central-repository-root "$CENTRAL_CHECKOUT" \
    --output "$resolved_file" \
    --target-repository "$TARGET_REPOSITORY" \
    --target-sha "$COMMIT" \
    --scenario "${BENCH_SCENARIO:-random-online}" \
    --spec-file "$SPEC_FILE" \
    --benchmark-git-repository "$BENCHMARK_REPOSITORY" \
    --plugin-git-repository "$PLUGIN_REPOSITORY" \
    --hardware-chip-model "${HARDWARE_CHIP_MODEL:-}" \
    --cann-version "$CANN_VERSION" \
    --github-env "$GITHUB_ENV"; then
  baseline_unavailable "Central exact baseline validation failed for $TARGET_REPOSITORY@$COMMIT"
fi

echo "Fetched central exact perfgate baseline: $TARGET_REPOSITORY@$COMMIT -> $resolved_file"
