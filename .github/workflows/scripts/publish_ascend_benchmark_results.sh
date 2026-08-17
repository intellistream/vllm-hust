#!/bin/bash
set -euo pipefail

WORKSPACE_ROOT=${WORKSPACE_ROOT:-${GITHUB_WORKSPACE:-$PWD}}
VLLM_HUST_REPO=${VLLM_HUST_REPO:-$WORKSPACE_ROOT}
VLLM_HUST_BENCHMARK_REPO=${VLLM_HUST_BENCHMARK_REPO:-$WORKSPACE_ROOT/vllm-hust-benchmark}
VLLM_HUST_WEBSITE_REPO=${VLLM_HUST_WEBSITE_REPO:-$WORKSPACE_ROOT/vllm-hust-website}
RUN_ID=${RUN_ID:-ci-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}-${TARGET_REPO_SHA:-local}}
RESULT_ROOT=${RESULT_ROOT:-$VLLM_HUST_REPO/.benchmarks/ci/$RUN_ID}
PUBLISHER_SCRIPT=${BENCHMARK_PUBLICATION_SYNC_SCRIPT:-$VLLM_HUST_REPO/.github/workflows/scripts/sync_benchmark_snapshots_to_github.sh}

publish_submission() {
  local run_id=$1
  local result_root=$2
  local submission_dir=$3

  if [[ "$(tr -d '[:space:]' <"$submission_dir/STATUS" 2>/dev/null || true)" != "OK" ]]; then
    echo "refusing to publish submission without STATUS=OK: $submission_dir" >&2
    return 2
  fi

  BENCHMARK_REPO_DIR="$VLLM_HUST_BENCHMARK_REPO" \
  WEBSITE_REPO_DIR="$VLLM_HUST_WEBSITE_REPO" \
  CURRENT_SUBMISSION_DIR="$submission_dir" \
  CURRENT_SUBMISSIONS_DIR="" \
  VLLM_HUST_REPO_DIR="$VLLM_HUST_REPO" \
  LOCAL_SNAPSHOT_OUTPUT_DIR="$result_root/leaderboard-data" \
  PYTHON_BIN="${PYTHON_BIN:-python3}" \
  BENCHMARK_REPO_SLUG="${BENCHMARK_REPO_SLUG:-vLLM-HUST/vllm-hust-benchmark}" \
  BENCHMARK_REPO_GH_TOKEN="${BENCHMARK_REPO_GH_TOKEN:-}" \
  BENCHMARK_REPO_SSH_KEY="${BENCHMARK_REPO_SSH_KEY:-}" \
  SNAPSHOT_TARGET_BRANCH="${SNAPSHOT_TARGET_BRANCH:-main}" \
  SNAPSHOT_COMMIT_MESSAGE="chore(data): sync benchmark publication $run_id" \
  RUN_ID="$run_id" \
    "$PUBLISHER_SCRIPT"
}

publish_submissions() {
  local run_id=$1
  local result_root=$2
  local submissions_dir=$3

  BENCHMARK_REPO_DIR="$VLLM_HUST_BENCHMARK_REPO" \
  WEBSITE_REPO_DIR="$VLLM_HUST_WEBSITE_REPO" \
  CURRENT_SUBMISSION_DIR="" \
  CURRENT_SUBMISSIONS_DIR="$submissions_dir" \
  VLLM_HUST_REPO_DIR="$VLLM_HUST_REPO" \
  LOCAL_SNAPSHOT_OUTPUT_DIR="$result_root/leaderboard-data" \
  PYTHON_BIN="${PYTHON_BIN:-python3}" \
  BENCHMARK_REPO_SLUG="${BENCHMARK_REPO_SLUG:-vLLM-HUST/vllm-hust-benchmark}" \
  BENCHMARK_REPO_GH_TOKEN="${BENCHMARK_REPO_GH_TOKEN:-}" \
  BENCHMARK_REPO_SSH_KEY="${BENCHMARK_REPO_SSH_KEY:-}" \
  SNAPSHOT_TARGET_BRANCH="${SNAPSHOT_TARGET_BRANCH:-main}" \
  SNAPSHOT_COMMIT_MESSAGE="chore(data): sync benchmark publication $run_id" \
  RUN_ID="$run_id" \
    "$PUBLISHER_SCRIPT"
}

if [[ ! -x "$PUBLISHER_SCRIPT" ]]; then
  echo "benchmark publication sync script is missing or not executable: $PUBLISHER_SCRIPT" >&2
  exit 2
fi

summary_file=${BENCHMARK_MULTI_SCENARIO_SUMMARY_FILE:-$RESULT_ROOT/multi_scenario_results.tsv}
if [[ -f "$summary_file" ]]; then
  run_ids=()
  result_roots=()
  submission_dirs=()
  while IFS=$'\t' read -r scenario run_id result_root _raw_result submission_dir exit_code; do
    if [[ "$scenario" == "scenario" || -z "$scenario" ]]; then
      continue
    fi
    if [[ "$exit_code" != "0" ]]; then
      echo "refusing partial multi-scenario publication because $scenario exited $exit_code" >&2
      exit 2
    fi
    if [[ "$(tr -d '[:space:]' <"$submission_dir/STATUS" 2>/dev/null || true)" != "OK" ]]; then
      echo "refusing partial multi-scenario publication without STATUS=OK: $submission_dir" >&2
      exit 2
    fi
    run_ids+=("$run_id")
    result_roots+=("$result_root")
    submission_dirs+=("$submission_dir")
  done <"$summary_file"
  if [[ "${#run_ids[@]}" -eq 0 ]]; then
    echo "multi-scenario summary contains no publishable submissions: $summary_file" >&2
    exit 2
  fi
  aggregate_dir=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/benchmark-publication.XXXXXX")
  # shellcheck disable=SC2329
  cleanup_aggregate_dir() {
    rm -rf "$aggregate_dir"
  }
  trap cleanup_aggregate_dir EXIT
  for index in "${!run_ids[@]}"; do
    aggregate_submission_dir="$aggregate_dir/${run_ids[$index]}"
    mkdir -p "$aggregate_submission_dir"
    cp -a "${submission_dirs[$index]}/." "$aggregate_submission_dir/"
  done
  publish_submissions "${RUN_ID}-multi" "${RESULT_ROOT}" "$aggregate_dir"
  exit 0
fi

publish_submission "$RUN_ID" "$RESULT_ROOT" "${SUBMISSION_DIR:-$RESULT_ROOT/submissions/$RUN_ID}"
