#!/bin/bash
set -euo pipefail

if [[ "${PERFGATE_REQUIRED:-0}" != "1" ]]; then
  echo "Required perfgate completion validation is disabled."
  exit 0
fi

failures=()

require_value() {
  local name=$1
  local expected=$2
  local actual=${!name:-}
  if [[ "$actual" != "$expected" ]]; then
    failures+=("$name expected '$expected', got '${actual:-unset}'")
  fi
}

require_file() {
  local name=$1
  local path=${!name:-}
  if [[ -z "$path" || ! -f "$path" ]]; then
    failures+=("$name does not reference an existing file: '${path:-unset}'")
  fi
}

require_value PERFGATE_MODE enforce
require_value BENCH_SCENARIO_COUNT 1
require_value BENCH_SCENARIO random-online
require_value PERFGATE_BASELINE_AVAILABLE 1
require_value PERFGATE_STAGE1_COMPLETED 1
require_value PERFGATE_STAGE1_RESULT pass
require_value PERFGATE_STAGE2_EXECUTED 1
require_value PERFGATE_STAGE2_BASELINE_AVAILABLE 1
require_value PERFGATE_STAGE2_COMPLETED 1
require_value PERFGATE_STAGE2_RESULT pass
require_value PERFGATE_STAGE2_SKIPPED 0
require_value PERFGATE_STAGE2_REBASE_CONFLICT 0
require_value PERFGATE_RESULT pass
require_file PERFGATE_BASELINE_FILE
require_file PERFGATE_STAGE2_B1PRIME_FILE
require_file PERFGATE_STAGE2_M2_BASELINE_FILE
require_file PERFGATE_REPORT_FILE

if (( ${#failures[@]} > 0 )); then
  echo "Required two-stage performance gate is incomplete or failed:" >&2
  printf '  - %s\n' "${failures[@]}" >&2
  exit 2
fi

echo "Required two-stage performance gate completed successfully."
