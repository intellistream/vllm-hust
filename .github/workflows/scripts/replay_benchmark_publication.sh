#!/bin/bash
set -euo pipefail

mode=${1:-}
case "$mode" in
  preflight|publish) ;;
  *)
    echo "usage: $0 <preflight|publish>" >&2
    exit 2
    ;;
esac

require_env() {
  local name=$1
  if [[ -z "${!name:-}" ]]; then
    echo "$name is required" >&2
    exit 2
  fi
}

for name in \
  SOURCE_RUN_ID SOURCE_RUN_ATTEMPT EXPECTED_TARGET_SHA \
  EXPECTED_BENCHMARK_MAIN_SHA REPLAY_CONFIRMATION REPLAY_ARTIFACT_ROOT \
  BENCHMARK_REPO_DIR WEBSITE_REPO_DIR VLLM_HUST_REPO_DIR \
  EXPECTED_WEBSITE_SHA EXPECTED_SYNC_SCRIPT_SHA256; do
  require_env "$name"
done

REPLAY_FETCH_MAX_ATTEMPTS=${REPLAY_FETCH_MAX_ATTEMPTS:-4}
REPLAY_FETCH_RETRY_SECONDS=${REPLAY_FETCH_RETRY_SECONDS:-5}
case "$REPLAY_FETCH_MAX_ATTEMPTS" in
  ''|*[!0-9]*|0*)
    echo "REPLAY_FETCH_MAX_ATTEMPTS must be a positive integer" >&2
    exit 2
    ;;
esac
case "$REPLAY_FETCH_RETRY_SECONDS" in
  ''|*[!0-9]*)
    echo "REPLAY_FETCH_RETRY_SECONDS must be a non-negative integer" >&2
    exit 2
    ;;
esac

if [[ "$mode" == "publish" ]]; then
  require_env REPLAY_WRITER_TOKEN
  require_env REPLAY_RECEIPT_FILE
fi

case "$SOURCE_RUN_ID" in
  ''|*[!0-9]*) echo "SOURCE_RUN_ID must contain only digits" >&2; exit 2 ;;
esac
case "$SOURCE_RUN_ATTEMPT" in
  ''|*[!0-9]*|0*) echo "SOURCE_RUN_ATTEMPT must be a positive integer" >&2; exit 2 ;;
esac
for value_name in EXPECTED_TARGET_SHA EXPECTED_BENCHMARK_MAIN_SHA EXPECTED_WEBSITE_SHA; do
  value=${!value_name}
  if [[ ! "$value" =~ ^[0-9a-f]{40}$ ]]; then
    echo "$value_name must be a full lowercase Git SHA" >&2
    exit 2
  fi
done
if [[ ! "$EXPECTED_SYNC_SCRIPT_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "EXPECTED_SYNC_SCRIPT_SHA256 must be a lowercase SHA-256" >&2
  exit 2
fi

expected_confirmation="publish-${SOURCE_RUN_ID}-${SOURCE_RUN_ATTEMPT}"
if [[ "$REPLAY_CONFIRMATION" != "$expected_confirmation" ]]; then
  echo "REPLAY_CONFIRMATION must equal $expected_confirmation" >&2
  exit 2
fi

submission_run_id="ci-${SOURCE_RUN_ID}-${SOURCE_RUN_ATTEMPT}-${EXPECTED_TARGET_SHA}"
relative_submission_dir="submissions/$submission_run_id"
artifact_name="ascend-benchmark-${SOURCE_RUN_ID}-${SOURCE_RUN_ATTEMPT}"
sync_script="$VLLM_HUST_REPO_DIR/.github/workflows/scripts/sync_benchmark_snapshots_to_github.sh"
required_submission_files=(
  leaderboard_manifest.json
  run_leaderboard.json
  env-manifest.json
  pip-packages.json
  checksums.sha256
  STATUS
)
required_snapshot_files=(
  leaderboard_single.json
  leaderboard_multi.json
  leaderboard_compare.json
  last_updated.json
)

if [[ ! -x "$sync_script" ]]; then
  echo "publication sync script is missing or not executable: $sync_script" >&2
  exit 2
fi
actual_sync_sha256=$(sha256sum "$sync_script" | awk '{print $1}')
if [[ "$actual_sync_sha256" != "$EXPECTED_SYNC_SCRIPT_SHA256" ]]; then
  echo "publication sync script digest mismatch" >&2
  echo "expected=$EXPECTED_SYNC_SCRIPT_SHA256" >&2
  echo "actual=$actual_sync_sha256" >&2
  exit 2
fi

if [[ ! -d "$BENCHMARK_REPO_DIR/.git" ]]; then
  echo "benchmark checkout is missing: $BENCHMARK_REPO_DIR" >&2
  exit 2
fi
if [[ ! -f "$WEBSITE_REPO_DIR/scripts/aggregate_results.py" ]]; then
  echo "website aggregator is missing: $WEBSITE_REPO_DIR" >&2
  exit 2
fi
website_sha=$(git -C "$WEBSITE_REPO_DIR" rev-parse HEAD)
if [[ "$website_sha" != "$EXPECTED_WEBSITE_SHA" ]]; then
  echo "website checkout mismatch: expected $EXPECTED_WEBSITE_SHA, got $website_sha" >&2
  exit 2
fi

if [[ -n "$(git -C "$BENCHMARK_REPO_DIR" status --porcelain)" ]]; then
  echo "benchmark checkout must be clean before replay" >&2
  exit 2
fi
remote_url=$(git -C "$BENCHMARK_REPO_DIR" remote get-url origin)
if [[ "${ALLOW_LOCAL_REPLAY_REMOTE:-0}" != "1" ]]; then
  case "$remote_url" in
    https://github.com/vLLM-HUST/vllm-hust-benchmark|https://github.com/vLLM-HUST/vllm-hust-benchmark.git) ;;
    *) echo "unexpected benchmark origin: $remote_url" >&2; exit 2 ;;
  esac
fi
fetch_target_branch_with_retry() {
  local phase=$1
  local attempt=1
  while (( attempt <= REPLAY_FETCH_MAX_ATTEMPTS )); do
    if [[ -n "${askpass_script:-}" ]]; then
      if env GIT_ASKPASS="$askpass_script" GIT_TERMINAL_PROMPT=0 \
        git -C "$BENCHMARK_REPO_DIR" fetch origin main; then
        return 0
      fi
    elif git -C "$BENCHMARK_REPO_DIR" fetch origin main; then
      return 0
    fi
    if (( attempt == REPLAY_FETCH_MAX_ATTEMPTS )); then
      echo "replay ${phase} fetch failed after ${REPLAY_FETCH_MAX_ATTEMPTS} attempts" >&2
      return 1
    fi
    echo "replay ${phase} fetch failed; retrying in ${REPLAY_FETCH_RETRY_SECONDS}s (attempt $attempt/$REPLAY_FETCH_MAX_ATTEMPTS)" >&2
    sleep "$REPLAY_FETCH_RETRY_SECONDS"
    attempt=$((attempt + 1))
  done
}

fetch_target_branch_with_retry preflight
benchmark_head=$(git -C "$BENCHMARK_REPO_DIR" rev-parse HEAD)
benchmark_remote_main=$(git -C "$BENCHMARK_REPO_DIR" rev-parse origin/main)
if [[ "$benchmark_head" != "$EXPECTED_BENCHMARK_MAIN_SHA" \
  || "$benchmark_remote_main" != "$EXPECTED_BENCHMARK_MAIN_SHA" ]]; then
  echo "benchmark main moved; repeat the credential-free rehearsal" >&2
  echo "expected=$EXPECTED_BENCHMARK_MAIN_SHA" >&2
  echo "head=$benchmark_head" >&2
  echo "origin/main=$benchmark_remote_main" >&2
  exit 2
fi

submission_candidates=()
while IFS= read -r candidate; do
  submission_candidates+=("$candidate")
done < <(find "$REPLAY_ARTIFACT_ROOT" -type d -path "*/submissions/$submission_run_id" -print)
if [[ "${#submission_candidates[@]}" -ne 1 ]]; then
  echo "expected exactly one $relative_submission_dir in $artifact_name; found ${#submission_candidates[@]}" >&2
  exit 2
fi
submission_dir=${submission_candidates[0]}

for file_name in "${required_submission_files[@]}"; do
  file_path="$submission_dir/$file_name"
  if [[ ! -f "$file_path" || -L "$file_path" ]]; then
    echo "required replay evidence must be a regular file: $file_path" >&2
    exit 2
  fi
done
python3 - "$submission_dir" "$EXPECTED_TARGET_SHA" <<'PY'
import json
import re
import sys
from pathlib import Path

submission = Path(sys.argv[1])
expected_sha = sys.argv[2]
env_manifest = json.loads((submission / "env-manifest.json").read_text())
leaderboard = json.loads((submission / "run_leaderboard.json").read_text())

git_info = env_manifest.get("git_info", {}).get("vllm_hust", {})
metadata = leaderboard.get("metadata", {})
runtime_engine = metadata.get("runtime_provenance", {}).get("engine", {})
expected_checksum_files = {
    "leaderboard_manifest.json",
    "run_leaderboard.json",
    "env-manifest.json",
    "pip-packages.json",
}
checksum_files = set()
checksum_pattern = re.compile(r"^[0-9a-f]{64}  \./([^/]+)$")
for line in (submission / "checksums.sha256").read_text().splitlines():
    match = checksum_pattern.fullmatch(line)
    if match is None or match.group(1) not in expected_checksum_files:
        raise SystemExit(f"unsafe checksum manifest entry: {line!r}")
    if match.group(1) in checksum_files:
        raise SystemExit(f"duplicate checksum manifest entry: {match.group(1)}")
    checksum_files.add(match.group(1))
if checksum_files != expected_checksum_files:
    raise SystemExit("checksum manifest does not cover the exact evidence set")
checks = {
    "env declared SHA": git_info.get("declared"),
    "env observed SHA": git_info.get("observed"),
    "leaderboard Git SHA": metadata.get("git_commit"),
    "runtime engine SHA": runtime_engine.get("commit"),
}
for label, actual in checks.items():
    if actual != expected_sha:
        raise SystemExit(f"{label} mismatch: expected {expected_sha}, got {actual}")
if metadata.get("github_repository") != "vLLM-HUST/vllm-hust":
    raise SystemExit("leaderboard repository identity mismatch")
if runtime_engine.get("repository") != "vLLM-HUST/vllm-hust":
    raise SystemExit("runtime repository identity mismatch")
PY

if [[ "$(tr -d '[:space:]' < "$submission_dir/STATUS")" != "OK" ]]; then
  echo "source artifact STATUS is not OK" >&2
  exit 2
fi
(
  cd "$submission_dir"
  sha256sum -c checksums.sha256
)

echo "Replay preflight passed for $artifact_name/$relative_submission_dir"
if [[ "$mode" == "preflight" ]]; then
  exit 0
fi

mkdir -p "$(dirname "$REPLAY_RECEIPT_FILE")"
: > "$REPLAY_RECEIPT_FILE"
if [[ -n "${LOCAL_SNAPSHOT_OUTPUT_DIR:-}" ]]; then
  mkdir -p "$LOCAL_SNAPSHOT_OUTPUT_DIR"
fi

askpass_root=${RUNNER_TEMP:-/tmp}
mkdir -p "$askpass_root"
askpass_dir=$(mktemp -d "$askpass_root/benchmark-replay-askpass.XXXXXX")
cleanup_askpass() {
  rm -rf "$askpass_dir"
}
trap cleanup_askpass EXIT
askpass_script="$askpass_dir/askpass.sh"
askpass_token_file="$askpass_dir/token"
umask 077
writer_token="$REPLAY_WRITER_TOKEN"
printf '%s' "$writer_token" > "$askpass_token_file"
unset REPLAY_WRITER_TOKEN
cat > "$askpass_script" <<EOF
#!/bin/sh
case "\$1" in
  *Username*) printf '%s\n' x-access-token ;;
  *Password*) cat "$askpass_token_file" ;;
  *) exit 1 ;;
esac
EOF
chmod 700 "$askpass_script"

env -u GITHUB_ACTIONS -u REPLAY_WRITER_TOKEN \
  PYTHONPATH="$BENCHMARK_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  BENCHMARK_REPO_DIR="$BENCHMARK_REPO_DIR" \
  WEBSITE_REPO_DIR="$WEBSITE_REPO_DIR" \
  CURRENT_SUBMISSION_DIR="$submission_dir" \
  VLLM_HUST_REPO_DIR="$VLLM_HUST_REPO_DIR" \
  PYTHON_BIN="${PYTHON_BIN:-python3}" \
  RUN_ID="$submission_run_id" \
  SNAPSHOT_TARGET_BRANCH=main \
  SNAPSHOT_EXPECTED_BASE_SHA="$EXPECTED_BENCHMARK_MAIN_SHA" \
  BENCHMARK_REPO_REMOTE=origin \
  BENCHMARK_REPO_SLUG=vLLM-HUST/vllm-hust-benchmark \
  BENCHMARK_REPO_GH_TOKEN= \
  BENCHMARK_REPO_SSH_KEY= \
  ALLOW_LOCAL_GIT_RESET=1 \
  GITHUB_ENV="$REPLAY_RECEIPT_FILE" \
  LOCAL_SNAPSHOT_OUTPUT_DIR="${LOCAL_SNAPSHOT_OUTPUT_DIR:-}" \
  SNAPSHOT_MAX_FETCH_ATTEMPTS=4 \
  SNAPSHOT_FETCH_RETRY_SECONDS=5 \
  SNAPSHOT_MAX_PUSH_ATTEMPTS=1 \
  SNAPSHOT_PUSH_RETRY_SECONDS=0 \
  SNAPSHOT_COMMIT_MESSAGE="chore(data): replay benchmark publication $submission_run_id" \
  GIT_COMMITTER_NAME="vLLM-HUST Benchmark Bot" \
  GIT_COMMITTER_EMAIL=benchmark-bot@vllm-hust.local \
  GIT_ASKPASS="$askpass_script" \
  GIT_TERMINAL_PROMPT=0 \
  bash "$sync_script"

receipt_value() {
  local key=$1
  local values
  values=$(sed -n "s/^${key}=//p" "$REPLAY_RECEIPT_FILE")
  if [[ -z "$values" || "$(printf '%s\n' "$values" | wc -l | tr -d ' ')" -ne 1 ]]; then
    echo "replay receipt must contain exactly one $key" >&2
    exit 2
  fi
  printf '%s\n' "$values"
}

receipt_sync_status() {
  local values
  local value_count
  local initial_status
  local terminal_status

  values=$(sed -n 's/^GITHUB_SNAPSHOT_SYNC_STATUS=//p' "$REPLAY_RECEIPT_FILE")
  value_count=$(printf '%s\n' "$values" | wc -l | tr -d ' ')
  initial_status=$(printf '%s\n' "$values" | sed -n '1p')
  terminal_status=$(printf '%s\n' "$values" | sed -n '2p')
  if [[ "$value_count" -ne 2 || "$initial_status" != "attempting" ]]; then
    echo "replay receipt must contain attempting followed by one terminal sync status" >&2
    exit 2
  fi
  case "$terminal_status" in
    pushed|unchanged) printf '%s\n' "$terminal_status" ;;
    *)
      echo "unexpected terminal replay sync status: $terminal_status" >&2
      exit 2
      ;;
  esac
}

sync_status=$(receipt_sync_status)
sync_verification=$(receipt_value GITHUB_SNAPSHOT_SYNC_VERIFICATION)
verified_commit=$(receipt_value GITHUB_SNAPSHOT_SYNC_VERIFIED_COMMIT)
if [[ "$sync_verification" != "verified" ]]; then
  echo "replay verification did not succeed: $sync_verification" >&2
  exit 2
fi
if [[ ! "$verified_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "invalid verified replay commit: $verified_commit" >&2
  exit 2
fi

fetch_target_branch_with_retry post-publish
remote_commit=$(git -C "$BENCHMARK_REPO_DIR" rev-parse origin/main)
if [[ "$remote_commit" != "$verified_commit" ]]; then
  echo "remote replay commit mismatch" >&2
  exit 2
fi
post_publish_remote_url=$(git -C "$BENCHMARK_REPO_DIR" remote get-url origin)
if [[ "$post_publish_remote_url" != "$remote_url" ]]; then
  echo "benchmark origin changed during replay" >&2
  exit 2
fi
if [[ "$sync_status" == "pushed" ]]; then
  pushed_commit=$(receipt_value GITHUB_SNAPSHOT_SYNC_COMMIT)
  if [[ "$pushed_commit" != "$verified_commit" ]]; then
    echo "pushed and verified replay commits differ" >&2
    exit 2
  fi
  parent_commit=$(git -C "$BENCHMARK_REPO_DIR" rev-parse "$verified_commit^")
  if [[ "$parent_commit" != "$EXPECTED_BENCHMARK_MAIN_SHA" ]]; then
    echo "replay commit parent is not the approved benchmark main" >&2
    exit 2
  fi
fi

for file_name in "${required_submission_files[@]}"; do
  git -C "$BENCHMARK_REPO_DIR" show \
    "$verified_commit:$relative_submission_dir/$file_name" \
    | cmp - "$submission_dir/$file_name"
done
for file_name in "${required_snapshot_files[@]}"; do
  git -C "$BENCHMARK_REPO_DIR" cat-file -e \
    "$verified_commit:leaderboard-data/snapshots/$file_name"
done

while IFS= read -r changed_path; do
  case "$changed_path" in
    "$relative_submission_dir"/*) ;;
    leaderboard-data/snapshots/leaderboard_single.json) ;;
    leaderboard-data/snapshots/leaderboard_multi.json) ;;
    leaderboard-data/snapshots/leaderboard_compare.json) ;;
    leaderboard-data/snapshots/last_updated.json) ;;
    *) echo "unexpected replay path: $changed_path" >&2; exit 2 ;;
  esac
done < <(git -C "$BENCHMARK_REPO_DIR" diff --name-only \
  "$EXPECTED_BENCHMARK_MAIN_SHA..$verified_commit")

if git -C "$BENCHMARK_REPO_DIR" cat-file -e "$EXPECTED_BENCHMARK_MAIN_SHA:archive" 2>/dev/null; then
  before_archive=$(git -C "$BENCHMARK_REPO_DIR" rev-parse "$EXPECTED_BENCHMARK_MAIN_SHA:archive")
  after_archive=$(git -C "$BENCHMARK_REPO_DIR" rev-parse "$verified_commit:archive")
  if [[ "$before_archive" != "$after_archive" ]]; then
    echo "archive tree changed during replay" >&2
    exit 2
  fi
fi

git -C "$BENCHMARK_REPO_DIR" diff --check \
  "$EXPECTED_BENCHMARK_MAIN_SHA..$verified_commit"
(
  cd "$BENCHMARK_REPO_DIR/$relative_submission_dir"
  sha256sum -c checksums.sha256
)
post_validation_python=${POST_VALIDATION_PYTHON_BIN:-python3}
PYTHONPATH="$BENCHMARK_REPO_DIR/src" "$post_validation_python" \
  "$BENCHMARK_REPO_DIR/scripts/verify_submission_checksums.py" \
  --root "$BENCHMARK_REPO_DIR/submissions"
PYTHONPATH="$BENCHMARK_REPO_DIR/src" "$post_validation_python" \
  "$BENCHMARK_REPO_DIR/scripts/validate_public_leaderboard_snapshots.py" \
  --snapshot-dir "$BENCHMARK_REPO_DIR/leaderboard-data/snapshots"
PYTHONPATH="$BENCHMARK_REPO_DIR/src" "$post_validation_python" - \
  "$BENCHMARK_REPO_DIR/leaderboard-data/snapshots" <<'PY'
import sys
from pathlib import Path

from vllm_hust_benchmark.integration import validate_public_snapshot_trend_admission

validate_public_snapshot_trend_admission(Path(sys.argv[1]))
PY

{
  echo "REPLAY_RESULT=verified"
  echo "REPLAY_SOURCE_RUN_ID=$SOURCE_RUN_ID"
  echo "REPLAY_SOURCE_RUN_ATTEMPT=$SOURCE_RUN_ATTEMPT"
  echo "REPLAY_SOURCE_ARTIFACT=$artifact_name"
  echo "REPLAY_EXPECTED_TARGET_SHA=$EXPECTED_TARGET_SHA"
  echo "REPLAY_APPROVED_BENCHMARK_BASE=$EXPECTED_BENCHMARK_MAIN_SHA"
} >> "$REPLAY_RECEIPT_FILE"

echo "Historical benchmark publication replay verified at $verified_commit"
