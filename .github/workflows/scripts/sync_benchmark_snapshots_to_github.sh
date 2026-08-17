#!/bin/bash
set -euo pipefail

BENCHMARK_REPO_DIR=${BENCHMARK_REPO_DIR:?BENCHMARK_REPO_DIR is required}
WEBSITE_REPO_DIR=${WEBSITE_REPO_DIR:?WEBSITE_REPO_DIR is required}
CURRENT_SUBMISSION_DIR=${CURRENT_SUBMISSION_DIR:-}
CURRENT_SUBMISSIONS_DIR=${CURRENT_SUBMISSIONS_DIR:-}
VLLM_HUST_REPO_DIR=${VLLM_HUST_REPO_DIR:-${VLLM_HUST_REPO:-$BENCHMARK_REPO_DIR/../vllm-hust}}
PYTHON_BIN=${PYTHON_BIN:-python3}
SNAPSHOT_TARGET_BRANCH=${SNAPSHOT_TARGET_BRANCH:-main}
SNAPSHOT_EXPECTED_BASE_SHA=${SNAPSHOT_EXPECTED_BASE_SHA:-}
SNAPSHOT_OUTPUT_DIR=${SNAPSHOT_OUTPUT_DIR:-$BENCHMARK_REPO_DIR/leaderboard-data/snapshots}
LOCAL_SNAPSHOT_OUTPUT_DIR=${LOCAL_SNAPSHOT_OUTPUT_DIR:-}
SNAPSHOT_MAX_PUSH_ATTEMPTS=${SNAPSHOT_MAX_PUSH_ATTEMPTS:-4}
SNAPSHOT_PUSH_RETRY_SECONDS=${SNAPSHOT_PUSH_RETRY_SECONDS:-5}
SNAPSHOT_MAX_FETCH_ATTEMPTS=${SNAPSHOT_MAX_FETCH_ATTEMPTS:-4}
SNAPSHOT_FETCH_RETRY_SECONDS=${SNAPSHOT_FETCH_RETRY_SECONDS:-5}
SNAPSHOT_COMMIT_MESSAGE=${SNAPSHOT_COMMIT_MESSAGE:-chore(data): sync benchmark publication}
GIT_COMMITTER_NAME=${GIT_COMMITTER_NAME:-vLLM-HUST Benchmark Bot}
GIT_COMMITTER_EMAIL=${GIT_COMMITTER_EMAIL:-benchmark-bot@vllm-hust.local}
BENCHMARK_REPO_REMOTE=${BENCHMARK_REPO_REMOTE:-origin}
BENCHMARK_REPO_SLUG=${BENCHMARK_REPO_SLUG:-vLLM-HUST/vllm-hust-benchmark}
BENCHMARK_REPO_GH_TOKEN=${BENCHMARK_REPO_GH_TOKEN:-}
BENCHMARK_REPO_SSH_KEY=${BENCHMARK_REPO_SSH_KEY:-}
AUTH_DIR=
ORIGINAL_REMOTE_URL=

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

if [[ -n "$CURRENT_SUBMISSION_DIR" && -n "$CURRENT_SUBMISSIONS_DIR" ]]; then
  echo "set only one of CURRENT_SUBMISSION_DIR or CURRENT_SUBMISSIONS_DIR" >&2
  exit 2
fi
if [[ -z "$CURRENT_SUBMISSION_DIR" && -z "$CURRENT_SUBMISSIONS_DIR" ]]; then
  echo "CURRENT_SUBMISSION_DIR or CURRENT_SUBMISSIONS_DIR is required" >&2
  exit 2
fi

submission_source_dirs=()
submission_names=()
if [[ -n "$CURRENT_SUBMISSIONS_DIR" ]]; then
  if [[ ! -d "$CURRENT_SUBMISSIONS_DIR" ]]; then
    echo "current submissions directory not found: $CURRENT_SUBMISSIONS_DIR" >&2
    exit 2
  fi
  while IFS= read -r -d '' submission_dir; do
    submission_source_dirs+=("$submission_dir")
    submission_names+=("$(basename "$submission_dir")")
  done < <(find "$CURRENT_SUBMISSIONS_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
  if [[ "${#submission_source_dirs[@]}" -eq 0 ]]; then
    echo "current submissions directory is empty: $CURRENT_SUBMISSIONS_DIR" >&2
    exit 2
  fi
else
  submission_source_dirs=("$CURRENT_SUBMISSION_DIR")
  submission_names=("${RUN_ID:-$(basename "$CURRENT_SUBMISSION_DIR")}")
fi

write_github_env() {
  local key=$1
  local value=$2
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    printf '%s=%s\n' "$key" "$value" >>"$GITHUB_ENV"
  fi
}

validate_fetch_retry_configuration() {
  case "$SNAPSHOT_MAX_FETCH_ATTEMPTS" in
    ''|*[!0-9]*|0*)
      echo "SNAPSHOT_MAX_FETCH_ATTEMPTS must be a positive integer" >&2
      return 2
      ;;
  esac
  case "$SNAPSHOT_FETCH_RETRY_SECONDS" in
    ''|*[!0-9]*)
      echo "SNAPSHOT_FETCH_RETRY_SECONDS must be a non-negative integer" >&2
      return 2
      ;;
  esac
}

fetch_target_branch_with_retry() {
  local phase=$1
  local attempt=1

  while (( attempt <= SNAPSHOT_MAX_FETCH_ATTEMPTS )); do
    if git -C "$BENCHMARK_REPO_DIR" fetch \
      "$BENCHMARK_REPO_REMOTE" "$SNAPSHOT_TARGET_BRANCH"; then
      return 0
    fi
    if (( attempt == SNAPSHOT_MAX_FETCH_ATTEMPTS )); then
      echo "benchmark publication ${phase} fetch failed after ${SNAPSHOT_MAX_FETCH_ATTEMPTS} attempts" >&2
      return 1
    fi
    echo "benchmark publication ${phase} fetch failed; retrying ${BENCHMARK_REPO_REMOTE}/${SNAPSHOT_TARGET_BRANCH} in ${SNAPSHOT_FETCH_RETRY_SECONDS}s (attempt $attempt/$SNAPSHOT_MAX_FETCH_ATTEMPTS)" >&2
    sleep "$SNAPSHOT_FETCH_RETRY_SECONDS"
    attempt=$((attempt + 1))
  done
}

validate_fetch_retry_configuration
write_github_env GITHUB_SNAPSHOT_SYNC_STATUS attempting

if [[ -n "$SNAPSHOT_EXPECTED_BASE_SHA" \
  && ! "$SNAPSHOT_EXPECTED_BASE_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SNAPSHOT_EXPECTED_BASE_SHA must be a full lowercase Git SHA" >&2
  exit 2
fi

configure_push_remote() {
  local remote_url=

  ORIGINAL_REMOTE_URL=$(git -C "$BENCHMARK_REPO_DIR" remote get-url "$BENCHMARK_REPO_REMOTE")
  # The preflight can prove repository write permission for the token via the
  # GitHub API. Use that same credential for publication when both are set.
  if [[ -n "$BENCHMARK_REPO_GH_TOKEN" ]]; then
    AUTH_DIR=$(mktemp -d "${RUNNER_TEMP:-/tmp}/benchmark-writer.XXXXXX")
    local askpass_file="$AUTH_DIR/askpass.sh"
    cat >"$askpass_file" <<'EOF'
#!/bin/sh
case "$1" in
  *Username*) printf '%s\n' 'x-access-token' ;;
  *Password*) printf '%s\n' "$BENCHMARK_REPO_GH_TOKEN" ;;
  *) printf '\n' ;;
esac
EOF
    chmod 700 "$askpass_file"
    export GIT_ASKPASS="$askpass_file"
    export GIT_TERMINAL_PROMPT=0
    remote_url="https://github.com/${BENCHMARK_REPO_SLUG}.git"
    git -C "$BENCHMARK_REPO_DIR" remote set-url "$BENCHMARK_REPO_REMOTE" "$remote_url"
    return 0
  fi

  if [[ -n "$BENCHMARK_REPO_SSH_KEY" ]]; then
    AUTH_DIR=$(mktemp -d "${RUNNER_TEMP:-/tmp}/benchmark-writer.XXXXXX")
    local key_file="$AUTH_DIR/writer_key"
    local known_hosts_file="$AUTH_DIR/known_hosts"
    printf '%s\n' "$BENCHMARK_REPO_SSH_KEY" >"$key_file"
    chmod 600 "$key_file"
    printf '%s\n' \
      '[ssh.github.com]:443 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqKqKNSCX3yYaTuk8CSQb8eFXh8UwMphjMrJnnIy9i' \
      >"$known_hosts_file"
    export GIT_SSH_COMMAND="ssh -i $key_file -o IdentitiesOnly=yes -o UserKnownHostsFile=$known_hosts_file -o StrictHostKeyChecking=yes"
    remote_url="ssh://git@ssh.github.com:443/${BENCHMARK_REPO_SLUG}.git"
    git -C "$BENCHMARK_REPO_DIR" remote set-url "$BENCHMARK_REPO_REMOTE" "$remote_url"
    return 0
  fi

  if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    echo "L3 benchmark repository publication is enabled, but no cross-repository write credential is available." >&2
    echo "Configure one of the following secrets on the vllm-hust workflow repository before enabling benchmark repo publish:" >&2
    echo "  - VLLM_ASCEND_HUST_BENCHMARK_SSH_KEY: SSH private key with write access to ${BENCHMARK_REPO_SLUG}" >&2
    echo "  - VLLM_HUST_BENCHMARK_GH_TOKEN: GitHub token with contents write access to ${BENCHMARK_REPO_SLUG}" >&2
    echo "Benchmark repo publish target: ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}" >&2
    exit 2
  fi

  echo "No benchmark repo credential configured outside GitHub Actions; using existing ${BENCHMARK_REPO_REMOTE} remote."
}

for submission_dir in "${submission_source_dirs[@]}"; do
  for file_name in "${required_submission_files[@]}"; do
    if [[ ! -f "$submission_dir/$file_name" ]]; then
      echo "missing current submission file: $submission_dir/$file_name" >&2
      exit 2
    fi
  done
done

if [[ ! -d "$BENCHMARK_REPO_DIR/.git" ]]; then
  echo "benchmark repository checkout not found: $BENCHMARK_REPO_DIR" >&2
  exit 2
fi

if [[ ! -f "$WEBSITE_REPO_DIR/scripts/aggregate_results.py" ]]; then
  echo "website aggregation script not found: $WEBSITE_REPO_DIR/scripts/aggregate_results.py" >&2
  exit 2
fi

if [[ ! -f "$VLLM_HUST_REPO_DIR/pyproject.toml" ]]; then
  echo "vllm-hust repository checkout not found: $VLLM_HUST_REPO_DIR" >&2
  exit 2
fi

if [[ "${GITHUB_ACTIONS:-}" != "true" && "${ALLOW_LOCAL_GIT_RESET:-0}" != "1" ]]; then
  echo "refusing to reset a local checkout outside GitHub Actions; set ALLOW_LOCAL_GIT_RESET=1 to override" >&2
  exit 2
fi

run_id=${RUN_ID:-$(basename "${submission_source_dirs[0]}")}
if [[ -n "$CURRENT_SUBMISSIONS_DIR" ]]; then
  relative_submission_dir="submissions"
else
  relative_submission_dir="submissions/${submission_names[0]}"
fi
relative_snapshot_dir="leaderboard-data/snapshots"
publication_staging_dir=$(mktemp -d "$BENCHMARK_REPO_DIR/.snapshot-publication.XXXXXX")
staged_submission_dir="$publication_staging_dir/submissions"
staged_snapshot_dir="$publication_staging_dir/snapshots"

# Invoked indirectly by the EXIT trap below.
# shellcheck disable=SC2317,SC2329
cleanup_publication_staging() {
  rm -rf "$publication_staging_dir"
  if [[ -n "$ORIGINAL_REMOTE_URL" ]]; then
    git -C "$BENCHMARK_REPO_DIR" remote set-url "$BENCHMARK_REPO_REMOTE" "$ORIGINAL_REMOTE_URL" || true
  fi
  unset BENCHMARK_REPO_GH_TOKEN BENCHMARK_REPO_SSH_KEY GIT_ASKPASS GIT_SSH_COMMAND GIT_TERMINAL_PROMPT
  if [[ -n "$AUTH_DIR" ]]; then
    rm -rf "$AUTH_DIR"
  fi
}
trap cleanup_publication_staging EXIT

reset_publication_staging() {
  rm -rf "$publication_staging_dir" || return $?
  mkdir -p "$publication_staging_dir" || return $?
}

git -C "$BENCHMARK_REPO_DIR" config user.name "$GIT_COMMITTER_NAME"
git -C "$BENCHMARK_REPO_DIR" config user.email "$GIT_COMMITTER_EMAIL"
configure_push_remote

export VLLM_HUST_BENCHMARK_REPO="$BENCHMARK_REPO_DIR"
export VLLM_HUST_WEBSITE_REPO="$WEBSITE_REPO_DIR"
export VLLM_HUST_REPO="$VLLM_HUST_REPO_DIR"

prepare_publication_commit() {
  local fetched_target_sha

  reset_publication_staging || return $?
  fetch_target_branch_with_retry prepare || return $?
  if [[ -n "$SNAPSHOT_EXPECTED_BASE_SHA" ]]; then
    fetched_target_sha=$(git -C "$BENCHMARK_REPO_DIR" rev-parse \
      "$BENCHMARK_REPO_REMOTE/$SNAPSHOT_TARGET_BRANCH") || return $?
    if [[ "$fetched_target_sha" != "$SNAPSHOT_EXPECTED_BASE_SHA" ]]; then
      echo "benchmark publication base moved: expected $SNAPSHOT_EXPECTED_BASE_SHA, got $fetched_target_sha" >&2
      return 2
    fi
  fi
  git -C "$BENCHMARK_REPO_DIR" checkout -B "$SNAPSHOT_TARGET_BRANCH" "$BENCHMARK_REPO_REMOTE/$SNAPSHOT_TARGET_BRANCH" || return $?

  mkdir -p "$staged_submission_dir" || return $?
  cp -a "$BENCHMARK_REPO_DIR/submissions/." "$staged_submission_dir/" || return $?
  for index in "${!submission_source_dirs[@]}"; do
    staged_current_submission_dir="$staged_submission_dir/${submission_names[$index]}"
    mkdir -p "$staged_current_submission_dir" || return $?
    for file_name in "${required_submission_files[@]}"; do
      cp "${submission_source_dirs[$index]}/$file_name" \
        "$staged_current_submission_dir/$file_name" || return $?
    done
  done

  "$PYTHON_BIN" -m vllm_hust_benchmark.cli publish-website \
    --source-dir "$staged_submission_dir" \
    --output-dir "$staged_snapshot_dir" \
    --execute || return $?

  for file_name in "${required_snapshot_files[@]}"; do
    if [[ ! -f "$staged_snapshot_dir/$file_name" ]]; then
      echo "missing generated snapshot file: $staged_snapshot_dir/$file_name" >&2
      return 2
    fi
  done

  if ! PYTHONPATH="$BENCHMARK_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" "$BENCHMARK_REPO_DIR/scripts/validate_public_leaderboard_snapshots.py" \
    --snapshot-dir "$staged_snapshot_dir"; then
    echo "publication admission failed at public snapshot validation" >&2
    return 2
  fi
  if ! PYTHONPATH="$BENCHMARK_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" - "$staged_snapshot_dir" <<'PY'
import sys
from pathlib import Path

from vllm_hust_benchmark.integration import validate_public_snapshot_trend_admission

validate_public_snapshot_trend_admission(Path(sys.argv[1]))
PY
  then
    echo "publication admission failed at trend validation" >&2
    return 2
  fi

  mkdir -p "$BENCHMARK_REPO_DIR/submissions" "$SNAPSHOT_OUTPUT_DIR" || return $?
  for index in "${!submission_source_dirs[@]}"; do
    target_submission_dir="$BENCHMARK_REPO_DIR/submissions/${submission_names[$index]}"
    mkdir -p "$target_submission_dir" || return $?
    for file_name in "${required_submission_files[@]}"; do
      cp "${submission_source_dirs[$index]}/$file_name" \
        "$target_submission_dir/$file_name" || return $?
    done
  done
  for file_name in "${required_snapshot_files[@]}"; do
    cp "$staged_snapshot_dir/$file_name" "$SNAPSHOT_OUTPUT_DIR/$file_name" || return $?
  done

  if [[ -n "$LOCAL_SNAPSHOT_OUTPUT_DIR" ]]; then
    mkdir -p "$LOCAL_SNAPSHOT_OUTPUT_DIR" || return $?
    for file_name in "${required_snapshot_files[@]}"; do
      cp "$SNAPSHOT_OUTPUT_DIR/$file_name" "$LOCAL_SNAPSHOT_OUTPUT_DIR/$file_name" || return $?
    done
  fi

  git -C "$BENCHMARK_REPO_DIR" add "$relative_submission_dir" "$relative_snapshot_dir" || return $?
  if git -C "$BENCHMARK_REPO_DIR" diff --cached --quiet; then
    return 3
  else
    diff_status=$?
    if [[ "$diff_status" -ne 1 ]]; then
      return "$diff_status"
    fi
  fi

  git -C "$BENCHMARK_REPO_DIR" commit -m "$SNAPSHOT_COMMIT_MESSAGE" || return $?
}

verify_published_benchmark_repo_state() {
  local expected_commit=$1
  local verified_commit
  local file_name
  local index
  local submission_path

  fetch_target_branch_with_retry verify || return $?
  verified_commit=$(git -C "$BENCHMARK_REPO_DIR" rev-parse "$BENCHMARK_REPO_REMOTE/$SNAPSHOT_TARGET_BRANCH") || return $?
  if [[ "$verified_commit" != "$expected_commit" ]]; then
    write_github_env GITHUB_SNAPSHOT_SYNC_VERIFICATION failed
    echo "benchmark publication verification failed: expected $expected_commit, got $verified_commit" >&2
    return 1
  fi

  for index in "${!submission_names[@]}"; do
    submission_path="submissions/${submission_names[$index]}"
    for file_name in "${required_submission_files[@]}"; do
      if ! git -C "$BENCHMARK_REPO_DIR" cat-file -e \
        "$verified_commit:$submission_path/$file_name"; then
        write_github_env GITHUB_SNAPSHOT_SYNC_VERIFICATION failed
        echo "benchmark publication verification failed: missing $submission_path/$file_name" >&2
        return 1
      fi
    done
  done

  for file_name in "${required_snapshot_files[@]}"; do
    if ! git -C "$BENCHMARK_REPO_DIR" cat-file -e \
      "$verified_commit:$relative_snapshot_dir/$file_name"; then
      write_github_env GITHUB_SNAPSHOT_SYNC_VERIFICATION failed
      echo "benchmark publication verification failed: missing $relative_snapshot_dir/$file_name" >&2
      return 1
    fi
  done

  write_github_env GITHUB_SNAPSHOT_SYNC_VERIFICATION verified
  write_github_env GITHUB_SNAPSHOT_SYNC_VERIFIED_COMMIT "$verified_commit"
  echo "Verified benchmark publication at ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}: $verified_commit"
}

for attempt in $(seq 1 "$SNAPSHOT_MAX_PUSH_ATTEMPTS"); do
  if prepare_publication_commit; then
    snapshot_commit=$(git -C "$BENCHMARK_REPO_DIR" rev-parse HEAD)
    if git -C "$BENCHMARK_REPO_DIR" push "$BENCHMARK_REPO_REMOTE" "HEAD:$SNAPSHOT_TARGET_BRANCH"; then
      write_github_env GITHUB_SNAPSHOT_SYNC_STATUS pushed
      write_github_env GITHUB_SNAPSHOT_SYNC_COMMIT "$snapshot_commit"
      write_github_env GITHUB_SNAPSHOT_SYNC_REPO "$BENCHMARK_REPO_SLUG"
      write_github_env GITHUB_SNAPSHOT_SYNC_BRANCH "$SNAPSHOT_TARGET_BRANCH"
      write_github_env GITHUB_SNAPSHOT_SYNC_SUBMISSION_PATH "$relative_submission_dir"
      write_github_env GITHUB_SNAPSHOT_SYNC_SNAPSHOT_PATH "$relative_snapshot_dir"
      if verify_published_benchmark_repo_state "$snapshot_commit"; then
        echo "Pushed benchmark publication to ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}: $snapshot_commit"
        echo "Submission path: $relative_submission_dir"
        echo "Snapshot path: $relative_snapshot_dir"
      else
        verification_status=$?
        write_github_env GITHUB_SNAPSHOT_SYNC_VERIFICATION failed
        echo "benchmark publication push succeeded, but verification failed for $snapshot_commit" >&2
        exit "$verification_status"
      fi
      exit 0
    fi

    if [[ "$attempt" -lt "$SNAPSHOT_MAX_PUSH_ATTEMPTS" ]]; then
      echo "benchmark publication push failed; retrying with fresh ${BENCHMARK_REPO_REMOTE}/${SNAPSHOT_TARGET_BRANCH} in ${SNAPSHOT_PUSH_RETRY_SECONDS}s (attempt $attempt/$SNAPSHOT_MAX_PUSH_ATTEMPTS)" >&2
      sleep "$SNAPSHOT_PUSH_RETRY_SECONDS"
      continue
    fi
    break
  else
    prepare_status=$?
    if [[ "$prepare_status" -eq 3 ]]; then
      echo "Benchmark publication already includes submission $run_id"
      echo "Benchmark repo target: ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}"
      echo "Submission path: $relative_submission_dir"
      echo "Snapshot path: $relative_snapshot_dir"
      write_github_env GITHUB_SNAPSHOT_SYNC_STATUS unchanged
      write_github_env GITHUB_SNAPSHOT_SYNC_REPO "$BENCHMARK_REPO_SLUG"
      write_github_env GITHUB_SNAPSHOT_SYNC_BRANCH "$SNAPSHOT_TARGET_BRANCH"
      write_github_env GITHUB_SNAPSHOT_SYNC_SUBMISSION_PATH "$relative_submission_dir"
      write_github_env GITHUB_SNAPSHOT_SYNC_SNAPSHOT_PATH "$relative_snapshot_dir"
      verify_published_benchmark_repo_state "$(git -C "$BENCHMARK_REPO_DIR" rev-parse "$BENCHMARK_REPO_REMOTE/$SNAPSHOT_TARGET_BRANCH")"
      exit 0
    fi
    write_github_env GITHUB_SNAPSHOT_SYNC_STATUS rejected
    exit "$prepare_status"
  fi
done

echo "failed to push benchmark publication after $SNAPSHOT_MAX_PUSH_ATTEMPTS attempts" >&2
echo "Benchmark repo target: ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}" >&2
echo "Submission path: $relative_submission_dir" >&2
echo "Snapshot path: $relative_snapshot_dir" >&2
write_github_env GITHUB_SNAPSHOT_SYNC_STATUS failed
exit 1
