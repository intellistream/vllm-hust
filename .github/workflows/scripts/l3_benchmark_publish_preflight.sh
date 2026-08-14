#!/bin/bash
set -euo pipefail

PUBLISH_TO_BENCHMARK_REPO=${PUBLISH_TO_BENCHMARK_REPO:-0}
BENCHMARK_REPO_SLUG=${BENCHMARK_REPO_SLUG:-vLLM-HUST/vllm-hust-benchmark}
SNAPSHOT_TARGET_BRANCH=${SNAPSHOT_TARGET_BRANCH:-main}
BENCHMARK_REPO_GH_TOKEN=${BENCHMARK_REPO_GH_TOKEN:-}
BENCHMARK_REPO_SSH_KEY=${BENCHMARK_REPO_SSH_KEY:-}
CURL_BIN=${CURL_BIN:-curl}
PYTHON_BIN=${PYTHON_BIN:-python3}
AUTH_DIR=

cleanup() {
  unset BENCHMARK_REPO_GH_TOKEN BENCHMARK_REPO_SSH_KEY GIT_ASKPASS GIT_SSH_COMMAND GIT_TERMINAL_PROMPT
  if [[ -n "$AUTH_DIR" ]]; then
    rm -rf "$AUTH_DIR"
  fi
}
trap cleanup EXIT

write_github_env() {
  local key=$1
  local value=$2
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    printf '%s=%s\n' "$key" "$value" >>"$GITHUB_ENV"
  fi
}

if [[ "$PUBLISH_TO_BENCHMARK_REPO" != "1" ]]; then
  write_github_env L3_BENCHMARK_PUBLISH_PREFLIGHT skipped
  echo "L3 benchmark repository publish preflight: skipped"
  exit 0
fi

write_github_env L3_BENCHMARK_PUBLISH_PREFLIGHT running
write_github_env L3_BENCHMARK_PUBLISH_TARGET "${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}"

if [[ -z "$BENCHMARK_REPO_GH_TOKEN" && -z "$BENCHMARK_REPO_SSH_KEY" ]]; then
  write_github_env L3_BENCHMARK_PUBLISH_PREFLIGHT credential-missing
  echo "L3 benchmark repository publication is enabled, but no cross-repository write credential is available." >&2
  echo "Target: ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}" >&2
  echo "Configure one of these secrets before enabling benchmark repo publish:" >&2
  echo "  - VLLM_ASCEND_HUST_BENCHMARK_SSH_KEY" >&2
  echo "  - VLLM_HUST_BENCHMARK_GH_TOKEN" >&2
  exit 2
fi

if [[ -z "$BENCHMARK_REPO_GH_TOKEN" ]]; then
  write_github_env L3_BENCHMARK_PUBLISH_PREFLIGHT authorization-failed
  echo "An SSH key alone cannot prove non-mutative write authorization for the exact benchmark target." >&2
  echo "Configure VLLM_HUST_BENCHMARK_GH_TOKEN with repository contents write permission." >&2
  exit 2
fi

AUTH_DIR=$(mktemp -d "${RUNNER_TEMP:-/tmp}/benchmark-writer-preflight.XXXXXX")
repo_metadata_file="$AUTH_DIR/repository.json"
branch_metadata_file="$AUTH_DIR/branch.json"
api_config_file="$AUTH_DIR/github-api.conf"
printf 'header = "Authorization: Bearer %s"\n' "$BENCHMARK_REPO_GH_TOKEN" >"$api_config_file"
chmod 600 "$api_config_file"
api_headers=(
  --config "$api_config_file"
  --fail
  --silent
  --show-error
  -H "Accept: application/vnd.github+json"
  -H "X-GitHub-Api-Version: 2022-11-28"
)
if ! env -u BENCHMARK_REPO_GH_TOKEN -u BENCHMARK_REPO_SSH_KEY \
  "$CURL_BIN" "${api_headers[@]}" \
  "https://api.github.com/repos/${BENCHMARK_REPO_SLUG}" >"$repo_metadata_file"; then
  write_github_env L3_BENCHMARK_PUBLISH_PREFLIGHT authorization-failed
  echo "Unable to verify benchmark repository writer permission: $BENCHMARK_REPO_SLUG" >&2
  exit 2
fi
if ! env -u BENCHMARK_REPO_GH_TOKEN -u BENCHMARK_REPO_SSH_KEY \
  "$CURL_BIN" "${api_headers[@]}" \
  "https://api.github.com/repos/${BENCHMARK_REPO_SLUG}/branches/${SNAPSHOT_TARGET_BRANCH}" \
  >"$branch_metadata_file"; then
  write_github_env L3_BENCHMARK_PUBLISH_PREFLIGHT authorization-failed
  echo "Unable to verify exact benchmark target branch: ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}" >&2
  exit 2
fi
if ! env -u BENCHMARK_REPO_GH_TOKEN -u BENCHMARK_REPO_SSH_KEY \
  "$PYTHON_BIN" - "$repo_metadata_file" "$branch_metadata_file" \
  "$BENCHMARK_REPO_SLUG" "$SNAPSHOT_TARGET_BRANCH" <<'PY'
import json
import sys
from pathlib import Path

repo_file, branch_file, expected_repo, expected_branch = sys.argv[1:]
repo = json.loads(Path(repo_file).read_text(encoding="utf-8"))
branch = json.loads(Path(branch_file).read_text(encoding="utf-8"))
errors = []
if repo.get("full_name") != expected_repo:
    errors.append("authenticated repository identity does not match")
if (repo.get("permissions") or {}).get("push") is not True:
    errors.append("authenticated credential does not have push permission")
if branch.get("name") != expected_branch:
    errors.append("target branch identity does not match")
if branch.get("protected") is not False:
    errors.append("target branch is protected; direct writer authorization is not proven")
if errors:
    print("; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
PY
then
  write_github_env L3_BENCHMARK_PUBLISH_PREFLIGHT authorization-failed
  echo "Non-mutating GitHub permission check rejected ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}" >&2
  exit 2
fi

remote_url=
askpass_file="$AUTH_DIR/askpass.sh"
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
write_github_env L3_BENCHMARK_PUBLISH_CREDENTIAL token

auth_probe_repo="$AUTH_DIR/repo"
git init --quiet "$auth_probe_repo"
git -C "$auth_probe_repo" remote add benchmark "$remote_url"
if ! git -C "$auth_probe_repo" fetch --quiet --depth 1 benchmark \
  "refs/heads/${SNAPSHOT_TARGET_BRANCH}"; then
  write_github_env L3_BENCHMARK_PUBLISH_PREFLIGHT authorization-failed
  echo "Unable to authenticate and fetch exact benchmark target: ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}" >&2
  exit 2
fi
if ! git -C "$auth_probe_repo" push --dry-run benchmark \
  "FETCH_HEAD:refs/heads/${SNAPSHOT_TARGET_BRANCH}"; then
  write_github_env L3_BENCHMARK_PUBLISH_PREFLIGHT authorization-failed
  echo "Benchmark writer is not authorized for exact target: ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}" >&2
  exit 2
fi

write_github_env L3_BENCHMARK_PUBLISH_PREFLIGHT ok
echo "L3 benchmark repository publish preflight: ok"
echo "Target: ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}"
