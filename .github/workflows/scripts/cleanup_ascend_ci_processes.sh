#!/bin/bash
set -euo pipefail

workspace_root=${GITHUB_WORKSPACE:-$PWD}
runtime_root=${VLLM_HUST_CI_RUNTIME_ROOT:-$workspace_root/.ci-runtime}
marker_dir="$runtime_root/process-markers"
marker_pid_file="$marker_dir/vllm-server.pid"
marker_pgid_file="$marker_dir/vllm-server.pgid"

print_matches() {
  ps -eo pid,ppid,pgid,sid,etimes,args \
    | grep -F "$workspace_root" \
    | grep -E 'vllm|python|pytest' \
    | grep -v grep || true
}

echo "Ascend CI cleanup workspace: $workspace_root"
echo "Ascend CI cleanup runtime root: $runtime_root"

if [[ -f "$marker_pgid_file" ]]; then
  marker_pgid=$(tr -d '[:space:]' < "$marker_pgid_file")
  if [[ -n "$marker_pgid" ]] && kill -0 "$marker_pgid" 2>/dev/null; then
    echo "Cleaning leftover vLLM process group: $marker_pgid"
    kill -TERM -- "-$marker_pgid" 2>/dev/null || true
    for _ in $(seq 1 10); do
      if ! kill -0 "$marker_pgid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    kill -KILL -- "-$marker_pgid" 2>/dev/null || true
  fi
fi

if [[ -f "$marker_pid_file" ]]; then
  marker_pid=$(tr -d '[:space:]' < "$marker_pid_file")
  if [[ -n "$marker_pid" ]] && kill -0 "$marker_pid" 2>/dev/null; then
    echo "Cleaning leftover vLLM process: $marker_pid"
    kill "$marker_pid" 2>/dev/null || true
  fi
fi

remaining_matches=$(print_matches)
if [[ -n "$remaining_matches" ]]; then
  echo "Remaining workspace-scoped vLLM/Python processes after cleanup:"
  echo "$remaining_matches"
else
  echo "No remaining workspace-scoped vLLM/Python processes detected."
fi

rm -f "$marker_pid_file" "$marker_pgid_file"