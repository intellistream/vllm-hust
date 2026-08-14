# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/scripts/l3_benchmark_publish_preflight.sh"
)


def run_preflight(
    tmp_path: Path, extra_env: dict[str, str]
) -> tuple[subprocess.CompletedProcess[str], str]:
    github_env = tmp_path / "github-env"
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    git_log = tmp_path / "git.log"
    fake_git = fake_bin / "git"
    fake_git.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s\\n' "$*" >> "$FAKE_GIT_LOG"
if [[ "$*" == *" fetch "* || "$*" == *" fetch" ]]; then
  exit "${FAKE_GIT_FETCH_EXIT:-0}"
fi
if [[ "$*" == *" push --dry-run "* ]]; then
  exit "${FAKE_GIT_PUSH_EXIT:-0}"
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    fake_curl = fake_bin / "curl"
    fake_curl.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ -n "${BENCHMARK_REPO_GH_TOKEN:-}" || -n "${BENCHMARK_REPO_SSH_KEY:-}" ]]; then
  echo "writer credential leaked into API subprocess environment" >&2
  exit 9
fi
if [[ "$*" == *fake-token* ]]; then
  echo "writer token leaked into API subprocess arguments" >&2
  exit 10
fi
url="${!#}"
if [[ "$url" == */branches/* ]]; then
  printf '{"name":"main","protected":%s}\\n' "${FAKE_BRANCH_PROTECTED:-false}"
else
  printf '{"full_name":"vLLM-HUST/vllm-hust-benchmark","permissions":{"push":%s}}\\n' \\
    "${FAKE_REPO_PUSH_PERMISSION:-true}"
fi
""",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "GITHUB_ENV": str(github_env),
            "BENCHMARK_REPO_SLUG": "vLLM-HUST/vllm-hust-benchmark",
            "FAKE_GIT_LOG": str(git_log),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )
    env.update(extra_env)

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    env_text = github_env.read_text(encoding="utf-8") if github_env.exists() else ""
    return result, env_text


def test_l3_publish_preflight_skips_when_publish_disabled(tmp_path):
    result, env_text = run_preflight(tmp_path, {"PUBLISH_TO_BENCHMARK_REPO": "0"})

    assert result.returncode == 0
    assert "L3 benchmark repository publish preflight: skipped" in result.stdout
    assert "L3_BENCHMARK_PUBLISH_PREFLIGHT=skipped" in env_text


def test_l3_publish_preflight_fails_when_credential_missing(tmp_path):
    result, env_text = run_preflight(tmp_path, {"PUBLISH_TO_BENCHMARK_REPO": "1"})

    assert result.returncode == 2
    assert "no cross-repository write credential" in result.stderr
    assert "Target: vLLM-HUST/vllm-hust-benchmark@main" in result.stderr
    assert "L3_BENCHMARK_PUBLISH_PREFLIGHT=credential-missing" in env_text
    assert "L3_BENCHMARK_PUBLISH_TARGET=vLLM-HUST/vllm-hust-benchmark@main" in env_text


def test_l3_publish_preflight_rejects_ssh_key_without_api_permission_proof(tmp_path):
    result, env_text = run_preflight(
        tmp_path,
        {
            "PUBLISH_TO_BENCHMARK_REPO": "1",
            "BENCHMARK_REPO_SSH_KEY": "fake-key",
        },
    )

    assert result.returncode == 2
    assert "SSH key alone cannot prove" in result.stderr
    assert "L3_BENCHMARK_PUBLISH_PREFLIGHT=authorization-failed" in env_text


def test_l3_publish_preflight_accepts_github_token(tmp_path):
    result, env_text = run_preflight(
        tmp_path,
        {
            "PUBLISH_TO_BENCHMARK_REPO": "1",
            "BENCHMARK_REPO_GH_TOKEN": "fake-token",
        },
    )

    assert result.returncode == 0
    assert "L3_BENCHMARK_PUBLISH_PREFLIGHT=ok" in env_text
    assert "L3_BENCHMARK_PUBLISH_CREDENTIAL=token" in env_text
    git_log = (tmp_path / "git.log").read_text(encoding="utf-8")
    assert "fetch --quiet --depth 1 benchmark refs/heads/main" in git_log
    assert "push --dry-run benchmark FETCH_HEAD:refs/heads/main" in git_log


def test_l3_publish_preflight_uses_proven_token_when_ssh_key_is_also_set(tmp_path):
    result, env_text = run_preflight(
        tmp_path,
        {
            "PUBLISH_TO_BENCHMARK_REPO": "1",
            "BENCHMARK_REPO_GH_TOKEN": "fake-token",
            "BENCHMARK_REPO_SSH_KEY": "fake-key",
        },
    )

    assert result.returncode == 0
    assert "L3_BENCHMARK_PUBLISH_CREDENTIAL=token" in env_text
    assert "L3_BENCHMARK_PUBLISH_CREDENTIAL=ssh-key" not in env_text


def test_l3_publish_preflight_rejects_writer_without_branch_authorization(tmp_path):
    result, env_text = run_preflight(
        tmp_path,
        {
            "PUBLISH_TO_BENCHMARK_REPO": "1",
            "BENCHMARK_REPO_GH_TOKEN": "fake-token",
            "FAKE_GIT_PUSH_EXIT": "1",
        },
    )

    assert result.returncode == 2
    assert "not authorized for exact target" in result.stderr
    assert "L3_BENCHMARK_PUBLISH_PREFLIGHT=authorization-failed" in env_text


def test_l3_publish_preflight_rejects_token_without_push_permission(tmp_path):
    result, env_text = run_preflight(
        tmp_path,
        {
            "PUBLISH_TO_BENCHMARK_REPO": "1",
            "BENCHMARK_REPO_GH_TOKEN": "fake-token",
            "FAKE_REPO_PUSH_PERMISSION": "false",
        },
    )

    assert result.returncode == 2
    assert "Non-mutating GitHub permission check rejected" in result.stderr
    assert "L3_BENCHMARK_PUBLISH_PREFLIGHT=authorization-failed" in env_text


def test_l3_publish_preflight_rejects_unproven_protected_branch_access(tmp_path):
    result, env_text = run_preflight(
        tmp_path,
        {
            "PUBLISH_TO_BENCHMARK_REPO": "1",
            "BENCHMARK_REPO_GH_TOKEN": "fake-token",
            "FAKE_BRANCH_PROTECTED": "true",
        },
    )

    assert result.returncode == 2
    assert "Non-mutating GitHub permission check rejected" in result.stderr
    assert "L3_BENCHMARK_PUBLISH_PREFLIGHT=authorization-failed" in env_text
