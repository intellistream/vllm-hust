# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/scripts/sync_benchmark_snapshots_to_github.sh"
)


def run(
    cmd: list[str], cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def init_bare_remote(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    remote.mkdir()
    run(["git", "init", "--bare", str(remote)], tmp_path)
    run(["git", "clone", str(remote), str(seed)], tmp_path)
    run(["git", "config", "user.name", "Test"], seed)
    run(["git", "config", "user.email", "test@example.com"], seed)
    (seed / "README.md").write_text("seed\n", encoding="utf-8")
    run(["git", "add", "README.md"], seed)
    run(["git", "commit", "-m", "seed"], seed)
    run(["git", "push", "origin", "HEAD:main"], seed)
    return remote, seed


def write_fake_python(tmp_path: Path) -> Path:
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "$*" == *"validate_public_leaderboard_snapshots.py"* ]]; then
  exit "${FAKE_PUBLIC_VALIDATOR_EXIT:-0}"
fi
if [[ "$*" == *"validate-trend"* ]]; then
  exit "${FAKE_TREND_VALIDATOR_EXIT:-0}"
fi
if [[ "$1" != "-m" \
  || "$2" != "vllm_hust_benchmark.cli" \
  || "$3" != "publish-website" ]]; then
  echo "unexpected fake python invocation: $*" >&2
  exit 2
fi
shift 3
source_dir=""
output_dir=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --source-dir)
      source_dir="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
mkdir -p "$output_dir"
if [[ -d "$source_dir/stale-ci" ]]; then
  stale_submission="stale-present"
else
  stale_submission="stale-absent"
fi
printf '{"stale_submission":"%s"}\\n' "$stale_submission" \\
  > "$output_dir/leaderboard_single.json"
printf '{}\\n' > "$output_dir/leaderboard_multi.json"
printf '{}\\n' > "$output_dir/leaderboard_compare.json"
printf '{}\\n' > "$output_dir/last_updated.json"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    return fake_python


def write_submission_evidence(submission: Path) -> None:
    """Write the complete evidence set required by publication admission."""
    submission.mkdir(parents=True, exist_ok=True)
    files = {
        "leaderboard_manifest.json": "{}\n",
        "run_leaderboard.json": "{}\n",
        "env-manifest.json": '{"git_info": {}}\n',
    }
    for name, content in files.items():
        (submission / name).write_text(content, encoding="utf-8")
    checksums = "".join(
        f"{hashlib.sha256((submission / name).read_bytes()).hexdigest()}  ./{name}\n"
        for name in files
    )
    (submission / "checksums.sha256").write_text(checksums, encoding="utf-8")
    (submission / "STATUS").write_text("OK\n", encoding="utf-8")


def write_flaky_git(tmp_path: Path) -> Path:
    fake_git = tmp_path / "fake-bin" / "git"
    fake_git.parent.mkdir()
    fake_git.write_text(
        """#!/bin/bash
set -euo pipefail

for argument in "$@"; do
  if [[ "$argument" == "push" && ! -f "$FAKE_GIT_PUSH_STATE" ]]; then
    touch "$FAKE_GIT_PUSH_STATE"
    "$REAL_GIT" -C "$FAKE_GIT_SEED" rm -r submissions/stale-ci
    "$REAL_GIT" -C "$FAKE_GIT_SEED" commit -m "remove stale submission"
    "$REAL_GIT" -C "$FAKE_GIT_SEED" push origin HEAD:main
    exit 1
  fi
done

exec "$REAL_GIT" "$@"
""",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    return fake_git


def test_sync_benchmark_snapshots_verifies_published_commit(tmp_path):
    remote, _seed = init_bare_remote(tmp_path)
    benchmark_repo = tmp_path / "benchmark"
    website_repo = tmp_path / "website"
    vllm_hust_repo = tmp_path / "vllm-hust"
    submission = tmp_path / "submission"
    github_env = tmp_path / "github-env"
    fake_python = write_fake_python(tmp_path)

    run(["git", "clone", str(remote), str(benchmark_repo)], tmp_path)
    (benchmark_repo / "submissions").mkdir()
    run(["git", "config", "user.name", "Test"], benchmark_repo)
    run(["git", "config", "user.email", "test@example.com"], benchmark_repo)
    (website_repo / "scripts").mkdir(parents=True)
    (website_repo / "scripts/aggregate_results.py").write_text(
        "# fake\n", encoding="utf-8"
    )
    vllm_hust_repo.mkdir()
    (vllm_hust_repo / "pyproject.toml").write_text(
        "[project]\nname='fake'\n", encoding="utf-8"
    )
    write_submission_evidence(submission)

    env = os.environ.copy()
    env.update(
        {
            "ALLOW_LOCAL_GIT_RESET": "1",
            "BENCHMARK_REPO_DIR": str(benchmark_repo),
            "BENCHMARK_REPO_REMOTE": "origin",
            "BENCHMARK_REPO_SLUG": "local/benchmark",
            "CURRENT_SUBMISSION_DIR": str(submission),
            "GITHUB_ENV": str(github_env),
            "LOCAL_SNAPSHOT_OUTPUT_DIR": str(tmp_path / "local-snapshots"),
            "PYTHON_BIN": str(fake_python),
            "RUN_ID": "ci-test",
            "SNAPSHOT_TARGET_BRANCH": "main",
            "VLLM_HUST_REPO_DIR": str(vllm_hust_repo),
            "WEBSITE_REPO_DIR": str(website_repo),
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    env_text = github_env.read_text(encoding="utf-8")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Verified benchmark publication" in result.stdout
    assert "GITHUB_SNAPSHOT_SYNC_STATUS=pushed" in env_text
    assert "GITHUB_SNAPSHOT_SYNC_VERIFICATION=verified" in env_text
    assert "GITHUB_SNAPSHOT_SYNC_VERIFIED_COMMIT=" in env_text
    assert "GITHUB_SNAPSHOT_SYNC_SUBMISSION_PATH=submissions/ci-test" in env_text
    assert "GITHUB_SNAPSHOT_SYNC_SNAPSHOT_PATH=leaderboard-data/snapshots" in env_text
    assert (benchmark_repo / "submissions" / "ci-test" / "STATUS").read_text(
        encoding="utf-8"
    ) == "OK\n"


@pytest.mark.parametrize(
    ("validator_environment", "failure_message"),
    [
        (
            "FAKE_PUBLIC_VALIDATOR_EXIT",
            "publication admission failed at public snapshot validation",
        ),
        (
            "FAKE_TREND_VALIDATOR_EXIT",
            "publication admission failed at trend validation",
        ),
    ],
)
def test_invalid_snapshot_does_not_change_benchmark_remote(
    tmp_path, validator_environment, failure_message
):
    remote, _seed = init_bare_remote(tmp_path)
    benchmark_repo = tmp_path / "benchmark"
    website_repo = tmp_path / "website"
    vllm_hust_repo = tmp_path / "vllm-hust"
    submission = tmp_path / "submission"
    github_env = tmp_path / "github-env"
    fake_python = write_fake_python(tmp_path)

    run(["git", "clone", str(remote), str(benchmark_repo)], tmp_path)
    (benchmark_repo / "submissions").mkdir()
    (website_repo / "scripts").mkdir(parents=True)
    (website_repo / "scripts/aggregate_results.py").write_text(
        "# fake\n", encoding="utf-8"
    )
    vllm_hust_repo.mkdir()
    (vllm_hust_repo / "pyproject.toml").write_text(
        "[project]\nname='fake'\n", encoding="utf-8"
    )
    write_submission_evidence(submission)
    remote_head = run(
        ["git", "--git-dir", str(remote), "rev-parse", "main"], tmp_path
    ).stdout.strip()

    env = os.environ.copy()
    env.update(
        {
            "ALLOW_LOCAL_GIT_RESET": "1",
            "BENCHMARK_REPO_DIR": str(benchmark_repo),
            "BENCHMARK_REPO_REMOTE": "origin",
            "BENCHMARK_REPO_SLUG": "local/benchmark",
            "CURRENT_SUBMISSION_DIR": str(submission),
            "GITHUB_ENV": str(github_env),
            "PYTHON_BIN": str(fake_python),
            "RUN_ID": "invalid-ci-test",
            "SNAPSHOT_TARGET_BRANCH": "main",
            "VLLM_HUST_REPO_DIR": str(vllm_hust_repo),
            "WEBSITE_REPO_DIR": str(website_repo),
        }
    )
    env[validator_environment] = "2"

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 2, result.stdout + result.stderr
    assert failure_message in result.stderr
    assert (
        run(
            ["git", "--git-dir", str(remote), "rev-parse", "main"], tmp_path
        ).stdout.strip()
        == remote_head
    )
    assert not (benchmark_repo / "submissions" / "invalid-ci-test").exists()
    assert not (benchmark_repo / "leaderboard-data" / "snapshots").exists()
    assert "GITHUB_SNAPSHOT_SYNC_STATUS=rejected" in github_env.read_text(
        encoding="utf-8"
    )


def test_push_retry_rebuilds_staging_from_fresh_remote(tmp_path):
    remote, seed = init_bare_remote(tmp_path)
    stale_submission = seed / "submissions" / "stale-ci"
    stale_submission.mkdir(parents=True)
    (stale_submission / "obsolete.txt").write_text("stale\n", encoding="utf-8")
    retained_submission = seed / "submissions" / "retained-ci"
    retained_submission.mkdir()
    (retained_submission / "result.txt").write_text("current\n", encoding="utf-8")
    run(["git", "add", "submissions"], seed)
    run(["git", "commit", "-m", "add stale submission"], seed)
    run(["git", "push", "origin", "HEAD:main"], seed)

    benchmark_repo = tmp_path / "benchmark"
    website_repo = tmp_path / "website"
    vllm_hust_repo = tmp_path / "vllm-hust"
    submission = tmp_path / "submission"
    github_env = tmp_path / "github-env"
    fake_python = write_fake_python(tmp_path)
    fake_git = write_flaky_git(tmp_path)

    run(["git", "clone", str(remote), str(benchmark_repo)], tmp_path)
    (website_repo / "scripts").mkdir(parents=True)
    (website_repo / "scripts/aggregate_results.py").write_text(
        "# fake\n", encoding="utf-8"
    )
    vllm_hust_repo.mkdir()
    (vllm_hust_repo / "pyproject.toml").write_text(
        "[project]\nname='fake'\n", encoding="utf-8"
    )
    write_submission_evidence(submission)

    env = os.environ.copy()
    env.update(
        {
            "ALLOW_LOCAL_GIT_RESET": "1",
            "BENCHMARK_REPO_DIR": str(benchmark_repo),
            "BENCHMARK_REPO_REMOTE": "origin",
            "BENCHMARK_REPO_SLUG": "local/benchmark",
            "CURRENT_SUBMISSION_DIR": str(submission),
            "FAKE_GIT_PUSH_STATE": str(tmp_path / "first-push-failed"),
            "FAKE_GIT_SEED": str(seed),
            "GITHUB_ENV": str(github_env),
            "PATH": f"{fake_git.parent}:{env['PATH']}",
            "PYTHON_BIN": str(fake_python),
            "REAL_GIT": shutil.which("git") or "git",
            "RUN_ID": "retry-ci-test",
            "SNAPSHOT_MAX_PUSH_ATTEMPTS": "2",
            "SNAPSHOT_PUSH_RETRY_SECONDS": "0",
            "SNAPSHOT_TARGET_BRANCH": "main",
            "VLLM_HUST_REPO_DIR": str(vllm_hust_repo),
            "WEBSITE_REPO_DIR": str(website_repo),
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "push failed; retrying with fresh origin/main" in result.stderr
    assert (tmp_path / "first-push-failed").is_file()
    assert (
        run(
            [
                "git",
                "--git-dir",
                str(remote),
                "show",
                "main:leaderboard-data/snapshots/leaderboard_single.json",
            ],
            tmp_path,
        ).stdout.strip()
        == '{"stale_submission":"stale-absent"}'
    )
