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
if [[ "$1" == "-" && "$#" == "2" ]]; then
  if [[ -n "${FAKE_TREND_VALIDATOR_INPUTS_FILE:-}" ]]; then
    printf '%s\n' "$2" >> "$FAKE_TREND_VALIDATOR_INPUTS_FILE"
  fi
  cat >/dev/null
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
        "pip-packages.json": "[]\n",
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


def write_flaky_fetch_git(tmp_path: Path) -> Path:
    fake_git = tmp_path / "fake-fetch-bin" / "git"
    fake_git.parent.mkdir()
    fake_git.write_text(
        """#!/bin/bash
set -euo pipefail

for argument in "$@"; do
  if [[ "$argument" == "fetch" ]]; then
    fetch_count=0
    if [[ -f "$FAKE_GIT_FETCH_COUNT" ]]; then
      read -r fetch_count < "$FAKE_GIT_FETCH_COUNT"
    fi
    fetch_count=$((fetch_count + 1))
    printf '%s\n' "$fetch_count" > "$FAKE_GIT_FETCH_COUNT"
    case ",${FAKE_GIT_FETCH_FAIL_CALLS:-}," in
      *",${fetch_count},"*) exit 1 ;;
    esac
    break
  fi
done

exec "$REAL_GIT" "$@"
""",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    return fake_git


def write_mismatched_verify_git(tmp_path: Path) -> Path:
    fake_git = tmp_path / "fake-verify-bin" / "git"
    fake_git.parent.mkdir()
    fake_git.write_text(
        """#!/bin/bash
set -euo pipefail

if [[ "$*" == *"rev-parse origin/main"* ]]; then
  printf '0000000000000000000000000000000000000000\n'
  exit 0
fi

exec "$REAL_GIT" "$@"
""",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    return fake_git


def prepare_sync_environment(
    tmp_path: Path, run_id: str
) -> tuple[dict[str, str], Path, Path, Path]:
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
            "RUN_ID": run_id,
            "SNAPSHOT_FETCH_RETRY_SECONDS": "0",
            "SNAPSHOT_TARGET_BRANCH": "main",
            "VLLM_HUST_REPO_DIR": str(vllm_hust_repo),
            "WEBSITE_REPO_DIR": str(website_repo),
        }
    )
    return env, remote, benchmark_repo, github_env


def test_sync_benchmark_snapshots_verifies_published_commit(tmp_path):
    remote, _seed = init_bare_remote(tmp_path)
    benchmark_repo = tmp_path / "benchmark"
    website_repo = tmp_path / "website"
    vllm_hust_repo = tmp_path / "vllm-hust"
    submission = tmp_path / "submission"
    github_env = tmp_path / "github-env"
    trend_validator_inputs = tmp_path / "trend-validator-inputs"
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
            "FAKE_TREND_VALIDATOR_INPUTS_FILE": str(trend_validator_inputs),
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
    assert run(
        ["git", "remote", "get-url", "origin"], benchmark_repo
    ).stdout.strip() == str(remote)
    assert (benchmark_repo / "submissions" / "ci-test" / "STATUS").read_text(
        encoding="utf-8"
    ) == "OK\n"
    assert (benchmark_repo / "submissions" / "ci-test" / "pip-packages.json").read_text(
        encoding="utf-8"
    ) == "[]\n"
    trend_inputs = trend_validator_inputs.read_text(encoding="utf-8").splitlines()
    assert len(trend_inputs) == 1
    trend_input = Path(trend_inputs[0])
    assert trend_input.name == "snapshots"
    assert trend_input.parent.name.startswith(".snapshot-publication.")


def test_multi_submission_sync_publishes_one_atomic_commit(tmp_path):
    env, remote, benchmark_repo, _github_env = prepare_sync_environment(
        tmp_path, "multi-ci"
    )
    submissions_dir = tmp_path / "multi-submissions"
    write_submission_evidence(submissions_dir / "run-random")
    write_submission_evidence(submissions_dir / "run-sharegpt")
    env.pop("CURRENT_SUBMISSION_DIR")
    env["CURRENT_SUBMISSIONS_DIR"] = str(submissions_dir)

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    remote_head = run(
        ["git", "--git-dir", str(remote), "rev-parse", "main"], tmp_path
    ).stdout.strip()
    parents = (
        run(
            ["git", "--git-dir", str(remote), "show", "-s", "--format=%P", remote_head],
            tmp_path,
        )
        .stdout.strip()
        .split()
    )
    assert len(parents) == 1
    for run_id in ("run-random", "run-sharegpt"):
        assert (
            run(
                [
                    "git",
                    "--git-dir",
                    str(remote),
                    "cat-file",
                    "-e",
                    f"{remote_head}:submissions/{run_id}/STATUS",
                ],
                tmp_path,
            ).returncode
            == 0
        )
    assert (
        benchmark_repo / "leaderboard-data/snapshots/leaderboard_single.json"
    ).exists()


def test_multi_submission_failure_after_staging_leaves_remote_unchanged(tmp_path):
    env, remote, benchmark_repo, _github_env = prepare_sync_environment(
        tmp_path, "multi-failure"
    )
    submissions_dir = tmp_path / "multi-submissions"
    write_submission_evidence(submissions_dir / "run-random")
    write_submission_evidence(submissions_dir / "run-sharegpt")
    env.pop("CURRENT_SUBMISSION_DIR")
    env["CURRENT_SUBMISSIONS_DIR"] = str(submissions_dir)
    env["FAKE_PUBLIC_VALIDATOR_EXIT"] = "2"
    remote_head = run(
        ["git", "--git-dir", str(remote), "rev-parse", "main"], tmp_path
    ).stdout.strip()

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 2, result.stdout + result.stderr
    assert (
        run(
            ["git", "--git-dir", str(remote), "rev-parse", "main"], tmp_path
        ).stdout.strip()
        == remote_head
    )
    assert not (benchmark_repo / "submissions" / "run-random").exists()
    assert not (benchmark_repo / "submissions" / "run-sharegpt").exists()


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


def test_expected_base_mismatch_rejects_before_publication(tmp_path):
    env, remote, benchmark_repo, github_env = prepare_sync_environment(
        tmp_path, "expected-base-mismatch"
    )
    remote_head = run(
        ["git", "--git-dir", str(remote), "rev-parse", "main"], tmp_path
    ).stdout.strip()
    env["SNAPSHOT_EXPECTED_BASE_SHA"] = "0" * 40

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 2, result.stdout + result.stderr
    assert "benchmark publication base moved" in result.stderr
    assert (
        run(
            ["git", "--git-dir", str(remote), "rev-parse", "main"], tmp_path
        ).stdout.strip()
        == remote_head
    )
    assert not (benchmark_repo / "submissions" / "expected-base-mismatch").exists()
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


@pytest.mark.parametrize(
    ("max_attempts", "expected_returncode", "expected_message"),
    [
        ("3", 0, "prepare fetch failed; retrying origin/main"),
        ("2", 1, "prepare fetch failed after 2 attempts"),
    ],
)
def test_prepare_fetch_retry_is_bounded(
    tmp_path, max_attempts, expected_returncode, expected_message
):
    env, _remote, benchmark_repo, github_env = prepare_sync_environment(
        tmp_path, "prepare-fetch-retry"
    )
    fake_git = write_flaky_fetch_git(tmp_path)
    fetch_count = tmp_path / "fetch-count"
    env.update(
        {
            "FAKE_GIT_FETCH_COUNT": str(fetch_count),
            "FAKE_GIT_FETCH_FAIL_CALLS": "1,2",
            "PATH": f"{fake_git.parent}:{env['PATH']}",
            "REAL_GIT": shutil.which("git") or "git",
            "SNAPSHOT_MAX_FETCH_ATTEMPTS": max_attempts,
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

    assert result.returncode == expected_returncode, result.stdout + result.stderr
    assert expected_message in result.stderr
    if expected_returncode == 0:
        assert fetch_count.read_text(encoding="utf-8").strip() == "4"
        assert "GITHUB_SNAPSHOT_SYNC_STATUS=pushed" in github_env.read_text(
            encoding="utf-8"
        )
    else:
        assert fetch_count.read_text(encoding="utf-8").strip() == "2"
        assert not (benchmark_repo / "submissions" / "prepare-fetch-retry").exists()
        assert "GITHUB_SNAPSHOT_SYNC_STATUS=rejected" in github_env.read_text(
            encoding="utf-8"
        )


def test_verify_fetch_retry_recovers_after_push(tmp_path):
    env, _remote, _benchmark_repo, github_env = prepare_sync_environment(
        tmp_path, "verify-fetch-retry"
    )
    fake_git = write_flaky_fetch_git(tmp_path)
    fetch_count = tmp_path / "fetch-count"
    env.update(
        {
            "FAKE_GIT_FETCH_COUNT": str(fetch_count),
            "FAKE_GIT_FETCH_FAIL_CALLS": "2,3",
            "PATH": f"{fake_git.parent}:{env['PATH']}",
            "REAL_GIT": shutil.which("git") or "git",
            "SNAPSHOT_MAX_FETCH_ATTEMPTS": "3",
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
    assert "verify fetch failed; retrying origin/main" in result.stderr
    assert fetch_count.read_text(encoding="utf-8").strip() == "4"
    assert "GITHUB_SNAPSHOT_SYNC_VERIFICATION=verified" in github_env.read_text(
        encoding="utf-8"
    )


def test_verify_fetch_retry_exhaustion_records_pushed_unverified_state(tmp_path):
    env, remote, _benchmark_repo, github_env = prepare_sync_environment(
        tmp_path, "verify-fetch-exhausted"
    )
    fake_git = write_flaky_fetch_git(tmp_path)
    fetch_count = tmp_path / "fetch-count"
    env.update(
        {
            "FAKE_GIT_FETCH_COUNT": str(fetch_count),
            "FAKE_GIT_FETCH_FAIL_CALLS": "2,3,4",
            "PATH": f"{fake_git.parent}:{env['PATH']}",
            "REAL_GIT": shutil.which("git") or "git",
            "SNAPSHOT_MAX_FETCH_ATTEMPTS": "3",
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

    assert result.returncode == 1, result.stdout + result.stderr
    assert "verify fetch failed after 3 attempts" in result.stderr
    assert "push succeeded, but verification failed" in result.stderr
    assert fetch_count.read_text(encoding="utf-8").strip() == "4"
    assert "GITHUB_SNAPSHOT_SYNC_STATUS=pushed" in env_text
    assert "GITHUB_SNAPSHOT_SYNC_COMMIT=" in env_text
    assert "GITHUB_SNAPSHOT_SYNC_VERIFICATION=failed" in env_text
    assert "GITHUB_SNAPSHOT_SYNC_VERIFIED_COMMIT=" not in env_text
    assert (
        run(
            [
                "git",
                "--git-dir",
                str(remote),
                "show",
                "main:submissions/verify-fetch-exhausted/STATUS",
            ],
            tmp_path,
        ).stdout.strip()
        == "OK"
    )


def test_verify_commit_mismatch_preserves_pushed_state(tmp_path):
    env, remote, _benchmark_repo, github_env = prepare_sync_environment(
        tmp_path, "verify-commit-mismatch"
    )
    fake_git = write_mismatched_verify_git(tmp_path)
    env.update(
        {
            "PATH": f"{fake_git.parent}:{env['PATH']}",
            "REAL_GIT": shutil.which("git") or "git",
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
    sync_statuses = [
        line
        for line in env_text.splitlines()
        if line.startswith("GITHUB_SNAPSHOT_SYNC_STATUS=")
    ]

    assert result.returncode == 1, result.stdout + result.stderr
    assert "benchmark publication verification failed: expected" in result.stderr
    assert "push succeeded, but verification failed" in result.stderr
    assert sync_statuses[-1] == "GITHUB_SNAPSHOT_SYNC_STATUS=pushed"
    assert "GITHUB_SNAPSHOT_SYNC_COMMIT=" in env_text
    assert "GITHUB_SNAPSHOT_SYNC_VERIFICATION=failed" in env_text
    assert "GITHUB_SNAPSHOT_SYNC_VERIFIED_COMMIT=" not in env_text
    assert (
        run(
            [
                "git",
                "--git-dir",
                str(remote),
                "show",
                "main:submissions/verify-commit-mismatch/STATUS",
            ],
            tmp_path,
        ).stdout.strip()
        == "OK"
    )


@pytest.mark.parametrize(
    ("variable", "value", "expected_message"),
    [
        (
            "SNAPSHOT_MAX_FETCH_ATTEMPTS",
            "0",
            "SNAPSHOT_MAX_FETCH_ATTEMPTS must be a positive integer",
        ),
        (
            "SNAPSHOT_MAX_FETCH_ATTEMPTS",
            "invalid",
            "SNAPSHOT_MAX_FETCH_ATTEMPTS must be a positive integer",
        ),
        (
            "SNAPSHOT_FETCH_RETRY_SECONDS",
            "-1",
            "SNAPSHOT_FETCH_RETRY_SECONDS must be a non-negative integer",
        ),
    ],
)
def test_invalid_fetch_retry_configuration_fails_closed(
    tmp_path, variable, value, expected_message
):
    env, _remote, benchmark_repo, github_env = prepare_sync_environment(
        tmp_path, "invalid-fetch-retry"
    )
    env[variable] = value

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 2, result.stdout + result.stderr
    assert expected_message in result.stderr
    assert not (benchmark_repo / "submissions" / "invalid-fetch-retry").exists()
    assert not github_env.exists()
