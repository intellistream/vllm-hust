# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / ".github/workflows/scripts/replay_benchmark_publication.sh"
SYNC_SCRIPT = (
    REPO_ROOT / ".github/workflows/scripts/sync_benchmark_snapshots_to_github.sh"
)


def run(
    command: list[str],
    cwd: Path,
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def init_repo(path: Path, files: dict[str, str]) -> str:
    path.mkdir(parents=True)
    run(["git", "init", "-b", "main"], path)
    run(["git", "config", "user.name", "Test"], path)
    run(["git", "config", "user.email", "test@example.com"], path)
    for relative_path, content in files.items():
        target = path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    run(["git", "add", "."], path)
    run(["git", "commit", "-m", "seed"], path)
    return run(["git", "rev-parse", "HEAD"], path).stdout.strip()


def write_artifact(
    root: Path,
    *,
    run_id: str,
    attempt: str,
    target_sha: str,
) -> Path:
    submission = (
        root
        / f"ci-{run_id}-{attempt}-{target_sha}"
        / "submissions"
        / f"ci-{run_id}-{attempt}-{target_sha}"
    )
    submission.mkdir(parents=True)
    files = {
        "leaderboard_manifest.json": json.dumps(
            {
                "schema_version": "leaderboard-export-manifest/v2",
                "entries": [],
            }
        )
        + "\n",
        "run_leaderboard.json": json.dumps(
            {
                "metadata": {
                    "git_commit": target_sha,
                    "github_repository": "vLLM-HUST/vllm-hust",
                    "runtime_provenance": {
                        "engine": {
                            "repository": "vLLM-HUST/vllm-hust",
                            "commit": target_sha,
                        }
                    },
                }
            }
        )
        + "\n",
        "env-manifest.json": json.dumps(
            {
                "git_info": {
                    "vllm_hust": {
                        "declared": target_sha,
                        "observed": target_sha,
                    }
                }
            }
        )
        + "\n",
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
    return submission


@pytest.fixture
def replay_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    run_id = "31004708110"
    attempt = "1"
    target_sha = "4a6f5b1ce78ace4b2b4d77229a9707a7f54ba5d0"
    artifact_root = tmp_path / "artifact"
    write_artifact(
        artifact_root,
        run_id=run_id,
        attempt=attempt,
        target_sha=target_sha,
    )

    seed = tmp_path / "benchmark-seed"
    benchmark_sha = init_repo(
        seed,
        {
            "README.md": "benchmark\n",
            "archive/README.md": "archive\n",
            "submissions/.gitkeep": "",
        },
    )
    remote = tmp_path / "benchmark-remote.git"
    run(["git", "init", "--bare", str(remote)], tmp_path)
    run(["git", "remote", "add", "origin", str(remote)], seed)
    run(["git", "push", "origin", "HEAD:main"], seed)
    run(["git", "symbolic-ref", "HEAD", "refs/heads/main"], remote)

    benchmark = tmp_path / "benchmark"
    run(["git", "clone", str(remote), str(benchmark)], tmp_path)
    website = tmp_path / "website"
    website_sha = init_repo(
        website,
        {"scripts/aggregate_results.py": "# fake\n"},
    )
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "$1" == "-m" && "$2" == "vllm_hust_benchmark.cli" ]]; then
  shift 3
  output_dir=""
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --output-dir) output_dir=$2; shift 2 ;;
      *) shift ;;
    esac
  done
  mkdir -p "$output_dir"
  printf '{}\\n' > "$output_dir/leaderboard_single.json"
  printf '{}\\n' > "$output_dir/leaderboard_multi.json"
  printf '{}\\n' > "$output_dir/leaderboard_compare.json"
  printf '{}\\n' > "$output_dir/last_updated.json"
  exit 0
fi
if [[ "$1" == "-" ]]; then
  cat >/dev/null
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "ALLOW_LOCAL_REPLAY_REMOTE": "1",
            "BENCHMARK_REPO_DIR": str(benchmark),
            "EXPECTED_BENCHMARK_MAIN_SHA": benchmark_sha,
            "EXPECTED_SYNC_SCRIPT_SHA256": hashlib.sha256(
                SYNC_SCRIPT.read_bytes()
            ).hexdigest(),
            "EXPECTED_TARGET_SHA": target_sha,
            "EXPECTED_WEBSITE_SHA": website_sha,
            "REPLAY_ARTIFACT_ROOT": str(artifact_root),
            "REPLAY_CONFIRMATION": f"publish-{run_id}-{attempt}",
            "REPLAY_RECEIPT_FILE": str(tmp_path / "receipt.env"),
            "RUNNER_TEMP": str(tmp_path / "runner-temp"),
            "PYTHON_BIN": str(fake_python),
            "POST_VALIDATION_PYTHON_BIN": str(fake_python),
            "SOURCE_RUN_ATTEMPT": attempt,
            "SOURCE_RUN_ID": run_id,
            "VLLM_HUST_REPO_DIR": str(REPO_ROOT),
            "WEBSITE_REPO_DIR": str(website),
        }
    )
    return env, remote


def test_preflight_accepts_exact_historical_artifact(
    replay_environment: tuple[dict[str, str], Path],
) -> None:
    env, _remote = replay_environment

    result = run(
        ["bash", str(SCRIPT_PATH), "preflight"],
        REPO_ROOT,
        env=env,
    )

    assert "Replay preflight passed" in result.stdout
    assert not Path(env["REPLAY_RECEIPT_FILE"]).exists()


def test_preflight_rejects_confirmation_before_publication(
    replay_environment: tuple[dict[str, str], Path],
) -> None:
    env, remote = replay_environment
    original_remote = run(
        ["git", "rev-parse", "refs/heads/main"], remote
    ).stdout.strip()
    env["REPLAY_CONFIRMATION"] = "publish-wrong"

    result = run(
        ["bash", str(SCRIPT_PATH), "preflight"],
        REPO_ROOT,
        env=env,
        check=False,
    )

    assert result.returncode == 2
    assert "REPLAY_CONFIRMATION must equal" in result.stderr
    assert (
        run(["git", "rev-parse", "refs/heads/main"], remote).stdout.strip()
        == original_remote
    )


def test_preflight_rejects_artifact_target_mismatch(
    replay_environment: tuple[dict[str, str], Path],
) -> None:
    env, _remote = replay_environment
    env_manifest = next(Path(env["REPLAY_ARTIFACT_ROOT"]).rglob("env-manifest.json"))
    payload = json.loads(env_manifest.read_text(encoding="utf-8"))
    payload["git_info"]["vllm_hust"]["observed"] = "0" * 40
    env_manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    submission = env_manifest.parent
    checksum_lines = []
    for name in (
        "leaderboard_manifest.json",
        "run_leaderboard.json",
        "env-manifest.json",
        "pip-packages.json",
    ):
        digest = hashlib.sha256((submission / name).read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  ./{name}\n")
    (submission / "checksums.sha256").write_text(
        "".join(checksum_lines), encoding="utf-8"
    )

    result = run(
        ["bash", str(SCRIPT_PATH), "preflight"],
        REPO_ROOT,
        env=env,
        check=False,
    )

    assert result.returncode != 0
    assert "env observed SHA mismatch" in result.stderr


def test_preflight_rejects_unsafe_checksum_manifest_path(
    replay_environment: tuple[dict[str, str], Path],
) -> None:
    env, _remote = replay_environment
    checksum_manifest = next(
        Path(env["REPLAY_ARTIFACT_ROOT"]).rglob("checksums.sha256")
    )
    checksum_manifest.write_text(
        checksum_manifest.read_text(encoding="utf-8")
        + f"{'0' * 64}  ./../../outside\n",
        encoding="utf-8",
    )

    result = run(
        ["bash", str(SCRIPT_PATH), "preflight"],
        REPO_ROOT,
        env=env,
        check=False,
    )

    assert result.returncode != 0
    assert "unsafe checksum manifest entry" in result.stderr


def test_preflight_rejects_benchmark_main_that_moved_after_rehearsal(
    replay_environment: tuple[dict[str, str], Path],
) -> None:
    env, remote = replay_environment
    seed = remote.parent / "benchmark-update"
    run(["git", "clone", str(remote), str(seed)], remote.parent)
    run(["git", "config", "user.name", "Test"], seed)
    run(["git", "config", "user.email", "test@example.com"], seed)
    (seed / "README.md").write_text("moved\n", encoding="utf-8")
    run(["git", "add", "README.md"], seed)
    run(["git", "commit", "-m", "move main"], seed)
    run(["git", "push", "origin", "HEAD:main"], seed)

    result = run(
        ["bash", str(SCRIPT_PATH), "preflight"],
        REPO_ROOT,
        env=env,
        check=False,
    )

    assert result.returncode == 2
    assert "benchmark main moved" in result.stderr
    assert not Path(env["REPLAY_RECEIPT_FILE"]).exists()


def test_publish_requires_writer_after_all_static_inputs_are_valid(
    replay_environment: tuple[dict[str, str], Path],
) -> None:
    env, _remote = replay_environment
    env.pop("REPLAY_WRITER_TOKEN", None)

    result = run(
        ["bash", str(SCRIPT_PATH), "publish"],
        REPO_ROOT,
        env=env,
        check=False,
    )

    assert result.returncode == 2
    assert "REPLAY_WRITER_TOKEN is required" in result.stderr


def test_publish_pushes_and_verifies_via_local_bare_remote(
    replay_environment: tuple[dict[str, str], Path],
) -> None:
    env, remote = replay_environment
    env["REPLAY_WRITER_TOKEN"] = "test-writer-token"
    benchmark = Path(env["BENCHMARK_REPO_DIR"])
    expected_base = env["EXPECTED_BENCHMARK_MAIN_SHA"]

    result = run(
        ["bash", str(SCRIPT_PATH), "publish"],
        REPO_ROOT,
        env=env,
    )

    assert "Historical benchmark publication replay verified" in result.stdout
    receipt = Path(env["REPLAY_RECEIPT_FILE"]).read_text(encoding="utf-8")
    assert "GITHUB_SNAPSHOT_SYNC_STATUS=pushed\n" in receipt
    assert "GITHUB_SNAPSHOT_SYNC_VERIFICATION=verified\n" in receipt
    assert "REPLAY_RESULT=verified\n" in receipt
    remote_head = run(["git", "rev-parse", "refs/heads/main"], remote).stdout.strip()
    assert remote_head != expected_base
    assert (
        run(["git", "rev-parse", f"{remote_head}^"], benchmark).stdout.strip()
        == expected_base
    )
    changed_paths = set(
        run(
            ["git", "diff", "--name-only", f"{expected_base}..{remote_head}"],
            benchmark,
        ).stdout.splitlines()
    )
    submission_prefix = (
        "submissions/ci-31004708110-1-4a6f5b1ce78ace4b2b4d77229a9707a7f54ba5d0/"
    )
    assert changed_paths
    assert all(
        path.startswith(submission_prefix)
        or path.startswith("leaderboard-data/snapshots/")
        for path in changed_paths
    )
    assert run(["git", "remote", "get-url", "origin"], benchmark).stdout.strip() == str(
        remote
    )


def test_publish_failure_cleans_askpass_and_preserves_remote(
    replay_environment: tuple[dict[str, str], Path],
) -> None:
    env, remote = replay_environment
    env["REPLAY_WRITER_TOKEN"] = "test-writer-token"
    original_remote = run(
        ["git", "rev-parse", "refs/heads/main"], remote
    ).stdout.strip()
    failing_python = Path(env["RUNNER_TEMP"]).parent / "failing-python"
    failing_python.write_text("#!/bin/bash\nexit 9\n", encoding="utf-8")
    failing_python.chmod(0o755)
    env["PYTHON_BIN"] = str(failing_python)

    result = run(
        ["bash", str(SCRIPT_PATH), "publish"],
        REPO_ROOT,
        env=env,
        check=False,
    )

    assert result.returncode == 9
    assert (
        run(["git", "rev-parse", "refs/heads/main"], remote).stdout.strip()
        == original_remote
    )
    runner_temp = Path(env["RUNNER_TEMP"])
    assert not list(runner_temp.glob("benchmark-replay-askpass.*"))
