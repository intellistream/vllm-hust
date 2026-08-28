# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_SCRIPT = (
    REPO_ROOT / ".github/workflows/scripts/perfgate_stage2_rebase_and_benchmark.sh"
)


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _prepare_fake_commands(tmp_path: Path) -> Path:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir(parents=True)
    _write_executable(
        fake_bin / "git",
        """#!/bin/bash
set -euo pipefail
case "${1:-}" in
  rev-parse)
    if [[ "${2:-}" == "--verify" ]]; then
      exit 1
    fi
    if [[ "${2:-}" == "HEAD" ]]; then
      echo "original-ref"
    else
      echo "${FAKE_M2_COMMIT}"
    fi
    ;;
  rebase)
    exit "${FAKE_REBASE_RC:-0}"
    ;;
  diff)
    echo "conflicting-file.py"
    ;;
  *)
    exit 0
    ;;
esac
""",
    )
    _write_executable(
        fake_bin / "bash",
        """#!/bin/bash
set -euo pipefail
target=${1:-}
printf '%s\n' "$target" >> "${FAKE_BASH_LOG}"
case "$target" in
  *run_ascend_benchmark_ci.sh)
    submission_dir="${RESULT_ROOT}/submissions/${RUN_ID}"
    mkdir -p "$submission_dir"
    printf '{}\n' > "$submission_dir/run_leaderboard.json"
    printf '{}\n' > "$submission_dir/perfgate-provenance.json"
    ;;
  *perfgate_fetch_baseline.sh)
    if [[ "${FAKE_FETCH_AVAILABLE}" == "1" ]]; then
      {
        echo "PERFGATE_BASELINE_AVAILABLE=1"
        echo "PERFGATE_BASELINE_FILE=${FAKE_BASELINE_FILE}"
        echo "PERFGATE_BASELINE_METADATA_FILE=${FAKE_METADATA_FILE}"
        echo "PERFGATE_BASELINE_COMMIT=${FAKE_M2_COMMIT}"
        echo "PERFGATE_BASELINE_SOURCE=central-exact"
      } > "${GITHUB_ENV}"
    else
      {
        echo "PERFGATE_BASELINE_AVAILABLE=0"
        echo "PERFGATE_BASELINE_UNAVAILABLE_REASON=No exact M2 baseline"
      } > "${GITHUB_ENV}"
    fi
    exit "${FAKE_FETCH_RC:-0}"
    ;;
  *)
    echo "Unexpected bash target: $target" >&2
    exit 99
    ;;
esac
""",
    )
    _write_executable(
        fake_bin / "python",
        """#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == "-m" && "${2:-}" == "pip" ]]; then
  exit 0
fi
echo "candidate provenance valid"
""",
    )
    return fake_bin


def _stage2_env(
    tmp_path: Path,
    *,
    mode: str = "enforce",
    fork_point: str = "m2-commit",
    rebase_rc: str = "0",
    fetch_available: str = "1",
    fetch_rc: str = "0",
) -> dict[str, str]:
    fake_bin = _prepare_fake_commands(tmp_path)
    return {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PYTHON_BIN": str(fake_bin / "python"),
        "PERFGATE_MODE": mode,
        "FORK_POINT": fork_point,
        "GITHUB_ENV": str(tmp_path / "github-env"),
        "GITHUB_WORKSPACE": str(REPO_ROOT),
        "RUNNER_TEMP": str(tmp_path),
        "PERFGATE_STAGE2_RESULT_ROOT": str(tmp_path / "stage2-result"),
        "PERFGATE_STAGE2_RUN_ID": "test-stage2",
        "FAKE_M2_COMMIT": "m2-commit",
        "FAKE_REBASE_RC": rebase_rc,
        "FAKE_FETCH_AVAILABLE": fetch_available,
        "FAKE_FETCH_RC": fetch_rc,
        "FAKE_BASELINE_FILE": str(tmp_path / "m2-baseline.json"),
        "FAKE_METADATA_FILE": str(tmp_path / "m2-metadata.json"),
        "FAKE_BASH_LOG": str(tmp_path / "bash.log"),
    }


def _run_stage2(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(STAGE2_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )


def test_enforce_revalidates_when_fork_point_is_latest_main(tmp_path: Path) -> None:
    env = _stage2_env(tmp_path)

    result = _run_stage2(env)

    assert result.returncode == 0
    assert "required revalidation" in result.stdout
    assert "run_ascend_benchmark_ci.sh" in Path(env["FAKE_BASH_LOG"]).read_text()
    github_env = Path(env["GITHUB_ENV"]).read_text()
    assert "PERFGATE_STAGE2_EXECUTED" in github_env
    assert "PERFGATE_STAGE2_BASELINE_AVAILABLE" in github_env


def test_report_mode_can_skip_when_fork_point_is_latest_main(tmp_path: Path) -> None:
    env = _stage2_env(tmp_path, mode="report")

    result = _run_stage2(env)

    assert result.returncode == 0
    assert "Stage 2 skipped" in result.stdout
    github_env = Path(env["GITHUB_ENV"]).read_text()
    assert "PERFGATE_STAGE2_SKIPPED" in github_env


def test_enforce_rebase_conflict_fails(tmp_path: Path) -> None:
    env = _stage2_env(tmp_path, fork_point="fork-point", rebase_rc="1")

    result = _run_stage2(env)

    assert result.returncode == 2
    assert "rebase conflict recorded" in result.stdout


def test_enforce_missing_m2_baseline_preserves_reason(tmp_path: Path) -> None:
    env = _stage2_env(tmp_path, fetch_available="0", fetch_rc="2")

    result = _run_stage2(env)

    assert result.returncode == 2
    github_env = Path(env["GITHUB_ENV"]).read_text()
    assert "PERFGATE_STAGE2_BASELINE_AVAILABLE" in github_env
    assert "No exact M2 baseline" in github_env
