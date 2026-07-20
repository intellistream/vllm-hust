# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / ".github/workflows/scripts"
STAGE2_SCRIPT = SCRIPT_DIR / "perfgate_stage2_rebase_and_benchmark.sh"
FETCH_BASELINE_SCRIPT = SCRIPT_DIR / "perfgate_fetch_baseline.sh"


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
  ls-remote)
    exit "${FAKE_LS_REMOTE_RC:-0}"
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
  *install_ascend_benchmark_with_dev_hub.sh)
    exit 0
    ;;
  *run_ascend_benchmark_ci.sh)
    mkdir -p "${RESULT_ROOT}/submissions/${RUN_ID}"
    printf '{}\n' > "${RESULT_ROOT}/submissions/${RUN_ID}/run_leaderboard.json"
    exit 0
    ;;
  *perfgate_fetch_baseline.sh)
    if [[ "${FAKE_FETCH_AVAILABLE}" == "1" ]]; then
      printf '{}\n' > "${FAKE_BASELINE_FILE}"
      {
        echo "PERFGATE_BASELINE_AVAILABLE=1"
        echo "PERFGATE_BASELINE_FILE=${FAKE_BASELINE_FILE}"
        echo "PERFGATE_BASELINE_COMMIT=${FAKE_M2_COMMIT}"
        echo "PERFGATE_BASELINE_SOURCE=exact"
      } > "${GITHUB_ENV}"
    else
      {
        echo "PERFGATE_BASELINE_AVAILABLE=0"
        echo "PERFGATE_BASELINE_COMMIT=${FAKE_M2_COMMIT}"
        echo "PERFGATE_BASELINE_SOURCE=unavailable"
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
    result_root = tmp_path / "stage2-result"
    baseline_file = tmp_path / "m2-baseline.json"
    return {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PERFGATE_MODE": mode,
        "FORK_POINT": fork_point,
        "GITHUB_ENV": str(tmp_path / "github-env"),
        "GITHUB_WORKSPACE": str(REPO_ROOT),
        "RUNNER_TEMP": str(tmp_path),
        "PERFGATE_STAGE2_RESULT_ROOT": str(result_root),
        "PERFGATE_STAGE2_RUN_ID": "test-stage2",
        "FAKE_M2_COMMIT": "m2-commit",
        "FAKE_REBASE_RC": rebase_rc,
        "FAKE_FETCH_AVAILABLE": fetch_available,
        "FAKE_FETCH_RC": fetch_rc,
        "FAKE_BASELINE_FILE": str(baseline_file),
        "FAKE_BASH_LOG": str(tmp_path / "bash.log"),
        "PYTHON_BIN": "",
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


def test_stage2_revalidates_latest_main_in_enforce_mode(tmp_path: Path) -> None:
    env = _stage2_env(tmp_path)

    result = _run_stage2(env)

    assert result.returncode == 0
    assert "required revalidation" in result.stdout
    assert "run_ascend_benchmark_ci.sh" in Path(env["FAKE_BASH_LOG"]).read_text(
        encoding="utf-8"
    )
    github_env = Path(env["GITHUB_ENV"]).read_text(encoding="utf-8")
    assert "PERFGATE_STAGE2_EXECUTED" in github_env
    assert "PERFGATE_STAGE2_BASELINE_AVAILABLE" in github_env


def test_stage2_rebase_conflict_fails_only_in_enforce_mode(tmp_path: Path) -> None:
    enforce_env = _stage2_env(
        tmp_path / "enforce",
        fork_point="fork-point",
        rebase_rc="1",
    )
    report_env = _stage2_env(
        tmp_path / "report",
        mode="report",
        fork_point="fork-point",
        rebase_rc="1",
    )

    enforce_result = _run_stage2(enforce_env)
    report_result = _run_stage2(report_env)

    assert enforce_result.returncode == 2
    assert report_result.returncode == 0
    assert "rebase conflict recorded" in enforce_result.stdout


def test_stage2_missing_m2_baseline_preserves_reason_and_fails(
    tmp_path: Path,
) -> None:
    env = _stage2_env(
        tmp_path,
        fetch_available="0",
        fetch_rc="2",
    )

    result = _run_stage2(env)

    assert result.returncode == 2
    github_env = Path(env["GITHUB_ENV"]).read_text(encoding="utf-8")
    assert "PERFGATE_STAGE2_BASELINE_AVAILABLE" in github_env
    assert "No exact M2 baseline" in github_env


def test_fetch_baseline_preserves_reason_in_enforce_mode(tmp_path: Path) -> None:
    fake_bin = _prepare_fake_commands(tmp_path)
    github_env = tmp_path / "fetch-github-env"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PERFGATE_MODE": "enforce",
        "FORK_POINT": "missing-commit",
        "GITHUB_ENV": str(github_env),
        "FAKE_LS_REMOTE_RC": "1",
        "FAKE_M2_COMMIT": "m2-commit",
        "FAKE_FETCH_AVAILABLE": "0",
        "FAKE_BASH_LOG": str(tmp_path / "bash.log"),
    }

    result = subprocess.run(
        ["/bin/bash", str(FETCH_BASELINE_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )

    assert result.returncode == 2
    written_env = github_env.read_text(encoding="utf-8")
    assert "PERFGATE_BASELINE_AVAILABLE" in written_env
    assert "Perfgate baseline branch not found" in written_env


def _run_compare_with_fake_python(
    tmp_path: Path,
    report: str,
    python_rc: str,
    extra_env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    fake_python = tmp_path / "fake-python"
    _write_executable(
        fake_python,
        f"""#!/bin/bash
set -euo pipefail
report_file=""
while (( $# > 0 )); do
  if [[ "$1" == "--report-file" ]]; then
    report_file=$2
    break
  fi
  shift
done
printf '%s\\n' '{report}' > "$report_file"
exit {python_rc}
""",
    )
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text("{}\n", encoding="utf-8")
    current.write_text("{}\n", encoding="utf-8")
    env = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "PERFGATE_MODE": "enforce",
        "PERFGATE_BASELINE_AVAILABLE": "1",
        "PERFGATE_BASELINE_FILE": str(baseline),
        "PERFGATE_STAGE1_CURRENT_FILE": str(current),
        "PERFGATE_REPORT_FILE": str(tmp_path / "report.md"),
        "GITHUB_ENV": str(tmp_path / "github-env"),
        **extra_env,
    }
    return subprocess.run(
        ["/bin/bash", str(SCRIPT_DIR / "perfgate_compare.sh")],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )


def test_final_compare_reports_rebase_conflict_and_fails(tmp_path: Path) -> None:
    conflict_file = tmp_path / "rebase-conflict.txt"
    conflict_file.write_text("CONFLICT (content): workflow.yml\n", encoding="utf-8")

    result = _run_compare_with_fake_python(
        tmp_path,
        "**Overall: FAIL**\n**Stage 2: FAIL**",
        "1",
        {
            "PERFGATE_STAGE2_REBASE_CONFLICT": "1",
            "PERFGATE_STAGE2_REBASE_CONFLICT_FILE": str(conflict_file),
        },
    )

    assert result.returncode == 1
    github_env = Path(tmp_path / "github-env").read_text(encoding="utf-8")
    assert "PERFGATE_STAGE2_COMPLETED" in github_env
    assert "PERFGATE_STAGE2_RESULT" in github_env


def test_final_compare_reports_stage2_not_run_and_fails(tmp_path: Path) -> None:
    result = _run_compare_with_fake_python(
        tmp_path,
        "**Overall: FAIL**\n**Stage 2: NOT RUN**",
        "1",
        {"PERFGATE_STAGE2_NOT_RUN_REASON": "Stage 1 did not pass"},
    )

    assert result.returncode == 1
    github_env = Path(tmp_path / "github-env").read_text(encoding="utf-8")
    assert "PERFGATE_STAGE2_COMPLETED" in github_env
    assert "PERFGATE_STAGE2_RESULT" in github_env
