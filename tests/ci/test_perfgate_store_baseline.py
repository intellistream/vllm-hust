# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/scripts/perfgate_store_baseline.sh"
)
RUNTIME_MANAGER_SHA = "d" * 40
TARGET_SHA = "a" * 40


def supports_bash_case_conversion() -> bool:
    return (
        subprocess.run(
            ["bash", "-c", '[[ "${BASH_VERSINFO[0]}" -ge 4 ]]'],
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def run_store_baseline(
    tmp_path: Path, *, fetch_failures: int
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture_file = tmp_path / "publish-arguments.json"
    fetch_counter_file = tmp_path / "fetch-counter"

    write_executable(
        fake_bin / "git",
        f"""#!/bin/bash
set -euo pipefail
if [[ "$*" == *"fetch --quiet origin main:refs/remotes/origin/main"* ]]; then
  counter_file=${{FETCH_COUNTER_FILE:?}}
  count=0
  if [[ -f "$counter_file" ]]; then
    count=$(<"$counter_file")
  fi
  count=$((count + 1))
  printf '%s\\n' "$count" > "$counter_file"
  if [[ "$count" -le "${{FETCH_FAILURES:?}}" ]]; then
    echo "fatal: simulated target fetch failure" >&2
    exit 128
  fi
  exit 0
fi
if [[ "$*" == *"rev-parse origin/main"* ]]; then
  printf '%s\\n' '{TARGET_SHA}'
fi
""",
    )
    write_executable(
        fake_bin / "jq",
        f"""#!/bin/bash
set -euo pipefail
case "$*" in
  *raw_result_sha256*) printf 'test-sha256\\n' ;;
  *.same_spec.scenario*) printf 'random-online\\n' ;;
  *.same_spec.spec_id*) printf 'official-spec\\n' ;;
  *.same_spec.resolved_spec_hash*) printf '%s\\n' '{"c" * 64}' ;;
  *.metadata.github_repository*) printf 'vLLM-HUST/vllm-hust\\n' ;;
  *.metadata.git_commit*) printf '%s\\n' '{TARGET_SHA}' ;;
  *.vllm_hust_sha*) printf '%s\\n' '{"b" * 40}' ;;
  *.vllm_ascend_hust_sha*) printf '%s\\n' '{"c" * 40}' ;;
  *.benchmark_runner_sha*) printf '%s\\n' '{"e" * 40}' ;;
  *.runtime_manager_sha*) printf '%s\\n' '{RUNTIME_MANAGER_SHA}' ;;
  *.hardware_chip_model*) printf '910B2\\n' ;;
  *.cann_version*) printf '9.0.0\\n' ;;
  *.torch_version*) printf '2.10.0\\n' ;;
  *.torch_npu_version*) printf '2.10.0\\n' ;;
esac
""",
    )
    write_executable(
        fake_bin / "sha256sum",
        """#!/bin/bash
set -euo pipefail
printf 'test-sha256  %s\\n' "$1"
""",
    )
    write_executable(
        fake_bin / "python",
        f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path

Path(os.environ["CAPTURE_FILE"]).write_text(
    json.dumps(sys.argv[1:]), encoding="utf-8"
)
""",
    )

    result_root = tmp_path / "results"
    submission_dir = result_root / "submissions" / "run"
    submission_dir.mkdir(parents=True)
    for raw_result in (
        result_root / "runs/warmup-1/raw_benchmark_result.json",
        result_root / "runs/1/raw_benchmark_result.json",
        result_root / "runs/2/raw_benchmark_result.json",
        result_root / "runs/3/raw_benchmark_result.json",
    ):
        raw_result.parent.mkdir(parents=True, exist_ok=True)
        raw_result.write_text("{}\\n", encoding="utf-8")

    baseline_file = submission_dir / "run_leaderboard.json"
    baseline_file.write_text("{}\\n", encoding="utf-8")
    measurement_file = submission_dir / "measurement.json"
    measurement_file.write_text("{}\\n", encoding="utf-8")
    provenance_file = submission_dir / "perfgate-provenance.json"
    provenance_file.write_text("{}\\n", encoding="utf-8")

    benchmark_repo = tmp_path / "benchmark"
    (benchmark_repo / "src/vllm_hust_benchmark").mkdir(parents=True)
    env = {
        **os.environ,
        "CAPTURE_FILE": str(capture_file),
        "FETCH_COUNTER_FILE": str(fetch_counter_file),
        "FETCH_FAILURES": str(fetch_failures),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PERFGATE_BASELINE_WRITER_TOKEN": "test-token",
        "PERFGATE_TARGET_FETCH_INITIAL_DELAY_SECONDS": "0",
        "PERFGATE_TARGET_FETCH_MAX_ATTEMPTS": "4",
        "PERFGATE_TARGET_REPOSITORY": "vLLM-HUST/vllm-hust",
        "PERFGATE_TARGET_SHA": TARGET_SHA,
        "PERFGATE_TARGET_GIT_REPOSITORY": str(tmp_path),
        "PERFGATE_BENCHMARK_REPO_DIR": str(benchmark_repo),
        "PERFGATE_BASELINE_SOURCE_FILE": str(baseline_file),
        "PERFGATE_MEASUREMENT_FILE": str(measurement_file),
        "PERFGATE_PROVENANCE_FILE": str(provenance_file),
        "PYTHON_BIN": str(fake_bin / "python"),
        "RESULT_ROOT": str(result_root),
    }

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    return result, capture_file, fetch_counter_file


@pytest.mark.skipif(
    not supports_bash_case_conversion(),
    reason="perfgate_store_baseline.sh requires Bash 4+ case conversion",
)
def test_store_baseline_retries_target_fetch_and_forwards_provenance(
    tmp_path: Path,
) -> None:
    result, capture_file, fetch_counter_file = run_store_baseline(
        tmp_path, fetch_failures=2
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert fetch_counter_file.read_text(encoding="utf-8").strip() == "3"
    assert "target main fetch attempt 1/4" in result.stdout
    assert "target main fetch attempt 3/4" in result.stdout
    arguments = json.loads(capture_file.read_text(encoding="utf-8"))
    runtime_manager_index = arguments.index("--runtime-manager-sha")
    assert arguments[runtime_manager_index + 1] == RUNTIME_MANAGER_SHA


@pytest.mark.skipif(
    not supports_bash_case_conversion(),
    reason="perfgate_store_baseline.sh requires Bash 4+ case conversion",
)
def test_store_baseline_fails_after_target_fetch_retries(tmp_path: Path) -> None:
    result, capture_file, fetch_counter_file = run_store_baseline(
        tmp_path, fetch_failures=4
    )

    assert result.returncode != 0
    assert fetch_counter_file.read_text(encoding="utf-8").strip() == "4"
    assert not capture_file.exists()
    assert "target main fetch attempt 4/4" in result.stdout
    assert "Failed to fetch target main after 4 attempts" in result.stderr
