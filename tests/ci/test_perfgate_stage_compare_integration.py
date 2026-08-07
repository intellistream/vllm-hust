# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration coverage for the shell-level Perfgate provenance gates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE1_SCRIPT = REPO_ROOT / ".github/workflows/scripts/perfgate_stage1_compare.sh"
STAGE2_SCRIPT = REPO_ROOT / (
    ".github/workflows/scripts/perfgate_stage2_rebase_and_benchmark.sh"
)
COMPARE_SCRIPT = REPO_ROOT / ".github/workflows/scripts/perfgate_compare.sh"
TARGET_SHA = "1" * 40


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def baseline_provenance() -> dict[str, str]:
    return {
        "vllm_hust_sha": "4" * 40,
        "vllm_ascend_hust_sha": "2" * 40,
        "benchmark_runner_sha": "3" * 40,
        "runtime_manager_sha": "8" * 40,
        "hardware_chip_model": "910B2",
        "cann_version": "8.2.RC1",
        "torch_version": "2.7.1",
        "torch_npu_version": "2.7.1",
    }


def candidate_provenance(**overrides: str) -> dict[str, str]:
    payload = {
        "schema_version": "perfgate-runtime-provenance/v1",
        **baseline_provenance(),
        "vllm_hust_sha": TARGET_SHA,
    }
    payload.update(overrides)
    return payload


def make_python_stub(path: Path) -> None:
    """Run the real validator, but fake the benchmark compare module."""
    write_executable(
        path,
        """#!/bin/bash
set -euo pipefail
if [[ "$1" == "-m" ]]; then
  report=""
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--report-file" ]]; then report="$2"; break; fi
    shift
  done
  mkdir -p "$(dirname "$report")"
  printf '## Performance Gate Report\\n\\n**Overall: PASS**\\n' > "$report"
  exit 0
fi
exec "${REAL_PYTHON:-python3}" "$@"
""",
    )


def env_value(env_file: Path, key: str) -> str:
    lines = env_file.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1]
        if line.startswith(f"{key}<<"):
            delimiter = line.split("<<", 1)[1]
            values: list[str] = []
            for value in lines[index + 1 :]:
                if value == delimiter:
                    return "\n".join(values)
                values.append(value)
    return ""


def stage1_fixture(
    tmp_path: Path,
    *,
    metadata: bool = True,
    candidate_overrides: dict[str, str] | None = None,
) -> tuple[dict[str, str], Path]:
    result_root = tmp_path / "results"
    submission = result_root / "submissions" / "run"
    submission.mkdir(parents=True)
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text("{}\n", encoding="utf-8")
    current_file = submission / "run_leaderboard.json"
    current_file.write_text("{}\n", encoding="utf-8")
    candidate_file = submission / "perfgate-provenance.json"
    write_json(candidate_file, candidate_provenance(**(candidate_overrides or {})))
    env_file = tmp_path / "github-env"
    env_file.touch()
    python_stub = tmp_path / "python-stub"
    make_python_stub(python_stub)
    env = {
        **os.environ,
        "GITHUB_ENV": str(env_file),
        "GITHUB_WORKSPACE": str(REPO_ROOT),
        "RESULT_ROOT": str(result_root),
        "RUN_ID": "run",
        "TARGET_REPO_SHA": TARGET_SHA,
        "PERFGATE_BASELINE_AVAILABLE": "1",
        "PERFGATE_BASELINE_FILE": str(baseline_file),
        "PERFGATE_STAGE1_CURRENT_FILE": str(current_file),
        "PERFGATE_STAGE1_PROVENANCE_FILE": str(candidate_file),
        "PYTHON_BIN": str(python_stub),
        "REAL_PYTHON": sys.executable,
    }
    if metadata:
        metadata_file = tmp_path / "baseline-metadata.json"
        write_json(metadata_file, {"provenance": baseline_provenance()})
        env["PERFGATE_BASELINE_METADATA_FILE"] = str(metadata_file)
    return env, env_file


def run_script(script: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_stage1_matching_provenance_runs_comparison(tmp_path: Path) -> None:
    env, env_file = stage1_fixture(tmp_path)
    result = run_script(STAGE1_SCRIPT, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert env_value(env_file, "PERFGATE_STAGE1_PROVENANCE_VALID") == "1"
    report = Path(env_value(env_file, "PERFGATE_REPORT_FILE")).read_text()
    assert "**Overall: PASS**" in report


@pytest.mark.parametrize(
    ("candidate_overrides", "metadata"),
    [
        ({"runtime_manager_sha": "not-a-sha"}, True),
        (None, False),
    ],
)
def test_stage1_provenance_failure_skips_comparison(
    tmp_path: Path,
    candidate_overrides: dict[str, str] | None,
    metadata: bool,
) -> None:
    env, env_file = stage1_fixture(
        tmp_path,
        metadata=metadata,
        candidate_overrides=candidate_overrides,
    )
    result = run_script(STAGE1_SCRIPT, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert env_value(env_file, "PERFGATE_STAGE1_PROVENANCE_VALID") == "0"
    report = Path(env_value(env_file, "PERFGATE_REPORT_FILE")).read_text()
    assert "**Overall: FAIL**" in report
    assert "Stage 2 was not run." in report


@pytest.mark.parametrize(("mode", "expected_rc"), [("report", 0), ("enforce", 2)])
def test_compare_provenance_failure_honors_report_and_enforce_modes(
    tmp_path: Path, mode: str, expected_rc: int
) -> None:
    env_file = tmp_path / "github-env"
    report_file = tmp_path / "report.md"
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text("{}\n", encoding="utf-8")
    env = {
        **os.environ,
        "GITHUB_ENV": str(env_file),
        "PERFGATE_MODE": mode,
        "PERFGATE_REPORT_FILE": str(report_file),
        "PERFGATE_BASELINE_AVAILABLE": "1",
        "PERFGATE_BASELINE_FILE": str(baseline_file),
        "PERFGATE_STAGE1_PROVENANCE_VALID": "0",
        "PERFGATE_STAGE1_PROVENANCE_FAILURE_REASON": "fixture mismatch",
    }
    result = run_script(COMPARE_SCRIPT, env)

    assert result.returncode == expected_rc, result.stdout + result.stderr
    report = report_file.read_text(encoding="utf-8")
    assert "Stage 1: NOT RUN" in report
    assert "Stage 2: NOT RUN" in report
    assert "**Overall: FAIL**" in report


def make_stage2_bash_stub(path: Path, candidate: Path, metadata: Path) -> None:
    write_executable(
        path,
        f"""#!/bin/bash
set -euo pipefail
script="${{1:-}}"
if [[ "$script" == *.github/workflows/scripts/run_ascend_benchmark_ci.sh ]]; then
  destination="$RESULT_ROOT/submissions/$RUN_ID"
  mkdir -p "$destination"
  cp "{candidate}" "$destination/perfgate-provenance.json"
  if [[ -z "${{BENCHMARK_FINALIZER_SCRIPT:-}}" ]]; then
    printf '{{}}\\n' > "$destination/run_leaderboard.json"
    exit 0
  fi
  cat > "$destination/run_leaderboard.json" <<'JSON'
{{"entry_id":"stage2-fixture","engine":"vllm-hust","engine_version":"test","config_type":"single_gpu","hardware":{{"chip_model":"910B2","chip_count":1}},"model":{{"canonical_id":"hf:Qwen/Qwen2.5-14B-Instruct","precision":"FP16"}},"workload":{{"name":"random-online"}},"metadata":{{"submitted_at":"2026-08-01T00:00:00Z"}}}}
JSON
  cat > "$destination/leaderboard_manifest.json" <<'JSON'
{{"schema_version":"leaderboard-export-manifest/v2","generated_at":"2026-08-01T00:00:00Z","entries":[{{"leaderboard_artifact":"run_leaderboard.json"}}]}}
JSON
  /bin/bash "$BENCHMARK_FINALIZER_SCRIPT" "$destination"
  case "${{STAGE2_ARTIFACT_MUTATION:-}}" in
    missing-manifest)
      rm "$destination/leaderboard_manifest.json"
      ;;
    missing-checksum-entry)
      grep -v 'env-manifest.json' "$destination/checksums.sha256" \
        > "$destination/checksums.sha256.tmp"
      mv "$destination/checksums.sha256.tmp" "$destination/checksums.sha256"
      ;;
    tamper)
      printf '%s\\n' '{{"tampered":true}}' > "$destination/run_leaderboard.json"
      ;;
  esac
  /bin/bash "$BENCHMARK_VALIDATOR_SCRIPT" "$destination"
  PYTHONPATH="$BENCHMARK_REPO_ROOT/src${{PYTHONPATH:+:$PYTHONPATH}}" \
    "$BENCHMARK_ADMISSION_PYTHON" - "$destination" <<'PY'
import sys
from pathlib import Path

from vllm_hust_benchmark.integration import _scan_submission_admission_failures

failures = _scan_submission_admission_failures(Path(sys.argv[1]).parent)
if failures:
    print(f"publication admission failed: {{failures}}", file=sys.stderr)
    raise SystemExit(1)
PY
  exit 0
fi
if [[ "$script" == *.github/workflows/scripts/perfgate_fetch_baseline.sh ]]; then
  printf 'PERFGATE_BASELINE_AVAILABLE=1\\n' >> "$GITHUB_ENV"
  printf 'PERFGATE_BASELINE_FILE={metadata}.baseline\\n' >> "$GITHUB_ENV"
  printf 'PERFGATE_BASELINE_METADATA_FILE={metadata}\\n' >> "$GITHUB_ENV"
  printf 'PERFGATE_BASELINE_COMMIT=%s\\n' "${{1:-}}" >> "$GITHUB_ENV"
  printf 'PERFGATE_BASELINE_SOURCE=fixture\\n' >> "$GITHUB_ENV"
  printf '{{}}\\n' > "{metadata}.baseline"
  exit 0
fi
exec /bin/bash "$@"
""",
    )


def make_stage2_git_stub(path: Path) -> None:
    write_executable(
        path,
        """#!/bin/bash
set -euo pipefail
if [[ "$1" == "fetch" ]]; then exit 0; fi
if [[ "$1" == "rebase" ]]; then exit 0; fi
if [[ "$1" == "checkout" ]]; then exit 0; fi
if [[ "$1" == "branch" && "${2:-}" == "-D" ]]; then exit 0; fi
if [[ "$1" == "rev-parse" && "${2:-}" == "HEAD" && -n "${TEST_TARGET_SHA:-}" ]]; then
  printf '%s\n' "$TEST_TARGET_SHA"
  exit 0
fi
if [[ "$1" == "rev-parse" && "${2:-}" == "origin/main" ]]; then
  printf '%s\n' "$TEST_TARGET_SHA"
  exit 0
fi
exec /usr/bin/git "$@"
""",
    )


def artifact_scripts() -> tuple[Path, Path, Path]:
    benchmark_repo = os.environ.get(
        "VLLM_HUST_BENCHMARK_REPO", str(REPO_ROOT / "vllm-hust-benchmark")
    )
    root = Path(benchmark_repo)
    return (
        root,
        root / "scripts/collect-run-artifact.sh",
        root / "scripts/validate-run-artifact.sh",
    )


@pytest.mark.parametrize("mismatch", [False, True])
def test_stage2_provenance_match_or_mismatch_is_recorded(
    tmp_path: Path, mismatch: bool
) -> None:
    target_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    metadata = tmp_path / "baseline-metadata.json"
    write_json(metadata, {"provenance": baseline_provenance()})
    candidate = tmp_path / "candidate.json"
    write_json(
        candidate,
        candidate_provenance(
            vllm_hust_sha=target_sha,
            **({"runtime_manager_sha": "not-a-sha"} if mismatch else {}),
        ),
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    make_stage2_bash_stub(fake_bin / "bash", candidate, metadata)
    make_stage2_git_stub(fake_bin / "git")
    write_executable(
        fake_bin / "python",
        f'#!/bin/bash\nexec {sys.executable} "$@"\n',
    )
    env_file = tmp_path / "github-env"
    env_file.touch()
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "GITHUB_ENV": str(env_file),
        "GITHUB_WORKSPACE": str(REPO_ROOT),
        "RUN_ID": "run",
        "PERFGATE_STAGE2_RUN_ID": "stage2-run",
        "PERFGATE_STAGE2_RESULT_ROOT": str(tmp_path / "stage2-results"),
        "FORK_POINT": "0" * 40,
        "TEST_TARGET_SHA": target_sha,
        "PYTHON_BIN": "",
        "NODE_ENV_RETRY_MAX_ATTEMPTS": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    result = run_script(STAGE2_SCRIPT, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert env_value(env_file, "PERFGATE_STAGE2_PROVENANCE_VALID") == (
        "0" if mismatch else "1"
    )


def test_stage2_provenance_failure_reaches_enforced_final_compare(
    tmp_path: Path,
) -> None:
    """A Stage 2 provenance failure must remain a failed enforce result."""
    target_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    metadata = tmp_path / "baseline-metadata.json"
    write_json(metadata, {"provenance": baseline_provenance()})
    candidate = tmp_path / "candidate.json"
    write_json(
        candidate,
        candidate_provenance(
            vllm_hust_sha=target_sha,
            runtime_manager_sha="not-a-sha",
        ),
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    make_stage2_bash_stub(fake_bin / "bash", candidate, metadata)
    make_stage2_git_stub(fake_bin / "git")
    write_executable(
        fake_bin / "python",
        f'#!/bin/bash\nexec {sys.executable} "$@"\n',
    )
    stage2_env_file = tmp_path / "stage2-github-env"
    stage2_env_file.touch()
    stage2_env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "GITHUB_ENV": str(stage2_env_file),
        "GITHUB_WORKSPACE": str(REPO_ROOT),
        "RUN_ID": "run",
        "PERFGATE_STAGE2_RUN_ID": "stage2-run",
        "PERFGATE_STAGE2_RESULT_ROOT": str(tmp_path / "stage2-results"),
        "FORK_POINT": "0" * 40,
        "TEST_TARGET_SHA": target_sha,
        "PYTHON_BIN": "",
        "NODE_ENV_RETRY_MAX_ATTEMPTS": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    stage2_result = run_script(STAGE2_SCRIPT, stage2_env)

    assert stage2_result.returncode == 0, stage2_result.stdout + stage2_result.stderr
    assert env_value(stage2_env_file, "PERFGATE_STAGE2_PROVENANCE_VALID") == "0"
    reason = env_value(stage2_env_file, "PERFGATE_STAGE2_NOT_RUN_REASON")
    assert "candidate provenance validation failed" in reason
    assert env_value(stage2_env_file, "PERFGATE_STAGE2_B1PRIME_FILE") == ""

    def write_result(path: Path) -> None:
        write_json(
            path,
            {
                "engine": "vllm-hust",
                "metrics": {
                    "throughput_tps": 100.0,
                    "ttft_ms": 50.0,
                    "tbt_ms": 10.0,
                },
                "same_spec": {
                    "spec_id": "perfgate-ascend-qwen25-3b-910b3",
                    "resolved_spec_hash": "abc123",
                },
            },
        )

    stage1_baseline = tmp_path / "stage1-baseline.json"
    stage1_current = tmp_path / "stage1-current.json"
    write_result(stage1_baseline)
    write_result(stage1_current)
    report_file = tmp_path / "final-report.md"
    compare_env_file = tmp_path / "compare-github-env"
    benchmark_repo = Path(
        os.environ.get(
            "VLLM_HUST_BENCHMARK_REPO", str(REPO_ROOT / "vllm-hust-benchmark")
        )
    )
    compare_env = {
        **os.environ,
        "PYTHONPATH": str(benchmark_repo / "src"),
        "PYTHON_BIN": sys.executable,
        "GITHUB_ENV": str(compare_env_file),
        "PERFGATE_MODE": "enforce",
        "PERFGATE_REPORT_FILE": str(report_file),
        "PERFGATE_BASELINE_AVAILABLE": "1",
        "PERFGATE_BASELINE_FILE": str(stage1_baseline),
        "PERFGATE_STAGE1_CURRENT_FILE": str(stage1_current),
        "PERFGATE_STAGE1_PROVENANCE_VALID": "1",
        "PERFGATE_STAGE2_PROVENANCE_VALID": "0",
        "PERFGATE_STAGE2_NOT_RUN_REASON": reason,
        "PERFGATE_M2_COMMIT": "m2-commit",
        "FORK_POINT": "0" * 40,
    }
    compare_result = run_script(COMPARE_SCRIPT, compare_env)

    assert compare_result.returncode == 1, compare_result.stdout + compare_result.stderr
    report = report_file.read_text(encoding="utf-8")
    assert "Stage 2: NOT RUN" in report
    assert "candidate provenance validation failed" in report
    assert "**Overall: FAIL**" in report


@pytest.mark.parametrize(
    "mutation", [None, "missing-manifest", "missing-checksum-entry", "tamper"]
)
def test_stage2_artifact_finalize_and_admission_contract(
    tmp_path: Path, mutation: str | None
) -> None:
    """Evidence must be finalized before the Stage 2 admission decision."""
    benchmark_repo, finalizer, validator = artifact_scripts()
    if not finalizer.is_file() or not validator.is_file():
        pytest.fail("benchmark artifact finalizer and validator must exist")

    target_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    metadata = tmp_path / "baseline-metadata.json"
    write_json(metadata, {"provenance": baseline_provenance()})
    candidate = tmp_path / "candidate.json"
    write_json(candidate, candidate_provenance(vllm_hust_sha=target_sha))
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    make_stage2_bash_stub(fake_bin / "bash", candidate, metadata)
    make_stage2_git_stub(fake_bin / "git")
    write_executable(
        fake_bin / "python",
        f'#!/bin/bash\nexec {sys.executable} "$@"\n',
    )
    env_file = tmp_path / "github-env"
    env_file.touch()
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "GITHUB_ENV": str(env_file),
        "GITHUB_WORKSPACE": str(REPO_ROOT),
        "RUN_ID": "run",
        "PERFGATE_STAGE2_RUN_ID": "stage2-run",
        "PERFGATE_STAGE2_RESULT_ROOT": str(tmp_path / "stage2-results"),
        "FORK_POINT": "0" * 40,
        "TEST_TARGET_SHA": target_sha,
        "STAGE2_ARTIFACT_MUTATION": mutation or "",
        "BENCHMARK_FINALIZER_SCRIPT": str(finalizer),
        "BENCHMARK_VALIDATOR_SCRIPT": str(validator),
        "BENCHMARK_REPO_ROOT": str(Path(benchmark_repo)),
        "BENCHMARK_ADMISSION_PYTHON": sys.executable,
        "CURRENT_RUNTIME_PYTHON": sys.executable,
        "PYTHON_BIN": "",
        "NODE_ENV_RETRY_MAX_ATTEMPTS": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    result = run_script(STAGE2_SCRIPT, env)

    if mutation is None:
        assert result.returncode == 0, result.stdout + result.stderr
        artifact = tmp_path / "stage2-results" / "submissions" / "stage2-run"
        assert (artifact / "env-manifest.json").is_file()
        assert (artifact / "checksums.sha256").is_file()
    else:
        assert result.returncode != 0, result.stdout + result.stderr
