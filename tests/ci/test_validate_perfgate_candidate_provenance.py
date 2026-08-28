# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/scripts/validate_perfgate_candidate_provenance.py"
)
TARGET_SHA = "1" * 40


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def provenance(**overrides: str) -> dict[str, str]:
    payload = {
        "schema_version": "perfgate-runtime-provenance/v1",
        "vllm_hust_sha": TARGET_SHA,
        "vllm_ascend_hust_sha": "2" * 40,
        "benchmark_runner_sha": "3" * 40,
        "runtime_manager_sha": "8" * 40,
        "hardware_chip_model": "910B2",
        "cann_version": "8.2.RC1",
        "torch_version": "2.7.1",
        "torch_npu_version": "2.7.1",
    }
    payload.update(overrides)
    return payload


def run_validator(
    tmp_path: Path, candidate: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    baseline = tmp_path / "baseline-metadata.json"
    candidate_file = tmp_path / "perfgate-provenance.json"
    baseline_provenance = provenance(vllm_hust_sha="4" * 40)
    baseline_provenance.pop("schema_version")
    write_json(baseline, {"provenance": baseline_provenance})
    write_json(candidate_file, candidate)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline-metadata",
            str(baseline),
            "--candidate-provenance",
            str(candidate_file),
            "--expected-target-sha",
            TARGET_SHA,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_candidate_runtime_provenance_matches_baseline(tmp_path: Path) -> None:
    result = run_validator(tmp_path, provenance())

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("overrides", "expected_error"),
    [
        ({"vllm_ascend_hust_sha": "5" * 40}, "vllm_ascend_hust_sha"),
        ({"benchmark_runner_sha": "6" * 40}, "benchmark_runner_sha"),
        ({"runtime_manager_sha": "not-a-sha"}, "runtime_manager_sha"),
        ({"torch_npu_version": "2.8.0"}, "torch_npu_version"),
        ({"vllm_hust_sha": "7" * 40}, "candidate Core SHA mismatch"),
    ],
)
def test_candidate_runtime_provenance_mismatch_fails_closed(
    tmp_path: Path,
    overrides: dict[str, str],
    expected_error: str,
) -> None:
    result = run_validator(tmp_path, provenance(**overrides))

    assert result.returncode == 2
    assert expected_error in result.stderr


def test_missing_runtime_manager_provenance_fails_closed(tmp_path: Path) -> None:
    candidate = provenance()
    candidate.pop("runtime_manager_sha")

    result = run_validator(tmp_path, candidate)

    assert result.returncode == 2
    assert "runtime_manager_sha" in result.stderr
