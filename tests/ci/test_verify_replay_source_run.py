# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import copy
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/scripts/verify_replay_source_run.py"
)
SOURCE_RUN_ID = 31004708110
SOURCE_RUN_ATTEMPT = 1
SOURCE_JOB_ID = 92301693209
TARGET_SHA = "4a6f5b1ce78ace4b2b4d77229a9707a7f54ba5d0"
ARTIFACT_NAME = "ascend-benchmark-31004708110-1"
EXPECTED_STEPS = {
    30: ("Run perfgate baseline producer benchmark", "success"),
    31: ("Upload perfgate producer artifact", "success"),
    32: ("Sanitize runner before central baseline publication", "success"),
    33: ("Publish central perfgate baseline", "success"),
    34: ("Run benchmark CI and optional formal publish", "failure"),
    38: ("Cleanup leftover Ascend CI processes", "success"),
    39: ("Release Ascend hardware lock", "success"),
    40: ("Build benchmark summary artifacts", "success"),
    41: ("Upload benchmark artifacts", "success"),
}


@pytest.fixture
def source_payloads() -> tuple[dict, dict, dict]:
    run = {
        "id": SOURCE_RUN_ID,
        "path": ".github/workflows/ascend-benchmark-leaderboard.yml",
        "event": "push",
        "head_branch": "main",
        "head_sha": TARGET_SHA,
        "run_attempt": SOURCE_RUN_ATTEMPT,
        "status": "completed",
        "conclusion": "failure",
    }
    steps = [
        {
            "number": number,
            "name": name,
            "status": "completed",
            "conclusion": conclusion,
        }
        for number, (name, conclusion) in EXPECTED_STEPS.items()
    ]
    jobs = {
        "total_count": 1,
        "jobs": [
            {
                "id": SOURCE_JOB_ID,
                "run_id": SOURCE_RUN_ID,
                "run_attempt": SOURCE_RUN_ATTEMPT,
                "head_sha": TARGET_SHA,
                "name": "ascend-benchmark",
                "status": "completed",
                "conclusion": "failure",
                "steps": steps,
            }
        ],
    }
    artifacts = {
        "total_count": 1,
        "artifacts": [
            {
                "name": ARTIFACT_NAME,
                "expired": False,
                "workflow_run": {
                    "id": SOURCE_RUN_ID,
                    "head_branch": "main",
                    "head_sha": TARGET_SHA,
                },
            }
        ],
    }
    return run, jobs, artifacts


def run_verifier(
    tmp_path: Path,
    payloads: tuple[dict, dict, dict],
) -> subprocess.CompletedProcess[str]:
    paths = []
    for name, payload in zip(("run", "jobs", "artifacts"), payloads, strict=True):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--run-json",
            str(paths[0]),
            "--jobs-json",
            str(paths[1]),
            "--artifacts-json",
            str(paths[2]),
            "--source-run-id",
            str(SOURCE_RUN_ID),
            "--source-run-attempt",
            str(SOURCE_RUN_ATTEMPT),
            "--source-job-id",
            str(SOURCE_JOB_ID),
            "--expected-target-sha",
            TARGET_SHA,
            "--expected-artifact-name",
            ARTIFACT_NAME,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_accepts_exact_failed_run_with_trusted_artifact_steps(
    tmp_path: Path,
    source_payloads: tuple[dict, dict, dict],
) -> None:
    result = run_verifier(tmp_path, source_payloads)

    assert result.returncode == 0, result.stderr
    assert "artifact identity verified" in result.stdout


@pytest.mark.parametrize(
    ("payload_index", "mutate", "expected_error"),
    [
        (0, lambda payload: payload.update(conclusion="success"), "run conclusion"),
        (
            1,
            lambda payload: payload["jobs"][0]["steps"][4].update(conclusion="success"),
            "step 34 conclusion",
        ),
        (
            1,
            lambda payload: payload["jobs"][0].update(id=SOURCE_JOB_ID + 1),
            "exact approved job once",
        ),
        (
            2,
            lambda payload: payload["artifacts"][0].update(expired=True),
            "artifact expired state",
        ),
    ],
)
def test_rejects_source_identity_drift(
    tmp_path: Path,
    source_payloads: tuple[dict, dict, dict],
    payload_index: int,
    mutate: Callable[[dict], None],
    expected_error: str,
) -> None:
    run, jobs, artifacts = (copy.deepcopy(payload) for payload in source_payloads)
    payloads = (run, jobs, artifacts)
    mutate(payloads[payload_index])

    result = run_verifier(tmp_path, payloads)

    assert result.returncode == 2
    assert expected_error in result.stderr


def test_rejects_truncated_jobs_response(
    tmp_path: Path,
    source_payloads: tuple[dict, dict, dict],
) -> None:
    run, jobs, artifacts = copy.deepcopy(source_payloads)
    jobs["total_count"] = 2

    result = run_verifier(tmp_path, (run, jobs, artifacts))

    assert result.returncode == 2
    assert "source jobs count mismatch" in result.stderr
