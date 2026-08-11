#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the immutable Actions source for a publication-only replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXPECTED_WORKFLOW_PATH = ".github/workflows/ascend-benchmark-leaderboard.yml"
EXPECTED_JOB_NAME = "ascend-benchmark"
EXPECTED_JOB_CONCLUSION = "failure"
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


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} response is unreadable") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} response must be an object")
    return payload


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def verify_source_run(
    run: dict[str, Any],
    jobs_payload: dict[str, Any],
    artifacts_payload: dict[str, Any],
    *,
    source_run_id: int,
    source_run_attempt: int,
    source_job_id: int,
    expected_target_sha: str,
    expected_artifact_name: str,
) -> None:
    _require_equal(run.get("id"), source_run_id, "source run id")
    _require_equal(run.get("path"), EXPECTED_WORKFLOW_PATH, "source workflow path")
    _require_equal(run.get("event"), "push", "source event")
    _require_equal(run.get("head_branch"), "main", "source branch")
    _require_equal(run.get("head_sha"), expected_target_sha, "source head SHA")
    _require_equal(run.get("run_attempt"), source_run_attempt, "source run attempt")
    _require_equal(run.get("status"), "completed", "source run status")
    _require_equal(run.get("conclusion"), "failure", "source run conclusion")

    jobs = jobs_payload.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("source jobs response must contain a jobs list")
    _require_equal(jobs_payload.get("total_count"), len(jobs), "source jobs count")
    matching_jobs = [job for job in jobs if job.get("id") == source_job_id]
    if len(matching_jobs) != 1:
        raise ValueError(
            "source jobs response must contain the exact approved job once"
        )
    job = matching_jobs[0]
    _require_equal(job.get("run_id"), source_run_id, "source job run id")
    _require_equal(job.get("run_attempt"), source_run_attempt, "source job attempt")
    _require_equal(job.get("head_sha"), expected_target_sha, "source job head SHA")
    _require_equal(job.get("name"), EXPECTED_JOB_NAME, "source job name")
    _require_equal(job.get("status"), "completed", "source job status")
    _require_equal(
        job.get("conclusion"), EXPECTED_JOB_CONCLUSION, "source job conclusion"
    )

    steps = job.get("steps")
    if not isinstance(steps, list):
        raise ValueError("approved source job must contain a steps list")
    steps_by_number: dict[int, dict[str, Any]] = {}
    for step in steps:
        number = step.get("number")
        if not isinstance(number, int):
            raise ValueError("source job step number must be an integer")
        if number in steps_by_number:
            raise ValueError(f"duplicate source job step number: {number}")
        steps_by_number[number] = step
    for number, (expected_name, expected_conclusion) in EXPECTED_STEPS.items():
        step = steps_by_number.get(number)
        if step is None:
            raise ValueError(f"approved source job step {number} is missing")
        _require_equal(step.get("name"), expected_name, f"source step {number} name")
        _require_equal(step.get("status"), "completed", f"source step {number} status")
        _require_equal(
            step.get("conclusion"),
            expected_conclusion,
            f"source step {number} conclusion",
        )

    artifacts = artifacts_payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("source artifacts response must contain an artifacts list")
    _require_equal(
        artifacts_payload.get("total_count"),
        len(artifacts),
        "source artifacts count",
    )
    matching_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.get("name") == expected_artifact_name
    ]
    if len(matching_artifacts) != 1:
        raise ValueError("source run must contain the exact approved artifact once")
    artifact = matching_artifacts[0]
    _require_equal(artifact.get("expired"), False, "source artifact expired state")
    artifact_run = artifact.get("workflow_run")
    if not isinstance(artifact_run, dict):
        raise ValueError("source artifact must include workflow_run identity")
    _require_equal(artifact_run.get("id"), source_run_id, "artifact source run id")
    _require_equal(artifact_run.get("head_branch"), "main", "artifact source branch")
    _require_equal(
        artifact_run.get("head_sha"), expected_target_sha, "artifact source head SHA"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--jobs-json", type=Path, required=True)
    parser.add_argument("--artifacts-json", type=Path, required=True)
    parser.add_argument("--source-run-id", type=int, required=True)
    parser.add_argument("--source-run-attempt", type=int, required=True)
    parser.add_argument("--source-job-id", type=int, required=True)
    parser.add_argument("--expected-target-sha", required=True)
    parser.add_argument("--expected-artifact-name", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        verify_source_run(
            _load_object(args.run_json, "source run"),
            _load_object(args.jobs_json, "source jobs"),
            _load_object(args.artifacts_json, "source artifacts"),
            source_run_id=args.source_run_id,
            source_run_attempt=args.source_run_attempt,
            source_job_id=args.source_job_id,
            expected_target_sha=args.expected_target_sha,
            expected_artifact_name=args.expected_artifact_name,
        )
    except ValueError as error:
        print(f"Replay source verification failed: {error}", file=sys.stderr)
        return 2
    print("Replay source run, job steps, and artifact identity verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
