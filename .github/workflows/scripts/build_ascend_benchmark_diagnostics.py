#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a secret-free structured summary for an Ascend benchmark run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "ascend-nightly-diagnostics/v1"
REQUIRED_EVIDENCE = (
    "run_leaderboard.json",
    "leaderboard_manifest.json",
    "env-manifest.json",
    "pip-packages.json",
    "checksums.sha256",
)
HEX_DIGITS = frozenset("0123456789abcdef")
SCENARIO_SUMMARY_FIELDS = (
    "scenario",
    "run_id",
    "result_root",
    "raw_result",
    "submission_dir",
    "exit_code",
)


def _parse_exit_code(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    return None


def _candidate_path(raw_path: str, root: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


def _path_is_contained(path: Path, root: Path) -> bool:
    root_absolute = Path(os.path.abspath(root))
    path_absolute = Path(os.path.abspath(path))
    try:
        relative_path = path_absolute.relative_to(root_absolute)
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False

    current = root_absolute
    if current.is_symlink():
        return False
    for part in relative_path.parts:
        current /= part
        if current.is_symlink():
            return False
    return True


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def load_step_results(raw_json: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(raw_json or "[]")
    except json.JSONDecodeError as error:
        raise ValueError("step results are not valid JSON") from error
    if not isinstance(payload, list):
        raise ValueError("step results must be a JSON list")

    results = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each step result must be an object")
        step_id = str(item.get("id") or "").strip()
        name = str(item.get("name") or step_id).strip()
        outcome = str(item.get("outcome") or "unknown").strip().lower()
        if not step_id or outcome not in {
            "success",
            "failure",
            "cancelled",
            "skipped",
            "unknown",
        }:
            raise ValueError("step results contain an invalid id or outcome")
        results.append(
            {
                "id": step_id,
                "name": name,
                "outcome": outcome,
                "exit_code": _parse_exit_code(item.get("exit_code")),
            }
        )
    return results


def _raw_result_is_valid(path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict)


def _rejected_scenario(
    bench_scenario: str,
    run_id: str,
    formal_exit_code: int | None,
    reason: str,
    *,
    row: dict[str | None, Any] | None = None,
) -> dict[str, Any]:
    row = row or {}
    scenario_value = row.get("scenario")
    run_id_value = row.get("run_id")
    scenario = (
        scenario_value.strip()
        if isinstance(scenario_value, str) and scenario_value.strip()
        else bench_scenario
    )
    scenario_run_id = (
        run_id_value.strip()
        if isinstance(run_id_value, str) and run_id_value.strip()
        else run_id
    )
    row_exit_code = _parse_exit_code(row.get("exit_code"))
    return {
        "scenario": scenario,
        "run_id": scenario_run_id,
        "status": "failed",
        "exit_code": (row_exit_code if row_exit_code is not None else formal_exit_code),
        "result_root": "rejected",
        "raw_result": "rejected",
        "submission_dir": "rejected",
        "path_errors": [reason],
        "_submission_path": None,
    }


def load_scenarios(
    result_root: Path,
    summary_path: Path,
    bench_scenario: str,
    run_id: str,
    formal_outcome: str,
    formal_exit_code: int | None,
) -> list[dict[str, Any]]:
    scenarios = []
    if summary_path.is_file():
        if summary_path.is_symlink() or not _path_is_contained(
            summary_path, result_root
        ):
            return [
                _rejected_scenario(
                    bench_scenario,
                    run_id,
                    formal_exit_code,
                    "scenario summary escapes the benchmark result root",
                )
            ]
        try:
            with summary_path.open(encoding="utf-8", newline="") as summary_file:
                reader = csv.DictReader(summary_file, delimiter="\t", strict=True)
                missing_columns = [
                    field
                    for field in SCENARIO_SUMMARY_FIELDS
                    if field not in (reader.fieldnames or [])
                ]
                saw_row = False
                for row in reader:
                    saw_row = True
                    if missing_columns:
                        scenarios.append(
                            _rejected_scenario(
                                bench_scenario,
                                run_id,
                                formal_exit_code,
                                "scenario summary is missing required columns: "
                                + ", ".join(missing_columns),
                                row=row,
                            )
                        )
                        continue
                    empty_fields = [
                        field
                        for field in SCENARIO_SUMMARY_FIELDS[:-1]
                        if not isinstance(row.get(field), str)
                        or not row[field].strip()
                        or "\x00" in row[field]
                    ]
                    if empty_fields:
                        scenarios.append(
                            _rejected_scenario(
                                bench_scenario,
                                run_id,
                                formal_exit_code,
                                "scenario summary has invalid required fields: "
                                + ", ".join(empty_fields),
                                row=row,
                            )
                        )
                        continue
                    try:
                        scenario_root = _candidate_path(row["result_root"], result_root)
                        submission_dir = _candidate_path(
                            row["submission_dir"], result_root
                        )
                        raw_result = _candidate_path(row["raw_result"], result_root)
                        command_exit_code = _parse_exit_code(row.get("exit_code"))
                        path_errors = []
                        if not _path_is_contained(scenario_root, result_root):
                            path_errors.append(
                                "result_root escapes the benchmark result root"
                            )
                        if not _path_is_contained(raw_result, scenario_root):
                            path_errors.append(
                                "raw_result escapes the scenario result root"
                            )
                        if not _path_is_contained(submission_dir, scenario_root):
                            path_errors.append("submission_dir escapes scenario root")

                        status_path = submission_dir / "STATUS"
                        execution_passed = False
                        if not path_errors and _raw_result_is_valid(raw_result):
                            try:
                                execution_passed = (
                                    status_path.is_file()
                                    and not status_path.is_symlink()
                                    and status_path.read_text(encoding="utf-8").strip()
                                    == "OK"
                                )
                            except (OSError, UnicodeDecodeError):
                                execution_passed = False
                        scenarios.append(
                            {
                                "scenario": row["scenario"],
                                "run_id": row["run_id"],
                                "status": ("passed" if execution_passed else "failed"),
                                "exit_code": command_exit_code,
                                "result_root": (
                                    _relative_path(scenario_root, result_root)
                                    if not path_errors
                                    else "rejected"
                                ),
                                "raw_result": (
                                    _relative_path(raw_result, result_root)
                                    if not path_errors
                                    else "rejected"
                                ),
                                "submission_dir": (
                                    _relative_path(submission_dir, result_root)
                                    if not path_errors
                                    else "rejected"
                                ),
                                "path_errors": path_errors,
                                "_submission_path": (
                                    submission_dir if not path_errors else None
                                ),
                            }
                        )
                    except (OSError, RuntimeError, TypeError, ValueError):
                        scenarios.append(
                            _rejected_scenario(
                                bench_scenario,
                                run_id,
                                formal_exit_code,
                                "scenario summary row contains invalid paths",
                                row=row,
                            )
                        )
                if missing_columns and not saw_row:
                    scenarios.append(
                        _rejected_scenario(
                            bench_scenario,
                            run_id,
                            formal_exit_code,
                            "scenario summary is missing required columns: "
                            + ", ".join(missing_columns),
                        )
                    )
        except (OSError, UnicodeDecodeError, csv.Error):
            scenarios.append(
                _rejected_scenario(
                    bench_scenario,
                    run_id,
                    formal_exit_code,
                    "scenario summary is unreadable or malformed",
                )
            )
        return scenarios

    submission_dir = result_root / "submissions" / run_id
    raw_result = result_root / "raw_benchmark.json"
    status_path = submission_dir / "STATUS"
    paths_are_safe = _path_is_contained(
        submission_dir, result_root
    ) and _path_is_contained(raw_result, result_root)
    try:
        execution_passed = (
            paths_are_safe
            and _raw_result_is_valid(raw_result)
            and status_path.is_file()
            and not status_path.is_symlink()
            and status_path.read_text(encoding="utf-8").strip() == "OK"
        )
    except OSError:
        execution_passed = False
    if execution_passed or formal_outcome in {"success", "failure"}:
        scenarios.append(
            {
                "scenario": bench_scenario,
                "run_id": run_id,
                "status": "passed" if execution_passed else "failed",
                "exit_code": formal_exit_code,
                "result_root": ".",
                "raw_result": _relative_path(raw_result, result_root),
                "submission_dir": _relative_path(submission_dir, result_root),
                "_submission_path": submission_dir,
            }
        )
    return scenarios


def verify_submission_evidence(submission_dir: Path) -> dict[str, Any]:
    missing = []
    for file_name in REQUIRED_EVIDENCE:
        evidence_path = submission_dir / file_name
        if (
            not _path_is_contained(evidence_path, submission_dir)
            or not evidence_path.is_file()
            or evidence_path.is_symlink()
        ):
            missing.append(file_name)
    checksum_errors: list[str] = []
    checksums_path = submission_dir / "checksums.sha256"
    covered_files: set[str] = set()

    if (
        checksums_path.is_file()
        and not checksums_path.is_symlink()
        and _path_is_contained(checksums_path, submission_dir)
    ):
        try:
            checksum_lines = checksums_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            checksum_lines = []
            checksum_errors.append("checksums.sha256 is unreadable")
        for line_number, line in enumerate(checksum_lines, start=1):
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                checksum_errors.append(f"invalid checksum line {line_number}")
                continue
            expected_digest, raw_name = parts
            if len(expected_digest) != 64 or not set(expected_digest).issubset(
                HEX_DIGITS
            ):
                checksum_errors.append(f"invalid checksum line {line_number}")
                continue
            raw_name = raw_name.removeprefix("*")
            normalized_name = raw_name.removeprefix("./")
            relative_name = Path(normalized_name)
            if relative_name.is_absolute() or ".." in relative_name.parts:
                checksum_errors.append(f"unsafe checksum path at line {line_number}")
                continue
            artifact_path = submission_dir / relative_name
            covered_files.add(relative_name.as_posix())
            if (
                not _path_is_contained(artifact_path, submission_dir)
                or not artifact_path.is_file()
                or artifact_path.is_symlink()
            ):
                checksum_errors.append(
                    f"unsafe or missing checksum target: {relative_name.as_posix()}"
                )
                continue
            actual_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if actual_digest != expected_digest:
                checksum_errors.append(f"checksum mismatch: {relative_name.as_posix()}")

        for required_name in REQUIRED_EVIDENCE[:-1]:
            if required_name not in covered_files:
                checksum_errors.append(f"checksum entry missing: {required_name}")

    return {
        "status": "passed" if not missing and not checksum_errors else "failed",
        "missing_files": missing,
        "checksum_errors": checksum_errors,
    }


def load_plugin_preflights(result_root: Path) -> list[dict[str, str]]:
    results = []
    for path in sorted((result_root / "preflight").glob("plugin-*.json")):
        if path.is_symlink() or not _path_is_contained(path, result_root):
            results.append(
                {
                    "mode": path.stem.removeprefix("plugin-"),
                    "status": "failed",
                    "reason": "preflight result path is unsafe",
                }
            )
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            results.append(
                {
                    "mode": path.stem.removeprefix("plugin-"),
                    "status": "failed",
                    "reason": "preflight result is unreadable",
                }
            )
            continue
        result = {
            "mode": str(payload.get("mode") or "unknown"),
            "status": str(payload.get("status") or "failed"),
        }
        if result["status"] == "failed":
            result["reason"] = str(payload.get("reason") or "unknown failure")
        results.append(result)
    return results


def build_diagnostics(
    *,
    result_root: Path,
    step_results: list[dict[str, Any]],
    scenario_summary: Path,
    run_id: str,
    bench_scenario: str,
    publish_enabled: bool,
    publication_status_value: str,
    publication_verification: str,
    run_metadata: dict[str, str],
) -> dict[str, Any]:
    formal_step = next(
        (step for step in step_results if step["id"] == "formal-benchmark"),
        {"outcome": "unknown", "exit_code": None},
    )
    scenarios = load_scenarios(
        result_root,
        scenario_summary,
        bench_scenario,
        run_id,
        formal_step["outcome"],
        formal_step["exit_code"],
    )
    passed_count = sum(item["status"] == "passed" for item in scenarios)
    failed_count = sum(item["status"] == "failed" for item in scenarios)
    if scenarios and failed_count == 0:
        execution_status = "success"
    elif passed_count > 0:
        execution_status = "partial"
    else:
        execution_status = "failed"

    quality_results = []
    if execution_status == "success":
        for scenario in scenarios:
            quality_results.append(
                {
                    "scenario": scenario["scenario"],
                    **verify_submission_evidence(scenario["_submission_path"]),
                }
            )
        data_quality_status = (
            "passed"
            if quality_results
            and all(item["status"] == "passed" for item in quality_results)
            else "failed"
        )
    else:
        data_quality_status = "not-evaluated"

    if not publish_enabled:
        publication_status = "not-attempted"
    elif (
        publication_status_value in {"pushed", "unchanged"}
        and publication_verification == "verified"
    ):
        publication_status = "succeeded"
    elif publication_status_value in {"attempting", "failed", "rejected"} or (
        publication_verification == "failed"
    ):
        publication_status = "failed"
    else:
        publication_status = "not-attempted"
    release_visibility_status = (
        "visible" if publication_status == "succeeded" else "not-published"
    )

    failed_steps = [
        step for step in step_results if step["outcome"] in {"failure", "cancelled"}
    ]
    public_scenarios = [
        {key: value for key, value in scenario.items() if not key.startswith("_")}
        for scenario in scenarios
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "run": run_metadata,
        "failed_steps": failed_steps,
        "plugin_preflight": load_plugin_preflights(result_root),
        "scenario_summary": {
            "passed": passed_count,
            "failed": failed_count,
            "scenarios": public_scenarios,
        },
        "data_quality": quality_results,
        "benchmark_execution_status": execution_status,
        "publication_status": publication_status,
        "data_quality_status": data_quality_status,
        "release_visibility_status": release_visibility_status,
    }


def append_github_summary(path: Path, payload: dict[str, Any]) -> None:
    failed_steps = payload["failed_steps"]

    def format_failed_step(item: dict[str, Any]) -> str:
        exit_code = item["exit_code"]
        exit_label = exit_code if exit_code is not None else "unknown"
        return f"{item['name']} (exit {exit_label})"

    failed_summary = (
        ", ".join(format_failed_step(item) for item in failed_steps) or "none"
    )
    scenario_summary = payload["scenario_summary"]
    scenario_status = (
        f"{scenario_summary['passed']} passed, {scenario_summary['failed']} failed"
    )
    lines = [
        "",
        "## Nightly Status",
        f"- Benchmark execution: `{payload['benchmark_execution_status']}`",
        f"- Publication: `{payload['publication_status']}`",
        f"- Data quality: `{payload['data_quality_status']}`",
        f"- Release visibility: `{payload['release_visibility_status']}`",
        f"- Scenarios: `{scenario_status}`",
        f"- Failed steps: {failed_summary}",
    ]
    with path.open("a", encoding="utf-8") as summary_file:
        summary_file.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario-summary", type=Path)
    parser.add_argument("--step-results-json", default="[]")
    parser.add_argument("--github-summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_root = args.result_root
    run_id = os.environ.get("RUN_ID") or result_root.name
    payload = build_diagnostics(
        result_root=result_root,
        step_results=load_step_results(args.step_results_json),
        scenario_summary=args.scenario_summary
        or result_root / "multi_scenario_results.tsv",
        run_id=run_id,
        bench_scenario=os.environ.get("BENCH_SCENARIO", "unknown"),
        publish_enabled=os.environ.get("PUBLISH_TO_BENCHMARK_REPO") == "1",
        publication_status_value=os.environ.get("GITHUB_SNAPSHOT_SYNC_STATUS", ""),
        publication_verification=os.environ.get(
            "GITHUB_SNAPSHOT_SYNC_VERIFICATION", ""
        ),
        run_metadata={
            "id": os.environ.get("GITHUB_RUN_ID", "local"),
            "attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
            "head_sha": os.environ.get("TARGET_REPO_SHA")
            or os.environ.get("GITHUB_SHA", "unknown"),
            "event": os.environ.get("GITHUB_EVENT_NAME", "local"),
            "url": (
                f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
                f"{os.environ.get('GITHUB_REPOSITORY', 'local/local')}/actions/runs/"
                f"{os.environ.get('GITHUB_RUN_ID', 'local')}"
            ),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.github_summary:
        append_github_summary(args.github_summary, payload)
    print(f"Nightly diagnostics: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
