# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/scripts/build_ascend_benchmark_diagnostics.py"
)
SPEC = importlib.util.spec_from_file_location("ascend_nightly_diagnostics", SCRIPT_PATH)
assert SPEC and SPEC.loader
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


def _write_submission(root: Path, run_id: str, *, valid_checksums: bool = True) -> Path:
    submission = root / "submissions" / run_id
    submission.mkdir(parents=True)
    evidence = {
        "run_leaderboard.json": "{}\n",
        "leaderboard_manifest.json": "{}\n",
        "env-manifest.json": "{}\n",
        "pip-packages.json": "[]\n",
    }
    for name, content in evidence.items():
        (submission / name).write_text(content, encoding="utf-8")
    checksum_lines = []
    for name, content in evidence.items():
        digest = hashlib.sha256(content.encode()).hexdigest()
        if name == "env-manifest.json" and not valid_checksums:
            digest = "0" * 64
        checksum_lines.append(f"{digest}  ./{name}")
    (submission / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    (submission / "STATUS").write_text("OK\n", encoding="utf-8")
    return submission


def _write_scenario(
    result_root: Path,
    scenario: str,
    exit_code: int,
    *,
    complete: bool,
) -> tuple[str, str, str, str, str, str]:
    run_id = f"nightly-{scenario}"
    scenario_root = result_root / scenario
    raw_result = scenario_root / "raw_benchmark.json"
    submission = scenario_root / "submissions" / run_id
    scenario_root.mkdir(parents=True)
    if complete:
        raw_result.write_text("{}\n", encoding="utf-8")
        submission = _write_submission(scenario_root, run_id)
    return (
        scenario,
        run_id,
        str(scenario_root),
        str(raw_result),
        str(submission),
        str(exit_code),
    )


def _write_scenario_summary(result_root: Path, rows: Sequence[tuple[str, ...]]) -> Path:
    summary = result_root / "multi_scenario_results.tsv"
    header = "scenario\trun_id\tresult_root\traw_result\tsubmission_dir\texit_code\n"
    summary.write_text(
        header + "".join("\t".join(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return summary


def _build(
    result_root: Path,
    summary: Path,
    steps: list[dict[str, str]],
    *,
    publish: bool = False,
    sync_status: str = "",
    verification: str = "",
) -> dict:
    return diagnostics.build_diagnostics(
        result_root=result_root,
        step_results=diagnostics.load_step_results(json.dumps(steps)),
        scenario_summary=summary,
        run_id="nightly",
        bench_scenario="random-online",
        publish_enabled=publish,
        publication_status_value=sync_status,
        publication_verification=verification,
        run_metadata={"id": "123", "head_sha": "a" * 40},
    )


def test_multi_scenario_success_reports_all_four_restored_states(tmp_path: Path):
    result_root = tmp_path / "results"
    result_root.mkdir()
    rows = [
        _write_scenario(result_root, "random-online", 0, complete=True),
        _write_scenario(result_root, "sharegpt-online", 0, complete=True),
    ]
    summary = _write_scenario_summary(result_root, rows)

    payload = _build(
        result_root,
        summary,
        [{"id": "formal-benchmark", "outcome": "success", "exit_code": "0"}],
        publish=True,
        sync_status="pushed",
        verification="verified",
    )

    assert payload["benchmark_execution_status"] == "success"
    assert payload["publication_status"] == "succeeded"
    assert payload["data_quality_status"] == "passed"
    assert payload["release_visibility_status"] == "visible"
    assert payload["scenario_summary"]["passed"] == 2
    assert payload["failed_steps"] == []


def test_partial_run_preserves_failed_step_and_scenario_exit_codes(tmp_path: Path):
    result_root = tmp_path / "results"
    result_root.mkdir()
    rows = [
        _write_scenario(result_root, "random-online", 0, complete=True),
        _write_scenario(result_root, "sharegpt-online", 87, complete=False),
    ]
    summary = _write_scenario_summary(result_root, rows)

    payload = _build(
        result_root,
        summary,
        [
            {
                "id": "formal-benchmark",
                "name": "Run formal benchmark",
                "outcome": "failure",
                "exit_code": "87",
            }
        ],
    )

    assert payload["benchmark_execution_status"] == "partial"
    assert payload["publication_status"] == "not-attempted"
    assert payload["data_quality_status"] == "not-evaluated"
    assert payload["release_visibility_status"] == "not-published"
    assert payload["failed_steps"][0]["exit_code"] == 87
    assert [item["exit_code"] for item in payload["scenario_summary"]["scenarios"]] == [
        0,
        87,
    ]


def test_scenario_summary_missing_columns_is_rejected(tmp_path: Path):
    result_root = tmp_path / "results"
    result_root.mkdir()
    summary = result_root / "multi_scenario_results.tsv"
    summary.write_text(
        "scenario\trun_id\tresult_root\tsubmission_dir\texit_code\n"
        "random-online\tnightly-random\t.\tsubmissions/nightly-random\t1\n",
        encoding="utf-8",
    )

    payload = _build(
        result_root,
        summary,
        [{"id": "formal-benchmark", "outcome": "failure", "exit_code": "1"}],
    )

    scenario = payload["scenario_summary"]["scenarios"][0]
    assert payload["benchmark_execution_status"] == "failed"
    assert scenario["scenario"] == "random-online"
    assert scenario["raw_result"] == "rejected"
    assert scenario["path_errors"] == [
        "scenario summary is missing required columns: raw_result"
    ]


def test_invalid_scenario_row_is_rejected_without_losing_valid_rows(tmp_path: Path):
    result_root = tmp_path / "results"
    result_root.mkdir()
    valid_row = _write_scenario(result_root, "random-online", 0, complete=True)
    invalid_row = (
        "sharegpt-online",
        "nightly-sharegpt-online",
        "",
        "",
        "",
        "0",
    )
    summary = _write_scenario_summary(result_root, [valid_row, invalid_row])

    payload = _build(
        result_root,
        summary,
        [{"id": "formal-benchmark", "outcome": "failure", "exit_code": "87"}],
    )

    assert payload["benchmark_execution_status"] == "partial"
    assert payload["scenario_summary"]["passed"] == 1
    assert payload["scenario_summary"]["failed"] == 1
    rejected = payload["scenario_summary"]["scenarios"][1]
    assert rejected["scenario"] == "sharegpt-online"
    assert rejected["exit_code"] == 0
    assert rejected["result_root"] == "rejected"
    assert rejected["path_errors"] == [
        "scenario summary has invalid required fields: "
        "result_root, raw_result, submission_dir"
    ]


def test_unreadable_scenario_summary_is_rejected(tmp_path: Path):
    result_root = tmp_path / "results"
    result_root.mkdir()
    summary = result_root / "multi_scenario_results.tsv"
    summary.write_bytes(b"scenario\trun_id\n\xff")
    output = result_root / "nightly-diagnostics.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--result-root",
            str(result_root),
            "--output",
            str(output),
            "--scenario-summary",
            str(summary),
            "--step-results-json",
            '[{"id":"formal-benchmark","outcome":"failure","exit_code":"1"}]',
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    scenario = payload["scenario_summary"]["scenarios"][0]
    assert payload["benchmark_execution_status"] == "failed"
    assert scenario["status"] == "failed"
    assert scenario["result_root"] == "rejected"
    assert scenario["path_errors"] == ["scenario summary is unreadable or malformed"]


def test_successful_execution_fails_data_quality_for_tampered_evidence(tmp_path: Path):
    result_root = tmp_path / "results"
    result_root.mkdir()
    run_id = "nightly"
    (result_root / "raw_benchmark.json").write_text("{}\n", encoding="utf-8")
    _write_submission(result_root, run_id, valid_checksums=False)

    payload = _build(
        result_root,
        result_root / "missing.tsv",
        [{"id": "formal-benchmark", "outcome": "success", "exit_code": "0"}],
    )

    assert payload["benchmark_execution_status"] == "success"
    assert payload["data_quality_status"] == "failed"
    assert payload["data_quality"][0]["checksum_errors"] == [
        "checksum mismatch: env-manifest.json"
    ]


def test_successful_execution_fails_data_quality_for_missing_pip_packages(
    tmp_path: Path,
):
    result_root = tmp_path / "results"
    result_root.mkdir()
    (result_root / "raw_benchmark.json").write_text("{}\n", encoding="utf-8")
    submission = _write_submission(result_root, "nightly")
    (submission / "pip-packages.json").unlink()

    payload = _build(
        result_root,
        result_root / "missing.tsv",
        [{"id": "formal-benchmark", "outcome": "success", "exit_code": "0"}],
    )

    assert payload["data_quality_status"] == "failed"
    assert payload["data_quality"][0]["missing_files"] == ["pip-packages.json"]
    assert (
        "unsafe or missing checksum target: pip-packages.json"
        in payload["data_quality"][0]["checksum_errors"]
    )


def test_successful_execution_fails_data_quality_for_missing_pip_checksum_entry(
    tmp_path: Path,
):
    result_root = tmp_path / "results"
    result_root.mkdir()
    (result_root / "raw_benchmark.json").write_text("{}\n", encoding="utf-8")
    submission = _write_submission(result_root, "nightly")
    checksums_path = submission / "checksums.sha256"
    checksum_lines = checksums_path.read_text(encoding="utf-8").splitlines()
    checksums_path.write_text(
        "\n".join(line for line in checksum_lines if "pip-packages.json" not in line)
        + "\n",
        encoding="utf-8",
    )

    payload = _build(
        result_root,
        result_root / "missing.tsv",
        [{"id": "formal-benchmark", "outcome": "success", "exit_code": "0"}],
    )

    assert payload["data_quality_status"] == "failed"
    assert payload["data_quality"][0]["missing_files"] == []
    assert payload["data_quality"][0]["checksum_errors"] == [
        "checksum entry missing: pip-packages.json"
    ]


def test_successful_execution_fails_data_quality_for_tampered_pip_packages(
    tmp_path: Path,
):
    result_root = tmp_path / "results"
    result_root.mkdir()
    (result_root / "raw_benchmark.json").write_text("{}\n", encoding="utf-8")
    submission = _write_submission(result_root, "nightly")
    (submission / "pip-packages.json").write_text(
        '[{"name": "tampered"}]\n', encoding="utf-8"
    )

    payload = _build(
        result_root,
        result_root / "missing.tsv",
        [{"id": "formal-benchmark", "outcome": "success", "exit_code": "0"}],
    )

    assert payload["data_quality_status"] == "failed"
    assert payload["data_quality"][0]["missing_files"] == []
    assert payload["data_quality"][0]["checksum_errors"] == [
        "checksum mismatch: pip-packages.json"
    ]


def test_enabled_publication_without_attempt_remains_not_attempted(tmp_path: Path):
    result_root = tmp_path / "results"
    result_root.mkdir()
    (result_root / "raw_benchmark.json").write_text("{}\n", encoding="utf-8")
    _write_submission(result_root, "nightly")

    payload = _build(
        result_root,
        result_root / "missing.tsv",
        [{"id": "formal-benchmark", "outcome": "failure", "exit_code": "1"}],
        publish=True,
    )

    assert payload["benchmark_execution_status"] == "success"
    assert payload["publication_status"] == "not-attempted"
    assert payload["release_visibility_status"] == "not-published"


def test_explicit_publication_failure_is_reported(tmp_path: Path):
    result_root = tmp_path / "results"
    result_root.mkdir()

    payload = _build(
        result_root,
        result_root / "missing.tsv",
        [{"id": "formal-benchmark", "outcome": "failure", "exit_code": "1"}],
        publish=True,
        sync_status="failed",
    )

    assert payload["publication_status"] == "failed"


def test_scenario_paths_outside_result_root_are_rejected(tmp_path: Path):
    result_root = tmp_path / "results"
    result_root.mkdir()
    external_root = tmp_path / "external"
    external_root.mkdir()
    external_raw = external_root / "raw_benchmark.json"
    external_raw.write_text("{}\n", encoding="utf-8")
    external_submission = _write_submission(external_root, "outside")
    summary = _write_scenario_summary(
        result_root,
        [
            (
                "random-online",
                "outside",
                str(external_root),
                str(external_raw),
                str(external_submission),
                "0",
            )
        ],
    )

    payload = _build(
        result_root,
        summary,
        [{"id": "formal-benchmark", "outcome": "success", "exit_code": "0"}],
    )

    scenario = payload["scenario_summary"]["scenarios"][0]
    assert payload["benchmark_execution_status"] == "failed"
    assert scenario["result_root"] == "rejected"
    assert scenario["submission_dir"] == "rejected"
    assert scenario["path_errors"]
    assert str(external_root) not in json.dumps(payload)


def test_scenario_symlink_and_checksum_symlink_are_rejected(tmp_path: Path):
    result_root = tmp_path / "results"
    result_root.mkdir()
    external_root = tmp_path / "external"
    external_root.mkdir()
    linked_scenario = result_root / "linked"
    linked_scenario.symlink_to(external_root, target_is_directory=True)
    summary = _write_scenario_summary(
        result_root,
        [
            (
                "random-online",
                "linked",
                str(linked_scenario),
                str(linked_scenario / "raw_benchmark.json"),
                str(linked_scenario / "submissions/linked"),
                "0",
            )
        ],
    )
    payload = _build(
        result_root,
        summary,
        [{"id": "formal-benchmark", "outcome": "success", "exit_code": "0"}],
    )
    assert payload["benchmark_execution_status"] == "failed"
    assert payload["scenario_summary"]["scenarios"][0]["path_errors"]

    submission = _write_submission(result_root, "nightly")
    external_manifest = tmp_path / "external-manifest.json"
    external_manifest.write_text("{}\n", encoding="utf-8")
    (submission / "env-manifest.json").unlink()
    (submission / "env-manifest.json").symlink_to(external_manifest)
    quality = diagnostics.verify_submission_evidence(submission)
    assert quality["status"] == "failed"
    assert "env-manifest.json" in quality["missing_files"]
    assert (
        "unsafe or missing checksum target: env-manifest.json"
        in quality["checksum_errors"]
    )


def test_scenario_summary_symlink_is_rejected_without_reading_target(tmp_path: Path):
    result_root = tmp_path / "results"
    result_root.mkdir()
    external_summary = tmp_path / "external.tsv"
    external_summary.write_text(
        "scenario\trun_id\tresult_root\traw_result\tsubmission_dir\texit_code\n"
        "secret-value\tnightly\t.\t.\t.\t0\n",
        encoding="utf-8",
    )
    summary = result_root / "multi_scenario_results.tsv"
    summary.symlink_to(external_summary)

    payload = _build(
        result_root,
        summary,
        [{"id": "formal-benchmark", "outcome": "success", "exit_code": "0"}],
    )

    assert payload["benchmark_execution_status"] == "failed"
    assert payload["scenario_summary"]["scenarios"][0]["result_root"] == "rejected"
    assert "secret-value" not in json.dumps(payload)


def test_plugin_preflight_symlink_is_rejected_without_reading_target(tmp_path: Path):
    result_root = tmp_path / "results"
    preflight_root = result_root / "preflight"
    preflight_root.mkdir(parents=True)
    external_result = tmp_path / "plugin-installed.json"
    external_result.write_text(
        json.dumps(
            {
                "mode": "installed",
                "status": "failed",
                "reason": "secret-value",
            }
        ),
        encoding="utf-8",
    )
    (preflight_root / "plugin-installed.json").symlink_to(external_result)

    payload = _build(
        result_root,
        result_root / "missing.tsv",
        [{"id": "formal-benchmark", "outcome": "failure", "exit_code": "1"}],
    )

    assert payload["plugin_preflight"] == [
        {
            "mode": "installed",
            "status": "failed",
            "reason": "preflight result path is unsafe",
        }
    ]
    assert "secret-value" not in json.dumps(payload)


def test_step_results_drop_unapproved_output_fields():
    secret = "not-for-artifacts"
    results = diagnostics.load_step_results(
        json.dumps(
            [
                {
                    "id": "install-plugin",
                    "name": "Install plugin",
                    "outcome": "failure",
                    "exit_code": "2",
                    "output": secret,
                }
            ]
        )
    )

    assert secret not in json.dumps(results)
    assert results == [
        {
            "id": "install-plugin",
            "name": "Install plugin",
            "outcome": "failure",
            "exit_code": 2,
        }
    ]
