# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/scripts/publish_ascend_benchmark_results.sh"
)


def _publisher(tmp_path: Path) -> tuple[Path, Path]:
    log = tmp_path / "publisher.log"
    publisher = tmp_path / "publisher.sh"
    publisher.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s|%s|%s|%s\\n' "$RUN_ID" "$CURRENT_SUBMISSION_DIR" \\
  "$BENCHMARK_REPO_GH_TOKEN" "$BENCHMARK_REPO_SSH_KEY" >> "$PUBLISHER_LOG"
""",
        encoding="utf-8",
    )
    publisher.chmod(0o755)
    return publisher, log


def _submission(path: Path, status: str = "OK") -> None:
    path.mkdir(parents=True)
    (path / "STATUS").write_text(f"{status}\n", encoding="utf-8")


def _run(tmp_path: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    publisher, log = _publisher(tmp_path)
    complete_env = {
        **os.environ,
        "BENCHMARK_PUBLICATION_SYNC_SCRIPT": str(publisher),
        "BENCHMARK_REPO_GH_TOKEN": "token-sentinel",
        "BENCHMARK_REPO_SSH_KEY": "key-sentinel",
        "PUBLISHER_LOG": str(log),
        "RESULT_ROOT": str(tmp_path / "results"),
        "RUN_ID": "single-run",
        "VLLM_HUST_BENCHMARK_REPO": str(tmp_path / "benchmark"),
        "VLLM_HUST_REPO": str(tmp_path / "core"),
        "VLLM_HUST_WEBSITE_REPO": str(tmp_path / "website"),
        **env,
    }
    return subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=complete_env,
    )


def test_single_submission_publishes_with_scoped_writer(tmp_path: Path) -> None:
    submission = tmp_path / "results/submissions/single-run"
    _submission(submission)

    result = _run(tmp_path, {})

    assert result.returncode == 0, result.stdout + result.stderr
    log = (tmp_path / "publisher.log").read_text(encoding="utf-8")
    assert log == f"single-run|{submission}|token-sentinel|key-sentinel\n"


def test_multi_scenario_publishes_every_validated_submission(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    submission_a = result_root / "random/submissions/run-random"
    submission_b = result_root / "sharegpt/submissions/run-sharegpt"
    _submission(submission_a)
    _submission(submission_b)
    summary = result_root / "multi_scenario_results.tsv"
    summary.write_text(
        "scenario\trun_id\tresult_root\traw_result\tsubmission_dir\texit_code\n"
        f"random-online\trun-random\t{result_root / 'random'}\t"
        f"unused\t{submission_a}\t0\n"
        f"sharegpt-online\trun-sharegpt\t{result_root / 'sharegpt'}\t"
        f"unused\t{submission_b}\t0\n",
        encoding="utf-8",
    )

    result = _run(tmp_path, {})

    assert result.returncode == 0, result.stdout + result.stderr
    lines = (tmp_path / "publisher.log").read_text(encoding="utf-8").splitlines()
    assert [line.split("|", 1)[0] for line in lines] == ["run-random", "run-sharegpt"]


def test_multi_scenario_refuses_partial_publication(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    submission = result_root / "random/submissions/run-random"
    _submission(submission)
    summary = result_root / "multi_scenario_results.tsv"
    summary.write_text(
        "scenario\trun_id\tresult_root\traw_result\tsubmission_dir\texit_code\n"
        f"random-online\trun-random\t{result_root / 'random'}\t"
        f"unused\t{submission}\t0\n"
        f"sharegpt-online\trun-sharegpt\t{result_root / 'sharegpt'}\t"
        "unused\tmissing\t2\n",
        encoding="utf-8",
    )

    result = _run(tmp_path, {})

    assert result.returncode == 2
    assert "refusing partial multi-scenario publication" in result.stderr
    assert not (tmp_path / "publisher.log").exists()
