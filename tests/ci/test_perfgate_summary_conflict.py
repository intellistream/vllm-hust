# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
import pathlib
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/ascend-benchmark-leaderboard.yml"


def test_summary_and_pr_comment_include_rebase_conflict_details(tmp_path: Path) -> None:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    run_script = next(
        step["run"]
        for job in workflow["jobs"].values()
        if isinstance(job, dict)
        for step in job.get("steps", [])
        if step.get("name") == "Build benchmark summary artifacts"
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    conflict_file = tmp_path / "rebase-conflict.txt"
    conflict_file.write_text(
        "CONFLICT (content): .github/workflows/ascend-benchmark-leaderboard.yml\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "GITHUB_WORKSPACE": str(workspace),
        "BENCHMARK_RESULTS_ROOT": str(tmp_path / "benchmarks"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REPOSITORY": "vLLM-HUST/test",
        "GITHUB_RUN_ID": "1",
        "GITHUB_RUN_ATTEMPT": "1",
        "TARGET_REPO_SHA": "abc123",
        "ASCEND_HUST_TARGET_SHA": "abc123",
        "GITHUB_SHA": "abc123",
        "GITHUB_REF_NAME": "test",
        "BENCH_SCENARIO": "random-online",
        "BENCH_SCENARIO_COUNT": "1",
        "PERFGATE_STAGE2_REBASE_CONFLICT_FILE": str(conflict_file),
    }
    result = subprocess.run(
        ["/bin/bash"],
        input=run_script,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    comment_files = list(tmp_path.rglob("benchmark_comment.md"))
    assert len(comment_files) == 1
    summary = Path(env["GITHUB_STEP_SUMMARY"]).read_text(encoding="utf-8")
    comment = comment_files[0].read_text(encoding="utf-8")
    for text in (summary, comment):
        assert "Stage 2 rebase conflict details" in text
        assert "CONFLICT (content)" in text

