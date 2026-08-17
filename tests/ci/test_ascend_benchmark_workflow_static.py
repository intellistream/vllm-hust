# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/ascend-benchmark-leaderboard.yml"
)


def test_benchmark_workflow_only_submits_registered_requests_to_112() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    job = workflow["jobs"]["submit-to-112"]

    assert "self-hosted" not in text
    assert "docker run" not in text
    assert "BENCHMARK_REPO_GH_TOKEN" not in text
    assert job["uses"].endswith("/.github/workflows/evaluation-request.yml@main")
    assert job["with"]["repeat_count"] == 3
    assert job["with"]["target_registry_version"] == "1.3.5"
    assert "official-ascend-jan-2026-v0.18.0-random-online" in text
    assert "official-ascend-jan-2026-v0.18.0-sharegpt-online" in text


def test_benchmark_workflow_keeps_trusted_triggers() -> None:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    triggers = workflow[True]

    assert triggers["push"]["branches"] == ["main"]
    assert set(triggers["pull_request"]["types"]) == {
        "labeled",
        "synchronize",
        "reopened",
    }
    assert triggers["schedule"][0]["cron"] == "0 17 * * *"
    assert set(
        triggers["workflow_dispatch"]["inputs"]["benchmark_scenario"]["options"]
    ) == {
        "random-online",
        "sharegpt-online",
    }
