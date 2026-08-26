# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
def load_workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_ordinary_pr_ci_is_hosted_and_does_not_run_ascend_performance() -> None:
    workflow_path = REPO_ROOT / ".github/workflows/pre-commit.yml"
    workflow = load_workflow(workflow_path)
    text = workflow_path.read_text(encoding="utf-8")

    assert workflow[True].get("pull_request") is not None
    assert workflow[True].get("push") is not None
    assert workflow["jobs"]["pre-commit"]["runs-on"] == "ubuntu-latest"
    assert "self-hosted" not in text
    assert "npu" not in text.lower()
    assert "evaluation-request.yml@main" not in text


def test_pre_commit_uses_available_github_hosted_runner():
    workflow_path = REPO_ROOT / ".github/workflows/pre-commit.yml"
    workflow = load_workflow(workflow_path)
    pre_commit = workflow["jobs"]["pre-commit"]
    text = workflow_path.read_text(encoding="utf-8")

    assert pre_commit["if"] != "false"
    assert pre_commit["runs-on"] == "ubuntu-latest"
    assert "fetch-depth: 0" in text
    assert "--from-ref" in text
    assert "--to-ref" in text
    assert "--all-files" not in text
    assert "--hook-stage manual" not in text
    assert "PRE_COMMIT_FROM_REF:" in text
    assert "PRE_COMMIT_TO_REF:" in text


def test_actionlint_knows_ascend_runner_labels():
    config_path = REPO_ROOT / ".github/actionlint.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    labels = set(config["self-hosted-runner"]["labels"])

    assert {"ascend", "910b", "docker", "npu-2"} <= labels
