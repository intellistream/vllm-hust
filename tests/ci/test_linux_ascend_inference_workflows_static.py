# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATHS = (
    REPO_ROOT / ".github/workflows/linux-ascend-inference-smoke.yml",
    REPO_ROOT / ".github/workflows/linux-ascend-inference-regression.yml",
)


def test_inference_workflows_submit_signed_requests_to_112() -> None:
    for workflow_path in WORKFLOW_PATHS:
        text = workflow_path.read_text(encoding="utf-8")
        workflow = yaml.safe_load(text)
        job = workflow["jobs"]["submit-to-112"]

        assert "self-hosted" not in text
        assert job["uses"].endswith("/.github/workflows/evaluation-request.yml@main")
        assert job["with"]["repository"] == "vLLM-HUST/vllm-hust"
        assert job["with"]["plugin_repository"] == "vLLM-HUST/vllm-ascend-hust"
        assert job["with"]["repeat_count"] == 3
        assert job["with"]["npu_count"] == 1
        assert set(job["secrets"]) == {
            "evaluation_api_url",
            "evaluation_api_token",
            "evaluation_hmac_secret",
        }


def test_inference_workflows_only_accept_trusted_test_ready_prs() -> None:
    for workflow_path in WORKFLOW_PATHS:
        text = workflow_path.read_text(encoding="utf-8")
        assert (
            "github.event.pull_request.head.repo.full_name == github.repository" in text
        )
        assert "contains(github.event.pull_request.labels.*.name, 'ready')" in text
        assert "contains(github.event.pull_request.labels.*.name, 'verified')" in text
