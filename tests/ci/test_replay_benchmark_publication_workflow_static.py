# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/replay-ascend-benchmark-publication.yml"
)


def workflow_text() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def workflow_yaml() -> dict:
    return yaml.safe_load(workflow_text())


def test_replay_workflow_is_manual_main_only_and_github_hosted() -> None:
    workflow = workflow_yaml()
    job = workflow["jobs"]["replay-publication"]

    assert set(workflow[True]) == {"workflow_dispatch"}
    assert job["if"] == "${{ github.ref == 'refs/heads/main' }}"
    assert job["runs-on"] == "ubuntu-latest"
    assert "self-hosted" not in workflow_text()
    assert "ascend-benchmark" not in str(job["runs-on"])


def test_replay_workflow_has_minimal_permissions_and_no_hf_path() -> None:
    workflow = workflow_yaml()

    assert workflow["permissions"] == {
        "actions": "read",
        "contents": "read",
    }
    assert "HF_TOKEN" not in workflow_text()
    assert "publish_to_hf" not in workflow_text()


def test_replay_workflow_downloads_exact_source_run_artifact() -> None:
    text = workflow_text()

    assert "source_run_id:" in text
    assert "source_run_attempt:" in text
    assert "expected_target_sha:" in text
    assert "expected_benchmark_main_sha:" in text
    assert "confirmation:" in text
    assert "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093" in text
    assert (
        "name: ascend-benchmark-${{ inputs.source_run_id }}-"
        "${{ inputs.source_run_attempt }}" in text
    )
    assert "run-id: ${{ inputs.source_run_id }}" in text
    assert "repository: ${{ github.repository }}" in text


def test_writer_secret_is_scoped_to_publish_step() -> None:
    text = workflow_text()
    preflight_start = text.index(
        "      - name: Validate replay without writer credential"
    )
    publish_start = text.index("      - name: Publish and verify historical artifact")
    summary_start = text.index("      - name: Write replay summary")

    assert "REPLAY_WRITER_TOKEN" not in text[:publish_start]
    assert (
        "REPLAY_WRITER_TOKEN: "
        "${{ secrets.VLLM_HUST_BENCHMARK_GH_TOKEN }}"
        in text[publish_start:summary_start]
    )
    assert (
        "replay_benchmark_publication.sh preflight"
        in text[preflight_start:publish_start]
    )
    assert (
        "replay_benchmark_publication.sh publish" in text[publish_start:summary_start]
    )


def test_replay_checkouts_do_not_persist_credentials() -> None:
    text = workflow_text()

    assert text.count("persist-credentials: false") == 3
    assert "cancel-in-progress: false" in text
    assert "EXPECTED_SYNC_SCRIPT_SHA256:" in text
    assert "REPLAY_RECEIPT_FILE:" in text
    assert "jsonschema==4.23.0" in text
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in text
