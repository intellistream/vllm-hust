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

    assert "confirmation:" in text
    assert 'SOURCE_RUN_ID: "31004708110"' in text
    assert 'SOURCE_RUN_ATTEMPT: "1"' in text
    assert 'EXPECTED_SOURCE_JOB_ID: "92301693209"' in text
    assert 'EXPECTED_TARGET_SHA: "4a6f5b1ce78ace4b2b4d77229a9707a7f54ba5d0"' in text
    assert (
        'EXPECTED_BENCHMARK_MAIN_SHA: "734f3fc3cc7a809889ac1780a1fd980e98226cef"'
        in text
    )
    assert "source_run_id:" not in text
    assert "expected_target_sha:" not in text
    assert "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093" in text
    assert "name: ascend-benchmark-31004708110-1" in text
    assert "run-id: 31004708110" in text
    assert "repository: ${{ github.repository }}" in text


def test_replay_workflow_verifies_source_run_job_and_artifact_before_download() -> None:
    text = workflow_text()

    verify_start = text.index(
        "      - name: Verify approved source run, job, and artifact metadata"
    )
    download_start = text.index("      - name: Download immutable source artifact")
    verify_block = text[verify_start:download_start]

    assert "actions/runs/${SOURCE_RUN_ID}" in verify_block
    assert "actions/runs/${SOURCE_RUN_ID}/jobs?per_page=100" in verify_block
    assert "actions/runs/${SOURCE_RUN_ID}/artifacts?per_page=100" in verify_block
    assert "verify_replay_source_run.py" in verify_block
    assert '--source-job-id "$EXPECTED_SOURCE_JOB_ID"' in verify_block
    assert "env -u GH_TOKEN python3" in verify_block
    assert '.conclusion == "success"' not in verify_block


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
