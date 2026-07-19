# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/ci/validate_dual_editable_install.sh"
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/dual-editable-install.yml"


def read_script() -> str:
    return SCRIPT_PATH.read_text(encoding="utf-8")


def read_workflow() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def workflow_yaml() -> dict:
    return yaml.safe_load(read_workflow())


def test_dual_editable_workflow_is_manual_and_targets_arm64_runner():
    workflow = workflow_yaml()
    text = read_workflow()

    assert True in workflow
    assert "workflow_dispatch" in workflow[True]
    assert "pull_request" not in workflow[True]
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["validate-dual-editable-install"]["runs-on"] == [
        "self-hosted",
        "ascend",
        "910b",
        "docker",
    ]
    assert "actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0" in text


def test_dual_editable_workflow_accepts_both_repo_refs_and_keep_work_dir():
    inputs = workflow_yaml()[True]["workflow_dispatch"]["inputs"]
    job_env = workflow_yaml()["jobs"]["validate-dual-editable-install"]["env"]
    text = read_workflow()

    assert inputs["vllm_hust_ref"]["default"] == "main"
    assert inputs["vllm_ascend_hust_ref"]["default"] == "main"
    assert inputs["keep_work_dir"]["default"] == "0"
    assert job_env["VLLM_HUST_REPO"] == "${{ github.workspace }}/vllm-hust"
    assert job_env["VLLM_ASCEND_HUST_REPO"] == (
        "${{ github.workspace }}/vllm-ascend-hust"
    )
    assert "ref: ${{ inputs.vllm_hust_ref }}" in text
    assert "ref: ${{ inputs.vllm_ascend_hust_ref }}" in text
    assert "KEEP_WORK_DIR: ${{ inputs.keep_work_dir }}" in text


def test_dual_editable_workflow_checks_out_plugin_and_runs_validation_script():
    text = read_workflow()

    assert "repository: vLLM-HUST/vllm-ascend-hust" in text
    assert "path: vllm-ascend-hust" in text
    assert "python3 -m pip install --user uv" in text
    assert "bash vllm-hust/scripts/ci/validate_dual_editable_install.sh" in text


def test_dual_editable_script_uses_clean_python_312_uv_environment():
    script = read_script()

    assert "set -euo pipefail" in script
    assert "PYTHON_VERSION=${PYTHON_VERSION:-3.12}" in script
    assert 'uv venv --python "${PYTHON_VERSION}" "${WORK_DIR}/.venv"' in script
    assert 'echo "UV_NO_CACHE=${UV_NO_CACHE:-<unset>}"' in script
    assert 'python3 -m pip install --user uv' in script


def test_dual_editable_script_installs_both_repos_with_expected_flags():
    script = read_script()

    assert "VLLM_HUST_REPO" in script
    assert "VLLM_ASCEND_HUST_REPO" in script
    assert "VLLM_USE_PRECOMPILED=1 uv pip install" in script
    assert "--torch-backend=auto" in script
    assert "COMPILE_CUSTOM_KERNELS=0 uv pip install" in script
    assert "--no-deps" in script


def test_dual_editable_script_keeps_issue_131_acceptance_checks():
    script = read_script()

    assert "import vllm" in script
    assert "import vllm_ascend" in script
    assert 'uv pip check --python "${PYTHON_BIN}"' in script
    assert 'uv pip freeze --python "${PYTHON_BIN}"' in script
    assert "vllm_module" in script
    assert "vllm_ascend_module" in script
    assert "vllm-ascend-hust" in script
    assert "torch-npu" in script
    assert "fastapi" in script
