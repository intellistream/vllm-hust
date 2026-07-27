# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static checks for the dual-editable install workflow and validation script.

These tests ensure that PR #137's CI artefacts retain the critical
verification steps and do not regress over time.

How to run manually:
    export VLLM_ASCEND_HUST_REPO=~/vllm/vllm-ascend-hust
    export VLLM_ASCEND_HUST_REF=HEAD
    export VLLM_HUST_REPO=~/vllm/vllm-hust
    export VLLM_HUST_REF=HEAD
    bash scripts/ci/validate_dual_editable_install.sh
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/dual-editable-install.yml"
SCRIPT_PATH = REPO_ROOT / "scripts/ci/validate_dual_editable_install.sh"


def test_workflow_exists():
    assert WORKFLOW_PATH.exists(), f"Missing workflow: {WORKFLOW_PATH}"


def test_validation_script_exists():
    assert SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}"


def test_script_rejects_url_for_ascend_repo():
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "http*" in text
    assert "must be a local path, not a URL" in text


def test_script_validates_directories_exist():
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert '[[ ! -d "$VLLM_ASCEND_HUST_REPO" ]]' in text
    assert '[[ ! -d "$VLLM_HUST_REPO" ]]' in text


def test_workflow_has_manual_trigger_with_ref_inputs():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "vllm_hust_ref:" in text
    assert "vllm_ascend_hust_ref:" in text


def test_workflow_clones_ascend_repo():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "Checkout vllm-ascend-hust" in text
    assert "vLLM-HUST/vllm-ascend-hust.git" in text
    assert "git clone --depth 1" in text


def test_workflow_runs_validation_script():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "validate_dual_editable_install.sh" in text


def test_workflow_does_not_require_ascend_hardware():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "runs-on: ubuntu-latest" in text
    assert "ascend" not in text.lower() or "vllm-ascend-hust" in text


def test_script_installs_both_packages_editable():
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "pip3 install -e" in text
    assert "VLLM_ASCEND_HUST_REPO" in text
    assert "VLLM_TARGET_DEVICE=empty" in text


def test_script_sets_soc_version_for_ascend_build():
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "SOC_VERSION" in text
    assert "ascend910b1" in text


def test_script_runs_import_smoke_tests():
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "import vllm" in text
    assert "import vllm_ascend" in text


def test_script_runs_dependency_check():
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "pip3 check" in text


def test_script_prints_dependency_versions():
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "fastapi" in text
    assert "transformers" in text


def test_script_uses_no_deps_for_editable_installs():
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "--no-deps" in text
