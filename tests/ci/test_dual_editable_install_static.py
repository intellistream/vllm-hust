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
    # SHA-safe checkout: clone --no-checkout + fetch + checkout (not -b which
    # only works for branch/tag refs, not 40-char SHA).
    assert "git clone --no-checkout" in text
    assert "git fetch origin" in text
    assert "git checkout" in text
    assert "-b " not in text.split("Checkout")[1] if "Checkout" in text else True


def test_workflow_runs_validation_script():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "validate_dual_editable_install.sh" in text


def test_workflow_does_not_require_ascend_hardware():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "runs-on: ubuntu-latest" in text
    assert "ascend" not in text.lower() or "vllm-ascend-hust" in text


def test_script_installs_both_packages_editable():
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "python -m pip install -e" in text
    assert "VLLM_ASCEND_HUST_REPO" in text
    assert "VLLM_TARGET_DEVICE=empty" in text


def test_script_sets_soc_version_for_ascend_build():
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "SOC_VERSION" in text
    assert "ascend910b1" in text


def test_script_runs_metadata_verification():
    """Step 5 verifies installed package metadata, not actual imports.

    Both editable installs use --no-deps, so runtime dependencies like numpy
    are not installed and  would fail. The script verifies package
    metadata via importlib.metadata instead.
    """
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "importlib.metadata" in text
    assert "PackageNotFoundError" in text
    # Must NOT attempt actual import (would fail with --no-deps)
    assert "import vllm;" not in text
    assert "import vllm_ascend;" not in text


def test_script_runs_dependency_check():
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "python -m pip check" in text


def test_script_prints_dependency_versions():
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "fastapi" in text
    assert "transformers" in text


def test_script_uses_no_deps_for_editable_installs():
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "--no-deps" in text


def test_workflow_supports_sha_ref():
    """The ascend-hust checkout must support SHA refs, not just branch/tag.

    git clone -b <ref> fails for 40-char SHA. The workflow uses
    clone --no-checkout + fetch + checkout to support all ref types.
    """
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "git clone --no-checkout" in text
    assert "git fetch origin" in text


def test_workflow_has_pull_request_trigger():
    """The workflow must auto-trigger on PR for packaging/workflow paths."""
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "pull_request:" in text
    assert "pyproject.toml" in text
    assert "scripts/ci/validate_dual_editable_install.sh" in text


def test_script_narrows_to_metadata_smoke():
    """Step 7 must be scoped to metadata/build smoke, not dependency compat."""
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "metadata/build smoke" in text.lower() or "metadata smoke" in text.lower()
    # Must NOT claim to verify dependency compatibility
    assert "without dependency conflicts" not in text


def test_script_uses_python_m_pip():
    """Script must use 'python -m pip' consistently, not bare pip3."""
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "python -m pip install -e" in text
    assert "python -m pip check" in text
    assert "pip3 install -e" not in text
    assert "pip3 check" not in text
