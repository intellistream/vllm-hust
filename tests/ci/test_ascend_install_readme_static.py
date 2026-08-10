# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_readme_documents_ascend_empty_target_install_contract():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("## Install For Development")
    end = readme.index("## Run On Ascend/NPU", start)
    install_section = readme[start:end]

    commands = [
        "-r requirements/common.txt",
        "-r /path/to/vllm-ascend-hust/requirements.txt",
        "-r requirements/build/empty.txt",
        "--extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi",
        "--index-strategy unsafe-best-match",
        "VLLM_TARGET_DEVICE=empty uv pip install -e .",
        "--no-build-isolation --no-deps",
    ]
    positions = [install_section.index(command) for command in commands]

    old_install = "VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto"

    assert positions == sorted(positions)
    assert old_install not in install_section
