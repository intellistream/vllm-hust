# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from tools.verify_extension_wheel import (
    REQUIRED_BUILTIN_BUNDLES,
    REQUIRED_CONTRACT_FILES,
    WheelContentError,
    verify_wheel,
)


def _write_wheel(path: Path, members: set[str]) -> None:
    with ZipFile(path, "w") as wheel:
        for member in members:
            wheel.writestr(member, json.dumps({"member": member}))


def test_verify_wheel_accepts_complete_contract(tmp_path: Path) -> None:
    wheel_path = tmp_path / "complete.whl"
    required = set(REQUIRED_CONTRACT_FILES | REQUIRED_BUILTIN_BUNDLES)
    _write_wheel(wheel_path, required)

    assert set(verify_wheel(wheel_path)) == required


def test_verify_wheel_rejects_missing_contract_member(tmp_path: Path) -> None:
    wheel_path = tmp_path / "missing.whl"
    required = set(REQUIRED_CONTRACT_FILES | REQUIRED_BUILTIN_BUNDLES)
    missing = required.pop()
    _write_wheel(wheel_path, required)

    with pytest.raises(WheelContentError, match=missing):
        verify_wheel(wheel_path)


def test_verify_wheel_optionally_requires_rust_frontend(tmp_path: Path) -> None:
    wheel_path = tmp_path / "rust.whl"
    required = set(REQUIRED_CONTRACT_FILES | REQUIRED_BUILTIN_BUNDLES)
    required.update({"vllm/vllm-rs", "vllm/_rust_tool_parser.abi3.so"})
    _write_wheel(wheel_path, required)

    verified = verify_wheel(wheel_path, require_rust_frontend=True)

    assert "vllm/vllm-rs" in verified
    assert "vllm/_rust_tool_parser.abi3.so" in verified
