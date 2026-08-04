# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import json
import os
import stat
from pathlib import Path

MODULE_PATH = (
    Path(__file__).parents[2]
    / "vllm"
    / "v1"
    / "kv_offload"
    / "cpu"
    / "b134_descriptor_layout.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("b134_layout_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_is_disabled_without_output_directory(monkeypatch) -> None:
    monkeypatch.delenv("B134_DESCRIPTOR_LAYOUT_DIR", raising=False)
    module = _load_module()
    assert module.CAPTURE_ENABLED is False
    assert module.capture_descriptor_layout(1, "d2h", []) is None


def test_capture_writes_only_region_relative_offsets(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("B134_DESCRIPTOR_LAYOUT_DIR", str(tmp_path))
    module = _load_module()
    output = module.capture_descriptor_layout(
        7,
        "d2h",
        [
            {
                "direction": "d2h",
                "dst_offset": 0,
                "dst_region": "dst_tensor_0",
                "size": 4096,
                "src_offset": 8192,
                "src_region": "src_tensor_0",
            }
        ],
    )
    assert output is not None
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "kv-transfer-descriptor-layout/v1"
    assert payload["evidence_label"] == "existing-server-probe"
    assert payload["descriptors"][0]["src_offset"] == 8192
    assert "address" not in output.read_text(encoding="utf-8")
    # os.open(mode=0o600) is a POSIX permission; Windows does not implement
    # it, so the mode bit assertion is only meaningful on POSIX platforms.
    if os.name != "nt":
        assert stat.S_IMODE(output.stat().st_mode) == 0o600
