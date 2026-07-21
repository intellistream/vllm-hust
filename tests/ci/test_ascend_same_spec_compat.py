from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github/workflows/scripts/ascend_same_spec_compat.py"
SPEC = importlib.util.spec_from_file_location("ascend_same_spec_compat", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_overlay_is_bounded_and_does_not_mutate_source() -> None:
    source = {
        "id": "perfgate-ascend-qwen25-3b-910b2",
        "server_parameters": {"gpu_memory_utilization": 0.92},
        "client_parameters": {"temperature": 0.4},
    }

    resolved = MODULE.apply_ascend_compatibility_overlay(source)

    assert source["server_parameters"] == {"gpu_memory_utilization": 0.92}
    assert resolved["server_parameters"] == {
        "gpu_memory_utilization": 0.85,
        "no_enable_chunked_prefill": True,
        "no_enable_prefix_caching": True,
    }
    assert resolved["client_parameters"] == {"temperature": 0.4}


@pytest.mark.parametrize("value", [None, "invalid", 0.9])
def test_overlay_uses_conservative_memory_limit(value: object) -> None:
    payload = {"server_parameters": {}}
    if value is not None:
        payload["server_parameters"]["gpu_memory_utilization"] = value

    resolved = MODULE.apply_ascend_compatibility_overlay(payload)

    assert resolved["server_parameters"]["gpu_memory_utilization"] == 0.85


def test_cli_writes_effective_spec(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "effective.json"
    source.write_text(
        json.dumps({"server_parameters": {}, "client_parameters": {}}) + "\n",
        encoding="utf-8",
    )

    status = MODULE.main(["--source", str(source), "--output", str(output)])

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["server_parameters"]["no_enable_chunked_prefill"] is True
    assert payload["client_parameters"]["temperature"] == 0
