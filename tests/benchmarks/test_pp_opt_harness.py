# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PP_OPT_DIR = REPO_ROOT / "benchmarks" / "pp_opt"
CONFIG_DIR = PP_OPT_DIR / "configs" / "qwen3_32b_pp2tp2_4x910b2"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def client_module():
    return load_module("pp_opt_client_test", PP_OPT_DIR / "client.py")


def test_client_records_output_token_checksum(client_module):
    token_ids = [4, 8, 15, 16, 23, 42]
    completions = SimpleNamespace(
        create=lambda **_: SimpleNamespace(
            choices=[SimpleNamespace(token_ids=token_ids)]
        )
    )
    client = object.__new__(client_module.VLLMClient)
    client._client = SimpleNamespace(completions=completions)
    client.model_name = "test-model"
    client.verbose = False
    request = SimpleNamespace(input_tokens=[1, 2], output_length=len(token_ids))

    result = client.send_request(request, request_id=0)

    expected = hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert result["success"] is True
    assert result["actual_output_length"] == len(token_ids)
    assert result["output_token_ids_sha256"] == expected


def test_client_leaves_checksum_empty_on_failure(client_module):
    def fail(**_):
        raise RuntimeError("request failed")

    client = object.__new__(client_module.VLLMClient)
    client._client = SimpleNamespace(completions=SimpleNamespace(create=fail))
    client.model_name = "test-model"
    client.verbose = False
    request = SimpleNamespace(input_tokens=[1], output_length=1)

    result = client.send_request(request, request_id=0)

    assert result["success"] is False
    assert result["actual_output_length"] == 0
    assert result["output_token_ids_sha256"] is None


def test_load_config_supports_explicit_model_dir_override(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "model-00001-of-00017.safetensors").touch()
    command = f"""
set -eu
SCRIPT_DIR={PP_OPT_DIR!s}
WORKSPACE_ROOT=/unused/workspace
CONFIG_PATH={CONFIG_DIR / "config.json"!s}
MODEL_DIR_OVERRIDE={model_dir!s}
source {PP_OPT_DIR / "load_config.sh"!s}
load_pp_opt_config qwen3_32b
printf '%s\n' "$MODEL_DIR" "$MODEL_CONFIG_PATH" "$EMBEDDING_SHARD"
"""

    result = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        str(model_dir),
        str(model_dir / "config.json"),
        str(model_dir / "model-00001-of-00017.safetensors"),
    ]


def test_910b2_config_and_fit_are_portable_and_bound():
    config_path = CONFIG_DIR / "config.json"
    fit_path = CONFIG_DIR / "fit_result.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fit = json.loads(fit_path.read_text(encoding="utf-8"))

    assert config["model_dir"] == "models/Qwen3-32B"
    assert not Path(fit["metadata"]["input_dir"]).is_absolute()
    assert (
        fit["metadata"]["config_sha256"]
        == hashlib.sha256(config_path.read_bytes()).hexdigest()
    )
    assert fit["metadata"]["calibration_run_id"] == ("910b2-pp2tp2-20260720T0244Z")


def test_run_experiment_records_reproducibility_and_determinism_controls():
    text = (PP_OPT_DIR / "run_experiment.sh").read_text(encoding="utf-8")

    export_position = text.index('export VLLM_BATCH_INVARIANT="${BATCH_INVARIANT}"')
    first_python_position = text.index('python "${SCRIPT_DIR}')
    assert export_position < first_python_position
    assert "BATCH_INVARIANT must be 0 or 1" in text
    assert 'REQUIRE_CLEAN_GIT="${REQUIRE_CLEAN_GIT:-1}"' in text
    assert "CORE_GIT_SHA" in text
    assert "ASCEND_GIT_SHA" in text
    assert "CORE_GIT_CLEAN" in text
    assert "ASCEND_GIT_CLEAN" in text
    assert "Formal benchmark requires clean core and Ascend worktrees" in text
    assert 'MODEL_LOAD_FORMAT="${MODEL_LOAD_FORMAT:-dummy}"' in text
    assert "MODEL_DIR_OVERRIDE" in text
    assert "MODEL_WEIGHT_INDEX_SHA256" in text
    assert "DECODE_BENCH_FILL_MEAN" in text
    assert "DECODE_BENCH_FILL_STD" in text
    assert "urllib.request.ProxyHandler({})" in text
    assert "--no-async-scheduling" in text


def test_profile_server_disables_async_scheduling():
    text = (PP_OPT_DIR / "profile_server.sh").read_text(encoding="utf-8")

    assert "--no-async-scheduling" in text
