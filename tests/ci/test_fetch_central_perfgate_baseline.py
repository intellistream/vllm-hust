from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / ".github/workflows/scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "fetch_central_perfgate_baseline.py"
SPEC = importlib.util.spec_from_file_location("fetch_central_perfgate_baseline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

TARGET_SHA = "1" * 40
PLUGIN_SHA = "2" * 40
BENCHMARK_SHA = "3" * 40
SPEC_HASH = "a" * 64


def _spec() -> dict[str, object]:
    return {
        "id": "perfgate-ascend-qwen25-3b-910b2",
        "scenario": "random-online",
        "server_parameters": {},
        "client_parameters": {},
    }


def _modules(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    calls: dict[str, object] = {}

    class Record:
        def __init__(self, **values: str) -> None:
            self.values = values

    class SameSpec:
        @staticmethod
        def load_benchmark_spec(_path: Path) -> dict[str, object]:
            return _spec()

        @staticmethod
        def build_same_spec_payload(
            payload: dict[str, object], **_kwargs: object
        ) -> dict[str, object]:
            server = payload["server_parameters"]
            client = payload["client_parameters"]
            assert isinstance(server, dict)
            assert isinstance(client, dict)
            assert server["no_enable_chunked_prefill"] is True
            assert server["no_enable_prefix_caching"] is True
            assert client["temperature"] == 0
            return {
                "scenario": "random-online",
                "spec_id": "perfgate-ascend-qwen25-3b-910b2",
                "resolved_spec_hash": SPEC_HASH,
            }

    class Protocol:
        BaselineIdentity = Record
        BaselineProvenance = Record

        @staticmethod
        def fetch_baseline(*args: object, **kwargs: object) -> Path:
            calls["args"] = args
            calls["kwargs"] = kwargs
            output = args[1]
            assert isinstance(output, Path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('{"metrics": {}}\n', encoding="utf-8")
            return output

    monkeypatch.setattr(
        MODULE,
        "_load_benchmark_modules",
        lambda _repository: (Protocol, SameSpec),
    )
    monkeypatch.setattr(
        MODULE,
        "_git_sha",
        lambda repository: PLUGIN_SHA if "plugin" in str(repository) else BENCHMARK_SHA,
    )
    monkeypatch.setattr(
        MODULE,
        "_package_version",
        lambda distribution, _module: {
            "torch": "2.10.0",
            "torch-npu": "2.10.0.post1",
        }[distribution],
    )
    return calls


def test_main_fetches_exact_identity_and_writes_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _modules(monkeypatch)
    env_file = tmp_path / "github.env"
    spec_file = tmp_path / "spec.json"
    spec_file.write_text("{}\n", encoding="utf-8")
    central = tmp_path / "central"
    benchmark = tmp_path / "benchmark"
    plugin = tmp_path / "plugin"
    (benchmark / "src").mkdir(parents=True)
    (plugin / ".git").mkdir(parents=True)

    status = MODULE.main(
        [
            "--central-repository-root",
            str(central),
            "--output",
            str(tmp_path / "baseline.json"),
            "--target-repository",
            "vLLM-HUST/vllm-hust",
            "--target-sha",
            TARGET_SHA,
            "--scenario",
            "random-online",
            "--spec-file",
            str(spec_file),
            "--benchmark-git-repository",
            str(benchmark),
            "--plugin-git-repository",
            str(plugin),
            "--hardware-chip-model",
            "910B2",
            "--cann-version",
            "9.0.0",
            "--github-env",
            str(env_file),
        ]
    )

    assert status == 0
    identity = calls["kwargs"]
    assert isinstance(identity, dict)
    assert identity["expected_provenance"].values == {
        "vllm_hust_sha": TARGET_SHA,
        "vllm_ascend_hust_sha": PLUGIN_SHA,
        "benchmark_runner_sha": BENCHMARK_SHA,
        "hardware_chip_model": "910B2",
        "cann_version": "9.0.0",
        "torch_version": "2.10.0",
        "torch_npu_version": "2.10.0.post1",
    }
    assert calls["args"][2].values == {
        "target_repository": "vLLM-HUST/vllm-hust",
        "target_sha": TARGET_SHA,
        "scenario": "random-online",
        "spec_id": "perfgate-ascend-qwen25-3b-910b2",
        "spec_hash": SPEC_HASH,
    }
    env_text = env_file.read_text(encoding="utf-8")
    assert "PERFGATE_BASELINE_SOURCE=exact" in env_text
    assert f"PERFGATE_BASELINE_COMMIT={TARGET_SHA}" in env_text


def test_main_rejects_scenario_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _modules(monkeypatch)
    monkeypatch.setattr(
        MODULE,
        "_resolve_spec_identity",
        lambda *_args: (_ for _ in ()).throw(
            ValueError("resolved same-spec scenario mismatch")
        ),
    )

    status = MODULE.main(
        [
            "--central-repository-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "baseline.json"),
            "--target-repository",
            "vLLM-HUST/vllm-hust",
            "--target-sha",
            TARGET_SHA,
            "--scenario",
            "random-online",
            "--spec-file",
            str(tmp_path / "spec.json"),
            "--benchmark-git-repository",
            str(tmp_path),
            "--plugin-git-repository",
            str(tmp_path),
            "--hardware-chip-model",
            "910B2",
            "--cann-version",
            "9.0.0",
        ]
    )

    assert status == 2
    assert "scenario mismatch" in capsys.readouterr().err


def test_fetch_shell_preserves_report_and_enforce_missing_baseline_semantics(
    tmp_path: Path,
) -> None:
    script = SCRIPT_DIR / "perfgate_fetch_baseline.sh"
    env_file = tmp_path / "github.env"
    common_env = {
        **os.environ,
        "GITHUB_ENV": str(env_file),
        "GITHUB_REPOSITORY": "vLLM-HUST/vllm-hust",
        "PERFGATE_BASELINE_OUTPUT_DIR": str(tmp_path / "output"),
        "SAME_SPEC_SPEC_FILE": str(tmp_path / "missing-spec.json"),
    }

    report = subprocess.run(
        ["bash", str(script), TARGET_SHA],
        env={**common_env, "PERFGATE_MODE": "report"},
        capture_output=True,
        text=True,
        check=False,
    )
    enforce = subprocess.run(
        ["bash", str(script), TARGET_SHA],
        env={**common_env, "PERFGATE_MODE": "enforce"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert report.returncode == 0
    assert enforce.returncode == 2
    assert "PERFGATE_BASELINE_AVAILABLE" in env_file.read_text(encoding="utf-8")
