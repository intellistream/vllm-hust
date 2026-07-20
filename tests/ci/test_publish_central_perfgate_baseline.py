from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github/workflows/scripts/publish_central_perfgate_baseline.py"
SPEC = importlib.util.spec_from_file_location(
    "publish_central_perfgate_baseline", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

TARGET_SHA = "1" * 40
PLUGIN_SHA = "2" * 40
BENCHMARK_SHA = "3" * 40
SPEC_HASH = "a" * 64


def _artifact() -> dict[str, object]:
    return {
        "metadata": {
            "github_repository": "vLLM-HUST/vllm-hust",
            "git_commit": TARGET_SHA,
            "runtime_provenance": {
                "engine": {"commit": TARGET_SHA},
                "plugin": {"commit": PLUGIN_SHA},
            },
        },
        "same_spec": {
            "scenario": "random-online",
            "spec_id": "perfgate-ascend-qwen25-3b-910b2",
            "resolved_spec_hash": SPEC_HASH,
        },
    }


def test_extract_artifact_fields_uses_exact_comparison_identity() -> None:
    fields = MODULE._extract_artifact_fields(_artifact())

    assert fields == {
        "target_repository": "vLLM-HUST/vllm-hust",
        "target_sha": TARGET_SHA,
        "scenario": "random-online",
        "spec_id": "perfgate-ascend-qwen25-3b-910b2",
        "spec_hash": SPEC_HASH,
        "vllm_hust_sha": TARGET_SHA,
        "vllm_ascend_hust_sha": PLUGIN_SHA,
    }


def test_extract_artifact_fields_rejects_incomplete_provenance() -> None:
    artifact = _artifact()
    metadata = artifact["metadata"]
    assert isinstance(metadata, dict)
    runtime = metadata["runtime_provenance"]
    assert isinstance(runtime, dict)
    del runtime["plugin"]

    with pytest.raises(ValueError, match="runtime_provenance.plugin"):
        MODULE._extract_artifact_fields(artifact)


def test_main_publishes_with_runtime_and_git_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "run_leaderboard.json"
    source.write_text(json.dumps(_artifact()) + "\n", encoding="utf-8")
    target_repo = tmp_path / "target"
    benchmark_repo = tmp_path / "benchmark"
    target_repo.mkdir()
    benchmark_repo.mkdir()
    published: dict[str, object] = {}

    class Record:
        def __init__(self, **values: str) -> None:
            self.values = values

    def publish_baseline(*args: object, **kwargs: object) -> str:
        published["args"] = args
        published["kwargs"] = kwargs
        return "published:baselines/example/run_leaderboard.json"

    protocol = SimpleNamespace(
        BaselineIdentity=Record,
        BaselineProvenance=Record,
        publish_baseline=publish_baseline,
    )
    monkeypatch.setattr(MODULE, "_load_protocol", lambda: protocol)
    monkeypatch.setattr(MODULE, "_git_sha", lambda _repository: BENCHMARK_SHA)
    monkeypatch.setattr(
        MODULE,
        "_package_version",
        lambda distribution, _module: {
            "torch": "2.10.0",
            "torch-npu": "2.10.0.post1",
        }[distribution],
    )

    status = MODULE.main(
        [
            "--source",
            str(source),
            "--target-git-repository",
            str(target_repo),
            "--benchmark-git-repository",
            str(benchmark_repo),
            "--central-remote",
            "git@example.invalid:vLLM-HUST/vllm-hust-benchmark.git",
            "--hardware-chip-model",
            "910B2",
            "--cann-version",
            "9.0.0",
            "--update-latest-pointer",
        ]
    )

    assert status == 0
    assert capsys.readouterr().out.startswith("published:")
    args = published["args"]
    identity = args[5]
    provenance = args[6]
    assert identity.values["target_sha"] == TARGET_SHA
    assert identity.values["spec_hash"] == SPEC_HASH
    assert provenance.values == {
        "vllm_hust_sha": TARGET_SHA,
        "vllm_ascend_hust_sha": PLUGIN_SHA,
        "benchmark_runner_sha": BENCHMARK_SHA,
        "hardware_chip_model": "910B2",
        "cann_version": "9.0.0",
        "torch_version": "2.10.0",
        "torch_npu_version": "2.10.0.post1",
    }
    assert published["kwargs"] == {"update_latest_pointer": True}
