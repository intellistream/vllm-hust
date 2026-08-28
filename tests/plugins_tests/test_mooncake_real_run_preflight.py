# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Standard
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = (
    ROOT / "examples" / "disaggregated" / "mooncake_connector" / "real_run_preflight.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("mooncake_real_run_preflight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_version_gate_handles_release_and_prerelease_versions() -> None:
    module = _module()
    assert module._version_at_least("0.3.8", (0, 3, 8))
    assert module._version_at_least("0.4.0rc1", (0, 3, 8))
    assert not module._version_at_least("0.3.7", (0, 3, 8))
    assert not module._version_at_least(None, (0, 3, 8))


def test_store_config_is_topology_specific(tmp_path: Path) -> None:
    module = _module()
    config = tmp_path / "mooncake.json"
    config.write_text(
        json.dumps(
            {
                "mode": "standalone-store",
                "master_server_address": "127.0.0.1:50051",
                "global_segment_size": 0,
                "protocol": "tcp",
            }
        ),
        encoding="utf-8",
    )
    assert module._store_config_probe(config, "store-standalone")["ok"]
    probe = module._store_config_probe(config, "store-embedded")
    assert not probe["ok"]
    assert "mode does not match selected topology" in probe["errors"]


def test_manifest_probe_requires_selected_topology_roles(tmp_path: Path) -> None:
    module = _module()
    bundle_dir = tmp_path / "vllm" / "plugins" / "builtin_kv_bundles"
    bundle_dir.mkdir(parents=True)
    components = [
        {"component_id": component}
        for component in module.TOPOLOGY_COMPONENTS["direct"]
    ]
    (bundle_dir / "mooncake.bundle.json").write_text(
        json.dumps(
            {
                "bundle_id": "vllm-core.mooncake-bridges",
                "components": components,
            }
        ),
        encoding="utf-8",
    )
    assert module._manifest_probe(tmp_path, "direct")["ok"]
    assert not module._manifest_probe(tmp_path, "combined-embedded")["ok"]


def test_blocked_preflight_writes_record_and_launches_nothing(
    monkeypatch, tmp_path: Path
) -> None:
    module = _module()
    project = tmp_path / "project"
    model = tmp_path / "model"
    project.mkdir()
    model.mkdir()
    monkeypatch.setattr(
        module,
        "_python_probe",
        lambda python: {
            "probe_ok": True,
            "modules": {
                "vllm": {"ok": True},
                "mooncake.engine": {"ok": False},
            },
            "mooncake_version": None,
        },
    )
    monkeypatch.setattr(module, "_manifest_probe", lambda root, topology: {"ok": True})
    monkeypatch.setattr(module, "_accelerator_probe", lambda kind, count: {"ok": True})
    monkeypatch.setattr(module, "_ports_probe", lambda ports: {"ok": True})
    monkeypatch.setattr(module, "_service_probe", lambda topology: {"ok": True})
    monkeypatch.setattr(module, "_git_revision", lambda path: "a" * 40)
    output = tmp_path / "evidence"
    args = argparse.Namespace(
        project_root=project,
        model=model,
        output_dir=output,
        python=Path(__file__),
        topology="direct",
        store_config=None,
        accelerator="ascend",
        expected_devices=2,
        ports=[8000],
    )
    record = module.build_record(args)
    assert record["provenance_label"] == "preflight-only"
    assert record["launch_performed"] is False
    assert record["ready_for_real_online"] is False
    assert json.loads((output / "preflight.json").read_text()) == record


def test_existing_output_is_never_overwritten(monkeypatch, tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "evidence"
    output.mkdir()
    sentinel = output / "owned"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(module, "_python_probe", lambda python: {"probe_ok": False})
    monkeypatch.setattr(module, "_manifest_probe", lambda root, topology: {"ok": False})
    monkeypatch.setattr(module, "_accelerator_probe", lambda kind, count: {"ok": False})
    monkeypatch.setattr(module, "_ports_probe", lambda ports: {"ok": False})
    monkeypatch.setattr(module, "_service_probe", lambda topology: {"ok": False})
    monkeypatch.setattr(module, "_git_revision", lambda path: None)
    args = argparse.Namespace(
        project_root=tmp_path,
        model=tmp_path,
        output_dir=output,
        python=Path(__file__),
        topology="direct",
        store_config=None,
        accelerator="ascend",
        expected_devices=2,
        ports=[8000],
    )
    record = module.build_record(args)
    output_check = next(
        check for check in record["checks"] if check["name"] == "output_directory_new"
    )
    assert not output_check["passed"]
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (output / "preflight.json").exists()


def test_virtual_environment_interpreter_path_is_preserved(
    monkeypatch, tmp_path: Path
) -> None:
    module = _module()
    real_python = tmp_path / "base-python"
    real_python.write_text("", encoding="utf-8")
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(real_python)
    observed: list[Path] = []

    def observe_python(python: Path) -> dict[str, bool]:
        observed.append(python)
        return {"probe_ok": False}

    monkeypatch.setattr(module, "_python_probe", observe_python)
    monkeypatch.setattr(module, "_manifest_probe", lambda root, topology: {"ok": False})
    monkeypatch.setattr(module, "_accelerator_probe", lambda kind, count: {"ok": False})
    monkeypatch.setattr(module, "_ports_probe", lambda ports: {"ok": False})
    monkeypatch.setattr(module, "_service_probe", lambda topology: {"ok": False})
    monkeypatch.setattr(module, "_git_revision", lambda path: None)
    args = argparse.Namespace(
        project_root=tmp_path,
        model=tmp_path,
        output_dir=tmp_path / "evidence",
        python=venv_python,
        topology="direct",
        store_config=None,
        accelerator="ascend",
        expected_devices=2,
        ports=[8000],
    )
    module.build_record(args)
    assert observed == [venv_python.absolute()]
