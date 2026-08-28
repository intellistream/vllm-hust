#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed readiness check for matched Mooncake connector runs.

The command launches no service, installs no package, and reports no benchmark
result. It writes a provenance record for a later process-owned runner.
"""

from __future__ import annotations

# Standard
import argparse
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

MINIMUM_MOONCAKE_VERSION = (0, 3, 8)
REQUIRED_RUNTIME_IMPORTS = (
    "vllm",
    "vllm.engine.arg_utils",
    "mooncake.engine",
)
TOPOLOGY_COMPONENTS = {
    "direct": ("direct-scheduler", "direct-worker", "direct-telemetry"),
    "store-embedded": ("store-scheduler", "store-worker", "store-telemetry"),
    "store-standalone": ("store-scheduler", "store-worker", "store-telemetry"),
    "combined-embedded": (
        "direct-scheduler",
        "direct-worker",
        "direct-telemetry",
        "store-scheduler",
        "store-worker",
        "store-telemetry",
    ),
    "combined-standalone": (
        "direct-scheduler",
        "direct-worker",
        "direct-telemetry",
        "store-scheduler",
        "store-worker",
        "store-telemetry",
    ),
}


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": passed, "evidence": evidence}


def _git_revision(path: Path) -> str | None:
    result = _run(["git", "rev-parse", "HEAD"], cwd=path)
    return result.stdout.strip() if result.returncode == 0 else None


def _last_json_object(output: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _python_probe(python: Path, project_root: Path) -> dict[str, Any]:
    probe = """
import importlib
import importlib.metadata
import json

modules = {}
for name in ("vllm", "vllm.engine.arg_utils", "mooncake.engine"):
    try:
        module = importlib.import_module(name)
        modules[name] = {"ok": True, "file": getattr(module, "__file__", None)}
    except Exception as exc:
        modules[name] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
try:
    version = importlib.metadata.version("mooncake-transfer-engine")
except importlib.metadata.PackageNotFoundError:
    version = None
print(json.dumps({"modules": modules, "mooncake_version": version}, sort_keys=True))
"""
    result = _run([str(python), "-c", probe], cwd=project_root)
    if result.returncode != 0:
        return {
            "probe_ok": False,
            "error": result.stderr.strip() or result.stdout.strip(),
        }
    payload = _last_json_object(result.stdout)
    if payload is None:
        return {"probe_ok": False, "error": "probe returned non-JSON output"}
    payload["probe_ok"] = True
    return payload


def _cli_probe(python: Path, project_root: Path) -> dict[str, Any]:
    """Require the actual service CLI to construct under the selected source tree."""
    result = _run(
        [str(python), "-m", "vllm.entrypoints.cli.main", "--help"],
        cwd=project_root,
    )
    return {
        "ok": result.returncode == 0 and "serve" in result.stdout,
        "command": [
            str(python),
            "-m",
            "vllm.entrypoints.cli.main",
            "--help",
        ],
        "returncode": result.returncode,
        "error": result.stderr.strip() or None,
    }


def _version_at_least(version: str | None, minimum: tuple[int, int, int]) -> bool:
    if version is None:
        return False
    numeric: list[int] = []
    for part in version.split(".")[:3]:
        digits = ""
        for character in part:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            return False
        numeric.append(int(digits))
    return len(numeric) == 3 and tuple(numeric) >= minimum


def _source_is_under(source: str | None, project_root: Path) -> bool:
    if source is None:
        return False
    try:
        Path(source).resolve().relative_to(project_root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _runtime_imports_ready(modules: dict[str, Any]) -> bool:
    return all(modules.get(name, {}).get("ok") for name in REQUIRED_RUNTIME_IMPORTS)


def _manifest_probe(project_root: Path, topology: str) -> dict[str, Any]:
    path = (
        project_root
        / "vllm"
        / "plugins"
        / "builtin_kv_bundles"
        / "mooncake.bundle.json"
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "path": str(path), "error": str(exc)}
    components = {item.get("component_id") for item in payload.get("components", [])}
    required = set(TOPOLOGY_COMPONENTS[topology])
    missing = sorted(required - components)
    return {
        "ok": payload.get("bundle_id") == "vllm-core.mooncake-bridges" and not missing,
        "path": str(path.resolve()),
        "bundle_id": payload.get("bundle_id"),
        "required_components": sorted(required),
        "missing_components": missing,
    }


def _store_config_probe(path: Path | None, topology: str) -> dict[str, Any]:
    if topology == "direct":
        return {"ok": True, "required": False}
    if path is None:
        return {"ok": False, "required": True, "error": "store config is required"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "required": True, "path": str(path), "error": str(exc)}
    expected_mode = (
        "standalone-store" if topology.endswith("standalone") else "embedded"
    )
    segment_size = str(payload.get("global_segment_size", "")).strip().upper()
    zero_segment = segment_size in {"0", "0B", "0KB", "0MB", "0GB", "0TB"}
    errors: list[str] = []
    if payload.get("mode", "embedded") != expected_mode:
        errors.append("mode does not match selected topology")
    if not payload.get("master_server_address"):
        errors.append("master_server_address is required")
    if payload.get("protocol") not in {"tcp", "rdma"}:
        errors.append("protocol must be tcp or rdma")
    if expected_mode == "embedded" and zero_segment:
        errors.append("embedded mode requires a nonzero global_segment_size")
    if expected_mode == "standalone-store" and not zero_segment:
        errors.append("standalone-store requires global_segment_size=0")
    return {
        "ok": not errors,
        "required": True,
        "path": str(path.resolve()),
        "mode": payload.get("mode", "embedded"),
        "errors": errors,
    }


def _accelerator_probe(
    kind: str, expected_devices: int, minimum_free_memory_mib: int
) -> dict[str, Any]:
    if kind == "nvidia":
        command = [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
        result = _run(command)
        devices = []
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3:
                continue
            try:
                free_mib, total_mib = int(fields[1]), int(fields[2])
            except ValueError:
                continue
            devices.append(
                {
                    "id": fields[0],
                    "free_memory_mib": free_mib,
                    "total_memory_mib": total_mib,
                    "eligible": free_mib >= minimum_free_memory_mib,
                }
            )
        ids = [device["id"] for device in devices if device["eligible"]]
    else:
        command = ["npu-smi", "info", "-l"]
        result = _run(command)
        ids = sorted(
            {
                line.partition(":")[2].strip()
                for line in result.stdout.splitlines()
                if "NPU ID" in line and line.partition(":")[2].strip().isdigit()
            }
        )
        devices = [{"id": device_id, "eligible": True} for device_id in ids]
    return {
        "ok": result.returncode == 0 and len(ids) >= expected_devices,
        "kind": kind,
        "expected_devices": expected_devices,
        "minimum_free_memory_mib": minimum_free_memory_mib,
        "eligible_device_ids": ids,
        "devices": devices,
        "error": result.stderr.strip() if result.returncode else None,
    }


def _ports_probe(ports: list[int]) -> dict[str, Any]:
    unavailable: list[int] = []
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                candidate.bind(("127.0.0.1", port))
            except OSError:
                unavailable.append(port)
    return {"ok": not unavailable, "ports": ports, "unavailable": unavailable}


def _service_probe(topology: str) -> dict[str, Any]:
    required: list[str] = []
    if topology != "direct":
        required.append("mooncake_master")
    if topology.endswith("standalone"):
        required.append("mooncake_client")
    resolved = {name: shutil.which(name) for name in required}
    return {"ok": all(resolved.values()), "required": required, "resolved": resolved}


def build_record(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.resolve()
    model = args.model.resolve()
    output_dir = args.output_dir.resolve()
    python = args.python.absolute()
    store_config = args.store_config.resolve() if args.store_config else None
    output_is_new = not output_dir.exists()
    python_probe = (
        _python_probe(python, project_root)
        if python.is_file()
        else {"probe_ok": False, "error": "python executable does not exist"}
    )
    cli = (
        _cli_probe(python, project_root)
        if python.is_file() and project_root.is_dir()
        else {"ok": False, "error": "python executable or project root is missing"}
    )
    modules = python_probe.get("modules", {})
    manifest = _manifest_probe(project_root, args.topology)
    config = _store_config_probe(store_config, args.topology)
    accelerator = _accelerator_probe(
        args.accelerator, args.expected_devices, args.minimum_free_memory_mib
    )
    ports = _ports_probe(args.ports)
    services = _service_probe(args.topology)
    revision = _git_revision(project_root)
    checks = [
        _check("project_root", project_root.is_dir(), str(project_root)),
        _check("model", model.is_dir(), str(model)),
        _check("output_directory_new", output_is_new, str(output_dir)),
        _check("python", python.is_file(), str(python)),
        _check(
            "required_imports",
            python_probe.get("probe_ok", False) and _runtime_imports_ready(modules),
            python_probe,
        ),
        _check(
            "vllm_source",
            _source_is_under(modules.get("vllm", {}).get("file"), project_root),
            {
                "required_root": str(project_root),
                "imported_file": modules.get("vllm", {}).get("file"),
            },
        ),
        _check("vllm_cli", cli["ok"], cli),
        _check(
            "mooncake_version",
            _version_at_least(
                python_probe.get("mooncake_version"), MINIMUM_MOONCAKE_VERSION
            ),
            {
                "required": ">=0.3.8",
                "installed": python_probe.get("mooncake_version"),
            },
        ),
        _check("bundle_manifest", manifest["ok"], manifest),
        _check("store_config", config["ok"], config),
        _check("service_executables", services["ok"], services),
        _check("accelerator_inventory", accelerator["ok"], accelerator),
        _check("ports", ports["ok"], ports),
        _check("revision", revision is not None, {"vllm": revision}),
    ]
    record = {
        "provenance_label": "preflight-only",
        "ready_for_real_online": all(check["passed"] for check in checks),
        "launch_performed": False,
        "topology": args.topology,
        "checks": checks,
        "required_matrix": ["legacy", "typed", "rollback-to-legacy"],
        "configuration_invariants": {
            "legacy": "built-in connector names only; no typed selection",
            "typed": {
                "bundle_id": "vllm-core.mooncake-bridges",
                "components": list(TOPOLOGY_COMPONENTS[args.topology]),
            },
            "rollback-to-legacy": (
                "fresh process; built-in names; no typed selection or manifest"
            ),
        },
    }
    if output_is_new:
        output_dir.mkdir(parents=True, mode=0o700)
        (output_dir / "preflight.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--topology", choices=sorted(TOPOLOGY_COMPONENTS), required=True
    )
    parser.add_argument("--store-config", type=Path)
    parser.add_argument("--accelerator", choices=("nvidia", "ascend"), required=True)
    parser.add_argument("--expected-devices", type=int, default=2)
    parser.add_argument("--minimum-free-memory-mib", type=int, default=0)
    parser.add_argument(
        "--ports", type=int, nargs="+", default=[8000, 8010, 8020, 8998, 50051]
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    record = build_record(parse_args(argv))
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["ready_for_real_online"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
