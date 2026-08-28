# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
from pathlib import Path

import pytest

import vllm.plugins as plugins
import vllm.plugins.startup as startup
from vllm.plugins.contracts import ComponentPermission
from vllm.plugins.startup import (
    ExtensionBundleAdmissionError,
    resolve_extension_startup,
)


def write_manifest(
    directory: Path,
    bundle_id: str,
    *,
    host_api_range: str = ">=1,<2",
    permissions: list[str] | None = None,
) -> Path:
    path = directory / f"{bundle_id}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "bundle_id": bundle_id,
                "bundle_version": "1.0.0",
                "host_api_range": host_api_range,
                "components": [
                    {
                        "component_id": "scheduler",
                        "contracts": ["vllm.scheduler.policy.v1"],
                        "execution_planes": ["scheduler"],
                        "isolation": "trusted_in_process",
                        "implementation_ref": "must_not_import:Policy",
                        "permissions": permissions or [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_resolver_builds_snapshot_without_importing_implementation(
    tmp_path: Path,
) -> None:
    first = write_manifest(tmp_path, "org.example.first")
    second = write_manifest(tmp_path, "org.example.second")

    resolution = resolve_extension_startup((first, second))

    assert [bundle.bundle_id for bundle in resolution.snapshot.bundles] == [
        "org.example.first",
        "org.example.second",
    ]
    assert resolution.disabled_bundle_ids == ()
    assert "must_not_import" not in sys.modules


def test_resolver_reports_bundles_disabled_by_allowlist(tmp_path: Path) -> None:
    first = write_manifest(tmp_path, "org.example.first")
    second = write_manifest(tmp_path, "org.example.second")

    resolution = resolve_extension_startup(
        (first, second),
        enabled_bundle_ids=("org.example.second",),
    )

    assert [bundle.bundle_id for bundle in resolution.snapshot.bundles] == [
        "org.example.second"
    ]
    assert resolution.disabled_bundle_ids == ("org.example.first",)


@pytest.mark.parametrize("host_api_range", [">=2", "not-a-specifier"])
def test_resolver_rejects_incompatible_or_invalid_host_range(
    tmp_path: Path,
    host_api_range: str,
) -> None:
    path = write_manifest(
        tmp_path,
        "org.example.incompatible",
        host_api_range=host_api_range,
    )

    with pytest.raises(ExtensionBundleAdmissionError, match="host"):
        resolve_extension_startup((path,))


def test_resolver_applies_explicit_permission_policy(tmp_path: Path) -> None:
    path = write_manifest(
        tmp_path,
        "org.example.networked",
        permissions=["network_egress"],
    )

    with pytest.raises(ExtensionBundleAdmissionError, match="denied by host policy"):
        resolve_extension_startup((path,), allowed_permissions=())

    resolution = resolve_extension_startup(
        (path,),
        allowed_permissions=(ComponentPermission.NETWORK_EGRESS,),
    )
    assert len(resolution.snapshot.components) == 1


def test_resolver_rejects_duplicate_manifest_paths(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, "org.example.duplicate-path")

    with pytest.raises(ExtensionBundleAdmissionError, match="paths must be unique"):
        resolve_extension_startup((path, path))


def test_resolver_rejects_unknown_enabled_bundle_id(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, "org.example.configured")

    with pytest.raises(ExtensionBundleAdmissionError, match="no configured manifest"):
        resolve_extension_startup(
            (path,),
            enabled_bundle_ids=("org.example.missing",),
        )


def test_general_loader_resolves_snapshot_before_legacy_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def resolve() -> None:
        events.append("typed-snapshot")

    def load_legacy(*, group: str) -> dict:
        events.append(f"legacy:{group}")
        return {}

    monkeypatch.setattr(plugins, "plugins_loaded", False)
    monkeypatch.setattr(startup, "get_configured_extension_startup", resolve)
    monkeypatch.setattr(plugins, "load_plugins_by_group", load_legacy)

    plugins.load_general_plugins()

    assert events == ["typed-snapshot", "legacy:vllm.general_plugins"]


def test_snapshot_admission_failure_does_not_mark_legacy_plugins_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject() -> None:
        raise ExtensionBundleAdmissionError("rejected")

    monkeypatch.setattr(plugins, "plugins_loaded", False)
    monkeypatch.setattr(startup, "get_configured_extension_startup", reject)

    with pytest.raises(ExtensionBundleAdmissionError, match="rejected"):
        plugins.load_general_plugins()

    assert plugins.plugins_loaded is False
