# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import sys
from pathlib import Path, PurePosixPath

import pytest

from vllm import envs
from vllm.plugins.installed import (
    InstalledExtensionBundleError,
    discover_installed_extension_bundles,
)
from vllm.plugins.selection import (
    apply_cli_extension_selection,
    extension_selection_from_argv,
)
from vllm.plugins.startup import (
    ExtensionBundleAdmissionError,
    get_configured_extension_startup,
    resolve_configured_extension_startup,
)


class FakeDistribution:
    def __init__(self, root: Path, name: str = "example-plugin") -> None:
        self.root = root
        self.metadata = {"Name": name}
        self.version = "1.2.3"
        self.files = tuple(
            PurePosixPath(path.relative_to(root).as_posix())
            for path in root.rglob("*")
            if path.is_file()
        )
        self.direct_url: str | None = None

    def locate_file(self, path: PurePosixPath) -> Path:
        return self.root / path

    def read_text(self, filename: str) -> str | None:
        if filename == "direct_url.json":
            return self.direct_url
        return None


class FakeEntryPoint:
    group = "vllm.extension_bundles"

    def __init__(self, name: str, value: str, dist: FakeDistribution) -> None:
        self.name = name
        self.value = value
        self.dist = dist


def write_registration(
    tmp_path: Path,
    *,
    bundle_id: str = "org.example.performance",
    registered_id: str | None = None,
    implementation_ref: str = "must_not_import:PerformanceLogger",
) -> FakeEntryPoint:
    manifest_dir = tmp_path / "example_plugin" / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "vllm-hust-extension-v1.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "bundle_id": bundle_id,
                "bundle_version": "1.0.0",
                "host_api_range": ">=1,<2",
                "components": [
                    {
                        "component_id": "logger",
                        "contracts": ["vllm.stat_logger.v1"],
                        "execution_planes": ["api"],
                        "isolation": "trusted_in_process",
                        "implementation_ref": implementation_ref,
                        "permissions": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    distribution = FakeDistribution(tmp_path)
    return FakeEntryPoint(
        registered_id or bundle_id,
        "example_plugin.manifests",
        distribution,
    )


def test_installed_bundle_discovery_reads_static_metadata_without_import(
    tmp_path: Path,
) -> None:
    entry_point = write_registration(tmp_path)

    bundles = discover_installed_extension_bundles(
        (entry_point.name,), discovered_entry_points=(entry_point,)
    )

    assert bundles[0].bundle_id == "org.example.performance"
    assert bundles[0].distribution_name == "example-plugin"
    assert "must_not_import" not in sys.modules


def test_editable_install_uses_static_direct_url_without_import(
    tmp_path: Path,
) -> None:
    entry_point = write_registration(tmp_path / "src")
    entry_point.dist.root = tmp_path
    entry_point.dist.files = ()
    entry_point.dist.direct_url = json.dumps(
        {"url": tmp_path.as_uri(), "dir_info": {"editable": True}}
    )

    bundles = discover_installed_extension_bundles(
        (entry_point.name,), discovered_entry_points=(entry_point,)
    )

    assert bundles[0].manifest_path.parent.name == "manifests"
    assert "must_not_import" not in sys.modules


def test_unselected_invalid_registration_does_not_affect_selected_startup(
    tmp_path: Path,
) -> None:
    selected = write_registration(tmp_path / "selected")
    invalid = FakeEntryPoint(
        "org.example.unrelated",
        "unrelated.module:provider",
        FakeDistribution(tmp_path / "unrelated"),
    )

    bundles = discover_installed_extension_bundles(
        (selected.name,), discovered_entry_points=(invalid, selected)
    )

    assert [bundle.bundle_id for bundle in bundles] == [selected.name]


@pytest.mark.parametrize(
    ("registered_id", "value", "match"),
    [
        ("org.example.other", "example_plugin.manifests", "does not match"),
        (
            "org.example.performance",
            "example_plugin.manifests:provider",
            "static module directory",
        ),
    ],
)
def test_invalid_installed_registration_fails_closed(
    tmp_path: Path,
    registered_id: str,
    value: str,
    match: str,
) -> None:
    entry_point = write_registration(tmp_path, registered_id=registered_id)
    entry_point.value = value

    with pytest.raises(InstalledExtensionBundleError, match=match):
        discover_installed_extension_bundles(
            (registered_id,), discovered_entry_points=(entry_point,)
        )


def test_duplicate_installed_registration_fails_closed(tmp_path: Path) -> None:
    first = write_registration(tmp_path / "first")
    second = write_registration(tmp_path / "second")

    with pytest.raises(InstalledExtensionBundleError, match="multiple distributions"):
        discover_installed_extension_bundles(
            (first.name,), discovered_entry_points=(first, second)
        )


def test_explicit_selection_resolves_installed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry_point = write_registration(tmp_path)
    installed = discover_installed_extension_bundles(
        (entry_point.name,), discovered_entry_points=(entry_point,)
    )
    monkeypatch.setattr(
        "vllm.plugins.installed.discover_installed_extension_bundles",
        lambda bundle_ids: installed,
    )

    resolution = resolve_configured_extension_startup(
        (), enabled_bundle_ids=(entry_point.name,)
    )

    assert resolution.diagnostics().admitted_bundle_ids == (entry_point.name,)


def test_unset_selection_never_scans_installed_distributions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.plugins.installed.discover_installed_extension_bundles",
        lambda bundle_ids: pytest.fail("installed discovery must remain disabled"),
    )

    resolution = resolve_configured_extension_startup((), enabled_bundle_ids=None)

    assert resolution.snapshot.bundles == ()


def test_duplicate_explicit_selection_is_rejected() -> None:
    with pytest.raises(ExtensionBundleAdmissionError, match="duplicates"):
        resolve_configured_extension_startup(
            (),
            enabled_bundle_ids=("org.example.performance",) * 2,
        )


def test_cli_selection_is_early_ordered_and_conflict_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envs.disable_envs_cache()
    with monkeypatch.context() as context:
        context.delenv("VLLM_EXTENSION_BUNDLES", raising=False)
        selected = extension_selection_from_argv(
            [
                "vllm",
                "serve",
                "model",
                "--extension",
                "org.example.first",
                "--extension=org.example.second",
            ]
        )

        apply_cli_extension_selection(selected)

        assert os.environ["VLLM_EXTENSION_BUNDLES"] == (
            "org.example.first,org.example.second"
        )
        with pytest.raises(ValueError, match="conflicts"):
            apply_cli_extension_selection(("org.example.other",))
        os.environ.pop("VLLM_EXTENSION_BUNDLES")

    get_configured_extension_startup.cache_clear()
