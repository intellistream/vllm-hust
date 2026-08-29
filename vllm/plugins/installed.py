# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Static discovery of extension manifests from installed distributions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

import regex as re

from vllm.plugins.manifest import load_extension_bundle_manifest

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from vllm.plugins.contracts import ExtensionBundleDescriptor


EXTENSION_BUNDLES_ENTRY_POINT_GROUP = "vllm.extension_bundles"
EXTENSION_BUNDLE_MANIFEST_FILENAMES = (
    "vllm-hust-extension-v1.json",
    "extension-bundle-v1.json",
)

_MODULE_PATH = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")


class InstalledExtensionBundleError(ValueError):
    """Reject invalid or ambiguous installed Bundle registration metadata."""


@dataclass(frozen=True, slots=True)
class InstalledExtensionBundle:
    """An installed Bundle located without importing its Python package."""

    bundle_id: str
    distribution_name: str
    distribution_version: str
    manifest_path: Path
    descriptor: ExtensionBundleDescriptor


def _distribution_identity(entry_point: Any) -> tuple[str, str]:
    distribution = getattr(entry_point, "dist", None)
    if distribution is None:
        raise InstalledExtensionBundleError(
            f"installed Bundle {entry_point.name!r} has no distribution metadata"
        )
    name = distribution.metadata.get("Name")
    if not isinstance(name, str) or not name:
        raise InstalledExtensionBundleError(
            f"installed Bundle {entry_point.name!r} has no distribution name"
        )
    version = distribution.version
    if not isinstance(version, str) or not version:
        raise InstalledExtensionBundleError(
            f"installed Bundle {entry_point.name!r} has no distribution version"
        )
    return name, version


def _manifest_path(entry_point: Any) -> Path:
    if not _MODULE_PATH.fullmatch(entry_point.value):
        raise InstalledExtensionBundleError(
            f"installed Bundle {entry_point.name!r} must register a static "
            "module directory without an attribute, extras, or filesystem path"
        )

    distribution = entry_point.dist
    files = distribution.files
    if files is None:
        raise InstalledExtensionBundleError(
            f"distribution for installed Bundle {entry_point.name!r} has no "
            "installed-file metadata"
        )
    relative_paths = tuple(
        PurePosixPath(*entry_point.value.split("."), filename)
        for filename in EXTENSION_BUNDLE_MANIFEST_FILENAMES
    )
    matches = tuple(
        file for file in files if PurePosixPath(str(file)) in relative_paths
    )
    if not matches:
        matches = _editable_manifest_paths(distribution, relative_paths)
    if len(matches) != 1:
        raise InstalledExtensionBundleError(
            f"installed Bundle {entry_point.name!r} must contain exactly one "
            "supported manifest file beneath its registered module directory: "
            f"{[path.as_posix() for path in relative_paths]}"
        )
    match = matches[0]
    return match if isinstance(match, Path) else Path(distribution.locate_file(match))


def _editable_manifest_paths(
    distribution: Any,
    relative_paths: tuple[PurePosixPath, ...],
) -> tuple[Path, ...]:
    """Locate package data in a PEP 660 source tree without importing it."""
    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        return ()
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as error:
        raise InstalledExtensionBundleError(
            "editable distribution has invalid direct_url.json metadata"
        ) from error
    if direct_url.get("dir_info", {}).get("editable") is not True:
        return ()
    parsed = urlparse(direct_url.get("url", ""))
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise InstalledExtensionBundleError(
            "editable Bundle registration must use a local file direct URL"
        )
    project_root = Path(unquote(parsed.path))
    if not project_root.is_absolute():
        raise InstalledExtensionBundleError(
            "editable Bundle registration has a non-absolute project path"
        )
    source_roots = (project_root, project_root / "src")
    return tuple(
        candidate
        for source_root in source_roots
        for relative_path in relative_paths
        if (candidate := source_root / relative_path).is_file()
    )


def _load_registration(entry_point: Any) -> InstalledExtensionBundle:
    distribution_name, distribution_version = _distribution_identity(entry_point)
    manifest_path = _manifest_path(entry_point)
    try:
        descriptor = load_extension_bundle_manifest(manifest_path)
    except ValueError as error:
        raise InstalledExtensionBundleError(
            f"installed Bundle {entry_point.name!r} from distribution "
            f"{distribution_name!r} has an invalid manifest: {error}"
        ) from error
    if descriptor.bundle_id != entry_point.name:
        raise InstalledExtensionBundleError(
            f"installed Bundle registration {entry_point.name!r} does not match "
            f"manifest bundle_id {descriptor.bundle_id!r}"
        )
    return InstalledExtensionBundle(
        bundle_id=descriptor.bundle_id,
        distribution_name=distribution_name,
        distribution_version=distribution_version,
        manifest_path=manifest_path,
        descriptor=descriptor,
    )


def discover_installed_extension_bundles(
    bundle_ids: Iterable[str] | None = None,
    *,
    discovered_entry_points: Sequence[Any] | None = None,
) -> tuple[InstalledExtensionBundle, ...]:
    """Discover installed static Bundle registrations without importing code.

    Args:
        bundle_ids: Optional ordered selection. Unselected registrations are not
            parsed, so unrelated installed packages cannot affect startup.
        discovered_entry_points: Test-only metadata injection.

    Returns:
        Valid registrations in requested order, or sorted by Bundle ID when no
        selection is supplied.

    Raises:
        InstalledExtensionBundleError: If selected metadata is ambiguous or
            invalid.
    """
    selected = None if bundle_ids is None else tuple(bundle_ids)
    selected_set = None if selected is None else frozenset(selected)
    registrations = (
        tuple(entry_points(group=EXTENSION_BUNDLES_ENTRY_POINT_GROUP))
        if discovered_entry_points is None
        else tuple(discovered_entry_points)
    )
    candidates: dict[str, list[Any]] = {}
    for entry_point in registrations:
        if selected_set is not None and entry_point.name not in selected_set:
            continue
        candidates.setdefault(entry_point.name, []).append(entry_point)

    for bundle_id, entries in candidates.items():
        if len(entries) != 1:
            distributions = sorted(
                _distribution_identity(entry)[0] for entry in entries
            )
            raise InstalledExtensionBundleError(
                f"installed Bundle {bundle_id!r} is registered by multiple "
                f"distributions: {distributions}"
            )

    loaded = {
        bundle_id: _load_registration(entries[0])
        for bundle_id, entries in candidates.items()
    }
    if selected is not None:
        return tuple(loaded[bundle_id] for bundle_id in selected if bundle_id in loaded)
    return tuple(loaded[bundle_id] for bundle_id in sorted(loaded))
