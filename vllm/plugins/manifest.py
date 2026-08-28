# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fail-closed parser for extension bundle manifests.

Parsing validates identities and placement before any implementation module is
imported. Domain-specific materializers remain responsible for constructing
runtime objects after a startup snapshot has been accepted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from vllm.plugins.contracts import (
    ComponentIsolation,
    ComponentPermission,
    DomainContract,
    ExecutionPlane,
    ExtensionBundleDescriptor,
    ExtensionComponentDescriptor,
)


_TOP_LEVEL_FIELDS = {
    "schema_version",
    "bundle_id",
    "bundle_version",
    "host_api_range",
    "components",
}
_COMPONENT_FIELDS = {
    "component_id",
    "contracts",
    "execution_planes",
    "isolation",
    "implementation_ref",
    "permissions",
}
_EnumT = TypeVar(
    "_EnumT",
    ComponentPermission,
    DomainContract,
    ExecutionPlane,
)


def _require_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{location} keys must be strings")
    return value


def _require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _require_string_tuple(value: Any, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"{location} must be an array of strings")
    return tuple(value)


def _parse_enum_tuple(
    value: Any,
    enum_type: type[_EnumT],
    location: str,
) -> tuple[_EnumT, ...]:
    values = _require_string_tuple(value, location)
    try:
        return tuple(enum_type(item) for item in values)
    except ValueError as error:
        raise ValueError(f"{location} contains an unsupported value") from error


def _reject_unknown_fields(
    value: dict[str, Any],
    allowed: set[str],
    location: str,
) -> None:
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"{location} contains unknown fields: {sorted(unknown)}")


def _parse_component(value: Any, index: int) -> ExtensionComponentDescriptor:
    location = f"components[{index}]"
    component = _require_object(value, location)
    _reject_unknown_fields(component, _COMPONENT_FIELDS, location)
    missing = _COMPONENT_FIELDS - {"permissions"} - component.keys()
    if missing:
        raise ValueError(f"{location} is missing fields: {sorted(missing)}")

    isolation_value = _require_string(component["isolation"], f"{location}.isolation")
    try:
        isolation = ComponentIsolation(isolation_value)
    except ValueError as error:
        raise ValueError(f"{location}.isolation is unsupported") from error

    return ExtensionComponentDescriptor(
        component_id=_require_string(
            component["component_id"], f"{location}.component_id"
        ),
        contracts=_parse_enum_tuple(
            component["contracts"], DomainContract, f"{location}.contracts"
        ),
        execution_planes=_parse_enum_tuple(
            component["execution_planes"],
            ExecutionPlane,
            f"{location}.execution_planes",
        ),
        isolation=isolation,
        implementation_ref=_require_string(
            component["implementation_ref"], f"{location}.implementation_ref"
        ),
        permissions=_parse_enum_tuple(
            component.get("permissions", []),
            ComponentPermission,
            f"{location}.permissions",
        ),
    )


def parse_extension_bundle_manifest(payload: Any) -> ExtensionBundleDescriptor:
    """Parse an already decoded extension bundle manifest."""
    manifest = _require_object(payload, "manifest")
    _reject_unknown_fields(manifest, _TOP_LEVEL_FIELDS, "manifest")
    missing = _TOP_LEVEL_FIELDS - manifest.keys()
    if missing:
        raise ValueError(f"manifest is missing fields: {sorted(missing)}")
    if manifest["schema_version"] != "1.0":
        raise ValueError("unsupported extension bundle schema_version")

    components_value = manifest["components"]
    if not isinstance(components_value, list):
        raise ValueError("components must be an array")

    return ExtensionBundleDescriptor(
        bundle_id=_require_string(manifest["bundle_id"], "bundle_id"),
        bundle_version=_require_string(manifest["bundle_version"], "bundle_version"),
        host_api_range=_require_string(manifest["host_api_range"], "host_api_range"),
        components=tuple(
            _parse_component(component, index)
            for index, component in enumerate(components_value)
        ),
    )


def load_extension_bundle_manifest(path: str | Path) -> ExtensionBundleDescriptor:
    """Read and parse one manifest without importing its implementation."""
    manifest_path = Path(path)
    if manifest_path.is_symlink():
        raise ValueError("extension bundle manifest must not be a symbolic link")
    if not manifest_path.is_file():
        raise ValueError("extension bundle manifest must be a regular file")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read extension bundle manifest") from error
    return parse_extension_bundle_manifest(payload)
