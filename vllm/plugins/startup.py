# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Admission and immutable startup resolution for extension bundles."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from vllm.plugins.contracts import (
    EXTENSION_HOST_API_VERSION,
    ComponentPermission,
    ExtensionBundleDescriptor,
)
from vllm.plugins.manifest import load_extension_bundle_manifest
from vllm.plugins.snapshot import ExtensionStartupSnapshot

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class ExtensionBundleAdmissionError(ValueError):
    """Reject an explicitly configured bundle before implementation import."""


_API_VERSION = re.compile(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?$")
_API_CLAUSE = re.compile(r"^(>=|<=|==|!=|>|<)(\d+(?:\.\d+){0,2})$")


def _parse_api_version(value: str) -> tuple[int, int, int]:
    match = _API_VERSION.fullmatch(value)
    if match is None:
        raise ExtensionBundleAdmissionError(
            f"extension API version must be one to three numeric parts: {value!r}"
        )
    major, minor, patch = match.groups()
    return int(major), int(minor or 0), int(patch or 0)


def _matches_api_range(version: tuple[int, int, int], api_range: str) -> bool:
    clauses = api_range.split(",")
    if not clauses or any(not clause for clause in clauses):
        raise ExtensionBundleAdmissionError(
            f"invalid host_api_range: {api_range!r}"
        )
    operations = {
        ">=": lambda left, right: left >= right,
        "<=": lambda left, right: left <= right,
        "==": lambda left, right: left == right,
        "!=": lambda left, right: left != right,
        ">": lambda left, right: left > right,
        "<": lambda left, right: left < right,
    }
    for clause in clauses:
        match = _API_CLAUSE.fullmatch(clause)
        if match is None:
            raise ExtensionBundleAdmissionError(
                f"invalid host_api_range clause: {clause!r}"
            )
        operation, required = match.groups()
        if not operations[operation](version, _parse_api_version(required)):
            return False
    return True


@dataclass(frozen=True, slots=True)
class ExtensionStartupResolution:
    """Accepted snapshot plus configured bundles disabled by policy."""

    snapshot: ExtensionStartupSnapshot
    disabled_bundle_ids: tuple[str, ...]


def _normalize_manifest_paths(paths: "Sequence[str | Path]") -> tuple[Path, ...]:
    normalized = tuple(Path(os.path.abspath(path)) for path in paths)
    if len(normalized) != len(set(normalized)):
        raise ExtensionBundleAdmissionError(
            "extension manifest paths must be unique"
        )
    return normalized


def _validate_host_api_range(
    bundle: ExtensionBundleDescriptor,
    host_api_version: str,
) -> None:
    host_version = _parse_api_version(host_api_version)
    try:
        compatible = _matches_api_range(host_version, bundle.host_api_range)
    except ExtensionBundleAdmissionError as error:
        raise ExtensionBundleAdmissionError(
            f"bundle {bundle.bundle_id!r} has an invalid host_api_range: "
            f"{bundle.host_api_range!r}"
        ) from error
    if not compatible:
        raise ExtensionBundleAdmissionError(
            f"bundle {bundle.bundle_id!r} requires host API "
            f"{bundle.host_api_range!r}, but this host provides "
            f"{host_api_version!r}"
        )


def _validate_permission_policy(
    bundle: ExtensionBundleDescriptor,
    allowed_permissions: "Iterable[ComponentPermission] | None",
) -> None:
    if allowed_permissions is None:
        return
    allowed = frozenset(allowed_permissions)
    for component in bundle.components:
        denied = set(component.permissions) - allowed
        if denied:
            values = sorted(permission.value for permission in denied)
            raise ExtensionBundleAdmissionError(
                f"component {bundle.bundle_id}/{component.component_id} "
                f"requests permissions denied by host policy: {values}"
            )


def resolve_extension_startup(
    manifest_paths: "Sequence[str | Path]",
    *,
    enabled_bundle_ids: "Iterable[str] | None" = None,
    allowed_permissions: "Iterable[ComponentPermission] | None" = None,
    host_api_version: str = EXTENSION_HOST_API_VERSION,
) -> ExtensionStartupResolution:
    """Validate configured manifests and build a deterministic startup view.

    Every explicitly configured manifest is parsed before allowlist filtering,
    so malformed disabled bundles cannot silently remain in deployment config.
    No ``implementation_ref`` is imported here.
    """
    enabled = None if enabled_bundle_ids is None else frozenset(enabled_bundle_ids)
    accepted: list[ExtensionBundleDescriptor] = []
    disabled: list[str] = []
    seen_bundle_ids: set[str] = set()
    for path in _normalize_manifest_paths(manifest_paths):
        try:
            bundle = load_extension_bundle_manifest(path)
        except ValueError as error:
            raise ExtensionBundleAdmissionError(
                f"extension manifest {path} was rejected: {error}"
            ) from error
        if bundle.bundle_id in seen_bundle_ids:
            raise ExtensionBundleAdmissionError(
                f"bundle id {bundle.bundle_id!r} is configured more than once"
            )
        seen_bundle_ids.add(bundle.bundle_id)
        _validate_host_api_range(bundle, host_api_version)
        _validate_permission_policy(bundle, allowed_permissions)
        if enabled is not None and bundle.bundle_id not in enabled:
            disabled.append(bundle.bundle_id)
            continue
        accepted.append(bundle)

    if enabled is not None:
        unknown_enabled = enabled - seen_bundle_ids
        if unknown_enabled:
            raise ExtensionBundleAdmissionError(
                "enabled bundle ids have no configured manifest: "
                f"{sorted(unknown_enabled)}"
            )

    return ExtensionStartupResolution(
        snapshot=ExtensionStartupSnapshot.build(tuple(accepted)),
        disabled_bundle_ids=tuple(disabled),
    )


@cache
def get_configured_extension_startup() -> ExtensionStartupResolution:
    """Resolve process startup configuration once into an immutable snapshot."""
    import vllm.envs as envs

    return resolve_extension_startup(
        envs.VLLM_EXTENSION_MANIFESTS,
        enabled_bundle_ids=envs.VLLM_EXTENSION_BUNDLES,
    )
