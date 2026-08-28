# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KV-domain selection topology for admitted extension components.

This module deliberately stops before implementation import or connector
construction. ``KVConnectorFactory`` remains the sole owner of role-specific
construction, constructor validation, HMA checks, and ``MultiConnector``
materialization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import regex as re

from vllm.plugins.contracts import DomainContract, ExecutionPlane
from vllm.plugins.snapshot import (
    ExtensionStartupSnapshot,
    ResolvedExtensionComponent,
)

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9.-]*$")


class KVConnectorSelectionError(ValueError):
    """Reject an invalid or unresolved KV connector selection topology."""


class KVConnectorComposition(str, Enum):
    """Composition semantics owned by the KV connector domain."""

    SINGLE = "single"
    ORDERED_MULTI = "ordered_multi"


@dataclass(frozen=True, slots=True)
class KVConnectorSchedulerCapabilities:
    """Configuration-time scheduler capability declarations."""

    supports_hma: bool


@dataclass(frozen=True, slots=True)
class KVConnectorWorkerCapabilities:
    """Configuration-time worker capability declarations."""

    supports_hma: bool
    requires_piecewise_for_cudagraph: bool


@dataclass(frozen=True, slots=True)
class KVConnectorPairSelection:
    """Pair scheduler and worker components for one logical connector."""

    connector_id: str
    scheduler_component: str
    worker_component: str
    scheduler_capabilities: KVConnectorSchedulerCapabilities
    worker_capabilities: KVConnectorWorkerCapabilities

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.connector_id):
            raise KVConnectorSelectionError(
                "connector_id must use lowercase letters, digits, dots, or hyphens"
            )
        _validate_qualified_component_id(
            self.scheduler_component, "scheduler_component"
        )
        _validate_qualified_component_id(self.worker_component, "worker_component")


@dataclass(frozen=True, slots=True)
class KVConnectorSelectionProfile:
    """Versioned, explicit connector topology independent of discovery order."""

    schema_version: str
    composition: KVConnectorComposition
    connectors: tuple[KVConnectorPairSelection, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise KVConnectorSelectionError(
                f"unsupported KV connector selection schema: {self.schema_version!r}"
            )
        if not self.connectors:
            raise KVConnectorSelectionError(
                "KV connector selection must contain at least one connector"
            )
        connector_ids = [connector.connector_id for connector in self.connectors]
        if len(connector_ids) != len(set(connector_ids)):
            raise KVConnectorSelectionError("connector_id values must be unique")
        component_pairs = [
            (connector.scheduler_component, connector.worker_component)
            for connector in self.connectors
        ]
        if len(component_pairs) != len(set(component_pairs)):
            raise KVConnectorSelectionError(
                "scheduler/worker component pairs must be unique"
            )
        if (
            self.composition is KVConnectorComposition.SINGLE
            and len(self.connectors) != 1
        ):
            raise KVConnectorSelectionError(
                "single composition requires exactly one connector"
            )
        if (
            self.composition is KVConnectorComposition.ORDERED_MULTI
            and len(self.connectors) < 2
        ):
            raise KVConnectorSelectionError(
                "ordered_multi composition requires at least two connectors"
            )


@dataclass(frozen=True, slots=True)
class ResolvedKVConnectorPair:
    """Bind an explicit logical connector to admitted role components."""

    connector_id: str
    scheduler_component: ResolvedExtensionComponent
    worker_component: ResolvedExtensionComponent
    scheduler_capabilities: KVConnectorSchedulerCapabilities
    worker_capabilities: KVConnectorWorkerCapabilities


@dataclass(frozen=True, slots=True)
class ResolvedKVConnectorSelection:
    """Immutable, ordered result for a future factory-owned adapter."""

    composition: KVConnectorComposition
    connectors: tuple[ResolvedKVConnectorPair, ...]

    @property
    def supports_hma(self) -> bool:
        """Return true only when every role of every child declares HMA."""
        return all(
            connector.scheduler_capabilities.supports_hma
            and connector.worker_capabilities.supports_hma
            for connector in self.connectors
        )

    @property
    def requires_piecewise_for_cudagraph(self) -> bool:
        """Return true when any ordered worker child requires piecewise mode."""
        return any(
            connector.worker_capabilities.requires_piecewise_for_cudagraph
            for connector in self.connectors
        )


def parse_kv_connector_selection(
    payload: Mapping[str, Any],
) -> KVConnectorSelectionProfile:
    """Parse the closed KV selection schema without accepting free-form data."""
    _reject_unknown_fields(
        payload,
        {"schema_version", "composition", "connectors"},
        "KV connector selection",
    )
    schema_version = _required_string(payload, "schema_version")
    composition_value = _required_string(payload, "composition")
    try:
        composition = KVConnectorComposition(composition_value)
    except ValueError as error:
        allowed = [item.value for item in KVConnectorComposition]
        raise KVConnectorSelectionError(
            f"composition must be one of {allowed}, got {composition_value!r}"
        ) from error

    raw_connectors = payload.get("connectors")
    if not isinstance(raw_connectors, Sequence) or isinstance(
        raw_connectors, (str, bytes)
    ):
        raise KVConnectorSelectionError("connectors must be an array")

    connectors: list[KVConnectorPairSelection] = []
    for index, raw_connector in enumerate(raw_connectors):
        if not isinstance(raw_connector, Mapping):
            raise KVConnectorSelectionError(f"connectors[{index}] must be an object")
        _reject_unknown_fields(
            raw_connector,
            {
                "connector_id",
                "scheduler_component",
                "worker_component",
                "scheduler_capabilities",
                "worker_capabilities",
            },
            f"connectors[{index}]",
        )
        scheduler_capabilities = _parse_scheduler_capabilities(
            raw_connector.get("scheduler_capabilities"), index
        )
        worker_capabilities = _parse_worker_capabilities(
            raw_connector.get("worker_capabilities"), index
        )
        connectors.append(
            KVConnectorPairSelection(
                connector_id=_required_string(raw_connector, "connector_id"),
                scheduler_component=_required_string(
                    raw_connector, "scheduler_component"
                ),
                worker_component=_required_string(raw_connector, "worker_component"),
                scheduler_capabilities=scheduler_capabilities,
                worker_capabilities=worker_capabilities,
            )
        )

    return KVConnectorSelectionProfile(
        schema_version=schema_version,
        composition=composition,
        connectors=tuple(connectors),
    )


def resolve_kv_connector_selection(
    profile: KVConnectorSelectionProfile,
    snapshot: ExtensionStartupSnapshot,
) -> ResolvedKVConnectorSelection:
    """Resolve exact scheduler/worker providers without importing them."""
    resolved = tuple(
        ResolvedKVConnectorPair(
            connector_id=connector.connector_id,
            scheduler_component=_resolve_component(
                snapshot,
                connector.scheduler_component,
                DomainContract.KV_CONNECTOR_SCHEDULER_V1,
                ExecutionPlane.SCHEDULER,
            ),
            worker_component=_resolve_component(
                snapshot,
                connector.worker_component,
                DomainContract.KV_CONNECTOR_WORKER_V1,
                ExecutionPlane.WORKER,
            ),
            scheduler_capabilities=connector.scheduler_capabilities,
            worker_capabilities=connector.worker_capabilities,
        )
        for connector in profile.connectors
    )
    return ResolvedKVConnectorSelection(
        composition=profile.composition,
        connectors=resolved,
    )


def _resolve_component(
    snapshot: ExtensionStartupSnapshot,
    qualified_id: str,
    contract: DomainContract,
    plane: ExecutionPlane,
) -> ResolvedExtensionComponent:
    by_id = {component.qualified_id: component for component in snapshot.components}
    component = by_id.get(qualified_id)
    if component is None:
        available = sorted(
            candidate.qualified_id
            for candidate in snapshot.components_for(contract, plane)
        )
        raise KVConnectorSelectionError(
            f"{qualified_id!r} is not an admitted {plane.value} provider for "
            f"{contract.value}; available providers: {available}"
        )
    if contract not in component.component.contracts:
        raise KVConnectorSelectionError(
            f"{qualified_id!r} does not implement {contract.value}"
        )
    if plane not in component.component.execution_planes:
        raise KVConnectorSelectionError(
            f"{qualified_id!r} is not admitted for the {plane.value} plane"
        )
    return component


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise KVConnectorSelectionError(f"{field} must be a non-empty string")
    return value


def _parse_scheduler_capabilities(
    value: Any, connector_index: int
) -> KVConnectorSchedulerCapabilities:
    location = f"connectors[{connector_index}].scheduler_capabilities"
    if not isinstance(value, Mapping):
        raise KVConnectorSelectionError(f"{location} must be an object")
    _reject_unknown_fields(value, {"supports_hma"}, location)
    return KVConnectorSchedulerCapabilities(
        supports_hma=_required_bool(value, "supports_hma", location)
    )


def _parse_worker_capabilities(
    value: Any, connector_index: int
) -> KVConnectorWorkerCapabilities:
    location = f"connectors[{connector_index}].worker_capabilities"
    if not isinstance(value, Mapping):
        raise KVConnectorSelectionError(f"{location} must be an object")
    _reject_unknown_fields(
        value,
        {"supports_hma", "requires_piecewise_for_cudagraph"},
        location,
    )
    return KVConnectorWorkerCapabilities(
        supports_hma=_required_bool(value, "supports_hma", location),
        requires_piecewise_for_cudagraph=_required_bool(
            value, "requires_piecewise_for_cudagraph", location
        ),
    )


def _required_bool(payload: Mapping[str, Any], field: str, location: str) -> bool:
    value = payload.get(field)
    if not isinstance(value, bool):
        raise KVConnectorSelectionError(f"{location}.{field} must be a boolean")
    return value


def _reject_unknown_fields(
    payload: Mapping[str, Any], allowed: set[str], location: str
) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise KVConnectorSelectionError(
            f"{location} contains unknown fields: {sorted(unknown)}"
        )


def _validate_qualified_component_id(value: str, field: str) -> None:
    bundle_id, separator, component_id = value.partition("/")
    if (
        not separator
        or not bundle_id
        or not component_id
        or not _IDENTIFIER.fullmatch(bundle_id)
        or not _IDENTIFIER.fullmatch(component_id)
    ):
        raise KVConnectorSelectionError(
            f"{field} must use the qualified bundle-id/component-id form"
        )
