# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Immutable startup snapshot for validated extension components."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.plugins.contracts import (
    DomainContract,
    ExecutionPlane,
    ExtensionBundleDescriptor,
    ExtensionComponentDescriptor,
)


@dataclass(frozen=True, slots=True)
class ResolvedExtensionComponent:
    """Bind a component descriptor to its containing bundle identity."""

    bundle_id: str
    bundle_version: str
    component: ExtensionComponentDescriptor

    @property
    def qualified_id(self) -> str:
        return f"{self.bundle_id}/{self.component.component_id}"


@dataclass(frozen=True, slots=True)
class ExtensionStartupSnapshot:
    """A deterministic, immutable view used by domain materializers."""

    bundles: tuple[ExtensionBundleDescriptor, ...]
    components: tuple[ResolvedExtensionComponent, ...]

    @classmethod
    def build(
        cls,
        bundles: tuple[ExtensionBundleDescriptor, ...],
    ) -> "ExtensionStartupSnapshot":
        bundle_ids = [bundle.bundle_id for bundle in bundles]
        if len(bundle_ids) != len(set(bundle_ids)):
            raise ValueError("bundle ids must be unique in a startup snapshot")

        components = tuple(
            ResolvedExtensionComponent(
                bundle_id=bundle.bundle_id,
                bundle_version=bundle.bundle_version,
                component=component,
            )
            for bundle in bundles
            for component in bundle.components
        )
        return cls(bundles=bundles, components=components)

    def components_for(
        self,
        contract: DomainContract,
        execution_plane: ExecutionPlane | None = None,
    ) -> tuple[ResolvedExtensionComponent, ...]:
        """Return providers without imposing a generic composition policy."""
        return tuple(
            resolved
            for resolved in self.components
            if contract in resolved.component.contracts
            and (
                execution_plane is None
                or execution_plane in resolved.component.execution_planes
            )
        )
