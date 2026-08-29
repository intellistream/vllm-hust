# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared import boundary for admitted extension components.

Admission and selection operate only on immutable descriptors. Domain owners
call this helper at their existing construction boundary, after selecting an
exact component, so import failures remain fail-closed and attributable.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from vllm.plugins.snapshot import ResolvedExtensionComponent


class ExtensionComponentMaterializationError(RuntimeError):
    """Fail closed after an admitted component is selected for import."""


def import_component_implementation(
    component: ResolvedExtensionComponent,
    *,
    domain: str,
) -> Any:
    """Import one ``module:attribute`` reference after typed admission."""
    implementation_ref = component.component.implementation_ref
    module_name, separator, attribute_path = implementation_ref.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ExtensionComponentMaterializationError(
            f"{domain} component {component.qualified_id!r} implementation_ref "
            f"must use module:attribute syntax: {implementation_ref!r}"
        )
    try:
        implementation = import_module(module_name)
        for attribute in attribute_path.split("."):
            if not attribute:
                raise AttributeError("empty attribute segment")
            implementation = getattr(implementation, attribute)
        return implementation
    except Exception as error:
        raise ExtensionComponentMaterializationError(
            f"failed to import {domain} component {component.qualified_id!r} "
            f"from {implementation_ref!r}"
        ) from error
