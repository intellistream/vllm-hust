# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Victim selector protocol and plugin discovery for vLLM HUST.

This module defines the lightweight protocol that scheduler preemption
victim selection plugins must implement, a no-op default selector (that
matches upstream vLLM behaviour), and a factory that discovers and loads
plugins via the ``vllm.victim_selector`` entry-point group.

Plugins (e.g. BidKV) are installed separately and auto-registered via::

    [project.entry-points."vllm.victim_selector"]
    bidkv = "bidkv.adapters.vllm_hust.selector:BidkvVictimSelector"
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Any, Protocol, runtime_checkable

from vllm.plugins.contracts import DomainContract, ExecutionPlane
from vllm.plugins.snapshot import ResolvedExtensionComponent
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.request import Request

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class VictimSelector(Protocol):
    """Protocol that victim selection plugins must implement.

    The protocol is intentionally minimal so that third-party plugins
    (e.g. BidKV) can be developed and released independently of
    vllm-hust.
    """

    @classmethod
    def from_vllm_config(cls, vllm_config) -> VictimSelector:
        """Factory: build a selector from a vLLM ``VllmConfig``."""
        ...

    def pick_victim(
        self,
        running: Sequence[Request],
        policy: SchedulingPolicy,
        *,
        kv_utilization: float | None = None,
        now_s: float | None = None,
    ) -> Request:
        """Pick the request to preempt from *running*."""
        ...

    def emit_observability_log(self, logger, scheduler_name: str) -> None:
        """Emit observability / metrics log line (optional)."""
        ...

    def export_metrics(self) -> dict[str, Any]:
        """Export internal metrics as a flat dict (optional)."""
        ...


# ---------------------------------------------------------------------------
# No-op default (equivalent to upstream vLLM behaviour)
# ---------------------------------------------------------------------------


class NoOpVictimSelector:
    """Default victim selector — behaves identically to upstream vLLM.

    * FCFS: always picks the last request in ``running``.
    * PRIORITY: picks the request with the highest priority (ties broken
      by latest arrival).
    """

    @classmethod
    def from_vllm_config(cls, vllm_config) -> NoOpVictimSelector:
        return cls()

    def pick_victim(
        self,
        running: Sequence[Request],
        policy: SchedulingPolicy,
        *,
        kv_utilization: float | None = None,
        now_s: float | None = None,
    ) -> Request:
        if not running:
            raise ValueError("running is empty, cannot pick victim")
        if policy == SchedulingPolicy.PRIORITY:
            return max(
                running,
                key=lambda request: (request.priority, request.arrival_time),
            )
        return running[-1]

    def emit_observability_log(self, logger, scheduler_name: str) -> None:
        pass

    def export_metrics(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Plugin discovery
# ---------------------------------------------------------------------------


class VictimSelectorMaterializationError(RuntimeError):
    """Fail closed after a typed scheduler-policy component was admitted."""


def _select_typed_component(
    vllm_config,
) -> ResolvedExtensionComponent | None:
    """Select one admitted scheduler-policy provider for this process.

    Selection is deliberately domain-specific: scheduler policies are
    exclusive, while other contracts may eventually define composition or
    fan-out rules.  A qualified component id is required to resolve ambiguity.
    """
    from vllm.plugins.startup import get_configured_extension_startup

    resolution = get_configured_extension_startup()
    providers = resolution.snapshot.components_for(
        DomainContract.SCHEDULER_POLICY_V1,
        ExecutionPlane.SCHEDULER,
    )
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    selected_id = additional_config.get("victim_selector_component")

    if selected_id is not None:
        matches = tuple(
            provider for provider in providers if provider.qualified_id == selected_id
        )
        if not matches:
            available = sorted(provider.qualified_id for provider in providers)
            raise VictimSelectorMaterializationError(
                "victim_selector_component selects no admitted scheduler-policy "
                f"provider: {selected_id!r}; available providers: {available}"
            )
        return matches[0]

    if len(providers) > 1:
        available = sorted(provider.qualified_id for provider in providers)
        raise VictimSelectorMaterializationError(
            "multiple admitted scheduler-policy providers require an explicit "
            f"victim_selector_component: {available}"
        )
    return providers[0] if providers else None


def _load_implementation(implementation_ref: str) -> Any:
    """Import a ``module:attribute`` reference only after typed admission."""
    module_name, separator, attribute_path = implementation_ref.partition(":")
    if not separator or not module_name or not attribute_path:
        raise VictimSelectorMaterializationError(
            "scheduler-policy implementation_ref must use module:attribute syntax: "
            f"{implementation_ref!r}"
        )
    try:
        implementation = import_module(module_name)
        for attribute in attribute_path.split("."):
            if not attribute:
                raise AttributeError("empty attribute segment")
            implementation = getattr(implementation, attribute)
        return implementation
    except Exception as error:
        raise VictimSelectorMaterializationError(
            f"failed to import scheduler-policy implementation {implementation_ref!r}"
        ) from error


def _materialize_typed_victim_selector(
    component: ResolvedExtensionComponent,
    vllm_config,
) -> VictimSelector:
    implementation = _load_implementation(component.component.implementation_ref)
    factory = getattr(implementation, "from_vllm_config", None)
    if not callable(factory):
        raise VictimSelectorMaterializationError(
            f"scheduler-policy component {component.qualified_id!r} does not expose "
            "a callable from_vllm_config factory"
        )
    try:
        selector = factory(vllm_config)
    except Exception as error:
        raise VictimSelectorMaterializationError(
            "scheduler-policy component "
            f"{component.qualified_id!r} failed to initialize"
        ) from error
    if not isinstance(selector, VictimSelector):
        raise VictimSelectorMaterializationError(
            f"scheduler-policy component {component.qualified_id!r} returned an "
            "object that does not implement VictimSelector"
        )
    return selector


def _materialize_legacy_victim_selector(
    entry_point: Any,
    vllm_config,
    *,
    fail_closed: bool,
) -> VictimSelector | None:
    """Load one legacy entry point, optionally treating it as explicit config."""
    try:
        selector_class = entry_point.load()
        factory = getattr(selector_class, "from_vllm_config", None)
        if not callable(factory):
            raise TypeError("selector does not expose from_vllm_config")
        selector = factory(vllm_config)
        if not isinstance(selector, VictimSelector):
            raise TypeError("selector does not implement VictimSelector")
        return selector
    except Exception as error:
        if fail_closed:
            raise VictimSelectorMaterializationError(
                "explicit legacy scheduler-policy entry point "
                f"{entry_point.name!r} failed to initialize"
            ) from error
        return None


def get_victim_selector(vllm_config) -> VictimSelector:
    """Discover and instantiate a victim selector.

    An admitted typed scheduler-policy component takes precedence and fails
    closed if selection, import, or protocol validation fails.  When no typed
    provider is configured, retain the legacy ``vllm.victim_selector`` entry
    point behavior for compatibility.
    """
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    if additional_config.get("victim_selector_plugin_disabled"):
        return NoOpVictimSelector()

    typed_component = _select_typed_component(vllm_config)
    if typed_component is not None:
        return _materialize_typed_victim_selector(typed_component, vllm_config)

    selected_legacy_name = additional_config.get("victim_selector_plugin")
    try:
        from importlib.metadata import EntryPoints, entry_points

        eps: EntryPoints = entry_points(group="vllm.victim_selector")
    except Exception as error:
        if selected_legacy_name is not None:
            raise VictimSelectorMaterializationError(
                "failed to discover explicitly selected legacy scheduler-policy "
                f"entry point {selected_legacy_name!r}"
            ) from error
        return NoOpVictimSelector()

    if selected_legacy_name is not None:
        matches = tuple(ep for ep in eps if ep.name == selected_legacy_name)
        if len(matches) != 1:
            available = sorted(ep.name for ep in eps)
            raise VictimSelectorMaterializationError(
                "victim_selector_plugin must select exactly one legacy "
                "scheduler-policy entry point: "
                f"{selected_legacy_name!r}; available entry points: {available}"
            )
        selector = _materialize_legacy_victim_selector(
            matches[0], vllm_config, fail_closed=True
        )
        assert selector is not None
        return selector

    for ep in eps:
        selector = _materialize_legacy_victim_selector(
            ep, vllm_config, fail_closed=False
        )
        if selector is not None:
            return selector

    return NoOpVictimSelector()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def infer_kv_utilization_from_scheduler(scheduler) -> float | None:
    """Return current KV-cache utilization ratio [0, 1] from a scheduler.

    Used by schedulers to pass ``kv_utilization`` to ``pick_victim`` so
    that plugins (e.g. BidKV) can gate utility-based selection on KV
    pressure without coupling to scheduler internals.
    """
    try:
        return scheduler.kv_cache_manager.usage
    except Exception:
        return None
