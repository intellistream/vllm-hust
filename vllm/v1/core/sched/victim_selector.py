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

When more than one selector is installed, choose one explicitly with
``--additional-config '{"victim_selector_plugin": "bidkv"}'``.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from vllm.logger import init_logger
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.request import Request

logger = init_logger(__name__)

VICTIM_SELECTOR_PLUGINS_GROUP = "vllm.victim_selector"
VICTIM_SELECTOR_PLUGIN_CONFIG_KEY = "victim_selector_plugin"
VICTIM_SELECTOR_API_VERSION = 1

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


def get_victim_selector(vllm_config) -> VictimSelector:
    """Discover and instantiate a victim selector.

    Loads a plugin from the ``vllm.victim_selector`` entry-point group only
    when the user explicitly requests it. Ambient site-packages must not
    change the default scheduler behavior.
    """
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    if additional_config.get("victim_selector_plugin_disabled"):
        return NoOpVictimSelector()

    requested_plugin = additional_config.get(VICTIM_SELECTOR_PLUGIN_CONFIG_KEY)
    if requested_plugin is None:
        return NoOpVictimSelector()
    if not isinstance(requested_plugin, str) or not requested_plugin.strip():
        raise ValueError(
            f"additional_config.{VICTIM_SELECTOR_PLUGIN_CONFIG_KEY} must be "
            "a non-empty string"
        )

    try:
        from importlib.metadata import EntryPoints, entry_points

        eps: EntryPoints = entry_points(group=VICTIM_SELECTOR_PLUGINS_GROUP)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to discover requested victim selector plugin {requested_plugin!r}"
        ) from exc

    candidates = [ep for ep in eps if ep.name == requested_plugin]
    if not candidates:
        available = ", ".join(sorted(ep.name for ep in eps))
        raise ValueError(
            f"Requested victim selector plugin {requested_plugin!r} is not "
            f"installed; available plugins: {available or 'none'}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple victim selector entry points are registered as "
            f"{requested_plugin!r}; uninstall duplicate distributions"
        )
    selected = candidates[0]

    try:
        selector_cls = selected.load()
        api_version = getattr(selector_cls, "vllm_victim_selector_api_version", None)
        if api_version != VICTIM_SELECTOR_API_VERSION:
            raise TypeError(
                "plugin declares victim-selector API version "
                f"{api_version!r}; expected {VICTIM_SELECTOR_API_VERSION}"
            )
        if not hasattr(selector_cls, "from_vllm_config"):
            raise TypeError("plugin does not define from_vllm_config()")
        selector = selector_cls.from_vllm_config(vllm_config)
        if not isinstance(selector, VictimSelector):
            raise TypeError("plugin does not implement the VictimSelector protocol")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load requested victim selector plugin {selected.name!r}: {exc}"
        ) from exc

    distribution = selected.dist
    distribution_name = getattr(distribution, "name", None) or "unknown"
    distribution_version = getattr(distribution, "version", None) or "unknown"
    try:
        source_path = inspect.getsourcefile(selector_cls) or inspect.getfile(
            selector_cls
        )
    except (OSError, TypeError):
        source_path = selected.value
    logger.info(
        "Loaded victim selector plugin %r distribution=%s==%s source=%s "
        "api_version=%s",
        selected.name,
        distribution_name,
        distribution_version,
        source_path,
        api_version,
    )
    return selector


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
