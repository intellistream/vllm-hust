# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Default-off worker lifecycle event hooks for observability plugins."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

logger = logging.getLogger(__name__)

ASYNC_OUTPUT_CREATED = "async_output.created"
ASYNC_OUTPUT_COPY_ISSUED = "async_output.copy_issued"
ASYNC_OUTPUT_COPY_FAILED = "async_output.copy_failed"
ASYNC_OUTPUT_WAIT_COMPLETE = "async_output.wait_complete"
ASYNC_OUTPUT_MATERIALIZED = "async_output.materialized"
ASYNC_OUTPUT_RETAINED = "async_output.retained"
ASYNC_OUTPUT_CONSUMED = "async_output.consumed"


@dataclass(frozen=True, slots=True)
class WorkerLifecycleEvent:
    """One immutable worker lifecycle event."""

    name: str
    monotonic_ns: int
    fields: Mapping[str, Any]


WorkerLifecycleListener = Callable[[WorkerLifecycleEvent], None]

_listener_lock = threading.Lock()
_listener_snapshot: tuple[tuple[str, WorkerLifecycleListener], ...] = ()


def register_worker_lifecycle_listener(
    name: str, listener: WorkerLifecycleListener
) -> None:
    """Register one process-local listener by stable plugin name."""
    if not name:
        raise ValueError("listener name must not be empty")
    if not callable(listener):
        raise TypeError("listener must be callable")

    global _listener_snapshot
    with _listener_lock:
        existing = dict(_listener_snapshot).get(name)
        if existing is listener:
            return
        if existing is not None:
            raise ValueError(f"worker lifecycle listener {name!r} is registered")
        _listener_snapshot = (*_listener_snapshot, (name, listener))


def unregister_worker_lifecycle_listener(name: str) -> None:
    """Remove a process-local listener if present."""
    global _listener_snapshot
    with _listener_lock:
        _listener_snapshot = tuple(
            item for item in _listener_snapshot if item[0] != name
        )


def has_worker_lifecycle_listeners() -> bool:
    """Return whether an event payload would have a consumer."""
    return bool(_listener_snapshot)


def emit_worker_lifecycle_event(name: str, **fields: Any) -> None:
    """Emit one event and isolate serving from listener failures."""
    global _listener_snapshot
    listeners = _listener_snapshot
    if not listeners:
        return

    event = WorkerLifecycleEvent(
        name=name,
        monotonic_ns=time.perf_counter_ns(),
        fields=MappingProxyType(fields),
    )
    failed_names: list[str] = []
    for listener_name, listener in listeners:
        try:
            listener(event)
        except Exception:
            failed_names.append(listener_name)
            logger.exception(
                "Disabling failed worker lifecycle listener %s", listener_name
            )

    if not failed_names:
        return
    failed_set = set(failed_names)
    with _listener_lock:
        _listener_snapshot = tuple(
            item for item in _listener_snapshot if item[0] not in failed_set
        )


def reset_worker_lifecycle_listeners_for_test() -> None:
    """Clear process-local listeners for isolated tests."""
    global _listener_snapshot
    with _listener_lock:
        _listener_snapshot = ()
