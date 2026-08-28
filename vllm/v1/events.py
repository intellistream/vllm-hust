# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic typed event outlet for engine-core instrumentation.

Design contract
---------------
- Core exposes only a *generic*, *typed*, *default-off* event outlet.
- Concrete emitters (e.g. the B134 TSV/JSONL logger) live out-of-tree as
  plugins that subscribe through :meth:`EventBus.register_sink`.
- No experiment-specific names, file protocols or vendor event chains are
  hard-coded in core. Event semantics are generic (request lifecycle,
  KV-offload transfer), never tied to a particular benchmark/experiment.

Overhead
--------
With no registered sinks, :attr:`EventBus.enabled` is ``False`` and call
sites skip event construction entirely (a single boolean check), so the
default-off instrumentation is zero-overhead. Sink failures are swallowed so
instrumentation can never affect serving correctness or latency.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Typed events (generic semantics)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineCoreEvent:
    """Base class for all engine-core events."""

    ts_monotonic_ns: int = field(init=False, default_factory=time.monotonic_ns)
    """Monotonic timestamp (ns) taken at construction time."""


@dataclass(frozen=True)
class RequestAdmitted(EngineCoreEvent):
    """A request was admitted into the running set."""

    request_id: str


@dataclass(frozen=True)
class RequestScheduled(EngineCoreEvent):
    """A request was scheduled for execution in this step."""

    request_id: str


@dataclass(frozen=True)
class RequestPreempted(EngineCoreEvent):
    """A request was preempted to free space for other requests."""

    request_id: str


@dataclass(frozen=True)
class RequestResumed(EngineCoreEvent):
    """A previously preempted request resumed execution."""

    request_id: str


@dataclass(frozen=True)
class KVOffloadStore(EngineCoreEvent):
    """Blocks were stored to the CPU tier."""

    request_id: str
    duration_us: int
    evicted_keys: int
    stored_keys: int


@dataclass(frozen=True)
class KVOffloadEvict(EngineCoreEvent):
    """Blocks were explicitly evicted from the CPU tier."""

    request_id: str
    duration_us: int
    evicted_keys: int


@dataclass(frozen=True)
class KVOffloadRestoreStart(EngineCoreEvent):
    """A restore (load) of blocks from a secondary tier began."""

    request_id: str
    keys: int


@dataclass(frozen=True)
class KVOffloadRestoreDone(EngineCoreEvent):
    """A restore (load) of blocks from a secondary tier finished."""

    request_id: str
    keys: int


@dataclass(frozen=True)
class KVOffloadTierEvict(EngineCoreEvent):
    """Blocks were evicted by the tiering manager."""

    request_id: str
    duration_us: int
    keys: int


@dataclass(frozen=True)
class KVOffloadSchedStep(EngineCoreEvent):
    """One scheduler step of the offloading connector finished."""

    duration_us: int


@dataclass(frozen=True)
class KVTransferSwapD2H(EngineCoreEvent):
    """NPU->CPU block swap kernel was submitted."""

    job_id: str
    descriptors: int
    duration_us: int


@dataclass(frozen=True)
class KVTransferGatherH2D(EngineCoreEvent):
    """Host page gather for a CPU->NPU transfer finished."""

    job_id: str
    dma_runs: int
    duration_us: int


@dataclass(frozen=True)
class KVTransferSubmit(EngineCoreEvent):
    """A transfer job was fully prepared and submitted."""

    job_id: str
    bytes: int
    dependency_us: int
    descriptor_us: int
    descriptors: int
    direction: str  # "d2h" (NPU->CPU) or "h2d" (CPU->NPU)
    submit_us: int


@dataclass(frozen=True)
class KVTransferCopyDone(EngineCoreEvent):
    """A transfer job completed (wall-clock and device-event timings)."""

    job_id: str
    bytes: int
    completion_observed_ms: float
    device_event_ms: float
    direction: str


# ---------------------------------------------------------------------------
# Event bus (default-off)
# ---------------------------------------------------------------------------


class EventSink(Protocol):
    """A consumer of engine-core events."""

    def emit(self, event: EngineCoreEvent) -> None:  # pragma: no cover
        ...


class EventBus:
    """Process-wide default-off event bus.

    Plugins register sinks (typically from an entry-point ``register()``
    function during engine startup). With no sinks registered, ``enabled`` is
    ``False`` and :meth:`emit` is a no-op, so the instrumentation is disabled
    by default.
    """

    _sinks: list[EventSink] = []
    _lock = threading.Lock()
    enabled: bool = False
    """True when at least one sink is registered. Call sites check this
    before constructing events so the default-off path has zero overhead."""

    @classmethod
    def register_sink(cls, sink: EventSink) -> None:
        """Register a sink. Idempotent per sink instance."""
        with cls._lock:
            if sink not in cls._sinks:
                cls._sinks.append(sink)
            cls.enabled = True

    @classmethod
    def unregister_sink(cls, sink: EventSink) -> None:
        """Remove a previously registered sink."""
        with cls._lock:
            if sink in cls._sinks:
                cls._sinks.remove(sink)
            cls.enabled = bool(cls._sinks)

    @classmethod
    def emit(cls, event: EngineCoreEvent) -> None:
        """Dispatch a typed event to all registered sinks.

        Sink exceptions are swallowed: instrumentation must never affect
        serving correctness or latency (error degradation).
        """
        sinks = cls._sinks
        if not sinks:
            return
        for sink in sinks:
            with contextlib.suppress(Exception):
                sink.emit(event)


# Re-export for plugin authors that introspect event payloads.
EventPayload = dict[str, Any]
