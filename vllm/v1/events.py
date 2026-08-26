# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic typed event outlet for engine-core instrumentation.

Design contract
---------------
- Core exposes only a *generic*, *typed*, *default-off* event outlet.
- Concrete emitters (e.g. the B134 TSV logger) live out-of-tree as plugins
  that subscribe through :meth:`EventBus.register_sink`.
- No experiment-specific names, file protocols or vendor event chains are
  hard-coded in core. Event semantics are generic (request lifecycle,
  KV-offload transfer), never tied to a particular benchmark/experiment.

Overhead
--------
With no registered sinks, :meth:`EventBus.emit` is a single list-length
check and returns immediately. Sink failures are swallowed so that
instrumentation can never affect serving correctness or latency.
"""

from __future__ import annotations

import dataclasses
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

    ts_monotonic_ns: int = field(
        init=False, default_factory=time.monotonic_ns
    )
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
class RequestFinished(EngineCoreEvent):
    """A request finished generation."""

    request_id: str
    output_tokens: int


@dataclass(frozen=True)
class KVOffloadStore(EngineCoreEvent):
    """Blocks were stored to the CPU tier."""

    request_id: str
    elapsed_us: float
    num_keys: int
    evicted: int


@dataclass(frozen=True)
class KVOffloadEvict(EngineCoreEvent):
    """Blocks were explicitly evicted from the CPU tier."""

    request_id: str
    elapsed_us: float
    num_keys: int


@dataclass(frozen=True)
class KVOffloadRestoreStart(EngineCoreEvent):
    """A restore (load) of blocks from a secondary tier began."""

    request_id: str
    num_keys: int


@dataclass(frozen=True)
class KVOffloadRestoreDone(EngineCoreEvent):
    """A restore (load) of blocks from a secondary tier finished."""

    request_id: str
    num_keys: int


@dataclass(frozen=True)
class KVOffloadTierEvict(EngineCoreEvent):
    """Blocks were evicted by the tiering manager."""

    request_id: str
    num_keys: int
    elapsed_us: float


@dataclass(frozen=True)
class KVOffloadSchedStep(EngineCoreEvent):
    """One scheduler step of the offloading connector finished."""

    elapsed_us: float


@dataclass(frozen=True)
class KVTransferPhase(EngineCoreEvent):
    """A transfer job was prepared and submitted (wall-clock phases)."""

    job_id: str
    direction: str  # "d2h" (NPU->CPU) or "h2d" (CPU->NPU)
    bytes: int
    num_ops: int
    desc_us: float
    sync_us: float
    submit_us: float


@dataclass(frozen=True)
class KVTransferGatherH2D(EngineCoreEvent):
    """Host page gather for a CPU->NPU transfer finished."""

    job_id: str
    elapsed_us: float
    num_dma_ops: int


@dataclass(frozen=True)
class KVTransferSwapD2H(EngineCoreEvent):
    """NPU->CPU block swap kernel finished."""

    job_id: str
    num_ops: int
    elapsed_us: float


@dataclass(frozen=True)
class KVTransferCopyDone(EngineCoreEvent):
    """A transfer job completed (both wall-clock and event timings)."""

    job_id: str
    direction: str
    bytes: int
    wall_ms: float
    event_ms: float


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
    function during engine startup). With no sinks registered, ``emit`` is a
    no-op, so the instrumentation is disabled by default.
    """

    _sinks: list[EventSink] = []
    _lock = threading.Lock()

    @classmethod
    def register_sink(cls, sink: EventSink) -> None:
        """Register a sink. Idempotent per sink instance."""
        with cls._lock:
            if sink not in cls._sinks:
                cls._sinks.append(sink)

    @classmethod
    def unregister_sink(cls, sink: EventSink) -> None:
        """Remove a previously registered sink."""
        with cls._lock:
            if sink in cls._sinks:
                cls._sinks.remove(sink)

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
            try:
                sink.emit(event)
            except Exception:  # noqa: BLE001 - degrade, never propagate
                pass


# Re-export dataclass helpers for plugin authors that need to introspect
# event payloads without importing the vLLM codebase.
EventPayload = dataclasses.asdict
