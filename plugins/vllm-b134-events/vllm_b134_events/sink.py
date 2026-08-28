# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B134 JSONL event sink plugin.

Subscribes to the generic typed event bus (``vllm.v1.events.EventBus``) and
serializes the subset of events needed by the B134 KV-tiering benchmark into
a line-buffered JSONL file.

Responsibilities (per review feedback):
- event selection: only B134-relevant event types are serialized;
- serialization: typed events -> JSONL lines, preserving the historical B134
  event names and field semantics so existing analysis scripts keep working;
- file/exporter lifecycle: the output file is opened lazily on first event
  and closed (flushed) by :meth:`B134JsonlSink.close`;
- bounded buffering: events are enqueued to a bounded queue consumed by a
  single background writer thread (overflow drops the event and bumps a
  counter instead of blocking the engine);
- error degradation: any writer failure is caught and counted, never
  propagated to the serving path.

Enable by installing this package (``pip install -e plugins/vllm-b134-events``)
and setting ``B134_EVENTS_FILE=/path/to/events.jsonl``.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from typing import Any

from vllm.v1.events import (
    EngineCoreEvent,
    KVOffloadEvict,
    KVOffloadRestoreDone,
    KVOffloadRestoreStart,
    KVOffloadSchedStep,
    KVOffloadStore,
    KVOffloadTierEvict,
    KVTransferCopyDone,
    KVTransferGatherH2D,
    KVTransferSubmit,
    KVTransferSwapD2H,
    RequestAdmitted,
    RequestPreempted,
    RequestResumed,
    RequestScheduled,
)

# Default bound on the pending-event queue (events). If the writer thread
# cannot keep up, new events are dropped and ``dropped_events`` is bumped.
_DEFAULT_QUEUE_MAX = 65536
_WRITER_STOP = object()


class B134JsonlSink:
    """Serialize selected engine-core events into the B134 JSONL format."""

    def __init__(
        self,
        path: str | None = None,
        queue_max: int = _DEFAULT_QUEUE_MAX,
    ) -> None:
        self._path = path or os.environ.get("B134_EVENTS_FILE", "")
        self._queue: queue.Queue[str | object] = queue.Queue(maxsize=queue_max)
        self._thread: threading.Thread | None = None
        self._file = None
        self._closed = False
        self.dropped_events = 0
        self._error_count = 0
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the background writer thread."""
        if self._thread is not None or not self._path:
            return
        self._thread = threading.Thread(
            target=self._writer_loop, name="b134-events-writer", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        """Flush and stop the writer thread."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(_WRITER_STOP)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._close_file()

    # -- EventBus protocol -------------------------------------------------

    def emit(self, event: EngineCoreEvent) -> None:
        """EventBus sink entry point (called on the engine thread)."""
        if self._closed or not self._path:
            return
        line = self._serialize(event)
        if line is None:
            return  # event type not selected by this sink
        try:
            self._queue.put_nowait(line)
        except queue.Full:
            with self._lock:
                self.dropped_events += 1

    # -- internals -----------------------------------------------------------

    def _serialize(self, event: EngineCoreEvent) -> str | None:
        """Map a typed event to a B134 JSONL line (or None if not selected)."""
        name: str
        request_id: str
        fields: dict[str, Any]

        if isinstance(event, RequestAdmitted):
            name, request_id, fields = "admission", event.request_id, {}
        elif isinstance(event, RequestScheduled):
            name, request_id, fields = "scheduled", event.request_id, {}
        elif isinstance(event, RequestPreempted):
            name, request_id, fields = "preempt", event.request_id, {}
        elif isinstance(event, RequestResumed):
            name, request_id, fields = "wakeup", event.request_id, {}
        elif isinstance(event, KVOffloadStore):
            name, request_id, fields = (
                "cpu_store",
                event.request_id,
                {
                    "duration_us": event.duration_us,
                    "evicted_keys": event.evicted_keys,
                    "stored_keys": event.stored_keys,
                },
            )
        elif isinstance(event, KVOffloadEvict):
            name, request_id, fields = (
                "cpu_evict",
                event.request_id,
                {
                    "duration_us": event.duration_us,
                    "evicted_keys": event.evicted_keys,
                },
            )
        elif isinstance(event, KVOffloadRestoreStart):
            name, request_id, fields = (
                "restore_start",
                event.request_id,
                {
                    "keys": event.keys,
                },
            )
        elif isinstance(event, KVOffloadRestoreDone):
            name, request_id, fields = (
                "restore_done",
                event.request_id,
                {
                    "keys": event.keys,
                },
            )
        elif isinstance(event, KVOffloadTierEvict):
            name, request_id, fields = (
                "evict",
                event.request_id,
                {
                    "duration_us": event.duration_us,
                    "keys": event.keys,
                },
            )
        elif isinstance(event, KVOffloadSchedStep):
            name, request_id, fields = (
                "sched_step",
                "step",
                {
                    "duration_us": event.duration_us,
                },
            )
        elif isinstance(event, KVTransferSwapD2H):
            name, request_id, fields = (
                "swap_d2h_submit",
                event.job_id,
                {
                    "descriptors": event.descriptors,
                    "duration_us": event.duration_us,
                },
            )
        elif isinstance(event, KVTransferGatherH2D):
            name, request_id, fields = (
                "gather_h2d",
                event.job_id,
                {
                    "dma_runs": event.dma_runs,
                    "duration_us": event.duration_us,
                },
            )
        elif isinstance(event, KVTransferSubmit):
            name, request_id, fields = (
                "transfer_submit",
                event.job_id,
                {
                    "bytes": event.bytes,
                    "dependency_us": event.dependency_us,
                    "descriptor_us": event.descriptor_us,
                    "descriptors": event.descriptors,
                    "direction": event.direction,
                    "submit_us": event.submit_us,
                },
            )
        elif isinstance(event, KVTransferCopyDone):
            name, request_id, fields = (
                "copy_observed_complete",
                event.job_id,
                {
                    "bytes": event.bytes,
                    "completion_observed_ms": event.completion_observed_ms,
                    "device_event_ms": event.device_event_ms,
                    "direction": event.direction,
                },
            )
        else:
            return None

        payload = {
            "event": name,
            "fields": fields,
            "pid": os.getpid(),
            "request_id": request_id,
            "ts_monotonic_ns": event.ts_monotonic_ns,
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _WRITER_STOP:
                return
            self._write_line(item)

    def _write_line(self, line: str) -> None:
        try:
            f = self._open_file()
            if f is None:
                return
            f.write(line)
            f.flush()  # line-buffered semantics preserved
        except Exception:  # noqa: BLE001 - error degradation
            with self._lock:
                self._error_count += 1

    def _open_file(self):
        if self._file is None:
            path = self._path
            if not path:
                return None
            try:
                # The sink intentionally owns this file across writes and closes it
                # in ``close``; a per-call context manager would defeat buffering.
                self._file = open(  # noqa: SIM115
                    path, "a", buffering=1, encoding="utf-8"
                )
            except OSError:
                with self._lock:
                    self._error_count += 1
                return None
        return self._file

    def _close_file(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except OSError:
                pass
            self._file = None
