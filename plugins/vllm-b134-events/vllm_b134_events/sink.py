# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B134 TSV event sink plugin.

Subscribes to the generic typed event bus (``vllm.v1.events.EventBus``) and
serializes the subset of events needed by the B134 KV-tiering benchmark into
a line-buffered TSV file.

Responsibilities (per review feedback):
- event selection: only B134-relevant event types are serialized;
- serialization: typed events -> TSV lines, preserving the historical B134
  event names and field semantics so existing analysis scripts keep working;
- file/exporter lifecycle: the output file is opened lazily on first event
  and closed (flushed) by :meth:`B134TsvSink.close`;
- bounded buffering: events are enqueued to a bounded queue consumed by a
  single background writer thread (overflow drops the event and bumps a
  counter instead of blocking the engine);
- error degradation: any writer failure is caught and counted, never
  propagated to the serving path.

Enable by installing this package (``pip install -e plugins/vllm-b134-events``)
and setting ``B134_EVENTS_FILE=/path/to/events.tsv``.
"""

from __future__ import annotations

import os
import queue
import threading
import time
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
    KVTransferPhase,
    KVTransferSwapD2H,
    RequestAdmitted,
    RequestFinished,
    RequestPreempted,
    RequestResumed,
    RequestScheduled,
)

# Default bound on the pending-event queue (events). If the writer thread
# cannot keep up, new events are dropped and ``dropped_events`` is bumped.
_DEFAULT_QUEUE_MAX = 65536
_WRITER_STOP = object()


class B134TsvSink:
    """Serialize selected engine-core events into the B134 TSV format."""

    def __init__(
        self,
        path: str | None = None,
        queue_max: int = _DEFAULT_QUEUE_MAX,
    ) -> None:
        self._path = path or os.environ.get("B134_EVENTS_FILE", "")
        self._queue: "queue.Queue[EngineCoreEvent | object]" = queue.Queue(
            maxsize=queue_max
        )
        self._thread: threading.Thread | None = None
        self._file = None
        self._closed = False
        self.dropped_events = 0
        self._error_count = 0
        self._dropped_lock = threading.Lock()

    # -- lifecycle -------------------------------------------------------

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

    # -- EventBus protocol ------------------------------------------------

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
            with self._dropped_lock:
                self.dropped_events += 1

    # -- internals ---------------------------------------------------------

    def _serialize(self, event: EngineCoreEvent) -> str | None:
        """Map a typed event to a B134 TSV line (or None if not selected)."""
        extra: dict[str, Any] = {}
        name: str
        req_id: str

        if isinstance(event, RequestAdmitted):
            name, req_id = "admission", event.request_id
        elif isinstance(event, RequestScheduled):
            name, req_id = "scheduled", event.request_id
        elif isinstance(event, RequestPreempted):
            name, req_id = "preempt", event.request_id
        elif isinstance(event, RequestResumed):
            name, req_id = "wakeup", event.request_id
        elif isinstance(event, RequestFinished):
            name, req_id = "finish", event.request_id
            extra["out_tokens"] = event.output_tokens
        elif isinstance(event, KVOffloadStore):
            name, req_id = "cpu_store", event.request_id
            extra = {
                "us": f"{event.elapsed_us:.0f}",
                "n": event.num_keys,
                "evict": event.evicted,
            }
        elif isinstance(event, KVOffloadEvict):
            name, req_id = "cpu_evict", event.request_id
            extra = {"us": f"{event.elapsed_us:.0f}", "n": event.num_keys}
        elif isinstance(event, KVOffloadRestoreStart):
            name, req_id = "restore_start", event.request_id
            extra = {"n": event.num_keys}
        elif isinstance(event, KVOffloadRestoreDone):
            name, req_id = "restore_done", event.request_id
            extra = {"n": event.num_keys}
        elif isinstance(event, KVOffloadTierEvict):
            name, req_id = "evict", event.request_id
            extra = {"n": event.num_keys, "us": f"{event.elapsed_us:.0f}"}
        elif isinstance(event, KVOffloadSchedStep):
            name, req_id = "sched_step", "step"
            extra = {"us": f"{event.elapsed_us:.0f}"}
        elif isinstance(event, KVTransferPhase):
            name, req_id = "transfer_phase", event.job_id
            extra = {
                "dir": event.direction,
                "bytes": event.bytes,
                "n_ops": event.num_ops,
                "desc_us": f"{event.desc_us:.0f}",
                "sync_us": f"{event.sync_us:.0f}",
                "submit_us": f"{event.submit_us:.0f}",
            }
        elif isinstance(event, KVTransferGatherH2D):
            name, req_id = "gather_h2d", event.job_id
            extra = {
                "us": f"{event.elapsed_us:.0f}",
                "chunks": event.num_dma_ops,
            }
        elif isinstance(event, KVTransferSwapD2H):
            name, req_id = "swap_d2h", event.job_id
            extra = {"n": event.num_ops, "us": f"{event.elapsed_us:.0f}"}
        elif isinstance(event, KVTransferCopyDone):
            name, req_id = "copy_done", event.job_id
            extra = {
                "dir": event.direction,
                "bytes": event.bytes,
                "copy_ms_wall": f"{event.wall_ms:.3f}",
                "copy_ms_event": f"{event.event_ms:.3f}",
                "note": "wall_authoritative",
            }
        else:
            return None

        fields = " ".join(f"{k}={v}" for k, v in extra.items())
        ts = event.ts_monotonic_ns / 1e9
        return f"{ts:.6f}\t{name}\t{req_id}\t{fields}\n"

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
            with self._dropped_lock:
                self._error_count += 1

    def _open_file(self):
        if self._file is None:
            path = self._path
            if not path:
                return None
            try:
                self._file = open(path, "a", buffering=1, encoding="utf-8")
            except OSError:
                with self._dropped_lock:
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
