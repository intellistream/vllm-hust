# SPDX-License-Identifier: Apache-2.0
"""Default-off event observer for the TP1 async output pathology probe.

The observer records source-path events only. It does not allocate device
buffers, change synchronization, or implement an optimization.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_PATH_ENV = "VLLM_TP1_OUTPUT_OBSERVER_PATH"
_MAX_EVENTS_ENV = "VLLM_TP1_OUTPUT_OBSERVER_MAX_EVENTS"
_FLUSH_EVENTS_ENV = "VLLM_TP1_OUTPUT_OBSERVER_FLUSH_EVENTS"


class OutputPathologyObserver:
    """Collect a reconstructable event ledger without changing output data."""

    def __init__(
        self,
        path: str,
        *,
        max_events: int = 1_000_000,
        flush_events: int = 4096,
    ) -> None:
        if max_events <= 0 or flush_events <= 0:
            raise ValueError("observer limits must be positive")
        self.path = Path(path)
        self.max_events = max_events
        self.flush_events = flush_events
        self._events: list[dict[str, Any]] = []
        self._next_output_id = 0
        self._storage_to_output: dict[int, int] = {}
        self._output_to_storage: dict[int, int] = {}
        self._retained_storages: set[int] = set()
        self._event_count = 0
        self._dropped_events = 0
        self._write_errors = 0
        self._checkpoint_dirty = False
        self._header_written = False
        self._disabled = False
        self._lock = threading.Lock()
        atexit.register(self.flush)

    @staticmethod
    def now_ns() -> int:
        return time.perf_counter_ns()

    def start_output(
        self,
        req_ids: Sequence[str],
        shape: Sequence[int],
        dtype: str,
    ) -> int:
        with self._lock:
            output_id = self._next_output_id
            self._next_output_id += 1
        fingerprint_input = json.dumps(
            [list(req_ids), list(shape), dtype],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        self._record(
            "output_created",
            output_id=output_id,
            request_fingerprint=hashlib.sha256(fingerprint_input).hexdigest(),
            request_count=len(req_ids),
            shape=list(shape),
            dtype=dtype,
        )
        return output_id

    def record_copy_issued(
        self,
        output_id: int,
        *,
        storage_id: int,
        event_id: int,
        nbytes: int,
        dispatch_ns: int,
    ) -> None:
        with self._lock:
            can_link_future_events = self._record_locked(
                "fresh_cpu_tensor_copy_issued",
                output_id=output_id,
                storage_id=storage_id,
                event_id=event_id,
                nbytes=nbytes,
                dispatch_ns=dispatch_ns,
            )
            if not can_link_future_events:
                return
            self._release_storage_locked(storage_id)
            self._release_output_locked(output_id)
            self._storage_to_output[storage_id] = output_id
            self._output_to_storage[output_id] = storage_id

    def record_output_wait(self, output_id: int, wait_ns: int) -> None:
        self._record("output_event_wait", output_id=output_id, wait_ns=wait_ns)

    def record_output_materialization(
        self, output_id: int, materialization_ns: int
    ) -> None:
        with self._lock:
            self._record_locked(
                "output_python_materialization",
                output_id=output_id,
                materialization_ns=materialization_ns,
            )
            storage_id = self._output_to_storage.get(output_id)
            if (
                storage_id is not None
                and storage_id not in self._retained_storages
            ):
                self._release_storage_locked(storage_id)

    def record_input_batch_retain(self, storage_id: int) -> None:
        with self._lock:
            output_id = self._storage_to_output.get(storage_id)
            can_link_future_events = self._record_locked(
                "input_batch_retain",
                output_id=output_id,
                storage_id=storage_id,
            )
            if can_link_future_events and output_id is not None:
                self._retained_storages.add(storage_id)

    def record_input_batch_consume(
        self,
        storage_id: int,
        *,
        wait_ns: int,
        materialization_ns: int,
    ) -> None:
        with self._lock:
            self._record_locked(
                "input_batch_consume",
                output_id=self._storage_to_output.get(storage_id),
                storage_id=storage_id,
                wait_ns=wait_ns,
                materialization_ns=materialization_ns,
            )
            self._release_storage_locked(storage_id)

    def record_exception(self, output_id: int, phase: str) -> None:
        self._record("observer_seam_exception", output_id=output_id, phase=phase)

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _record(self, event: str, **fields: Any) -> None:
        with self._lock:
            self._record_locked(event, **fields)

    def _record_locked(self, event: str, **fields: Any) -> bool:
        """Record an event and report whether future links remain useful."""
        if self._disabled:
            return False
        if self._event_count >= self.max_events:
            self._dropped_events += 1
            self._checkpoint_dirty = True
            self._clear_auxiliary_state_locked()
            return False
        self._event_count += 1
        self._checkpoint_dirty = True
        self._events.append(
            {
                "event": event,
                "monotonic_ns": time.perf_counter_ns(),
                **fields,
            }
        )
        if self._event_count >= self.max_events:
            self._clear_auxiliary_state_locked()
        if len(self._events) >= self.flush_events:
            self._flush_locked()
        return not self._disabled and self._event_count < self.max_events

    def _release_storage_locked(self, storage_id: int) -> None:
        output_id = self._storage_to_output.pop(storage_id, None)
        self._retained_storages.discard(storage_id)
        if (
            output_id is not None
            and self._output_to_storage.get(output_id) == storage_id
        ):
            self._output_to_storage.pop(output_id, None)

    def _release_output_locked(self, output_id: int) -> None:
        storage_id = self._output_to_storage.pop(output_id, None)
        if storage_id is None:
            return
        self._retained_storages.discard(storage_id)
        if self._storage_to_output.get(storage_id) == output_id:
            self._storage_to_output.pop(storage_id, None)

    def _clear_auxiliary_state_locked(self) -> None:
        self._storage_to_output.clear()
        self._output_to_storage.clear()
        self._retained_storages.clear()

    def _flush_locked(self) -> None:
        if self._disabled or not self._checkpoint_dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if self._header_written else "w"
            with self.path.open(mode, encoding="utf-8", newline="\n") as file:
                if not self._header_written:
                    file.write(
                        json.dumps(
                            {
                                "event": "observer_header",
                                "schema_version": "tp1-output-pathology/v1",
                                "pid": os.getpid(),
                                "path_env": _PATH_ENV,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    self._header_written = True
                for event in self._events:
                    file.write(json.dumps(event, separators=(",", ":")) + "\n")
                file.write(
                    json.dumps(
                        {
                            "event": "observer_checkpoint",
                            "event_count": self._event_count,
                            "dropped_events": self._dropped_events,
                            "write_errors": self._write_errors,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            self._events.clear()
            self._checkpoint_dirty = False
        except OSError:
            # Observation must never alter serving correctness.
            self._write_errors += 1
            self._events.clear()
            self._checkpoint_dirty = False
            self._disabled = True
            self._clear_auxiliary_state_locked()


_observer: OutputPathologyObserver | None = None
_observer_initialized = False
_observer_lock = threading.Lock()


def get_output_pathology_observer() -> OutputPathologyObserver | None:
    global _observer, _observer_initialized
    if _observer_initialized:
        return _observer
    with _observer_lock:
        if _observer_initialized:
            return _observer
        path = os.getenv(_PATH_ENV)
        if path:
            try:
                _observer = OutputPathologyObserver(
                    path,
                    max_events=int(os.getenv(_MAX_EVENTS_ENV, "1000000")),
                    flush_events=int(os.getenv(_FLUSH_EVENTS_ENV, "4096")),
                )
            except (OSError, ValueError):
                _observer = None
        _observer_initialized = True
        return _observer


def reset_output_pathology_observer_for_test() -> None:
    global _observer, _observer_initialized
    with _observer_lock:
        if _observer is not None:
            _observer.flush()
        _observer = None
        _observer_initialized = False
