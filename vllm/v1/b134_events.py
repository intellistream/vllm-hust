# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional JSONL events for the KV tiering benchmark contract."""

import json
import os
import threading
import time
from typing import Any

_EVENTS_PATH = os.environ.get("B134_EVENTS_FILE")
EVENTS_ENABLED = bool(_EVENTS_PATH)
_FILE_DESCRIPTOR: int | None = None
_OPEN_LOCK = threading.Lock()


def _file_descriptor() -> int:
    global _FILE_DESCRIPTOR
    if _FILE_DESCRIPTOR is None:
        with _OPEN_LOCK:
            if _FILE_DESCRIPTOR is None:
                assert _EVENTS_PATH is not None
                _FILE_DESCRIPTOR = os.open(
                    _EVENTS_PATH,
                    os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                    0o600,
                )
    return _FILE_DESCRIPTOR


def emit(event: str, request_id: str, **fields: Any) -> None:
    """Append one event when ``B134_EVENTS_FILE`` is configured."""
    if not EVENTS_ENABLED:
        return
    payload = {
        "event": event,
        "fields": fields,
        "pid": os.getpid(),
        "request_id": request_id,
        "ts_monotonic_ns": time.monotonic_ns(),
    }
    line = json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
    os.write(_file_descriptor(), line.encode("utf-8"))
