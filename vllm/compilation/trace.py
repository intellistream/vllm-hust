# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in structured tracing for compilation and graph-cache behavior."""

import json
import os
import time
from typing import Any

TRACE_PATH_ENV = "VLLM_COMPILATION_TRACE_PATH"


def emit_compilation_trace(event: str, **fields: Any) -> None:
    """Append one best-effort JSON event without affecting serving behavior."""
    path = os.getenv(TRACE_PATH_ENV, "").strip()
    if not path:
        return

    record = {
        "schema_version": 1,
        "event": event,
        "timestamp_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        "rank": os.getenv("RANK"),
        "local_rank": os.getenv("LOCAL_RANK"),
        **fields,
    }
    try:
        payload = (json.dumps(record, sort_keys=True, default=str) + "\n").encode()
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
    except OSError:
        return
