"""B134 event emitter (minimal, observe-only instrumentation)."""
import os
import time

_FILE = os.environ.get("B134_EVENTS_FILE", "")

if _FILE:
    _f = open(_FILE, "a", buffering=1)  # line-buffered


def emit(event: str, req_id: str, extra: str = "") -> None:
    if not _FILE:
        return
    _f.write(f"{time.monotonic():.6f}\t{event}\t{req_id}\t{extra}\n")
