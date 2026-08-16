# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
import random
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

# O_DIRECT is Linux-specific and not available on macOS
O_DIRECT = getattr(os, "O_DIRECT", 0)

# Thread-local storage for unique temporary file suffixes
_thread_local = threading.local()

InstrumentationCallback = Callable[[dict[str, object]], None]


def _emit(
    callback: InstrumentationCallback | None,
    event: str,
    *,
    operation: str,
    path: str,
    identity: dict[str, object] | None,
    **fields: object,
) -> None:
    if callback is not None:
        try:
            callback(
                {
                    "event": event,
                    "timestamp_ns": time.monotonic_ns(),
                    "operation": operation,
                    "path": path,
                    **(identity or {}),
                    **fields,
                }
            )
        except Exception:
            logger.exception("Filesystem tier instrumentation callback failed")


def _get_tmp_suffix() -> str:
    """Generate a thread-local unique suffix for temporary files."""
    try:
        return _thread_local.tmp_suffix
    except AttributeError:
        _thread_local.tmp_suffix = f"_{random.randint(0, 2**63 - 1)}.tmp"
        return _thread_local.tmp_suffix


def _ensure_dirs(path: str) -> None:
    """Create parent directories of *path* if they don't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def store_block(
    dest_path: str,
    buffer: memoryview,
    offset: int,
    block_size: int,
    *,
    instrumentation_callback: InstrumentationCallback | None = None,
    instrumentation_identity: dict[str, object] | None = None,
) -> None:
    """
    Store callback: Writes to a temp file then atomically replaces the destination.
    """
    # Check if block already exists to avoid redundant writes
    if os.path.exists(dest_path):
        _emit(
            instrumentation_callback,
            "io_skip",
            operation="write",
            path=dest_path,
            identity=instrumentation_identity,
            reason="destination_exists",
        )
        return

    tmp_path = dest_path + _get_tmp_suffix()
    # Ensure parent directories exist
    _ensure_dirs(dest_path)

    # Write block atomically. Cast to a flat byte view so the slice uses byte
    # indices; the raw memoryview may be multi-dimensional with itemsize > 1.
    view_slice = buffer.cast("B")[offset : offset + block_size]
    try:
        fd = os.open(
            tmp_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_TRUNC | O_DIRECT,
            0o644,
        )
        try:
            call_start_ns = time.monotonic_ns()
            _emit(
                instrumentation_callback,
                "io_call_start",
                operation="write",
                path=dest_path,
                identity=instrumentation_identity,
                requested_bytes=len(view_slice),
                call_start_ns=call_start_ns,
            )
            written = os.write(fd, view_slice)
            call_finish_ns = time.monotonic_ns()
            _emit(
                instrumentation_callback,
                "io_call_finish",
                operation="write",
                path=dest_path,
                identity=instrumentation_identity,
                requested_bytes=len(view_slice),
                completed_bytes=written,
                call_start_ns=call_start_ns,
                call_finish_ns=call_finish_ns,
                call_ns=call_finish_ns - call_start_ns,
                success=written == len(view_slice),
            )
            if written < len(view_slice):
                raise OSError(
                    f"Short write: expected {len(view_slice)} bytes, wrote {written}"
                )
        finally:
            os.close(fd)
        os.replace(tmp_path, dest_path)
    except Exception as exc:
        _emit(
            instrumentation_callback,
            "io_error",
            operation="write",
            path=dest_path,
            identity=instrumentation_identity,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        try:
            os.remove(tmp_path)
        except OSError as cleanup_exc:
            logger.warning("Failed to remove temp file %s: %s", tmp_path, cleanup_exc)
        raise


def load_block(
    source_path: str,
    view: memoryview,
    offset: int,
    block_size: int,
    *,
    instrumentation_callback: InstrumentationCallback | None = None,
    instrumentation_identity: dict[str, object] | None = None,
) -> None:
    """
    Load callback: read one KV block from disk. Remove the file on failure.
    """
    fd: int | None = None
    view_slice = view.cast("B")[offset : offset + block_size]
    try:
        fd = os.open(source_path, os.O_RDONLY | O_DIRECT)
        call_start_ns = time.monotonic_ns()
        _emit(
            instrumentation_callback,
            "io_call_start",
            operation="readv",
            path=source_path,
            identity=instrumentation_identity,
            requested_bytes=block_size,
            call_start_ns=call_start_ns,
        )
        bytes_read = os.readv(fd, [view_slice])
        call_finish_ns = time.monotonic_ns()
        _emit(
            instrumentation_callback,
            "io_call_finish",
            operation="readv",
            path=source_path,
            identity=instrumentation_identity,
            requested_bytes=block_size,
            completed_bytes=bytes_read,
            call_start_ns=call_start_ns,
            call_finish_ns=call_finish_ns,
            call_ns=call_finish_ns - call_start_ns,
            success=bytes_read == block_size,
        )
        if bytes_read < block_size:
            raise OSError(f"Short read: expected {block_size} bytes, read {bytes_read}")
    except Exception as exc:
        _emit(
            instrumentation_callback,
            "io_error",
            operation="readv",
            path=source_path,
            identity=instrumentation_identity,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        try:
            os.remove(source_path)
        except OSError as cleanup_exc:
            logger.warning(
                "Failed to remove unreadable file %s: %s", source_path, cleanup_exc
            )
        raise
    finally:
        if fd is not None:
            os.close(fd)
