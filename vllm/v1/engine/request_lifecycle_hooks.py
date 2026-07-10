# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Optional request-lifecycle profiler bridge.

The vLLM-HUST runtime must remain usable when the research profiler package is
not installed. This shim therefore imports the profiler bridge only when
`VLLM_RLP_TRACE_EXPORT_PATH` is set and silently degrades to no-op otherwise.
"""

import logging
import os
import time
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

TRACE_EXPORT_ENV = "VLLM_RLP_TRACE_EXPORT_PATH"


def monotonic_ms(timestamp_s: float | None = None) -> float:
    source = time.monotonic() if timestamp_s is None else timestamp_s
    return source * 1000.0


@lru_cache(maxsize=1)
def get_request_lifecycle_hooks() -> Any | None:
    if not os.environ.get(TRACE_EXPORT_ENV, "").strip():
        return None
    try:
        from vllm_request_lifecycle_profiler.runtime_hooks import (
            RuntimeLifecycleHooks,
        )
    except Exception as exc:  # pragma: no cover - depends on optional overlay.
        logger.warning(
            "Request lifecycle profiler export requested, but the profiler "
            "overlay could not be imported: %s",
            exc,
        )
        return None
    try:
        return RuntimeLifecycleHooks.from_env()
    except Exception as exc:  # pragma: no cover - defensive runtime guard.
        logger.warning("Request lifecycle profiler hook initialization failed: %s", exc)
        return None


def emit_lifecycle(method_name: str, request_id: str, **metadata: Any) -> None:
    hooks = get_request_lifecycle_hooks()
    if hooks is None:
        return
    timestamp_ms = metadata.pop("timestamp_ms", None)
    try:
        method = getattr(hooks, method_name)
        method(request_id, timestamp_ms=timestamp_ms, **_sanitize_metadata(metadata))
    except Exception as exc:  # pragma: no cover - tracing must not break serving.
        logger.debug("Request lifecycle profiler hook failed: %s", exc)


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, bool | float | int | str]:
    result: dict[str, bool | float | int | str] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, bool | float | int | str):
            result[key] = value
        else:
            result[key] = str(value)
    return result
