# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Default-off, fail-open lifecycle observer seam."""

import os
from functools import lru_cache
from typing import Any

TRACE_EXPORT_ENV = "VLLM_RLP_TRACE_EXPORT_PATH"


@lru_cache(maxsize=1)
def get_request_lifecycle_observer() -> Any | None:
    if not os.environ.get(TRACE_EXPORT_ENV, "").strip():
        return None
    try:
        from vllm_request_lifecycle_profiler.plugin import get_lifecycle_observer

        return get_lifecycle_observer()
    except Exception:
        return None


def _observe(method_name: str, *args: object, **kwargs: object) -> None:
    observer = get_request_lifecycle_observer()
    if observer is None:
        return
    try:
        callback = getattr(observer, method_name, None)
        if callable(callback):
            callback(*args, **kwargs)
    except Exception:
        return


def observe_request_started(runtime_request_id: str) -> None:
    _observe("request_started", runtime_request_id)


def observe_abort_attempt(runtime_request_id: str) -> None:
    _observe("abort_attempt", runtime_request_id)


def observe_request_terminal(
    runtime_request_id: str,
    terminal_cause: str,
    *,
    generated_tokens_total: int = 0,
) -> None:
    _observe(
        "request_terminal",
        runtime_request_id,
        terminal_cause,
        generated_tokens_total=generated_tokens_total,
    )


def observe_engine_failure(runtime_request_ids: object) -> None:
    _observe("engine_failure", runtime_request_ids)


def observe_worker_generation_exit(
    worker_name: str,
    *,
    worker_pid: int | None,
    exit_code: int | None,
) -> None:
    _observe(
        "worker_generation_exit",
        worker_name,
        worker_pid=worker_pid,
        exit_code=exit_code,
    )


def close_engine_failure_observers() -> None:
    """Fail-open close of optional process-local observers before core death."""

    if not os.environ.get(TRACE_EXPORT_ENV, "").strip():
        return
    try:
        from vllm_request_lifecycle_profiler.plugin import (
            close_engine_failure_observers as close_plugin_observers,
        )

        close_plugin_observers()
    except Exception:
        return


def observe_resource_transition(
    runtime_request_id: str,
    resource_type: str,
    resource_id: str,
    transition: str,
    worker_generation: str,
    resource_units: int = 1,
) -> None:
    _observe(
        "resource_transition",
        runtime_request_id,
        resource_type,
        resource_id,
        transition,
        worker_generation,
        resource_units,
    )


def is_resource_observation_enabled() -> bool:
    """Return whether the optional lifecycle exporter was explicitly enabled."""

    return bool(os.environ.get(TRACE_EXPORT_ENV, "").strip())
