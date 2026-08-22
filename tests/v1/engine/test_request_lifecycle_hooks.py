# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.engine import request_lifecycle_hooks as hooks

pytestmark = pytest.mark.cpu_test


class RecordingObserver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def __getattr__(self, name: str):
        def record(*args: object, **kwargs: object) -> None:
            self.calls.append((name, args, kwargs))

        return record


def test_disabled_seam_does_not_resolve_optional_profiler(monkeypatch) -> None:
    monkeypatch.delenv(hooks.TRACE_EXPORT_ENV, raising=False)
    hooks.get_request_lifecycle_observer.cache_clear()

    assert hooks.get_request_lifecycle_observer() is None
    hooks.observe_request_started("request-0")
    hooks.observe_request_terminal("request-0", "complete")


def test_all_observer_callbacks_are_forwarded_and_fail_open(monkeypatch) -> None:
    observer = RecordingObserver()
    monkeypatch.setattr(hooks, "get_request_lifecycle_observer", lambda: observer)

    hooks.observe_request_started("request-0")
    hooks.observe_abort_attempt("request-0")
    hooks.observe_request_terminal(
        "request-0", "explicit_cancel", generated_tokens_total=2
    )
    hooks.observe_engine_failure(("request-1",))
    hooks.observe_worker_generation_exit("worker-0", worker_pid=10, exit_code=1)
    hooks.observe_resource_transition(
        "request-0", "kv", "block-0", "release", "generation-0"
    )

    assert [name for name, _args, _kwargs in observer.calls] == [
        "request_started",
        "abort_attempt",
        "request_terminal",
        "engine_failure",
        "worker_generation_exit",
        "resource_transition",
    ]

    monkeypatch.setattr(
        hooks,
        "get_request_lifecycle_observer",
        lambda: type(
            "RaisingObserver",
            (),
            {"request_started": lambda *_args: (_ for _ in ()).throw(RuntimeError())},
        )(),
    )
    hooks.observe_request_started("request-0")


def test_engine_failure_close_is_disabled_by_default_and_forwarded_when_enabled(
    monkeypatch,
) -> None:
    from vllm_request_lifecycle_profiler import plugin

    calls: list[str] = []
    monkeypatch.delenv(hooks.TRACE_EXPORT_ENV, raising=False)
    monkeypatch.setattr(
        plugin, "close_engine_failure_observers", lambda: calls.append("close")
    )
    hooks.close_engine_failure_observers()
    assert calls == []

    monkeypatch.setenv(hooks.TRACE_EXPORT_ENV, "/tmp/trace")
    hooks.close_engine_failure_observers()
    assert calls == ["close"]
