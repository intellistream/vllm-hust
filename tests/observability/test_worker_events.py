# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / "vllm" / "observability" / "worker_events.py"
SPEC = importlib.util.spec_from_file_location("_worker_events_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
worker_events = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker_events
SPEC.loader.exec_module(worker_events)


@pytest.fixture(autouse=True)
def reset_listeners():
    worker_events.reset_worker_lifecycle_listeners_for_test()
    yield
    worker_events.reset_worker_lifecycle_listeners_for_test()


def test_default_off_has_no_listener() -> None:
    assert worker_events.has_worker_lifecycle_listeners() is False
    worker_events.emit_worker_lifecycle_event("ignored", value=1)


def test_listener_receives_immutable_event() -> None:
    events = []
    worker_events.register_worker_lifecycle_listener("test", events.append)

    worker_events.emit_worker_lifecycle_event("created", value=1)

    assert worker_events.has_worker_lifecycle_listeners() is True
    assert len(events) == 1
    assert events[0].name == "created"
    assert events[0].fields == {"value": 1}
    with pytest.raises(TypeError):
        events[0].fields["value"] = 2


def test_registration_is_idempotent_only_for_same_listener() -> None:
    def listener(event):
        pass

    worker_events.register_worker_lifecycle_listener("test", listener)
    worker_events.register_worker_lifecycle_listener("test", listener)

    with pytest.raises(ValueError):
        worker_events.register_worker_lifecycle_listener("test", lambda event: None)


def test_failing_listener_is_disabled_without_escaping() -> None:
    calls = []

    def failing(event):
        raise RuntimeError("observer failure")

    worker_events.register_worker_lifecycle_listener("failing", failing)
    worker_events.register_worker_lifecycle_listener("healthy", calls.append)

    worker_events.emit_worker_lifecycle_event("first")
    worker_events.emit_worker_lifecycle_event("second")

    assert [event.name for event in calls] == ["first", "second"]


def test_unregister_removes_listener() -> None:
    events = []
    worker_events.register_worker_lifecycle_listener("test", events.append)
    worker_events.unregister_worker_lifecycle_listener("test")

    worker_events.emit_worker_lifecycle_event("ignored")

    assert events == []
    assert worker_events.has_worker_lifecycle_listeners() is False
