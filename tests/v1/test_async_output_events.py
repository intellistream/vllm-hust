# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parents[2]
EVENTS_PATH = ROOT / "vllm" / "v1" / "events.py"
SPEC = importlib.util.spec_from_file_location("_async_output_events", EVENTS_PATH)
assert SPEC is not None and SPEC.loader is not None
events = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = events
SPEC.loader.exec_module(events)


@pytest.fixture(autouse=True)
def reset_event_bus():
    events.EventBus._sinks = []
    events.EventBus.enabled = False
    yield
    events.EventBus._sinks = []
    events.EventBus.enabled = False


def test_async_output_lifecycle_uses_existing_typed_bus() -> None:
    received: list[Any] = []

    class Sink:
        def emit(self, event: Any) -> None:
            received.append(event)

    sink = Sink()
    events.EventBus.register_sink(sink)
    lifecycle = [
        events.AsyncOutputCreated(1, ("req-1",), (1, 1), "int64"),
        events.AsyncOutputCopyIssued(1, 2, 3, 8, 4),
        events.AsyncOutputWaitComplete(1, 5),
        events.AsyncOutputMaterialized(1, 6),
        events.AsyncOutputRetained(2),
        events.AsyncOutputConsumed(2, 7, 8),
    ]

    for event in lifecycle:
        events.EventBus.emit(event)

    assert received == lifecycle
    assert all(event.ts_monotonic_ns > 0 for event in received)


def test_async_output_copy_failure_is_typed_and_fail_open() -> None:
    received: list[Any] = []

    class FailingSink:
        def emit(self, event: Any) -> None:
            raise RuntimeError("observer failure")

    class HealthySink:
        def emit(self, event: Any) -> None:
            received.append(event)

    events.EventBus.register_sink(FailingSink())
    events.EventBus.register_sink(HealthySink())

    event = events.AsyncOutputCopyFailed(1, "sampled_token_d2h")
    events.EventBus.emit(event)

    assert received == [event]


def test_actual_worker_call_sites_emit_complete_lifecycle() -> None:
    expected = {
        "AsyncOutputCreated",
        "AsyncOutputCopyIssued",
        "AsyncOutputCopyFailed",
        "AsyncOutputWaitComplete",
        "AsyncOutputMaterialized",
        "AsyncOutputRetained",
        "AsyncOutputConsumed",
    }
    emitted: set[str] = set()
    for relative_path in (
        "vllm/v1/worker/gpu_model_runner.py",
        "vllm/v1/worker/gpu_input_batch.py",
    ):
        tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id in expected:
                emitted.add(node.func.id)

    assert emitted == expected
