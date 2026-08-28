# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the generic engine-core event bus and the B134 JSONL sink.

Pure-Python: no NPU/GPU required. Run from the repo root with
``vllm-b134-events`` installed (``pip install -e plugins/vllm-b134-events``):

    python -m pytest tests/v1/test_b134_events.py -v
"""

import importlib
import json
import os
import tempfile

import pytest
from vllm_b134_events.sink import B134JsonlSink

from vllm.v1.events import (
    EventBus,
    KVOffloadStore,
    KVOffloadTierEvict,
    KVTransferCopyDone,
    KVTransferSubmit,
    RequestAdmitted,
    RequestPreempted,
)


class CollectingSink:
    def __init__(self):
        self.events = []
        self.fail = False

    def emit(self, event):
        if self.fail:
            raise RuntimeError("sink failure")
        self.events.append(event)


@pytest.fixture(autouse=True)
def _clean_bus():
    EventBus._sinks = []
    EventBus.enabled = False
    yield
    EventBus._sinks = []
    EventBus.enabled = False


def test_event_bus_default_off():
    """With no sinks, emit is a no-op (no exception, no side effect)."""
    assert EventBus.enabled is False
    EventBus.emit(RequestAdmitted("req-1"))  # must not raise


def test_event_bus_enabled_after_register():
    s = CollectingSink()
    EventBus.register_sink(s)
    assert EventBus.enabled is True


def test_event_bus_dispatches_to_all_sinks():
    s1, s2 = CollectingSink(), CollectingSink()
    EventBus.register_sink(s1)
    EventBus.register_sink(s2)
    EventBus.emit(RequestAdmitted("req-1"))
    assert len(s1.events) == 1
    assert len(s2.events) == 1
    assert s1.events[0].request_id == "req-1"


def test_event_bus_swallows_sink_errors():
    """A failing sink must not break the serving path."""
    good, bad = CollectingSink(), CollectingSink()
    bad.fail = True
    EventBus.register_sink(good)
    EventBus.register_sink(bad)
    EventBus.emit(RequestAdmitted("req-1"))  # must not raise
    assert len(good.events) == 1


def test_event_bus_register_idempotent():
    s = CollectingSink()
    EventBus.register_sink(s)
    EventBus.register_sink(s)
    EventBus.emit(RequestPreempted("req-1"))
    assert len(s.events) == 1


def test_b134_sink_selects_only_relevant_events(tmp_path):
    """Unselected event types are ignored; selected ones are serialized."""
    path = tmp_path / "events.jsonl"
    sink = B134JsonlSink(str(path))
    sink.start()
    try:
        sink.emit(RequestAdmitted("req-1"))
        sink.emit(
            KVOffloadStore("req-1", duration_us=12, evicted_keys=1, stored_keys=4)
        )
        sink.emit(
            KVTransferSubmit(
                "job3",
                bytes=1024,
                dependency_us=2,
                descriptor_us=1,
                descriptors=2,
                direction="h2d",
                submit_us=3,
            )
        )
        sink.emit(KVOffloadTierEvict("req-2", duration_us=5, keys=7))
        sink.emit(
            KVTransferCopyDone(
                "job3",
                bytes=1024,
                completion_observed_ms=1.5,
                device_event_ms=1.2,
                direction="h2d",
            )
        )
        sink.close()
    finally:
        sink.close()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    parsed = [json.loads(line) for line in lines]

    assert parsed[0]["event"] == "admission"
    assert parsed[0]["request_id"] == "req-1"
    assert parsed[1]["event"] == "cpu_store"
    assert parsed[1]["fields"]["stored_keys"] == 4
    assert parsed[1]["fields"]["duration_us"] == 12
    assert parsed[2]["event"] == "transfer_submit"
    assert parsed[2]["fields"]["direction"] == "h2d"
    assert parsed[2]["fields"]["descriptors"] == 2
    assert parsed[3]["event"] == "evict"
    assert parsed[3]["fields"]["keys"] == 7
    assert parsed[4]["event"] == "copy_observed_complete"
    assert parsed[4]["fields"]["device_event_ms"] == 1.2
    # every line carries a monotonic timestamp
    for p in parsed:
        assert p["ts_monotonic_ns"] > 0


def test_b134_sink_disabled_without_path():
    """Without B134_EVENTS_FILE / path the sink never writes."""
    old = os.environ.pop("B134_EVENTS_FILE", None)
    try:
        sink = B134JsonlSink()
        sink.start()
        sink.emit(RequestAdmitted("req-1"))
        sink.close()
    finally:
        if old is not None:
            os.environ["B134_EVENTS_FILE"] = old


def test_b134_plugin_does_not_enable_bus_without_path(monkeypatch):
    """Installing the plugin must keep event production disabled by default."""
    monkeypatch.delenv("B134_EVENTS_FILE", raising=False)
    import vllm_b134_events

    plugin = importlib.reload(vllm_b134_events)
    plugin.register()

    assert EventBus.enabled is False
    assert EventBus._sinks == []


def test_b134_sink_queue_overflow_degrades():
    """Bounded queue overflow drops events instead of blocking/raising."""
    path = tempfile.mktemp(suffix=".jsonl")
    try:
        sink = B134JsonlSink(path, queue_max=2)
        sink.start()
        try:
            for i in range(100):
                sink.emit(RequestAdmitted(f"req-{i}"))
        finally:
            sink.close()
        assert sink.dropped_events > 0
    finally:
        if os.path.exists(path):
            os.unlink(path)
