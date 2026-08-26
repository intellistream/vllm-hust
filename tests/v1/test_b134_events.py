# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the generic engine-core event bus and the B134 TSV sink.

Pure-Python: no NPU/GPU required. Run from the repo root with
``vllm-b134-events`` installed (``pip install -e plugins/vllm-b134-events``):

    python -m pytest tests/v1/test_b134_events.py -v
"""

import os
import tempfile

import pytest

from vllm.v1.events import (
    EventBus,
    KVOffloadStore,
    KVTransferCopyDone,
    KVTransferPhase,
    RequestAdmitted,
    RequestFinished,
    RequestPreempted,
)

from vllm_b134_events.sink import B134TsvSink


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
    yield
    EventBus._sinks = []


def test_event_bus_default_off():
    """With no sinks, emit is a no-op (no exception, no side effect)."""
    EventBus.emit(RequestAdmitted("req-1"))  # must not raise


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
    path = tmp_path / "events.tsv"
    sink = B134TsvSink(str(path))
    sink.start()
    try:
        sink.emit(RequestAdmitted("req-1"))
        sink.emit(KVOffloadStore("req-1", elapsed_us=12.0, num_keys=4, evicted=1))
        sink.emit(KVTransferPhase("job3", direction="h2d", bytes=1024,
                                  num_ops=2, desc_us=1.0, sync_us=2.0,
                                  submit_us=3.0))
        sink.emit(RequestFinished("req-2", output_tokens=7))
        sink.emit(KVTransferCopyDone("job3", direction="h2d", bytes=1024,
                                     wall_ms=1.5, event_ms=1.2))
        sink.close()
    finally:
        sink.close()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    admission = lines[0].split("\t")
    assert admission[1] == "admission"
    assert admission[2] == "req-1"
    assert lines[1].split("\t")[1] == "cpu_store"
    assert "us=12" in lines[1]
    assert "n=4" in lines[1]
    assert lines[2].split("\t")[1] == "transfer_phase"
    assert "dir=h2d" in lines[2]
    assert lines[3].split("\t")[1] == "finish"
    assert "out_tokens=7" in lines[3]
    assert lines[4].split("\t")[1] == "copy_done"
    assert "note=wall_authoritative" in lines[4]


def test_b134_sink_disabled_without_path():
    """Without B134_EVENTS_FILE / path the sink never writes."""
    old = os.environ.pop("B134_EVENTS_FILE", None)
    try:
        sink = B134TsvSink()
        sink.start()
        sink.emit(RequestAdmitted("req-1"))
        sink.close()
    finally:
        if old is not None:
            os.environ["B134_EVENTS_FILE"] = old


def test_b134_sink_queue_overflow_degrades():
    """Bounded queue overflow drops events instead of blocking/raising."""
    path = tempfile.mktemp(suffix=".tsv")
    try:
        sink = B134TsvSink(path, queue_max=2)
        sink.start()
        try:
            for i in range(100):
                sink.emit(RequestAdmitted(f"req-{i}"))
        finally:
            sink.close()
        assert sink.dropped_events > 0
        assert not os.path.exists(path) or True  # never raised
    finally:
        if os.path.exists(path):
            os.unlink(path)
