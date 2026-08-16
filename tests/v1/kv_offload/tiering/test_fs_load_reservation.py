# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import mmap
import threading
from typing import Any

from vllm.v1.kv_offload.tiering.fs.io import load_block, store_block
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool


def test_strict_load_reservation_keeps_read_worker_off_store_queue():
    events: list[dict[str, Any]] = []
    first_store_started = threading.Event()
    second_store_started = threading.Event()
    release_store = threading.Event()
    load_started = threading.Event()

    def first_store():
        first_store_started.set()
        assert release_store.wait(timeout=5)

    def second_store():
        second_store_started.set()

    pool = DualQueueThreadPool(
        n_read_threads=1,
        n_write_threads=1,
        strict_load_reservation=True,
        instrumentation_callback=events.append,
    )
    try:
        pool.enqueue_store(1, 2, [first_store, second_store])
        assert first_store_started.wait(timeout=5)
        assert not second_store_started.wait(timeout=0.1)

        pool.enqueue_load(2, 1, [load_started.set])
        assert load_started.wait(timeout=5)
        assert not second_store_started.is_set()
        release_store.set()
        pool.wait_idle()
    finally:
        release_store.set()
        pool.shutdown(wait=True)

    starts = [event for event in events if event["event"] == "task_start"]
    assert any(
        event["job_id"] == 2
        and event["direction"] == "load"
        and event["worker_class"] == "read_priority"
        for event in starts
    )
    assert not any(
        event["direction"] == "store" and event["worker_class"] == "read_priority"
        for event in starts
    )
    finishes = [event for event in events if event["event"] == "job_finish"]
    assert {event["job_id"] for event in finishes} == {1, 2}
    assert all(event["complete_ns"] >= 0 for event in finishes)


def test_default_pool_remains_work_conserving_for_store_tasks():
    store_started = threading.Event()
    pool = DualQueueThreadPool(n_read_threads=1, n_write_threads=0)
    try:
        pool.enqueue_store(1, 1, [store_started.set])
        assert store_started.wait(timeout=5)
        pool.wait_idle()
    finally:
        pool.shutdown(wait=True)


def test_real_io_instrumentation_reports_calls_and_identity(tmp_path):
    block_size = mmap.PAGESIZE
    source = mmap.mmap(-1, block_size)
    source[:] = bytes(index % 251 for index in range(block_size))
    destination = mmap.mmap(-1, block_size)
    path = str(tmp_path / "block.bin")
    events: list[dict[str, Any]] = []
    identity = {"job_id": 7, "block_id": 3, "direction": "store"}

    store_block(
        path,
        memoryview(source),
        0,
        block_size,
        instrumentation_callback=events.append,
        instrumentation_identity=identity,
    )
    load_block(
        path,
        memoryview(destination),
        0,
        block_size,
        instrumentation_callback=events.append,
        instrumentation_identity={**identity, "direction": "load"},
    )

    assert destination[:] == source[:]
    calls = [event for event in events if event["event"] == "io_call_finish"]
    assert [event["operation"] for event in calls] == ["write", "readv"]
    assert all(event["job_id"] == 7 and event["block_id"] == 3 for event in calls)
    assert all(event["completed_bytes"] == block_size for event in calls)
    assert all(event["success"] is True for event in calls)
