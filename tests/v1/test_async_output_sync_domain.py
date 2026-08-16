# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import logging
import queue

from vllm.v1.executor import multiproc_executor
from vllm.v1.executor.multiproc_executor import WorkerProc
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput


class FakeAsyncOutput(AsyncModelRunnerOutput):
    def __init__(self, domain: object, *, sync_error: bool = False) -> None:
        self._domain = domain
        self.sync_calls = 0
        self.sync_error = sync_error
        self.materialized_with_sync = False
        self.materialized_without_sync = False

    @property
    def synchronization_domain(self) -> object:
        return self._domain

    def synchronize(self) -> None:
        if self.sync_error:
            raise RuntimeError("synchronize failed")
        self.sync_calls += 1

    def get_output(self) -> ModelRunnerOutput:
        self.materialized_with_sync = True
        return ModelRunnerOutput(req_ids=[str(id(self))], req_id_to_index={})

    def get_output_without_sync(self) -> ModelRunnerOutput:
        self.materialized_without_sync = True
        return ModelRunnerOutput(req_ids=[str(id(self))], req_id_to_index={})


class FakeResponseQueue:
    def __init__(self) -> None:
        self.responses: list[tuple] = []

    def enqueue(self, item) -> None:
        self.responses.append(item)


def make_workerproc() -> WorkerProc:
    proc = WorkerProc.__new__(WorkerProc)
    proc.rank = 0
    proc.worker_response_mq = FakeResponseQueue()
    proc.host_batching_stats_path = None
    proc.host_batching_guard = True
    proc.host_batching_max_outputs = 4
    proc.host_batching_max_wait_us = 0
    proc.async_output_queue = queue.Queue()
    proc._host_batching_stats = {
        "guard_enabled": True,
        "max_outputs": 4,
        "max_wait_us": 0,
        "batches": 0,
        "outputs": 0,
        "batched_batches": 0,
        "max_batch_size": 0,
        "max_queue_depth": 0,
        "queue_wait_us": 0.0,
        "sync_calls": 0,
        "sync_calls_saved": 0,
    }
    return proc


def test_batch_synchronization_requires_stream_identity() -> None:
    first_domain = object()
    first = FakeAsyncOutput(first_domain)

    assert first.can_batch_synchronize_with(FakeAsyncOutput(first_domain))
    assert not first.can_batch_synchronize_with(FakeAsyncOutput(object()))


def test_same_domain_batch_syncs_last_event_and_materializes_all() -> None:
    proc = make_workerproc()
    domain = object()
    outputs = [FakeAsyncOutput(domain) for _ in range(3)]

    proc.enqueue_output_batch(outputs)

    # Safety property (1): only the last output's event is synchronized.
    assert [output.sync_calls for output in outputs] == [0, 0, 1]
    # All outputs are materialized without an additional wait.
    assert all(output.materialized_without_sync for output in outputs)
    assert all(not output.materialized_with_sync for output in outputs)
    # Every output is delivered to the response queue, none lost.
    assert len(proc.worker_response_mq.responses) == len(outputs)
    # Bookkeeping: one batch, sync calls saved for all but the last output.
    assert proc._host_batching_stats["batches"] == 1
    assert proc._host_batching_stats["batched_batches"] == 1
    assert proc._host_batching_stats["sync_calls"] == 1
    assert proc._host_batching_stats["sync_calls_saved"] == len(outputs) - 1


def test_mixed_domain_batch_syncs_each_output_individually() -> None:
    proc = make_workerproc()
    outputs = [FakeAsyncOutput(object()) for _ in range(3)]

    proc.enqueue_output_batch(outputs)

    # Safety property (2): per-output synchronization, no batch path taken.
    assert all(output.materialized_with_sync for output in outputs)
    assert all(not output.materialized_without_sync for output in outputs)
    assert len(proc.worker_response_mq.responses) == len(outputs)
    # No batching bookkeeping since the batch guard bailed out.
    assert proc._host_batching_stats["batches"] == 0
    assert proc._host_batching_stats["sync_calls_saved"] == 0


def test_batch_sync_exception_falls_back_to_per_output_sync(caplog) -> None:
    proc = make_workerproc()
    domain = object()
    # Same domain (so batching is attempted), but the last output's `synchronize`
    # raises to exercise the defensive fallback.
    outputs = [FakeAsyncOutput(domain) for _ in range(2)]
    outputs[-1].sync_error = True

    with caplog.at_level(
        logging.CRITICAL, logger=multiproc_executor.__name__
    ):
        proc.enqueue_output_batch(outputs)

    # Safety property (3): every output is synchronized individually via
    # `get_output`, so none is materialized without synchronization and none is lost.
    assert all(output.materialized_with_sync for output in outputs)
    assert all(not output.materialized_without_sync for output in outputs)
    assert len(proc.worker_response_mq.responses) == len(outputs)