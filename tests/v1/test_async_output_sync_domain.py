# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import logging
import queue

from vllm.v1.executor import multiproc_executor
from vllm.v1.executor.multiproc_executor import WorkerProc
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput


class FakeAsyncOutput(AsyncModelRunnerOutput):
    """In-memory stand-in mirroring the production ``AsyncOutput`` contract:

    ``get_output()`` = ``synchronize()`` then ``get_output_without_sync()``, so
    a failed ``synchronize()`` propagates out of ``get_output()`` exactly as it
    does for the real async outputs this PR changed.

    Failure model:

    - ``sync_failures=N`` fails the first ``N`` synchronize() calls and then
      succeeds (used to simulate a transient batch-wait failure that a
      per-output retry recovers from).
    - ``persistent_sync_error=True`` always raises (used to prove fail-closed
      behavior when the readiness event can never be awaited).
    """

    def __init__(
        self,
        domain: object,
        *,
        sync_failures: int = 0,
        persistent_sync_error: bool = False,
    ) -> None:
        self._domain = domain
        self._req_id = f"req-{id(self)}"
        self.sync_calls = 0
        self.sync_failures = sync_failures
        self.persistent_sync_error = persistent_sync_error
        # Observability mirrors the two production materialization paths:
        #  - materialized_with_sync: materialized via get_output() (which issued
        #    / waited on its own synchronize()).
        #  - materialized_without_sync: materialized via get_output_without_sync()
        #    after the synchronization domain was already synchronized (batched
        #    path, or inside get_output()).
        self.materialized_with_sync = False
        self.materialized_without_sync = False

    @property
    def synchronization_domain(self) -> object:
        return self._domain

    def synchronize(self) -> None:
        self.sync_calls += 1
        if self.persistent_sync_error or self.sync_calls <= self.sync_failures:
            raise RuntimeError("synchronize failed")

    def get_output(self) -> ModelRunnerOutput:
        # Faithful to production AsyncOutput.get_output(): wait, then materialize.
        self.synchronize()
        self.materialized_with_sync = True
        return self.get_output_without_sync()

    def get_output_without_sync(self) -> ModelRunnerOutput:
        self.materialized_without_sync = True
        return ModelRunnerOutput(req_ids=[self._req_id], req_id_to_index={})


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
    # All outputs are materialized without an additional per-output wait.
    assert all(output.materialized_without_sync for output in outputs)
    assert all(not output.materialized_with_sync for output in outputs)
    # Every output is delivered to the response queue, none lost.
    assert len(proc.worker_response_mq.responses) == len(outputs)
    assert all(
        status is WorkerProc.ResponseStatus.SUCCESS
        for status, _ in proc.worker_response_mq.responses
    )
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
    assert [output.sync_calls for output in outputs] == [1, 1, 1]
    assert len(proc.worker_response_mq.responses) == len(outputs)
    assert all(
        status is WorkerProc.ResponseStatus.SUCCESS
        for status, _ in proc.worker_response_mq.responses
    )
    # No batching bookkeeping since the batch guard bailed out.
    assert proc._host_batching_stats["batches"] == 0
    assert proc._host_batching_stats["sync_calls_saved"] == 0


def test_batch_sync_transient_failure_recovers_per_output(caplog) -> None:
    """A batch-wait failure that a per-output retry can recover from must
    still deliver every output successfully."""
    proc = make_workerproc()
    domain = object()
    # Same domain (so batching is attempted), but the last output fails the
    # batch-path synchronize() once and succeeds when retried per-output.
    outputs = [FakeAsyncOutput(domain) for _ in range(2)]
    outputs[-1].sync_failures = 1

    with caplog.at_level(logging.CRITICAL, logger=multiproc_executor.__name__):
        proc.enqueue_output_batch(outputs)

    # The batch-path synchronize() fails on output[-1] once, then the per-output
    # fallback re-synchronizes it successfully: sync calls are [output0: 1, out1: 2].
    assert [output.sync_calls for output in outputs] == [1, 2]
    assert all(output.materialized_with_sync for output in outputs)
    assert all(output.materialized_without_sync for output in outputs)
    assert len(proc.worker_response_mq.responses) == len(outputs)
    assert all(
        status is WorkerProc.ResponseStatus.SUCCESS
        for status, _ in proc.worker_response_mq.responses
    )
    # Each response carries the correct output payload, in order.
    assert [payload.req_ids[0] for _, payload in proc.worker_response_mq.responses] == [
        output._req_id for output in outputs
    ]


def test_batch_sync_persistent_error_falls_back_closed(caplog) -> None:
    """A readiness event that can never be awaited must fail closed: the
    affected output is surfaced as a FAILURE response, never silently
    materialized with unsynchronized data."""
    proc = make_workerproc()
    domain = object()
    # Same domain (so batching is attempted); the last output's synchronize()
    # raises persistently. The fallback re-synchronizes each output via
    # get_output(), which raises again for the broken output.
    outputs = [
        FakeAsyncOutput(domain),
        FakeAsyncOutput(domain, persistent_sync_error=True),
    ]

    with caplog.at_level(logging.CRITICAL, logger=multiproc_executor.__name__):
        proc.enqueue_output_batch(outputs)

    statuses = [status for status, _ in proc.worker_response_mq.responses]
    assert len(statuses) == len(outputs)
    assert statuses[0] is WorkerProc.ResponseStatus.SUCCESS
    assert statuses[1] is WorkerProc.ResponseStatus.FAILURE
    # The healthy output is materialized; the broken output must NOT be
    # materialized as if it had synchronized data.
    assert outputs[0].materialized_with_sync is True
    assert outputs[1].materialized_with_sync is False
    assert outputs[1].materialized_without_sync is False
