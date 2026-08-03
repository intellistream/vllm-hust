# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle
from collections.abc import Iterable

import pytest

import vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker as worker_module
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    TransferResult,
)
from vllm.v1.kv_recovery_profile import (
    KV_RECOVERY_PROFILE_BINDING,
    MAX_PENDING_H2D_CONTEXTS_PER_PROCESS,
    MAX_TRANSFER_IDS_PER_WAIT_SET,
    BoundedKVRecoveryWorkerObserver,
    KVRecoveryH2DReceipt,
    KVRecoveryIdentity,
    KVRecoveryLogicalBlock,
    KVRecoveryTransferAttempt,
    KVRecoveryTransferContext,
    KVRecoveryWaitAttempt,
    KVRecoveryWaitMembership,
    canonical_block_set_id,
)

pytestmark = pytest.mark.cpu_test


class CPULoadStoreSpec(LoadStoreSpec):
    @staticmethod
    def medium() -> str:
        return "CPU"


class RecordingBackend:
    def __init__(self, events: list[tuple]):
        self.events = events
        self.finished: list[TransferResult] = []
        self.submit_load_result = True
        self.submit_store_result = True
        self.wait_error: Exception | None = None
        self.shutdown_error: Exception | None = None
        self.shutdown_called = False

    def submit_load(self, job_id, src_spec, dst_spec):
        self.events.append(("backend_submit_load", job_id, src_spec, dst_spec))
        return self.submit_load_result

    def submit_store(self, job_id, src_spec, dst_spec):
        self.events.append(("backend_submit_store", job_id, src_spec, dst_spec))
        return self.submit_store_result

    def get_finished(self):
        finished = self.finished
        self.finished = []
        return finished

    def wait(self, job_ids):
        self.events.append(("backend_wait", frozenset(job_ids)))
        if self.wait_error is not None:
            raise self.wait_error

    def shutdown(self):
        self.shutdown_called = True
        if self.shutdown_error is not None:
            raise self.shutdown_error


class RecordingObserver:
    def __init__(self, events: list[tuple], raise_at: str | None = None):
        self.events = events
        self.raise_at = raise_at
        self.attempts: dict[int, KVRecoveryTransferAttempt] = {}
        self.transfer_seq = 0
        self.wait_completed_calls: list[KVRecoveryWaitAttempt] = []
        self.receipt_truncations: list[KVRecoveryH2DReceipt] = []
        self.closed = False

    def _raise_if_requested(self, stage: str):
        if self.raise_at == stage:
            raise RuntimeError(stage)

    def begin_transfer(self, connector_job_id, context):
        self._raise_if_requested("begin_transfer")
        self.events.append(("observer_begin", connector_job_id, context))
        attempt = KVRecoveryTransferAttempt(
            connector_job_id=connector_job_id,
            transfer_id=f"{'a' * 32}:t:{self.transfer_seq}",
            context=context,
        )
        self.transfer_seq += 1
        self.attempts[connector_job_id] = attempt
        return attempt

    def transfer_submitted(self, attempt, timestamp_ns):
        self._raise_if_requested("transfer_submitted")
        self.events.append(("observer_submitted", attempt.connector_job_id))

    def transfer_not_submitted(self, attempt):
        self._raise_if_requested("transfer_not_submitted")
        self.attempts.pop(attempt.connector_job_id, None)
        self.events.append(("observer_not_submitted", attempt.connector_job_id))

    def transfer_completed(
        self,
        connector_job_id,
        timestamp_ns,
        success,
        bytes_moved,
        device_duration_ns,
    ):
        self._raise_if_requested("transfer_completed")
        self.events.append(
            (
                "observer_completed",
                connector_job_id,
                success,
                bytes_moved,
                device_duration_ns,
            )
        )
        attempt = self.attempts.pop(connector_job_id, None)
        if (
            attempt is None
            or attempt.context.operation != "h2d_restore"
            or not success
            or bytes_moved is None
            or bytes_moved <= 0
        ):
            return None
        return KVRecoveryH2DReceipt(
            binding=attempt.context.binding,
            connector_job_id=connector_job_id,
            transfer_id=attempt.transfer_id,
            identity=attempt.context.identity,
            block_set_id=attempt.context.block_set_id,
            process_uuid="a" * 32,
            rank=0,
            world_size=1,
            clock_domain_id="c" * 32,
            communication_done_event_id=(f"{'a' * 32}:e:{connector_job_id}"),
            restore_done_profile_record_id=(f"{'a' * 32}:k:{connector_job_id}"),
            timestamp_ns=timestamp_ns,
            bytes_moved=bytes_moved,
        )

    def prepare_wait(self, connector_job_ids):
        self._raise_if_requested("prepare_wait")
        self.events.append(("observer_prepare_wait", frozenset(connector_job_ids)))
        transfer_ids = tuple(
            sorted(self.attempts[job_id].transfer_id for job_id in connector_job_ids)
        )
        return KVRecoveryWaitMembership(transfer_ids)

    def wait_completed(self, attempt):
        self._raise_if_requested("wait_completed")
        self.events.append(("observer_wait_completed", attempt.transfer_ids))
        self.wait_completed_calls.append(attempt)

    def h2d_receipt_truncated(self, receipt):
        self.receipt_truncations.append(receipt)

    def close(self):
        self._raise_if_requested("close")
        self.closed = True


class RecordingEvidenceSink:
    def __init__(self):
        self.submissions: list[tuple[KVRecoveryTransferAttempt, int]] = []
        self.not_submitted: list[KVRecoveryTransferAttempt] = []
        self.completions: list[int] = []
        self.capacity_losses: list[tuple[KVRecoveryTransferAttempt, int, str]] = []
        self.failures: list[tuple[str, tuple[int, ...]]] = []
        self.waits: list[KVRecoveryWaitAttempt] = []
        self.truncated: list[KVRecoveryH2DReceipt] = []
        self.close_calls: list[tuple[tuple[KVRecoveryTransferAttempt, ...], bool]] = []

    def transfer_submitted(self, attempt, timestamp_ns):
        self.submissions.append((attempt, timestamp_ns))

    def transfer_not_submitted(self, attempt):
        self.not_submitted.append(attempt)

    def transfer_completed(
        self,
        attempt,
        submit_timestamp_ns,
        timestamp_ns,
        success,
        bytes_moved,
        device_duration_ns,
    ):
        self.completions.append(attempt.connector_job_id)
        if (
            attempt.context.operation != "h2d_restore"
            or not success
            or bytes_moved is None
            or bytes_moved <= 0
        ):
            return None
        return KVRecoveryH2DReceipt(
            binding=attempt.context.binding,
            connector_job_id=attempt.connector_job_id,
            transfer_id=attempt.transfer_id,
            identity=attempt.context.identity,
            block_set_id=attempt.context.block_set_id,
            process_uuid="a" * 32,
            rank=0,
            world_size=1,
            clock_domain_id="c" * 32,
            communication_done_event_id=(f"{'a' * 32}:e:{attempt.connector_job_id}"),
            restore_done_profile_record_id=(f"{'a' * 32}:k:{attempt.connector_job_id}"),
            timestamp_ns=timestamp_ns,
            bytes_moved=bytes_moved,
        )

    def pending_h2d_capacity_exhausted(self, attempt, timestamp_ns, loss_reason):
        self.capacity_losses.append((attempt, timestamp_ns, loss_reason))

    def evidence_failure(
        self,
        reason,
        connector_job_ids,
        transfer_ids,
        timestamp_ns,
    ):
        self.failures.append((reason, connector_job_ids))

    def wait_completed(self, attempt):
        self.waits.append(attempt)

    def h2d_receipt_truncated(self, receipt):
        self.truncated.append(receipt)

    def close(self, open_attempts, evidence_disabled):
        self.close_calls.append((open_attempts, evidence_disabled))


def make_context(operation: str) -> KVRecoveryTransferContext:
    trace_id = "1" * 32
    lifecycle_id = f"{trace_id}:e:0"
    is_recovery = operation == "h2d_restore"
    identity = KVRecoveryIdentity(
        run_id="0" * 32,
        trace_id=trace_id,
        engine_lifecycle_id=lifecycle_id,
        runtime_request_id="request-0",
        recovery_epoch=1 if is_recovery else None,
        episode_id=f"{lifecycle_id}:k:1" if is_recovery else None,
        base_preempted_event_id=f"{'b' * 32}:e:0" if is_recovery else None,
    )
    logical_blocks = (KVRecoveryLogicalBlock(0, 0, "b" * 32),)
    return KVRecoveryTransferContext(
        binding=KV_RECOVERY_PROFILE_BINDING,
        identity=identity,
        operation=operation,  # type: ignore[arg-type]
        block_set_id=canonical_block_set_id(
            KV_RECOVERY_PROFILE_BINDING, identity, logical_blocks
        ),
        logical_blocks=logical_blocks,
    )


def make_load_job() -> TransferJob:
    return TransferJob(
        req_id="request-0",
        src_spec=CPULoadStoreSpec(),
        dst_spec=GPULoadStoreSpec([3], group_sizes=[1], block_indices=[0]),
    )


def make_store_job() -> TransferJob:
    return TransferJob(
        req_id="request-0",
        src_spec=GPULoadStoreSpec([3], group_sizes=[1], block_indices=[0]),
        dst_spec=CPULoadStoreSpec(),
    )


def make_worker(observer=None):
    events: list[tuple] = []
    backend = RecordingBackend(events)
    worker = OffloadingConnectorWorker(
        spec=object(),  # type: ignore[arg-type]
        kv_recovery_observer=observer,
    )
    worker.worker = backend  # type: ignore[assignment]
    return worker, backend, events


def metadata(
    *,
    load_jobs: Iterable[tuple[int, TransferJob]] = (),
    store_jobs: Iterable[tuple[int, TransferJob]] = (),
    jobs_to_flush: set[int] | None = None,
    kv_recovery_contexts: Iterable[tuple[int, KVRecoveryTransferContext]] = (),
) -> OffloadingConnectorMetadata:
    contexts = dict(kv_recovery_contexts)
    return OffloadingConnectorMetadata(
        load_jobs=dict(load_jobs),
        store_jobs=dict(store_jobs),
        jobs_to_flush=jobs_to_flush,
        kv_recovery_contexts=contexts or None,
    )


def test_transfer_job_remains_backward_compatible():
    src_spec = CPULoadStoreSpec()
    dst_spec = CPULoadStoreSpec()

    job = TransferJob("request-0", src_spec, dst_spec)

    assert job.req_id == "request-0"
    assert job.src_spec is src_spec
    assert job.dst_spec is dst_spec
    assert not hasattr(job, "kv_recovery_context")


def test_connector_metadata_sidecar_round_trips_and_defaults_off():
    job = make_load_job()
    context = make_context("h2d_restore")
    disabled = metadata(load_jobs=((7, job),))
    active = metadata(
        load_jobs=((7, job),),
        kv_recovery_contexts=((7, context),),
    )

    disabled_copy = pickle.loads(pickle.dumps(disabled))
    active_copy = pickle.loads(pickle.dumps(active))

    assert disabled_copy.kv_recovery_contexts is None
    assert active_copy.kv_recovery_contexts == {7: context}


def test_disabled_worker_path_preserves_load_semantics():
    worker, backend, events = make_worker()
    context = make_context("h2d_restore")
    job = make_load_job()

    worker.start_kv_transfers(
        metadata(
            load_jobs=((7, job),),
            kv_recovery_contexts=((7, context),),
        )
    )
    backend.finished.append(TransferResult(7, True, 128, 0.25))
    finished_sending, finished_recving = worker.get_finished(set())
    worker_meta = worker.build_connector_worker_meta()

    assert events == [("backend_submit_load", 7, job.src_spec, job.dst_spec)]
    assert finished_sending == set()
    assert finished_recving == {"request-0"}
    assert worker_meta is not None
    assert worker_meta.completed_jobs == {7: 1}
    assert worker_meta.transfer_stats.load.bytes == 128
    assert worker_meta.kv_recovery_h2d_receipts == ()
    assert not worker_meta.kv_recovery_h2d_receipts_truncated


def test_disabled_worker_path_does_not_read_profile_clock(
    monkeypatch: pytest.MonkeyPatch,
):
    worker, backend, _ = make_worker()
    context = make_context("h2d_restore")
    job = make_load_job()

    def unexpected_clock_read():
        raise AssertionError("disabled KV-recovery path read the clock")

    monkeypatch.setattr(worker_module.time, "monotonic_ns", unexpected_clock_read)
    worker.start_kv_transfers(
        metadata(
            load_jobs=((7, job),),
            kv_recovery_contexts=((7, context),),
        )
    )
    backend.finished.append(TransferResult(7, True, 128, 0.25))

    assert worker.get_finished(set()) == (set(), {"request-0"})


def test_submit_clock_failure_keeps_serving_but_cannot_emit_receipt(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[tuple] = []
    observer = RecordingObserver(events)
    worker, backend, _ = make_worker(observer)
    backend.events = events

    def unavailable_clock():
        raise RuntimeError("clock unavailable")

    monkeypatch.setattr(worker_module.time, "monotonic_ns", unavailable_clock)
    context = make_context("h2d_restore")
    job = make_load_job()
    worker.start_kv_transfers(
        metadata(
            load_jobs=((7, job),),
            kv_recovery_contexts=((7, context),),
        )
    )
    backend.finished.append(TransferResult(7, True, 128, 0.25))

    assert worker.get_finished(set()) == (set(), {"request-0"})
    worker_meta = worker.build_connector_worker_meta()
    assert worker_meta is not None
    assert worker_meta.completed_jobs == {7: 1}
    assert worker_meta.kv_recovery_h2d_receipts == ()
    assert not any(event[0] == "observer_completed" for event in events)


def test_h2d_sidecar_reaches_submit_completion_and_scheduler_receipt():
    events: list[tuple] = []
    observer = RecordingObserver(events)
    worker, backend, _ = make_worker(observer)
    backend.events = events
    context = make_context("h2d_restore")
    job = make_load_job()

    worker.start_kv_transfers(
        metadata(
            load_jobs=((7, job),),
            kv_recovery_contexts=((7, context),),
        )
    )
    backend.finished.append(TransferResult(7, True, 128, 0.25))
    _, finished_recving = worker.get_finished(set())
    worker_meta = worker.build_connector_worker_meta()

    assert [event[0] for event in events[:3]] == [
        "observer_begin",
        "backend_submit_load",
        "observer_submitted",
    ]
    assert finished_recving == {"request-0"}
    assert worker_meta is not None
    assert worker_meta.completed_jobs == {7: 1}
    assert len(worker_meta.kv_recovery_h2d_receipts) == 1
    receipt = worker_meta.kv_recovery_h2d_receipts[0]
    assert receipt.connector_job_id == 7
    assert receipt.identity == context.identity
    assert receipt.block_set_id == context.block_set_id
    assert events[-1] == ("observer_completed", 7, True, 128, 250_000_000)


def test_submit_and_completion_clocks_are_first_profile_observations(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[tuple] = []
    observer = RecordingObserver(events)
    worker, backend, _ = make_worker(observer)
    backend.events = events
    original_get_finished = backend.get_finished

    def observed_get_finished():
        events.append(("backend_get_finished",))
        return original_get_finished()

    timestamps = iter((10, 20))

    def observed_clock():
        timestamp_ns = next(timestamps)
        events.append(("profile_clock", timestamp_ns))
        return timestamp_ns

    backend.get_finished = observed_get_finished  # type: ignore[method-assign]
    monkeypatch.setattr(worker_module.time, "monotonic_ns", observed_clock)
    context = make_context("h2d_restore")
    job = make_load_job()

    worker.start_kv_transfers(
        metadata(
            load_jobs=((7, job),),
            kv_recovery_contexts=((7, context),),
        )
    )
    backend.finished.append(TransferResult(7, True, 128, 0.25))
    worker.get_finished(set())

    assert [event[0] for event in events] == [
        "observer_begin",
        "backend_submit_load",
        "profile_clock",
        "observer_submitted",
        "backend_get_finished",
        "profile_clock",
        "observer_completed",
    ]
    assert observer.wait_completed_calls == []


def test_deferred_store_keeps_full_job_and_submits_before_load():
    events: list[tuple] = []
    observer = RecordingObserver(events)
    worker, backend, _ = make_worker(observer)
    backend.events = events
    store_context = make_context("d2h_preserve")
    load_context = make_context("h2d_restore")
    store_job = make_store_job()
    load_job = make_load_job()

    worker.prepare_store_kv(
        metadata(
            store_jobs=((3, store_job),),
            kv_recovery_contexts=((3, store_context),),
        )
    )
    assert worker._unsubmitted_store_jobs == [
        (3, store_job.src_spec, store_job.dst_spec)
    ]
    assert events == []

    worker.start_kv_transfers(
        metadata(
            load_jobs=((7, load_job),),
            kv_recovery_contexts=((7, load_context),),
        )
    )
    worker.start_kv_transfers(metadata())

    backend_calls = [event for event in events if event[0].startswith("backend_")]
    assert backend_calls == [
        ("backend_submit_store", 3, store_job.src_spec, store_job.dst_spec),
        ("backend_submit_load", 7, load_job.src_spec, load_job.dst_spec),
    ]
    assert [event[2] for event in events if event[0] == "observer_begin"] == [
        store_context,
        load_context,
    ]


@pytest.mark.parametrize(
    "raise_at",
    ["begin_transfer", "transfer_submitted", "transfer_completed"],
)
def test_observer_transfer_failure_does_not_change_serving(raise_at: str):
    events: list[tuple] = []
    observer = RecordingObserver(events, raise_at=raise_at)
    worker, backend, _ = make_worker(observer)
    backend.events = events
    context = make_context("h2d_restore")
    job = make_load_job()

    worker.start_kv_transfers(
        metadata(
            load_jobs=((7, job),),
            kv_recovery_contexts=((7, context),),
        )
    )
    backend.finished.append(TransferResult(7, True, 128, 0.25))
    _, finished_recving = worker.get_finished(set())
    worker_meta = worker.build_connector_worker_meta()

    assert finished_recving == {"request-0"}
    assert worker_meta is not None
    assert worker_meta.completed_jobs == {7: 1}
    assert worker_meta.kv_recovery_h2d_receipts == ()
    assert [event[0] for event in events].count("backend_submit_load") == 1


@pytest.mark.parametrize("malformed_stage", ["begin", "completion"])
def test_malformed_worker_observer_return_does_not_change_serving(
    malformed_stage: str,
):
    class MalformedObserver(RecordingObserver):
        def begin_transfer(self, connector_job_id, context):
            if malformed_stage == "begin":
                return object()
            return super().begin_transfer(connector_job_id, context)

        def transfer_completed(self, *args, **kwargs):
            if malformed_stage == "completion":
                return object()
            return super().transfer_completed(*args, **kwargs)

    events: list[tuple] = []
    observer = MalformedObserver(events)
    worker, backend, _ = make_worker(observer)
    backend.events = events
    context = make_context("h2d_restore")
    job = make_load_job()

    worker.start_kv_transfers(
        metadata(
            load_jobs=((7, job),),
            kv_recovery_contexts=((7, context),),
        )
    )
    backend.finished.append(TransferResult(7, True, 128, 0.25))

    assert worker.get_finished(set()) == (set(), {"request-0"})
    worker_meta = worker.build_connector_worker_meta()
    assert worker_meta is not None
    assert worker_meta.completed_jobs == {7: 1}
    assert worker_meta.kv_recovery_h2d_receipts == ()


def test_backend_submit_rejection_keeps_existing_assertion_semantics():
    events: list[tuple] = []
    observer = RecordingObserver(events)
    worker, backend, _ = make_worker(observer)
    backend.events = events
    backend.submit_load_result = False
    context = make_context("h2d_restore")
    job = make_load_job()

    with pytest.raises(AssertionError):
        worker.start_kv_transfers(
            metadata(
                load_jobs=((7, job),),
                kv_recovery_contexts=((7, context),),
            )
        )

    assert [event[0] for event in events] == [
        "observer_begin",
        "backend_submit_load",
        "observer_not_submitted",
    ]


def test_wait_observer_is_fail_open_but_backend_error_is_preserved():
    events: list[tuple] = []
    observer = RecordingObserver(events, raise_at="prepare_wait")
    worker, backend, _ = make_worker(observer)
    backend.events = events

    worker.handle_preemptions(metadata(jobs_to_flush={3, 4}))
    assert events == [("backend_wait", frozenset({3, 4}))]

    backend.wait_error = RuntimeError("backend wait failed")
    with pytest.raises(RuntimeError, match="backend wait failed"):
        worker.handle_preemptions(metadata(jobs_to_flush={3, 4}))


def test_malformed_wait_membership_does_not_change_backend_wait():
    events: list[tuple] = []
    observer = RecordingObserver(events)
    observer.prepare_wait = lambda connector_job_ids: object()  # type: ignore[method-assign]
    worker, backend, _ = make_worker(observer)
    backend.events = events

    worker.handle_preemptions(metadata(jobs_to_flush={3}))

    assert events == [("backend_wait", frozenset({3}))]
    assert observer.wait_completed_calls == []


def test_overbound_wait_caps_profiler_copy_but_preserves_backend_membership():
    events: list[tuple] = []
    observer = RecordingObserver(events)
    observed_sizes: list[int] = []

    def prepare_overbound_wait(connector_job_ids):
        observed_sizes.append(len(connector_job_ids))
        return None

    observer.prepare_wait = prepare_overbound_wait  # type: ignore[method-assign]
    worker, backend, _ = make_worker(observer)
    backend.events = events
    job_ids = set(range(MAX_TRANSFER_IDS_PER_WAIT_SET + 10_000))

    worker.handle_preemptions(metadata(jobs_to_flush=job_ids))

    assert observed_sizes == [MAX_TRANSFER_IDS_PER_WAIT_SET + 1]
    assert events == [("backend_wait", frozenset(job_ids))]


def test_successful_wait_preparation_is_not_recorded_when_backend_raises():
    events: list[tuple] = []
    observer = RecordingObserver(events)
    worker, backend, _ = make_worker(observer)
    backend.events = events
    observer.begin_transfer(3, make_context("d2h_preserve"))
    backend.wait_error = RuntimeError("backend wait failed")

    with pytest.raises(RuntimeError, match="backend wait failed"):
        worker.handle_preemptions(metadata(jobs_to_flush={3}))

    assert events[-2:] == [
        ("observer_prepare_wait", frozenset({3})),
        ("backend_wait", frozenset({3})),
    ]
    assert observer.wait_completed_calls == []


def test_device_duration_overflow_is_unavailable():
    assert OffloadingConnectorWorker._device_duration_ns(1e308) is None


def test_observer_closes_even_when_backend_shutdown_raises():
    events: list[tuple] = []
    observer = RecordingObserver(events)
    worker, backend, _ = make_worker(observer)
    backend.shutdown_error = RuntimeError("backend shutdown failed")

    with pytest.raises(RuntimeError, match="backend shutdown failed"):
        worker.shutdown()

    assert backend.shutdown_called
    assert observer.closed


def test_successful_wait_is_recorded_only_after_backend_returns():
    events: list[tuple] = []
    observer = RecordingObserver(events)
    worker, backend, _ = make_worker(observer)
    backend.events = events
    context = make_context("d2h_preserve")
    observer.begin_transfer(3, context)

    worker.handle_preemptions(metadata(jobs_to_flush={3}))

    assert events[-3:] == [
        ("observer_prepare_wait", frozenset({3})),
        ("backend_wait", frozenset({3})),
        ("observer_wait_completed", (f"{'a' * 32}:t:0",)),
    ]
    assert len(observer.wait_completed_calls) == 1
    assert observer.wait_completed_calls[0].transfer_ids == (f"{'a' * 32}:t:0",)


def test_4097th_h2d_keeps_serving_and_records_one_capacity_loss():
    sink = RecordingEvidenceSink()
    observer = BoundedKVRecoveryWorkerObserver("a" * 32, "0" * 32, "c" * 32, sink)
    worker, backend, events = make_worker(observer)
    context = make_context("h2d_restore")
    job = make_load_job()
    total_jobs = MAX_PENDING_H2D_CONTEXTS_PER_PROCESS + 1
    jobs = tuple((job_id, job) for job_id in range(total_jobs))
    contexts = tuple((job_id, context) for job_id in range(total_jobs))

    worker.start_kv_transfers(metadata(load_jobs=jobs, kv_recovery_contexts=contexts))

    assert len(events) == total_jobs
    assert all(event[0] == "backend_submit_load" for event in events)
    assert observer.transfer_seq == total_jobs
    assert observer.pending_h2d_count == MAX_PENDING_H2D_CONTEXTS_PER_PROCESS
    assert len(sink.submissions) == MAX_PENDING_H2D_CONTEXTS_PER_PROCESS
    assert len(sink.capacity_losses) == 1
    dropped_attempt, _, loss_reason = sink.capacity_losses[0]
    assert dropped_attempt.connector_job_id == total_jobs - 1
    assert loss_reason == "serialization_failure"
    assert sink.failures == []
    assert observer.evidence_disabled

    backend.finished.append(TransferResult(total_jobs - 1, True, 128, 0.25))
    assert worker.get_finished(set()) == (set(), {"request-0"})
    worker_meta = worker.build_connector_worker_meta()

    assert worker_meta is not None
    assert worker_meta.completed_jobs == {total_jobs - 1: 1}
    assert worker_meta.kv_recovery_h2d_receipts == ()
    assert sink.completions == []
    assert len(sink.capacity_losses) == 1
    assert sink.failures == []

    worker.shutdown()
    observer.close()
    assert len(sink.close_calls) == 1
    open_attempts, evidence_disabled = sink.close_calls[0]
    assert len(open_attempts) == MAX_PENDING_H2D_CONTEXTS_PER_PROCESS
    assert evidence_disabled
