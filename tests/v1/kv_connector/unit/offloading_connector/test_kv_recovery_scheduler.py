# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from types import SimpleNamespace

import pytest

import vllm.v1.kv_recovery_profile as recovery_profile
from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
)
from tests.v1.kv_connector.unit.utils import EOS_TOKEN_ID
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    offloading_connector as offloading_connector_module,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
    scheduler as scheduler_module,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.v1.kv_recovery_profile import (
    KV_RECOVERY_PROFILE_BINDING,
    KVRecoveryComputeContext,
    KVRecoveryH2DReceipt,
    KVRecoveryIdentity,
    KVRecoveryLogicalBlock,
    KVRecoveryTransferAttempt,
    KVRecoveryTransferContext,
    KVRecoveryWaitAttempt,
    KVRecoveryWaitMembership,
    canonical_block_set_id,
)
from vllm.v1.outputs import KVConnectorOutput

pytestmark = pytest.mark.cpu_test


class RecordingSchedulerObserver:
    def __init__(self):
        self.logical_ids: dict[tuple[str, int, int], str] = {}
        self.contexts: list[KVRecoveryTransferContext] = []
        self.receipt_batches: list[tuple[tuple[KVRecoveryH2DReceipt, ...], bool]] = []
        self.preemptions: list[tuple[str, int]] = []
        self.admission_starts: list[tuple[str, int]] = []
        self.admissions: list[tuple[str, int, str]] = []
        self.requeues: list[tuple[str, int, str]] = []
        self.terminals: list[str] = []
        self.runtime_events: list[tuple[str, str, int | None]] = []
        self.receipt_ready_request_ids: set[str] = set()
        self.reset_thresholds: list[int] = []
        self.closed = False

    def request_preempted(self, runtime_request_id, recovery_epoch):
        self.preemptions.append((runtime_request_id, recovery_epoch))
        self.runtime_events.append(("preempted", runtime_request_id, recovery_epoch))
        return f"{'b' * 32}:k:0"

    def prepare_transfer_context(self, runtime_request_id, operation, coordinates):
        self.runtime_events.append((operation, runtime_request_id, None))
        trace_id = hashlib.sha256(runtime_request_id.encode()).hexdigest()[:32]
        lifecycle_id = f"{trace_id}:e:0"
        is_recovery = operation == "h2d_restore"
        identity = KVRecoveryIdentity(
            run_id="0" * 32,
            trace_id=trace_id,
            engine_lifecycle_id=lifecycle_id,
            runtime_request_id=runtime_request_id,
            recovery_epoch=1 if is_recovery else None,
            episode_id=f"{lifecycle_id}:k:1" if is_recovery else None,
            base_preempted_event_id=(f"{'b' * 32}:e:0" if is_recovery else None),
            preempt_profile_record_id=(f"{'b' * 32}:k:0" if is_recovery else None),
        )
        logical_blocks = tuple(
            KVRecoveryLogicalBlock(
                coordinate.group_index,
                coordinate.logical_ordinal,
                self.logical_ids.setdefault(
                    (
                        runtime_request_id,
                        coordinate.group_index,
                        coordinate.logical_ordinal,
                    ),
                    hashlib.sha256(
                        (
                            f"{runtime_request_id}:"
                            f"{coordinate.group_index}:"
                            f"{coordinate.logical_ordinal}"
                        ).encode()
                    ).hexdigest()[:32],
                ),
            )
            for coordinate in coordinates
        )
        context = KVRecoveryTransferContext(
            binding=KV_RECOVERY_PROFILE_BINDING,
            identity=identity,
            operation=operation,
            block_set_id=canonical_block_set_id(
                KV_RECOVERY_PROFILE_BINDING, identity, logical_blocks
            ),
            logical_blocks=logical_blocks,
        )
        self.contexts.append(context)
        return context

    def consume_h2d_receipts(self, receipts, receipt_capacity_exhausted):
        self.receipt_batches.append((receipts, receipt_capacity_exhausted))
        self.runtime_events.extend(
            ("receipt", receipt.identity.runtime_request_id, None)
            for receipt in receipts
        )
        self.receipt_ready_request_ids.update(
            receipt.identity.runtime_request_id for receipt in receipts
        )

    def request_admission_started(self, runtime_request_id, recovery_epoch):
        if runtime_request_id not in self.receipt_ready_request_ids:
            return
        self.admission_starts.append((runtime_request_id, recovery_epoch))
        self.runtime_events.append(
            ("admission_started", runtime_request_id, recovery_epoch)
        )

    def request_admitted(self, runtime_request_id, recovery_epoch, compute_kind):
        self.admissions.append((runtime_request_id, recovery_epoch, compute_kind))
        self.runtime_events.append(("admitted", runtime_request_id, recovery_epoch))
        receipt = next(
            receipt
            for batch, _truncated in reversed(self.receipt_batches)
            for receipt in batch
            if receipt.identity.runtime_request_id == runtime_request_id
        )
        return KVRecoveryComputeContext(
            binding=receipt.binding,
            identity=receipt.identity,
            transfer_id=receipt.transfer_id,
            block_set_id=receipt.block_set_id,
            bytes_moved=receipt.bytes_moved,
            admission_profile_record_id=f"{'b' * 32}:k:1",
            compute_kind=compute_kind,
            base_phase_start_event_id=f"{'d' * 32}:e:0",
        )

    def request_requeued(self, runtime_request_id, recovery_epoch, reason):
        self.requeues.append((runtime_request_id, recovery_epoch, reason))
        self.runtime_events.append(("requeued", runtime_request_id, recovery_epoch))

    def request_terminal(self, runtime_request_id):
        self.terminals.append(runtime_request_id)
        self.runtime_events.append(("terminal", runtime_request_id, None))

    def reset(self, stale_job_threshold):
        self.reset_thresholds.append(stale_job_threshold)

    def close(self):
        self.closed = True


class RecordingWorkerObserver:
    def __init__(self):
        self.attempts: dict[int, KVRecoveryTransferAttempt] = {}
        self.contexts: list[KVRecoveryTransferContext] = []
        self.transfer_seq = 0
        self.waits: list[KVRecoveryWaitAttempt] = []
        self.first_compute_observations: list[tuple[KVRecoveryComputeContext, int]] = []
        self.missing_first_compute: list[KVRecoveryComputeContext] = []

    def begin_transfer(self, connector_job_id, context):
        attempt = KVRecoveryTransferAttempt(
            connector_job_id=connector_job_id,
            transfer_id=f"{'a' * 32}:t:{self.transfer_seq}",
            context=context,
        )
        self.transfer_seq += 1
        self.attempts[connector_job_id] = attempt
        self.contexts.append(context)
        return attempt

    def transfer_submitted(self, attempt, timestamp_ns):
        return None

    def transfer_not_submitted(self, attempt):
        self.attempts.pop(attempt.connector_job_id, None)

    def transfer_completed(
        self,
        connector_job_id,
        timestamp_ns,
        success,
        bytes_moved,
        device_duration_ns,
    ):
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
        transfer_ids = tuple(
            sorted(self.attempts[job_id].transfer_id for job_id in connector_job_ids)
        )
        return KVRecoveryWaitMembership(transfer_ids)

    def wait_completed(self, attempt):
        self.waits.append(attempt)

    def h2d_receipt_capacity_exhausted(self, receipt, loss_reason):
        raise AssertionError("receipt capacity unexpectedly exhausted")

    def first_compute(self, context, timestamp_ns):
        self.first_compute_observations.append((context, timestamp_ns))

    def first_compute_not_observed(self, context):
        self.missing_first_compute.append(context)

    def close(self):
        return None


class RecordingFactory:
    def __init__(self):
        self.scheduler = RecordingSchedulerObserver()
        self.worker = RecordingWorkerObserver()

    def reinitialize_after_fork(self, binding):
        assert binding == KV_RECOVERY_PROFILE_BINDING

    def create_scheduler_observer(self, binding):
        assert binding == KV_RECOVERY_PROFILE_BINDING
        return self.scheduler

    def create_worker_observer(self, binding):
        assert binding == KV_RECOVERY_PROFILE_BINDING
        return self.worker


def enable_test_observer_factory(
    monkeypatch: pytest.MonkeyPatch,
    factory: RecordingFactory,
) -> None:
    monkeypatch.setattr(recovery_profile, "_activation_gate_open", lambda: True)
    monkeypatch.setattr(
        offloading_connector_module,
        "kv_recovery_runtime_scope_authorized",
        lambda *args: True,
    )
    monkeypatch.setattr(recovery_profile, "_observer_factory", factory)


def make_receipt(
    context: KVRecoveryTransferContext,
    connector_job_id: int,
) -> KVRecoveryH2DReceipt:
    return KVRecoveryH2DReceipt(
        binding=context.binding,
        connector_job_id=connector_job_id,
        transfer_id=f"{'a' * 32}:t:{connector_job_id}",
        identity=context.identity,
        block_set_id=context.block_set_id,
        process_uuid="a" * 32,
        rank=0,
        world_size=1,
        clock_domain_id="c" * 32,
        communication_done_event_id=f"{'a' * 32}:e:{connector_job_id}",
        restore_done_profile_record_id=f"{'a' * 32}:k:{connector_job_id}",
        timestamp_ns=10,
        bytes_moved=128,
    )


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_pressure_preemption_preserves_identity_through_real_connector_path(
    request_runner,
    monkeypatch: pytest.MonkeyPatch,
    async_scheduling: bool,
):
    factory = RecordingFactory()
    enable_test_observer_factory(monkeypatch, factory)

    block_size = 4
    block_size_factor = 3
    offloaded_block_size = block_size * block_size_factor
    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=100,
        async_scheduling=async_scheduling,
        block_size_factor=block_size_factor,
    )

    original_complete_transfers = runner.offloading_spec.complete_transfers

    def complete_transfers_with_measurements():
        original_complete_transfers()
        for result in runner.offloading_spec.handler.completed_transfers:
            result.transfer_size = 128
            result.transfer_time = 0.25

    runner.offloading_spec.complete_transfers = complete_transfers_with_measurements

    free_blocks = runner.scheduler.kv_cache_manager.block_pool.free_block_queue
    initial_free_blocks = free_blocks.num_free_blocks
    runner.new_request(token_ids=[0] * offloaded_block_size * 2)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(decoded_tokens=[0], complete_transfers=False)

    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[0] * (2 * offloaded_block_size - block_size),
        complete_transfers=False,
    )

    free_blocks.num_free_blocks = 0
    runner.run(
        decoded_tokens=[],
        complete_transfers=False,
        expected_flushed=tuple(range(9)),
        expected_stored=tuple(range(9)),
    )

    free_blocks.num_free_blocks = initial_free_blocks
    runner.scheduler.reset_prefix_cache()
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, context: 3
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[0] * block_size,
        expected_loaded=tuple(range(9)),
    )

    h2d_contexts = [
        context
        for context in factory.scheduler.contexts
        if context.operation == "h2d_restore"
    ]
    assert len(h2d_contexts) == 1
    assert h2d_contexts[0].coordinates == tuple(
        recovery_profile.KVRecoveryBlockCoordinate(0, ordinal) for ordinal in range(3)
    )
    assert h2d_contexts[0] in factory.worker.contexts
    d2h_blocks_by_coordinate = {
        block.coordinate: block.logical_block_id
        for context in factory.scheduler.contexts
        if context.operation == "d2h_preserve"
        for block in context.logical_blocks
    }
    assert {
        block.coordinate: block.logical_block_id
        for block in h2d_contexts[0].logical_blocks
    }.items() <= d2h_blocks_by_coordinate.items()

    receipts = [
        receipt
        for batch, truncated in factory.scheduler.receipt_batches
        if not truncated
        for receipt in batch
    ]
    assert len(receipts) == 1
    assert receipts[0].identity == h2d_contexts[0].identity
    assert receipts[0].block_set_id == h2d_contexts[0].block_set_id
    runtime_request_id = h2d_contexts[0].identity.runtime_request_id
    assert factory.scheduler.preemptions == [(runtime_request_id, 1)]
    assert factory.scheduler.admission_starts == [(runtime_request_id, 1)]
    assert factory.scheduler.admissions == [(runtime_request_id, 1, "decode")]
    connector_worker = runner.worker_connector.connector_worker
    assert connector_worker is not None
    compute_contexts = connector_worker._kv_recovery_compute_contexts
    assert compute_contexts is not None
    assert compute_contexts == {}
    assert [
        context.identity.runtime_request_id
        for context in factory.worker.missing_first_compute
    ] == [runtime_request_id]
    event_names = [
        event_name
        for event_name, request_id, _ in factory.scheduler.runtime_events
        if request_id == runtime_request_id
    ]
    assert event_names.index("preempted") < event_names.index("h2d_restore")
    assert event_names.index("h2d_restore") < event_names.index("receipt")
    assert event_names.index("receipt") < event_names.index("admission_started")
    assert event_names.index("admission_started") < event_names.index("admitted")


def test_scheduler_observer_failure_is_fail_open(
    request_runner,
    monkeypatch: pytest.MonkeyPatch,
):
    class RaisingSchedulerObserver(RecordingSchedulerObserver):
        def prepare_transfer_context(self, *args, **kwargs):
            raise RuntimeError("observer failed")

    factory = RecordingFactory()
    factory.scheduler = RaisingSchedulerObserver()
    enable_test_observer_factory(monkeypatch, factory)
    runner = request_runner(
        block_size=4,
        num_gpu_blocks=100,
        async_scheduling=False,
    )

    runner.new_request(token_ids=[0] * 4)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=(0,))

    assert factory.worker.contexts == []
    assert factory.scheduler.terminals == ["0"]


def test_post_wakeup_requeue_requires_matching_ready_epoch():
    observer = RecordingSchedulerObserver()
    scheduler = object.__new__(scheduler_module.OffloadingConnectorScheduler)
    scheduler._kv_recovery_observer = observer
    scheduler._kv_recovery_receipt_ready = {"request-0": 1}
    scheduler._req_status = {
        "request-0": SimpleNamespace(
            req=SimpleNamespace(num_preemptions=1),
        )
    }

    scheduler.observe_kv_recovery_requeue("request-0", "block_capacity")
    scheduler._req_status["request-0"].req.num_preemptions = 2
    scheduler.observe_kv_recovery_requeue("request-0", "token_budget")

    assert observer.requeues == [("request-0", 1, "block_capacity")]


def test_malformed_scheduler_observer_return_is_fail_open(
    request_runner,
    monkeypatch: pytest.MonkeyPatch,
):
    class MalformedSchedulerObserver(RecordingSchedulerObserver):
        def prepare_transfer_context(self, *args, **kwargs):
            return object()

    factory = RecordingFactory()
    factory.scheduler = MalformedSchedulerObserver()
    enable_test_observer_factory(monkeypatch, factory)
    runner = request_runner(
        block_size=4,
        num_gpu_blocks=100,
        async_scheduling=False,
    )
    runner.new_request(token_ids=[0] * 4)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )

    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=(0,))

    assert factory.worker.contexts == []


@pytest.mark.parametrize("operation", ["h2d_restore", "d2h_preserve"])
def test_recovery_coordinate_capture_is_bounded_by_one_overflow_sentinel(
    operation: str,
):
    observed_lengths: list[int] = []

    class OverboundObserver(RecordingSchedulerObserver):
        def prepare_transfer_context(
            self,
            runtime_request_id,
            observed_operation,
            coordinates,
        ):
            assert observed_operation == operation
            observed_lengths.append(len(coordinates))
            return None

    coordinates = []
    for ordinal in range(recovery_profile.MAX_LOGICAL_BLOCKS_PER_SET + 100):
        scheduler_module.OffloadingConnectorScheduler._append_kv_recovery_coordinate(
            coordinates,
            0,
            ordinal,
        )

    assert len(coordinates) == recovery_profile.MAX_LOGICAL_BLOCKS_PER_SET + 1
    scheduler = object.__new__(scheduler_module.OffloadingConnectorScheduler)
    scheduler._kv_recovery_observer = OverboundObserver()
    assert (
        scheduler._prepare_kv_recovery_context(
            "request-0",
            operation,  # type: ignore[arg-type]
            coordinates,
        )
        is None
    )
    assert observed_lengths == [recovery_profile.MAX_LOGICAL_BLOCKS_PER_SET + 1]


def test_disabled_scheduler_never_constructs_recovery_coordinates(
    request_runner,
    monkeypatch: pytest.MonkeyPatch,
):
    def unexpected_coordinate(*args, **kwargs):
        raise AssertionError("hard-off scheduler constructed recovery identity")

    monkeypatch.setattr(
        scheduler_module,
        "KVRecoveryBlockCoordinate",
        unexpected_coordinate,
    )
    runner = request_runner(
        block_size=4,
        num_gpu_blocks=100,
        async_scheduling=False,
    )
    runner.new_request(token_ids=[0] * 4)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )

    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=(0,))


def test_reset_notifies_observer_and_filters_stale_or_unmatched_receipts(
    request_runner,
    monkeypatch: pytest.MonkeyPatch,
):
    factory = RecordingFactory()
    enable_test_observer_factory(monkeypatch, factory)
    runner = request_runner(
        block_size=4,
        num_gpu_blocks=100,
        async_scheduling=False,
    )
    scheduler = runner.connector_scheduler
    context = factory.scheduler.prepare_transfer_context(
        "request-0",
        "h2d_restore",
        (recovery_profile.KVRecoveryBlockCoordinate(0, 0),),
    )

    scheduler._stale_job_threshold = 10
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                kv_recovery_h2d_receipts=(make_receipt(context, 9),),
            )
        )
    )
    assert factory.scheduler.receipt_batches == []

    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                kv_recovery_h2d_receipts=(make_receipt(context, 10),),
            )
        )
    )
    assert factory.scheduler.receipt_batches == [((), True)]

    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                kv_recovery_h2d_receipts=(object(),),  # type: ignore[arg-type]
            )
        )
    )
    assert factory.scheduler.receipt_batches == [((), True), ((), True)]

    scheduler.reset_cache()
    assert factory.scheduler.reset_thresholds == [scheduler._stale_job_threshold]

    scheduler.shutdown()
    assert factory.scheduler.closed
