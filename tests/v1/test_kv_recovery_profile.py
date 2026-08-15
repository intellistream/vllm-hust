# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import pickle
import select
import signal
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

import vllm.v1.kv_recovery_profile as recovery_profile
from vllm.v1.kv_recovery_profile import (
    BoundedKVRecoveryWorkerObserver,
    KVRecoveryBlockCoordinate,
    KVRecoveryComputeContext,
    KVRecoveryH2DReceipt,
    KVRecoveryIdentity,
    KVRecoveryLogicalBlock,
    KVRecoveryTransferAttempt,
    KVRecoveryTransferContext,
    KVRecoveryWaitAttempt,
    canonical_block_set_id,
    create_kv_recovery_scheduler_observer,
    create_kv_recovery_worker_observer,
    prepare_kv_recovery_profile_after_fork,
    register_kv_recovery_observer_factory,
    reinitialize_kv_recovery_profile_after_fork,
)

pytestmark = pytest.mark.cpu_test

PROCESS_UUID = "a" * 32
RUN_ID = "0" * 32
CLOCK_DOMAIN_ID = "c" * 32


@pytest.fixture(autouse=True)
def reset_observer_factory(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(recovery_profile, "_observer_factory", None)


def make_identity(*, recovery: bool = True) -> KVRecoveryIdentity:
    trace_id = "1" * 32
    lifecycle_id = f"{trace_id}:e:0"
    return KVRecoveryIdentity(
        run_id="0" * 32,
        trace_id=trace_id,
        engine_lifecycle_id=lifecycle_id,
        runtime_request_id="request-0",
        recovery_epoch=1 if recovery else None,
        episode_id=f"{lifecycle_id}:k:1" if recovery else None,
        base_preempted_event_id=f"{'b' * 32}:e:0" if recovery else None,
        preempt_profile_record_id=f"{'b' * 32}:k:0" if recovery else None,
    )


def make_context(*, operation: str = "h2d_restore") -> KVRecoveryTransferContext:
    identity = make_identity(recovery=operation == "h2d_restore")
    logical_blocks = (
        KVRecoveryLogicalBlock(0, 2, "2" * 32),
        KVRecoveryLogicalBlock(1, 3, "3" * 32),
    )
    return KVRecoveryTransferContext(
        identity=identity,
        operation=operation,  # type: ignore[arg-type]
        block_set_id=canonical_block_set_id(identity, logical_blocks),
        logical_blocks=logical_blocks,
    )


def make_receipt(
    *,
    transfer_id: str | None = None,
    process_uuid: str = "a" * 32,
) -> KVRecoveryH2DReceipt:
    context = make_context()
    resolved_transfer_id = transfer_id or f"{process_uuid}:t:0"
    return KVRecoveryH2DReceipt(
        connector_job_id=7,
        transfer_id=resolved_transfer_id,
        identity=context.identity,
        block_set_id=context.block_set_id,
        process_uuid=process_uuid,
        rank=0,
        world_size=1,
        clock_domain_id=CLOCK_DOMAIN_ID,
        communication_done_event_id=f"{process_uuid}:e:8",
        restore_done_profile_record_id=f"{process_uuid}:k:9",
        timestamp_ns=10,
        bytes_moved=128,
    )


def make_attempt_receipt(
    attempt: KVRecoveryTransferAttempt,
    timestamp_ns: int,
    bytes_moved: int,
    *,
    clock_domain_id: str = CLOCK_DOMAIN_ID,
) -> KVRecoveryH2DReceipt:
    context = attempt.context
    return KVRecoveryH2DReceipt(
        connector_job_id=attempt.connector_job_id,
        transfer_id=attempt.transfer_id,
        identity=context.identity,
        block_set_id=context.block_set_id,
        process_uuid=PROCESS_UUID,
        rank=0,
        world_size=1,
        clock_domain_id=clock_domain_id,
        communication_done_event_id=(f"{PROCESS_UUID}:e:{attempt.connector_job_id}"),
        restore_done_profile_record_id=(f"{PROCESS_UUID}:k:{attempt.connector_job_id}"),
        timestamp_ns=timestamp_ns,
        bytes_moved=bytes_moved,
    )


class NoopEvidenceSink:
    def __init__(self):
        self.failures: list[str] = []
        self.close_calls = 0
        self.first_compute_observations: list[tuple[object, int]] = []

    def transfer_submitted(self, attempt, timestamp_ns):
        return None

    def transfer_not_submitted(self, attempt):
        return None

    def transfer_completed(
        self,
        attempt,
        submit_timestamp_ns,
        timestamp_ns,
        success,
        bytes_moved,
        device_duration_ns,
    ):
        return None

    def transfer_capacity_exhausted(
        self,
        attempt,
        capacity,
        timestamp_ns,
        loss_reason,
    ):
        return None

    def evidence_failure(
        self,
        reason,
        connector_job_ids,
        transfer_ids,
        timestamp_ns,
    ):
        self.failures.append(reason)

    def wait_completed(self, attempt):
        return None

    def h2d_receipt_capacity_exhausted(self, receipt, loss_reason):
        return None

    def first_compute(self, context, timestamp_ns):
        self.first_compute_observations.append((context, timestamp_ns))

    def close(self, open_attempts, evidence_disabled):
        self.close_calls += 1


def test_recovery_abi_round_trips_through_pickle():
    context = make_context()
    attempt = KVRecoveryTransferAttempt(
        connector_job_id=7,
        transfer_id=f"{'a' * 32}:t:0",
        context=context,
    )
    compute_context = KVRecoveryComputeContext(
        identity=context.identity,
        transfer_id=attempt.transfer_id,
        block_set_id=context.block_set_id,
        bytes_moved=128,
        admission_profile_record_id=f"{'b' * 32}:k:1",
        compute_kind="prefill",
        base_phase_start_event_id=f"{'d' * 32}:e:0",
    )

    assert pickle.loads(pickle.dumps(context)) == context
    assert pickle.loads(pickle.dumps(attempt)) == attempt
    assert pickle.loads(pickle.dumps(compute_context)) == compute_context


def test_first_compute_requires_exact_context_and_base_event():
    context = make_context()
    compute_context = KVRecoveryComputeContext(
        identity=context.identity,
        transfer_id=f"{PROCESS_UUID}:t:0",
        block_set_id=context.block_set_id,
        bytes_moved=128,
        admission_profile_record_id=f"{'b' * 32}:k:1",
        compute_kind="prefill",
        base_phase_start_event_id=f"{'d' * 32}:e:0",
    )
    sink = NoopEvidenceSink()
    observer = BoundedKVRecoveryWorkerObserver(
        PROCESS_UUID, RUN_ID, CLOCK_DOMAIN_ID, sink
    )

    observer.first_compute(compute_context, 20)
    assert sink.first_compute_observations == [(compute_context, 20)]

    invalid_sink = NoopEvidenceSink()
    invalid_observer = BoundedKVRecoveryWorkerObserver(
        PROCESS_UUID, RUN_ID, CLOCK_DOMAIN_ID, invalid_sink
    )
    invalid_observer.first_compute(
        replace(
            compute_context,
            identity=replace(compute_context.identity, run_id="9" * 32),
        ),
        20,
    )
    assert invalid_observer.evidence_disabled
    assert invalid_sink.failures == ["invalid_first_compute_observation"]

    missing_sink = NoopEvidenceSink()
    missing_observer = BoundedKVRecoveryWorkerObserver(
        PROCESS_UUID, RUN_ID, CLOCK_DOMAIN_ID, missing_sink
    )
    missing_observer.first_compute_not_observed(compute_context)
    assert missing_observer.evidence_disabled
    assert missing_sink.failures == ["missing_first_compute_observation"]


def test_context_rejects_coordinate_or_digest_drift():
    context = make_context()
    reversed_blocks = tuple(reversed(context.logical_blocks))

    with pytest.raises(ValueError, match="canonical coordinate order"):
        KVRecoveryTransferContext(
            identity=context.identity,
            operation=context.operation,
            block_set_id=context.block_set_id,
            logical_blocks=reversed_blocks,
        )

    with pytest.raises(ValueError, match="does not match"):
        KVRecoveryTransferContext(
            identity=context.identity,
            operation=context.operation,
            block_set_id="0" * 64,
            logical_blocks=context.logical_blocks,
        )


def test_h2d_requires_episode_and_d2h_forbids_episode():
    h2d_context = make_context(operation="h2d_restore")
    d2h_context = make_context(operation="d2h_preserve")

    # Episode-driven H2D needs the episode fields; an unassociated H2D
    # (block-level tiering migration of a running request) is valid without
    # them, so the non-recovery identity must not be rejected.
    unassociated = KVRecoveryTransferContext(
        identity=d2h_context.identity,
        operation="h2d_restore",
        block_set_id=canonical_block_set_id(
            d2h_context.identity, h2d_context.logical_blocks
        ),
        logical_blocks=h2d_context.logical_blocks,
    )
    assert unassociated.identity.recovery_epoch is None

    # A d2h context may never carry a recovery episode.
    with pytest.raises(ValueError, match="D2H preserve cannot"):
        KVRecoveryTransferContext(
            identity=h2d_context.identity,
            operation="d2h_preserve",
            block_set_id=canonical_block_set_id(
                h2d_context.identity, d2h_context.logical_blocks
            ),
            logical_blocks=d2h_context.logical_blocks,
        )


def test_unregistered_factory_is_a_noop():
    assert create_kv_recovery_scheduler_observer() is None
    assert create_kv_recovery_worker_observer() is None


def test_runtime_scope_requires_explicit_enable_and_supported_spec():
    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
    from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

    class CustomTieringSpec(TieringOffloadingSpec):
        pass

    tiering_spec = object.__new__(TieringOffloadingSpec)
    enabled = SimpleNamespace(
        additional_config={
            "kv_recovery_profile_enabled": True,
            "recompute_scheduler_enable": False,
        }
    )

    assert recovery_profile.kv_recovery_runtime_scope_enabled(tiering_spec, enabled)
    assert not recovery_profile.kv_recovery_runtime_scope_enabled(
        object.__new__(CPUOffloadingSpec), enabled
    )
    assert not recovery_profile.kv_recovery_runtime_scope_enabled(
        object.__new__(CustomTieringSpec), enabled
    )
    assert not recovery_profile.kv_recovery_runtime_scope_enabled(
        tiering_spec,
        SimpleNamespace(additional_config={"recompute_scheduler_enable": False}),
    )
    assert not recovery_profile.kv_recovery_runtime_scope_enabled(
        tiering_spec,
        SimpleNamespace(
            additional_config={
                "kv_recovery_profile_enabled": True,
                "recompute_scheduler_enable": True,
            }
        ),
    )


def test_factory_failures_degrade_to_none():
    class BrokenFactory:
        def reinitialize_after_fork(self):
            raise RuntimeError("broken bootstrap")

        def create_scheduler_observer(self):
            raise RuntimeError("broken scheduler")

        def create_worker_observer(self):
            raise RuntimeError("broken worker")

    register_kv_recovery_observer_factory(BrokenFactory())
    reinitialize_kv_recovery_profile_after_fork()
    assert recovery_profile._observer_factory is None
    assert create_kv_recovery_scheduler_observer() is None
    assert create_kv_recovery_worker_observer() is None


def test_factory_registration_rejects_replacement():
    class Factory:
        def reinitialize_after_fork(self):
            return None

        def create_scheduler_observer(self):
            return None

        def create_worker_observer(self):
            return None

    first = Factory()
    register_kv_recovery_observer_factory(first)
    register_kv_recovery_observer_factory(first)

    with pytest.raises(RuntimeError, match="already registered"):
        register_kv_recovery_observer_factory(Factory())


def test_first_create_after_fork_reinitializes_factory_once(
    monkeypatch: pytest.MonkeyPatch,
):
    scheduler_observer = object()

    class Factory:
        def __init__(self):
            self.reinitialize_calls = 0

        def reinitialize_after_fork(self):
            self.reinitialize_calls += 1

        def create_scheduler_observer(self):
            return scheduler_observer

        def create_worker_observer(self):
            return None

    factory = Factory()
    register_kv_recovery_observer_factory(factory)
    monkeypatch.setattr(recovery_profile, "_observer_factory_pid", -1)
    monkeypatch.setattr(recovery_profile, "_observer_factory_ready_pid", -1)

    assert create_kv_recovery_scheduler_observer() is scheduler_observer
    assert create_kv_recovery_worker_observer() is None
    assert factory.reinitialize_calls == 1


def test_child_registration_can_replace_inherited_factory(
    monkeypatch: pytest.MonkeyPatch,
):
    class Factory:
        def __init__(self, scheduler_observer):
            self.scheduler_observer = scheduler_observer

        def reinitialize_after_fork(self):
            raise AssertionError("fresh child factory must not be reinitialized")

        def create_scheduler_observer(self):
            return self.scheduler_observer

        def create_worker_observer(self):
            return None

    register_kv_recovery_observer_factory(Factory(object()))
    monkeypatch.setattr(recovery_profile, "_observer_factory_pid", -1)
    monkeypatch.setattr(recovery_profile, "_observer_factory_ready_pid", -1)
    child_scheduler_observer = object()
    child_factory = Factory(child_scheduler_observer)

    register_kv_recovery_observer_factory(child_factory)

    assert recovery_profile._observer_factory is child_factory
    assert create_kv_recovery_scheduler_observer() is child_scheduler_observer


def test_coordinate_rejects_negative_values():
    with pytest.raises(ValueError, match="uint32"):
        KVRecoveryBlockCoordinate(-1, 0)


def test_all_abi_uint_fields_reject_bool_and_float_values():
    for value in (False, 0.0):
        with pytest.raises(ValueError, match="uint32"):
            KVRecoveryBlockCoordinate(value, 0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="ordinal"):
            KVRecoveryBlockCoordinate(0, value)  # type: ignore[arg-type]

    identity = make_identity()
    lifecycle_id = identity.engine_lifecycle_id
    for value in (True, 1.0):
        with pytest.raises(ValueError, match="recovery_epoch"):
            replace(
                identity,
                recovery_epoch=value,  # type: ignore[arg-type]
                episode_id=f"{lifecycle_id}:k:{value}",
            )

    context = make_context()
    for value in (False, 7.0):
        with pytest.raises(ValueError, match="connector_job_id"):
            KVRecoveryTransferAttempt(
                connector_job_id=value,  # type: ignore[arg-type]
                transfer_id=f"{PROCESS_UUID}:t:0",
                context=context,
            )

    receipt = make_receipt()
    for field_name, value, message in (
        ("connector_job_id", 7.0, "connector_job_id"),
        ("rank", False, "rank=0"),
        ("world_size", True, "rank=0"),
        ("timestamp_ns", 10.0, "timestamp/bytes"),
        ("bytes_moved", 128.0, "timestamp/bytes"),
    ):
        with pytest.raises(ValueError, match=message):
            replace(receipt, **{field_name: value})

    for value in (False, 10.0):
        with pytest.raises(ValueError, match="wait-entry"):
            KVRecoveryWaitAttempt(
                recovery_profile.KVRecoveryWaitMembership((f"{PROCESS_UUID}:t:0",)),
                value,  # type: ignore[arg-type]
            )


def test_identity_and_scoped_ids_reject_noncanonical_values():
    identity = make_identity()

    with pytest.raises(ValueError, match="run_id"):
        replace(identity, run_id="run-0")
    with pytest.raises(ValueError, match="base_preempted_event_id"):
        replace(identity, base_preempted_event_id="preempted-0")
    with pytest.raises(ValueError, match="transfer_id"):
        KVRecoveryTransferAttempt(
            connector_job_id=7,
            transfer_id=f"{'a' * 32}:t:{2**64}",
            context=make_context(),
        )

    receipt = make_receipt()
    with pytest.raises(ValueError, match="belong to process_uuid"):
        replace(receipt, transfer_id=f"{'b' * 32}:t:0")
    with pytest.raises(ValueError, match="rank=0"):
        replace(receipt, rank=1, world_size=2)
    with pytest.raises(ValueError, match="communication_done_event_id"):
        replace(receipt, communication_done_event_id=f"{'b' * 32}:e:8")
    with pytest.raises(ValueError, match="restore_done_profile_record_id"):
        replace(receipt, restore_done_profile_record_id=f"{'a' * 32}:k:{2**64}")


def test_transfer_seq_uses_last_uint64_then_saturates_without_wrap():
    sink = NoopEvidenceSink()
    observer = BoundedKVRecoveryWorkerObserver(
        PROCESS_UUID, RUN_ID, CLOCK_DOMAIN_ID, sink
    )
    observer._transfer_seq = 2**64 - 1

    last = observer.begin_transfer(7, make_context())
    assert last is not None
    assert last.transfer_id == f"{'a' * 32}:t:{2**64 - 1}"
    observer.transfer_not_submitted(last)

    assert observer.begin_transfer(8, make_context()) is None
    assert observer.transfer_seq == 2**64
    assert observer.evidence_disabled
    assert sink.failures == ["transfer_seq_exhausted"]

    assert observer.begin_transfer(9, make_context()) is None
    assert observer.transfer_seq == 2**64
    assert sink.failures == ["transfer_seq_exhausted"]


def test_bounded_observer_close_is_idempotent_and_terminal():
    sink = NoopEvidenceSink()
    observer = BoundedKVRecoveryWorkerObserver(
        PROCESS_UUID, RUN_ID, CLOCK_DOMAIN_ID, sink
    )
    attempt = observer.begin_transfer(7, make_context())
    assert attempt is not None

    observer.close()
    observer.close()

    assert sink.close_calls == 1
    assert observer.pending_h2d_count == 0
    assert observer.begin_transfer(8, make_context()) is None
    observer.transfer_not_submitted(attempt)
    assert sink.close_calls == 1
    assert sink.failures == []


def test_close_racing_begin_cannot_leave_a_new_attempt():
    close_entered = threading.Event()
    close_release = threading.Event()

    class BlockingCloseSink(NoopEvidenceSink):
        def close(self, open_attempts, evidence_disabled):
            close_entered.set()
            assert close_release.wait(timeout=3)
            super().close(open_attempts, evidence_disabled)

    sink = BlockingCloseSink()
    observer = BoundedKVRecoveryWorkerObserver(
        PROCESS_UUID, RUN_ID, CLOCK_DOMAIN_ID, sink
    )
    results: list[KVRecoveryTransferAttempt | None] = []
    close_thread = threading.Thread(target=observer.close)
    close_thread.start()
    assert close_entered.wait(timeout=3)

    begin_thread = threading.Thread(
        target=lambda: results.append(observer.begin_transfer(7, make_context()))
    )
    begin_thread.start()
    close_release.set()
    close_thread.join(timeout=3)
    begin_thread.join(timeout=3)

    assert not close_thread.is_alive()
    assert not begin_thread.is_alive()
    assert results == [None]
    assert observer.pending_h2d_count == 0
    assert sink.close_calls == 1


def test_bounded_observer_rejects_sink_receipt_for_another_transfer():
    class InvalidReceiptSink(NoopEvidenceSink):
        def transfer_completed(
            self,
            attempt,
            submit_timestamp_ns,
            timestamp_ns,
            success,
            bytes_moved,
            device_duration_ns,
        ):
            context = attempt.context
            return KVRecoveryH2DReceipt(
                connector_job_id=attempt.connector_job_id,
                transfer_id=f"{'a' * 32}:t:999",
                identity=context.identity,
                block_set_id=context.block_set_id,
                process_uuid="a" * 32,
                rank=0,
                world_size=1,
                clock_domain_id=CLOCK_DOMAIN_ID,
                communication_done_event_id=f"{'a' * 32}:e:8",
                restore_done_profile_record_id=f"{'a' * 32}:k:9",
                timestamp_ns=timestamp_ns,
                bytes_moved=bytes_moved,
            )

    sink = InvalidReceiptSink()
    observer = BoundedKVRecoveryWorkerObserver(
        PROCESS_UUID, RUN_ID, CLOCK_DOMAIN_ID, sink
    )
    attempt = observer.begin_transfer(7, make_context())
    assert attempt is not None
    observer.transfer_submitted(attempt, 10)

    receipt = observer.transfer_completed(7, 20, True, 128, 5)

    assert receipt is None
    assert sink.failures == ["invalid_h2d_receipt"]


def test_bounded_observer_rejects_foreign_clock_receipt():
    class ForeignClockSink(NoopEvidenceSink):
        def transfer_completed(
            self,
            attempt,
            submit_timestamp_ns,
            timestamp_ns,
            success,
            bytes_moved,
            device_duration_ns,
        ):
            return make_attempt_receipt(
                attempt,
                timestamp_ns,
                bytes_moved,
                clock_domain_id="d" * 32,
            )

    sink = ForeignClockSink()
    observer = BoundedKVRecoveryWorkerObserver(
        PROCESS_UUID, RUN_ID, CLOCK_DOMAIN_ID, sink
    )
    attempt = observer.begin_transfer(7, make_context())
    assert attempt is not None
    observer.transfer_submitted(attempt, 10)

    assert observer.transfer_completed(7, 20, True, 128, 5) is None
    assert sink.failures == ["invalid_h2d_receipt"]


def test_foreign_run_transfer_cannot_enter_wait_membership():
    sink = NoopEvidenceSink()
    observer = BoundedKVRecoveryWorkerObserver(
        PROCESS_UUID, RUN_ID, CLOCK_DOMAIN_ID, sink
    )
    local_context = make_context(operation="d2h_preserve")
    local_attempt = observer.begin_transfer(7, local_context)
    assert local_attempt is not None
    observer.transfer_submitted(local_attempt, 10)

    foreign_identity = replace(local_context.identity, run_id="f" * 32)
    foreign_context = replace(
        local_context,
        identity=foreign_identity,
        block_set_id=canonical_block_set_id(
            foreign_identity, local_context.logical_blocks
        ),
    )
    assert observer.begin_transfer(8, foreign_context) is None
    assert observer.prepare_wait(frozenset({7, 8})) is None
    assert sink.failures == ["foreign_run_transfer", "invalid_wait_membership"]


@pytest.mark.parametrize("operation", ["h2d_restore", "d2h_preserve"])
def test_connector_flush_invalidates_pending_context_exactly_once(operation):
    sink = NoopEvidenceSink()
    observer = BoundedKVRecoveryWorkerObserver(
        PROCESS_UUID, RUN_ID, CLOCK_DOMAIN_ID, sink
    )
    attempt = observer.begin_transfer(7, make_context(operation=operation))
    assert attempt is not None
    observer.transfer_submitted(attempt, 10)

    observer.invalidate_transfers(frozenset({7, 8}))
    observer.invalidate_transfers(frozenset({7}))

    assert observer.prepared_transfer_count == 0
    assert observer.pending_h2d_count == 0
    assert observer.pending_d2h_count == 0
    # A scheduler discard handoff is an expected lifecycle transition, not a
    # state-capacity failure. The invalidated contexts fail closed, but the
    # observer remains usable so later H2D restore evidence is still captured.
    assert not observer.evidence_disabled
    assert sink.failures == ["connector_flush_invalidation"]
    # A late completion for the invalidated context is now observable as an
    # unknown completion instead of being silently swallowed by the latch.
    assert observer.transfer_completed(7, 20, True, 128, 5) is None
    assert sink.failures == [
        "connector_flush_invalidation",
        "unknown_completion",
    ]
    later = observer.begin_transfer(9, make_context())
    assert later is not None


@pytest.mark.parametrize(
    ("connector_job_id", "timestamp_ns", "success", "bytes_moved", "duration_ns"),
    [
        (7.0, 20, True, 128, 5),
        (7, 20.0, True, 128, 5),
        (7, 20, 1, 128, 5),
        (7, 20, True, 128.0, 5),
        (7, 20, True, 128, 5.0),
    ],
)
def test_completion_rejects_non_exact_integer_inputs_without_consuming_attempt(
    connector_job_id,
    timestamp_ns,
    success,
    bytes_moved,
    duration_ns,
):
    sink = NoopEvidenceSink()
    observer = BoundedKVRecoveryWorkerObserver(
        PROCESS_UUID, RUN_ID, CLOCK_DOMAIN_ID, sink
    )
    attempt = observer.begin_transfer(7, make_context())
    assert attempt is not None
    observer.transfer_submitted(attempt, 10)

    assert (
        observer.transfer_completed(
            connector_job_id,
            timestamp_ns,
            success,
            bytes_moved,
            duration_ns,
        )
        is None
    )
    assert observer.prepare_wait(frozenset({7})) is not None
    assert sink.failures == ["invalid_completion_measurement"]


def test_equal_h2d_timestamps_reach_profile_sink_but_cannot_emit_receipt():
    class RecordingReceiptSink(NoopEvidenceSink):
        def __init__(self):
            super().__init__()
            self.completion_calls = 0

        def transfer_completed(
            self,
            attempt,
            submit_timestamp_ns,
            timestamp_ns,
            success,
            bytes_moved,
            device_duration_ns,
        ):
            self.completion_calls += 1
            return make_attempt_receipt(attempt, timestamp_ns, bytes_moved)

    sink = RecordingReceiptSink()
    observer = BoundedKVRecoveryWorkerObserver(
        PROCESS_UUID, RUN_ID, CLOCK_DOMAIN_ID, sink
    )
    attempt = observer.begin_transfer(7, make_context())
    assert attempt is not None
    observer.transfer_submitted(attempt, 10)

    assert observer.transfer_completed(7, 10, True, 128, 5) is None
    assert sink.completion_calls == 1
    assert sink.failures == ["invalid_h2d_receipt"]


def test_equal_d2h_timestamps_fail_before_profile_completion():
    class RecordingCompletionSink(NoopEvidenceSink):
        def __init__(self):
            super().__init__()
            self.completion_calls = 0

        def transfer_completed(
            self,
            attempt,
            submit_timestamp_ns,
            timestamp_ns,
            success,
            bytes_moved,
            device_duration_ns,
        ):
            self.completion_calls += 1
            return None

    sink = RecordingCompletionSink()
    observer = BoundedKVRecoveryWorkerObserver(
        PROCESS_UUID, RUN_ID, CLOCK_DOMAIN_ID, sink
    )
    attempt = observer.begin_transfer(7, make_context(operation="d2h_preserve"))
    assert attempt is not None
    observer.transfer_submitted(attempt, 10)

    assert observer.transfer_completed(7, 10, True, 128, 5) is None
    assert sink.completion_calls == 0
    assert sink.failures == ["invalid_completion_measurement"]


def test_overbound_wait_is_rejected_before_membership_resolution():
    sink = NoopEvidenceSink()
    observer = BoundedKVRecoveryWorkerObserver(
        PROCESS_UUID, RUN_ID, CLOCK_DOMAIN_ID, sink
    )

    membership = observer.prepare_wait(frozenset(range(4097)))

    assert membership is None
    assert sink.failures == ["invalid_wait_membership"]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_prepare_after_fork_replaces_inherited_factory_lock(
    monkeypatch: pytest.MonkeyPatch,
):
    class Factory:
        def reinitialize_after_fork(self):
            return None

        def create_scheduler_observer(self):
            return None

        def create_worker_observer(self):
            return None

    read_fd, write_fd = os.pipe()
    inherited_lock = recovery_profile._observer_factory_lock
    inherited_lock.acquire()
    child_pid = os.fork()
    if child_pid == 0:
        os.close(read_fd)
        try:
            prepare_kv_recovery_profile_after_fork()
            register_kv_recovery_observer_factory(Factory())
            os.write(write_fd, b"ok")
        except BaseException as exc:
            os.write(write_fd, f"error:{exc!r}".encode())
        finally:
            os.close(write_fd)
            os._exit(0)

    os.close(write_fd)
    inherited_lock.release()
    ready, _, _ = select.select([read_fd], [], [], 3.0)
    if not ready:
        os.kill(child_pid, signal.SIGKILL)
        os.waitpid(child_pid, 0)
        pytest.fail("fork child deadlocked on inherited factory lock")
    result = os.read(read_fd, 1024)
    os.close(read_fd)
    _, status = os.waitpid(child_pid, 0)

    assert os.waitstatus_to_exitcode(status) == 0
    assert result == b"ok"
