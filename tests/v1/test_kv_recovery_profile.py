# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import pickle
import select
import signal
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import vllm.v1.kv_recovery_profile as recovery_profile
from vllm.v1.kv_recovery_profile import (
    KV_RECOVERY_PROFILE_BINDING,
    MAX_H2D_RECEIPTS_PER_WORKER_STEP,
    MAX_PENDING_D2H_CONTEXTS_PER_PROCESS,
    MAX_PENDING_H2D_CONTEXTS_PER_PROCESS,
    MAX_PREPARED_TRANSFER_ATTEMPTS_PER_PROCESS,
    BoundedKVRecoveryWorkerObserver,
    KVRecoveryBlockCoordinate,
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
    )


def make_context(*, operation: str = "h2d_restore") -> KVRecoveryTransferContext:
    identity = make_identity(recovery=operation == "h2d_restore")
    logical_blocks = (
        KVRecoveryLogicalBlock(0, 2, "2" * 32),
        KVRecoveryLogicalBlock(1, 3, "3" * 32),
    )
    return KVRecoveryTransferContext(
        binding=KV_RECOVERY_PROFILE_BINDING,
        identity=identity,
        operation=operation,  # type: ignore[arg-type]
        block_set_id=canonical_block_set_id(
            KV_RECOVERY_PROFILE_BINDING, identity, logical_blocks
        ),
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
        binding=context.binding,
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
        binding=context.binding,
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

    def transfer_submitted(self, attempt, timestamp_ns):
        return None

    def transfer_not_submitted(self, attempt):
        return None

    def transfer_completed(self, **kwargs):
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

    def close(self, open_attempts, evidence_disabled):
        self.close_calls += 1


def test_binding_locks_authorized_candidate_digests():
    binding = KV_RECOVERY_PROFILE_BINDING
    assert binding.profile_sha256 == (
        "b363532884d1cae8049ab080d2b85a629f3b33a75f6621788d1e4c8f30737666"
    )
    assert binding.profile_owner_approval_record_sha256 == (
        "2831ce52802e7cbe4ec092431c71c18de05491da7ac8d48512014b1d43b3cb0c"
    )
    assert binding.observer_policy_sha256 == (
        "fbee3bc4f74b8c5c20e929c4e37596b06b31c1b9cd02517d8461961a8aedf420"
    )
    assert binding.observer_policy_approval_candidate_sha256 == (
        "27cb52269fff8f10e376f3128384c4aac37f149ad685b89501f98ba50bc0cf29"
    )
    assert binding.observer_policy_profile_p0_owner_approval_sha256 == (
        "322df19ade75797fc4c3f43fe96e689404ef60a270c92081afea03683e734c91"
    )
    assert binding.communication_mapping_sha256 == (
        "095944bbbb1a3ad3518aebfdd61c820ade3affdebd6024b47389cdaec24a3fa3"
    )
    assert binding.communication_mapping_approval_candidate_sha256 == (
        "a686243ffd9c650790e6421e5f976a2a5c010a5d2480e7a30ad8722a9218f3f7"
    )
    assert binding.communication_mapping_authority_approval_sha256 == (
        "42b761177c4bea13833be19ddc143f89c32eee02fbbb98afc2c03249c2783fe5"
    )
    assert binding.base_mode_overlay_sha256 == (
        "d80771b793a9513cdbd4fc2001e21771c56a59d49340ee88d4381027d70b1d5a"
    )
    assert binding.base_mode_overlay_approval_candidate_sha256 == (
        "de889357af659d8e9c1f7daf0aa32656761e80dcf0fea29ef60809912f513694"
    )
    assert binding.base_mode_overlay_owner_approval_sha256 == (
        "50d7deb8f98fcdca36303297a4ce6f7958ae619574d2e1994872936b6fc583a3"
    )
    assert binding.g0_remediation_cpu_result_sha256 == (
        "4b378d10cef6be10fa3fb5a840402da77b1da755781118127a5d30b1cc4b7eb2"
    )
    assert binding.g1_cpu_source_reauthorization_sha256 == (
        "448ff68cd139077a2f767fd16359bb5dd8f4cf61cfe258fd48f53a26569718f5"
    )
    assert binding.communication_mapping_status == "approved_mapping_only"
    assert binding.base_mode_overlay_status == "approved_overlay_only"
    assert binding.observer_policy_status == (
        "profile_P0_items_1_5_approved_runtime_items_6_8_pending"
    )
    assert binding.complete_configuration_status == "pending"
    assert binding.runtime_conformance_status == "pending"
    assert binding.joint_admission_status == "pending"
    assert binding.communication_mode == "none"
    assert not binding.runtime_activation_authorized
    assert not recovery_profile.KV_RECOVERY_RUNTIME_ACTIVATION_AUTHORIZED
    assert binding.runtime_commit == "f229ba7cad21a4dba58681af6738a9fd947388e2"
    assert binding.device_plugin_audit_commit == (
        "cafad89a5e103f31ea517c1edb56130578c3cd56"
    )
    assert {
        MAX_PREPARED_TRANSFER_ATTEMPTS_PER_PROCESS,
        MAX_PENDING_H2D_CONTEXTS_PER_PROCESS,
        MAX_PENDING_D2H_CONTEXTS_PER_PROCESS,
        MAX_H2D_RECEIPTS_PER_WORKER_STEP,
    } == {4096}


def test_mapping_and_overlay_approval_cannot_open_activation_gate(
    monkeypatch: pytest.MonkeyPatch,
):
    binding = replace(
        KV_RECOVERY_PROFILE_BINDING,
        communication_mode=KV_RECOVERY_PROFILE_BINDING.communication_mapping_id,
        runtime_activation_authorized=True,
    )
    monkeypatch.setattr(
        recovery_profile,
        "KV_RECOVERY_RUNTIME_ACTIVATION_AUTHORIZED",
        True,
    )
    monkeypatch.setattr(recovery_profile, "KV_RECOVERY_PROFILE_BINDING", binding)

    assert not recovery_profile._activation_gate_open()

    fully_admitted = replace(
        binding,
        observer_policy_status="fully_ratified",
        complete_configuration_status="approved",
        runtime_conformance_status="conformant",
        joint_admission_status="approved",
    )
    monkeypatch.setattr(
        recovery_profile,
        "KV_RECOVERY_PROFILE_BINDING",
        fully_admitted,
    )
    assert recovery_profile._activation_gate_open()


def test_recovery_abi_round_trips_through_pickle():
    context = make_context()
    attempt = KVRecoveryTransferAttempt(
        connector_job_id=7,
        transfer_id=f"{'a' * 32}:t:0",
        context=context,
    )

    assert pickle.loads(pickle.dumps(context)) == context
    assert pickle.loads(pickle.dumps(attempt)) == attempt


def test_context_rejects_coordinate_or_digest_drift():
    context = make_context()
    reversed_blocks = tuple(reversed(context.logical_blocks))

    with pytest.raises(ValueError, match="canonical coordinate order"):
        KVRecoveryTransferContext(
            binding=context.binding,
            identity=context.identity,
            operation=context.operation,
            block_set_id=context.block_set_id,
            logical_blocks=reversed_blocks,
        )

    with pytest.raises(ValueError, match="does not match"):
        KVRecoveryTransferContext(
            binding=context.binding,
            identity=context.identity,
            operation=context.operation,
            block_set_id="0" * 64,
            logical_blocks=context.logical_blocks,
        )


def test_h2d_requires_episode_and_d2h_forbids_episode():
    h2d_context = make_context(operation="h2d_restore")
    d2h_context = make_context(operation="d2h_preserve")

    with pytest.raises(ValueError, match="H2D restore requires"):
        KVRecoveryTransferContext(
            binding=h2d_context.binding,
            identity=d2h_context.identity,
            operation="h2d_restore",
            block_set_id=canonical_block_set_id(
                h2d_context.binding,
                d2h_context.identity,
                h2d_context.logical_blocks,
            ),
            logical_blocks=h2d_context.logical_blocks,
        )

    with pytest.raises(ValueError, match="D2H preserve cannot"):
        KVRecoveryTransferContext(
            binding=d2h_context.binding,
            identity=h2d_context.identity,
            operation="d2h_preserve",
            block_set_id=canonical_block_set_id(
                d2h_context.binding,
                h2d_context.identity,
                d2h_context.logical_blocks,
            ),
            logical_blocks=d2h_context.logical_blocks,
        )


def test_default_path_has_no_observer_or_profiler_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    class UnexpectedFactory:
        def reinitialize_after_fork(self, binding):
            raise AssertionError("hard-off factory was called")

        def create_scheduler_observer(self, binding):
            raise AssertionError("hard-off factory was called")

        def create_worker_observer(self, binding):
            raise AssertionError("hard-off factory was called")

    forbidden_output = tmp_path / "must-not-exist.jsonl"
    monkeypatch.setenv("VLLM_RLP_TRACE_EXPORT_PATH", str(forbidden_output))
    monkeypatch.setenv("VLLM_RLP_KV_RECOVERY_PROFILE", "rlp.kv-recovery/v1alpha1")
    register_kv_recovery_observer_factory(UnexpectedFactory())
    assert recovery_profile._observer_factory is None
    initial_threads = tuple(thread.ident for thread in threading.enumerate())

    assert create_kv_recovery_scheduler_observer() is None
    assert create_kv_recovery_worker_observer() is None
    monkeypatch.setattr(
        recovery_profile.os,
        "getpid",
        lambda: (_ for _ in ()).throw(
            AssertionError("hard-off path queried fork state")
        ),
    )
    prepare_kv_recovery_profile_after_fork()
    reinitialize_kv_recovery_profile_after_fork()

    assert not any(
        module_name.startswith("vllm_request_lifecycle_profiler")
        for module_name in sys.modules
    )
    assert tuple(thread.ident for thread in threading.enumerate()) == initial_threads
    assert not forbidden_output.exists()


def test_pending_authorities_keep_activation_gate_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        recovery_profile, "KV_RECOVERY_RUNTIME_ACTIVATION_AUTHORIZED", True
    )

    assert not recovery_profile._activation_gate_open()


def test_runtime_scope_requires_exact_tiering_spec_and_recompute_off(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
    from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

    class CustomTieringSpec(TieringOffloadingSpec):
        pass

    monkeypatch.setattr(recovery_profile, "_activation_gate_open", lambda: True)
    tiering_spec = object.__new__(TieringOffloadingSpec)
    recompute_off = SimpleNamespace(
        additional_config={"recompute_scheduler_enable": False}
    )

    assert recovery_profile.kv_recovery_runtime_scope_authorized(
        tiering_spec, recompute_off
    )
    assert not recovery_profile.kv_recovery_runtime_scope_authorized(
        object.__new__(CPUOffloadingSpec), recompute_off
    )
    assert not recovery_profile.kv_recovery_runtime_scope_authorized(
        object.__new__(CustomTieringSpec), recompute_off
    )
    assert not recovery_profile.kv_recovery_runtime_scope_authorized(
        tiering_spec, SimpleNamespace(additional_config={})
    )
    assert not recovery_profile.kv_recovery_runtime_scope_authorized(
        tiering_spec,
        SimpleNamespace(additional_config={"recompute_scheduler_enable": True}),
    )


def test_factory_failures_degrade_to_none(monkeypatch: pytest.MonkeyPatch):
    class BrokenFactory:
        def reinitialize_after_fork(self, binding):
            raise RuntimeError("broken bootstrap")

        def create_scheduler_observer(self, binding):
            raise RuntimeError("broken scheduler")

        def create_worker_observer(self, binding):
            raise RuntimeError("broken worker")

    monkeypatch.setattr(recovery_profile, "_activation_gate_open", lambda: True)
    factory = BrokenFactory()
    register_kv_recovery_observer_factory(factory)

    reinitialize_kv_recovery_profile_after_fork()
    assert recovery_profile._observer_factory is None
    assert create_kv_recovery_scheduler_observer() is None
    assert create_kv_recovery_worker_observer() is None


def test_factory_registration_rejects_replacement(
    monkeypatch: pytest.MonkeyPatch,
):
    class Factory:
        def reinitialize_after_fork(self, binding):
            return None

        def create_scheduler_observer(self, binding):
            return None

        def create_worker_observer(self, binding):
            return None

    monkeypatch.setattr(recovery_profile, "_activation_gate_open", lambda: True)
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

        def reinitialize_after_fork(self, binding):
            self.reinitialize_calls += 1

        def create_scheduler_observer(self, binding):
            return scheduler_observer

        def create_worker_observer(self, binding):
            return None

    monkeypatch.setattr(recovery_profile, "_activation_gate_open", lambda: True)
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

        def reinitialize_after_fork(self, binding):
            raise AssertionError("fresh child factory must not be reinitialized")

        def create_scheduler_observer(self, binding):
            return self.scheduler_observer

        def create_worker_observer(self, binding):
            return None

    monkeypatch.setattr(recovery_profile, "_activation_gate_open", lambda: True)
    parent_factory = Factory(object())
    register_kv_recovery_observer_factory(parent_factory)
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
                binding=context.binding,
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
            local_context.binding,
            foreign_identity,
            local_context.logical_blocks,
        ),
    )
    assert observer.begin_transfer(8, foreign_context) is None
    assert observer.prepare_wait(frozenset({7, 8})) is None
    assert sink.failures == ["foreign_run_transfer", "invalid_wait_membership"]


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

        def transfer_completed(self, **kwargs):
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
        def reinitialize_after_fork(self, binding):
            return None

        def create_scheduler_observer(self, binding):
            return None

        def create_worker_observer(self, binding):
            return None

    monkeypatch.setattr(recovery_profile, "_activation_gate_open", lambda: True)
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
