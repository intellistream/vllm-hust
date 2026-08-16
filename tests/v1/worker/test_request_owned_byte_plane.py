# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from uuid import UUID

import pytest

from vllm.v1.core.sched.ownership import (
    AttentionLeaseManager,
    OwnerAssignmentObservation,
    OwnerLeaseCoordinator,
    OwnerLeaseKey,
    OwnerReceipt,
)
from vllm.v1.worker.request_owned_byte_plane import (
    StateHarborBytePlaneAdapter,
    StateHarborBytePlaneError,
    StateHarborCommitOutcome,
    StateHarborGroupPayload,
    StateHarborRedockTicket,
    StateHarborSourceImage,
    StateHarborWriteLease,
    StateHarborWriterFence,
    derive_lmcache_instance_id,
)
from vllm.v1.worker.request_owned_kv import RequestOwnedKVSnapshot
from vllm.v1.worker.request_owned_offload import (
    OwnerOffloadPlan,
    RequestOwnedBulkOffloadLedger,
    RequestOwnedOffloadError,
    make_request_owned_offload_keys,
)


class _MemoryBytePlane:
    """Deterministic immutable-object backend with injectable terminals."""

    def __init__(self) -> None:
        self.pending: dict[StateHarborWriteLease, bytes | None] = {}
        self.objects: dict[bytes, bytes] = {}
        self.retired: set[StateHarborWriterFence] = set()
        self.aborted: list[StateHarborWriteLease] = []
        self.deleted: list[bytes] = []
        self.unknown_publish_sequences: set[int] = set()
        self.unknown_miss_sequences: set[int] = set()
        self.raise_after_publish_sequences: set[int] = set()
        self.raise_without_publish_sequences: set[int] = set()
        self.refuse_commit_sequences: set[int] = set()
        self.fail_write_sequences: set[int] = set()
        self.fail_delete_keys: set[bytes] = set()

    def prepare_write(self, lease: StateHarborWriteLease, byte_length: int) -> bool:
        if lease.writer in self.retired:
            raise RuntimeError("writer registration was reaped")
        if byte_length <= 0 or lease in self.pending:
            return False
        self.pending[lease] = None
        return True

    def write(self, lease: StateHarborWriteLease, payload: bytes) -> None:
        if lease.writer in self.retired:
            raise RuntimeError("writer registration was reaped")
        if lease not in self.pending:
            raise RuntimeError("write lease is not pending")
        if lease.sequence in self.fail_write_sequences:
            raise RuntimeError("injected write failure")
        self.pending[lease] = bytes(payload)

    def commit_write(self, lease: StateHarborWriteLease) -> StateHarborCommitOutcome:
        if lease.writer in self.retired:
            raise RuntimeError("writer registration was reaped")
        payload = self.pending.pop(lease, None)
        if payload is None:
            raise RuntimeError("write lease has no complete payload")
        if lease.sequence in self.refuse_commit_sequences:
            return StateHarborCommitOutcome.REFUSED
        if lease.sequence in self.unknown_miss_sequences:
            return StateHarborCommitOutcome.UNKNOWN
        if lease.sequence in self.raise_without_publish_sequences:
            raise TimeoutError("injected commit timeout before publication")
        self.objects[lease.object_key] = payload
        if lease.sequence in self.raise_after_publish_sequences:
            raise TimeoutError("injected commit timeout after publication")
        if lease.sequence in self.unknown_publish_sequences:
            return StateHarborCommitOutcome.UNKNOWN
        return StateHarborCommitOutcome.COMMITTED

    def abort_write(self, lease: StateHarborWriteLease) -> None:
        self.pending.pop(lease, None)
        self.aborted.append(lease)

    def read(self, object_key: bytes) -> bytes | None:
        return self.objects.get(object_key)

    def delete(self, object_key: bytes) -> bool:
        if object_key in self.fail_delete_keys:
            return False
        self.objects.pop(object_key, None)
        self.deleted.append(object_key)
        return True

    def reap(self, writer: StateHarborWriterFence) -> None:
        self.retired.add(writer)
        for lease in tuple(self.pending):
            if lease.writer == writer:
                self.pending.pop(lease)


def _snapshot(
    *,
    epoch: int = 4,
    owner_rank: int = 3,
    generation: int = 7,
    first_block: int = 1,
) -> RequestOwnedKVSnapshot:
    return RequestOwnedKVSnapshot(
        key=OwnerLeaseKey("stateharbor-request", epoch),
        owner_rank=owner_rank,
        allocation_generation=generation,
        num_computed_tokens=8,
        reserved_num_tokens=16,
        pending_free=False,
        tables=tuple((first_block + group,) for group in range(6)),
    )


def _source(snapshot: RequestOwnedKVSnapshot) -> StateHarborSourceImage:
    return StateHarborSourceImage(
        model_fingerprint=b"m" * 32,
        layout_fingerprint=b"l" * 32,
        session_id="session-a",
        request_id=snapshot.key.request_id,
        source_owner_rank=snapshot.owner_rank,
        source_owner_epoch=snapshot.key.owner_epoch,
        source_activation_generation=snapshot.allocation_generation,
    )


def _plan(snapshot: RequestOwnedKVSnapshot) -> OwnerOffloadPlan:
    keys = make_request_owned_offload_keys(snapshot, (8, 8, 8, 8, 8, 8))
    return OwnerOffloadPlan.from_snapshot(snapshot, keys)


def _payloads() -> tuple[StateHarborGroupPayload, ...]:
    sizes = (17, 129, 1025, 4097, 65539, 278605)
    return tuple(
        StateHarborGroupPayload(
            group_index=group,
            logical_token_span=(0, 8),
            valid_extents=(8,),
            payload=bytes((group * 31 + index) % 251 for index in range(size)),
        )
        for group, size in enumerate(sizes)
    )


def _writer(registration_generation: int = 1) -> StateHarborWriterFence:
    return StateHarborWriterFence(
        UUID("4e3f2dd2-59ab-4a91-a0e7-7c6066d85dc8"),
        registration_generation,
    )


def _begin_source_store(
    backend: _MemoryBytePlane,
    *,
    snapshot: RequestOwnedKVSnapshot | None = None,
    writer: StateHarborWriterFence | None = None,
):
    snapshot = snapshot or _snapshot()
    ledger = RequestOwnedBulkOffloadLedger(snapshot.owner_rank)
    adapter = StateHarborBytePlaneAdapter(ledger=ledger, backend=backend)
    identity = ledger.bind(snapshot, active=True)
    ledger.retire(identity)
    plan = _plan(snapshot)
    work = adapter.begin_store(
        plan=plan,
        source=_source(snapshot),
        groups=_payloads(),
        writer=writer or _writer(),
    )
    return ledger, adapter, identity, plan, work


def test_six_group_store_and_new_owner_restore_are_exact_and_not_hot_early() -> None:
    backend = _MemoryBytePlane()
    source_ledger, _, source_identity, source_plan, store = _begin_source_store(backend)
    result = store.execute_after_staging()

    assert result.receipt.success
    assert (
        source_ledger.take_reclaimable(source_identity) == source_plan.device_block_ids
    )
    assert sum(len(payload) for payload in store.payloads) == 349_412

    destination = _snapshot(epoch=5, owner_rank=6, generation=12, first_block=31)
    destination_ledger = RequestOwnedBulkOffloadLedger(destination.owner_rank)
    destination_adapter = StateHarborBytePlaneAdapter(
        ledger=destination_ledger, backend=backend
    )
    destination_identity = destination_ledger.bind(destination, active=False)
    restore_plan = OwnerOffloadPlan.from_snapshot(
        destination,
        source_plan.offload_keys,
        logical_block_indices=source_plan.logical_block_indices,
    )

    restore = destination_adapter.prepare_restore(
        plan=restore_plan, manifest=result.manifest
    )
    assert restore.payloads == store.payloads
    assert not destination_ledger.is_hot(destination_identity)
    with pytest.raises(RequestOwnedOffloadError, match="restore completion"):
        destination_ledger.activate(destination_identity)

    receipt = restore.complete_after_h2d(success=True)
    assert receipt.success
    assert destination_ledger.is_hot(destination_identity)
    destination_ledger.activate(destination_identity)


def test_unknown_commit_is_reconciled_exactly_and_unknown_miss_fails_closed() -> None:
    backend = _MemoryBytePlane()
    backend.unknown_publish_sequences = {2}
    backend.raise_after_publish_sequences = {6}
    ledger, _, identity, plan, store = _begin_source_store(backend)
    result = store.execute_after_staging()

    assert result.receipt.success
    assert len(result.reconciled_object_keys) == 2
    assert ledger.take_reclaimable(identity) == plan.device_block_ids

    failing_backend = _MemoryBytePlane()
    failing_backend.unknown_miss_sequences = {3}
    ledger, _, identity, _, store = _begin_source_store(failing_backend)
    failed = store.execute_after_staging()
    assert not failed.receipt.success
    assert failed.residual_object_keys == ()
    assert failing_backend.objects == {}
    with pytest.raises(RequestOwnedOffloadError, match="durable store"):
        ledger.take_reclaimable(identity)

    exception_backend = _MemoryBytePlane()
    exception_backend.raise_without_publish_sequences = {3}
    ledger, _, identity, _, store = _begin_source_store(exception_backend)
    failed = store.execute_after_staging()
    assert not failed.receipt.success
    assert exception_backend.objects == {}
    with pytest.raises(RequestOwnedOffloadError, match="durable store"):
        ledger.take_reclaimable(identity)


def test_partial_failure_aborts_pending_and_never_publishes_manifest() -> None:
    backend = _MemoryBytePlane()
    backend.fail_write_sequences = {3}
    ledger, _, identity, _, store = _begin_source_store(backend)

    result = store.execute_after_staging()

    assert not result.receipt.success
    assert len(backend.aborted) == 1
    assert backend.objects == {}
    assert backend.read(result.manifest.object_key) is None
    with pytest.raises(RequestOwnedOffloadError, match="durable store"):
        ledger.take_reclaimable(identity)


def test_source_destination_layout_and_payload_corruption_fail_closed() -> None:
    backend = _MemoryBytePlane()
    source = _snapshot()
    ledger = RequestOwnedBulkOffloadLedger(source.owner_rank)
    adapter = StateHarborBytePlaneAdapter(ledger=ledger, backend=backend)
    identity = ledger.bind(source, active=True)
    ledger.retire(identity)
    plan = _plan(source)

    with pytest.raises(StateHarborBytePlaneError, match="source identity"):
        adapter.begin_store(
            plan=plan,
            source=replace(_source(source), source_owner_rank=4),
            groups=_payloads(),
            writer=_writer(),
        )

    store = adapter.begin_store(
        plan=plan,
        source=_source(source),
        groups=_payloads(),
        writer=_writer(),
    )
    result = store.execute_after_staging()
    assert result.receipt.success

    wrong_request = replace(
        _snapshot(epoch=5, owner_rank=6, generation=12, first_block=31),
        key=OwnerLeaseKey("other-request", 5),
    )
    wrong_ledger = RequestOwnedBulkOffloadLedger(wrong_request.owner_rank)
    wrong_adapter = StateHarborBytePlaneAdapter(ledger=wrong_ledger, backend=backend)
    wrong_ledger.bind(wrong_request, active=False)
    wrong_plan = OwnerOffloadPlan.from_snapshot(
        wrong_request,
        plan.offload_keys,
        logical_block_indices=plan.logical_block_indices,
    )
    with pytest.raises(StateHarborBytePlaneError, match="destination request"):
        wrong_adapter.prepare_restore(plan=wrong_plan, manifest=result.manifest)

    destination = _snapshot(epoch=5, owner_rank=6, generation=12, first_block=31)
    destination_ledger = RequestOwnedBulkOffloadLedger(destination.owner_rank)
    destination_adapter = StateHarborBytePlaneAdapter(
        ledger=destination_ledger, backend=backend
    )
    destination_ledger.bind(destination, active=False)
    restore_plan = OwnerOffloadPlan.from_snapshot(
        destination,
        plan.offload_keys,
        logical_block_indices=plan.logical_block_indices,
    )
    wrong_source = replace(
        result.manifest.source,
        layout_fingerprint=b"x" * 32,
    )
    wrong_layout_manifest = replace(
        result.manifest,
        source=wrong_source,
        groups=tuple(
            replace(group, source=wrong_source) for group in result.manifest.groups
        ),
    )
    with pytest.raises(StateHarborBytePlaneError, match="manifest is missing"):
        destination_adapter.prepare_restore(
            plan=restore_plan,
            manifest=wrong_layout_manifest,
        )

    first_group = result.manifest.groups[0]
    backend.objects[first_group.object_key] += b"corrupt"
    with pytest.raises(StateHarborBytePlaneError, match="length/digest"):
        destination_adapter.prepare_restore(plan=restore_plan, manifest=result.manifest)


def test_duplicate_terminals_and_exact_generation_release_are_fenced() -> None:
    backend = _MemoryBytePlane()
    _, adapter1, _, plan1, store1 = _begin_source_store(backend)
    result1 = store1.execute_after_staging()
    with pytest.raises(StateHarborBytePlaneError, match="already executed"):
        store1.execute_after_staging()

    snapshot2 = _snapshot(generation=8, first_block=21)
    _, adapter2, _, _, store2 = _begin_source_store(
        backend, snapshot=snapshot2, writer=_writer(2)
    )
    result2 = store2.execute_after_staging()
    assert result1.manifest.object_key != result2.manifest.object_key

    with pytest.raises(StateHarborBytePlaneError, match="exact source generation"):
        adapter1.release(
            authority=result1.manifest.source,
            plan=plan1,
            manifest=result2.manifest,
        )
    assert backend.read(result2.manifest.object_key) == (
        result2.manifest.canonical_bytes()
    )

    release = adapter1.release(
        authority=result1.manifest.source,
        plan=plan1,
        manifest=result1.manifest,
    )
    assert release.manifest_key == result1.manifest.object_key
    assert (
        backend.read(result2.manifest.object_key) == result2.manifest.canonical_bytes()
    )
    assert all(
        backend.read(group.object_key) is not None for group in result2.manifest.groups
    )
    with pytest.raises(StateHarborBytePlaneError, match="already released"):
        adapter1.release(
            authority=result1.manifest.source,
            plan=plan1,
            manifest=result1.manifest,
        )

    # The second generation remains independently releasable.
    adapter2.release(
        authority=result2.manifest.source,
        manifest=result2.manifest,
    )


def test_restore_duplicate_terminal_and_failed_h2d_never_publish_hot() -> None:
    backend = _MemoryBytePlane()
    _, _, _, source_plan, store = _begin_source_store(backend)
    result = store.execute_after_staging()
    destination = _snapshot(epoch=5, owner_rank=6, generation=12, first_block=31)
    ledger = RequestOwnedBulkOffloadLedger(destination.owner_rank)
    adapter = StateHarborBytePlaneAdapter(ledger=ledger, backend=backend)
    destination_identity = ledger.bind(destination, active=False)
    plan = OwnerOffloadPlan.from_snapshot(
        destination,
        source_plan.offload_keys,
        logical_block_indices=source_plan.logical_block_indices,
    )
    restore = adapter.prepare_restore(plan=plan, manifest=result.manifest)

    receipt = restore.complete_after_h2d(success=False, error="copy failed")
    assert not receipt.success
    assert not ledger.is_hot(destination_identity)
    with pytest.raises(StateHarborBytePlaneError, match="already completed"):
        restore.complete_after_h2d(success=True)


def test_reaped_writer_cannot_publish_and_registration_generation_changes_id() -> None:
    backend = _MemoryBytePlane()
    old_writer = _writer(7)
    new_writer = _writer(8)
    assert derive_lmcache_instance_id(old_writer, b"group-0") != (
        derive_lmcache_instance_id(new_writer, b"group-0")
    )
    assert derive_lmcache_instance_id(old_writer, b"group-0") != (
        derive_lmcache_instance_id(old_writer, b"group-1")
    )

    lease = StateHarborWriteLease(b"pending-object", old_writer, 99)
    assert backend.prepare_write(lease, 7)
    backend.write(lease, b"partial")
    backend.reap(old_writer)
    assert backend.read(lease.object_key) is None
    with pytest.raises(RuntimeError, match="reaped"):
        backend.commit_write(lease)

    ledger, _, identity, plan, store = _begin_source_store(backend, writer=new_writer)
    result = store.execute_after_staging()
    assert result.receipt.success
    assert ledger.take_reclaimable(identity) == plan.device_block_ids


def test_new_owner_redock_holds_reserve_until_exact_h2d_terminal() -> None:
    backend = _MemoryBytePlane()
    source_ledger, _, source_identity, source_plan, store = _begin_source_store(backend)
    result = store.execute_after_staging()
    assert source_ledger.take_reclaimable(source_identity) == (
        source_plan.device_block_ids
    )

    coordinator = OwnerLeaseCoordinator()
    coordinator.observe(OwnerAssignmentObservation(owner_id=3, observation_seq=1))
    coordinator.observe(OwnerAssignmentObservation(owner_id=6, observation_seq=1))
    source_worker = AttentionLeaseManager(owner_rank=3, capacity=64)
    destination_worker = AttentionLeaseManager(owner_rank=6, capacity=64)

    source_key = OwnerLeaseKey("stateharbor-request", 4)
    assert coordinator.assign(source_key, explicit_owner=3) == 3
    source_reserve = source_worker.apply(coordinator.reserve(source_key, 8))
    assert coordinator.apply_receipt(source_reserve)
    published = coordinator.publish(1, {source_key})
    assert len(published) == 1
    source_worker.record_published(published[0])

    source_preempt = source_worker.apply(coordinator.preempt(source_key))
    assert coordinator.apply_receipt(source_preempt)
    source_release = source_worker.apply(coordinator.finish(source_key))
    assert source_release.released
    assert coordinator.apply_receipt(source_release)

    destination_key = OwnerLeaseKey("stateharbor-request", 5)
    assert coordinator.assign(destination_key, explicit_owner=6) == 6
    buffered_reserve = destination_worker.apply(coordinator.reserve(destination_key, 8))
    assert buffered_reserve.accepted
    assert coordinator.runnable_num_tokens_of(destination_key) is None
    assert coordinator.publish(2) == []

    destination = _snapshot(epoch=5, owner_rank=6, generation=12, first_block=31)
    destination_ledger = RequestOwnedBulkOffloadLedger(destination.owner_rank)
    destination_adapter = StateHarborBytePlaneAdapter(
        ledger=destination_ledger, backend=backend
    )
    destination_identity = destination_ledger.bind(destination, active=False)
    restore_plan = OwnerOffloadPlan.from_snapshot(
        destination,
        source_plan.offload_keys,
        logical_block_indices=source_plan.logical_block_indices,
    )
    ticket = StateHarborRedockTicket(
        source=result.manifest.source,
        source_manifest_key=result.manifest.object_key,
        source_release_receipt=source_release,
        destination_identity=destination_identity,
        destination_reserve_receipt=buffered_reserve,
    )
    redock = destination_adapter.prepare_redock(
        ticket=ticket,
        plan=restore_plan,
        manifest=result.manifest,
    )
    assert redock.payloads == tuple(payload.payload for payload in _payloads())
    assert not destination_ledger.is_hot(destination_identity)
    assert coordinator.runnable_num_tokens_of(destination_key) is None

    terminal = redock.complete_after_h2d(success=True)
    assert destination_ledger.is_hot(destination_identity)
    assert coordinator.runnable_num_tokens_of(destination_key) is None

    assert coordinator.apply_receipt(terminal.admit_destination_reserve())
    destination_ledger.activate(destination_identity)
    published = coordinator.publish(3, {destination_key})
    assert len(published) == 1
    assert published[0].key == destination_key
    assert published[0].owner_id == 6
    with pytest.raises(StateHarborBytePlaneError, match="already admitted"):
        terminal.admit_destination_reserve()

    with pytest.raises(StateHarborBytePlaneError, match="advance the owner epoch"):
        StateHarborRedockTicket(
            source=result.manifest.source,
            source_manifest_key=result.manifest.object_key,
            source_release_receipt=source_release,
            destination_identity=replace(
                destination_identity,
                key=source_key,
            ),
            destination_reserve_receipt=replace(
                buffered_reserve,
                key=source_key,
            ),
        )


def test_failed_redock_h2d_never_releases_buffered_reserve() -> None:
    backend = _MemoryBytePlane()
    _, _, _, source_plan, store = _begin_source_store(backend)
    result = store.execute_after_staging()

    destination = _snapshot(epoch=5, owner_rank=6, generation=12, first_block=31)
    destination_ledger = RequestOwnedBulkOffloadLedger(destination.owner_rank)
    destination_adapter = StateHarborBytePlaneAdapter(
        ledger=destination_ledger, backend=backend
    )
    destination_identity = destination_ledger.bind(destination, active=False)
    restore_plan = OwnerOffloadPlan.from_snapshot(
        destination,
        source_plan.offload_keys,
        logical_block_indices=source_plan.logical_block_indices,
    )
    source_release = OwnerReceipt(
        key=OwnerLeaseKey("stateharbor-request", 4),
        owner_id=3,
        command_seq=9,
        accepted=True,
        runnable_num_tokens=8,
        released=True,
        pending_dma=0,
    )
    buffered_reserve = OwnerReceipt(
        key=destination.key,
        owner_id=destination.owner_rank,
        command_seq=1,
        accepted=True,
        runnable_num_tokens=8,
    )
    ticket = StateHarborRedockTicket(
        source=result.manifest.source,
        source_manifest_key=result.manifest.object_key,
        source_release_receipt=source_release,
        destination_identity=destination_identity,
        destination_reserve_receipt=buffered_reserve,
    )
    redock = destination_adapter.prepare_redock(
        ticket=ticket,
        plan=restore_plan,
        manifest=result.manifest,
    )
    terminal = redock.complete_after_h2d(success=False, error="copy failed")
    assert not destination_ledger.is_hot(destination_identity)
    with pytest.raises(StateHarborBytePlaneError, match="failed redock H2D"):
        terminal.admit_destination_reserve()
