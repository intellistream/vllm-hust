# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
import torch

from vllm.v1.core.sched.ownership import OwnerLeaseKey
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    OffloadKey,
    TransferResult,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.worker.request_owned_kv import RequestOwnedKVSnapshot
from vllm.v1.worker.request_owned_offload import (
    OwnerBulkTransferDirection,
    OwnerBulkTransferReceipt,
    OwnerOffloadIdentity,
    OwnerOffloadPlan,
    RequestOwnedBulkOffloadAdapter,
    RequestOwnedBulkOffloadLedger,
    RequestOwnedBulkRestoreWork,
    RequestOwnedOffloadError,
    make_request_owned_offload_keys,
)


class _FakeOffloadingWorker(OffloadingWorker):
    def __init__(self) -> None:
        self.store_submissions: list[tuple[int, GPULoadStoreSpec, LoadStoreSpec]] = []
        self.load_submissions: list[tuple[int, LoadStoreSpec, GPULoadStoreSpec]] = []
        self.finished: list[TransferResult] = []
        self.waited: list[set[int]] = []

    def submit_store(
        self,
        job_id: int,
        src_spec: GPULoadStoreSpec,
        dst_spec: LoadStoreSpec,
    ) -> bool:
        self.store_submissions.append((job_id, src_spec, dst_spec))
        return True

    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
    ) -> bool:
        self.load_submissions.append((job_id, src_spec, dst_spec))
        return True

    def get_finished(self) -> list[TransferResult]:
        finished, self.finished = self.finished, []
        return finished

    def wait(self, job_ids: set[int]) -> None:
        self.waited.append(job_ids)

    def finish(self, job_id: int, *, success: bool = True) -> None:
        self.finished.append(TransferResult(job_id=job_id, success=success))


class _TensorOffloadingWorker(_FakeOffloadingWorker):
    """Synchronous CPU tensor oracle for exact bulk-copy direction/layout."""

    def __init__(self, device: torch.Tensor, num_host_blocks: int) -> None:
        super().__init__()
        self.device = device
        self.host = torch.zeros((num_host_blocks, device.shape[1]), dtype=device.dtype)

    def submit_store(
        self,
        job_id: int,
        src_spec: GPULoadStoreSpec,
        dst_spec: LoadStoreSpec,
    ) -> bool:
        assert isinstance(dst_spec, CPULoadStoreSpec)
        self.host[dst_spec.block_ids.tolist()] = self.device[
            src_spec.block_ids.tolist()
        ]
        self.finish(job_id)
        return True

    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
    ) -> bool:
        assert isinstance(src_spec, CPULoadStoreSpec)
        self.device[dst_spec.block_ids.tolist()] = self.host[
            src_spec.block_ids.tolist()
        ]
        self.finish(job_id)
        return True


def _snapshot(
    *,
    request_id: str = "req",
    epoch: int = 0,
    owner_rank: int = 3,
    generation: int = 1,
    tables: tuple[tuple[int, ...], ...] = ((1, 2), (3, 4)),
) -> RequestOwnedKVSnapshot:
    return RequestOwnedKVSnapshot(
        key=OwnerLeaseKey(request_id, epoch),
        owner_rank=owner_rank,
        allocation_generation=generation,
        num_computed_tokens=8,
        reserved_num_tokens=16,
        pending_free=False,
        tables=tables,
    )


def _keys() -> tuple[tuple[OffloadKey, ...], ...]:
    return (
        (make_offload_key(b"hash-a", 0), make_offload_key(b"hash-b", 0)),
        (make_offload_key(b"hash-a", 1), make_offload_key(b"hash-b", 1)),
    )


def test_bulk_store_is_durable_before_reclaim_and_restore_before_active() -> None:
    ledger = RequestOwnedBulkOffloadLedger(owner_rank=3)
    source = _snapshot()
    source_id = ledger.bind(source, active=True)
    source_plan = OwnerOffloadPlan.from_snapshot(source, _keys())

    with pytest.raises(RequestOwnedOffloadError, match="ACTIVE"):
        ledger.begin_store(source_plan)

    ledger.retire(source_id)
    store_job = ledger.begin_store(source_plan)
    assert store_job.direction is OwnerBulkTransferDirection.STORE
    assert not ledger.is_host_durable(source_plan)
    with pytest.raises(RequestOwnedOffloadError, match="durable store"):
        ledger.take_reclaimable(source_id)

    ledger.complete(OwnerBulkTransferReceipt.for_job(store_job, success=True))
    assert ledger.is_host_durable(source_plan)
    assert ledger.take_reclaimable(source_id) == source_plan.device_block_ids
    with pytest.raises(RequestOwnedOffloadError, match="already consumed"):
        ledger.take_reclaimable(source_id)
    with pytest.raises(RequestOwnedOffloadError, match="duplicate"):
        ledger.complete(OwnerBulkTransferReceipt.for_job(store_job, success=True))

    # Same lease/epoch, new physical generation and exact new destination.
    destination = _snapshot(generation=2, tables=((5, 6), (7, 8)))
    destination_id = ledger.bind(destination, active=False)
    destination_plan = OwnerOffloadPlan.from_snapshot(destination, _keys())
    restore_job = ledger.begin_restore(destination_plan)
    assert restore_job.direction is OwnerBulkTransferDirection.RESTORE
    assert not ledger.is_hot(destination_id)
    with pytest.raises(RequestOwnedOffloadError, match="restore completion"):
        ledger.activate(destination_id)

    ledger.complete(OwnerBulkTransferReceipt.for_job(restore_job, success=True))
    assert ledger.is_hot(destination_id)
    ledger.activate(destination_id)


def test_receipt_must_match_owner_epoch_generation_and_destination_exactly() -> None:
    ledger = RequestOwnedBulkOffloadLedger(owner_rank=3)
    snapshot = _snapshot()
    identity = ledger.bind(snapshot, active=True)
    ledger.retire(identity)
    plan = OwnerOffloadPlan.from_snapshot(snapshot, _keys())
    job = ledger.begin_store(plan)
    receipt = OwnerBulkTransferReceipt.for_job(job, success=True)

    wrong_generation = replace(
        receipt,
        identity=replace(receipt.identity, allocation_generation=2),
    )
    with pytest.raises(RequestOwnedOffloadError, match="does not match"):
        ledger.complete(wrong_generation)

    wrong_destination = replace(
        receipt,
        device_block_ids=((1, 2), (3, 9)),
    )
    with pytest.raises(RequestOwnedOffloadError, match="does not match"):
        ledger.complete(wrong_destination)

    assert ledger.pending_jobs == (job,)
    ledger.complete(receipt)


def test_failed_store_never_becomes_durable_or_reclaimable() -> None:
    ledger = RequestOwnedBulkOffloadLedger(owner_rank=3)
    snapshot = _snapshot()
    identity = ledger.bind(snapshot, active=True)
    ledger.retire(identity)
    plan = OwnerOffloadPlan.from_snapshot(snapshot, _keys())
    job = ledger.begin_store(plan)

    ledger.complete(
        OwnerBulkTransferReceipt.for_job(
            job,
            success=False,
            error="copy failed",
        )
    )
    assert not ledger.is_host_durable(plan)
    with pytest.raises(RequestOwnedOffloadError, match="durable store"):
        ledger.take_reclaimable(identity)


def test_abort_invalidates_completion_and_new_generation_is_aba_safe() -> None:
    ledger = RequestOwnedBulkOffloadLedger(owner_rank=3)
    old = _snapshot()
    old_id = ledger.bind(old, active=True)
    ledger.retire(old_id)
    old_job = ledger.begin_store(OwnerOffloadPlan.from_snapshot(old, _keys()))

    ledger.abort(old_id)
    new = _snapshot(generation=2, tables=((5, 6), (7, 8)))
    ledger.bind(new, active=False)
    with pytest.raises(RequestOwnedOffloadError, match="unknown, stale, or duplicate"):
        ledger.complete(OwnerBulkTransferReceipt.for_job(old_job, success=True))


def test_epoch_and_owner_fences_fail_closed() -> None:
    ledger = RequestOwnedBulkOffloadLedger(owner_rank=3)
    old = _snapshot(epoch=0)
    old_id = ledger.bind(old, active=True)
    ledger.retire(old_id)
    ledger.abort(old_id)

    new_epoch = _snapshot(epoch=1, generation=1, tables=((5, 6), (7, 8)))
    ledger.bind(new_epoch, active=False)
    with pytest.raises(RequestOwnedOffloadError, match="stale request epoch"):
        ledger.bind(old, active=False)

    with pytest.raises(RequestOwnedOffloadError, match="wrong owner"):
        ledger.bind(_snapshot(request_id="other", owner_rank=4), active=False)


def test_plan_rejects_non_exact_geometry_and_null_or_duplicate_blocks() -> None:
    snapshot = _snapshot()
    with pytest.raises(ValueError, match="same groups"):
        OwnerOffloadPlan.from_snapshot(snapshot, (_keys()[0],))
    with pytest.raises(ValueError, match="only 2 blocks"):
        OwnerOffloadPlan.from_snapshot(
            snapshot,
            (_keys()[0] + (make_offload_key(b"hash-c", 0),), _keys()[1]),
        )

    identity = OwnerOffloadIdentity.from_snapshot(snapshot)
    with pytest.raises(TypeError, match="null block"):
        OwnerOffloadPlan(
            identity=identity,
            device_block_ids=((0,), (3,)),
            offload_keys=((make_offload_key(b"a", 0),), (make_offload_key(b"b", 1),)),
        )
    with pytest.raises(ValueError, match="multiple plan positions"):
        OwnerOffloadPlan(
            identity=identity,
            device_block_ids=((1,), (1,)),
            offload_keys=((make_offload_key(b"a", 0),), (make_offload_key(b"b", 1),)),
        )

    with pytest.raises(ValueError, match="encodes group"):
        OwnerOffloadPlan(
            identity=identity,
            device_block_ids=((1,), (3,)),
            offload_keys=((make_offload_key(b"a", 1),), (make_offload_key(b"b", 1),)),
        )


def test_owner_host_keys_survive_generation_but_fence_partial_extension() -> None:
    partial = _snapshot(
        generation=1,
        tables=((1, 2), (3, 4)),
    )
    keys = make_request_owned_offload_keys(partial, (4, 4))
    same_logical_bytes = make_request_owned_offload_keys(
        replace(partial, allocation_generation=2, tables=((5, 6), (7, 8))),
        (4, 4),
    )
    assert keys == same_logical_bytes

    extended = replace(partial, num_computed_tokens=12)
    extended_keys = make_request_owned_offload_keys(extended, (4, 4))
    assert extended_keys[0][0] == keys[0][0]
    assert extended_keys[0][1] == keys[0][1]

    short_partial = replace(partial, num_computed_tokens=6)
    short_keys = make_request_owned_offload_keys(short_partial, (4, 4))
    assert short_keys[0][0] == keys[0][0]
    assert short_keys[0][1] != keys[0][1]
    assert (
        make_request_owned_offload_keys(replace(partial, owner_rank=4), (4, 4)) != keys
    )


def test_adapter_reuses_manager_worker_for_exact_store_and_bulk_restore() -> None:
    manager = CPUOffloadingManager(num_blocks=8)
    worker = _FakeOffloadingWorker()
    adapter = RequestOwnedBulkOffloadAdapter(
        owner_rank=3,
        manager=manager,
        worker=worker,
    )
    source = _snapshot()
    source_id = adapter.bind(source, active=True)
    adapter.retire(source_id)
    source_plan = OwnerOffloadPlan.from_snapshot(source, _keys())

    store_job = adapter.submit_store(source_plan)
    assert adapter.poll() == ()
    assert len(worker.store_submissions) == 1
    _, gpu_src, _ = worker.store_submissions[0]
    assert gpu_src.block_ids.tolist() == [1, 2, 3, 4]
    assert list(gpu_src.group_sizes) == [2, 2]
    assert list(gpu_src.block_indices) == [0, 0]
    with pytest.raises(RequestOwnedOffloadError, match="durable store"):
        adapter.take_reclaimable(source_id)

    worker.finish(store_job.job_id)
    (store_receipt,) = adapter.poll()
    assert store_receipt.success
    assert adapter.take_reclaimable(source_id) == ((1, 2), (3, 4))

    destination = _snapshot(generation=2, tables=((5, 6), (7, 8)))
    destination_id = adapter.bind(destination, active=False)
    restore_plan = OwnerOffloadPlan.from_snapshot(destination, _keys())
    restore_job = adapter.submit_restore(restore_plan)
    assert len(worker.load_submissions) == 1
    _, _, gpu_dst = worker.load_submissions[0]
    assert gpu_dst.block_ids.tolist() == [5, 6, 7, 8]
    assert list(gpu_dst.group_sizes) == [2, 2]
    with pytest.raises(RequestOwnedOffloadError, match="restore completion"):
        adapter.activate(destination_id)

    worker.finish(restore_job.job_id)
    (restore_receipt,) = adapter.poll()
    assert restore_receipt.success
    adapter.activate(destination_id)


def test_adapter_bulk_restore_is_byte_exact_at_the_named_destination() -> None:
    device = torch.arange(10 * 8, dtype=torch.int16).view(10, 8)
    original = device[[1, 2, 3, 4]].clone()
    manager = CPUOffloadingManager(num_blocks=8)
    worker = _TensorOffloadingWorker(device, num_host_blocks=8)
    adapter = RequestOwnedBulkOffloadAdapter(
        owner_rank=3,
        manager=manager,
        worker=worker,
    )
    source = _snapshot()
    source_id = adapter.bind(source, active=True)
    adapter.retire(source_id)
    source_plan = OwnerOffloadPlan.from_snapshot(source, _keys())
    store_job = adapter.submit_store(source_plan)
    adapter.wait((store_job,))
    (store_receipt,) = adapter.poll()
    assert store_receipt.success
    adapter.take_reclaimable(source_id)

    device[[1, 2, 3, 4, 5, 6, 7, 8]] = -1
    destination = _snapshot(generation=2, tables=((5, 6), (7, 8)))
    destination_id = adapter.bind(destination, active=False)
    restore_plan = OwnerOffloadPlan.from_snapshot(destination, _keys())
    restore_job = adapter.submit_restore(restore_plan)
    adapter.wait((restore_job,))
    (restore_receipt,) = adapter.poll()
    assert restore_receipt.success
    assert torch.equal(device[[5, 6, 7, 8]], original)
    assert torch.equal(device[[1, 2, 3, 4]], torch.full_like(original, -1))
    adapter.activate(destination_id)
    assert adapter.poll() == ()


def test_restore_work_is_post_zero_completion_and_replay_fenced() -> None:
    device = torch.arange(10 * 8, dtype=torch.int16).view(10, 8)
    manager = CPUOffloadingManager(num_blocks=8)
    worker = _TensorOffloadingWorker(device, num_host_blocks=8)
    adapter = RequestOwnedBulkOffloadAdapter(
        owner_rank=3,
        manager=manager,
        worker=worker,
    )
    source = _snapshot()
    source_id = adapter.bind(source, active=True)
    adapter.retire(source_id)
    plan = OwnerOffloadPlan.from_snapshot(source, _keys())
    store_job = adapter.submit_store(plan)
    adapter.wait((store_job,))
    adapter.poll()
    adapter.take_reclaimable(source_id)

    destination = _snapshot(generation=2, tables=((5, 6), (7, 8)))
    adapter.bind(destination, active=False)
    restore_plan = OwnerOffloadPlan.from_snapshot(destination, _keys())
    work = RequestOwnedBulkRestoreWork(
        step_seq=9,
        adapter=adapter,
        plan=restore_plan,
        zero_block_ids=((5, 6, 9), (7, 8)),
    )
    receipt = work.execute_after_zero()
    assert receipt.success
    with pytest.raises(RequestOwnedOffloadError, match="already executed"):
        work.execute_after_zero()


def test_adapter_failure_or_abort_never_reclaims_or_publishes_durable() -> None:
    manager = CPUOffloadingManager(num_blocks=8)
    worker = _FakeOffloadingWorker()
    adapter = RequestOwnedBulkOffloadAdapter(
        owner_rank=3,
        manager=manager,
        worker=worker,
    )
    source = _snapshot()
    source_id = adapter.bind(source, active=True)
    adapter.retire(source_id)
    plan = OwnerOffloadPlan.from_snapshot(source, _keys())
    failed_job = adapter.submit_store(plan)
    worker.finish(failed_job.job_id, success=False)
    (failed_receipt,) = adapter.poll()
    assert not failed_receipt.success
    with pytest.raises(RequestOwnedOffloadError, match="durable store"):
        adapter.take_reclaimable(source_id)

    retry_job = adapter.submit_store(plan)
    adapter.abort(source_id)
    worker.finish(retry_job.job_id)
    (late_receipt,) = adapter.poll()
    assert not late_receipt.success
    assert late_receipt.error is not None
    assert "aborted" in late_receipt.error

    replacement = _snapshot(generation=2, tables=((5, 6), (7, 8)))
    adapter.bind(replacement, active=False)


def test_adapter_host_capacity_rejection_is_an_immediate_failed_receipt() -> None:
    manager = CPUOffloadingManager(num_blocks=1)
    worker = _FakeOffloadingWorker()
    adapter = RequestOwnedBulkOffloadAdapter(
        owner_rank=3,
        manager=manager,
        worker=worker,
    )
    source = _snapshot()
    source_id = adapter.bind(source, active=True)
    adapter.retire(source_id)
    job = adapter.submit_store(OwnerOffloadPlan.from_snapshot(source, _keys()))

    (receipt,) = adapter.poll()
    assert receipt.job_id == job.job_id
    assert not receipt.success
    assert receipt.error == "host tier has no store capacity"
    assert worker.store_submissions == []


def test_adapter_release_closes_bound_generation_for_next_epoch() -> None:
    adapter = RequestOwnedBulkOffloadAdapter(
        owner_rank=3,
        manager=CPUOffloadingManager(num_blocks=8),
        worker=_FakeOffloadingWorker(),
    )
    source = _snapshot()
    identity = adapter.bind(source, active=True)
    assert adapter.ledger.is_current(identity)
    adapter.release(source)

    replacement = _snapshot(request_id="req", epoch=1, generation=2)
    replacement_id = adapter.bind(replacement, active=True)
    assert adapter.ledger.is_current(replacement_id)


def test_adapter_release_evicts_exact_durable_host_image() -> None:
    manager = CPUOffloadingManager(num_blocks=8)
    worker = _FakeOffloadingWorker()
    adapter = RequestOwnedBulkOffloadAdapter(
        owner_rank=3,
        manager=manager,
        worker=worker,
    )
    source = _snapshot()
    source_id = adapter.bind(source, active=True)
    adapter.retire(source_id)
    plan = OwnerOffloadPlan.from_snapshot(source, _keys())
    job = adapter.submit_store(plan)
    worker.finish(job.job_id)
    adapter.poll()
    adapter.take_reclaimable(source_id)
    assert adapter.ledger.is_host_durable(plan)

    adapter.evict_owned_host_keys(source.key)
    assert not adapter.ledger.is_host_durable(plan)


def test_adapter_new_store_retires_stale_partial_tail_image() -> None:
    manager = CPUOffloadingManager(num_blocks=8)
    worker = _FakeOffloadingWorker()
    adapter = RequestOwnedBulkOffloadAdapter(
        owner_rank=3,
        manager=manager,
        worker=worker,
    )
    partial = replace(_snapshot(), num_computed_tokens=6)
    partial_id = adapter.bind(partial, active=True)
    adapter.retire(partial_id)
    partial_plan = OwnerOffloadPlan.from_snapshot(
        partial,
        make_request_owned_offload_keys(partial, (4, 4)),
    )
    first = adapter.submit_store(partial_plan)
    worker.finish(first.job_id)
    adapter.poll()
    adapter.take_reclaimable(partial_id)

    extended = replace(
        partial,
        allocation_generation=2,
        num_computed_tokens=8,
        tables=((5, 6), (7, 8)),
    )
    extended_id = adapter.bind(extended, active=True)
    adapter.retire(extended_id)
    extended_plan = OwnerOffloadPlan.from_snapshot(
        extended,
        make_request_owned_offload_keys(extended, (4, 4)),
    )
    second = adapter.submit_store(extended_plan)
    worker.finish(second.job_id)
    adapter.poll()
    adapter.take_reclaimable(extended_id)

    assert adapter.ledger.is_host_durable(extended_plan)
    assert adapter._owned_host_keys[extended.key] == {
        key for group in extended_plan.offload_keys for key in group
    }
    assert manager.resident_blocks == 4
    adapter.evict_owned_host_keys(extended.key)
    assert manager.resident_blocks == 0
