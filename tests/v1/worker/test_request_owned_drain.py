# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.ownership import (
    AttentionLeaseManager,
    OwnerCachePoolSnapshot,
    OwnerCommand,
    OwnerCommandKind,
    OwnerLeaseKey,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.request_owned_kv import (
    DeferredFreeResult,
    RequestOwnedKVSnapshot,
    RequestOwnedStepMarkResult,
    RequestOwnedStepMetadata,
)
from vllm.v1.worker.request_owned_offload import (
    RequestOwnedBulkOffloadAdapter,
    RequestOwnedOffloadError,
)
from vllm.v1.worker.worker_base import WorkerWrapperBase

pytestmark = pytest.mark.cpu_test


class _DeferredOffloadingWorker(OffloadingWorker):
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.pending: set[int] = set()
        self.finished: list[TransferResult] = []
        self.waited: list[set[int]] = []

    def submit_store(
        self,
        job_id: int,
        src_spec: GPULoadStoreSpec,
        dst_spec: LoadStoreSpec,
    ) -> bool:
        self.order.append("submit_d2h")
        self.pending.add(job_id)
        return True

    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
    ) -> bool:
        raise AssertionError("H2D is outside the background drain tests")

    def get_finished(self) -> list[TransferResult]:
        finished, self.finished = self.finished, []
        return finished

    def wait(self, job_ids: set[int]) -> None:
        self.waited.append(set(job_ids))
        for job_id in job_ids:
            if job_id in self.pending:
                self.finish(job_id)

    def finish(self, job_id: int, *, success: bool = True) -> None:
        assert job_id in self.pending
        self.pending.remove(job_id)
        self.finished.append(TransferResult(job_id=job_id, success=success))


class _ModelWorker:
    def __init__(self, order: list[str], *, fail: bool = False) -> None:
        self.order = order
        self.fail = fail
        self.calls = 0
        self.metadata_handoffs: list[RequestOwnedStepMetadata | None] = []

    def execute_model(self, scheduler_output):
        self.calls += 1
        self.order.append("model")
        if self.fail:
            raise RuntimeError("model step failed")
        return ModelRunnerOutput(req_ids=[], req_id_to_index={})

    def set_request_owned_step_metadata(self, metadata) -> None:
        self.metadata_handoffs.append(metadata)


class _DrainStore:
    group_block_sizes = (4,)

    def __init__(self, key: OwnerLeaseKey) -> None:
        self.key = key
        self.source = RequestOwnedKVSnapshot(
            key=key,
            owner_rank=0,
            allocation_generation=1,
            num_computed_tokens=8,
            reserved_num_tokens=8,
            pending_free=False,
            tables=((1, 2),),
        )
        self.pending_free = False
        self.calls: list[str] = []

    def computed_prefix_snapshot(self, key):
        return self.source if key == self.key else None

    def snapshot(self, key):
        return self.source if key == self.key else None

    def preempt(self, command):
        self.calls.append("preempt")
        if command.key != self.key or self.source is None:
            return DeferredFreeResult(
                accepted=False,
                key=command.key,
                error="source generation disappeared",
            )
        self.pending_free = True
        return DeferredFreeResult(accepted=True, key=command.key, deferred=True)

    def release(self, command):
        self.calls.append("release")
        if command.key != self.key:
            return DeferredFreeResult(
                accepted=False,
                key=command.key,
                error="wrong release key",
            )
        self.pending_free = self.source is not None
        return DeferredFreeResult(
            accepted=True,
            key=command.key,
            deferred=self.pending_free,
        )

    def flush(self):
        self.calls.append("flush")
        if not self.pending_free:
            return ()
        self.pending_free = False
        self.source = None
        return (self.key,)

    def build_step_metadata(
        self,
        step_seq,
        tokens,
        request_token_counts,
        scheduled_spec_decode_tokens,
    ):
        self.calls.append("build")
        return SimpleNamespace(
            accepted=True,
            metadata=RequestOwnedStepMetadata(
                step_seq=step_seq,
                owner_rank=0,
                entries=(),
            ),
            error=None,
        )

    def mark_computed_batch(self, metadata, committed_num_tokens=None):
        self.calls.append("mark")
        return RequestOwnedStepMarkResult(accepted=True, step_seq=metadata.step_seq)

    def pool_snapshot(self):
        self.calls.append("pool_snapshot")
        return OwnerCachePoolSnapshot(
            owner_rank=0,
            total_blocks=8,
            free_blocks=7 if self.source is None else 5,
        )


def _command(key: OwnerLeaseKey, seq: int, kind: OwnerCommandKind) -> OwnerCommand:
    return OwnerCommand(
        key=key,
        owner_id=0,
        command_seq=seq,
        kind=kind,
        required_num_tokens=8,
    )


def _step(step_seq: int, *, useful_tokens: int) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.step_seq = step_seq
    output.total_num_scheduled_tokens = useful_tokens
    output.num_scheduled_tokens = (
        {"unrelated-running-request": useful_tokens} if useful_tokens else {}
    )
    return output


def _wrapper(*, fail_model: bool = False):
    order: list[str] = []
    key = OwnerLeaseKey("drain", 0)
    store = _DrainStore(key)
    model_worker = _ModelWorker(order, fail=fail_model)
    offload_worker = _DeferredOffloadingWorker(order)
    wrapper = WorkerWrapperBase(global_rank=0)
    wrapper.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=64),
        scheduler_config=SimpleNamespace(
            enable_request_owned_attention=True,
            enable_request_owned_sampling=True,
            enable_request_owned_kv_offload=True,
            max_num_batched_tokens=16,
            max_num_seqs=4,
        ),
    )
    wrapper.worker = model_worker
    wrapper.mm_receiver_cache = None
    wrapper._request_owned_kv_store = store
    wrapper._request_owned_offload_adapter = RequestOwnedBulkOffloadAdapter(
        owner_rank=0,
        manager=CPUOffloadingManager(num_blocks=8),
        worker=offload_worker,
    )
    manager = AttentionLeaseManager(owner_rank=0, capacity=64)
    assert manager.apply(_command(key, 1, OwnerCommandKind.RESERVE)).accepted
    manager.emit_batch(1)
    wrapper._request_owned_control_manager = manager
    return wrapper, store, model_worker, offload_worker, order, key


def test_token_work_overlaps_d2h_and_withholds_preempt_receipt() -> None:
    wrapper, store, model, offload, order, key = _wrapper()
    preempt = _step(2, useful_tokens=1)
    preempt.owner_commands = [_command(key, 2, OwnerCommandKind.PREEMPT)]

    first = wrapper.execute_model(preempt)

    assert order == ["submit_d2h", "model"]
    assert offload.waited == []
    assert first.owner_receipt_batches[0].events == ()
    assert first.owner_receipt_batches[0].pending_dma == 1
    assert store.source is not None

    second = wrapper.execute_model(_step(3, useful_tokens=1))
    assert second.owner_receipt_batches[0].events == ()
    assert second.owner_receipt_batches[0].pending_dma == 1
    assert model.calls == 2

    offload.finish(0)
    terminal = wrapper.execute_model(_step(4, useful_tokens=1))
    (receipt,) = terminal.owner_receipt_batches[0].events
    assert receipt.accepted
    assert receipt.key == key
    assert receipt.command_seq == 2
    assert terminal.owner_receipt_batches[0].pending_dma == 0
    assert store.source is None
    assert offload.waited == []


def test_zero_token_control_step_waits_once_for_liveness() -> None:
    wrapper, store, _, offload, _, key = _wrapper()
    preempt = _step(2, useful_tokens=0)
    preempt.owner_commands = [_command(key, 2, OwnerCommandKind.PREEMPT)]

    result = wrapper.execute_model(preempt)

    (receipt,) = result.owner_receipt_batches[0].events
    assert receipt.accepted
    assert receipt.command_seq == 2
    assert result.owner_receipt_batches[0].pending_dma == 0
    assert offload.waited == [{0}]
    assert store.source is None


def test_failed_background_d2h_latches_fail_stop_without_reclaim() -> None:
    wrapper, store, _, offload, _, key = _wrapper()
    preempt = _step(2, useful_tokens=1)
    preempt.owner_commands = [_command(key, 2, OwnerCommandKind.PREEMPT)]
    wrapper.execute_model(preempt)
    offload.finish(0, success=False)

    with pytest.raises(RequestOwnedOffloadError, match="D2H failed"):
        wrapper.execute_model(_step(3, useful_tokens=1))
    assert store.source is not None
    assert "preempt" not in store.calls

    with pytest.raises(RequestOwnedOffloadError, match="fail-stop"):
        wrapper.execute_model(_step(4, useful_tokens=1))
    assert store.source is not None


def test_model_failure_after_submit_latches_uncommitted_drain() -> None:
    wrapper, store, _, offload, _, key = _wrapper(fail_model=True)
    preempt = _step(2, useful_tokens=1)
    preempt.owner_commands = [_command(key, 2, OwnerCommandKind.PREEMPT)]

    with pytest.raises(RuntimeError, match="model step failed"):
        wrapper.execute_model(preempt)
    assert offload.pending == {0}
    assert store.source is not None

    with pytest.raises(RequestOwnedOffloadError, match="fail-stop"):
        wrapper.execute_model(_step(3, useful_tokens=1))
    assert store.source is not None


def test_release_waits_for_same_key_drain_and_suppresses_stale_preempt() -> None:
    wrapper, store, _, offload, _, key = _wrapper()
    preempt = _step(2, useful_tokens=1)
    preempt.owner_commands = [_command(key, 2, OwnerCommandKind.PREEMPT)]
    assert wrapper.execute_model(preempt).owner_receipt_batches[0].events == ()

    release = _step(3, useful_tokens=0)
    release.owner_commands = [_command(key, 3, OwnerCommandKind.RELEASE)]
    result = wrapper.execute_model(release)

    assert offload.waited == [{0}]
    assert store.source is None
    assert [event.command_seq for event in result.owner_receipt_batches[0].events] == [
        3
    ]
    assert result.owner_receipt_batches[0].events[0].released
