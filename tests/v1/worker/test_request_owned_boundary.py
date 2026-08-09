# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU tests for the G2 request-owned worker envelope boundary.

Command processing is exercised against an injected fake physical store for
wrapper ordering, plus at least one real :class:`KVCacheManager` integration
through ``WorkerWrapperBase.initialize_from_config`` (the store itself has
its own real-manager tests in test_request_owned_kv.py).
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.worker_base as worker_base_module
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.ownership import (
    AttentionLeaseManager,
    OwnerAdmissionStatus,
    OwnerAllocationDescriptor,
    OwnerCachePoolSnapshot,
    OwnerCommand,
    OwnerCommandKind,
    OwnerLeaseKey,
    OwnerLeaseToken,
    PublicationViolationError,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
)
from vllm.v1.worker.request_owned_kv import (
    AllocationResult,
    DeferredFreeResult,
    RequestOwnedStepMetadata,
)
from vllm.v1.worker.worker_base import WorkerWrapperBase

pytestmark = pytest.mark.cpu_test


class _FakeWorker:
    def __init__(self, output=EMPTY_MODEL_RUNNER_OUTPUT, before_return=None) -> None:
        self.output = output
        self.before_return = before_return
        self.calls = 0
        self.initialized_config = None

    def initialize_from_config(self, kv_cache_config):
        self.initialized_config = kv_cache_config

    def execute_model(self, scheduler_output):
        self.calls += 1
        if self.before_return is not None:
            self.before_return(scheduler_output)
        return self.output


class _FakeAsyncOutput(AsyncModelRunnerOutput):
    def get_output(self) -> ModelRunnerOutput:
        return EMPTY_MODEL_RUNNER_OUTPUT


class _FakeStore:
    """Wrapper-ordering fake for the physical store.

    RESERVE/EXTEND/PREEMPT/RELEASE accept unless their kind is in ``reject``;
    RESTORE always rejects (mirroring the real store).  Every call is logged
    so tests can assert ordering and that stale/wrong commands never touch
    the store.  pool_snapshot() returns a real protocol snapshot so receipt
    construction validates.
    """

    def __init__(
        self, owner_rank: int, reject=(), reject_keys=(), reject_build=False
    ) -> None:
        self.owner_rank = owner_rank
        self.reject = set(reject)
        self.reject_keys = set(reject_keys)
        self.reject_build = reject_build
        self.calls: list[str] = []
        self.last_build_step: int | None = None
        self.last_build_tokens: tuple = ()
        self.last_build_counts: dict = {}

    def reserve(self, command):
        self.calls.append("reserve")
        return self._allocation(command)

    def extend(self, command):
        self.calls.append("extend")
        return self._allocation(command)

    def preempt(self, command):
        self.calls.append("preempt")
        return self._free(command)

    def release(self, command):
        self.calls.append("release")
        return self._free(command)

    def restore(self, command):
        self.calls.append("restore")
        return DeferredFreeResult(
            accepted=False,
            key=command.key,
            error="RESTORE is out of scope for the physical KV store",
        )

    def build_step_metadata(self, step_seq, tokens, request_token_counts):
        self.calls.append("build")
        self.last_build_step = step_seq
        self.last_build_tokens = tuple(tokens)
        self.last_build_counts = dict(request_token_counts)
        if self.reject_build:
            return SimpleNamespace(
                accepted=False,
                step_seq=step_seq,
                metadata=None,
                error="fake build failure",
            )
        return SimpleNamespace(
            accepted=True,
            step_seq=step_seq,
            metadata=RequestOwnedStepMetadata(
                step_seq=step_seq, owner_rank=self.owner_rank, entries=()
            ),
            error=None,
        )

    def flush(self):
        self.calls.append("flush")
        return ()

    def pool_snapshot(self):
        self.calls.append("pool_snapshot")
        return OwnerCachePoolSnapshot(
            owner_rank=self.owner_rank, total_blocks=32, free_blocks=32
        )

    def _allocation(self, command):
        if command.kind in self.reject or command.key.request_id in self.reject_keys:
            return AllocationResult(
                accepted=False, key=command.key, error="physical reserve failure"
            )
        return AllocationResult(accepted=True, key=command.key)

    def _free(self, command):
        if command.kind in self.reject:
            return DeferredFreeResult(
                accepted=False, key=command.key, error="physical free failure"
            )
        return DeferredFreeResult(accepted=True, key=command.key, deferred=True)


def _wrapper(
    rank: int, worker: _FakeWorker, store: _FakeStore | None = None
) -> WorkerWrapperBase:
    wrapper = WorkerWrapperBase(global_rank=rank)
    wrapper.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=64),
        scheduler_config=SimpleNamespace(
            enable_request_owned_attention=True,
            max_num_batched_tokens=16,
            max_num_seqs=4,
        ),
    )
    wrapper.worker = worker
    wrapper.mm_receiver_cache = None
    wrapper._request_owned_kv_store = store if store is not None else _FakeStore(rank)
    return wrapper


def _output(step_seq: int = 1) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.step_seq = step_seq
    return output


def _reserve(owner_id: int, command_seq: int = 1) -> OwnerCommand:
    return OwnerCommand(
        key=OwnerLeaseKey("req", 0),
        owner_id=owner_id,
        command_seq=command_seq,
        kind=OwnerCommandKind.RESERVE,
        required_num_tokens=8,
    )


def _real_reserve(
    owner_id: int,
    command_seq: int,
    required: int = 10,
    request_id: str = "req",
) -> OwnerCommand:
    """RESERVE with the allocation descriptor the real store needs."""
    key = OwnerLeaseKey(request_id, 0)
    return OwnerCommand(
        key=key,
        owner_id=owner_id,
        command_seq=command_seq,
        kind=OwnerCommandKind.RESERVE,
        required_num_tokens=required,
        allocation=OwnerAllocationDescriptor(
            key=key,
            num_prompt_tokens=required,
            num_computed_tokens=0,
            num_tokens=required,
            status=OwnerAdmissionStatus.WAITING,
        ),
    )


def _real_vllm_config(dcp: int = 1, pcp: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=64),
        scheduler_config=SimpleNamespace(
            enable_request_owned_attention=True,
            max_num_batched_tokens=16,
            max_num_seqs=4,
        ),
        cache_config=SimpleNamespace(block_size=4, enable_prefix_caching=False),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
        ),
    )


def _kv_cache_config(num_blocks: int = 32, block_size: int = 4) -> KVCacheConfig:
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=4,
        head_size=8,
        dtype=torch.float32,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["a"], kv_cache_spec=spec)],
    )


def _real_wrapper(worker: _FakeWorker, num_blocks: int = 32) -> WorkerWrapperBase:
    wrapper = WorkerWrapperBase(global_rank=0)
    wrapper.vllm_config = _real_vllm_config()
    wrapper.worker = worker
    wrapper.mm_receiver_cache = None
    wrapper.initialize_from_config([_kv_cache_config(num_blocks=num_blocks)])
    return wrapper


class _RecordingKVCacheManager(KVCacheManager):
    last_kwargs: dict | None = None

    def __init__(self, **kwargs) -> None:
        _RecordingKVCacheManager.last_kwargs = kwargs
        super().__init__(**kwargs)


# -- constructor ------------------------------------------------------------


def test_store_constructed_after_worker_init_with_correct_config(
    monkeypatch,
) -> None:
    worker = _FakeWorker()
    wrapper = WorkerWrapperBase(global_rank=2)
    wrapper.vllm_config = _real_vllm_config(dcp=2, pcp=1)
    wrapper.worker = worker
    wrapper.mm_receiver_cache = None
    kv_cache_config = _kv_cache_config()
    monkeypatch.setattr(worker_base_module, "KVCacheManager", _RecordingKVCacheManager)
    assert wrapper._request_owned_kv_store is None
    wrapper.initialize_from_config([kv_cache_config] * 3)

    # The store is bound only after the underlying worker initialized.
    assert worker.initialized_config is kv_cache_config
    assert wrapper._request_owned_kv_store is not None
    assert wrapper._request_owned_kv_store._owner_rank == 2
    assert wrapper._request_owned_kv_store._manager is not None
    kwargs = _RecordingKVCacheManager.last_kwargs
    assert kwargs["kv_cache_config"] is kv_cache_config
    assert kwargs["max_model_len"] == 64
    assert kwargs["max_num_batched_tokens"] == 16
    # DCP/PCP and block sizes come from the same facts the scheduler uses:
    # scheduler_block_size = block_size * dcp * pcp for a single group.
    assert kwargs["scheduler_block_size"] == 8
    assert kwargs["hash_block_size"] == 8
    assert kwargs["dcp_world_size"] == 2
    assert kwargs["pcp_world_size"] == 1
    assert kwargs["enable_caching"] is False
    assert kwargs["use_eagle"] is False
    assert kwargs["log_stats"] is False
    assert kwargs["enable_kv_cache_events"] is False
    assert kwargs.get("metrics_collector") is None


def test_store_not_constructed_when_feature_disabled(monkeypatch) -> None:
    worker = _FakeWorker()
    wrapper = WorkerWrapperBase(global_rank=0)
    vllm_config = _real_vllm_config()
    vllm_config.scheduler_config.enable_request_owned_attention = False
    wrapper.vllm_config = vllm_config
    wrapper.worker = worker
    wrapper.mm_receiver_cache = None
    monkeypatch.setattr(worker_base_module, "KVCacheManager", _RecordingKVCacheManager)
    _RecordingKVCacheManager.last_kwargs = None
    wrapper.initialize_from_config([_kv_cache_config()])
    assert wrapper._request_owned_kv_store is None
    assert _RecordingKVCacheManager.last_kwargs is None


# -- failure-atomic command composition --------------------------------------


def test_physical_reserve_accepted_yields_accepted_receipt_and_snapshot() -> None:
    worker = _FakeWorker()
    store = _FakeStore(0)
    wrapper = _wrapper(0, worker, store)
    step = _output()
    step.owner_commands = [_reserve(owner_id=0)]

    result = wrapper.execute_model(step)
    batch = result.owner_receipt_batches[0]
    assert batch.owner_rank == 0
    assert batch.emitted_step_seq == 1
    assert len(batch.events) == 1
    receipt = batch.events[0]
    assert receipt.accepted
    assert receipt.runnable_num_tokens == 8
    # G2: the receipt carries a block-ID-free physical capacity snapshot.
    assert batch.cache_pool is not None
    assert batch.cache_pool.owner_rank == 0
    assert batch.cache_pool.total_blocks == 32
    assert batch.cache_pool.free_blocks == 32
    assert store.calls == ["reserve", "build", "flush", "pool_snapshot"]


def test_physical_reserve_failure_no_logical_grant_next_command_works() -> None:
    worker = _FakeWorker()
    store = _FakeStore(0, reject_keys={"req"})
    wrapper = _wrapper(0, worker, store)
    step = _output(step_seq=1)
    step.owner_commands = [_reserve(owner_id=0, command_seq=1)]

    result = wrapper.execute_model(step)
    batch = result.owner_receipt_batches[0]
    assert len(batch.events) == 1
    receipt = batch.events[0]
    assert not receipt.accepted
    assert receipt.error == "physical reserve failure"
    # The logical manager committed the external-reject fences.
    assert wrapper._request_owned_control_manager is not None

    # No logical grant for the rejected key: a RELEASE for it is refused on
    # logical grounds and the store is never consulted.
    store.calls.clear()
    step2 = _output(step_seq=2)
    step2.owner_commands = [
        OwnerCommand(
            key=OwnerLeaseKey("req", 0),
            owner_id=0,
            command_seq=2,
            kind=OwnerCommandKind.RELEASE,
            required_num_tokens=8,
        )
    ]
    result2 = wrapper.execute_model(step2)
    receipt2 = result2.owner_receipt_batches[0].events[0]
    assert not receipt2.accepted
    assert receipt2.error == "no lease to release"
    assert store.calls == ["build", "flush", "pool_snapshot"]

    # Next higher command on a fresh key works.
    store.calls.clear()
    step3 = _output(step_seq=3)
    step3.owner_commands = [
        OwnerCommand(
            key=OwnerLeaseKey("req2", 0),
            owner_id=0,
            command_seq=3,
            kind=OwnerCommandKind.RESERVE,
            required_num_tokens=8,
        )
    ]
    result3 = wrapper.execute_model(step3)
    receipt3 = result3.owner_receipt_batches[0].events[0]
    assert receipt3.accepted
    assert receipt3.key.request_id == "req2"


def test_logical_stale_command_never_touches_store() -> None:
    worker = _FakeWorker()
    store = _FakeStore(0)
    wrapper = _wrapper(0, worker, store)
    step = _output(step_seq=1)
    # seq=1 twice: the duplicate is a stale logical command.
    step.owner_commands = [
        _reserve(owner_id=0, command_seq=1),
        _reserve(owner_id=0, command_seq=1),
    ]

    result = wrapper.execute_model(step)
    events = result.owner_receipt_batches[0].events
    assert len(events) == 2
    assert events[0].accepted
    assert not events[1].accepted
    assert events[1].error == "stale or duplicate command sequence"
    # The stale command never reached the physical store.
    assert store.calls == ["reserve", "build", "flush", "pool_snapshot"]


def test_foreign_commands_and_publications_are_ignored() -> None:
    step = _output(step_seq=4)
    command = _reserve(owner_id=1)
    step.owner_commands = [command]
    step.scheduled_owner_leases = [
        OwnerLeaseToken(
            key=command.key,
            owner_id=1,
            step_seq=4,
            command_seq=1,
            runnable_num_tokens=8,
        )
    ]
    worker = _FakeWorker()
    store = _FakeStore(0)
    result = _wrapper(0, worker, store).execute_model(step)
    assert worker.calls == 1
    assert result.owner_receipt_batches[0].events == ()
    assert store.calls == ["build", "flush", "pool_snapshot"]


def test_own_rank_command_accepted_while_other_rank_emits_empty_batch() -> None:
    step = _output(step_seq=3)
    step.owner_commands = [_reserve(owner_id=1)]

    rank0 = _wrapper(0, _FakeWorker(), _FakeStore(0))
    rank0_result = rank0.execute_model(step)
    rank0_batch = rank0_result.owner_receipt_batches[0]
    assert rank0_batch.owner_rank == 0
    assert rank0_batch.events == ()
    assert rank0_batch.cache_pool is not None
    assert rank0_batch.cache_pool.owner_rank == 0

    rank1 = _wrapper(1, _FakeWorker(), _FakeStore(1))
    rank1_result = rank1.execute_model(step)
    rank1_batch = rank1_result.owner_receipt_batches[0]
    assert rank1_batch.owner_rank == 1
    assert len(rank1_batch.events) == 1
    assert rank1_batch.events[0].accepted
    assert rank1_batch.cache_pool.owner_rank == 1


def test_restore_rejected_even_after_preempt() -> None:
    worker = _FakeWorker()
    wrapper = _wrapper(0, worker)
    step1 = _output(step_seq=1)
    step1.owner_commands = [_reserve(owner_id=0, command_seq=1)]
    assert wrapper.execute_model(step1).owner_receipt_batches[0].events[0].accepted

    step2 = _output(step_seq=2)
    step2.owner_commands = [
        OwnerCommand(
            key=OwnerLeaseKey("req", 0),
            owner_id=0,
            command_seq=2,
            kind=OwnerCommandKind.PREEMPT,
            required_num_tokens=8,
        )
    ]
    assert wrapper.execute_model(step2).owner_receipt_batches[0].events[0].accepted

    step3 = _output(step_seq=3)
    step3.owner_commands = [
        OwnerCommand(
            key=OwnerLeaseKey("req", 0),
            owner_id=0,
            command_seq=3,
            kind=OwnerCommandKind.RESTORE,
            required_num_tokens=8,
        )
    ]
    result = wrapper.execute_model(step3)
    receipt = result.owner_receipt_batches[0].events[0]
    assert not receipt.accepted
    assert receipt.error == "RESTORE is out of scope for the physical KV store"


# -- flush ordering ----------------------------------------------------------


def test_deferred_free_flushed_before_accepted_receipt_and_snapshot() -> None:
    worker = _FakeWorker(before_return=lambda _: store.calls.append("worker"))
    store = _FakeStore(0)
    wrapper = _wrapper(0, worker, store)
    step1 = _output(step_seq=1)
    step1.owner_commands = [_reserve(owner_id=0, command_seq=1)]
    assert wrapper.execute_model(step1).owner_receipt_batches[0].events[0].accepted

    store.calls.clear()
    step2 = _output(step_seq=2)
    step2.owner_commands = [
        OwnerCommand(
            key=OwnerLeaseKey("req", 0),
            owner_id=0,
            command_seq=2,
            kind=OwnerCommandKind.RELEASE,
            required_num_tokens=8,
        )
    ]
    result = wrapper.execute_model(step2)
    batch = result.owner_receipt_batches[0]
    assert len(batch.events) == 1
    assert batch.events[0].accepted
    # The physical free is deferred until after the synchronous execute and
    # is flushed before the receipt batch and capacity snapshot are emitted.
    assert store.calls == ["release", "build", "worker", "flush", "pool_snapshot"]


# -- real KVCacheManager integration -----------------------------------------


def test_real_store_reserve_release_flush_integration() -> None:
    wrapper = _real_wrapper(_FakeWorker())
    store = wrapper._request_owned_kv_store
    manager = store._manager
    initial_free = manager.block_pool.get_num_free_blocks()
    # One block is the pool null block, so 32 configured -> 31 free.
    assert initial_free == 31

    step1 = _output(step_seq=1)
    step1.owner_commands = [_real_reserve(owner_id=0, command_seq=1)]
    result1 = wrapper.execute_model(step1)
    batch1 = result1.owner_receipt_batches[0]
    receipt1 = batch1.events[0]
    assert receipt1.accepted
    # 10 tokens at block_size 4 -> 3 blocks; snapshot taken after reserve.
    assert batch1.cache_pool is not None
    assert batch1.cache_pool.total_blocks == 32
    assert batch1.cache_pool.free_blocks == initial_free - 3
    # Receipts carry no block IDs by construction.
    assert batch1.cache_pool.groups[0].allocated_blocks == 3
    assert receipt1.key == OwnerLeaseKey("req", 0)

    step2 = _output(step_seq=2)
    step2.owner_commands = [
        OwnerCommand(
            key=OwnerLeaseKey("req", 0),
            owner_id=0,
            command_seq=2,
            kind=OwnerCommandKind.RELEASE,
            required_num_tokens=8,
        )
    ]
    result2 = wrapper.execute_model(step2)
    batch2 = result2.owner_receipt_batches[0]
    assert batch2.events[0].accepted
    # The deferred free was flushed before the snapshot: the pool is whole
    # again and the record is gone.
    assert manager.block_pool.get_num_free_blocks() == initial_free
    assert batch2.cache_pool.free_blocks == initial_free
    assert store.get_block_table(OwnerLeaseKey("req", 0)) is None


def test_real_store_physical_reserve_failure() -> None:
    wrapper = _real_wrapper(_FakeWorker(), num_blocks=2)
    store = wrapper._request_owned_kv_store
    step = _output(step_seq=1)
    step.owner_commands = [_real_reserve(owner_id=0, command_seq=1)]

    result = wrapper.execute_model(step)
    receipt = result.owner_receipt_batches[0].events[0]
    assert not receipt.accepted
    assert receipt.error == "insufficient KV cache to reserve"
    assert store.get_block_table(OwnerLeaseKey("req", 0)) is None


# -- unchanged G1 validations -------------------------------------------------


def test_command_processing_precedes_underlying_worker(monkeypatch) -> None:
    step = _output()
    step.owner_commands = [_reserve(owner_id=1)]
    order = []
    original_apply = AttentionLeaseManager.apply

    def record_apply(manager, command):
        order.append("apply")
        return original_apply(manager, command)

    monkeypatch.setattr(AttentionLeaseManager, "apply", record_apply)
    worker = _FakeWorker(before_return=lambda _: order.append("worker"))
    wrapper = _wrapper(1, worker, _FakeStore(1))
    wrapper.execute_model(step)
    assert worker.calls == 1
    assert order == ["apply", "worker"]


def test_scheduled_tokens_fail_before_underlying_worker() -> None:
    worker = _FakeWorker()
    wrapper = _wrapper(0, worker)
    step = _output()
    step.total_num_scheduled_tokens = 1

    with pytest.raises(RuntimeError, match="replicated KV"):
        wrapper.execute_model(step)
    assert worker.calls == 0
    assert wrapper._request_owned_control_manager is None


@pytest.mark.parametrize(
    "total, per_request",
    [
        (False, {}),
        (0.0, {}),
        (0, {"req": 1}),
        (1, {}),
        (0, {"req": True}),
        (0, {"req": -1, "other": 1}),
    ],
)
def test_inconsistent_token_envelope_fails_before_worker(total, per_request) -> None:
    worker = _FakeWorker()
    wrapper = _wrapper(0, worker)
    step = _output()
    step.total_num_scheduled_tokens = total
    step.num_scheduled_tokens = per_request
    with pytest.raises(RuntimeError, match="inconsistent token schedule"):
        wrapper.execute_model(step)
    assert worker.calls == 0
    assert wrapper._request_owned_control_manager is None


def test_empty_singleton_is_never_mutated() -> None:
    assert EMPTY_MODEL_RUNNER_OUTPUT.owner_receipt_batches is None
    wrapper = _wrapper(0, _FakeWorker())
    result = wrapper.execute_model(_output())
    assert result is not EMPTY_MODEL_RUNNER_OUTPUT
    assert result.owner_receipt_batches is not None
    assert EMPTY_MODEL_RUNNER_OUTPUT.owner_receipt_batches is None


def test_none_and_async_outputs_fail_explicitly() -> None:
    step = _output()
    step.owner_commands = [_reserve(owner_id=0)]
    none_worker = _FakeWorker(output=None)
    none_wrapper = _wrapper(0, none_worker)
    with pytest.raises(RuntimeError, match="split sampling"):
        none_wrapper.execute_model(step)
    assert none_wrapper._request_owned_control_manager is None

    async_worker = _FakeWorker(output=_FakeAsyncOutput())
    async_wrapper = _wrapper(0, async_worker)
    with pytest.raises(RuntimeError, match="step-keyed receipt FIFO"):
        async_wrapper.execute_model(step)
    assert async_wrapper._request_owned_control_manager is None


def test_underlying_exception_is_fail_stop() -> None:
    def explode(_):
        raise RuntimeError("gpu exploded")

    worker = _FakeWorker(before_return=explode)
    wrapper = _wrapper(0, worker)
    step = _output()
    step.owner_commands = [_reserve(owner_id=0)]
    with pytest.raises(RuntimeError, match="gpu exploded"):
        wrapper.execute_model(step)
    # Fail-stop: the logical manager is never committed and the real shared
    # pool is not pretended to roll back.
    assert wrapper._request_owned_control_manager is None


@pytest.mark.parametrize("step_seq", [0, -1, True, False, 1.5, "1", None])
def test_invalid_step_fails_before_underlying_worker(step_seq) -> None:
    worker = _FakeWorker()
    wrapper = _wrapper(0, worker)
    with pytest.raises(RuntimeError, match="positive non-bool"):
        wrapper.execute_model(_output(step_seq=step_seq))
    assert worker.calls == 0


def test_local_publication_without_physical_grant_fails_before_worker() -> None:
    step = _output(step_seq=4)
    command = _reserve(owner_id=0)
    step.owner_commands = [command]
    step.scheduled_owner_leases = [
        OwnerLeaseToken(
            key=command.key,
            owner_id=0,
            step_seq=4,
            command_seq=1,
            runnable_num_tokens=8,
        )
    ]
    worker = _FakeWorker()
    store = _FakeStore(0, reject={OwnerCommandKind.RESERVE})
    with pytest.raises(PublicationViolationError, match="no lease"):
        _wrapper(0, worker, store).execute_model(step)
    assert worker.calls == 0


def test_default_off_path_is_unchanged() -> None:
    worker = _FakeWorker()
    wrapper = _wrapper(0, worker)
    wrapper.vllm_config.scheduler_config.enable_request_owned_attention = False
    result = wrapper.execute_model(SchedulerOutput.make_empty())
    assert result is EMPTY_MODEL_RUNNER_OUTPUT
    assert worker.calls == 1


# -- G3 step metadata seam ----------------------------------------------------


def test_g3_seam_builds_step_metadata_after_command_publication_validation() -> None:
    step = _output(step_seq=1)
    step.owner_commands = [_reserve(owner_id=0, command_seq=1)]
    step.scheduled_owner_leases = [
        OwnerLeaseToken(
            key=OwnerLeaseKey("other", 0),
            owner_id=1,
            step_seq=1,
            command_seq=1,
            runnable_num_tokens=8,
        )
    ]
    worker = _FakeWorker()
    store = _FakeStore(0)
    wrapper = _wrapper(0, worker, store)

    result = wrapper.execute_model(step)
    assert worker.calls == 1
    assert result.owner_receipt_batches[0].events[0].accepted

    # The seam runs after command+publication validation and receives only
    # the exact own-rank tokens; ordering against the underlying worker is
    # covered by test_deferred_free_flushed_before_accepted_receipt_and_snapshot.
    assert store.calls == ["reserve", "build", "flush", "pool_snapshot"]
    assert store.last_build_step == 1
    assert store.last_build_tokens == ()
    assert store.last_build_counts == {}
    metadata = wrapper._request_owned_step_metadata
    assert metadata is not None
    assert metadata.step_seq == 1
    assert metadata.owner_rank == 0
    assert metadata.entries == ()


def test_g3_seam_fails_closed_when_build_rejected() -> None:
    step = _output(step_seq=1)
    step.owner_commands = [_reserve(owner_id=0, command_seq=1)]
    worker = _FakeWorker()
    store = _FakeStore(0, reject_build=True)
    wrapper = _wrapper(0, worker, store)

    with pytest.raises(RuntimeError, match="step metadata build failed"):
        wrapper.execute_model(step)
    assert worker.calls == 0
    # Fail-stop: neither the logical manager nor the worker advanced.
    assert wrapper._request_owned_control_manager is None
    assert wrapper._request_owned_step_metadata is None
    assert store.calls == ["reserve", "build"]


def test_no_local_id_on_scheduler_wires_while_metadata_is_worker_local() -> None:
    wrapper = _real_wrapper(_FakeWorker())
    store = wrapper._request_owned_kv_store
    step = _output(step_seq=1)
    command = _real_reserve(owner_id=0, command_seq=1)
    step.owner_commands = [command]

    result = wrapper.execute_model(step)
    batch = result.owner_receipt_batches[0]
    receipt = batch.events[0]
    assert receipt.accepted

    # Receipts and pool snapshots on the scheduler-facing output carry no
    # local block ids; the metadata stays worker-local and is not attached.
    assert not hasattr(receipt, "tables")
    assert not hasattr(receipt, "block_ids")
    assert not hasattr(receipt, "delta")
    assert not hasattr(batch.cache_pool, "tables")
    assert not hasattr(batch.cache_pool, "block_ids")
    assert not hasattr(result, "request_owned_step_metadata")

    # Wire objects are untouched: the same command object is still on the
    # scheduler output, and the metadata is the immutable empty batch.
    assert step.owner_commands[0] is command
    metadata = wrapper._request_owned_step_metadata
    assert metadata is not None
    assert metadata.owner_rank == 0
    assert metadata.entries == ()

    # The store's G3 snapshot of a later executed lease carries local ids
    # (worker-local), while the receipt for the same key still does not.
    assert store.build_step_metadata(2, [], {}).accepted
    built = store.build_step_metadata(3, [], {})
    assert built.accepted
    assert built.metadata.step_seq == 3
    assert built.metadata.entries == ()
