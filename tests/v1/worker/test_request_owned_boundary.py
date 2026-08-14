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
    KVCacheTensor,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
    OwnerSamplingBatch,
)
from vllm.v1.worker.request_owned_kv import (
    AllocationResult,
    DeferredFreeResult,
    RequestOwnedKVSnapshot,
    RequestOwnedStepMarkResult,
    RequestOwnedStepMetadata,
)
from vllm.v1.worker.request_owned_offload import RequestOwnedBulkOffloadAdapter
from vllm.v1.worker.worker_base import WorkerWrapperBase

pytestmark = pytest.mark.cpu_test


class _FakeWorker:
    def __init__(self, output=EMPTY_MODEL_RUNNER_OUTPUT, before_return=None) -> None:
        self.output = output
        self.before_return = before_return
        self.calls = 0
        self.initialized_config = None
        self.metadata_handoffs: list[RequestOwnedStepMetadata] = []

    def initialize_from_config(self, kv_cache_config):
        self.initialized_config = kv_cache_config

    def execute_model(self, scheduler_output):
        self.calls += 1
        if self.before_return is not None:
            self.before_return(scheduler_output)
        return self.output

    def set_request_owned_step_metadata(self, metadata):
        self.metadata_handoffs.append(metadata)


class _BulkRestoreWorker(_FakeWorker):
    def __init__(self) -> None:
        super().__init__()
        self.restore_zero_plans: list[tuple[tuple[int, ...], ...]] = []

    def execute_request_owned_bulk_restore(self, work) -> None:
        for item in work:
            self.restore_zero_plans.append(item.zero_block_ids)
            item.execute_after_zero()


class _DeferredBulkRestoreWorker(_BulkRestoreWorker):
    """Ascend-shaped zero-token owner-sampling heartbeat worker."""

    def __init__(self, owner_rank: int = 0) -> None:
        super().__init__()
        self.owner_rank = owner_rank
        self.output = None
        self.sample_calls = 0
        self.sample_grammar = object()
        self.return_none_from_sample = False

    def sample_tokens(self, grammar_output):
        self.sample_calls += 1
        self.sample_grammar = grammar_output
        if self.return_none_from_sample:
            return None
        metadata = self.metadata_handoffs[-1]
        output = ModelRunnerOutput(req_ids=[], req_id_to_index={})
        output.owner_sampling_batches = [
            OwnerSamplingBatch(
                owner_rank=self.owner_rank,
                emitted_step_seq=metadata.step_seq,
                row_ids=(),
            )
        ]
        return output


class _SyncOffloadingWorker(OffloadingWorker):
    """Synchronous transfer receipt fixture for the wrapper lifecycle."""

    def __init__(self) -> None:
        self.finished: list[TransferResult] = []

    def submit_store(
        self,
        job_id: int,
        src_spec: GPULoadStoreSpec,
        dst_spec: LoadStoreSpec,
    ) -> bool:
        self.finished.append(TransferResult(job_id=job_id, success=True))
        return True

    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
    ) -> bool:
        self.finished.append(TransferResult(job_id=job_id, success=True))
        return True

    def get_finished(self) -> list[TransferResult]:
        finished, self.finished = self.finished, []
        return finished

    def wait(self, job_ids: set[int]) -> None:
        return


class _ScriptedDeferredWorker(_FakeWorker):
    """Real-store integration worker for one deferred speculative step."""

    def __init__(self, sample_output: ModelRunnerOutput) -> None:
        super().__init__(output=EMPTY_MODEL_RUNNER_OUTPUT)
        self.sample_output = sample_output
        self.sample_calls = 0

    def execute_model(self, scheduler_output):
        self.calls += 1
        # The RESERVE heartbeat completes synchronously.  The following
        # token-bearing target verification defers its terminal output to
        # sample_tokens, matching the production split-sampling lifecycle.
        return EMPTY_MODEL_RUNNER_OUTPUT if self.calls == 1 else None

    def sample_tokens(self, grammar_output):
        self.sample_calls += 1
        return self.sample_output


class _NoHookWorker:
    """A worker that does not implement the G3 handoff hook."""

    def __init__(self, output=EMPTY_MODEL_RUNNER_OUTPUT) -> None:
        self.output = output
        self.calls = 0

    def execute_model(self, scheduler_output):
        self.calls += 1
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

    def build_step_metadata(
        self,
        step_seq,
        tokens,
        request_token_counts,
        scheduled_spec_decode_tokens,
    ):
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

    def mark_computed_batch(self, metadata):
        self.calls.append("mark")
        return RequestOwnedStepMarkResult(
            accepted=True,
            step_seq=metadata.step_seq,
        )

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


def _mla_spec(block_size: int, compress_ratio: int) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        compress_ratio=compress_ratio,
    )


def _uniform_kv_cache_config(
    num_blocks: int = 32,
    block_size: int = 4,
    inner_specs: dict[str, MLAAttentionSpec] | None = None,
) -> KVCacheConfig:
    """DeepSeek-V4-Flash-shaped config with one uniform wrapper group.

    The inner specs share a block size and base type but differ in
    ``compress_ratio``; the default dict order is deliberately adverse
    (128 first, 4 second) so a naive first-spec pick is observable.
    """
    if inner_specs is None:
        inner_specs = {
            "high": _mla_spec(block_size, compress_ratio=128),
            "low": _mla_spec(block_size, compress_ratio=4),
        }
    uniform = UniformTypeKVCacheSpecs(block_size=block_size, kv_cache_specs=inner_specs)
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=1024,
                shared_by=["a", "b"],
                offset=7,
                block_stride=9,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["a"], kv_cache_spec=uniform)],
    )


def _real_wrapper(worker: _FakeWorker, num_blocks: int = 32) -> WorkerWrapperBase:
    wrapper = WorkerWrapperBase(global_rank=0)
    wrapper.vllm_config = _real_vllm_config()
    wrapper.worker = worker
    wrapper.mm_receiver_cache = None
    wrapper.initialize_from_config([_kv_cache_config(num_blocks=num_blocks)])
    return wrapper


def _speculative_real_wrapper(
    worker: _ScriptedDeferredWorker, num_blocks: int = 32
) -> WorkerWrapperBase:
    wrapper = _real_wrapper(worker, num_blocks=num_blocks)
    wrapper.vllm_config.scheduler_config.enable_request_owned_sampling = True
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


def test_exclusive_offload_adapter_constructed_after_worker_cache_init(
    monkeypatch,
) -> None:
    worker = _FakeWorker()
    wrapper = WorkerWrapperBase(global_rank=2)
    wrapper.vllm_config = _real_vllm_config()
    wrapper.vllm_config.scheduler_config.enable_request_owned_kv_offload = True
    wrapper.worker = worker
    wrapper.mm_receiver_cache = None
    sentinel = object()
    build_calls: list[int] = []

    def build_adapter():
        build_calls.append(worker.calls)
        assert worker.initialized_config is not None
        return sentinel

    monkeypatch.setattr(wrapper, "_create_request_owned_offload_adapter", build_adapter)
    wrapper.initialize_from_config([_kv_cache_config()] * 3)

    assert wrapper._request_owned_offload_adapter is sentinel
    assert build_calls == [0]


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


def test_plain_config_normalization_keeps_identity() -> None:
    raw = _kv_cache_config()
    assert worker_base_module._normalize_request_owned_kv_cache_config(raw) is raw


def test_uniform_store_binds_min_compress_ratio_representative(
    monkeypatch,
) -> None:
    worker = _FakeWorker()
    wrapper = WorkerWrapperBase(global_rank=0)
    wrapper.vllm_config = _real_vllm_config()
    wrapper.worker = worker
    wrapper.mm_receiver_cache = None
    raw = _uniform_kv_cache_config()
    monkeypatch.setattr(worker_base_module, "KVCacheManager", _RecordingKVCacheManager)
    wrapper.initialize_from_config([raw])

    # The underlying worker still initialized with the original raw config.
    assert worker.initialized_config is raw
    # The store manager ran on a normalized copy whose uniform group is bound
    # to the allocation-binding inner spec: minimum positive compress_ratio
    # (4), not the dict-first spec (128).
    manager_config = _RecordingKVCacheManager.last_kwargs["kv_cache_config"]
    assert manager_config is not raw
    group_spec = manager_config.kv_cache_groups[0].kv_cache_spec
    assert isinstance(group_spec, MLAAttentionSpec)
    assert group_spec.compress_ratio == 4
    # Store metadata sees the concrete spec: concrete kind and
    # block_size * binding ratio effective tokens per block.
    snapshot = wrapper._request_owned_kv_store.pool_snapshot()
    assert snapshot.groups[0].spec_kind == "mla_attention"
    assert snapshot.groups[0].effective_tokens_per_block == 4 * 4
    # The original wrapper, dict insertion order, and inner specs are
    # untouched by store construction.
    original_group = raw.kv_cache_groups[0].kv_cache_spec
    assert isinstance(original_group, UniformTypeKVCacheSpecs)
    assert list(original_group.kv_cache_specs) == ["high", "low"]
    assert original_group.kv_cache_specs["high"].compress_ratio == 128
    assert original_group.kv_cache_specs["low"].compress_ratio == 4


def test_normalized_config_is_distinct_copy_preserving_metadata() -> None:
    raw = _uniform_kv_cache_config(num_blocks=48)
    raw.kv_cache_groups.append(
        KVCacheGroupSpec(
            layer_names=["b"],
            kv_cache_spec=FullAttentionSpec(
                block_size=4,
                num_kv_heads=2,
                head_size=8,
                dtype=torch.float32,
            ),
        )
    )

    normalized = worker_base_module._normalize_request_owned_kv_cache_config(raw)

    assert normalized is not raw
    assert normalized.num_blocks == raw.num_blocks == 48
    # Group order and layer names are preserved; only the uniform wrapper is
    # replaced by its binding representative, and the plain group is copied
    # untouched.
    assert [g.layer_names for g in normalized.kv_cache_groups] == [
        ["a"],
        ["b"],
    ]
    uniform_spec = normalized.kv_cache_groups[0].kv_cache_spec
    assert isinstance(uniform_spec, MLAAttentionSpec)
    assert uniform_spec.compress_ratio == 4
    assert isinstance(normalized.kv_cache_groups[1].kv_cache_spec, FullAttentionSpec)
    # KVCacheTensor geometry is preserved on distinct deep-copied objects.
    assert len(normalized.kv_cache_tensors) == len(raw.kv_cache_tensors) == 1
    normalized_tensor = normalized.kv_cache_tensors[0]
    raw_tensor = raw.kv_cache_tensors[0]
    assert normalized_tensor is not raw_tensor
    assert (
        normalized_tensor.size,
        normalized_tensor.offset,
        normalized_tensor.block_stride,
    ) == (raw_tensor.size, raw_tensor.offset, raw_tensor.block_stride)
    assert normalized_tensor.shared_by == raw_tensor.shared_by
    # The raw config (wrapper, dict order, inner specs) is unmodified.
    original_group = raw.kv_cache_groups[0].kv_cache_spec
    assert isinstance(original_group, UniformTypeKVCacheSpecs)
    assert list(original_group.kv_cache_specs) == ["high", "low"]
    assert original_group.kv_cache_specs["high"].compress_ratio == 128


def test_uniform_empty_wrapper_fails_closed(monkeypatch) -> None:
    worker = _FakeWorker()
    wrapper = WorkerWrapperBase(global_rank=0)
    wrapper.vllm_config = _real_vllm_config()
    wrapper.worker = worker
    wrapper.mm_receiver_cache = None
    raw = _uniform_kv_cache_config(inner_specs={})
    monkeypatch.setattr(worker_base_module, "KVCacheManager", _RecordingKVCacheManager)
    _RecordingKVCacheManager.last_kwargs = None
    with pytest.raises(ValueError, match="no inner KV cache specs"):
        wrapper.initialize_from_config([raw])
    assert _RecordingKVCacheManager.last_kwargs is None


@pytest.mark.parametrize("bad_ratio", [True, 0, -1])
def test_uniform_malformed_ratio_fails_closed(monkeypatch, bad_ratio) -> None:
    worker = _FakeWorker()
    wrapper = WorkerWrapperBase(global_rank=0)
    wrapper.vllm_config = _real_vllm_config()
    wrapper.worker = worker
    wrapper.mm_receiver_cache = None
    raw = _uniform_kv_cache_config(
        inner_specs={
            "bad": _mla_spec(4, compress_ratio=bad_ratio),
            "good": _mla_spec(4, compress_ratio=4),
        }
    )
    monkeypatch.setattr(worker_base_module, "KVCacheManager", _RecordingKVCacheManager)
    _RecordingKVCacheManager.last_kwargs = None
    with pytest.raises(ValueError, match="invalid compress_ratio"):
        wrapper.initialize_from_config([raw])
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


def test_exclusive_bulk_preempt_restore_resume_emits_terminal_receipt() -> None:
    worker = _BulkRestoreWorker()
    wrapper = _real_wrapper(worker)
    wrapper.vllm_config.scheduler_config.enable_request_owned_kv_offload = True
    wrapper._request_owned_offload_adapter = RequestOwnedBulkOffloadAdapter(
        owner_rank=0,
        manager=CPUOffloadingManager(num_blocks=8),
        worker=_SyncOffloadingWorker(),
    )
    store = wrapper._request_owned_kv_store
    assert store is not None
    key = OwnerLeaseKey("req", 0)

    reserve_step = _output(step_seq=1)
    reserve_step.owner_commands = [_real_reserve(owner_id=0, command_seq=1)]
    assert (
        wrapper.execute_model(reserve_step).owner_receipt_batches[0].events[0].accepted
    )
    assert store.mark_computed(key, 6)

    preempt_step = _output(step_seq=2)
    preempt_step.owner_commands = [
        OwnerCommand(
            key=key,
            owner_id=0,
            command_seq=2,
            kind=OwnerCommandKind.PREEMPT,
            required_num_tokens=10,
        )
    ]
    preempt_receipt = wrapper.execute_model(preempt_step).owner_receipt_batches[0]
    assert preempt_receipt.events[0].accepted
    assert store.snapshot(key) is None

    restore_step = _output(step_seq=3)
    restore_step.owner_commands = [
        OwnerCommand(
            key=key,
            owner_id=0,
            command_seq=3,
            kind=OwnerCommandKind.RESTORE,
            required_num_tokens=10,
        )
    ]
    restore_batch = wrapper.execute_model(restore_step).owner_receipt_batches[0]
    assert restore_batch.events[0].accepted
    assert restore_batch.events[0].pending_dma == 0
    assert restore_batch.pending_dma == 0
    restored = store.snapshot(key)
    assert restored is not None
    assert restored.num_computed_tokens == 6
    assert worker.restore_zero_plans == [restored.tables]
    assert store.is_restore_ready(key)

    resume_step = _output(step_seq=4)
    resume_step.owner_commands = [
        OwnerCommand(
            key=key,
            owner_id=0,
            command_seq=4,
            kind=OwnerCommandKind.RESERVE,
            required_num_tokens=10,
            allocation=OwnerAllocationDescriptor(
                key=key,
                num_prompt_tokens=10,
                num_computed_tokens=6,
                num_tokens=10,
                status=OwnerAdmissionStatus.PREEMPTED,
            ),
        )
    ]
    resume_batch = wrapper.execute_model(resume_step).owner_receipt_batches[0]
    assert resume_batch.events[0].accepted
    assert not store.is_restore_ready(key)


def test_bulk_preempt_skips_hybrid_null_block_placeholders() -> None:
    wrapper = _wrapper(0, _FakeWorker())
    wrapper._request_owned_offload_adapter = RequestOwnedBulkOffloadAdapter(
        owner_rank=0,
        manager=CPUOffloadingManager(num_blocks=8),
        worker=_SyncOffloadingWorker(),
    )
    key = OwnerLeaseKey("req", 0)
    snapshot = RequestOwnedKVSnapshot(
        key=key,
        owner_rank=0,
        allocation_generation=1,
        num_computed_tokens=8,
        reserved_num_tokens=8,
        pending_free=True,
        tables=((0, 2), (3, 0)),
    )

    class _NullPlaceholderStore:
        group_block_sizes = (4, 4)

        def computed_prefix_snapshot(self, candidate):
            assert candidate == key
            return snapshot

        def preempt(self, command):
            return DeferredFreeResult(accepted=True, key=command.key, deferred=True)

    command = OwnerCommand(
        key=key,
        owner_id=0,
        command_seq=1,
        kind=OwnerCommandKind.PREEMPT,
        required_num_tokens=8,
    )
    result = wrapper._store_request_owned_preempt(command, _NullPlaceholderStore())
    assert result.accepted


def test_bulk_restore_rolls_back_on_late_step_failure_and_is_retryable() -> None:
    worker = _BulkRestoreWorker()
    wrapper = _real_wrapper(worker)
    wrapper.vllm_config.scheduler_config.enable_request_owned_kv_offload = True
    wrapper._request_owned_offload_adapter = RequestOwnedBulkOffloadAdapter(
        owner_rank=0,
        manager=CPUOffloadingManager(num_blocks=8),
        worker=_SyncOffloadingWorker(),
    )
    store = wrapper._request_owned_kv_store
    assert store is not None
    key = OwnerLeaseKey("req", 0)

    reserve_step = _output(step_seq=1)
    reserve_step.owner_commands = [_real_reserve(owner_id=0, command_seq=1)]
    assert (
        wrapper.execute_model(reserve_step).owner_receipt_batches[0].events[0].accepted
    )
    assert store.mark_computed(key, 6)

    preempt_step = _output(step_seq=2)
    preempt_step.owner_commands = [
        OwnerCommand(
            key=key,
            owner_id=0,
            command_seq=2,
            kind=OwnerCommandKind.PREEMPT,
            required_num_tokens=10,
        )
    ]
    assert (
        wrapper.execute_model(preempt_step).owner_receipt_batches[0].events[0].accepted
    )

    restore_step = _output(step_seq=3)
    restore_step.owner_commands = [
        OwnerCommand(
            key=key,
            owner_id=0,
            command_seq=3,
            kind=OwnerCommandKind.RESTORE,
            required_num_tokens=10,
        )
    ]

    def fail_after_restore(_scheduler_output) -> None:
        raise RuntimeError("late model failure")

    worker.before_return = fail_after_restore
    with pytest.raises(RuntimeError, match="late model failure"):
        wrapper.execute_model(restore_step)

    # H2D completed, but no terminal receipt committed.  The exact final
    # destination is gone while the cold host image remains retryable.
    assert store.snapshot(key) is None
    assert wrapper._request_owned_restore_guard is None

    worker.before_return = None
    wrapper.vllm_config.scheduler_config.enable_request_owned_sampling = True
    original_pool_snapshot = store.pool_snapshot

    def fail_terminal_snapshot():
        raise RuntimeError("late receipt failure")

    store.pool_snapshot = fail_terminal_snapshot
    with pytest.raises(RuntimeError, match="late receipt failure"):
        wrapper.execute_model(restore_step)
    assert store.snapshot(key) is None
    assert wrapper._request_owned_restore_guard is None
    assert wrapper._request_owned_fail_stop is None

    store.pool_snapshot = original_pool_snapshot
    retry_batch = wrapper.execute_model(restore_step).owner_receipt_batches[0]
    assert retry_batch.events[0].accepted
    assert retry_batch.events[0].pending_dma == 0
    assert store.is_restore_ready(key)
    assert len(worker.restore_zero_plans) == 3


def test_bulk_restore_commits_deferred_sampling_heartbeat_synchronously() -> None:
    worker = _DeferredBulkRestoreWorker()
    wrapper = _real_wrapper(worker)
    wrapper.vllm_config.scheduler_config.enable_request_owned_sampling = True
    wrapper.vllm_config.scheduler_config.enable_request_owned_kv_offload = True
    wrapper._request_owned_offload_adapter = RequestOwnedBulkOffloadAdapter(
        owner_rank=0,
        manager=CPUOffloadingManager(num_blocks=8),
        worker=_SyncOffloadingWorker(),
    )
    store = wrapper._request_owned_kv_store
    assert store is not None
    key = OwnerLeaseKey("req", 0)

    worker.output = EMPTY_MODEL_RUNNER_OUTPUT
    reserve_step = _output(step_seq=1)
    reserve_step.owner_commands = [_real_reserve(owner_id=0, command_seq=1)]
    wrapper.execute_model(reserve_step)
    assert store.mark_computed(key, 6)

    preempt_step = _output(step_seq=2)
    preempt_step.owner_commands = [
        OwnerCommand(
            key=key,
            owner_id=0,
            command_seq=2,
            kind=OwnerCommandKind.PREEMPT,
            required_num_tokens=10,
        )
    ]
    wrapper.execute_model(preempt_step)

    worker.output = None
    restore_step = _output(step_seq=3)
    restore_step.owner_commands = [
        OwnerCommand(
            key=key,
            owner_id=0,
            command_seq=3,
            kind=OwnerCommandKind.RESTORE,
            required_num_tokens=10,
        )
    ]
    worker.return_none_from_sample = True
    with pytest.raises(RuntimeError, match="heartbeat sample_tokens returned None"):
        wrapper.execute_model(restore_step)
    assert store.snapshot(key) is None
    assert wrapper._request_owned_restore_guard is None
    assert wrapper._request_owned_deferred is None

    worker.return_none_from_sample = False
    result = wrapper.execute_model(restore_step)

    assert worker.sample_calls == 2
    assert worker.sample_grammar is None
    assert result.owner_sampling_batches == [
        OwnerSamplingBatch(owner_rank=0, emitted_step_seq=3, row_ids=())
    ]
    assert result.owner_receipt_batches is not None
    receipt = result.owner_receipt_batches[0].events[0]
    assert receipt.accepted
    assert receipt.pending_dma == 0
    assert store.is_restore_ready(key)
    assert wrapper._request_owned_restore_guard is None
    assert wrapper._request_owned_deferred is None
    assert len(worker.restore_zero_plans) == 2


def test_remote_rank_joins_global_restore_heartbeat_synchronously() -> None:
    worker = _DeferredBulkRestoreWorker(owner_rank=1)
    wrapper = _wrapper(1, worker)
    wrapper.vllm_config.scheduler_config.enable_request_owned_sampling = True
    step = _output(step_seq=7)
    step.owner_commands = [
        OwnerCommand(
            key=OwnerLeaseKey("remote", 0),
            owner_id=0,
            command_seq=4,
            kind=OwnerCommandKind.RESTORE,
            required_num_tokens=10,
        )
    ]

    result = wrapper.execute_model(step)

    assert worker.sample_calls == 1
    assert result.owner_sampling_batches == [
        OwnerSamplingBatch(owner_rank=1, emitted_step_seq=7, row_ids=())
    ]
    assert result.owner_receipt_batches is not None
    assert result.owner_receipt_batches[0].events == ()
    assert wrapper._request_owned_restore_guard is None
    assert wrapper._request_owned_deferred is None


def test_global_restore_rejects_mixed_remote_command_before_execution() -> None:
    worker = _DeferredBulkRestoreWorker(owner_rank=1)
    store = _FakeStore(1)
    wrapper = _wrapper(1, worker, store)
    wrapper.vllm_config.scheduler_config.enable_request_owned_sampling = True
    step = _output(step_seq=7)
    step.owner_commands = [
        OwnerCommand(
            key=OwnerLeaseKey("remote-restore", 0),
            owner_id=0,
            command_seq=4,
            kind=OwnerCommandKind.RESTORE,
            required_num_tokens=10,
        ),
        OwnerCommand(
            key=OwnerLeaseKey("remote-extend", 0),
            owner_id=0,
            command_seq=5,
            kind=OwnerCommandKind.EXTEND,
            required_num_tokens=12,
        ),
    ]

    with pytest.raises(RuntimeError, match="exclusive zero-token global"):
        wrapper.execute_model(step)

    assert worker.calls == 0
    assert worker.sample_calls == 0
    assert store.calls == []


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


@pytest.mark.parametrize("accepted_draft_count", range(8))
def test_real_store_deferred_dspark_transaction_commits_verified_prefix(
    accepted_draft_count: int,
) -> None:
    """Compose the public worker seam with the real owner-local KV store.

    A K=7 target verification physically executes all K+1 positions, while
    the deferred sampler emits each possible accepted prefix A plus its one
    terminal correction/bonus token.  The wrapper must carry that evidence
    through the exact metadata handoff and atomically commit only A+1 logical
    positions without discarding or reallocating the provisional tail.
    """
    emitted = list(range(accepted_draft_count)) + [99]
    worker = _ScriptedDeferredWorker(
        ModelRunnerOutput(
            req_ids=["spec"],
            req_id_to_index={"spec": 0},
            sampled_token_ids=[emitted],
        )
    )
    wrapper = _speculative_real_wrapper(worker)
    key = OwnerLeaseKey("spec", 0)

    reserve = _output(step_seq=1)
    reserve.owner_commands = [
        _real_reserve(
            owner_id=0,
            command_seq=1,
            required=16,
            request_id="spec",
        )
    ]
    reserve_result = wrapper.execute_model(reserve)
    assert reserve_result.owner_receipt_batches[0].events[0].accepted

    verify = _output(step_seq=2)
    verify.total_num_scheduled_tokens = 8
    verify.num_scheduled_tokens = {"spec": 8}
    verify.scheduled_spec_decode_tokens = {"spec": list(range(7))}
    verify.scheduled_owner_leases = [
        OwnerLeaseToken(
            key=key,
            owner_id=0,
            step_seq=2,
            command_seq=1,
            runnable_num_tokens=16,
        )
    ]

    assert wrapper.execute_model(verify) is None
    metadata = worker.metadata_handoffs[-1]
    (entry,) = metadata.entries
    assert entry.pre_step_num_computed_tokens == 0
    assert entry.post_step_num_tokens == 8
    assert entry.num_speculative_tokens == 7
    provisional_tables = entry.tables

    result = wrapper.sample_tokens(object())
    assert result.sampled_token_ids == [emitted]
    assert result.owner_receipt_batches[0].emitted_step_seq == 2
    assert worker.sample_calls == 1
    store = wrapper._request_owned_kv_store
    assert store is not None
    assert store._records[key].num_computed_tokens == accepted_draft_count + 1
    assert store._records[key].reserved_num_tokens == 16
    assert store.get_block_table(key) == provisional_tables
    assert store._pending_marks == {}


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
    # The wrapper first actively cleared stale runner state with a None
    # handoff, then delivered the immutable metadata through the private
    # hook, and the wire objects were untouched.
    assert worker.metadata_handoffs == [None, metadata]


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


def test_g3_handoff_delivers_empty_heartbeat_metadata() -> None:
    """The hook receives the empty heartbeat metadata after a successful
    build, and no wire object carries or is mutated by the handoff."""
    step = _output(step_seq=1)
    command = _reserve(owner_id=0, command_seq=1)
    step.owner_commands = [command]
    worker = _FakeWorker()
    store = _FakeStore(0)
    wrapper = _wrapper(0, worker, store)

    result = wrapper.execute_model(step)
    metadata = wrapper._request_owned_step_metadata

    # The zero-token terminal gate still runs the underlying worker, and the
    # delivered metadata is exactly the built empty heartbeat batch, after
    # the start-of-call None clear.
    assert worker.calls == 1
    assert worker.metadata_handoffs == [None, metadata]
    assert metadata is wrapper._request_owned_step_metadata
    assert metadata.step_seq == 1
    assert metadata.owner_rank == 0
    assert metadata.entries == ()
    assert store.last_build_counts == {}

    # No wire attachment or mutation: the command object is untouched and no
    # scheduler-facing object carries the metadata.
    assert step.owner_commands[0] is command
    assert not hasattr(step, "request_owned_step_metadata")
    assert not hasattr(result, "request_owned_step_metadata")


def test_g3_handoff_fails_closed_for_unsupported_worker() -> None:
    """A worker without the private hook fails closed on the start-of-call
    None clear, before any command processing or underlying worker run."""
    step = _output(step_seq=1)
    step.owner_commands = [_reserve(owner_id=0, command_seq=1)]
    worker = _NoHookWorker()
    store = _FakeStore(0)
    wrapper = _wrapper(0, worker, store)

    with pytest.raises(RuntimeError, match="does not support the request-owned"):
        wrapper.execute_model(step)
    assert worker.calls == 0
    # The fail-closed None delivery happens before any command processing,
    # so the store was never touched.
    assert store.calls == []
    assert wrapper._request_owned_step_metadata is None


def test_g3_stale_metadata_cleared_at_start_of_request_owned_call() -> None:
    """Stale metadata from a previous step is cleared at the start of the
    next request-owned call, so a failure before the next successful build
    never exposes it."""
    step = _output(step_seq=1)
    step.owner_commands = [_reserve(owner_id=0, command_seq=1)]
    worker = _FakeWorker()
    store = _FakeStore(0)
    wrapper = _wrapper(0, worker, store)

    wrapper.execute_model(step)
    metadata1 = wrapper._request_owned_step_metadata
    assert metadata1 is not None
    assert worker.metadata_handoffs == [None, metadata1]

    # A token-bearing schedule is refused by the existing zero-token gate;
    # the stale metadata was already cleared at the start of the call.
    token_step = _output(step_seq=2)
    token_step.owner_commands = [_reserve(owner_id=0, command_seq=2)]
    token_step.total_num_scheduled_tokens = 8
    token_step.num_scheduled_tokens = {"req": 8}
    with pytest.raises(RuntimeError, match="control-only"):
        wrapper.execute_model(token_step)
    assert wrapper._request_owned_step_metadata is None
    # The concrete worker's runner state was actively cleared with a None
    # handoff at the start of the second call, before the gate refusal.
    assert worker.metadata_handoffs == [None, metadata1, None]
    # The gate refusal never reached the worker's execute_model.
    assert worker.calls == 1
