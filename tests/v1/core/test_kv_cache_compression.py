# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from dataclasses import replace
from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.utils import create_scheduler
from vllm.config import KVCacheCompressionConfig
from vllm.platforms.interface import Platform
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import (
    KVCacheCompressionCommitResult,
    KVCacheManager,
)
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import core as engine_core_module
from vllm.v1.engine.core import (
    EngineCore,
    _scheduler_accepts_kv_cache_compression_runtime_spec,
)
from vllm.v1.kv_cache_compression import (
    KVCacheCompressionCompatibility,
    KVCacheCompressionError,
    KVCacheCompressionPlan,
    KVCacheCompressionRuntimeSpec,
    ensure_kv_cache_compression_compatible,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.kv_cache_spec_registry import _REGISTRY_KVCACHESPEC_LIST
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.worker.worker_base import WorkerBase

init_none_hash(sha256)


@pytest.fixture(autouse=True)
def restore_kv_cache_spec_registry():
    registry = _REGISTRY_KVCACHESPEC_LIST.copy()
    _REGISTRY_KVCACHESPEC_LIST.clear()
    yield
    _REGISTRY_KVCACHESPEC_LIST.clear()
    _REGISTRY_KVCACHESPEC_LIST.update(registry)


def _config() -> KVCacheCompressionConfig:
    return KVCacheCompressionConfig(
        provider="pyramidkv_ascend",
        provider_config={"window_size": 8},
    )


def _runtime_spec() -> KVCacheCompressionRuntimeSpec:
    return KVCacheCompressionRuntimeSpec(
        schema_version=1,
        provider="pyramidkv_ascend",
        requires_private_destination=True,
        compression_threshold_tokens=256,
        required_recompute_tokens=8,
        max_physical_num_tokens=512,
    )


def _report(
    *,
    supported: bool = True,
    reasons: tuple[str, ...] = (),
    provider: str = "pyramidkv_ascend",
    schema_version: int = 1,
    platform: str = "npu",
) -> KVCacheCompressionCompatibility:
    return KVCacheCompressionCompatibility(
        schema_version=schema_version,
        provider=provider,
        supported=supported,
        reasons=reasons,
        platform=platform,
        provider_factory="fake.provider:create",
        runtime_spec=_runtime_spec() if supported else None,
    )


def test_platform_default_has_no_provider_factory() -> None:
    assert Platform.get_kv_cache_compression_provider_factory() is None


def test_disabled_engine_skips_worker_rpc() -> None:
    calls = 0

    def collective_rpc(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("disabled path must not issue an RPC")

    engine = SimpleNamespace(collective_rpc=collective_rpc)
    config = SimpleNamespace(kv_cache_compression_config=None)

    EngineCore._validate_kv_cache_compression(engine, config)

    assert calls == 0


def test_enabled_engine_uses_worker_report() -> None:
    calls = []

    def collective_rpc(method):
        calls.append(method)
        return [_report()]

    engine = SimpleNamespace(collective_rpc=collective_rpc)
    config = SimpleNamespace(kv_cache_compression_config=_config())

    EngineCore._validate_kv_cache_compression(engine, config)

    assert calls == ["validate_kv_cache_compression"]


def test_compatibility_check_precedes_memory_and_kv_allocation(monkeypatch) -> None:
    events = []

    def get_kv_cache_specs():
        events.append("specs")
        return [{"layer": object()}]

    def determine_available_memory():
        events.append("memory")
        return [1024]

    def initialize_from_config(configs):
        events.append("initialize")

    model_executor = SimpleNamespace(
        get_kv_cache_specs=get_kv_cache_specs,
        determine_available_memory=determine_available_memory,
        initialize_from_config=initialize_from_config,
    )
    scheduler_cache_config = SimpleNamespace(num_blocks=0, kv_cache_groups=[])
    monkeypatch.setattr(
        engine_core_module, "register_all_kvcache_specs", lambda config: None
    )
    monkeypatch.setattr(
        engine_core_module,
        "get_kv_cache_configs",
        lambda config, specs, memory: [object()],
    )
    monkeypatch.setattr(
        engine_core_module,
        "generate_scheduler_kv_cache_config",
        lambda configs: scheduler_cache_config,
    )
    engine = SimpleNamespace(
        model_executor=model_executor,
        available_gpu_memory_for_kv_cache=-1,
        _validate_kv_cache_compression=lambda config: events.append("validate"),
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1024),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        compilation_config=SimpleNamespace(
            compilation_time=0,
            encoder_compilation_time=0,
        ),
        validate_block_size=lambda: None,
    )

    result = EngineCore._initialize_kv_caches(engine, config)

    assert result is scheduler_cache_config
    assert events == ["specs", "validate", "memory", "initialize"]


def test_async_scheduler_accepts_runtime_spec_via_kwargs() -> None:
    class UnsupportedScheduler:
        def __init__(self, value) -> None:
            self.value = value

    assert _scheduler_accepts_kv_cache_compression_runtime_spec(Scheduler)
    assert _scheduler_accepts_kv_cache_compression_runtime_spec(AsyncScheduler)
    assert not _scheduler_accepts_kv_cache_compression_runtime_spec(
        UnsupportedScheduler
    )


def test_disabled_engine_preserves_legacy_scheduler_constructor(monkeypatch) -> None:
    class SchedulerConstructed(Exception):
        pass

    class LegacyScheduler:
        def __init__(
            self,
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            hash_block_size=None,
            include_finished_set=False,
            log_stats=False,
        ) -> None:
            raise SchedulerConstructed

    class FakeExecutor:
        def __init__(self, vllm_config) -> None:
            pass

    def initialize_kv_caches(engine, vllm_config):
        engine.kv_cache_compression_runtime_spec = None
        return SimpleNamespace(kv_cache_groups=[object()])

    monkeypatch.setattr("vllm.plugins.load_general_plugins", lambda: None)
    monkeypatch.setattr(
        engine_core_module.EngineCore,
        "_initialize_kv_caches",
        initialize_kv_caches,
    )
    monkeypatch.setattr(
        engine_core_module,
        "StructuredOutputManager",
        lambda vllm_config: object(),
    )
    monkeypatch.setattr(
        engine_core_module,
        "resolve_kv_cache_block_sizes",
        lambda kv_cache_config, vllm_config: (128, 128),
    )
    monkeypatch.setattr(
        engine_core_module.envs,
        "VLLM_ELASTIC_EP_SCALE_UP_LAUNCH",
        False,
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_rank_local=True),
        scheduler_config=SimpleNamespace(
            get_scheduler_cls=lambda: LegacyScheduler,
            enable_chunked_prefill=False,
        ),
    )

    with pytest.raises(SchedulerConstructed):
        EngineCore(config, FakeExecutor, log_stats=False)


def test_all_worker_incompatibilities_are_reported() -> None:
    reports = [
        _report(
            supported=False,
            reasons=("backend unsupported", "dtype unsupported"),
        ),
        _report(
            supported=False,
            reasons=("block size unsupported",),
            platform="npu:1",
        ),
    ]

    with pytest.raises(KVCacheCompressionError) as exc_info:
        ensure_kv_cache_compression_compatible(_config(), reports)

    message = str(exc_info.value)
    assert "worker 0 (npu): backend unsupported" in message
    assert "worker 0 (npu): dtype unsupported" in message
    assert "worker 1 (npu:1): block size unsupported" in message


@pytest.mark.parametrize(
    "report",
    [
        _report(provider="wrong"),
        _report(schema_version=2),
        _report(supported=True, reasons=("contradictory",)),
    ],
)
def test_malformed_worker_report_is_rejected(report) -> None:
    with pytest.raises(KVCacheCompressionError):
        ensure_kv_cache_compression_compatible(_config(), [report])


def test_worker_runtime_specs_must_match() -> None:
    mismatched = replace(
        _report(),
        runtime_spec=replace(_runtime_spec(), max_physical_num_tokens=513),
    )
    with pytest.raises(KVCacheCompressionError, match="runtime spec"):
        ensure_kv_cache_compression_compatible(_config(), [_report(), mismatched])


def test_base_worker_does_not_import_provider() -> None:
    provider_module = "fake_kv_cache_compression_provider"
    sys.modules.pop(provider_module, None)
    platform = SimpleNamespace(
        device_type="fake",
        get_kv_cache_compression_provider_factory=lambda: f"{provider_module}:create",
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(kv_cache_compression_config=_config()),
        current_platform=platform,
    )

    report = WorkerBase.validate_kv_cache_compression(worker)

    assert report.supported is False
    assert report.provider_factory == f"{provider_module}:create"
    assert provider_module not in sys.modules


def _manager(
    *,
    block_size: int = 128,
    num_blocks: int = 8,
    enable_caching: bool = False,
    compression_config: KVCacheCompressionConfig | None = None,
    compression_threshold_tokens: int = 256,
    required_recompute_tokens: int = 8,
) -> KVCacheManager:
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )
    cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer0", "layer1"], spec)],
    )
    return KVCacheManager(
        kv_cache_config=cache_config,
        max_model_len=1024,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        enable_caching=enable_caching,
        kv_cache_compression_config=compression_config,
        kv_cache_compression_runtime_spec=(
            replace(
                _runtime_spec(),
                compression_threshold_tokens=compression_threshold_tokens,
                required_recompute_tokens=required_recompute_tokens,
            )
            if compression_config is not None
            else None
        ),
    )


def _running_request(request_id: str = "request", prompt_len: int = 384) -> Request:
    request = Request(
        request_id=request_id,
        prompt_token_ids=[1] * prompt_len,
        sampling_params=SamplingParams(max_tokens=128),
        pooling_params=None,
        block_hasher=get_request_block_hasher(128, sha256),
    )
    request.status = RequestStatus.RUNNING
    return request


def _allocated_prefill(
    manager: KVCacheManager,
    request: Request,
) -> tuple[int, ...]:
    blocks = manager.allocate_slots(
        request,
        num_new_tokens=request.num_prompt_tokens,
        reserved_blocks=manager.get_num_compression_destination_blocks(request),
        has_scheduled_reqs=False,
    )
    assert blocks is not None
    if manager.get_num_compression_destination_blocks(request):
        manager.reserve_compression_destination(request)
    request.num_computed_tokens = request.num_prompt_tokens
    return tuple(blocks.get_block_ids()[0])


def _plan(
    request: Request,
    expected_block_ids: tuple[int, ...],
) -> KVCacheCompressionPlan:
    return KVCacheCompressionPlan(
        schema_version=1,
        provider="pyramidkv_ascend",
        request_id=request.request_id,
        semantic_num_tokens=request.num_prompt_tokens,
        physical_num_tokens=192,
        per_layer_physical_num_tokens=(("layer0", 192), ("layer1", 160)),
        expected_block_ids=(expected_block_ids,),
    )


def _commit_result() -> KVCacheCompressionCommitResult:
    return KVCacheCompressionCommitResult(
        source_block_ids=(0, 1, 2),
        destination_block_ids=(0, 1),
        released_block_ids=(2,),
        retained_hashed_source_block_ids=(),
    )


def _scheduler_output_with_transactions(
    transactions: dict[str, int],
) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.kv_cache_compression_transaction_ids = transactions
    return output


def _async_compression_scheduler(
    *,
    enable_prefix_caching: bool = False,
    long_prefill_token_threshold: int = 0,
    num_blocks: int = 64,
) -> AsyncScheduler:
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_model_len=1024,
        block_size=128,
        num_blocks=num_blocks,
        async_scheduling=True,
        enable_prefix_caching=enable_prefix_caching,
        long_prefill_token_threshold=long_prefill_token_threshold,
        kv_cache_compression_config=_config(),
        kv_cache_compression_runtime_spec=_runtime_spec(),
    )
    assert isinstance(scheduler, AsyncScheduler)
    return scheduler


def _waiting_request(request_id: str, prompt_len: int) -> Request:
    request = _running_request(request_id, prompt_len)
    request.status = RequestStatus.WAITING
    return request


def _single_layer_plan(
    request: Request, expected_block_ids: tuple[int, ...]
) -> KVCacheCompressionPlan:
    return KVCacheCompressionPlan(
        schema_version=1,
        provider="pyramidkv_ascend",
        request_id=request.request_id,
        semantic_num_tokens=request.num_prompt_tokens,
        physical_num_tokens=192,
        per_layer_physical_num_tokens=(("layer", 192),),
        expected_block_ids=(expected_block_ids,),
    )


def test_compression_plan_is_pickle_serializable() -> None:
    request = _running_request()
    plan = _plan(request, (0, 1, 2))

    assert ForkingPickler.loads(ForkingPickler.dumps(plan)) == plan


def test_plan_commit_reclaims_tail_and_decode_uses_physical_length() -> None:
    manager = _manager(compression_config=_config())
    request = _running_request()
    block_ids = _allocated_prefill(manager, request)
    free_before = manager.block_pool.get_num_free_blocks()

    commit = manager.apply_compression_plan(request, _plan(request, block_ids))

    assert commit.released_block_ids == (block_ids[-1],)
    assert manager.get_block_ids(request.request_id) == ([*block_ids[:2]],)
    assert manager.block_pool.get_num_free_blocks() == free_before + 1
    assert manager.get_compressed_physical_num_tokens(request.request_id) == 192

    decode_blocks = manager.allocate_slots(request, num_new_tokens=1)
    assert decode_blocks is not None
    assert decode_blocks.get_block_ids() == ([],)
    assert manager.get_block_ids(request.request_id) == ([*block_ids[:2]],)
    assert manager.get_compressed_physical_num_tokens(request.request_id) == 193

    other = _running_request("other", prompt_len=128)
    other_blocks = manager.allocate_slots(
        other,
        num_new_tokens=other.num_prompt_tokens,
        has_scheduled_reqs=False,
    )
    assert other_blocks is not None
    assert other_blocks.get_block_ids() == ([commit.released_block_ids[0]],)

    manager.free(request)
    assert manager.get_compressed_physical_num_tokens(request.request_id) is None
    manager.free(other)
    # One of the configured blocks is the pool's permanent null block.
    assert manager.block_pool.get_num_free_blocks() == 7


def test_prefix_cache_commit_swaps_to_private_unhashed_destination() -> None:
    manager = _manager(enable_caching=True, compression_config=_config())
    request = _running_request()
    source_ids = _allocated_prefill(manager, request)
    destination_ids = manager.get_compression_destination_block_ids(request.request_id)
    assert destination_ids is not None
    assert set(source_ids).isdisjoint(destination_ids[0])
    new_block_ids = manager.take_new_block_ids()
    assert set(destination_ids[0]).issubset(new_block_ids)
    assert manager.take_new_block_ids() == []
    source_blocks = [manager.block_pool.blocks[i] for i in source_ids]
    assert all(block.block_hash is not None for block in source_blocks)

    commit = manager.apply_compression_plan(request, _plan(request, source_ids))

    assert commit.source_block_ids == source_ids
    assert commit.destination_block_ids == tuple(destination_ids[0][:2])
    assert manager.get_block_ids(request.request_id) == (
        list(commit.destination_block_ids),
    )
    assert commit.retained_hashed_source_block_ids == source_ids
    assert all(block.ref_cnt == 0 for block in source_blocks)
    destination_blocks = [
        manager.block_pool.blocks[i] for i in commit.destination_block_ids
    ]
    assert all(
        block.ref_cnt == 1 and block.block_hash is None for block in destination_blocks
    )

    manager.allocate_slots(request, num_new_tokens=1)
    assert all(block.block_hash is None for block in destination_blocks)
    manager.free(request)


@pytest.mark.parametrize(
    ("required_recompute_tokens", "expected_cached_tokens"),
    [(1, 768), (8, 640), (130, 512)],
)
def test_required_recompute_tokens_caps_prefix_cache_admission(
    required_recompute_tokens: int,
    expected_cached_tokens: int,
) -> None:
    manager = _manager(
        enable_caching=True,
        compression_config=_config(),
        required_recompute_tokens=required_recompute_tokens,
    )
    warm = _running_request("warm", prompt_len=769)
    blocks = manager.allocate_slots(
        warm,
        num_new_tokens=warm.num_prompt_tokens,
        has_scheduled_reqs=False,
    )
    assert blocks is not None
    manager.free(warm)

    cached = _running_request("cached", prompt_len=769)
    computed, num_tokens = manager.get_computed_blocks(cached)

    assert num_tokens == expected_cached_tokens
    assert cached.num_prompt_tokens - num_tokens >= required_recompute_tokens
    assert len(computed.get_block_ids()[0]) == expected_cached_tokens // 128


def test_recompute_window_clamps_short_prompt_cache_admission() -> None:
    manager = _manager(
        enable_caching=True,
        compression_config=_config(),
        compression_threshold_tokens=2,
        required_recompute_tokens=8,
    )
    request = _running_request(prompt_len=4)

    computed, num_tokens = manager.get_computed_blocks(request)

    assert num_tokens == 0
    assert computed.get_block_ids() == ([],)


def test_prefix_plan_requires_live_private_destination() -> None:
    manager = _manager(enable_caching=True, compression_config=_config())
    request = _running_request()
    blocks = manager.allocate_slots(
        request,
        num_new_tokens=request.num_prompt_tokens,
        has_scheduled_reqs=False,
    )
    assert blocks is not None
    request.num_computed_tokens = request.num_prompt_tokens
    source_ids = tuple(blocks.get_block_ids()[0])

    with pytest.raises(KVCacheCompressionError, match="no private"):
        manager.apply_compression_plan(request, _plan(request, source_ids))

    manager.free(request)


def test_abort_releases_uncommitted_private_destination() -> None:
    manager = _manager(enable_caching=True, compression_config=_config())
    request = _running_request()
    _allocated_prefill(manager, request)
    manager.free(request)
    assert manager.get_compression_destination_block_ids(request.request_id) is None
    assert manager.block_pool.get_num_free_blocks() == 7


def test_source_and_private_destination_must_fit_atomically() -> None:
    manager = _manager(
        num_blocks=6,
        enable_caching=True,
        compression_config=_config(),
    )
    request = _running_request()
    free_before = manager.block_pool.get_num_free_blocks()

    blocks = manager.allocate_slots(
        request,
        num_new_tokens=request.num_prompt_tokens,
        reserved_blocks=manager.get_num_compression_destination_blocks(request),
        has_scheduled_reqs=False,
    )

    assert blocks is None
    assert manager.get_compression_destination_block_ids(request.request_id) is None
    assert manager.block_pool.get_num_free_blocks() == free_before


def test_shared_prefix_requests_receive_disjoint_private_destinations() -> None:
    manager = _manager(
        num_blocks=16,
        enable_caching=True,
        compression_config=_config(),
    )
    warm = _running_request("warm")
    warm_blocks = manager.allocate_slots(
        warm,
        num_new_tokens=warm.num_prompt_tokens,
        has_scheduled_reqs=False,
    )
    assert warm_blocks is not None
    manager.free(warm)

    requests = [_running_request("first"), _running_request("second")]
    sources = []
    destinations = []
    for request in requests:
        computed_blocks, num_computed_tokens = manager.get_computed_blocks(request)
        assert num_computed_tokens == 256
        new_blocks = manager.allocate_slots(
            request,
            num_new_tokens=request.num_prompt_tokens - num_computed_tokens,
            num_new_computed_tokens=num_computed_tokens,
            new_computed_blocks=computed_blocks,
            reserved_blocks=manager.get_num_compression_destination_blocks(request),
            has_scheduled_reqs=False,
        )
        assert new_blocks is not None
        manager.reserve_compression_destination(request)
        sources.append(manager.get_block_ids(request.request_id)[0])
        destination = manager.get_compression_destination_block_ids(request.request_id)
        assert destination is not None
        destinations.append(destination[0])

    shared_source_ids = (
        set(sources[0])
        .intersection(sources[1])
        .difference({manager.block_pool.null_block.block_id})
    )
    assert shared_source_ids
    assert set(sources[0]).isdisjoint(destinations[0])
    assert set(sources[1]).isdisjoint(destinations[1])
    assert set(destinations[0]).isdisjoint(destinations[1])
    for block_id in shared_source_ids:
        assert manager.block_pool.blocks[block_id].ref_cnt == 2

    for request in requests:
        manager.free(request)


def test_partial_prefill_plan_is_rejected_and_final_boundary_is_accepted() -> None:
    manager = _manager(compression_config=_config())
    request = _running_request()
    block_ids = _allocated_prefill(manager, request)
    free_before = manager.block_pool.get_num_free_blocks()
    request.num_computed_tokens = 256

    with pytest.raises(
        KVCacheCompressionError,
        match="not at the completed full-prefill boundary",
    ):
        manager.apply_compression_plan(request, _plan(request, block_ids))

    assert manager.get_block_ids(request.request_id) == ([*block_ids],)
    assert manager.block_pool.get_num_free_blocks() == free_before
    request.num_computed_tokens = request.num_prompt_tokens
    commit = manager.apply_compression_plan(request, _plan(request, block_ids))
    assert commit.released_block_ids == (block_ids[-1],)


def test_commit_ack_is_not_visible_until_plan_commit_and_next_schedule() -> None:
    request = _running_request()
    plan = _plan(request, (0, 1, 2))
    scheduler = SimpleNamespace(
        _pending_kv_cache_compression_block_table_updates=set(),
        _inflight_kv_cache_compression_transactions={request.request_id: 1},
        requests={request.request_id: request},
        kv_cache_manager=SimpleNamespace(
            validate_compression_plan=lambda actual_request, actual_plan: 2,
            apply_compression_plan=lambda actual_request, actual_plan: _commit_result(),
            get_block_ids=lambda request_id: ([0, 1],),
        ),
    )

    assert (
        Scheduler._take_kv_cache_compression_block_table_updates(
            scheduler, {request.request_id: 1}
        )
        is None
    )
    Scheduler._apply_kv_cache_compression_plans(
        scheduler,
        _scheduler_output_with_transactions({request.request_id: 1}),
        ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            kv_cache_compression_plans=[plan],
        ),
    )
    assert (
        Scheduler._take_kv_cache_compression_block_table_updates(
            scheduler, {"other": 1}
        )
        is None
    )
    assert Scheduler._take_kv_cache_compression_block_table_updates(
        scheduler, {request.request_id: 1}
    ) == {request.request_id: ([0, 1],)}


def test_stale_plan_is_rejected_without_mutating_blocks_or_pool() -> None:
    manager = _manager(compression_config=_config())
    request = _running_request()
    block_ids = _allocated_prefill(manager, request)
    free_before = manager.block_pool.get_num_free_blocks()
    stale_plan = replace(
        _plan(request, block_ids),
        expected_block_ids=((block_ids[0], block_ids[2], block_ids[1]),),
    )

    with pytest.raises(KVCacheCompressionError, match="block table changed"):
        manager.apply_compression_plan(request, stale_plan)

    assert manager.get_block_ids(request.request_id) == ([*block_ids],)
    assert manager.block_pool.get_num_free_blocks() == free_before
    assert manager.get_compressed_physical_num_tokens(request.request_id) is None


@pytest.mark.parametrize(
    ("plan_update", "error_match"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"provider": "wrong"}, "does not match"),
        ({"request_id": "wrong"}, "request_id"),
        (
            {"per_layer_physical_num_tokens": (("layer0", 192),)},
            "layers do not match",
        ),
        (
            {"per_layer_physical_num_tokens": (("layer0", 191), ("layer1", 160))},
            "must equal the maximum",
        ),
    ],
)
def test_invalid_plan_is_rejected_before_commit(plan_update, error_match) -> None:
    manager = _manager(compression_config=_config())
    request = _running_request()
    block_ids = _allocated_prefill(manager, request)
    free_before = manager.block_pool.get_num_free_blocks()
    plan = replace(_plan(request, block_ids), **plan_update)

    with pytest.raises(KVCacheCompressionError, match=error_match):
        manager.apply_compression_plan(request, plan)

    assert manager.get_block_ids(request.request_id) == ([*block_ids],)
    assert manager.block_pool.get_num_free_blocks() == free_before
    assert manager.get_compressed_physical_num_tokens(request.request_id) is None


def test_disabled_transfer_and_wrong_block_size_are_rejected() -> None:
    cases: list[tuple[KVCacheManager, dict | None, str]] = [
        (_manager(), None, "feature is disabled"),
        (_manager(compression_config=_config()), {}, "KV transfer"),
    ]
    for index, (manager, kv_transfer_params, error_match) in enumerate(cases):
        request = _running_request(f"request-{index}")
        request.num_computed_tokens = request.num_prompt_tokens
        request.kv_transfer_params = kv_transfer_params
        free_before = manager.block_pool.get_num_free_blocks()

        with pytest.raises(KVCacheCompressionError, match=error_match):
            manager.apply_compression_plan(request, _plan(request, (1, 2, 3)))
        assert manager.block_pool.get_num_free_blocks() == free_before

    with pytest.raises(KVCacheCompressionError, match="block_size"):
        _manager(block_size=64, compression_config=_config())


def test_repeated_plan_and_unsupported_decode_modes_are_rejected() -> None:
    manager = _manager(compression_config=_config())
    request = _running_request()
    block_ids = _allocated_prefill(manager, request)
    plan = _plan(request, block_ids)
    manager.apply_compression_plan(request, plan)

    with pytest.raises(KVCacheCompressionError, match="already compressed"):
        manager.apply_compression_plan(request, plan)
    with pytest.raises(KVCacheCompressionError, match="exactly one"):
        manager.allocate_slots(request, num_new_tokens=2)
    with pytest.raises(KVCacheCompressionError, match="ordinary single-token"):
        manager.allocate_slots(request, num_new_tokens=1, num_lookahead_tokens=1)


def test_scheduler_sends_full_replacement_once_request_is_scheduled() -> None:
    scheduler = SimpleNamespace(
        _pending_kv_cache_compression_block_table_updates={"request"},
        kv_cache_manager=SimpleNamespace(get_block_ids=lambda request_id: ([2, 3, 7],)),
    )

    assert (
        Scheduler._take_kv_cache_compression_block_table_updates(
            scheduler, {"other": 1}
        )
        is None
    )
    assert scheduler._pending_kv_cache_compression_block_table_updates == {"request"}
    assert Scheduler._take_kv_cache_compression_block_table_updates(
        scheduler, {"request": 1}
    ) == {"request": ([2, 3, 7],)}
    assert not scheduler._pending_kv_cache_compression_block_table_updates


def test_scheduler_commits_plans_and_rejects_them_when_disabled() -> None:
    request = _running_request()
    plan = _plan(request, (0, 1, 2))
    calls = []
    validations = []

    def validate_compression_plan(actual_request, actual_plan):
        validations.append((actual_request, actual_plan))
        return 2

    def apply_compression_plan(actual_request, actual_plan):
        calls.append((actual_request, actual_plan))
        return _commit_result()

    output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        kv_cache_compression_plans=[plan],
    )
    scheduler = SimpleNamespace(
        _pending_kv_cache_compression_block_table_updates=set(),
        _inflight_kv_cache_compression_transactions={request.request_id: 1},
        requests={request.request_id: request},
        kv_cache_manager=SimpleNamespace(
            validate_compression_plan=validate_compression_plan,
            apply_compression_plan=apply_compression_plan,
        ),
    )

    scheduler_output = _scheduler_output_with_transactions({request.request_id: 1})
    Scheduler._apply_kv_cache_compression_plans(scheduler, scheduler_output, output)

    assert validations == [(request, plan)]
    assert calls == [(request, plan)]
    assert scheduler._pending_kv_cache_compression_block_table_updates == {
        request.request_id
    }

    scheduler._pending_kv_cache_compression_block_table_updates = None
    scheduler._inflight_kv_cache_compression_transactions = {request.request_id: 2}
    with pytest.raises(RuntimeError, match="feature is disabled"):
        Scheduler._apply_kv_cache_compression_plans(
            scheduler,
            _scheduler_output_with_transactions({request.request_id: 2}),
            output,
        )


def test_scheduler_validates_all_plans_before_reclaiming_any_blocks() -> None:
    manager = _manager(compression_config=_config())
    first = _running_request("first")
    second = _running_request("second")
    first_blocks = _allocated_prefill(manager, first)
    second_blocks = _allocated_prefill(manager, second)
    free_before = manager.block_pool.get_num_free_blocks()
    stale_second = replace(
        _plan(second, second_blocks),
        expected_block_ids=((second_blocks[0], second_blocks[2], second_blocks[1]),),
    )
    output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        kv_cache_compression_plans=[
            _plan(first, first_blocks),
            stale_second,
        ],
    )
    scheduler = SimpleNamespace(
        _pending_kv_cache_compression_block_table_updates=set(),
        _inflight_kv_cache_compression_transactions={"first": 1, "second": 1},
        requests={first.request_id: first, second.request_id: second},
        kv_cache_manager=manager,
    )

    with pytest.raises(KVCacheCompressionError, match="block table changed"):
        Scheduler._apply_kv_cache_compression_plans(
            scheduler,
            _scheduler_output_with_transactions({"first": 1, "second": 1}),
            output,
        )

    assert manager.get_block_ids(first.request_id) == ([*first_blocks],)
    assert manager.get_block_ids(second.request_id) == ([*second_blocks],)
    assert manager.block_pool.get_num_free_blocks() == free_before
    assert not scheduler._pending_kv_cache_compression_block_table_updates


def test_scheduler_rejects_missing_duplicate_and_unexpected_plans() -> None:
    request = _running_request()
    plan = _plan(request, (0, 1, 2))
    scheduler = SimpleNamespace(
        _pending_kv_cache_compression_block_table_updates=set(),
        _inflight_kv_cache_compression_transactions={request.request_id: 1},
        requests={request.request_id: request},
        kv_cache_manager=SimpleNamespace(),
    )
    scheduler_output = _scheduler_output_with_transactions({request.request_id: 1})

    with pytest.raises(RuntimeError, match="did not return"):
        Scheduler._apply_kv_cache_compression_plans(
            scheduler,
            scheduler_output,
            ModelRunnerOutput(req_ids=[], req_id_to_index={}),
        )

    duplicate_output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        kv_cache_compression_plans=[plan, plan],
    )
    with pytest.raises(RuntimeError, match="duplicate"):
        Scheduler._apply_kv_cache_compression_plans(
            scheduler, scheduler_output, duplicate_output
        )

    unexpected_output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        kv_cache_compression_plans=[plan],
    )
    with pytest.raises(RuntimeError, match="unexpected"):
        Scheduler._apply_kv_cache_compression_plans(
            scheduler,
            _scheduler_output_with_transactions({"other": 1}),
            unexpected_output,
        )


def test_stale_transaction_output_is_ignored_after_same_id_restarts() -> None:
    request = _running_request()
    plan = _plan(request, (0, 1, 2))
    calls = []
    scheduler = SimpleNamespace(
        _pending_kv_cache_compression_block_table_updates=set(),
        _inflight_kv_cache_compression_transactions={request.request_id: 2},
        requests={request.request_id: request},
        kv_cache_manager=SimpleNamespace(
            validate_compression_plan=lambda *args: calls.append("validate"),
            apply_compression_plan=lambda *args: calls.append("apply"),
        ),
    )

    Scheduler._apply_kv_cache_compression_plans(
        scheduler,
        _scheduler_output_with_transactions({request.request_id: 1}),
        ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            kv_cache_compression_plans=[plan],
        ),
    )

    assert calls == []
    assert scheduler._inflight_kv_cache_compression_transactions == {
        request.request_id: 2
    }
    assert not scheduler._pending_kv_cache_compression_block_table_updates


@pytest.mark.parametrize("enable_prefix_caching", [False, True])
def test_async_scheduler_fences_final_prefill_until_commit_ack(
    enable_prefix_caching: bool,
) -> None:
    scheduler = _async_compression_scheduler(
        enable_prefix_caching=enable_prefix_caching
    )
    request = _waiting_request("compressed", 384)
    scheduler.add_request(request)

    final_prefill_output = scheduler.schedule()
    transaction_id = final_prefill_output.kv_cache_compression_transaction_ids
    assert transaction_id == {request.request_id: 1}
    assert final_prefill_output.num_scheduled_tokens == {request.request_id: 384}
    destinations = final_prefill_output.kv_cache_compression_destination_block_ids
    assert bool(destinations) is enable_prefix_caching

    blocked_output = scheduler.schedule()
    assert request.request_id not in blocked_output.num_scheduled_tokens

    other = _waiting_request("ordinary", 128)
    scheduler.add_request(other)
    overlap_output = scheduler.schedule()
    assert overlap_output.num_scheduled_tokens == {other.request_id: 128}
    assert overlap_output.kv_cache_compression_transaction_ids is None

    source_blocks = tuple(
        scheduler.kv_cache_manager.get_block_ids(request.request_id)[0]
    )
    Scheduler._apply_kv_cache_compression_plans(
        scheduler,
        final_prefill_output,
        ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            kv_cache_compression_plans=[_single_layer_plan(request, source_blocks)],
        ),
    )

    first_decode_output = scheduler.schedule()
    assert request.request_id in first_decode_output.num_scheduled_tokens
    assert first_decode_output.kv_cache_compression_block_table_updates == {
        request.request_id: scheduler.kv_cache_manager.get_block_ids(request.request_id)
    }
    next_output = scheduler.schedule()
    assert next_output.kv_cache_compression_block_table_updates is None


def test_async_chunked_prefill_only_fences_the_final_chunk() -> None:
    scheduler = _async_compression_scheduler(long_prefill_token_threshold=256)
    request = _waiting_request("chunked", 384)
    scheduler.add_request(request)

    intermediate = scheduler.schedule()
    assert intermediate.num_scheduled_tokens == {request.request_id: 256}
    assert intermediate.kv_cache_compression_transaction_ids is None

    final = scheduler.schedule()
    assert final.num_scheduled_tokens == {request.request_id: 128}
    assert final.kv_cache_compression_transaction_ids == {request.request_id: 2}

    blocked = scheduler.schedule()
    assert request.request_id not in blocked.num_scheduled_tokens


def test_async_abort_cancels_transaction_and_ignores_returned_plan() -> None:
    scheduler = _async_compression_scheduler()
    request = _waiting_request("aborted", 384)
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    source_blocks = tuple(
        scheduler.kv_cache_manager.get_block_ids(request.request_id)[0]
    )

    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    Scheduler._apply_kv_cache_compression_plans(
        scheduler,
        scheduler_output,
        ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            kv_cache_compression_plans=[_single_layer_plan(request, source_blocks)],
        ),
    )

    assert request.request_id not in scheduler.requests
    assert not scheduler._inflight_kv_cache_compression_transactions
    assert not scheduler._pending_kv_cache_compression_block_table_updates


def test_async_prefix_reset_restarts_request_with_a_new_transaction() -> None:
    scheduler = _async_compression_scheduler(enable_prefix_caching=True)
    request = _waiting_request("reset", 384)
    scheduler.add_request(request)
    stale_output = scheduler.schedule()
    stale_transaction_id = stale_output.kv_cache_compression_transaction_ids
    assert stale_transaction_id == {request.request_id: 1}
    stale_source_blocks = tuple(
        scheduler.kv_cache_manager.get_block_ids(request.request_id)[0]
    )

    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    assert not scheduler._inflight_kv_cache_compression_transactions

    restarted_output = scheduler.schedule()
    restarted_transaction_id = restarted_output.kv_cache_compression_transaction_ids
    assert restarted_transaction_id == {request.request_id: 2}

    Scheduler._apply_kv_cache_compression_plans(
        scheduler,
        stale_output,
        ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            kv_cache_compression_plans=[
                _single_layer_plan(request, stale_source_blocks)
            ],
        ),
    )

    assert scheduler._inflight_kv_cache_compression_transactions == {
        request.request_id: 2
    }
    assert not scheduler._pending_kv_cache_compression_block_table_updates


def test_preemption_selection_excludes_compression_transactions() -> None:
    eligible = SimpleNamespace(request_id="eligible", priority=10, arrival_time=10.0)
    fenced = SimpleNamespace(request_id="fenced", priority=100, arrival_time=100.0)

    class StubVictimSelector:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object, dict[str, object]]] = []

        def pick_victim(self, running, policy, **kwargs):
            self.calls.append((running, policy, kwargs))
            return running[0]

    victim_selector = StubVictimSelector()
    scheduler = SimpleNamespace(
        running=[eligible, fenced],
        policy=SchedulingPolicy.PRIORITY,
        victim_selector=victim_selector,
        kv_cache_manager=SimpleNamespace(usage=0.5),
        _inflight_kv_cache_compression_transactions={"fenced": 1},
    )

    assert Scheduler._select_preemption_candidate(scheduler, 42.0) is eligible
    assert victim_selector.calls == [
        (
            [eligible],
            SchedulingPolicy.PRIORITY,
            {"kv_utilization": 0.5, "now_s": 42.0},
        )
    ]


def test_normal_preemption_rejects_compression_transaction() -> None:
    request = _running_request()
    scheduler = SimpleNamespace(
        _inflight_kv_cache_compression_transactions={request.request_id: 1}
    )

    with pytest.raises(RuntimeError, match="compression transaction is in flight"):
        Scheduler._preempt_request(scheduler, request, 0.0)


def test_output_contracts_are_none_on_default_path() -> None:
    assert (
        ModelRunnerOutput(req_ids=[], req_id_to_index={}).kv_cache_compression_plans
        is None
    )
    assert SchedulerOutput.make_empty().kv_cache_compression_block_table_updates is None
    assert (
        SchedulerOutput.make_empty().kv_cache_compression_destination_block_ids is None
    )
    assert SchedulerOutput.make_empty().kv_cache_compression_transaction_ids is None
