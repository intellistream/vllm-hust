# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request

from .test_prefix_caching import make_kv_cache_config, make_kv_cache_manager

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 4


@pytest.fixture(autouse=True)
def _init_hash() -> None:
    init_none_hash(sha256)


def _request(
    request_id: str,
    tokens: list[int],
    *,
    domain: str = "tenant-a",
    tool_protocol: str = "tools-v1",
    admit: bool = True,
    min_reuse_tokens: int = 0,
) -> Request:
    params = SamplingParams(
        max_tokens=1,
        extra_args={
            "kv_prefix_sharing": {
                "identity": "shared-system-scaffold-v1",
                "share_domain": domain,
                "isolation": {"tool_protocol": tool_protocol},
                "admit": admit,
                "min_reuse_tokens": min_reuse_tokens,
            }
        },
    )
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _manager(num_blocks: int = 8):
    return make_kv_cache_manager(
        make_kv_cache_config(BLOCK_SIZE, num_blocks),
        max_model_len=64,
        enable_caching=True,
    )


def _publish(manager, request: Request) -> None:
    blocks, hit_tokens = manager.get_computed_blocks(request)
    assert hit_tokens == 0
    assert manager.allocate_slots(request, request.num_tokens) is not None
    manager.free(request)


def test_native_attach_consumes_kv_and_releases_ownership() -> None:
    manager = _manager()
    tokens = list(range(12))
    producer = _request("producer", tokens)
    _publish(manager, producer)

    consumer = _request("consumer", tokens)
    blocks, hit_tokens = manager.get_computed_blocks(consumer)
    assert hit_tokens == 8
    hit_ids = blocks.get_block_ids()[0]
    assert len(hit_ids) == 2

    assert (
        manager.allocate_slots(
            consumer,
            num_new_tokens=consumer.num_tokens - hit_tokens,
            num_new_computed_tokens=hit_tokens,
            new_computed_blocks=blocks,
        )
        is not None
    )
    assert manager.get_block_ids(consumer)[0][:2] == hit_ids
    assert all(manager.block_pool.blocks[block_id].ref_cnt == 1 for block_id in hit_ids)

    stats = manager.make_prefix_sharing_runtime_stats()
    assert stats["native_attached_blocks"] == 2
    assert stats["consumed_kv_tokens"] == 8
    assert stats["realized_reuse_requests"] == 1
    assert stats["avoided_prefill_tokens"] == 8
    assert stats["counter_provenance"] == "serving_runtime_kv_cache"
    assert stats["counter_semantics"] == "realized_runtime_reuse"

    manager.free(consumer)
    assert all(manager.block_pool.blocks[block_id].ref_cnt == 0 for block_id in hit_ids)
    assert manager.make_prefix_sharing_runtime_stats()["ownership_releases"] == 1


@pytest.mark.parametrize(
    ("domain", "tool_protocol"),
    [("tenant-b", "tools-v1"), ("tenant-a", "tools-v2")],
)
def test_tenant_and_tool_isolation_prevent_native_attach(
    domain: str, tool_protocol: str
) -> None:
    manager = _manager()
    tokens = list(range(12))
    _publish(manager, _request("producer", tokens))

    isolated = _request("isolated", tokens, domain=domain, tool_protocol=tool_protocol)
    blocks, hit_tokens = manager.get_computed_blocks(isolated)
    assert hit_tokens == 0
    assert blocks.get_block_ids() == ([],)


def test_admission_denial_is_fail_closed_for_reads_and_writes() -> None:
    manager = _manager()
    tokens = list(range(12))
    denied = _request("denied", tokens, admit=False)
    _publish(manager, denied)

    admitted = _request("admitted", tokens)
    _, hit_tokens = manager.get_computed_blocks(admitted)
    assert hit_tokens == 0
    stats = manager.make_prefix_sharing_runtime_stats()
    assert stats["denied_lookups"] == 1
    assert stats["fallback_reason_histogram"]["admission_denied"] == 1


def test_wait_threshold_declines_partial_native_hit_without_taking_ownership() -> None:
    manager = _manager()
    tokens = list(range(12))
    producer = _request("producer", tokens)
    _publish(manager, producer)
    cached_ids = [
        block.block_id
        for block in manager.block_pool.blocks
        if block.block_hash is not None
    ]

    consumer = _request("consumer", tokens, min_reuse_tokens=12)
    blocks, hit_tokens = manager.get_computed_blocks(consumer)
    assert hit_tokens == 0
    assert blocks.get_block_ids() == ([],)
    assert all(
        manager.block_pool.blocks[block_id].ref_cnt == 0 for block_id in cached_ids
    )
    stats = manager.make_prefix_sharing_runtime_stats()
    assert stats["fallback_reason_histogram"]["reuse_wait_threshold_unmet"] == 1
    assert stats["consumed_kv_tokens"] == 0


def test_allocation_exception_rolls_back_attached_native_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager()
    tokens = list(range(12))
    _publish(manager, _request("producer", tokens))
    consumer = _request("consumer", tokens)
    blocks, hit_tokens = manager.get_computed_blocks(consumer)
    hit_ids = blocks.get_block_ids()[0]

    def fail_allocate(*args, **kwargs):
        raise RuntimeError("injected allocation failure")

    monkeypatch.setattr(manager.coordinator, "allocate_new_blocks", fail_allocate)
    with pytest.raises(RuntimeError, match="injected allocation failure"):
        manager.allocate_slots(
            consumer,
            num_new_tokens=consumer.num_tokens - hit_tokens,
            num_new_computed_tokens=hit_tokens,
            new_computed_blocks=blocks,
        )

    assert all(manager.block_pool.blocks[block_id].ref_cnt == 0 for block_id in hit_ids)
    stats = manager.make_prefix_sharing_runtime_stats()
    assert stats["rollback_count"] == 1
    assert stats["realized_reuse_requests"] == 0
    assert stats["fallback_reason_histogram"]["allocation_exception_rollback"] == 1
