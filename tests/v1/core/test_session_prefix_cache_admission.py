# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-free tests for source-backed session prefix-cache admission."""

import pytest

from tests.v1.core.test_prefix_caching import (
    _make_hybrid_kv_cache_config,
    make_kv_cache_config,
    make_request,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import init_none_hash

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.fixture(autouse=True)
def _initialize_hash_function() -> None:
    init_none_hash(sha256)


def _controlled_request(
    request_id: str,
    token_ids: list[int],
    *,
    block_size: int,
    namespace: str,
    allow: bool,
):
    request = make_request(
        request_id,
        token_ids,
        block_size,
        sha256,
        cache_salt=namespace,
    )
    assert request.sampling_params is not None
    request.sampling_params.extra_args = {
        "kvplane_admit_prefix_cache": "allow" if allow else "deny",
        "kvplane_bypass_prefix_cache": "false" if allow else "true",
        "qbi_prefix_cache_policy_version": "session-apc-v2",
    }
    request.skip_reading_prefix_cache = request.get_skip_reading_prefix_cache()
    return request


def _run_prefill(manager: KVCacheManager, request):
    computed_blocks, hit_tokens = manager.get_computed_blocks(request)
    blocks = manager.allocate_slots(
        request,
        request.num_tokens - hit_tokens,
        hit_tokens,
        computed_blocks,
    )
    assert blocks is not None
    return hit_tokens


def _manager(*, block_size: int = 16) -> KVCacheManager:
    return KVCacheManager(
        make_kv_cache_config(block_size, 32),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
        log_stats=True,
    )


def test_session_apc_allow_reuses_prefix_and_emits_prompt_free_receipts() -> None:
    block_size = 16
    tokens = [value for value in range(3) for _ in range(block_size)] + [3] * 7
    manager = _manager(block_size=block_size)

    first = _controlled_request(
        "allow-first",
        tokens,
        block_size=block_size,
        namespace="session-a",
        allow=True,
    )
    assert _run_prefill(manager, first) == 0
    manager.free(first)

    second = _controlled_request(
        "allow-second",
        tokens,
        block_size=block_size,
        namespace="session-a",
        allow=True,
    )
    assert _run_prefill(manager, second) == 3 * block_size
    manager.free(second)

    audit = manager.take_qbi_prefix_cache_policy_receipts()
    assert audit["dropped_receipts"] == 0
    assert any(
        row["request_id"] == "allow-second"
        and row["operation"] == "lookup"
        and row["decision"] == "allow"
        and row["hit_tokens"] == 3 * block_size
        for row in audit["receipts"]
    )
    assert all("cache_salt" not in row for row in audit["receipts"])
    assert all(
        row["namespace_sha256"] != "session-a" for row in audit["receipts"]
    )


def test_session_apc_bypass_neither_reads_nor_writes_reusable_prefix() -> None:
    block_size = 16
    tokens = [value for value in range(3) for _ in range(block_size)] + [9] * 7
    manager = _manager(block_size=block_size)

    bypass = _controlled_request(
        "bypass",
        tokens,
        block_size=block_size,
        namespace="session-b",
        allow=False,
    )
    assert _run_prefill(manager, bypass) == 0
    manager.free(bypass)

    later_allow = _controlled_request(
        "later-allow",
        tokens,
        block_size=block_size,
        namespace="session-b",
        allow=True,
    )
    assert _run_prefill(manager, later_allow) == 0
    manager.free(later_allow)

    receipts = manager.take_qbi_prefix_cache_policy_receipts()["receipts"]
    bypass_rows = [row for row in receipts if row["request_id"] == "bypass"]
    assert {row["operation"] for row in bypass_rows} == {"lookup", "admit"}
    assert all(row["decision"] == "bypass" for row in bypass_rows)
    assert all(row["read_allowed"] is False for row in bypass_rows)
    assert all(row["write_allowed"] is False for row in bypass_rows)


def test_session_apc_cache_salt_isolates_namespaces() -> None:
    block_size = 16
    tokens = [value for value in range(3) for _ in range(block_size)] + [4] * 7
    manager = _manager(block_size=block_size)

    namespace_a = _controlled_request(
        "namespace-a-first",
        tokens,
        block_size=block_size,
        namespace="session-a",
        allow=True,
    )
    assert _run_prefill(manager, namespace_a) == 0
    manager.free(namespace_a)

    namespace_b = _controlled_request(
        "namespace-b",
        tokens,
        block_size=block_size,
        namespace="session-b",
        allow=True,
    )
    assert _run_prefill(manager, namespace_b) == 0
    manager.free(namespace_b)

    namespace_a_again = _controlled_request(
        "namespace-a-second",
        tokens,
        block_size=block_size,
        namespace="session-a",
        allow=True,
    )
    assert _run_prefill(manager, namespace_a_again) == 3 * block_size
    manager.free(namespace_a_again)

    lookup_rows = [
        row
        for row in manager.take_qbi_prefix_cache_policy_receipts()["receipts"]
        if row["operation"] == "lookup"
    ]
    namespace_hashes = {
        row["request_id"]: row["namespace_sha256"] for row in lookup_rows
    }
    assert namespace_hashes["namespace-a-first"] == namespace_hashes[
        "namespace-a-second"
    ]
    assert namespace_hashes["namespace-a-first"] != namespace_hashes[
        "namespace-b"
    ]


def test_session_apc_hybrid_mamba_allow_and_bypass_do_not_corrupt_manager() -> None:
    block_size = 16
    tokens = [value for value in range(3) for _ in range(block_size)] + [3] * 7
    manager = KVCacheManager(
        _make_hybrid_kv_cache_config(
            block_size,
            30,
            ["full", "mamba_align"],
        ),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
        log_stats=True,
    )

    allow = _controlled_request(
        "hybrid-allow",
        tokens,
        block_size=block_size,
        namespace="hybrid-session",
        allow=True,
    )
    assert _run_prefill(manager, allow) == 0
    manager.free(allow)

    reuse = _controlled_request(
        "hybrid-reuse",
        tokens,
        block_size=block_size,
        namespace="hybrid-session",
        allow=True,
    )
    _run_prefill(manager, reuse)
    manager.free(reuse)

    bypass = _controlled_request(
        "hybrid-bypass",
        tokens,
        block_size=block_size,
        namespace="hybrid-session",
        allow=False,
    )
    assert _run_prefill(manager, bypass) == 0
    manager.free(bypass)

    audit = manager.take_qbi_prefix_cache_policy_receipts()
    assert audit["dropped_receipts"] == 0
    assert any(
        row["request_id"] == "hybrid-reuse"
        and row["operation"] == "lookup"
        and row["decision"] == "allow"
        for row in audit["receipts"]
    )


@pytest.mark.parametrize(
    ("extra_args", "skip_reading", "message"),
    [
        (
            {"qbi_prefix_cache_policy_version": "session-apc-v2"},
            "false",
            "write admission is required",
        ),
        (
            {
                "qbi_prefix_cache_policy_version": "session-apc-v2",
                "kvplane_admit_prefix_cache": "allow",
            },
            "true",
            "allow both lookup/write or bypass both",
        ),
    ],
)
def test_session_apc_malformed_or_split_policy_fails_closed(
    extra_args: dict, skip_reading: str, message: str
) -> None:
    manager = _manager()
    request = make_request(
        "invalid-policy",
        [1] * 55,
        16,
        sha256,
        cache_salt="session-a",
    )
    assert request.sampling_params is not None
    request.sampling_params.extra_args = {
        **extra_args,
        "kvplane_bypass_prefix_cache": skip_reading,
    }
    request.skip_reading_prefix_cache = request.get_skip_reading_prefix_cache()

    with pytest.raises(ValueError, match=message):
        manager.get_computed_blocks(request)
