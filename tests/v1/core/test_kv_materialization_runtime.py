# SPDX-License-Identifier: Apache-2.0

import hashlib

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request


def _sampling_params(runtime_control: dict) -> SamplingParams:
    return SamplingParams.from_optional(
        max_tokens=1,
        extra_args={"kv_materialization_runtime_control": runtime_control},
    )


def test_request_carries_materialization_metadata() -> None:
    runtime_control = {
        "effective_decision": "partial_reuse",
        "target_reuse_tokens": 16,
    }
    request = Request("request", [1, 2], _sampling_params(runtime_control), None)

    assert request.kv_materialization_runtime_control is runtime_control


def test_partial_reuse_caps_lookup_and_commit_boundaries() -> None:
    manager = object.__new__(KVCacheManager)
    manager.hash_block_size = 4
    request = Request(
        "request",
        [1, 2],
        _sampling_params(
            {
                "effective_decision": "partial_reuse",
                "target_reuse_tokens": 9,
            }
        ),
        None,
    )
    request.block_hashes = [b"0", b"1", b"2"]

    assert manager._get_lookup_block_hashes(request) == [b"0", b"1"]
    assert manager._get_cacheable_num_tokens(request, 20) == 9


def test_segmented_tail_hash_isolated_from_prefix() -> None:
    def stable_hash(value: object) -> bytes:
        return hashlib.sha256(repr(value).encode()).digest()

    init_none_hash(stable_hash)
    block_hasher = get_request_block_hasher(2, stable_hash)
    params = _sampling_params(
        {
            "effective_decision": "partial_reuse",
            "target_reuse_tokens": 4,
            "segmented_tail_cache_salt": "kvmat:segment:test",
        }
    )
    request_a = Request(
        "a",
        [1, 2, 3, 4, 9, 10, 11, 12],
        params,
        None,
        block_hasher=block_hasher,
    )
    request_b = Request(
        "b",
        [5, 6, 7, 8, 9, 10, 11, 12],
        params,
        None,
        block_hasher=block_hasher,
    )

    assert request_a.block_hashes[:2] != request_b.block_hashes[:2]
    assert request_a.block_hashes[2:] == request_b.block_hashes[2:]
