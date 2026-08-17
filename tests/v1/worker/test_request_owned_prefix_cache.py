# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU correctness seals for owner-local hybrid prefix caching."""

import pytest
import torch

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlockCopy,
    make_block_hash_with_group_id,
)
from vllm.v1.core.request_owned_prefix import OwnerPrefixDescriptor
from vllm.v1.core.sched.ownership import (
    OwnerAdmissionStatus,
    OwnerAllocationDescriptor,
    OwnerCommand,
    OwnerCommandKind,
    OwnerLeaseKey,
    OwnerLeaseToken,
    OwnerReceipt,
    OwnerReceiptBatch,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.request_owned_kv import RequestOwnedKVStore

pytestmark = pytest.mark.cpu_test


def _manager() -> KVCacheManager:
    block_size = 4
    config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["full"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=2,
                    head_size=8,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["swa"],
                kv_cache_spec=SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=2,
                    head_size=8,
                    dtype=torch.float32,
                    sliding_window=8,
                ),
            ),
        ],
    )
    return KVCacheManager(
        kv_cache_config=config,
        max_model_len=64,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        enable_caching=True,
    )


def _reserve(request_id: str, seq: int) -> OwnerCommand:
    key = OwnerLeaseKey(request_id, 0)
    return OwnerCommand(
        key=key,
        owner_id=0,
        command_seq=seq,
        kind=OwnerCommandKind.RESERVE,
        required_num_tokens=12,
        allocation=OwnerAllocationDescriptor(
            key=key,
            num_prompt_tokens=8,
            num_computed_tokens=0,
            num_tokens=12,
            status=OwnerAdmissionStatus.WAITING,
            prefix=OwnerPrefixDescriptor((b"prefix-0", b"prefix-1")),
        ),
    )


def _token(key: OwnerLeaseKey, step_seq: int) -> OwnerLeaseToken:
    return OwnerLeaseToken(
        key=key,
        owner_id=0,
        step_seq=step_seq,
        command_seq=1,
        runnable_num_tokens=12,
    )


def test_reserve_never_publishes_uncomputed_blocks_and_commit_enables_hit():
    store = RequestOwnedKVStore(_manager(), owner_rank=0)

    first_command = _reserve("first", 1)
    first = store.reserve(first_command)
    assert first.accepted
    first_snapshot = store.snapshot(first_command.key)
    assert first_snapshot is not None
    assert first_snapshot.num_computed_tokens == 0

    # A second reservation before any successful forward must cold-miss. If
    # RESERVE had used KVCacheManager's default eager publication, these
    # unwritten blocks would already be reachable here.
    early_command = _reserve("early", 2)
    early = store.reserve(early_command)
    assert early.accepted
    early_snapshot = store.snapshot(early_command.key)
    assert early_snapshot is not None
    assert early_snapshot.num_computed_tokens == 0
    assert early.tables is not None and first.tables is not None
    assert early.tables[0][0] != first.tables[0][0]

    # Consume the allocation/control heartbeat, then execute and commit the
    # first prompt. Publication happens inside the all-entry terminal mark.
    empty = store.build_step_metadata(1, [], {})
    assert empty.accepted and empty.metadata is not None
    assert store.mark_computed_batch(empty.metadata).accepted
    step = store.build_step_metadata(
        2,
        [_token(first_command.key, 2)],
        {"first": 8},
    )
    assert step.accepted and step.metadata is not None
    assert store.mark_computed_batch(step.metadata).accepted

    hit_command = _reserve("hit", 3)
    hit = store.reserve(hit_command)
    assert hit.accepted
    hit_snapshot = store.snapshot(hit_command.key)
    assert hit_snapshot is not None
    # The manager retains one logits-producing block for recomputation, so an
    # eight-token prompt with four-token alignment has an exact four-token hit.
    assert hit_snapshot.num_computed_tokens == 4
    assert hit.tables is not None
    assert hit.tables[0][0] == first.tables[0][0]
    assert hit.tables[1][0] == first.tables[1][0]

    logical = OwnerReceiptBatch(
        owner_rank=0,
        emitted_step_seq=3,
        events=(
            OwnerReceipt(
                key=hit_command.key,
                owner_id=0,
                command_seq=3,
                accepted=True,
                runnable_num_tokens=12,
            ),
        ),
    )
    decorated = store.decorate_prefix_cache_receipts(logical)
    assert decorated.events[0].prefix_cache_hit_tokens == 4
    store.acknowledge_prefix_cache_receipts(decorated)
    assert (
        store.decorate_prefix_cache_receipts(logical).events[0].prefix_cache_hit_tokens
        is None
    )


def test_dsv4_shaped_four_group_hit_is_joint_and_hash_granularity_safe():
    groups = [
        KVCacheGroupSpec(
            layer_names=[f"mla-{ratio}"],
            kv_cache_spec=MLAAttentionSpec(
                block_size=8,
                num_kv_heads=1,
                head_size=8,
                dtype=torch.float32,
                compress_ratio=ratio,
                model_version="deepseek_v4",
            ),
        )
        for ratio in (1, 4, 8)
    ]
    groups.append(
        KVCacheGroupSpec(
            layer_names=["swa"],
            kv_cache_spec=SlidingWindowSpec(
                block_size=4,
                num_kv_heads=1,
                head_size=8,
                dtype=torch.float32,
                sliding_window=8,
            ),
        )
    )
    manager = KVCacheManager(
        kv_cache_config=KVCacheConfig(
            num_blocks=128,
            kv_cache_tensors=[],
            kv_cache_groups=groups,
        ),
        max_model_len=64,
        scheduler_block_size=8,
        hash_block_size=4,
        enable_caching=True,
    )
    store = RequestOwnedKVStore(manager, owner_rank=0)
    hashes = tuple(f"prefix-{index}".encode() for index in range(4))

    def reserve(request_id: str, seq: int):
        key = OwnerLeaseKey(request_id, 0)
        command = OwnerCommand(
            key=key,
            owner_id=0,
            command_seq=seq,
            kind=OwnerCommandKind.RESERVE,
            required_num_tokens=24,
            allocation=OwnerAllocationDescriptor(
                key=key,
                num_prompt_tokens=16,
                num_computed_tokens=0,
                num_tokens=24,
                status=OwnerAdmissionStatus.WAITING,
                prefix=OwnerPrefixDescriptor(hashes),
            ),
        )
        return key, store.reserve(command)

    first_key, first = reserve("dsv4-first", 1)
    assert first.accepted and first.tables is not None
    assert len(first.tables) == 4
    assert store.mark_computed(first_key, 16)

    hit_key, hit = reserve("dsv4-hit", 2)
    assert hit.accepted and hit.tables is not None
    snapshot = store.snapshot(hit_key)
    assert snapshot is not None and snapshot.num_computed_tokens == 8
    assert len(hit.tables) == 4
    assert all(
        hit_group[0] == first_group[0]
        for hit_group, first_group in zip(hit.tables, first.tables)
    )


def test_prefix_reset_refuses_live_lease_then_clears_idle_physical_index():
    store = RequestOwnedKVStore(_manager(), owner_rank=0)
    first_command = _reserve("reset-source", 1)
    first = store.reserve(first_command)
    assert first.accepted
    assert store.mark_computed(first_command.key, 8)
    assert not store.reset_prefix_cache()

    logical = OwnerReceiptBatch(
        owner_rank=0,
        emitted_step_seq=1,
        events=(
            OwnerReceipt(
                key=first_command.key,
                owner_id=0,
                command_seq=1,
                accepted=True,
                runnable_num_tokens=12,
            ),
        ),
    )
    store.acknowledge_prefix_cache_receipts(
        store.decorate_prefix_cache_receipts(logical)
    )
    release = OwnerCommand(
        key=first_command.key,
        owner_id=0,
        command_seq=2,
        kind=OwnerCommandKind.RELEASE,
        required_num_tokens=0,
    )
    assert store.release(release).accepted
    assert store.flush() == (first_command.key,)
    assert store.reset_prefix_cache()

    after = store.reserve(_reserve("after-reset", 3))
    assert after.accepted
    snapshot = store.snapshot(OwnerLeaseKey("after-reset", 0))
    assert snapshot is not None
    assert snapshot.num_computed_tokens == 0


def test_prefix_hit_is_capped_by_partial_reserve_horizon():
    store = RequestOwnedKVStore(_manager(), owner_rank=0)
    hashes = tuple(f"prefix-{index}".encode() for index in range(4))

    def reserve(request_id: str, seq: int, required: int) -> OwnerCommand:
        key = OwnerLeaseKey(request_id, 0)
        return OwnerCommand(
            key=key,
            owner_id=0,
            command_seq=seq,
            kind=OwnerCommandKind.RESERVE,
            required_num_tokens=required,
            allocation=OwnerAllocationDescriptor(
                key=key,
                num_prompt_tokens=16,
                num_computed_tokens=0,
                num_tokens=required,
                status=OwnerAdmissionStatus.WAITING,
                prefix=OwnerPrefixDescriptor(hashes),
            ),
        )

    source = reserve("long-source", 1, 20)
    assert store.reserve(source).accepted
    assert store.mark_computed(source.key, 16)

    partial = reserve("short-horizon", 2, 8)
    result = store.reserve(partial)
    assert result.accepted
    snapshot = store.snapshot(partial.key)
    assert snapshot is not None
    assert snapshot.num_computed_tokens == 4
    assert snapshot.num_computed_tokens < partial.required_num_tokens


def test_forward_prefix_cow_copy_consumes_only_its_allocation_delta(monkeypatch):
    manager = _manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    command = _reserve("forward-cow", 1)
    allocation = store.reserve(command)
    assert allocation.accepted and allocation.delta is not None
    destination_id = allocation.delta[0][0]
    destination = manager.block_pool.blocks[destination_id]
    source = manager.block_pool.get_new_blocks(1)[0]
    manager.block_pool._insert_block_hash(
        make_block_hash_with_group_id(b"forward-source", 0),
        source,
        num_tokens=4,
    )
    destination.ref_cnt += 1  # worker-copy retention beyond request ownership
    copies = [KVCacheBlockCopy(source.block_id, destination.block_id)]
    retained = [source, destination]
    monkeypatch.setattr(
        manager,
        "take_kv_cache_block_copies",
        lambda: (copies, retained),
    )

    prepared = store.prepare_prefix_cache_block_copies()
    assert prepared == tuple(copies)
    store.complete_prefix_cache_block_copies(prepared)

    record = store._records[command.key]
    assert record.pending_delta is not None
    assert all(destination_id not in group for group in record.pending_delta)
    assert source.ref_cnt == 0
    assert destination.ref_cnt == 1


def test_reverse_prefix_cow_snapshot_preserves_request_delta(monkeypatch):
    manager = _manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    command = _reserve("reverse-cow", 1)
    allocation = store.reserve(command)
    assert (
        allocation.accepted
        and allocation.delta is not None
        and allocation.tables is not None
    )
    original_delta = allocation.delta
    source_id = allocation.tables[0][0]
    source = manager.block_pool.blocks[source_id]
    destination = manager.block_pool.get_new_blocks(1)[0]
    snapshot_hash = make_block_hash_with_group_id(b"snapshot", 0)
    manager.block_pool._insert_block_hash(snapshot_hash, destination, num_tokens=4)
    source.ref_cnt += 1  # copy retention beyond request ownership
    copies = [KVCacheBlockCopy(source.block_id, destination.block_id)]
    retained = [source, destination]
    monkeypatch.setattr(
        manager,
        "take_kv_cache_block_copies",
        lambda: (copies, retained),
    )

    prepared = store.prepare_prefix_cache_block_copies()
    store.complete_prefix_cache_block_copies(prepared)

    record = store._records[command.key]
    assert record.pending_delta == original_delta
    assert source.ref_cnt == 1
    assert destination.ref_cnt == 0
    assert manager.block_pool.get_cached_block(b"snapshot", [0]) == [destination]
