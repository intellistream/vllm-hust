# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the G2 worker-local physical KV store.

The store under test (:class:`RequestOwnedKVStore`,
vllm.v1.worker.request_owned_kv) is exercised against a real
:class:`KVCacheManager` with a synthetic two-group config (full attention +
sliding window, both block_size 4, prefix caching disabled) so the manager
stays the authority on block counts and pool accounting.
"""

import pytest
import torch

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.ownership import (
    OwnerAdmissionStatus,
    OwnerAllocationDescriptor,
    OwnerCacheGroupSnapshot,
    OwnerCachePoolSnapshot,
    OwnerCommand,
    OwnerCommandKind,
    OwnerLeaseKey,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.request_owned_kv import RequestOwnedKVStore

pytestmark = pytest.mark.cpu_test


def _make_manager(num_blocks: int = 32, block_size: int = 4) -> KVCacheManager:
    spec0 = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=4,
        head_size=8,
        dtype=torch.float32,
    )
    spec1 = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=4,
        head_size=8,
        dtype=torch.float32,
        sliding_window=8,
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["a"], kv_cache_spec=spec0),
            KVCacheGroupSpec(layer_names=["b"], kv_cache_spec=spec1),
        ],
    )
    return KVCacheManager(
        kv_cache_config=config,
        max_model_len=64,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        enable_caching=False,
    )


def _command(
    key: OwnerLeaseKey,
    kind: OwnerCommandKind,
    required: int,
    computed: int = 0,
    prompt: int | None = None,
    seq: int = 0,
    owner_id: int = 0,
) -> OwnerCommand:
    allocation = None
    if kind is OwnerCommandKind.RESERVE:
        allocation = OwnerAllocationDescriptor(
            key=key,
            num_prompt_tokens=max(required, computed) if prompt is None else prompt,
            num_computed_tokens=computed,
            num_tokens=required,
            status=OwnerAdmissionStatus.WAITING,
        )
    return OwnerCommand(
        key=key,
        owner_id=owner_id,
        command_seq=seq,
        kind=kind,
        required_num_tokens=required,
        allocation=allocation,
    )


def _key(request_id: str, epoch: int = 0) -> OwnerLeaseKey:
    return OwnerLeaseKey(request_id=request_id, owner_epoch=epoch)


def _free(manager: KVCacheManager) -> int:
    return manager.block_pool.get_num_free_blocks()


def _sizes(tables: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return tuple(len(group) for group in tables)


def test_reserve_and_extend_tables():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("r1")
    initial = _free(manager)

    reserve = store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=10, seq=1)
    )
    assert reserve.accepted
    assert reserve.error is None
    # 10 tokens at block_size 4 -> 3 blocks per group, all newly allocated.
    assert _sizes(reserve.tables) == (3, 3)
    assert _sizes(reserve.delta) == (3, 3)
    assert reserve.tables == store.get_block_table(key)
    assert reserve.tables == tuple(
        tuple(group)
        for group in manager.get_block_ids(store._records[key].allocator_id)
    )
    assert _free(manager) == initial - 6

    extend = store.extend(
        _command(key, OwnerCommandKind.EXTEND, required=14, seq=2)
    )
    assert extend.accepted
    assert _sizes(extend.tables) == (4, 4)
    assert _sizes(extend.delta) == (1, 1)
    assert _free(manager) == initial - 8


def test_zero_token_reserve_and_noop_extend():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("z")
    initial = _free(manager)

    reserve = store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=0, seq=1)
    )
    assert reserve.accepted
    assert reserve.tables == ((), ())
    assert reserve.delta == ((), ())
    assert _free(manager) == initial
    # The reserved horizon is 0: no progress may be recorded on it.
    assert not store.mark_computed(key, 1)

    # The first real allocation arrives via EXTEND (horizon 0 -> 4).
    extend = store.extend(
        _command(key, OwnerCommandKind.EXTEND, required=4, seq=2)
    )
    assert extend.accepted
    assert _sizes(extend.delta) == (1, 1)
    assert _free(manager) == initial - 2

    # A no-op EXTEND (nothing new to allocate) never touches the manager.
    assert store.mark_computed(key, 4)
    noop = store.extend(
        _command(key, OwnerCommandKind.EXTEND, required=4, seq=3)
    )
    assert noop.accepted
    assert noop.delta == ((), ())
    assert _free(manager) == initial - 2


def test_two_stores_reuse_local_ids_with_distinct_allocators():
    manager = _make_manager()
    store0 = RequestOwnedKVStore(manager, owner_rank=0)
    store1 = RequestOwnedKVStore(manager, owner_rank=1)
    key = _key("shared")
    initial = _free(manager)

    r0 = store0.reserve(_command(key, OwnerCommandKind.RESERVE, required=10, seq=1))
    r1 = store1.reserve(_command(key, OwnerCommandKind.RESERVE, required=10, seq=1))
    assert r0.accepted and r1.accepted
    # Identical numeric local IDs are safe: allocator ids are epoch-qualified
    # and rank-prefixed, so the two tables live under distinct manager ids.
    assert store0._records[key].allocator_id != store1._records[key].allocator_id
    assert store0._records[key].allocator_id.startswith("0:")
    assert store1._records[key].allocator_id.startswith("1:")
    for group in range(2):
        assert set(r0.tables[group]).isdisjoint(r1.tables[group])
    assert _free(manager) == initial - 12
    # Each store reports only its own records; the pool free count is shared.
    snap0 = store0.pool_snapshot()
    snap1 = store1.pool_snapshot()
    assert [g.allocated_blocks for g in snap0.groups] == [3, 3]
    assert [g.allocated_blocks for g in snap1.groups] == [3, 3]
    assert snap0.free_blocks == snap1.free_blocks == initial - 12


def test_per_group_tables_are_private():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key_a = _key("a")
    key_b = _key("b")

    r_a = store.reserve(_command(key_a, OwnerCommandKind.RESERVE, required=10, seq=1))
    r_b = store.reserve(_command(key_b, OwnerCommandKind.RESERVE, required=10, seq=2))
    assert r_a.accepted and r_b.accepted
    for group in range(2):
        assert set(r_a.tables[group]).isdisjoint(r_b.tables[group])


def test_failed_reserve_and_extend_are_failure_atomic():
    manager = _make_manager(num_blocks=8)  # 7 free blocks (block 0 = null).
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("a")
    higher = _key("a", epoch=1)

    reserve = store.reserve(_command(key, OwnerCommandKind.RESERVE, required=10, seq=1))
    assert reserve.accepted
    assert _free(manager) == 1

    # A reserve that cannot fit leaves no record and no pool change.
    denied = store.reserve(
        _command(_key("b"), OwnerCommandKind.RESERVE, required=10, seq=2)
    )
    assert not denied.accepted
    assert "insufficient KV cache" in denied.error
    assert store.get_block_table(_key("b")) is None
    assert _free(manager) == 1

    # A failed higher-epoch reserve changes nothing: the lower-epoch lease
    # stays active (no local epoch fence advanced or superseded anything).
    denied_epoch = store.reserve(
        _command(higher, OwnerCommandKind.RESERVE, required=10, seq=3)
    )
    assert not denied_epoch.accepted
    assert store.get_block_table(key) == reserve.tables
    assert store.mark_computed(key, 5)
    assert _free(manager) == 1

    # Once the old lease is flushed, the higher-epoch retry succeeds.
    store.release(_command(key, OwnerCommandKind.RELEASE, required=10, seq=4))
    store.flush()
    retry = store.reserve(
        _command(higher, OwnerCommandKind.RESERVE, required=10, seq=5)
    )
    assert retry.accepted
    assert _sizes(retry.tables) == (3, 3)

    # A failed EXTEND leaves the record untouched (tables, horizon).
    store.release(_command(higher, OwnerCommandKind.RELEASE, required=10, seq=6))
    store.flush()
    store.reserve(_command(key, OwnerCommandKind.RESERVE, required=10, seq=7))
    denied_extend = store.extend(
        _command(key, OwnerCommandKind.EXTEND, required=30, seq=8)
    )
    assert not denied_extend.accepted
    assert "insufficient KV cache" in denied_extend.error
    assert _sizes(store.get_block_table(key)) == (3, 3)
    assert store.mark_computed(key, 9)  # reserved horizon still 10
    assert not store.mark_computed(key, 12)


def test_stale_epoch_cannot_target_newer_record():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    newer = _key("x", epoch=1)
    stale = _key("x", epoch=0)

    assert store.reserve(
        _command(newer, OwnerCommandKind.RESERVE, required=10, seq=1)
    ).accepted
    table = store.get_block_table(newer)
    after_reserve = _free(manager)

    # Physical records are keyed by the full lease key: a stale-epoch
    # command misses the lookup and changes nothing.
    stale_extend = store.extend(
        _command(stale, OwnerCommandKind.EXTEND, required=14, seq=2)
    )
    assert not stale_extend.accepted
    stale_release = store.release(
        _command(stale, OwnerCommandKind.RELEASE, required=10, seq=3)
    )
    assert not stale_release.accepted
    assert "no lease to release" in stale_release.error
    assert store.get_block_table(newer) == table
    assert not store.mark_computed(stale, 1)
    assert _free(manager) == after_reserve


def test_preempt_and_release_deferred_until_flush():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("p")
    initial = _free(manager)

    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=10, seq=1)
    ).accepted
    assert _free(manager) == initial - 6

    preempt = store.preempt(
        _command(key, OwnerCommandKind.PREEMPT, required=10, seq=2)
    )
    assert preempt.accepted and preempt.deferred
    # Blocks stay resident (and the allocator id non-reusable) pre-flush.
    assert _free(manager) == initial - 6
    assert _sizes(store.get_block_table(key)) == (3, 3)
    assert not store.mark_computed(key, 3)
    assert not store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=10, seq=3)
    ).accepted
    snapshot = store.pool_snapshot()
    assert [g.allocated_blocks for g in snapshot.groups] == [3, 3]
    assert [g.resident_blocks for g in snapshot.groups] == [3, 3]

    assert store.flush() == (key,)
    assert _free(manager) == initial
    assert store.get_block_table(key) is None
    assert [g.allocated_blocks for g in store.pool_snapshot().groups] == [0, 0]

    # The key is reusable after flush and allocates a fresh table.
    again = store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=10, seq=4)
    )
    assert again.accepted
    assert _sizes(again.tables) == (3, 3)
    assert _free(manager) == initial - 6

    # RELEASE follows the same deferred path.
    store.release(_command(key, OwnerCommandKind.RELEASE, required=10, seq=5))
    assert _free(manager) == initial - 6
    store.flush()
    assert _free(manager) == initial


def test_mark_computed_respects_reserved_horizon():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("m")

    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=10, seq=1)
    ).accepted
    assert not store.mark_computed(_key("missing"), 1)
    assert store.mark_computed(key, 8)
    assert not store.mark_computed(key, 12)  # beyond reserved horizon
    assert store.mark_computed(key, 10)
    assert not store.mark_computed(key, 9)  # regressive
    assert not store.mark_computed(key, True)  # bool is not a count
    assert not store.mark_computed(key, -1)

    # EXTEND raises the horizon; mark_computed follows the new cap.
    assert store.extend(
        _command(key, OwnerCommandKind.EXTEND, required=14, seq=2)
    ).accepted
    assert store.mark_computed(key, 12)
    assert store.mark_computed(key, 14)
    assert not store.mark_computed(key, 15)


def test_pool_snapshot_is_canonical_and_block_id_free():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=7)
    key = _key("s")

    snapshot = store.pool_snapshot()
    assert isinstance(snapshot, OwnerCachePoolSnapshot)
    assert isinstance(snapshot.groups, tuple)
    assert [isinstance(g, OwnerCacheGroupSnapshot) for g in snapshot.groups]
    assert [g.group_index for g in snapshot.groups] == [0, 1]
    assert [g.spec_kind for g in snapshot.groups] == ["full_attention",
                                                      "sliding_window"]
    assert [g.effective_tokens_per_block for g in snapshot.groups] == [4, 4]
    assert [g.allocated_blocks for g in snapshot.groups] == [0, 0]
    assert snapshot.owner_rank == 7
    assert snapshot.total_blocks == 32
    assert snapshot.free_blocks == 31  # block 0 is the pool null block
    assert snapshot.bytes_per_block is None  # tensor-less config: unknown

    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=10, seq=1)
    ).accepted
    snapshot = store.pool_snapshot()
    assert [g.allocated_blocks for g in snapshot.groups] == [3, 3]
    assert [g.resident_blocks for g in snapshot.groups] == [3, 3]
    assert snapshot.free_blocks == 25

    # The snapshot is block-ID free: every leaf field is a scalar.
    stack = [snapshot]
    while stack:
        value = stack.pop()
        if isinstance(value, (OwnerCachePoolSnapshot, OwnerCacheGroupSnapshot)):
            stack.extend(vars(value).values())
        elif isinstance(value, tuple):
            stack.extend(value)
        else:
            assert value is None or isinstance(value, (int, str))


def test_restore_and_prefix_paths_rejected():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("r")

    restore = store.restore(
        _command(key, OwnerCommandKind.RESTORE, required=10, seq=1)
    )
    assert not restore.accepted
    assert "out of scope" in restore.error
    assert store.get_block_table(key) is None

    # Prefix caching / computed-block APIs are out of scope: absent.
    assert not hasattr(store, "get_prefix_cache_snapshot")
    assert not hasattr(store, "get_computed_blocks")


def _make_dsv4_manager(
    num_blocks: int = 32, tensors: list[KVCacheTensor] | None = None
) -> KVCacheManager:
    spec0 = MLAAttentionSpec(
        block_size=4,
        num_kv_heads=4,
        head_size=8,
        dtype=torch.float32,
        compress_ratio=4,
    )
    spec1 = SlidingWindowSpec(
        block_size=4,
        num_kv_heads=4,
        head_size=8,
        dtype=torch.float32,
        sliding_window=8,
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=(
            [KVCacheTensor(size=num_blocks * 1024, shared_by=["a"]),
             KVCacheTensor(size=num_blocks * 2048, shared_by=["b"])]
            if tensors is None
            else tensors
        ),
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["a"], kv_cache_spec=spec0),
            KVCacheGroupSpec(layer_names=["b"], kv_cache_spec=spec1),
        ],
    )
    return KVCacheManager(
        kv_cache_config=config,
        max_model_len=64,
        scheduler_block_size=4,
        hash_block_size=4,
        enable_caching=False,
    )


def test_dsv4_shaped_heterogeneous_groups():
    """Ascend DSV4-style config: MLA compression and a tensor-derived
    pool-wide bytes_per_block instead of any one group's page size."""
    manager = _make_dsv4_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    snapshot = store.pool_snapshot()
    assert [g.spec_kind for g in snapshot.groups] == ["mla_attention",
                                                      "sliding_window"]
    # Effective tokens per block honors MLA compression (block_size * 4x).
    assert [g.effective_tokens_per_block for g in snapshot.groups] == [16, 4]
    # One globally unique shared-pool block ID spans every tensor: the
    # per-tensor bytes divide evenly across all pool blocks.
    assert snapshot.bytes_per_block == 1024 + 2048
    assert snapshot.total_blocks == 32

    key = _key("d")
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=10, seq=1)
    ).accepted
    snapshot = store.pool_snapshot()
    assert [g.allocated_blocks for g in snapshot.groups] == [3, 3]
    assert [g.resident_blocks for g in snapshot.groups] == [3, 3]
    assert snapshot.free_blocks == 25

    # A tensor that does not divide evenly across the pool -> unknown.
    odd = _make_dsv4_manager(tensors=[KVCacheTensor(size=1000, shared_by=["a"])])
    assert RequestOwnedKVStore(odd, 0).pool_snapshot().bytes_per_block is None


def test_wrong_kind_and_duplicate_guards():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("g")

    wrong_kind = store.reserve(
        _command(key, OwnerCommandKind.EXTEND, required=10, seq=1)
    )
    assert not wrong_kind.accepted and "RESERVE" in wrong_kind.error
    no_allocation = OwnerCommand(
        key=key,
        owner_id=0,
        command_seq=2,
        kind=OwnerCommandKind.RESERVE,
        required_num_tokens=10,
        allocation=None,
    )
    assert not store.reserve(no_allocation).accepted

    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=10, seq=3)
    ).accepted
    duplicate = store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=10, seq=4)
    )
    assert not duplicate.accepted and "duplicate" in duplicate.error
    wrong_free = store.preempt(
        _command(key, OwnerCommandKind.RELEASE, required=10, seq=5)
    )
    assert not wrong_free.accepted and "PREEMPT" in wrong_free.error
