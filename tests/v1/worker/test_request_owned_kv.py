# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the G2 worker-local physical KV store.

The store under test (:class:`RequestOwnedKVStore`,
vllm.v1.worker.request_owned_kv) is exercised against a real
:class:`KVCacheManager` with a synthetic two-group config (full attention +
sliding window, both block_size 4, prefix caching disabled) so the manager
stays the authority on block counts and pool accounting.
"""

from dataclasses import FrozenInstanceError, replace

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
    OwnerLeaseToken,
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
    status: OwnerAdmissionStatus = OwnerAdmissionStatus.WAITING,
) -> OwnerCommand:
    allocation = None
    if kind is OwnerCommandKind.RESERVE:
        allocation = OwnerAllocationDescriptor(
            key=key,
            num_prompt_tokens=max(required, computed) if prompt is None else prompt,
            num_computed_tokens=computed,
            num_tokens=required,
            status=status,
        )
    return OwnerCommand(
        key=key,
        owner_id=owner_id,
        command_seq=seq,
        kind=kind,
        required_num_tokens=required,
        allocation=allocation,
    )


def _token(
    key: OwnerLeaseKey,
    runnable: int,
    step_seq: int = 1,
    owner_id: int = 0,
    command_seq: int = 1,
) -> OwnerLeaseToken:
    return OwnerLeaseToken(
        key=key,
        owner_id=owner_id,
        step_seq=step_seq,
        command_seq=command_seq,
        runnable_num_tokens=runnable,
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

    reserve = store.reserve(_command(key, OwnerCommandKind.RESERVE, required=10, seq=1))
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

    extend = store.extend(_command(key, OwnerCommandKind.EXTEND, required=14, seq=2))
    assert extend.accepted
    assert _sizes(extend.tables) == (4, 4)
    assert _sizes(extend.delta) == (1, 1)
    assert _free(manager) == initial - 8


def test_zero_token_reserve_and_noop_extend():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("z")
    initial = _free(manager)

    reserve = store.reserve(_command(key, OwnerCommandKind.RESERVE, required=0, seq=1))
    assert reserve.accepted
    assert reserve.tables == ((), ())
    assert reserve.delta == ((), ())
    assert _free(manager) == initial
    # The reserved horizon is 0: no progress may be recorded on it.
    assert not store.mark_computed(key, 1)

    # The first real allocation arrives via EXTEND (horizon 0 -> 4).
    extend = store.extend(_command(key, OwnerCommandKind.EXTEND, required=4, seq=2))
    assert extend.accepted
    assert _sizes(extend.delta) == (1, 1)
    assert _free(manager) == initial - 2

    # A no-op EXTEND (nothing new to allocate) never touches the manager.
    assert store.mark_computed(key, 4)
    noop = store.extend(_command(key, OwnerCommandKind.EXTEND, required=4, seq=3))
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

    preempt = store.preempt(_command(key, OwnerCommandKind.PREEMPT, required=10, seq=2))
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
    again = store.reserve(_command(key, OwnerCommandKind.RESERVE, required=10, seq=4))
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


def test_preempt_flush_release_tombstone():
    """A request aborted while preempted: PREEMPT + flush frees the blocks
    and leaves an exact-key tombstone, so the later valid RELEASE is a
    non-deferred no-op instead of a rejection."""
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("pfr")
    initial = _free(manager)

    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=10, seq=1)
    ).accepted
    assert store.preempt(
        _command(key, OwnerCommandKind.PREEMPT, required=10, seq=2)
    ).accepted
    assert store.flush() == (key,)
    assert _free(manager) == initial
    assert store.get_block_table(key) is None
    # The request is still alive logically; other commands still miss.
    assert not store.mark_computed(key, 1)
    assert not store.extend(
        _command(key, OwnerCommandKind.EXTEND, required=14, seq=3)
    ).accepted

    # The aborting RELEASE is accepted as a non-deferred no-op and ends
    # the tombstone: a repeated RELEASE is now unknown again.
    release = store.release(_command(key, OwnerCommandKind.RELEASE, required=10, seq=4))
    assert release.accepted
    assert not release.deferred
    assert _free(manager) == initial
    again = store.release(_command(key, OwnerCommandKind.RELEASE, required=10, seq=5))
    assert not again.accepted
    assert "no lease to release" in again.error


def test_preempt_flush_resume():
    """A request resumed while preempted: RESERVE on the tombstoned key
    consumes the tombstone and allocates a fresh table."""
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("pfrs")
    initial = _free(manager)

    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=10, seq=1)
    ).accepted
    assert store.preempt(
        _command(key, OwnerCommandKind.PREEMPT, required=10, seq=2)
    ).accepted
    assert store.flush() == (key,)
    assert _free(manager) == initial

    resume = store.reserve(_command(key, OwnerCommandKind.RESERVE, required=10, seq=3))
    assert resume.accepted
    assert _sizes(resume.tables) == (3, 3)
    assert _free(manager) == initial - 6
    # Tombstone consumed: the resumed lease frees via the normal RELEASE
    # deferral path, and the release-flush leaves no tombstone behind.
    release = store.release(_command(key, OwnerCommandKind.RELEASE, required=10, seq=4))
    assert release.accepted and release.deferred
    store.flush()
    assert _free(manager) == initial
    assert not store.release(
        _command(key, OwnerCommandKind.RELEASE, required=10, seq=5)
    ).accepted


def test_pool_snapshot_is_canonical_and_block_id_free():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=7)
    key = _key("s")

    snapshot = store.pool_snapshot()
    assert isinstance(snapshot, OwnerCachePoolSnapshot)
    assert isinstance(snapshot.groups, tuple)
    assert [isinstance(g, OwnerCacheGroupSnapshot) for g in snapshot.groups]
    assert [g.group_index for g in snapshot.groups] == [0, 1]
    assert [g.spec_kind for g in snapshot.groups] == [
        "full_attention",
        "sliding_window",
    ]
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


def test_owner_kv_snapshot_binds_generation_and_survives_pending_free():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=7)
    key = _key("snapshot")
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=10, seq=1)
    ).accepted
    assert store.mark_computed(key, 6)

    snapshot = store.snapshot(key)
    assert snapshot is not None
    assert snapshot.key == key
    assert snapshot.owner_rank == 7
    assert snapshot.allocation_generation == 1
    assert snapshot.num_computed_tokens == 6
    assert snapshot.reserved_num_tokens == 10
    assert not snapshot.pending_free
    assert snapshot.tables == store.get_block_table(key)
    computed = store.computed_prefix_snapshot(key)
    assert computed is not None
    assert computed.tables == tuple(group[:2] for group in snapshot.tables)
    assert store.group_block_sizes == (4, 4)

    assert store.preempt(
        _command(key, OwnerCommandKind.PREEMPT, required=10, seq=2)
    ).accepted
    pending = store.snapshot(key)
    assert pending is not None
    assert pending.pending_free
    assert pending.tables == snapshot.tables
    pending_computed = store.computed_prefix_snapshot(key)
    assert pending_computed is not None
    assert pending_computed.pending_free
    assert pending_computed.tables == computed.tables

    assert store.flush() == (key,)
    assert store.snapshot(key) is None


def test_restore_requires_cold_lease_and_reactivates_exact_destination():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("r")

    restore = store.restore(_command(key, OwnerCommandKind.RESTORE, required=10, seq=1))
    assert not restore.accepted
    assert "no durable cold lease" in restore.error
    assert store.get_block_table(key) is None

    assert store.reserve(
        _command(
            key,
            OwnerCommandKind.RESERVE,
            required=10,
            computed=6,
            seq=2,
        )
    ).accepted
    assert store.preempt(
        _command(key, OwnerCommandKind.PREEMPT, required=10, seq=3)
    ).accepted
    store.flush()

    restored = store.restore(
        _command(key, OwnerCommandKind.RESTORE, required=10, seq=4)
    )
    assert restored.accepted
    snapshot = store.snapshot(key)
    assert snapshot is not None
    assert snapshot.num_computed_tokens == 6
    assert store.restore_source_block_indices(key, snapshot.allocation_generation) == (
        (0, 1),
        (0, 1),
    )
    assert not store.is_restore_ready(key)
    reserve = _command(
        key,
        OwnerCommandKind.RESERVE,
        required=10,
        computed=6,
        seq=5,
        status=OwnerAdmissionStatus.PREEMPTED,
    )
    assert not store.reserve(reserve).accepted
    assert store.mark_restore_ready(key, snapshot.allocation_generation)
    assert store.is_restore_ready(key)
    assert store.reserve(reserve).accepted
    assert store.mark_reactivated(key, snapshot.allocation_generation)
    assert not store.is_restore_ready(key)
    assert (
        store.restore_source_block_indices(key, snapshot.allocation_generation) is None
    )

    # Prefix caching / computed-block APIs are out of scope: absent.
    assert not hasattr(store, "get_prefix_cache_snapshot")
    assert not hasattr(store, "get_computed_blocks")


def test_computed_prefix_block_indices_preserve_hybrid_null_positions():
    assert RequestOwnedKVStore._computed_prefix_block_indices(
        ((0, 0, 0, 4, 5), (6, 0, 8)),
        num_computed_tokens=33,
        group_block_sizes=(8, 16),
    ) == ((3, 4), (0, 2))


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
            [
                KVCacheTensor(size=num_blocks * 1024, shared_by=["a"]),
                KVCacheTensor(size=num_blocks * 2048, shared_by=["b"]),
            ]
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


def test_compressed_prefix_uses_effective_physical_block_extent():
    store = RequestOwnedKVStore(_make_dsv4_manager(), owner_rank=0)
    key = _key("compressed")
    assert store.reserve(
        _command(
            key,
            OwnerCommandKind.RESERVE,
            required=20,
            computed=0,
            seq=1,
        )
    ).accepted
    assert store.mark_computed(key, 6)

    snapshot = store.snapshot(key)
    computed = store.computed_prefix_snapshot(key)
    assert snapshot is not None and computed is not None
    assert store.group_block_sizes == (16, 4)
    assert computed.tables[0] == snapshot.tables[0][:1]
    assert computed.tables[1] == snapshot.tables[1][:2]


def test_dsv4_shaped_heterogeneous_groups():
    """Ascend DSV4-style config: MLA compression and a tensor-derived
    pool-wide bytes_per_block instead of any one group's page size."""
    manager = _make_dsv4_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    snapshot = store.pool_snapshot()
    assert [g.spec_kind for g in snapshot.groups] == ["mla_attention", "sliding_window"]
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


# -- G3 execution metadata ---------------------------------------------------


def test_build_zero_local_owner_batch_valid_and_one_step_fenced():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("z")
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted

    empty = store.build_step_metadata(1, [], {})
    assert empty.accepted
    assert empty.error is None
    assert empty.metadata is not None
    assert empty.metadata.step_seq == 1
    assert empty.metadata.owner_rank == 0
    assert empty.metadata.entries == ()

    # One-step fence: the same or an older step cannot be rebuilt.
    assert not store.build_step_metadata(1, [], {}).accepted
    assert not store.build_step_metadata(0, [], {}).accepted
    assert not store.build_step_metadata(True, [], {}).accepted
    assert not store.build_step_metadata(-1, [], {}).accepted
    assert not store.build_step_metadata("1", [], {}).accepted
    assert not store.build_step_metadata(None, [], {}).accepted


def test_build_rejects_same_step_allocation_plus_authorization():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("s")
    # RESERVE for this step, then an execution token for the same key in the
    # same step: the scheduler must allocate and authorize in different steps.
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    same = store.build_step_metadata(1, [_token(key, 8, step_seq=1)], {"s": 8})
    assert not same.accepted
    assert "same-step" in same.error

    # The allocation step itself is valid; the token belongs to the next step.
    assert store.build_step_metadata(1, [], {}).accepted
    next_step = store.build_step_metadata(2, [_token(key, 8, step_seq=2)], {"s": 8})
    assert next_step.accepted
    assert next_step.metadata.entries[0].key == key


def test_build_rejects_wrong_missing_extra_duplicate_owner_step_state():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("a")
    tail_key = _key("c")
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.reserve(
        _command(tail_key, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.build_step_metadata(1, [], {}).accepted

    wrong_owner = store.build_step_metadata(
        2, [_token(key, 8, step_seq=2, owner_id=1)], {"a": 8}
    )
    assert not wrong_owner.accepted
    assert "wrong-owner" in wrong_owner.error

    # A token for a request with no local active record is extra: the count
    # is foreign on this rank and ignored, so the token has no match.
    extra_token = store.build_step_metadata(
        2, [_token(_key("ghost"), 8, step_seq=2)], {"ghost": 8}
    )
    assert not extra_token.accepted
    assert "extra lease token" in extra_token.error

    out_of_horizon = store.build_step_metadata(
        2, [_token(key, 12, step_seq=2)], {"a": 12}
    )
    assert not out_of_horizon.accepted
    assert "out-of-horizon" in out_of_horizon.error

    beyond_lease = store.build_step_metadata(2, [_token(key, 8, step_seq=2)], {"a": 12})
    assert not beyond_lease.accepted
    assert "exceeds the lease horizon" in beyond_lease.error

    for count in (-1, True, "8"):
        bad_count = store.build_step_metadata(
            2, [_token(key, 8, step_seq=2)], {"a": count}
        )
        assert not bad_count.accepted
        assert "scheduled count" in bad_count.error
    # A zero global count is not positive: the step is a heartbeat.
    zero = store.build_step_metadata(2, [_token(key, 8, step_seq=2)], {"a": 0})
    assert zero.accepted
    assert zero.metadata.entries == ()

    # A foreign positive count (no local record) alongside a valid local
    # execution is ignored on this rank.
    foreign_ok = store.build_step_metadata(
        3, [_token(key, 8, step_seq=3)], {"a": 8, "b": 8}
    )
    assert foreign_ok.accepted
    assert [entry.key for entry in foreign_ok.metadata.entries] == [key]
    # Consume the handoff mark with the exact post-step target before any
    # later token-bearing build for the same key.
    assert store.mark_computed(key, 8)

    wrong_step = store.build_step_metadata(4, [_token(key, 8, step_seq=99)], {"a": 8})
    assert not wrong_step.accepted
    assert "does not match" in wrong_step.error

    duplicate = store.build_step_metadata(
        4,
        [_token(key, 8, step_seq=4), _token(key, 8, step_seq=4)],
        {"a": 8},
    )
    assert not duplicate.accepted
    assert "duplicate" in duplicate.error

    # The original key has reached its reserved horizon (8), so the valid
    # token-bearing build uses a separate reserved key.
    ok = store.build_step_metadata(4, [_token(tail_key, 8, step_seq=4)], {"c": 8})
    assert ok.accepted
    assert [entry.key for entry in ok.metadata.entries] == [tail_key]

    stale = store.build_step_metadata(4, [_token(tail_key, 8, step_seq=4)], {"c": 8})
    assert not stale.accepted
    assert "stale" in stale.error
    assert not store.build_step_metadata(1, [], {}).accepted


def test_build_rejects_local_positive_count_without_token():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("m")
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.build_step_metadata(1, [], {}).accepted

    # A local active request with a positive scheduled count but no
    # own-rank authorization token must fail closed instead of being
    # invisible to a token-iterating derivation.
    missing = store.build_step_metadata(2, [], {"m": 8})
    assert not missing.accepted
    assert "missing authorization token" in missing.error
    assert store._records[key].pending_delta is not None

    # The same step is retryable with the token present and still hands the
    # full pending delta.
    retry = store.build_step_metadata(2, [_token(key, 8, step_seq=2)], {"m": 8})
    assert retry.accepted
    assert retry.metadata.entries[0].delta is not None
    assert retry.metadata.entries[0].post_step_num_tokens == 8


def test_build_rejects_ambiguous_same_request_active_records():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    epoch0 = _key("a", epoch=0)
    epoch1 = _key("a", epoch=1)
    assert store.reserve(
        _command(epoch0, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.build_step_metadata(1, [], {}).accepted
    # A second active record for the same request id makes the count
    # ambiguous: the store cannot pick the exact lease key.
    assert store.reserve(
        _command(epoch1, OwnerCommandKind.RESERVE, required=8, seq=2)
    ).accepted

    ambiguous = store.build_step_metadata(2, [], {"a": 8})
    assert not ambiguous.accepted
    assert "ambiguous same-request" in ambiguous.error


def test_build_rejects_pending_free_lease():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("p")
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.build_step_metadata(1, [], {}).accepted
    assert store.preempt(
        _command(key, OwnerCommandKind.PREEMPT, required=8, seq=2)
    ).accepted

    pending = store.build_step_metadata(2, [_token(key, 8, step_seq=2)], {"p": 8})
    assert not pending.accepted
    assert "pending free" in pending.error


def test_build_exact_epoch():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    epoch0 = _key("e", epoch=0)
    assert store.reserve(
        _command(epoch0, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.build_step_metadata(1, [], {}).accepted

    # The derived local key keeps the exact epoch: the count maps to the
    # (e, epoch 0) record, so a token for epoch 1 leaves the local key
    # missing its authorization token (and the epoch-1 token extra).
    wrong_epoch = store.build_step_metadata(
        2, [_token(_key("e", epoch=1), 8, step_seq=2)], {"e": 8}
    )
    assert not wrong_epoch.accepted
    assert "missing authorization token" in wrong_epoch.error
    assert "owner_epoch=0" in wrong_epoch.error

    exact = store.build_step_metadata(2, [_token(epoch0, 8, step_seq=2)], {"e": 8})
    assert exact.accepted
    assert [entry.key for entry in exact.metadata.entries] == [epoch0]


def test_step_metadata_detached_immutability_and_pre_flush_readability():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("d")
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=10, seq=1)
    ).accepted
    assert store.build_step_metadata(1, [], {}).accepted
    first = store.build_step_metadata(2, [_token(key, 10, step_seq=2)], {"d": 10})
    assert first.accepted
    snapshot = first.metadata
    entry = snapshot.entries[0]

    # The snapshot is readable pre-flush and matches the live tables.
    assert store.get_block_table(key) == entry.tables

    # Frozen: neither the batch nor an entry can be mutated.
    with pytest.raises(FrozenInstanceError):
        entry.tables = ()
    with pytest.raises(FrozenInstanceError):
        snapshot.entries = ()
    with pytest.raises(FrozenInstanceError):
        entry.key = _key("other")

    # Fully detached: a later EXTEND changes the live tables but never the
    # already-built snapshot; the first handoff must be completed first.
    assert store.mark_computed(key, 10)
    assert store.extend(
        _command(key, OwnerCommandKind.EXTEND, required=14, seq=3)
    ).accepted
    assert store.build_step_metadata(3, [], {}).accepted
    later = store.build_step_metadata(4, [_token(key, 14, step_seq=4)], {"d": 4})
    assert later.accepted
    later_entry = later.metadata.entries[0]
    assert later_entry.tables != entry.tables
    assert _sizes(later_entry.tables) > _sizes(entry.tables)
    assert later_entry.tables == store.get_block_table(key)

    # Flush removes the record, but the detached snapshots keep their facts.
    assert store.release(
        _command(key, OwnerCommandKind.RELEASE, required=14, seq=5)
    ).accepted
    store.flush()
    assert store.get_block_table(key) is None
    assert later_entry.tables == later.metadata.entries[0].tables


def test_step_metadata_heterogeneous_group_order():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("h")
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.build_step_metadata(1, [], {}).accepted
    built = store.build_step_metadata(2, [_token(key, 8, step_seq=2)], {"h": 8})
    assert built.accepted
    entry = built.metadata.entries[0]

    assert len(store._group_specs) == 2
    assert len(entry.tables) == 2
    assert len(entry.delta) == 2
    # Group order is the store's heterogeneous group order and both tables
    # and deltas follow it; the two groups hold distinct block id spaces.
    assert _sizes(entry.tables) == (2, 2)
    assert _sizes(entry.delta) == (2, 2)
    assert entry.tables == store.get_block_table(key)
    assert set(entry.tables[0]).isdisjoint(entry.tables[1])


def test_extend_full_table_replacement_and_delta():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("x")
    reserve_result = store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=10, seq=1)
    )
    assert reserve_result.accepted
    # Heartbeat steps: allocation without execution retains the deltas.
    assert store.build_step_metadata(1, [], {}).accepted
    extend_result = store.extend(
        _command(key, OwnerCommandKind.EXTEND, required=14, seq=2)
    )
    assert extend_result.accepted
    assert store.build_step_metadata(2, [], {}).accepted

    built = store.build_step_metadata(3, [_token(key, 14, step_seq=3)], {"x": 4})
    assert built.accepted
    entry = built.metadata.entries[0]

    # Full replacement: the snapshot table is the complete current table.
    assert entry.tables == store.get_block_table(key)
    # The delta accumulates every block allocated since the last handoff
    # (reserve blocks plus extend blocks), needed for local zeroing.
    expected_delta = tuple(
        reserve_result.delta[g] + extend_result.delta[g]
        for g in range(len(reserve_result.delta))
    )
    assert entry.delta == expected_delta
    assert entry.pre_step_num_computed_tokens == 0
    # Post-step target is pre-step computed + scheduled count (a partial
    # chunk below the 14-token lease horizon), not the horizon itself.
    assert entry.post_step_num_tokens == 4
    assert entry.allocation_generation >= 1

    # The delta was handed into the step; the next handoff sees nothing
    # pending and builds on the marked computed progress.
    assert store.mark_computed(key, 4)
    assert store.build_step_metadata(4, [], {}).accepted
    second = store.build_step_metadata(5, [_token(key, 14, step_seq=5)], {"x": 4})
    assert second.accepted
    assert second.metadata.entries[0].delta == ((), ())
    assert second.metadata.entries[0].pre_step_num_computed_tokens == 4
    assert second.metadata.entries[0].post_step_num_tokens == 8


def test_partial_chunk_below_horizon_and_exact_mark_target():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("c")
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=20, seq=1)
    ).accepted
    assert store.build_step_metadata(1, [], {}).accepted

    built = store.build_step_metadata(2, [_token(key, 20, step_seq=2)], {"c": 6})
    assert built.accepted
    entry = built.metadata.entries[0]
    # The lease horizon is 20, but this step only runs 6 tokens: the target
    # sits strictly below the horizon and mark must accept exactly it.
    assert entry.post_step_num_tokens == 6
    assert entry.post_step_num_tokens < 20

    assert not store.mark_computed(key, 5)
    assert not store.mark_computed(key, 20)
    assert store.mark_computed(key, 6)
    assert store._records[key].num_computed_tokens == 6

    # A later partial chunk continues from the marked progress.
    assert store.build_step_metadata(3, [], {}).accepted
    later = store.build_step_metadata(4, [_token(key, 20, step_seq=4)], {"c": 8})
    assert later.accepted
    assert later.metadata.entries[0].pre_step_num_computed_tokens == 6
    assert later.metadata.entries[0].post_step_num_tokens == 14
    assert not store.mark_computed(key, 13)
    assert store.mark_computed(key, 14)


def test_zero_token_grant_heartbeat_retains_deltas():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("q")
    reserve_result = store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=8, seq=1)
    )
    assert reserve_result.accepted
    assert store.build_step_metadata(1, [], {}).accepted

    # A zero-global-token heartbeat may carry a newly published grant but
    # must build empty execution metadata and keep the allocation delta
    # pending.
    heartbeat = store.build_step_metadata(2, [_token(key, 8, step_seq=2)], {})
    assert heartbeat.accepted
    assert heartbeat.metadata.entries == ()
    assert store._records[key].pending_delta is not None

    # The later token-bearing step still receives the full pending delta.
    built = store.build_step_metadata(3, [_token(key, 8, step_seq=3)], {"q": 8})
    assert built.accepted
    assert built.metadata.entries[0].delta == reserve_result.delta
    assert built.metadata.entries[0].post_step_num_tokens == 8


def test_same_key_preempt_flush_reserve_generation_change():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("g")
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.build_step_metadata(1, [], {}).accepted
    first = store.build_step_metadata(2, [_token(key, 8, step_seq=2)], {"g": 8})
    assert first.accepted
    first_generation = first.metadata.entries[0].allocation_generation

    # Same-key ABA: preempt, flush, and re-reserve the exact same key before
    # the handed-in step is ever marked complete.
    assert store.preempt(
        _command(key, OwnerCommandKind.PREEMPT, required=8, seq=3)
    ).accepted
    store.flush()
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=12, seq=4)
    ).accepted
    # The stale handed-in snapshot cannot mark the recycled record: the
    # physical allocation generation changed.
    assert not store.mark_computed(key, first.metadata.entries[0].post_step_num_tokens)

    assert store.build_step_metadata(3, [], {}).accepted
    second = store.build_step_metadata(4, [_token(key, 12, step_seq=4)], {"g": 12})
    assert second.accepted
    second_generation = second.metadata.entries[0].allocation_generation
    assert second_generation != first_generation

    # The recycled record's own post-step target is accepted exactly.
    assert not store.mark_computed(key, first.metadata.entries[0].post_step_num_tokens)
    assert not store.mark_computed(key, 8)
    assert store.mark_computed(key, 12)


def test_build_rejects_unconsumed_pending_mark():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("u")
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=16, seq=1)
    ).accepted
    assert store.build_step_metadata(1, [], {}).accepted
    assert store.build_step_metadata(
        2, [_token(key, 16, step_seq=2)], {"u": 8}
    ).accepted

    # The handed-in execution was never marked complete: a new build for the
    # same key is rejected until the exact target is marked.
    rejected = store.build_step_metadata(3, [_token(key, 16, step_seq=3)], {"u": 8})
    assert not rejected.accepted
    assert "unconsumed pending mark" in rejected.error

    assert store.mark_computed(key, 8)
    assert store.build_step_metadata(
        3, [_token(key, 16, step_seq=3)], {"u": 8}
    ).accepted


def test_allocation_deltas_not_lost_on_failed_build():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("n")
    reserve_result = store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=8, seq=1)
    )
    assert reserve_result.accepted
    assert store.build_step_metadata(1, [], {}).accepted

    # A failing build (duplicate token) must not consume or lose the delta.
    failed = store.build_step_metadata(
        2,
        [_token(key, 8, step_seq=2), _token(key, 8, step_seq=2)],
        {"n": 8},
    )
    assert not failed.accepted
    assert "duplicate" in failed.error

    # The same step is retryable and still hands the full pending delta.
    retry = store.build_step_metadata(2, [_token(key, 8, step_seq=2)], {"n": 8})
    assert retry.accepted
    assert retry.metadata.entries[0].delta == reserve_result.delta


def test_mark_computed_strict_after_step_handoff():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("m3")
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=10, seq=1)
    ).accepted
    assert store.build_step_metadata(1, [], {}).accepted
    built = store.build_step_metadata(2, [_token(key, 10, step_seq=2)], {"m3": 10})
    assert built.accepted
    record = store._records[key]
    assert record.num_computed_tokens == 0

    # While handed into the step, only the exact post-step target is legal.
    assert not store.mark_computed(key, 8)
    assert not store.mark_computed(key, 12)
    assert not store.mark_computed(key, True)
    # Explicit success: the exact post-step target is accepted once.
    assert store.mark_computed(key, 10)
    assert record.num_computed_tokens == 10

    # After success, duplicate target is idempotent; anything else is not.
    assert store.mark_computed(key, 10)
    assert not store.mark_computed(key, 9)
    assert not store.mark_computed(key, 11)


def test_step_metadata_exposes_local_ids_but_never_wire_fields():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("w")
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.build_step_metadata(1, [], {}).accepted
    built = store.build_step_metadata(2, [_token(key, 8, step_seq=2)], {"w": 8})
    assert built.accepted
    entry = built.metadata.entries[0]

    # The metadata does carry the local block ids the local zeroing needs...
    assert all(
        isinstance(block_id, int) and block_id >= 0
        for group in entry.tables
        for block_id in group
    )
    assert all(
        isinstance(block_id, int) and block_id >= 0
        for group in entry.delta
        for block_id in group
    )
    # ...but never allocator ids, block objects, managers, or mutation/free
    # methods.
    assert not hasattr(entry, "allocator_id")
    assert not hasattr(entry, "blocks")
    assert not hasattr(entry, "manager")
    assert not hasattr(entry, "free")
    assert not hasattr(entry, "flush")
    assert not hasattr(built.metadata, "allocator_id")
    assert not hasattr(built.metadata, "manager")

    # Wire protocol objects carry no local block id fields at all.
    from dataclasses import fields

    token_fields = {field.name for field in fields(OwnerLeaseToken)}
    assert token_fields == {
        "key",
        "owner_id",
        "step_seq",
        "command_seq",
        "runnable_num_tokens",
    }
    command_fields = {field.name for field in fields(OwnerCommand)}
    assert "tables" not in command_fields
    assert "block_ids" not in command_fields
    assert "delta" not in command_fields
    pool_fields = {field.name for field in fields(OwnerCachePoolSnapshot)}
    assert "tables" not in pool_fields
    assert "block_ids" not in pool_fields


# -- G3 atomic batch mark ----------------------------------------------------


def test_mark_computed_batch_atomic_advance():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key_a = _key("a")
    key_b = _key("b")
    assert store.reserve(
        _command(key_a, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.reserve(
        _command(key_b, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.build_step_metadata(1, [], {}).accepted

    built = store.build_step_metadata(
        2,
        [_token(key_a, 8, step_seq=2), _token(key_b, 8, step_seq=2)],
        {"a": 8, "b": 8},
    )
    assert built.accepted
    mark = store.mark_computed_batch(built.metadata)
    assert mark.accepted
    assert mark.error is None
    assert mark.step_seq == 2

    # Every record advanced to its exact post-step target and every mark
    # expectation was consumed in the same all-or-nothing commit.
    assert store._records[key_a].num_computed_tokens == 8
    assert store._records[key_b].num_computed_tokens == 8
    assert store._pending_marks == {}

    # The consumed handoffs free the keys for the next token-bearing build
    # (the reserved horizons are first extended past the marked progress).
    assert store.build_step_metadata(3, [], {}).accepted
    assert store.extend(
        _command(key_a, OwnerCommandKind.EXTEND, required=16, seq=2)
    ).accepted
    assert store.extend(
        _command(key_b, OwnerCommandKind.EXTEND, required=16, seq=2)
    ).accepted
    assert store.build_step_metadata(4, [], {}).accepted
    again = store.build_step_metadata(
        5,
        [_token(key_a, 16, step_seq=5), _token(key_b, 16, step_seq=5)],
        {"a": 8, "b": 8},
    )
    assert again.accepted
    assert again.metadata.entries[0].pre_step_num_computed_tokens == 8
    assert again.metadata.entries[0].post_step_num_tokens == 16


def test_mark_computed_batch_partial_rejected_without_changes():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key_a = _key("a")
    key_b = _key("b")
    assert store.reserve(
        _command(key_a, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.reserve(
        _command(key_b, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.build_step_metadata(1, [], {}).accepted
    built = store.build_step_metadata(
        2,
        [_token(key_a, 8, step_seq=2), _token(key_b, 8, step_seq=2)],
        {"a": 8, "b": 8},
    )
    assert built.accepted
    metadata = built.metadata
    entry_a, entry_b = metadata.entries

    # One bad entry (wrong post-step target) rejects the whole batch before
    # any record advances or any expectation is consumed.
    bad = replace(
        metadata, entries=(replace(entry_a, post_step_num_tokens=99), entry_b)
    )
    rejected = store.mark_computed_batch(bad)
    assert not rejected.accepted
    assert "target mismatch" in rejected.error
    assert store._records[key_a].num_computed_tokens == 0
    assert store._records[key_b].num_computed_tokens == 0
    assert set(store._pending_marks) == {key_a, key_b}

    # A duplicate batch entry is also rejected without changes.
    dup = replace(metadata, entries=(entry_a, entry_a))
    assert not store.mark_computed_batch(dup).accepted

    # The untouched expectations still accept the exact original batch.
    assert store.mark_computed_batch(metadata).accepted
    assert store._records[key_a].num_computed_tokens == 8
    assert store._records[key_b].num_computed_tokens == 8


def test_mark_computed_batch_stale_duplicate_wrong_owner_foreign():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("f")
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.build_step_metadata(1, [], {}).accepted
    built = store.build_step_metadata(2, [_token(key, 8, step_seq=2)], {"f": 8})
    assert built.accepted
    metadata = built.metadata
    entry = metadata.entries[0]

    # Wrong-owner metadata never touches this store.
    wrong_owner = replace(metadata, owner_rank=1)
    assert not store.mark_computed_batch(wrong_owner).accepted
    assert "wrong-owner" in store.mark_computed_batch(wrong_owner).error

    # Foreign key: no active record for the snapshot's key.
    foreign = replace(metadata, entries=(replace(entry, key=_key("ghost")),))
    rejected = store.mark_computed_batch(foreign)
    assert not rejected.accepted
    assert "foreign" in rejected.error

    # A batch for a step that was never built (or an older one) is stale.
    future = replace(metadata, step_seq=99)
    assert not store.mark_computed_batch(future).accepted
    assert "stale" in store.mark_computed_batch(future).error

    # The exact batch succeeds once...
    assert store.mark_computed_batch(metadata).accepted
    assert store._records[key].num_computed_tokens == 8
    # ...and a second mark of the same step is a duplicate.
    duplicate = store.mark_computed_batch(metadata)
    assert not duplicate.accepted
    assert "duplicate" in duplicate.error

    # A newer build makes the old step stale as well.
    assert store.build_step_metadata(3, [], {}).accepted
    stale = store.mark_computed_batch(metadata)
    assert not stale.accepted
    assert "stale" in stale.error


def test_mark_computed_batch_empty_heartbeat_metadata():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("h")
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted

    heartbeat = store.build_step_metadata(1, [], {})
    assert heartbeat.accepted
    assert heartbeat.metadata.entries == ()
    # An empty batch is a valid no-op mark: nothing to advance, step fenced.
    assert store.mark_computed_batch(heartbeat.metadata).accepted
    assert store._records[key].num_computed_tokens == 0
    # Marking the same heartbeat again is a duplicate.
    assert not store.mark_computed_batch(heartbeat.metadata).accepted


def test_build_rejects_any_live_pending_mark_across_disjoint_keys():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key_a = _key("a")
    key_b = _key("b")
    assert store.reserve(
        _command(key_a, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.reserve(
        _command(key_b, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.build_step_metadata(1, [], {}).accepted
    first = store.build_step_metadata(2, [_token(key_a, 8, step_seq=2)], {"a": 8})
    assert first.accepted

    # Key A's execution is pending and unmarked.  A token-bearing build for
    # the disjoint key B must not advance the fence and strand A...
    disjoint = store.build_step_metadata(3, [_token(key_b, 8, step_seq=3)], {"b": 8})
    assert not disjoint.accepted
    assert "unconsumed pending mark" in disjoint.error
    assert "request_id='a'" in disjoint.error
    # ...and a heartbeat must not advance the fence over it either.
    heartbeat = store.build_step_metadata(3, [], {})
    assert not heartbeat.accepted
    assert "unconsumed pending mark" in heartbeat.error

    # Atomicity: the rejected builds changed nothing; A's expectation is
    # still pending and the exact batch can still be marked.
    assert store._records[key_a].num_computed_tokens == 0
    assert store._records[key_b].num_computed_tokens == 0
    assert set(store._pending_marks) == {key_a}
    assert store.mark_computed_batch(first.metadata).accepted
    assert store._records[key_a].num_computed_tokens == 8

    # With the pending execution consumed, the next build for B proceeds.
    next_step = store.build_step_metadata(3, [_token(key_b, 8, step_seq=3)], {"b": 8})
    assert next_step.accepted
    assert [entry.key for entry in next_step.metadata.entries] == [key_b]


def test_mark_computed_batch_missing_entry_rejected_atomically():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key_a = _key("a")
    key_b = _key("b")
    assert store.reserve(
        _command(key_a, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.reserve(
        _command(key_b, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.build_step_metadata(1, [], {}).accepted
    built = store.build_step_metadata(
        2,
        [_token(key_a, 8, step_seq=2), _token(key_b, 8, step_seq=2)],
        {"a": 8, "b": 8},
    )
    assert built.accepted
    metadata = built.metadata
    entry_a = metadata.entries[0]

    # A metadata object missing one expected entry must not mark the subset
    # and strand the remainder: the whole batch is rejected with no changes.
    partial = replace(metadata, entries=(entry_a,))
    rejected = store.mark_computed_batch(partial)
    assert not rejected.accepted
    assert "missing" in rejected.error
    assert store._records[key_a].num_computed_tokens == 0
    assert store._records[key_b].num_computed_tokens == 0
    assert set(store._pending_marks) == {key_a, key_b}

    # The untouched expectations still accept the exact full batch.
    assert store.mark_computed_batch(metadata).accepted
    assert store._records[key_a].num_computed_tokens == 8
    assert store._records[key_b].num_computed_tokens == 8


def test_mark_computed_batch_generation_fence():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("g")
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.build_step_metadata(1, [], {}).accepted
    built = store.build_step_metadata(2, [_token(key, 8, step_seq=2)], {"g": 8})
    assert built.accepted
    first_metadata = built.metadata

    # Same-key ABA: preempt, flush, and re-reserve the exact key before the
    # handed-in step is marked.  The stale batch can never advance the
    # recycled record because its allocation generation changed.
    assert store.preempt(
        _command(key, OwnerCommandKind.PREEMPT, required=8, seq=3)
    ).accepted
    store.flush()
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=12, seq=4)
    ).accepted
    rejected = store.mark_computed_batch(first_metadata)
    assert not rejected.accepted
    assert "generation" in rejected.error
    assert store._records[key].num_computed_tokens == 0

    # The recycled record builds and marks normally with its own snapshot.
    assert store.build_step_metadata(3, [], {}).accepted
    second = store.build_step_metadata(4, [_token(key, 12, step_seq=4)], {"g": 12})
    assert second.accepted
    assert store.mark_computed_batch(second.metadata).accepted
    assert store._records[key].num_computed_tokens == 12


def test_mark_computed_batch_expectation_generation_mismatch():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("x")
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.build_step_metadata(1, [], {}).accepted
    built = store.build_step_metadata(2, [_token(key, 8, step_seq=2)], {"x": 8})
    assert built.accepted
    metadata = built.metadata

    # Corrupt the armed expectation's generation (as if the record had been
    # recycled out from under the expectation without a new build).  The
    # batch must fail without advancing anything even though the snapshot
    # entry still matches the record.
    store._pending_marks[key] = replace(
        store._pending_marks[key], allocation_generation=999
    )
    rejected = store.mark_computed_batch(metadata)
    assert not rejected.accepted
    assert "expectation generation" in rejected.error
    assert store._records[key].num_computed_tokens == 0
    assert key in store._pending_marks


@pytest.mark.parametrize("accepted_draft_count", range(8))
def test_speculative_mark_commits_every_verified_prefix_and_reuses_reserved_tail(
    accepted_draft_count: int,
):
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("spec")
    initial = _free(manager)
    reserve = store.reserve(_command(key, OwnerCommandKind.RESERVE, required=16, seq=1))
    assert reserve.accepted
    assert _sizes(reserve.tables) == (4, 4)

    # The allocation publication heartbeat separates RESERVE from execution.
    assert store.build_step_metadata(1, [], {}).accepted
    built = store.build_step_metadata(
        2,
        [_token(key, 16, step_seq=2)],
        {"spec": 8},
        {"spec": list(range(7))},
    )
    assert built.accepted
    (entry,) = built.metadata.entries
    assert entry.pre_step_num_computed_tokens == 0
    assert entry.post_step_num_tokens == 8
    assert entry.num_speculative_tokens == 7

    # A accepted drafts plus the terminal correction/bonus token commit A+1
    # positions; the complete eight-position execution remains resident.
    committed = accepted_draft_count + 1
    marked = store.mark_computed_batch(built.metadata, {key: committed})
    assert marked.accepted
    assert store._records[key].num_computed_tokens == committed
    assert store._records[key].reserved_num_tokens == 16
    assert store.get_block_table(key) == reserve.tables
    assert _free(manager) == initial - 8

    # The next target step starts from the verified prefix and overwrites the
    # rejected suffix without allocating it again.
    next_count = min(8, 16 - committed)
    next_step = store.build_step_metadata(
        3,
        [_token(key, 16, step_seq=3)],
        {"spec": next_count},
        {"spec": list(range(next_count - 1))},
    )
    assert next_step.accepted
    (next_entry,) = next_step.metadata.entries
    assert next_entry.pre_step_num_computed_tokens == committed
    assert next_entry.post_step_num_tokens == committed + next_count
    assert next_entry.tables == reserve.tables
    assert store.mark_computed_batch(
        next_step.metadata, {key: committed + next_count}
    ).accepted

    # A later EXTEND preserves the logical watermark for sliding-window
    # retirement while the manager's existing table prevents duplicate tail
    # allocation.
    extended = store.extend(_command(key, OwnerCommandKind.EXTEND, required=20, seq=2))
    assert extended.accepted
    assert _sizes(extended.tables) == (5, 5)
    assert _sizes(extended.delta) == (1, 1)
    if accepted_draft_count <= 1:
        # With at most ten committed positions, the eight-token window has
        # not crossed a whole four-token block. Treating the reserved horizon
        # (16) as computed would incorrectly null the first two blocks here.
        assert extended.tables[1][:4] == reserve.tables[1]
    # The synthetic second cache group may now retire blocks that are outside
    # the *committed* window. At most the two genuinely new horizon blocks are
    # added; the provisional suffix is never allocated a second time.
    assert initial - _free(manager) <= 10


def test_speculative_abort_discards_logical_progress_before_preempt_flush():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("aborted")
    initial = _free(manager)
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.build_step_metadata(1, [], {}).accepted
    built = store.build_step_metadata(
        2,
        [_token(key, 8, step_seq=2)],
        {"aborted": 8},
        {"aborted": list(range(7))},
    )
    assert built.accepted

    preempted = store.preempt(
        _command(key, OwnerCommandKind.PREEMPT, required=8, seq=2)
    )
    assert preempted.accepted and preempted.deferred
    rejected = store.mark_computed_batch(built.metadata, {key: 8})
    assert not rejected.accepted
    assert "pending free" in rejected.error
    assert store._records[key].num_computed_tokens == 0

    assert store.flush() == (key,)
    assert store.get_block_table(key) is None
    assert _free(manager) == initial


def test_speculative_mark_requires_atomic_in_range_commit_mapping():
    manager = _make_manager()
    store = RequestOwnedKVStore(manager, owner_rank=0)
    key = _key("spec")
    assert store.reserve(
        _command(key, OwnerCommandKind.RESERVE, required=8, seq=1)
    ).accepted
    assert store.build_step_metadata(1, [], {}).accepted
    built = store.build_step_metadata(
        2,
        [_token(key, 8, step_seq=2)],
        {"spec": 4},
        {"spec": [1, 2, 3]},
    )
    assert built.accepted

    missing = store.mark_computed_batch(built.metadata)
    assert not missing.accepted
    assert "missing speculative commits" in missing.error
    assert store._records[key].num_computed_tokens == 0

    too_low = store.mark_computed_batch(built.metadata, {key: 0})
    assert not too_low.accepted
    assert "must be in [1, 4]" in too_low.error
    assert store._records[key].num_computed_tokens == 0

    too_high = store.mark_computed_batch(built.metadata, {key: 5})
    assert not too_high.accepted
    assert "must be in [1, 4]" in too_high.error
    assert store._records[key].num_computed_tokens == 0
    assert store.mark_computed_batch(built.metadata, {key: 1}).accepted
