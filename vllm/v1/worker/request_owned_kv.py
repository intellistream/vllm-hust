# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G2 worker-local physical KV store for request-owned attention.

Composition boundary: ``AttentionLeaseManager``
(vllm.v1.core.sched.ownership) remains the sole logical fence -- it
consumes the owner command stream, enforces the per-owner command sequence
and the per-request-id epoch fence, tracks publication, and emits exactly
one ``OwnerReceiptBatch`` per step.  This module is the other half of the
pair: it directly reuses one injected real :class:`KVCacheManager` and owns
only the physical state the logical manager does not model -- private
epoch-qualified allocator ids derived from :class:`OwnerLeaseKey` (so
identical numeric local ids across independent stores never collide in the
shared per-rank block pool), private per-group block tables and newly
allocated deltas returned to the worker integration, the block-ID-free
capacity snapshot of the shared per-rank pool, and deferred frees for
PREEMPT/RELEASE released to the manager only at an explicit post-execute
completion fence (:meth:`RequestOwnedKVStore.flush`).  It emits no
:class:`OwnerReceipt` and performs no token/block estimation of its own.

Scope (fail closed): prefix caching, KV connectors, spec decode, and
RESTORE are out of scope; :meth:`restore` always rejects.  The store keeps
no local epoch fence: physical records are keyed by the full
:class:`OwnerLeaseKey` (request id plus epoch), so a command whose epoch
does not match the record simply misses the lookup and is rejected without
touching any state.  Commands whose kind or record state the store does not
own are rejected with ``accepted=False`` and never advance local records.
PREEMPT/RELEASE only mark the record pending a free; blocks stay resident
(and the allocator id non-reusable) until :meth:`flush`.  A PREEMPT flushed
while the request is still alive leaves an exact-key tombstone so the
subsequent valid RELEASE (request aborted while preempted) is accepted as a
non-deferred no-op; a RESERVE on that key consumes the tombstone and
allocates fresh blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.ownership import (
    OwnerCacheGroupSnapshot,
    OwnerCachePoolSnapshot,
    OwnerCommand,
    OwnerCommandKind,
    OwnerLeaseKey,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    get_kv_cache_spec_kind,
)
from vllm.v1.request import RequestStatus


class _AllocationRequest:
    """Private Request-like facade: only the fields the real manager
    observes on the prefix-cache-disabled path, never a copied prompt
    token payload."""

    __slots__ = (
        "request_id",
        "num_tokens",
        "num_computed_tokens",
        "num_prompt_tokens",
        "status",
    )

    def __init__(
        self,
        request_id: str,
        num_tokens: int,
        num_computed_tokens: int,
        num_prompt_tokens: int,
    ) -> None:
        self.request_id = request_id
        self.num_tokens = num_tokens
        self.num_computed_tokens = num_computed_tokens
        self.num_prompt_tokens = num_prompt_tokens
        self.status = RequestStatus.RUNNING


@dataclass
class _Record:
    """Private per-lease physical state held by the store.  Block counts
    are never accumulated here: tables are always derived from the real
    manager via the epoch-qualified allocator id."""

    key: OwnerLeaseKey
    allocator_id: str
    num_prompt_tokens: int
    num_computed_tokens: int = 0
    #: Chunk horizon of the last accepted RESERVE/EXTEND: the exclusive
    #: upper bound that ``mark_computed`` must not exceed.
    reserved_num_tokens: int = 0
    pending_free: bool = False
    #: True when the pending free was caused by PREEMPT (a flushed preempt
    #: leaves an exact-key tombstone for the later logical RELEASE).
    preempted: bool = False


@dataclass(frozen=True)
class AllocationResult:
    """Outcome of a RESERVE/EXTEND: private worker-facing per-group block
    tables (``tables`` full, ``delta`` newly allocated); ``None`` on
    rejection, empty tuples for an accepted empty lease."""

    accepted: bool
    key: OwnerLeaseKey
    tables: tuple[tuple[int, ...], ...] | None = None
    delta: tuple[tuple[int, ...], ...] | None = None
    error: str | None = None


@dataclass(frozen=True)
class DeferredFreeResult:
    """Outcome of a PREEMPT/RELEASE/RESTORE."""

    accepted: bool
    key: OwnerLeaseKey
    deferred: bool = False
    error: str | None = None


def _reject_allocation(command: OwnerCommand, error: str) -> AllocationResult:
    return AllocationResult(accepted=False, key=command.key, error=error)


def _reject_free(command: OwnerCommand, error: str) -> DeferredFreeResult:
    return DeferredFreeResult(accepted=False, key=command.key, error=error)


def _derivable_bytes_per_block(config: KVCacheConfig, total_blocks: int) -> int | None:
    """Pool-wide physical bytes represented by one shared-pool block ID.

    Ascend DSV4 uses one globally unique block-ID space for the whole
    per-rank pool, so a single block ID covers ``size / total_blocks`` bytes
    of every KV tensor.  ``None`` when no tensors exist or a tensor does not
    divide evenly across the pool (the emitter cannot express it)."""
    if not config.kv_cache_tensors or total_blocks <= 0:
        return None
    if any(tensor.size % total_blocks for tensor in config.kv_cache_tensors):
        return None
    return sum(tensor.size // total_blocks for tensor in config.kv_cache_tensors)


class RequestOwnedKVStore:
    """Worker-local physical KV store over one injected KVCacheManager;
    the injected manager stays the authority on the shared block pool."""

    def __init__(self, kv_cache_manager: KVCacheManager, owner_rank: int) -> None:
        self._manager = kv_cache_manager
        self._owner_rank = owner_rank
        self._records: dict[OwnerLeaseKey, _Record] = {}
        #: Exact keys physically freed by a flushed PREEMPT while the
        #: request may still be alive; consumed by RELEASE or RESERVE.
        self._tombstones: set[OwnerLeaseKey] = set()
        self._num_groups = kv_cache_manager.num_kv_cache_groups
        self._group_specs: tuple[KVCacheSpec, ...] = tuple(
            group.kv_cache_spec
            for group in kv_cache_manager.kv_cache_config.kv_cache_groups
        )

    def reserve(self, command: OwnerCommand) -> AllocationResult:
        """Physically reserve the chunk of a RESERVE command; request
        facts come from ``command.allocation``.  The manager gets
        ``full_sequence_must_fit=True``, ``has_scheduled_reqs=False`` and
        facade ``num_tokens`` = the chunk horizon; on ``None`` no record is
        created and no local state changes."""
        if command.kind is not OwnerCommandKind.RESERVE:
            return _reject_allocation(command, "reserve() requires a RESERVE command")
        if command.allocation is None:
            return _reject_allocation(
                command, "RESERVE without an allocation descriptor is unsupported"
            )
        existing = self._records.get(command.key)
        if existing is not None:
            if existing.pending_free:
                return _reject_allocation(
                    command, "lease held pending free until the flush fence"
                )
            return _reject_allocation(command, "duplicate reserve for active lease")

        allocator_id = self._allocator_id(command.key)
        computed = command.allocation.num_computed_tokens
        num_new = command.required_num_tokens - computed
        blocks: KVCacheBlocks | None = None
        if num_new > 0:
            blocks = self._allocate(
                allocator_id,
                command.allocation.num_prompt_tokens,
                computed,
                num_new,
                command.required_num_tokens,
            )
            if blocks is None:
                return _reject_allocation(command, "insufficient KV cache to reserve")

        record = _Record(
            key=command.key,
            allocator_id=allocator_id,
            num_prompt_tokens=command.allocation.num_prompt_tokens,
            num_computed_tokens=computed,
            reserved_num_tokens=command.required_num_tokens,
        )
        self._records[command.key] = record
        self._tombstones.discard(command.key)
        return self._accepted_allocation(command, record, blocks)

    def extend(self, command: OwnerCommand) -> AllocationResult:
        """Physically extend an active lease using stored request facts;
        on ``None`` from the manager the record stays unchanged."""
        if command.kind is not OwnerCommandKind.EXTEND:
            return _reject_allocation(command, "extend() requires an EXTEND command")
        record = self._records.get(command.key)
        if record is None:
            return _reject_allocation(command, "no active lease to extend")
        if record.pending_free:
            return _reject_allocation(
                command, "lease held pending free until the flush fence"
            )

        num_new = command.required_num_tokens - record.num_computed_tokens
        if num_new <= 0:
            return self._accepted_allocation(command, record, None)
        blocks = self._allocate(
            record.allocator_id,
            record.num_prompt_tokens,
            record.num_computed_tokens,
            num_new,
            command.required_num_tokens,
        )
        if blocks is None:
            return _reject_allocation(command, "insufficient KV cache to extend")
        record.reserved_num_tokens = command.required_num_tokens
        return self._accepted_allocation(command, record, blocks)

    def preempt(self, command: OwnerCommand) -> DeferredFreeResult:
        """Defer the physical free of a PREEMPT until the flush fence."""
        if command.kind is not OwnerCommandKind.PREEMPT:
            return _reject_free(command, "preempt() requires a PREEMPT command")
        record = self._records.get(command.key)
        if record is None:
            return _reject_free(command, "no active lease to preempt")
        record.pending_free = True
        record.preempted = True
        return DeferredFreeResult(accepted=True, key=command.key, deferred=True)

    def release(self, command: OwnerCommand) -> DeferredFreeResult:
        """Defer the physical free of a RELEASE until the flush fence.

        The lookup is keyed by the full lease key (request id plus epoch),
        so a stale-epoch RELEASE can never free the newer epoch's blocks:
        it misses the newer record and is rejected.  A RELEASE for an
        exact key that was physically freed by a flushed PREEMPT is a
        valid non-deferred no-op that ends the tombstone."""
        if command.kind is not OwnerCommandKind.RELEASE:
            return _reject_free(command, "release() requires a RELEASE command")
        record = self._records.get(command.key)
        if record is None:
            if command.key not in self._tombstones:
                return _reject_free(command, "no lease to release")
            self._tombstones.discard(command.key)
            return DeferredFreeResult(accepted=True, key=command.key, deferred=False)
        record.pending_free = True
        record.preempted = False
        return DeferredFreeResult(accepted=True, key=command.key, deferred=True)

    def restore(self, command: OwnerCommand) -> DeferredFreeResult:
        """RESTORE is out of scope: always fails closed."""
        if command.kind is not OwnerCommandKind.RESTORE:
            return _reject_free(command, "restore() requires a RESTORE command")
        return _reject_free(
            command, "RESTORE is out of scope for the physical KV store"
        )

    def mark_computed(self, key: OwnerLeaseKey, num_tokens: int) -> bool:
        """Record monotonic computed progress for an active lease; False
        (changing nothing) on unknown/pending-free/malformed/regressive
        updates or on progress beyond the reserved chunk horizon."""
        record = self._records.get(key)
        if record is None or record.pending_free:
            return False
        if isinstance(num_tokens, bool) or not isinstance(num_tokens, int):
            return False
        if num_tokens < 0 or num_tokens < record.num_computed_tokens:
            return False
        if num_tokens > record.reserved_num_tokens:
            return False
        record.num_computed_tokens = num_tokens
        return True

    def get_block_table(self, key: OwnerLeaseKey) -> tuple[tuple[int, ...], ...] | None:
        """Private full per-group tables; readable until :meth:`flush`."""
        record = self._records.get(key)
        if record is None:
            return None
        return self._tables(record)

    def flush(self) -> tuple[OwnerLeaseKey, ...]:
        """Post-execute completion fence: free all deferred blocks now
        that the executing GPU step finished writing them."""
        freed: list[OwnerLeaseKey] = []
        for key, record in list(self._records.items()):
            if not record.pending_free:
                continue
            self._manager.free(
                _AllocationRequest(
                    request_id=record.allocator_id,
                    num_tokens=0,
                    num_computed_tokens=0,
                    num_prompt_tokens=record.num_prompt_tokens,
                )
            )
            if record.preempted:
                self._tombstones.add(key)
            del self._records[key]
            freed.append(key)
        return tuple(freed)

    def pool_snapshot(self) -> OwnerCachePoolSnapshot:
        """Immutable, block-ID-free protocol snapshot of the shared pool.
        All blocks of non-flushed records count as both allocated and
        resident: pages stay resident (and the allocator id non-reusable)
        until :meth:`flush`."""
        pool = self._manager.block_pool
        total = pool.num_gpu_blocks
        free = pool.get_num_free_blocks()
        allocated: list[int] = [0] * self._num_groups
        for record in self._records.values():
            for group, table in enumerate(self._tables(record)):
                allocated[group] += len(table)
        groups = tuple(
            OwnerCacheGroupSnapshot(
                group_index=index,
                spec_kind=get_kv_cache_spec_kind(spec).value,
                effective_tokens_per_block=spec.block_size
                * max(1, int(getattr(spec, "compress_ratio", 1))),
                allocated_blocks=allocated[index],
                resident_blocks=allocated[index],
            )
            for index, spec in enumerate(self._group_specs)
        )
        return OwnerCachePoolSnapshot(
            owner_rank=self._owner_rank,
            total_blocks=total,
            free_blocks=free,
            bytes_per_block=_derivable_bytes_per_block(
                self._manager.kv_cache_config, total
            ),
            groups=groups,
        )

    def _allocator_id(self, key: OwnerLeaseKey) -> str:
        return f"{self._owner_rank}:{key.request_id}:e{key.owner_epoch}"

    def _allocate(
        self,
        allocator_id: str,
        num_prompt_tokens: int,
        num_computed_tokens: int,
        num_new_tokens: int,
        num_tokens: int,
    ) -> KVCacheBlocks | None:
        """One real-manager allocation; the sole source of block counts."""
        request = _AllocationRequest(
            request_id=allocator_id,
            num_tokens=num_tokens,
            num_computed_tokens=num_computed_tokens,
            num_prompt_tokens=num_prompt_tokens,
        )
        return self._manager.allocate_slots(
            request,
            num_new_tokens=num_new_tokens,
            full_sequence_must_fit=True,
            has_scheduled_reqs=False,
        )

    def _tables(self, record: _Record) -> tuple[tuple[int, ...], ...]:
        ids = self._manager.get_block_ids(record.allocator_id)
        return tuple(tuple(group) for group in ids)

    def _accepted_allocation(
        self,
        command: OwnerCommand,
        record: _Record,
        blocks: KVCacheBlocks | None,
    ) -> AllocationResult:
        return AllocationResult(
            accepted=True,
            key=command.key,
            tables=self._tables(record),
            delta=(
                tuple(tuple(group) for group in blocks.get_block_ids())
                if blocks is not None
                else tuple(() for _ in range(self._num_groups))
            ),
        )
