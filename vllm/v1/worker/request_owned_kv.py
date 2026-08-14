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

Scope (fail closed): prefix caching and spec decode remain out of scope.
O1 adds an exclusive host-KV lifecycle: a durable PREEMPT flush retains only
block-ID-free cold allocation facts; RESTORE allocates the final destination,
which cannot become runnable until the concrete worker zeros it and completes
the exact bulk H2D.  The store keeps no local epoch fence: physical records are
keyed by the full
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

G3 adds the worker-local one-step execution metadata API
(:meth:`RequestOwnedKVStore.build_step_metadata`): it freezes an immutable,
fully detached per-step snapshot (step sequence, exact lease keys, full
per-group block tables, pre-step computed counts, post-step targets, a
physical allocation generation, and the newly allocated local block deltas
needed for local zeroing) from the exact own-rank scheduled lease tokens
plus the exact per-key scheduled token counts.  The post-step target is
``pre-step computed + scheduled count`` and may sit strictly below the
cumulative lease horizon (``OwnerLeaseToken.runnable_num_tokens``), which
is a horizon, not this step's target.  A zero-token heartbeat builds empty
execution metadata even when new grants were published, and retains every
pending delta for the later token-bearing step.  The snapshot never exposes
allocator ids, block objects, the injected manager, or any mutation/free
method, and no local block id ever travels on a scheduler wire: the
metadata is handed only to the local step consumer.  The builder is
one-step fenced (stale step reuse is rejected), rejects missing/extra
lease and count state, duplicates, wrong-owner, pending-free,
out-of-horizon targets, same-step RESERVE/EXTEND-plus-authorization for one
key, and a new build for a key with an unconsumed pending mark, and hands
newly allocated deltas into a step only when the build succeeds: a failed
build retains every pending delta.  A handed-in mark expectation is fenced
by the record's physical allocation generation, so a stale snapshot can
never advance computed progress on a recycled same-key record.
  The atomic batch mark
(:meth:`RequestOwnedKVStore.mark_computed_batch`) validates every entry
(exact key, generation, pending expectation, pre-step base, post-step
target) before advancing any record, and each step may be marked at
most once: stale, duplicate, wrong-owner, foreign, or partial batches
fail without changes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.ownership import (
    OwnerAdmissionStatus,
    OwnerCacheGroupSnapshot,
    OwnerCachePoolSnapshot,
    OwnerCommand,
    OwnerCommandKind,
    OwnerLeaseKey,
    OwnerLeaseToken,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
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


def _request_owned_compress_ratio(kv_cache_spec: KVCacheSpec) -> int:
    """Return one fail-closed allocation-binding compression ratio."""

    ratio = getattr(kv_cache_spec, "compress_ratio", 1)
    if ratio is None:
        ratio = 1
    if isinstance(ratio, bool) or not isinstance(ratio, int) or ratio <= 0:
        raise ValueError(
            "request-owned KV cache spec has invalid compress_ratio "
            f"{ratio!r}; expected a positive non-bool integer"
        )
    return ratio


def request_owned_allocation_binding_spec(
    kv_cache_spec: KVCacheSpec,
) -> KVCacheSpec:
    """Resolve the concrete spec that binds a request-owned group allocation.

    DeepSeek-V4 may wrap MLA specs with the same physical ``block_size`` but
    different compression ratios in one :class:`UniformTypeKVCacheSpecs`.
    The real request-owned manager binds that group to the smallest ratio,
    which has the largest storage footprint.  Every consumer of group table
    cardinality must use the same representative rather than treating the
    ratio-less outer wrapper as uncompressed.
    """

    if not isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        _request_owned_compress_ratio(kv_cache_spec)
        return kv_cache_spec
    inner_specs = tuple(kv_cache_spec.kv_cache_specs.values())
    if not inner_specs:
        raise ValueError(
            "request-owned uniform KV cache group has no inner KV cache specs"
        )
    ratios = tuple(_request_owned_compress_ratio(spec) for spec in inner_specs)
    representative = inner_specs[ratios.index(min(ratios))]
    if representative.block_size != kv_cache_spec.block_size:
        raise ValueError(
            "request-owned uniform KV cache representative block_size "
            f"{representative.block_size} != wrapper block_size "
            f"{kv_cache_spec.block_size}"
        )
    return representative


def request_owned_effective_tokens_per_block(kv_cache_spec: KVCacheSpec) -> int:
    """Logical token capacity of one real group-table physical block ID."""

    representative = request_owned_allocation_binding_spec(kv_cache_spec)
    return representative.block_size * _request_owned_compress_ratio(representative)


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
    #: Monotonic physical allocation generation: changes on every record
    #: creation so a recycled same-key record (preempt/flush/reserve) can
    #: never be confused with its predecessor by a stale snapshot.
    generation: int = 0
    #: Newly allocated local block ids since the last successful step build
    #: handoff (accumulated across RESERVE/EXTEND), or ``None`` when nothing
    #: is pending.  Only a successful build may clear it.
    pending_delta: tuple[tuple[int, ...], ...] | None = None
    #: True while the record's last allocation happened after the previous
    #: successful build: the scheduler must not both allocate and authorize
    #: execution for the same key in one step.
    allocated_since_build: bool = False
    #: True after RESTORE allocated the final destination and until the
    #: following RESERVE reactivates it.
    restored_from_host: bool = False
    #: True only after the owner bulk H2D completed at the pre-forward seam.
    restore_ready: bool = False
    #: Block-ID-free logical positions that held concrete computed-prefix KV
    #: in the preempted source. A fresh restore destination may allocate real
    #: blocks where the source hybrid table carried null placeholders; H2D
    #: must still address only the original concrete positions.
    restore_source_block_indices: tuple[tuple[int, ...], ...] | None = None


@dataclass(frozen=True, slots=True)
class RequestOwnedColdSnapshot:
    """Block-ID-free facts needed to address one durable host prefix."""

    key: OwnerLeaseKey
    owner_rank: int
    num_prompt_tokens: int
    num_computed_tokens: int
    reserved_num_tokens: int
    source_block_indices: tuple[tuple[int, ...], ...]


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


@dataclass(frozen=True, slots=True)
class RequestOwnedStepEntry:
    """Immutable worker-local execution metadata for one lease of one step.

    Carries only detached facts: the exact lease key, the physical
    allocation generation (rejects same-key ABA), the pre-step computed
    count, the post-step token target the step must mark on explicit
    success, the full detached per-group block tables, and the newly
    allocated local block ids (per group) that still need local zeroing.
    No allocator id, block object, manager reference, or mutation/free
    method is exposed.
    """

    key: OwnerLeaseKey
    allocation_generation: int
    pre_step_num_computed_tokens: int
    post_step_num_tokens: int
    tables: tuple[tuple[int, ...], ...]
    delta: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.key, OwnerLeaseKey):
            raise TypeError(f"key must be an OwnerLeaseKey, got {self.key!r}.")
        for name in (
            "allocation_generation",
            "pre_step_num_computed_tokens",
            "post_step_num_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(
                    f"{name} must be a nonnegative non-bool int, got {value!r}."
                )
        if self.post_step_num_tokens == 0:
            raise ValueError(
                "post_step_num_tokens must be positive: a published lease "
                "token is never empty, got 0."
            )
        if not isinstance(self.tables, tuple) or not isinstance(self.delta, tuple):
            raise TypeError("tables and delta must be tuples of group tables")
        if len(self.tables) != len(self.delta):
            raise ValueError(
                "tables and delta must cover the same groups, got "
                f"{len(self.tables)} != {len(self.delta)}."
            )
        for tables in (self.tables, self.delta):
            for group in tables:
                if not isinstance(group, tuple):
                    raise TypeError("each group table must be a tuple of block ids")
                for block_id in group:
                    if (
                        isinstance(block_id, bool)
                        or not isinstance(block_id, int)
                        or block_id < 0
                    ):
                        raise TypeError(
                            "block ids must be nonnegative non-bool ints, "
                            f"got {block_id!r}."
                        )


@dataclass(frozen=True, slots=True)
class RequestOwnedKVSnapshot:
    """Detached identity and physical layout of one owner-local lease.

    The snapshot is worker-private because it carries rank-local block ids; it
    must never cross the scheduler wire.  ``allocation_generation`` fences an
    offload completion against same-key ABA after preempt/reallocation.

    This is only an observation.  Holding a snapshot does not pin the backing
    allocation or grant authority to mutate/free it.
    """

    key: OwnerLeaseKey
    owner_rank: int
    allocation_generation: int
    num_computed_tokens: int
    reserved_num_tokens: int
    pending_free: bool
    tables: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.key, OwnerLeaseKey):
            raise TypeError(f"key must be an OwnerLeaseKey, got {self.key!r}.")
        for name in (
            "owner_rank",
            "allocation_generation",
            "num_computed_tokens",
            "reserved_num_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(
                    f"{name} must be a nonnegative non-bool int, got {value!r}."
                )
        if not isinstance(self.pending_free, bool):
            raise TypeError(
                "pending_free must be a bool, got "
                f"{type(self.pending_free).__name__} ({self.pending_free!r})."
            )
        if self.num_computed_tokens > self.reserved_num_tokens:
            raise ValueError(
                "num_computed_tokens must not exceed reserved_num_tokens, got "
                f"{self.num_computed_tokens} > {self.reserved_num_tokens}."
            )
        if not isinstance(self.tables, tuple):
            raise TypeError("tables must be a tuple of group tables")
        for group in self.tables:
            if not isinstance(group, tuple):
                raise TypeError("each group table must be a tuple of block ids")
            for block_id in group:
                if (
                    isinstance(block_id, bool)
                    or not isinstance(block_id, int)
                    or block_id < 0
                ):
                    raise TypeError(
                        "block ids must be nonnegative non-bool ints, "
                        f"got {block_id!r}."
                    )


@dataclass(frozen=True, slots=True)
class RequestOwnedStepMetadata:
    """Immutable, fully detached execution metadata batch for one step.

    ``entries`` carries exactly the scheduled own-rank lease keys of the
    step (in batch order) and nothing else; a zero-local-owner batch is a
    valid empty tuple.  The batch is worker-local by contract: it may carry
    local block ids, but no scheduler wire object ever does.
    """

    step_seq: int
    owner_rank: int
    entries: tuple[RequestOwnedStepEntry, ...]

    def __post_init__(self) -> None:
        for name in ("step_seq", "owner_rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(
                    f"{name} must be a nonnegative non-bool int, got {value!r}."
                )
        if self.step_seq == 0:
            raise ValueError("step_seq must be positive, got 0.")
        if not isinstance(self.entries, tuple):
            raise TypeError(f"entries must be a tuple, got {self.entries!r}.")
        for entry in self.entries:
            if not isinstance(entry, RequestOwnedStepEntry):
                raise TypeError(
                    f"entries must contain RequestOwnedStepEntry, got {entry!r}."
                )


@dataclass(frozen=True, slots=True)
class RequestOwnedStepBuildCheckpoint:
    """Private rollback facts for an uncommitted empty control-step build."""

    last_built_step_seq: int | None
    marked_step_seq: int | None
    active_metadata: RequestOwnedStepMetadata | None
    allocated_since_build: tuple[tuple[OwnerLeaseKey, int, bool], ...]


@dataclass(frozen=True, slots=True)
class RequestOwnedStepMetadataResult:
    """Outcome of :meth:`RequestOwnedKVStore.build_step_metadata`."""

    accepted: bool
    step_seq: int
    metadata: RequestOwnedStepMetadata | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RequestOwnedStepMarkResult:
    """Outcome of :meth:`RequestOwnedKVStore.mark_computed_batch`."""

    accepted: bool
    step_seq: int
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


@dataclass(frozen=True, slots=True)
class _MarkExpectation:
    """Private handed-in mark contract for one lease key.

    Set when a successful step build hands the key into a step and consumed
    by the exact post-step-target ``mark_computed``.  ``allocation_generation``
    fences the expectation against a recycled same-key record (ABA), and
    ``step_seq`` documents which step handed the key in.
    """

    step_seq: int
    allocation_generation: int
    post_step_num_tokens: int


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
        self._cold_records: dict[OwnerLeaseKey, RequestOwnedColdSnapshot] = {}
        self._num_groups = kv_cache_manager.num_kv_cache_groups
        self._group_specs: tuple[KVCacheSpec, ...] = tuple(
            group.kv_cache_spec
            for group in kv_cache_manager.kv_cache_config.kv_cache_groups
        )
        #: Monotonic physical allocation generation; every record creation
        #: consumes the next value so same-key ABA is observable.
        self._generation_counter: int = 0
        #: Step fence of the last successful G3 build (one-step fenced).
        self._last_built_step_seq: int | None = None
        #: The immutable metadata of the last successful build, readable
        #: pre-flush; replaced by the next successful build.
        self._active_metadata: RequestOwnedStepMetadata | None = None
        #: Handed-in mark expectations keyed by lease key; fenced by the
        #: record's allocation generation and consumed by a successful
        #: ``mark_computed``.  Deliberately survives record flush so a stale
        #: snapshot cannot advance a recycled same-key record.
        self._pending_marks: dict[OwnerLeaseKey, _MarkExpectation] = {}
        #: Step fence of the last successfully marked batch (each step may
        #: be marked at most once; a duplicate mark attempt is rejected).
        self._marked_step_seq: int | None = None

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
            if existing.restored_from_host:
                if not existing.restore_ready:
                    return _reject_allocation(command, "bulk restore is not complete")
                assert command.allocation is not None
                if command.required_num_tokens != existing.reserved_num_tokens:
                    return _reject_allocation(
                        command,
                        "RESERVE after RESTORE must match the reserved destination "
                        "horizon",
                    )
                if (
                    command.allocation.num_prompt_tokens != existing.num_prompt_tokens
                    or command.allocation.num_computed_tokens
                    != existing.num_computed_tokens
                    or command.allocation.status is not OwnerAdmissionStatus.PREEMPTED
                ):
                    return _reject_allocation(
                        command,
                        "RESERVE after RESTORE must exactly describe the preserved "
                        "preempted prefix",
                    )
                return self._accepted_allocation(command, existing, None)
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

        self._generation_counter += 1
        record = _Record(
            key=command.key,
            allocator_id=allocator_id,
            num_prompt_tokens=command.allocation.num_prompt_tokens,
            num_computed_tokens=computed,
            reserved_num_tokens=command.required_num_tokens,
            generation=self._generation_counter,
            pending_delta=(
                tuple(tuple(group) for group in blocks.get_block_ids())
                if blocks is not None
                else None
            ),
            allocated_since_build=True,
        )
        self._records[command.key] = record
        self._tombstones.discard(command.key)
        self._cold_records.pop(command.key, None)
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
        base = (
            record.pending_delta
            if record.pending_delta is not None
            else tuple(() for _ in range(self._num_groups))
        )
        new_blocks = tuple(tuple(group) for group in blocks.get_block_ids())
        record.pending_delta = tuple(
            base[group_index] + new_blocks[group_index]
            for group_index in range(self._num_groups)
        )
        record.allocated_since_build = True
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
            self._cold_records.pop(command.key, None)
            return DeferredFreeResult(accepted=True, key=command.key, deferred=False)
        record.pending_free = True
        record.preempted = False
        return DeferredFreeResult(accepted=True, key=command.key, deferred=True)

    def restore(self, command: OwnerCommand) -> AllocationResult | DeferredFreeResult:
        """Allocate the final destination for a cold computed prefix.

        The bytes are not ready here.  The owner bulk adapter must zero the
        exact destination and complete H2D before :meth:`mark_restore_ready`;
        only a later RESERVE may reactivate the record.
        """
        if command.kind is not OwnerCommandKind.RESTORE:
            return _reject_free(command, "restore() requires a RESTORE command")
        if command.key in self._records:
            return _reject_allocation(command, "restore destination already exists")
        cold = self._cold_records.get(command.key)
        if cold is None:
            return _reject_allocation(command, "no durable cold lease to restore")
        if command.required_num_tokens < cold.num_computed_tokens:
            return _reject_allocation(
                command, "restore destination refuses computed prefix tokens"
            )
        blocks = self._allocate(
            self._allocator_id(command.key),
            cold.num_prompt_tokens,
            0,
            command.required_num_tokens,
            command.required_num_tokens,
        )
        if blocks is None:
            return _reject_allocation(
                command, "insufficient KV cache for restore destination"
            )
        self._generation_counter += 1
        record = _Record(
            key=command.key,
            allocator_id=self._allocator_id(command.key),
            num_prompt_tokens=cold.num_prompt_tokens,
            num_computed_tokens=cold.num_computed_tokens,
            reserved_num_tokens=command.required_num_tokens,
            generation=self._generation_counter,
            pending_delta=tuple(tuple(group) for group in blocks.get_block_ids()),
            allocated_since_build=True,
            restored_from_host=True,
            restore_source_block_indices=cold.source_block_indices,
        )
        self._records[command.key] = record
        self._tombstones.discard(command.key)
        return self._accepted_allocation(command, record, blocks)

    def mark_restore_ready(self, key: OwnerLeaseKey, generation: int) -> bool:
        """Consume restore zeroing deltas after exact bulk H2D completion."""

        record = self._records.get(key)
        if (
            record is None
            or record.generation != generation
            or not record.restored_from_host
            or record.restore_ready
            or record.pending_free
        ):
            return False
        record.pending_delta = None
        record.allocated_since_build = False
        record.restore_ready = True
        return True

    def mark_reactivated(self, key: OwnerLeaseKey, generation: int) -> bool:
        """Close the restored-record state after owner-ledger activation."""

        record = self._records.get(key)
        if (
            record is None
            or record.generation != generation
            or not record.restored_from_host
            or not record.restore_ready
        ):
            return False
        record.restored_from_host = False
        record.restore_ready = False
        record.restore_source_block_indices = None
        self._cold_records.pop(key, None)
        return True

    def restore_source_block_indices(
        self, key: OwnerLeaseKey, generation: int
    ) -> tuple[tuple[int, ...], ...] | None:
        """Return the exact logical source mask for one restore generation."""

        record = self._records.get(key)
        if (
            record is None
            or record.generation != generation
            or not record.restored_from_host
        ):
            return None
        return record.restore_source_block_indices

    def is_restore_ready(self, key: OwnerLeaseKey) -> bool:
        record = self._records.get(key)
        return bool(
            record is not None and record.restored_from_host and record.restore_ready
        )

    def abort_restore(self, key: OwnerLeaseKey, generation: int) -> bool:
        """Free a failed exact restore destination while retaining cold KV."""

        record = self._records.get(key)
        if (
            record is None
            or record.generation != generation
            or not record.restored_from_host
        ):
            return False
        self._manager.free(
            _AllocationRequest(
                request_id=record.allocator_id,
                num_tokens=0,
                num_computed_tokens=0,
                num_prompt_tokens=record.num_prompt_tokens,
            )
        )
        del self._records[key]
        self._tombstones.add(key)
        return True

    def mark_computed(self, key: OwnerLeaseKey, num_tokens: int) -> bool:
        """Record monotonic computed progress for an active lease; False
        (changing nothing) on unknown/pending-free/malformed/regressive
        updates or on progress beyond the reserved chunk horizon.

        Once a step build handed the key into a step (a mark expectation is
        pending), the strict G3 contract applies: the only accepted value is
        the exact post-step target of that handoff, fenced by the record's
        physical allocation generation so a stale snapshot can never advance
        a recycled same-key record.  Accepting the target is the explicit
        success declaration and consumes the expectation.  Outside a
        handoff the legacy monotonic behavior is unchanged."""
        if isinstance(num_tokens, bool) or not isinstance(num_tokens, int):
            return False
        record = self._records.get(key)
        expectation = self._pending_marks.get(key)
        if record is not None and expectation is not None:
            if record.pending_free:
                return False
            if record.generation != expectation.allocation_generation:
                return False
            if num_tokens != expectation.post_step_num_tokens:
                return False
            record.num_computed_tokens = num_tokens
            del self._pending_marks[key]
            return True
        if record is None or record.pending_free:
            return False
        if num_tokens < 0 or num_tokens < record.num_computed_tokens:
            return False
        if num_tokens > record.reserved_num_tokens:
            return False
        record.num_computed_tokens = num_tokens
        return True

    def mark_computed_batch(
        self, metadata: RequestOwnedStepMetadata
    ) -> RequestOwnedStepMarkResult:
        """Atomically advance every entry of a handed-in G3 step metadata
        batch to its exact post-step target.

        All-or-nothing: every entry is validated (exact lease key, physical
        allocation generation, an armed pending mark expectation from the
        matching step build fenced by the same generation, the exact
        post-step target, and the pre-step computed base) and the batch
        must cover exactly every expectation armed by the step, before any
        record advances; a stale (non-current or already-marked step),
        duplicate, wrong-owner, foreign, missing-entry, or partial batch
        fails with no changes.  Success consumes every expectation and
        sets the records' computed progress to the exact post-step
        targets."""
        if not isinstance(metadata, RequestOwnedStepMetadata):
            return RequestOwnedStepMarkResult(
                accepted=False,
                step_seq=getattr(metadata, "step_seq", 0),
                error=f"metadata must be a RequestOwnedStepMetadata, got {metadata!r}.",
            )
        step_seq = metadata.step_seq
        if metadata.owner_rank != self._owner_rank:
            return RequestOwnedStepMarkResult(
                accepted=False,
                step_seq=step_seq,
                error=f"wrong-owner metadata: owner {metadata.owner_rank} != "
                f"store rank {self._owner_rank}.",
            )
        if self._last_built_step_seq is None or step_seq != self._last_built_step_seq:
            return RequestOwnedStepMarkResult(
                accepted=False,
                step_seq=step_seq,
                error=f"stale metadata: step_seq {step_seq} is not the last "
                f"built step {self._last_built_step_seq}.",
            )
        if self._marked_step_seq is not None and step_seq == self._marked_step_seq:
            return RequestOwnedStepMarkResult(
                accepted=False,
                step_seq=step_seq,
                error=f"duplicate mark: step {step_seq} was already marked.",
            )
        seen: set[OwnerLeaseKey] = set()
        for entry in metadata.entries:
            if not isinstance(entry, RequestOwnedStepEntry):
                return RequestOwnedStepMarkResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"batch entry must be a RequestOwnedStepEntry, "
                    f"got {entry!r}.",
                )
            if entry.key in seen:
                return RequestOwnedStepMarkResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"duplicate batch entry for {entry.key!r}.",
                )
            seen.add(entry.key)
            record = self._records.get(entry.key)
            if record is None:
                return RequestOwnedStepMarkResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"foreign batch entry: no active record for {entry.key!r}.",
                )
            if record.pending_free:
                return RequestOwnedStepMarkResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"lease {entry.key!r} is held pending free until "
                    f"the flush fence.",
                )
            if record.generation != entry.allocation_generation:
                return RequestOwnedStepMarkResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"stale batch entry for {entry.key!r}: record "
                    f"generation {record.generation} != snapshot "
                    f"{entry.allocation_generation}.",
                )
            expectation = self._pending_marks.get(entry.key)
            if expectation is None:
                return RequestOwnedStepMarkResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"no pending mark expectation for {entry.key!r}.",
                )
            if expectation.allocation_generation != record.generation:
                return RequestOwnedStepMarkResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"stale mark expectation for {entry.key!r}: "
                    f"expectation generation "
                    f"{expectation.allocation_generation} != record "
                    f"generation {record.generation}.",
                )
            if expectation.step_seq != step_seq:
                return RequestOwnedStepMarkResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"mark expectation for {entry.key!r} is from step "
                    f"{expectation.step_seq}, not {step_seq}.",
                )
            if expectation.post_step_num_tokens != entry.post_step_num_tokens:
                return RequestOwnedStepMarkResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"post-step target mismatch for {entry.key!r}: "
                    f"expected {expectation.post_step_num_tokens}, got "
                    f"{entry.post_step_num_tokens}.",
                )
            if record.num_computed_tokens != entry.pre_step_num_computed_tokens:
                return RequestOwnedStepMarkResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"pre-step computed mismatch for {entry.key!r}: "
                    f"record {record.num_computed_tokens} != snapshot "
                    f"{entry.pre_step_num_computed_tokens}.",
                )
        # Batch completeness: the batch must cover exactly the expectations
        # armed by this step, so a metadata object missing an entry can
        # never mark a subset and strand the remainder.
        expected_keys = {
            key
            for key, expectation in self._pending_marks.items()
            if expectation.step_seq == step_seq
        }
        batch_keys = {entry.key for entry in metadata.entries}
        if batch_keys != expected_keys:
            missing = expected_keys - batch_keys
            extra = batch_keys - expected_keys
            return RequestOwnedStepMarkResult(
                accepted=False,
                step_seq=step_seq,
                error=(
                    f"incomplete batch for step {step_seq}: missing "
                    f"{sorted(missing, key=lambda k: (k.request_id, k.owner_epoch))}, "
                    f"extra "
                    f"{sorted(extra, key=lambda k: (k.request_id, k.owner_epoch))}."
                ),
            )
        for entry in metadata.entries:
            record = self._records[entry.key]
            record.num_computed_tokens = entry.post_step_num_tokens
            del self._pending_marks[entry.key]
        self._marked_step_seq = step_seq
        return RequestOwnedStepMarkResult(accepted=True, step_seq=step_seq)

    def build_step_metadata(
        self,
        step_seq: int,
        tokens: Sequence[OwnerLeaseToken],
        request_token_counts: Mapping[str, int],
    ) -> RequestOwnedStepMetadataResult:
        """Freeze the immutable worker-local G3 execution metadata for one
        step from the exact own-rank lease tokens and the GLOBAL per-request
        scheduled counts of the step (``SchedulerOutput.num_scheduled_tokens``).

        Local scheduled keys are derived inside the store by matching each
        positive global count against the active records' ``request_id``
        (the full exact :class:`OwnerLeaseKey` is retained); multiple active
        records for one request id are ambiguous and rejected, and requests
        with no local active record are foreign to this rank and ignored.
        On a token-bearing step (any positive global count) the derived
        local positive-count key set must match the own-rank authorization
        tokens exactly: a local active request with a positive count but a
        missing token fails, and an own token without a positive local count
        is extra.  The post-step target is ``record.num_computed_tokens +
        count`` and must not exceed the lease horizon
        (``OwnerLeaseToken.runnable_num_tokens``, a cumulative horizon, not
        this step's target) nor the reserved chunk horizon; a target
        strictly below the lease horizon is legal (partial chunk).  A
        zero-global-token heartbeat (no positive counts anywhere) may carry
        newly published grants but always builds empty execution metadata
        and retains every pending delta for the later token-bearing step.
        Any build (token-bearing or heartbeat) also rejects while any live
        unconsumed pending mark exists -- even for a disjoint key -- so the
        build fence never advances over pending execution; a stale recycled
        expectation (record missing or generation changed) is ignored under
        the existing ABA policy.  On success the pending deltas of the batch
        keys are handed into the step and mark expectations are armed; on
        any rejection nothing changes, so pending deltas are never silently
        lost and the same step can be retried.  No scheduler wire object is
        mutated."""
        if isinstance(step_seq, bool) or not isinstance(step_seq, int) or step_seq <= 0:
            return RequestOwnedStepMetadataResult(
                accepted=False,
                step_seq=step_seq,
                error=f"step_seq must be a positive non-bool int, got {step_seq!r}.",
            )
        if (
            self._last_built_step_seq is not None
            and step_seq <= self._last_built_step_seq
        ):
            return RequestOwnedStepMetadataResult(
                accepted=False,
                step_seq=step_seq,
                error=f"stale step_seq {step_seq}: already built through "
                f"{self._last_built_step_seq}.",
            )
        if tokens is None or request_token_counts is None:
            return RequestOwnedStepMetadataResult(
                accepted=False,
                step_seq=step_seq,
                error="tokens and request_token_counts must be the exact "
                "own-rank lease tokens and the global per-request counts.",
            )
        if not isinstance(request_token_counts, Mapping):
            return RequestOwnedStepMetadataResult(
                accepted=False,
                step_seq=step_seq,
                error="request_token_counts must be a mapping keyed by "
                f"request_id, got {request_token_counts!r}.",
            )
        lease_tokens = tuple(tokens)
        seen: set[OwnerLeaseKey] = set()
        for token in lease_tokens:
            if not isinstance(token, OwnerLeaseToken):
                return RequestOwnedStepMetadataResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"batch entry must be an OwnerLeaseToken, got {token!r}.",
                )
            if token.owner_id != self._owner_rank:
                return RequestOwnedStepMetadataResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"wrong-owner lease token for {token.key!r}: "
                    f"owner {token.owner_id} != store rank {self._owner_rank}.",
                )
            if token.step_seq != step_seq:
                return RequestOwnedStepMetadataResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"lease token step_seq {token.step_seq} does not "
                    f"match the built step {step_seq} for {token.key!r}.",
                )
            count = token.runnable_num_tokens
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                return RequestOwnedStepMetadataResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"runnable_num_tokens must be a positive non-bool "
                    f"int, got {count!r} for {token.key!r}.",
                )
            if token.key in seen:
                return RequestOwnedStepMetadataResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"duplicate lease token for {token.key!r}.",
                )
            seen.add(token.key)

        positive_requests: list[str] = []
        for request_id, count in request_token_counts.items():
            if not isinstance(request_id, str):
                return RequestOwnedStepMetadataResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"scheduled count key must be a request_id string, "
                    f"got {request_id!r}.",
                )
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                return RequestOwnedStepMetadataResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"scheduled count must be a nonnegative non-bool "
                    f"int, got {count!r} for {request_id!r}.",
                )
            if count > 0:
                positive_requests.append(request_id)

        # Pre-build global live-pending fence: no new build (token-bearing
        # or heartbeat) may advance the build fence over an unconsumed
        # execution from an earlier handoff, even for a disjoint key --
        # otherwise step N for key A could stay pending while step N+1 for
        # key B advances the fence and A becomes permanently unmarkable.
        # A live pending mark is one whose record still exists with the
        # same allocation generation; stale recycled expectations (record
        # missing or generation changed) are ignored under the existing ABA
        # policy.  The executed batch must be marked before the next build,
        # so the atomic mark is always reachable.
        live_pending = [
            key
            for key, expectation in self._pending_marks.items()
            if (record := self._records.get(key)) is not None
            and expectation.allocation_generation == record.generation
        ]
        if live_pending:
            pending_key = sorted(
                live_pending, key=lambda k: (k.request_id, k.owner_epoch)
            )[0]
            return RequestOwnedStepMetadataResult(
                accepted=False,
                step_seq=step_seq,
                error=(
                    "unconsumed pending mark for "
                    f"{pending_key!r} from step "
                    f"{self._pending_marks[pending_key].step_seq}: the "
                    "executed batch must be marked before the next build."
                ),
            )

        if not positive_requests:
            # Zero-global-token heartbeat: grants may be newly published but
            # nothing executes this step.  Build empty execution metadata,
            # retain every pending delta for the later token-bearing step,
            # and still close this step's fence (which clears the same-step
            # allocation marker so the later authorization step is legal).
            metadata = RequestOwnedStepMetadata(
                step_seq=step_seq, owner_rank=self._owner_rank, entries=()
            )
            for record in self._records.values():
                record.allocated_since_build = False
            self._last_built_step_seq = step_seq
            self._active_metadata = metadata
            return RequestOwnedStepMetadataResult(
                accepted=True, step_seq=step_seq, metadata=metadata
            )

        # Derive the local scheduled keys from the positive global counts by
        # matching active records on request_id (full exact key retained).
        # Requests without a local active record are foreign on this rank.
        local_counts: dict[OwnerLeaseKey, int] = {}
        for request_id in positive_requests:
            matches = [key for key in self._records if key.request_id == request_id]
            if not matches:
                continue
            if len(matches) > 1:
                return RequestOwnedStepMetadataResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=(
                        "ambiguous same-request active records for "
                        f"{request_id!r}: "
                        f"{sorted(matches, key=lambda k: k.owner_epoch)!r}."
                    ),
                )
            local_counts[matches[0]] = request_token_counts[request_id]

        token_keys = {token.key for token in lease_tokens}
        count_keys = set(local_counts)
        missing_tokens = count_keys - token_keys
        if missing_tokens:
            missing_key = sorted(
                missing_tokens, key=lambda k: (k.request_id, k.owner_epoch)
            )[0]
            return RequestOwnedStepMetadataResult(
                accepted=False,
                step_seq=step_seq,
                error=f"missing authorization token for {missing_key!r}.",
            )
        extra_tokens = token_keys - count_keys
        if extra_tokens:
            extra_key = sorted(
                extra_tokens, key=lambda k: (k.request_id, k.owner_epoch)
            )[0]
            return RequestOwnedStepMetadataResult(
                accepted=False,
                step_seq=step_seq,
                error=f"extra lease token without a scheduled count for {extra_key!r}.",
            )

        for token in lease_tokens:
            key = token.key
            record = self._records.get(key)
            if record is None:
                return RequestOwnedStepMetadataResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"missing lease: no active record for {key!r}.",
                )
            if record.pending_free:
                return RequestOwnedStepMetadataResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"lease {key!r} is held pending free until the flush fence.",
                )
            if record.allocated_since_build:
                return RequestOwnedStepMetadataResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"same-step RESERVE/EXTEND plus execution "
                    f"authorization for {key!r}.",
                )
            expectation = self._pending_marks.get(key)
            if (
                expectation is not None
                and expectation.allocation_generation == record.generation
            ):
                return RequestOwnedStepMetadataResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"unconsumed pending mark for {key!r} from step "
                    f"{expectation.step_seq}.",
                )
            target = record.num_computed_tokens + local_counts[key]
            if target > token.runnable_num_tokens:
                return RequestOwnedStepMetadataResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"post-step target {target} exceeds the lease "
                    f"horizon {token.runnable_num_tokens} for {key!r}.",
                )
            if target > record.reserved_num_tokens:
                return RequestOwnedStepMetadataResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"out-of-horizon lease token for {key!r}: target "
                    f"{target} > reserved {record.reserved_num_tokens}.",
                )

        empty_delta = tuple(() for _ in range(self._num_groups))
        entries = tuple(
            RequestOwnedStepEntry(
                key=token.key,
                allocation_generation=self._records[token.key].generation,
                pre_step_num_computed_tokens=(
                    self._records[token.key].num_computed_tokens
                ),
                post_step_num_tokens=(
                    self._records[token.key].num_computed_tokens
                    + local_counts[token.key]
                ),
                tables=self._tables(self._records[token.key]),
                delta=(
                    self._records[token.key].pending_delta
                    if self._records[token.key].pending_delta is not None
                    else empty_delta
                ),
            )
            for token in lease_tokens
        )
        metadata = RequestOwnedStepMetadata(
            step_seq=step_seq,
            owner_rank=self._owner_rank,
            entries=entries,
        )

        # Commit atomically: hand the pending deltas of the batch keys into
        # the step, arm the mark expectations with the exact post-step
        # target, and advance the one-step fence.  The rejected paths above
        # changed nothing.
        for token in lease_tokens:
            record = self._records[token.key]
            record.pending_delta = None
            self._pending_marks[token.key] = _MarkExpectation(
                step_seq=step_seq,
                allocation_generation=record.generation,
                post_step_num_tokens=(
                    record.num_computed_tokens + local_counts[token.key]
                ),
            )
        for record in self._records.values():
            record.allocated_since_build = False
        self._last_built_step_seq = step_seq
        self._active_metadata = metadata
        return RequestOwnedStepMetadataResult(
            accepted=True, step_seq=step_seq, metadata=metadata
        )

    def checkpoint_step_build(self) -> RequestOwnedStepBuildCheckpoint:
        """Capture only the reversible fence facts touched by an empty build."""

        return RequestOwnedStepBuildCheckpoint(
            last_built_step_seq=self._last_built_step_seq,
            marked_step_seq=self._marked_step_seq,
            active_metadata=self._active_metadata,
            allocated_since_build=tuple(
                (key, record.generation, record.allocated_since_build)
                for key, record in self._records.items()
            ),
        )

    def rollback_empty_step_build(
        self, checkpoint: RequestOwnedStepBuildCheckpoint, step_seq: int
    ) -> None:
        """Undo one uncommitted heartbeat build, or verify it never happened."""

        if not isinstance(checkpoint, RequestOwnedStepBuildCheckpoint):
            raise TypeError("checkpoint must be a RequestOwnedStepBuildCheckpoint")
        if (
            self._last_built_step_seq == checkpoint.last_built_step_seq
            and self._marked_step_seq == checkpoint.marked_step_seq
            and self._active_metadata == checkpoint.active_metadata
        ):
            return
        metadata = self._active_metadata
        if (
            self._last_built_step_seq != step_seq
            or metadata is None
            or metadata.step_seq != step_seq
            or metadata.entries
            or self._marked_step_seq not in (checkpoint.marked_step_seq, step_seq)
            or any(mark.step_seq == step_seq for mark in self._pending_marks.values())
        ):
            raise RuntimeError(
                "cannot roll back a nonempty or nonmatching request-owned step build"
            )
        for key, generation, allocated in checkpoint.allocated_since_build:
            record = self._records.get(key)
            if record is not None and record.generation == generation:
                record.allocated_since_build = allocated
        self._last_built_step_seq = checkpoint.last_built_step_seq
        self._marked_step_seq = checkpoint.marked_step_seq
        self._active_metadata = checkpoint.active_metadata

    def get_block_table(self, key: OwnerLeaseKey) -> tuple[tuple[int, ...], ...] | None:
        """Private full per-group tables; readable until :meth:`flush`."""
        record = self._records.get(key)
        if record is None:
            return None
        return self._tables(record)

    def snapshot(self, key: OwnerLeaseKey) -> RequestOwnedKVSnapshot | None:
        """Observe an exact allocation, including a pending-free source.

        Retirement needs to bind D2H to the old generation before a later free
        fence runs.  The O-line adapter, not this snapshot, is responsible for
        withholding that fence until a durable store receipt exists.
        """

        record = self._records.get(key)
        if record is None:
            return None
        return RequestOwnedKVSnapshot(
            key=record.key,
            owner_rank=self._owner_rank,
            allocation_generation=record.generation,
            num_computed_tokens=record.num_computed_tokens,
            reserved_num_tokens=record.reserved_num_tokens,
            pending_free=record.pending_free,
            tables=self._tables(record),
        )

    def computed_prefix_snapshot(
        self, key: OwnerLeaseKey
    ) -> RequestOwnedKVSnapshot | None:
        """Observe only blocks containing immutable computed KV bytes.

        Reserved-but-uncomputed tail blocks must never enter a durable store
        receipt.  A partially computed final block is included; the O-line
        host key also carries its valid-token extent so a later extension
        cannot mistake the earlier partial image for the newer block.
        """

        snapshot = self.snapshot(key)
        if snapshot is None:
            return None
        tables = tuple(
            table[
                : (snapshot.num_computed_tokens + effective_block_size - 1)
                // effective_block_size
            ]
            for table, effective_block_size in zip(
                snapshot.tables, self.group_block_sizes
            )
        )
        return RequestOwnedKVSnapshot(
            key=snapshot.key,
            owner_rank=snapshot.owner_rank,
            allocation_generation=snapshot.allocation_generation,
            num_computed_tokens=snapshot.num_computed_tokens,
            reserved_num_tokens=snapshot.reserved_num_tokens,
            pending_free=snapshot.pending_free,
            tables=tables,
        )

    @property
    def group_block_sizes(self) -> tuple[int, ...]:
        """Worker-private logical tokens represented by each group block."""

        return tuple(
            request_owned_effective_tokens_per_block(spec) for spec in self._group_specs
        )

    def flush(self) -> tuple[OwnerLeaseKey, ...]:
        """Post-execute completion fence: free all deferred blocks now
        that the executing GPU step finished writing them."""
        freed: list[OwnerLeaseKey] = []
        for key, record in list(self._records.items()):
            if not record.pending_free:
                continue
            source_block_indices = None
            if record.preempted:
                source_block_indices = self._computed_prefix_block_indices(
                    self._tables(record),
                    record.num_computed_tokens,
                    self.group_block_sizes,
                )
            self._manager.free(
                _AllocationRequest(
                    request_id=record.allocator_id,
                    num_tokens=0,
                    num_computed_tokens=0,
                    num_prompt_tokens=record.num_prompt_tokens,
                )
            )
            if record.preempted:
                assert source_block_indices is not None
                self._tombstones.add(key)
                self._cold_records[key] = RequestOwnedColdSnapshot(
                    key=key,
                    owner_rank=self._owner_rank,
                    num_prompt_tokens=record.num_prompt_tokens,
                    num_computed_tokens=record.num_computed_tokens,
                    reserved_num_tokens=record.reserved_num_tokens,
                    source_block_indices=source_block_indices,
                )
            else:
                self._cold_records.pop(key, None)
            del self._records[key]
            freed.append(key)
        return tuple(freed)

    @staticmethod
    def _computed_prefix_block_indices(
        tables: tuple[tuple[int, ...], ...],
        num_computed_tokens: int,
        group_block_sizes: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        """Logical positions of non-null blocks in the computed prefix."""

        return tuple(
            tuple(
                index
                for index, block_id in enumerate(
                    table[: (num_computed_tokens + block_size - 1) // block_size]
                )
                if block_id != 0
            )
            for table, block_size in zip(tables, group_block_sizes)
        )

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
                effective_tokens_per_block=(
                    request_owned_effective_tokens_per_block(spec)
                ),
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
