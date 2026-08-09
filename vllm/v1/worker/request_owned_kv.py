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
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.ownership import (
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
class RequestOwnedStepMetadataResult:
    """Outcome of :meth:`RequestOwnedKVStore.build_step_metadata`."""

    accepted: bool
    step_seq: int
    metadata: RequestOwnedStepMetadata | None = None
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

    def build_step_metadata(
        self,
        step_seq: int,
        tokens: Sequence[OwnerLeaseToken],
        scheduled_counts: Mapping[OwnerLeaseKey, int],
    ) -> RequestOwnedStepMetadataResult:
        """Freeze the immutable worker-local G3 execution metadata for one
        step from the exact scheduled own-rank lease tokens and the exact
        per-key scheduled token counts.

        The batch is one-step fenced (``step_seq`` must be strictly newer
        than the last successful build).  ``tokens`` is the exact own-rank
        authorization set and ``scheduled_counts`` the exact per-key
        scheduled counts: for a token-bearing step the two key sets must
        match exactly, counts must be positive non-bool ints, and the
        post-step target ``record.num_computed_tokens + count`` must not
        exceed the lease horizon (``OwnerLeaseToken.runnable_num_tokens``,
        a cumulative horizon, not this step's target) nor the reserved
        chunk horizon; a target strictly below the lease horizon is legal
        (partial chunk).  A zero-count heartbeat (empty ``scheduled_counts``)
        may carry newly published grants but always builds empty execution
        metadata and retains every pending delta for the later token-bearing
        step.  A token-bearing build also rejects a key with an unconsumed
        pending mark from an earlier handoff unless the record was recycled
        (generation changed).  On success the pending deltas of the batch
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
        if tokens is None or scheduled_counts is None:
            return RequestOwnedStepMetadataResult(
                accepted=False,
                step_seq=step_seq,
                error="tokens and scheduled_counts must be the exact "
                "scheduled own-rank lease tokens and per-key counts.",
            )
        if not isinstance(scheduled_counts, Mapping):
            return RequestOwnedStepMetadataResult(
                accepted=False,
                step_seq=step_seq,
                error="scheduled_counts must be a mapping keyed by "
                f"OwnerLeaseKey, got {scheduled_counts!r}.",
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

        count_items = tuple(scheduled_counts.items())
        for key, count in count_items:
            if not isinstance(key, OwnerLeaseKey):
                return RequestOwnedStepMetadataResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"scheduled count key must be an OwnerLeaseKey, got {key!r}.",
                )
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                return RequestOwnedStepMetadataResult(
                    accepted=False,
                    step_seq=step_seq,
                    error=f"scheduled count must be a positive non-bool int, "
                    f"got {count!r} for {key!r}.",
                )

        token_keys = {token.key for token in lease_tokens}
        count_keys = {key for key, _ in count_items}
        if not count_keys:
            # Zero-token heartbeat: grants may be newly published but nothing
            # executes this step.  Build empty execution metadata, retain
            # every pending delta for the later token-bearing step, and still
            # close this step's fence (which clears the same-step allocation
            # marker so the later authorization step is legal).
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

        def _first_key(keys: set[OwnerLeaseKey]) -> OwnerLeaseKey:
            return sorted(keys, key=lambda k: (k.request_id, k.owner_epoch))[0]

        missing_counts = token_keys - count_keys
        if missing_counts:
            return RequestOwnedStepMetadataResult(
                accepted=False,
                step_seq=step_seq,
                error=f"missing scheduled count for {_first_key(missing_counts)!r}.",
            )
        extra_counts = count_keys - token_keys
        if extra_counts:
            return RequestOwnedStepMetadataResult(
                accepted=False,
                step_seq=step_seq,
                error=f"extra scheduled count for {_first_key(extra_counts)!r}.",
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
            target = record.num_computed_tokens + scheduled_counts[key]
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
                    + scheduled_counts[token.key]
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
                    record.num_computed_tokens + scheduled_counts[token.key]
                ),
            )
        for record in self._records.values():
            record.allocated_since_build = False
        self._last_built_step_seq = step_seq
        self._active_metadata = metadata
        return RequestOwnedStepMetadataResult(
            accepted=True, step_seq=step_seq, metadata=metadata
        )

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
