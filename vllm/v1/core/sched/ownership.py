# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G0 request-owner protocol for request-owned attention.

This module is intentionally dependency-neutral: it only uses the standard
library and never imports the engine, scheduler, workers, or torch.  It
defines the wire-level protocol types (commands, receipts, batches, lease
tokens) together with two pure-Python reference implementations:

* :class:`OwnerLeaseCoordinator`: the scheduler-side reference state machine
  implementing deterministic least-committed-work assignment with local
  per-owner charges, epoch fencing for reused request ids, abandon/retry of
  provisional assignments, horizon gating, and exact-once release.
* :class:`AttentionLeaseManager`: the worker-side reference lease manager
  that consumes commands, produces receipts, and emits exactly one
  :class:`OwnerReceiptBatch` per step even when no events occurred.

Request lifecycle: ``runnable -> preempting -> waiting -> reserving ->
resumed``.  A PREEMPT receipt releases the active runnable capacity while
preserving the sticky owner/key; a later RESERVE reacquires a lease on the
same owner.  RESTORE is the separate DMA/cold-residency intent and does not
reacquire capacity.

All token counts are exact and are exclusive 0-based upper bounds: a lease
covering ``N`` tokens makes exactly positions ``0 <= position < N``
runnable, and ``N`` itself is never a legal position.  ``required_num_tokens``
and ``runnable_num_tokens`` are such counts.  A zero-token RESERVE is a legal
empty lease: it is accepted, commits zero tokens, and publishes no lease
token; a nonzero request against zero physical/reference capacity is still
rejected (``0`` is a real value, never a stand-in for a missing count).
Commands to a given owner are delivered reliably and in order: the per-owner
``command_seq`` fence increases monotonically across all request keys and
the worker consumes that stream without reordering or buffering.
Nothing here is wired into the scheduler; consumers (G1+) integrate through
the public types below.

G2 physical capacity vocabulary: :class:`OwnerCacheGroupSnapshot` and
:class:`OwnerCachePoolSnapshot` describe the single unified per-rank KV block
pool of Ascend DSV4 in exact block-ID-free terms (pool-wide block counts plus
per-group table facts).  The legacy logical ``free_capacity`` /
``resident_pages`` fields on receipts and batches are reference-token facts of
the G1 protocol only; they must not be used as G2 physical admission.
"""

import enum
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Protocol types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OwnerLeaseKey:
    """Identity of a request lease, fenced by the request-id reuse epoch."""

    request_id: str
    owner_epoch: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str):
            raise TypeError(f"request_id must be a string, got {self.request_id!r}.")
        if (
            isinstance(self.owner_epoch, bool)
            or not isinstance(self.owner_epoch, int)
            or self.owner_epoch < 0
        ):
            raise TypeError(
                "owner_epoch must be a nonnegative non-bool int, got "
                f"{self.owner_epoch!r}."
            )


class OwnerCommandKind(enum.Enum):
    """Kinds of owner commands issued by the coordinator."""

    RESERVE = "RESERVE"
    EXTEND = "EXTEND"
    PREEMPT = "PREEMPT"
    RELEASE = "RELEASE"
    RESTORE = "RESTORE"


class OwnerAdmissionStatus(enum.Enum):
    """Immutable admission status of a request on its owner.

    Carried by :class:`OwnerAllocationDescriptor`: ``WAITING`` marks a
    request that is admitted (owner-assigned) but not yet runnable, and
    ``PREEMPTED`` marks a request whose active runnable capacity was
    released while the sticky owner/key are preserved.
    """

    WAITING = "WAITING"
    PREEMPTED = "PREEMPTED"


@dataclass(frozen=True)
class OwnerAssignmentObservation:
    """Observation used by the coordinator for least-work assignment.

    ``owner_id`` and ``observation_seq`` are required; all workload fields
    are optional so partial observations can be fed as they arrive.  Later
    observations with the same ``owner_id`` supersede earlier ones.
    """

    owner_id: int
    observation_seq: int
    #: Prefix tokens already computed for the candidate request on this owner.
    prefix_len: int | None = None
    #: External/base workload observation (scheduled/queued tokens on this
    #: owner).  Explicitly excludes coordinator-local projected charges,
    #: which ``assign()`` adds separately; producers must report the same
    #: base value with or without this coordinator's recent assignments.
    work: int | None = None
    #: Total token capacity of this owner.
    capacity: int | None = None
    #: Resident KV pages held on this owner (preferred on ties).
    residency: int | None = None
    #: Total request length of the candidate request.
    request_length: int | None = None
    #: Pending DMA work (pages/bytes) on this owner.
    pending_dma: int | None = None


@dataclass(frozen=True)
class OwnerAllocationDescriptor:
    """Immutable RESERVE allocation descriptor for one request lease.

    Minimal before-first-publication descriptor: identifies the request
    (:attr:`key`), its prompt length, how many prompt tokens are already
    computed on the owner, the total runnable token count being reserved,
    and the request's admission status.  Counts are exact nonnegative
    non-bool integers and ``num_computed_tokens <= num_tokens``.
    """

    key: OwnerLeaseKey
    num_prompt_tokens: int
    num_computed_tokens: int
    num_tokens: int
    status: OwnerAdmissionStatus

    def __post_init__(self) -> None:
        if not isinstance(self.key, OwnerLeaseKey):
            raise TypeError(f"key must be an OwnerLeaseKey, got {self.key!r}.")
        for name in (
            "num_prompt_tokens",
            "num_computed_tokens",
            "num_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(
                    f"{name} must be a nonnegative non-bool int, got {value!r}."
                )
        if self.num_computed_tokens > self.num_tokens:
            raise ValueError(
                "num_computed_tokens must not exceed num_tokens, got "
                f"{self.num_computed_tokens} > {self.num_tokens}."
            )
        if not isinstance(self.status, OwnerAdmissionStatus):
            raise TypeError(
                f"status must be an OwnerAdmissionStatus, got {self.status!r}."
            )


@dataclass(frozen=True)
class OwnerCommand:
    """A fenced command from the coordinator to a worker lease manager.

    Commands for a given owner are delivered reliably and in order: the
    per-owner ``command_seq`` increases monotonically across all request
    keys, so the worker consumes a single strictly increasing stream.
    """

    key: OwnerLeaseKey
    owner_id: int
    #: Monotonically increasing per-owner fence; stale/duplicate commands are
    #: rejected by the receiver.
    command_seq: int
    kind: OwnerCommandKind
    #: Exact number of tokens this command requires, as an exclusive
    #: 0-based upper bound (positions ``0 <= p < required_num_tokens``).
    #: Zero is a legal empty RESERVE; nonzero against zero capacity is
    #: rejected by the receiver.
    required_num_tokens: int
    #: Optional RESERVE allocation descriptor (key must match; counts are
    #: validated by the descriptor itself).  ``None`` when the caller does
    #: not participate in allocation publication.
    allocation: OwnerAllocationDescriptor | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.required_num_tokens, bool)
            or not isinstance(self.required_num_tokens, int)
            or self.required_num_tokens < 0
        ):
            raise TypeError(
                "required_num_tokens must be a nonnegative non-bool int, "
                f"got {self.required_num_tokens!r}."
            )
        if self.allocation is None:
            return
        if not isinstance(self.allocation, OwnerAllocationDescriptor):
            raise TypeError(
                "allocation must be an OwnerAllocationDescriptor or None, "
                f"got {self.allocation!r}."
            )
        if self.allocation.key != self.key:
            raise ValueError(
                f"allocation key {self.allocation.key!r} must match "
                f"command key {self.key!r}."
            )


@dataclass(frozen=True)
class OwnerReceipt:
    """Per-command/request receipt from a worker lease manager."""

    key: OwnerLeaseKey
    owner_id: int
    command_seq: int
    accepted: bool
    #: Exact number of tokens the worker can legally run (exclusive 0-based
    #: upper bound: positions ``0 <= p < runnable_num_tokens``).  ``0`` is
    #: the legal grant of an accepted empty lease.  ``None`` on rejection.
    runnable_num_tokens: int | None = None
    #: True when a RELEASE fully completed; set exactly once per lease.
    released: bool = False
    pending_dma: int | None = None
    #: G1 logical reference-token fact (unused tokens of the reference
    #: manager's token budget).  NOT G2 physical capacity: must not be used
    #: as physical admission; see :class:`OwnerCachePoolSnapshot`.
    free_capacity: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class OwnerLeaseToken:
    """Immutable token covering a lease published at a scheduler step.

    Published tokens are legally binding: a worker must not refuse tokens in
    the range ``0 <= p < runnable_num_tokens``.  An accepted empty lease
    (count 0) publishes no token.  ``step_seq`` increases monotonically per
    lease; the worker rejects regressed or duplicate steps.
    """

    key: OwnerLeaseKey
    owner_id: int
    step_seq: int
    command_seq: int
    #: Exact count of tokens the worker must honor (exclusive 0-based upper
    #: bound, <= worker-granted count); always positive on a published token.
    runnable_num_tokens: int


@dataclass(frozen=True)
class OwnerCacheGroupSnapshot:
    """Block-ID-free physical-capacity fact for one KV group of a per-rank
    block pool.

    Ascend DSV4 exposes one shared block pool (``KVCacheConfig.num_blocks``)
    and one shared block-ID space per rank, while each KV group keeps its own
    lookup table and its own effective tokens per block.  This snapshot is
    block-ID-free by contract: it carries only counts and table metadata,
    never local block/page/slot IDs, and it never models an independent
    per-group capacity pool.

    ``allocated_blocks`` counts the blocks the pool has handed out for this
    group (including prefix/shared references and NULL blocks); it is not
    required to sum across groups to the pool's used total.
    """

    group_index: int
    #: Stable identifier of the group's table/layout spec kind (opaque to
    #: the protocol); must be a nonempty string.
    spec_kind: str
    #: Effective number of tokens this group's table stores per block;
    #: heterogeneous across groups by design.
    effective_tokens_per_block: int
    #: Blocks of the unified pool allocated for this group (may exceed the
    #: resident set while blocks are paged in or shared).
    allocated_blocks: int
    #: Blocks of this group currently resident in the pool.
    resident_blocks: int

    def __post_init__(self) -> None:
        for name in (
            "group_index",
            "effective_tokens_per_block",
            "allocated_blocks",
            "resident_blocks",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(
                    f"{name} must be a nonnegative non-bool int, got {value!r}."
                )
        if not isinstance(self.spec_kind, str) or not self.spec_kind:
            raise TypeError(
                f"spec_kind must be a nonempty string, got {self.spec_kind!r}."
            )
        if self.resident_blocks > self.allocated_blocks:
            raise ValueError(
                "resident_blocks must not exceed allocated_blocks, got "
                f"{self.resident_blocks} > {self.allocated_blocks}."
            )


@dataclass(frozen=True)
class OwnerCachePoolSnapshot:
    """Block-ID-free physical capacity of one unified per-rank KV block pool.

    Ascend DSV4 has exactly one shared block pool and one shared block-ID
    space per rank; each KV group has its own table and effective
    tokens-per-block.  This snapshot carries pool-wide counts plus the
    per-group table facts and never exposes local block/page/slot IDs, so
    consumers cannot assume any per-family pool splitting or ID namespace.

    ``groups`` must be sorted by strictly increasing ``group_index`` (unique
    and sorted).  The sum of group ``allocated_blocks`` is NOT required to
    equal ``total_blocks - free_blocks``: prefix, NULL, and shared references
    may break that equality.
    """

    owner_rank: int
    total_blocks: int
    free_blocks: int
    #: Physical bytes per block, or ``None`` when unknown to the emitter.
    bytes_per_block: int | None = None
    #: Per-group table snapshots, sorted by ascending unique ``group_index``.
    groups: tuple[OwnerCacheGroupSnapshot, ...] = ()

    def __post_init__(self) -> None:
        for name in ("owner_rank", "total_blocks", "free_blocks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(
                    f"{name} must be a nonnegative non-bool int, got {value!r}."
                )
        if self.bytes_per_block is not None and (
            isinstance(self.bytes_per_block, bool)
            or not isinstance(self.bytes_per_block, int)
            or self.bytes_per_block < 0
        ):
            raise TypeError(
                "bytes_per_block must be a nonnegative non-bool int or None, "
                f"got {self.bytes_per_block!r}."
            )
        if self.free_blocks > self.total_blocks:
            raise ValueError(
                "free_blocks must not exceed total_blocks, got "
                f"{self.free_blocks} > {self.total_blocks}."
            )
        if not isinstance(self.groups, tuple):
            raise TypeError(f"groups must be a tuple, got {self.groups!r}.")
        for group in self.groups:
            if not isinstance(group, OwnerCacheGroupSnapshot):
                raise TypeError(
                    "groups must contain only OwnerCacheGroupSnapshot "
                    f"instances, got {group!r}."
                )
        indices = [group.group_index for group in self.groups]
        if any(a >= b for a, b in zip(indices, indices[1:])):
            raise ValueError(
                "group_index values must be unique and sorted ascending, "
                f"got {indices!r}."
            )


@dataclass(frozen=True)
class OwnerReceiptBatch:
    """One envelope a worker emits per step; empty events differ from a
    missing response, so an enabled worker always emits exactly one batch.

    The legacy logical ``free_capacity`` / ``resident_pages`` fields are
    reference-token facts of the G1 protocol only; they must not be used as
    G2 physical admission.  Physical capacity travels block-ID-free in the
    optional :attr:`cache_pool` snapshot.
    """

    owner_rank: int
    emitted_step_seq: int
    events: tuple[OwnerReceipt, ...]
    #: G1 logical reference-token fact; NOT G2 physical capacity (see
    #: :class:`OwnerCachePoolSnapshot`).
    free_capacity: int | None = None
    #: G1 logical resident-page bookkeeping fact; NOT G2 physical capacity
    #: (see :class:`OwnerCachePoolSnapshot`).
    resident_pages: int | None = None
    pending_dma: int | None = None
    #: Optional G2 physical-capacity snapshot of this owner's unified
    #: per-rank KV block pool.  ``None`` when the emitter does not
    #: participate in G2 capacity publication (the G1 reference manager
    #: always emits ``None``).
    cache_pool: OwnerCachePoolSnapshot | None = None

    def __post_init__(self) -> None:
        if self.cache_pool is not None and not isinstance(
            self.cache_pool, OwnerCachePoolSnapshot
        ):
            raise TypeError(
                "cache_pool must be an OwnerCachePoolSnapshot or None, "
                f"got {self.cache_pool!r}."
            )


class OwnershipError(Exception):
    """Base error for the request-owner protocol."""


class EpochFenceError(OwnershipError):
    """Assignment attempted with a request-id epoch below the current fence."""


class PublicationViolationError(OwnershipError):
    """A publish/receipt would exceed the granted token count or refuse
    published tokens."""


# ---------------------------------------------------------------------------
# Scheduler-side reference coordinator
# ---------------------------------------------------------------------------


@dataclass
class _LeaseState:
    """Per-lease bookkeeping kept by the reference coordinator."""

    owner_id: int
    required_num_tokens: int = 0
    last_required_num_tokens: int = 0
    runnable_num_tokens: int | None = None
    preempted: bool = False
    restored: bool = False
    superseded: bool = False
    release_pending: bool = False
    released: bool = False
    command_seq: int = 0
    command_kind: OwnerCommandKind | None = None
    #: command_seq of the last receipt applied (0 = none yet).  Publication
    #: waits until the current command is receipted.
    receipt_seq: int = 0
    #: The outstanding RELEASE command issued by ``finish``, so repeated
    #: finish while ``release_pending`` returns the identical command.
    release_command: OwnerCommand | None = None


class OwnerLeaseCoordinator:
    """Pure-Python reference coordinator for request-owned attention.

    Assignment picks the owner with the least committed work, where the
    score combines the latest observation (``work`` + ``pending_dma``) with
    coordinator-local projected charges made exactly once per assignment
    (refunded exactly once on release or abandon).  Ties resolve by higher
    residency, then by stable numeric global rank (``owner_id``).

    Other invariants: idempotent same-request assignment; an epoch fence for
    reused request ids that keeps the old-epoch lease as a tombstone until
    its own RELEASE receipt (commitments are never silently freed); reserve/
    extend required-vs-granted token-count gating; accepted receipts fail
    closed when they exceed the command's required count or regress below
    the published watermark; publication only at or below the granted
    count (a zero-token empty lease publishes no token); preempt preserves
    the owner and releases active capacity;
    finish/abort leaves ``release_pending`` until the matching RELEASE
    receipt frees the commitment (and its charge) exactly once.  Owner
    commands are delivered reliably and in order (monotonic ``command_seq``
    across keys); nothing here buffers or reorders them.
    """

    def __init__(self) -> None:
        self._leases: dict[OwnerLeaseKey, _LeaseState] = {}
        self._epoch_fence: dict[str, int] = {}
        self._observations: dict[int, OwnerAssignmentObservation] = {}
        self._owner_command_seq: dict[int, int] = {}
        self._processed: dict[OwnerLeaseKey, int] = {}
        self._publish_watermark: dict[OwnerLeaseKey, int] = {}
        #: Coordinator-local projected work per owner (charge bookkeeping).
        self._charges: dict[int, int] = {}
        self._key_charge: dict[OwnerLeaseKey, int] = {}
        self._release_count = 0

    # -- observations and assignment ----------------------------------------

    def observe(self, observation: OwnerAssignmentObservation) -> None:
        """Feed the latest observation for ``observation.owner_id``."""
        current = self._observations.get(observation.owner_id)
        if current is None or observation.observation_seq >= current.observation_seq:
            self._observations[observation.owner_id] = observation

    def assign(
        self,
        key: OwnerLeaseKey,
        required_num_tokens: int | None = None,
        projected_work: int | None = None,
    ) -> int:
        """Assign ``key`` to the least-committed-work owner.

        Returns the existing owner unchanged when the key is already
        assigned (idempotent; the projected charge is applied exactly once).
        Raises :class:`EpochFenceError` when the request id was already
        admitted at a higher epoch; the lower-epoch lease (if any) is kept
        as a tombstone and never silently freed.
        """
        fence = self._epoch_fence.get(key.request_id)
        if fence is not None and key.owner_epoch < fence:
            raise EpochFenceError(
                f"request {key.request_id} already fenced at epoch {fence}; "
                f"cannot assign stale epoch {key.owner_epoch}"
            )
        if fence is None:
            self._epoch_fence[key.request_id] = key.owner_epoch
        elif key.owner_epoch > fence:
            # Request-id reuse: tombstone the old-epoch lease(s).  Their
            # commitments stay charged until their own RELEASE receipts.
            for old_key in [
                k
                for k in self._leases
                if k.request_id == key.request_id and k.owner_epoch < key.owner_epoch
            ]:
                self._leases[old_key].superseded = True
            self._epoch_fence[key.request_id] = key.owner_epoch

        existing = self._leases.get(key)
        if existing is not None:
            return existing.owner_id

        candidates = [o for o in self._observations.values() if o.owner_id >= 0]
        if not candidates:
            raise OwnershipError(
                f"cannot assign {key}: no owner observations available"
            )

        def score(obs: OwnerAssignmentObservation) -> tuple[int, int, int]:
            committed = (
                (obs.work or 0)
                + (obs.pending_dma or 0)
                + self._charges.get(obs.owner_id, 0)
            )
            # Prefer higher residency, then lower global rank (owner_id).
            return (committed, -(obs.residency or 0), obs.owner_id)

        owner = min(candidates, key=score).owner_id
        projected = (
            projected_work
            if projected_work is not None
            else (required_num_tokens if required_num_tokens is not None else 0)
        )
        self._charges[owner] = self._charges.get(owner, 0) + projected
        self._key_charge[key] = projected
        self._leases[key] = _LeaseState(
            owner_id=owner,
            required_num_tokens=(
                required_num_tokens if required_num_tokens is not None else 0
            ),
        )
        return owner

    # -- command issuance ----------------------------------------------------

    def _issue(
        self,
        key: OwnerLeaseKey,
        kind: OwnerCommandKind,
        required_num_tokens: int,
    ) -> OwnerCommand:
        lease = self._leases.get(key)
        if lease is None:
            raise OwnershipError(f"cannot issue {kind.name} for unassigned {key}")
        self._owner_command_seq[lease.owner_id] = (
            self._owner_command_seq.get(lease.owner_id, 0) + 1
        )
        lease.command_seq = self._owner_command_seq[lease.owner_id]
        lease.command_kind = kind
        lease.last_required_num_tokens = required_num_tokens
        lease.required_num_tokens = max(lease.required_num_tokens, required_num_tokens)
        return OwnerCommand(
            key=key,
            owner_id=lease.owner_id,
            command_seq=lease.command_seq,
            kind=kind,
            required_num_tokens=required_num_tokens,
        )

    def reserve(self, key: OwnerLeaseKey, required_num_tokens: int) -> OwnerCommand:
        lease = self._leases[key]
        if lease.released or lease.release_pending:
            raise OwnershipError(f"cannot RESERVE released {key}")
        if lease.superseded:
            raise OwnershipError(f"cannot RESERVE superseded {key}")
        if lease.command_seq != 0:
            raise OwnershipError(f"cannot RESERVE already-reserved {key}")
        return self._issue(key, OwnerCommandKind.RESERVE, required_num_tokens)

    def resume(self, key: OwnerLeaseKey, required_num_tokens: int) -> OwnerCommand:
        """Reacquire a lease on the same (sticky) owner after a preempt.

        Issues a RESERVE-kind command; the worker grants a fresh runnable
        token count instead of silently retaining the preempt-released
        capacity.
        """
        lease = self._leases[key]
        if lease.released or lease.release_pending:
            raise OwnershipError(f"cannot RESUME released {key}")
        if lease.superseded:
            raise OwnershipError(f"cannot RESUME superseded {key}")
        if not lease.preempted:
            raise OwnershipError(f"cannot RESUME non-preempted {key}")
        return self._issue(key, OwnerCommandKind.RESERVE, required_num_tokens)

    def extend(self, key: OwnerLeaseKey, required_num_tokens: int) -> OwnerCommand:
        lease = self._leases[key]
        if lease.released or lease.release_pending:
            raise OwnershipError(f"cannot EXTEND released {key}")
        if lease.superseded:
            raise OwnershipError(f"cannot EXTEND superseded {key}")
        if lease.command_seq == 0:
            raise OwnershipError(f"cannot EXTEND un-reserved {key}")
        return self._issue(key, OwnerCommandKind.EXTEND, required_num_tokens)

    def preempt(
        self,
        key: OwnerLeaseKey,
        preempt_num_tokens: int | None = None,
    ) -> OwnerCommand:
        lease = self._leases[key]
        if lease.released or lease.release_pending:
            raise OwnershipError(f"cannot PREEMPT released {key}")
        if lease.superseded:
            raise OwnershipError(f"cannot PREEMPT superseded {key}")
        num_tokens = (
            preempt_num_tokens
            if preempt_num_tokens is not None
            else (
                lease.runnable_num_tokens
                if lease.runnable_num_tokens is not None
                else lease.required_num_tokens
            )
        )
        return self._issue(key, OwnerCommandKind.PREEMPT, num_tokens)

    def restore(self, key: OwnerLeaseKey, required_num_tokens: int) -> OwnerCommand:
        """Request the separate DMA/cold-residency restore intent.

        Does not reacquire runnable capacity; use :meth:`resume` for that.
        """
        lease = self._leases[key]
        if lease.released or lease.release_pending:
            raise OwnershipError(f"cannot RESTORE released {key}")
        if lease.superseded:
            raise OwnershipError(f"cannot RESTORE superseded {key}")
        if not lease.preempted:
            raise OwnershipError(f"cannot RESTORE non-preempted {key}")
        return self._issue(key, OwnerCommandKind.RESTORE, required_num_tokens)

    def finish(self, key: OwnerLeaseKey) -> OwnerCommand:
        """Finish/abort the lease: leave ``release_pending`` until the
        matching RELEASE receipt frees the commitment exactly once.

        Idempotent while ``release_pending``: a repeated call returns the
        identical outstanding RELEASE command without advancing
        ``command_seq``, so a double finish before the first receipt
        converges and the refund happens exactly once.  Raises
        :class:`OwnershipError` when no RESERVE has been accepted yet
        (``runnable_num_tokens`` is None); the caller must :meth:`abandon`
        before reservation or after a refused RESERVE.
        """
        lease = self._leases[key]
        if lease.released:
            raise OwnershipError(f"cannot finish already-released {key}")
        if lease.runnable_num_tokens is None:
            raise OwnershipError(
                f"cannot finish {key}: no accepted RESERVE (abandon instead)"
            )
        if lease.release_pending:
            if lease.release_command is None:
                raise OwnershipError(
                    f"cannot finish {key}: missing outstanding RELEASE"
                )
            return lease.release_command
        lease.release_pending = True
        num_tokens = (
            lease.runnable_num_tokens
            if lease.runnable_num_tokens is not None
            else lease.required_num_tokens
        )
        command = self._issue(key, OwnerCommandKind.RELEASE, num_tokens)
        lease.release_command = command
        return command

    def abandon(self, key: OwnerLeaseKey) -> bool:
        """Abandon a provisional (not-yet-admitted) assignment.

        Refunds the projected charge and purges the lease so the scheduler
        can retry another owner.  Raises :class:`OwnershipError` once the
        lease is admitted (accepted receipt or publication): the owner is
        sticky after admission.  Returns False when no such lease exists.
        """
        lease = self._leases.get(key)
        if lease is None:
            return False
        if lease.runnable_num_tokens is not None:
            raise OwnershipError(f"cannot abandon admitted lease {key}")
        if lease.release_pending or lease.released or lease.preempted:
            raise OwnershipError(f"cannot abandon non-provisional lease {key}")
        charge = self._key_charge.pop(key, 0)
        if charge:
            self._charges[lease.owner_id] -= charge
        del self._leases[key]
        self._publish_watermark.pop(key, None)
        self._processed.pop(key, None)
        return True

    # -- receipts ------------------------------------------------------------

    def apply_receipt(self, receipt: OwnerReceipt) -> bool:
        """Apply a worker receipt; returns True when state advanced.

        Receipts are fenced by full key (request id + epoch): stale,
        duplicate, wrong-owner, and unknown-key receipts are ignored and
        never advance or free state.  An accepted receipt that exceeds the
        command's required count or regresses below the published
        watermark fails closed with :class:`PublicationViolationError`
        instead of silently advancing.
        """
        lease = self._leases.get(receipt.key)
        if lease is None:
            return False
        if receipt.owner_id != lease.owner_id:
            return False
        if receipt.command_seq != lease.command_seq:
            # Stale: a newer command superseded this sequence.
            return False
        if self._processed.get(receipt.key, 0) >= receipt.command_seq:
            # Duplicate: this sequence was already applied.
            return False
        if not receipt.accepted:
            return False
        if lease.superseded and lease.command_kind is not OwnerCommandKind.RELEASE:
            # Tombstoned lease: only its own RELEASE receipt may apply.
            return False

        if receipt.runnable_num_tokens is not None:
            if receipt.runnable_num_tokens > lease.last_required_num_tokens:
                raise PublicationViolationError(
                    f"receipt for {receipt.key} grants {receipt.runnable_num_tokens} "
                    f"beyond required count {lease.last_required_num_tokens}"
                )
            watermark = self._publish_watermark.get(receipt.key, 0)
            if receipt.runnable_num_tokens < watermark:
                raise PublicationViolationError(
                    f"receipt for {receipt.key} regresses below published "
                    f"count {watermark}"
                )
            lease.runnable_num_tokens = receipt.runnable_num_tokens
        self._processed[receipt.key] = receipt.command_seq
        lease.receipt_seq = receipt.command_seq

        if lease.command_kind is OwnerCommandKind.PREEMPT:
            # PREEMPT preserves the owner; the active runnable capacity was
            # released by the worker, so the receipt count is the honored
            # (published) count only.
            lease.preempted = True
        elif lease.command_kind is OwnerCommandKind.RESTORE:
            lease.restored = True
        elif lease.command_kind is OwnerCommandKind.RESERVE:
            lease.preempted = False
        if (
            receipt.released
            and lease.command_kind is OwnerCommandKind.RELEASE
            and lease.release_pending
            and not lease.released
        ):
            lease.released = True
            lease.release_pending = False
            self._release_count += 1
            charge = self._key_charge.pop(receipt.key, 0)
            if charge:
                self._charges[lease.owner_id] -= charge
        return True

    # -- publication ----------------------------------------------------------

    def publish(self, step_seq: int) -> list[OwnerLeaseToken]:
        """Publish lease tokens for the step, only at/below granted counts.

        A lease with zero runnable tokens (an accepted empty RESERVE)
        publishes no token.  Raises :class:`PublicationViolationError` when
        a publish would exceed the worker-granted count or regress an
        already-published count.
        """
        tokens: list[OwnerLeaseToken] = []
        for key, lease in sorted(
            self._leases.items(),
            key=lambda kv: (kv[0].request_id, kv[0].owner_epoch),
        ):
            if (
                lease.released
                or lease.release_pending
                or lease.preempted
                or lease.superseded
            ):
                continue
            if lease.command_seq != lease.receipt_seq:
                # The current command is still in flight: nothing may
                # publish until its receipt is applied.
                continue
            if lease.command_kind not in (
                OwnerCommandKind.RESERVE,
                OwnerCommandKind.EXTEND,
            ):
                # Only accepted RESERVE/EXTEND grants are publishable;
                # PREEMPT, RELEASE, and RESTORE receipts never advance
                # publication.
                continue
            if lease.runnable_num_tokens is None:
                # Not yet granted: nothing legal to publish.
                continue
            horizon = min(lease.runnable_num_tokens, lease.required_num_tokens)
            watermark = self._publish_watermark.get(key, 0)
            if horizon < watermark:
                raise PublicationViolationError(
                    f"publish count {horizon} for {key} regresses "
                    f"published count {watermark}"
                )
            # Zero runnable tokens: the empty lease publishes no token, and
            # the watermark never advances below an already-published count.
            if horizon <= watermark:
                continue
            self._publish_watermark[key] = horizon
            tokens.append(
                OwnerLeaseToken(
                    key=key,
                    owner_id=lease.owner_id,
                    step_seq=step_seq,
                    command_seq=lease.command_seq,
                    runnable_num_tokens=horizon,
                )
            )
        return tokens

    # -- introspection --------------------------------------------------------

    def owner_of(self, key: OwnerLeaseKey) -> int | None:
        lease = self._leases.get(key)
        return lease.owner_id if lease is not None else None

    def required_num_tokens_of(self, key: OwnerLeaseKey) -> int:
        return self._leases[key].required_num_tokens

    def runnable_num_tokens_of(self, key: OwnerLeaseKey) -> int | None:
        return self._leases[key].runnable_num_tokens

    def published_num_tokens(self, key: OwnerLeaseKey) -> int:
        return self._publish_watermark.get(key, 0)

    def is_preempted(self, key: OwnerLeaseKey) -> bool:
        return self._leases[key].preempted

    def is_restored(self, key: OwnerLeaseKey) -> bool:
        return self._leases[key].restored

    def is_superseded(self, key: OwnerLeaseKey) -> bool:
        return self._leases[key].superseded

    def is_release_pending(self, key: OwnerLeaseKey) -> bool:
        return self._leases[key].release_pending

    def is_released(self, key: OwnerLeaseKey) -> bool:
        return self._leases[key].released

    def release_count(self) -> int:
        """Number of leases whose commitments were released (exactly once)."""
        return self._release_count


# ---------------------------------------------------------------------------
# Worker-side reference lease manager
# ---------------------------------------------------------------------------


@dataclass
class _WorkerLease:
    """Per-lease bookkeeping kept by the reference worker manager.

    ``runnable_num_tokens`` (exclusive 0-based bound) and ``published_num_tokens``
    are the logical token counts the worker is legally bound to honor;
    ``committed_tokens`` is the separate active physical commitment that
    occupies capacity.  PREEMPT zeroes the commitment while retaining the
    logical fence, so a post-step preempt is not a count regression.
    """

    runnable_num_tokens: int
    published_num_tokens: int = 0
    committed_tokens: int = 0
    #: command_seq of the last accepted grant (RESERVE/EXTEND); published
    #: tokens must carry exactly this sequence.
    grant_command_seq: int = 0
    #: Last accepted publication step; strictly increasing per lease.
    last_step_seq: int = 0
    preempted: bool = False
    restoring: bool = False
    superseded: bool = False
    released: bool = False
    resident_pages: int = 0
    pending_dma: int = 0


class AttentionLeaseManager:
    """Pure-Python reference worker-side lease manager (no GPU).

    ``capacity`` models the worker's total request-owned attention budget in
    tokens; each active lease commits ``min(required_num_tokens, capacity)``
    tokens.  ``grant_ceiling`` optionally caps how many tokens a single
    command may grant, which models chunk-count gating.  A zero-token
    RESERVE is a legal empty lease: accepted, commits zero tokens, and
    publishes no token; a nonzero request against zero physical/reference
    capacity is still rejected.

    The manager enforces the protocol invariants locally: monotonically
    fenced command sequences, a per-request-id epoch fence (a stale
    lower-epoch RESERVE cannot recreate state), refusal of published tokens
    is illegal, PREEMPT releases the active runnable capacity while the
    owner/key stay sticky, RESUME (a RESERVE on a preempted lease) reacquires
    capacity on the same owner, RESTORE only signals the DMA/cold-residency
    intent, and each lease's commitment is released exactly once.
    ``record_published`` enforces the published-token invariant (active
    lease, matching grant command sequence, monotonic step sequence,
    count at or below the grant), and RELEASE clears the DMA/restore/
    residency facts so freed leases never report stale physical state.
    """

    def __init__(
        self,
        owner_rank: int,
        capacity: int,
        grant_ceiling: int | None = None,
    ) -> None:
        self.owner_rank = owner_rank
        self.capacity = capacity
        self.grant_ceiling = grant_ceiling
        self._leases: dict[OwnerLeaseKey, _WorkerLease] = {}
        self._command_fence: dict[int, int] = {}
        self._epoch_fence: dict[str, int] = {}
        self._outbox: list[OwnerReceipt] = []
        self._released_count = 0

    # -- command handling ------------------------------------------------------

    def apply(self, command: OwnerCommand) -> OwnerReceipt:
        """Consume one owner command and produce its receipt."""
        if command.owner_id != self.owner_rank:
            return self._receipt(command, accepted=False, error="wrong owner rank")
        fence = self._command_fence.get(command.owner_id, 0)
        if command.command_seq <= fence:
            return self._receipt(
                command,
                accepted=False,
                error="stale or duplicate command sequence",
            )
        self._command_fence[command.owner_id] = command.command_seq

        # Per-request epoch fence: a stale lower-epoch command cannot
        # recreate state for a superseded request id.  The one exception is
        # a matching lower-epoch RELEASE for an existing superseded
        # tombstone: it must be honored so the old commitment is freed
        # exactly once instead of leaking forever.
        epoch_fence = self._epoch_fence.get(command.key.request_id, -1)
        if command.key.owner_epoch < epoch_fence:
            tombstone = self._leases.get(command.key)
            if not (
                command.kind is OwnerCommandKind.RELEASE
                and tombstone is not None
                and tombstone.superseded
            ):
                return self._receipt(
                    command, accepted=False, error="stale request epoch"
                )
        elif command.kind is not OwnerCommandKind.RELEASE:
            if command.key.owner_epoch > epoch_fence:
                for old_key in [
                    k
                    for k in self._leases
                    if k.request_id == command.key.request_id
                    and k.owner_epoch < command.key.owner_epoch
                ]:
                    # Tombstone, do not silently free: capacity stays
                    # committed until the old lease's own RELEASE receipt.
                    self._leases[old_key].superseded = True
                self._epoch_fence[command.key.request_id] = command.key.owner_epoch

        lease = self._leases.get(command.key)
        if command.kind is OwnerCommandKind.RESERVE:
            return self._on_reserve(command, lease)
        if command.kind is OwnerCommandKind.EXTEND:
            return self._on_extend(command, lease)
        if command.kind is OwnerCommandKind.PREEMPT:
            return self._on_preempt(command, lease)
        if command.kind is OwnerCommandKind.RESTORE:
            return self._on_restore(command, lease)
        if command.kind is OwnerCommandKind.RELEASE:
            return self._on_release(command, lease)
        return self._receipt(command, accepted=False, error="unknown command kind")

    def _grant(self, command: OwnerCommand, lease: _WorkerLease) -> int:
        """Grant an exclusive token count for ``command`` on ``lease``."""
        cap = max(0, self.capacity - self._committed_others(lease))
        if self.grant_ceiling is not None:
            cap = min(cap, lease.runnable_num_tokens + self.grant_ceiling)
        return min(command.required_num_tokens, cap)

    def _committed_others(self, lease: _WorkerLease) -> int:
        return sum(
            other.committed_tokens
            for other in self._leases.values()
            if other is not lease and not other.released
        )

    def _on_reserve(
        self, command: OwnerCommand, lease: _WorkerLease | None
    ) -> OwnerReceipt:
        if lease is not None:
            if lease.released:
                return self._receipt(
                    command, accepted=False, error="lease already released"
                )
            if lease.superseded:
                return self._receipt(command, accepted=False, error="lease superseded")
            if not lease.preempted:
                return self._receipt(command, accepted=False, error="duplicate reserve")
            # Resume: reacquire capacity on the same owner; the old runnable
            # capacity was released by the PREEMPT receipt.
            if command.required_num_tokens < lease.published_num_tokens:
                return self._receipt(
                    command,
                    accepted=False,
                    error="refuses published tokens",
                )
            granted = self._grant(command, lease)
            if granted < lease.published_num_tokens or (
                granted <= 0 and command.required_num_tokens > 0
            ):
                # Reacquiring below the honored (published) count would
                # refuse published tokens; a nonzero reacquisition that
                # grants nothing is a refusal.  A zero-token resume is a
                # legal empty lease and commits nothing.
                return self._receipt(
                    command,
                    accepted=False,
                    error="insufficient capacity to resume",
                )
            lease.runnable_num_tokens = granted
            lease.committed_tokens = granted
            lease.grant_command_seq = command.command_seq
            lease.preempted = False
            lease.restoring = False
            lease.pending_dma = 0
            return self._receipt(command, accepted=True, runnable_num_tokens=granted)
        committed = sum(
            active.committed_tokens
            for active in self._leases.values()
            if not active.released
        )
        granted = min(command.required_num_tokens, max(0, self.capacity - committed))
        if self.grant_ceiling is not None:
            granted = min(granted, self.grant_ceiling)
        if granted <= 0 and command.required_num_tokens > 0:
            # Real pre-publication refusal: a nonzero request against zero
            # physical/reference capacity must not be reported as an
            # accepted empty lease.  A zero-token request is the one legal
            # empty lease and commits nothing.
            return self._receipt(
                command,
                accepted=False,
                error="insufficient capacity to reserve",
            )
        self._leases[command.key] = _WorkerLease(
            runnable_num_tokens=granted,
            committed_tokens=granted,
            grant_command_seq=command.command_seq,
        )
        return self._receipt(command, accepted=True, runnable_num_tokens=granted)

    def _on_extend(
        self, command: OwnerCommand, lease: _WorkerLease | None
    ) -> OwnerReceipt:
        if lease is None or lease.released:
            return self._receipt(
                command, accepted=False, error="no active lease to extend"
            )
        if lease.superseded:
            return self._receipt(command, accepted=False, error="lease superseded")
        if lease.preempted:
            return self._receipt(command, accepted=False, error="lease is preempted")
        if command.required_num_tokens < lease.published_num_tokens:
            return self._receipt(
                command,
                accepted=False,
                error="refuses published tokens",
            )
        granted = self._grant(command, lease)
        if granted < lease.runnable_num_tokens:
            reason = (
                "insufficient capacity to extend"
                if command.required_num_tokens >= lease.runnable_num_tokens
                else "extend would shrink the runnable token count"
            )
            return self._receipt(command, accepted=False, error=reason)
        if granted <= 0 and command.required_num_tokens > 0:
            # A nonzero extension that grants nothing (e.g. extending an
            # empty lease with zero capacity available) is a refusal, not
            # an accepted no-op.
            return self._receipt(
                command,
                accepted=False,
                error="insufficient capacity to extend",
            )
        lease.committed_tokens += granted - lease.runnable_num_tokens
        lease.runnable_num_tokens = granted
        lease.grant_command_seq = command.command_seq
        return self._receipt(command, accepted=True, runnable_num_tokens=granted)

    def _on_preempt(
        self, command: OwnerCommand, lease: _WorkerLease | None
    ) -> OwnerReceipt:
        if lease is None or lease.released:
            return self._receipt(
                command, accepted=False, error="no active lease to preempt"
            )
        if lease.superseded:
            return self._receipt(command, accepted=False, error="lease superseded")
        if command.required_num_tokens < lease.published_num_tokens:
            return self._receipt(
                command,
                accepted=False,
                error="refuses published tokens",
            )
        lease.preempted = True
        # Release the active physical commitment while retaining the
        # logical run/published fence: published tokens stay honored and
        # the preempt is not a count regression.
        lease.committed_tokens = 0
        return self._receipt(
            command, accepted=True, runnable_num_tokens=lease.runnable_num_tokens
        )

    def _on_restore(
        self, command: OwnerCommand, lease: _WorkerLease | None
    ) -> OwnerReceipt:
        if lease is None or lease.released:
            return self._receipt(
                command, accepted=False, error="no active lease to restore"
            )
        if lease.superseded:
            return self._receipt(command, accepted=False, error="lease superseded")
        if not lease.preempted:
            return self._receipt(
                command, accepted=False, error="lease is not preempted"
            )
        if command.required_num_tokens < lease.published_num_tokens:
            return self._receipt(
                command,
                accepted=False,
                error="refuses published tokens",
            )
        # Separate DMA/cold-residency intent: marks the restore in flight
        # without reacquiring runnable capacity.
        lease.restoring = True
        lease.pending_dma = 1
        return self._receipt(
            command,
            accepted=True,
            runnable_num_tokens=lease.runnable_num_tokens,
            pending_dma=1,
        )

    def _on_release(
        self, command: OwnerCommand, lease: _WorkerLease | None
    ) -> OwnerReceipt:
        if lease is None:
            return self._receipt(command, accepted=False, error="no lease to release")
        if lease.released:
            return self._receipt(
                command, accepted=False, error="lease already released"
            )
        if command.required_num_tokens < lease.published_num_tokens:
            return self._receipt(
                command,
                accepted=False,
                error="refuses published tokens",
            )
        lease.released = True
        # Release clears the DMA/restore/residency facts (and the physical
        # commitment) so a freed lease never reports stale physical state.
        lease.committed_tokens = 0
        lease.pending_dma = 0
        lease.restoring = False
        lease.resident_pages = 0
        self._released_count += 1
        return self._receipt(
            command,
            accepted=True,
            runnable_num_tokens=lease.runnable_num_tokens,
            released=True,
        )

    def _receipt(
        self,
        command: OwnerCommand,
        *,
        accepted: bool,
        runnable_num_tokens: int | None = None,
        released: bool = False,
        pending_dma: int | None = None,
        error: str | None = None,
    ) -> OwnerReceipt:
        lease = self._leases.get(command.key)
        receipt = OwnerReceipt(
            key=command.key,
            owner_id=command.owner_id,
            command_seq=command.command_seq,
            accepted=accepted,
            runnable_num_tokens=runnable_num_tokens,
            released=released,
            pending_dma=(
                pending_dma
                if pending_dma is not None
                else (lease.pending_dma if lease is not None else None)
            ),
            free_capacity=self.free_capacity(),
            error=error,
        )
        self._outbox.append(receipt)
        return receipt

    # -- publication ----------------------------------------------------------

    def record_published(self, token: OwnerLeaseToken) -> None:
        """Record a published lease token; refuses are illegal from here on.

        Tokens for the matching owner must satisfy the full publication
        invariant: an existing, active (not released/superseded/preempted)
        lease, the accepted grant's ``command_seq``, a strictly increasing
        ``step_seq``, and a token count at or below the granted count.
        Anything else raises :class:`PublicationViolationError`.  Tokens
        for other owners are ignored.
        """
        if token.owner_id != self.owner_rank:
            return
        lease = self._leases.get(token.key)
        if lease is None:
            raise PublicationViolationError(f"no lease for {token.key}")
        if lease.released:
            raise PublicationViolationError(f"lease {token.key} is released")
        if lease.superseded:
            raise PublicationViolationError(f"lease {token.key} is superseded")
        if lease.preempted:
            raise PublicationViolationError(f"lease {token.key} is preempted")
        if token.command_seq != lease.grant_command_seq:
            raise PublicationViolationError(
                f"token for {token.key} carries stale command sequence "
                f"{token.command_seq}, expected grant {lease.grant_command_seq}"
            )
        if token.step_seq <= lease.last_step_seq:
            raise PublicationViolationError(
                f"token for {token.key} regresses or duplicates step "
                f"sequence {token.step_seq} (last {lease.last_step_seq})"
            )
        if token.runnable_num_tokens > lease.runnable_num_tokens:
            raise PublicationViolationError(
                f"token for {token.key} exceeds granted count "
                f"{lease.runnable_num_tokens}"
            )
        lease.published_num_tokens = max(
            lease.published_num_tokens, token.runnable_num_tokens
        )
        lease.last_step_seq = token.step_seq

    # -- batch emission --------------------------------------------------------

    def emit_batch(self, emitted_step_seq: int) -> OwnerReceiptBatch:
        """Emit exactly one receipt batch, draining pending events.

        An enabled worker always emits one batch per step, even when
        ``events`` is empty, so empty work differs from a missing response.
        """
        events = tuple(self._outbox)
        self._outbox.clear()
        return OwnerReceiptBatch(
            owner_rank=self.owner_rank,
            emitted_step_seq=emitted_step_seq,
            events=events,
            free_capacity=self.free_capacity(),
            resident_pages=sum(
                lease.resident_pages
                for lease in self._leases.values()
                if not lease.released
            ),
            pending_dma=sum(
                lease.pending_dma
                for lease in self._leases.values()
                if not lease.released
            ),
        )

    # -- introspection ----------------------------------------------------------

    def free_capacity(self) -> int:
        committed = sum(
            lease.committed_tokens
            for lease in self._leases.values()
            if not lease.released
        )
        return max(0, self.capacity - committed)

    def published_num_tokens(self, key: OwnerLeaseKey) -> int:
        lease = self._leases.get(key)
        return lease.published_num_tokens if lease is not None else 0

    def is_released(self, key: OwnerLeaseKey) -> bool:
        lease = self._leases.get(key)
        return lease is not None and lease.released

    def release_count(self) -> int:
        return self._released_count
