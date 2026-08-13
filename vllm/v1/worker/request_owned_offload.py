# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owner-local correctness ledger for request-owned bulk KV offload.

This module deliberately does not schedule DMA.  Existing offloading workers
remain responsible for copying bytes; this ledger supplies the compatibility
contract missing from their request-id/job-id interface:

* every job is fenced by owner, request epoch, and physical allocation
  generation;
* a receipt must name the exact device blocks and durable host keys submitted;
* an ACTIVE allocation is never a D2H/reclaim victim;
* device blocks become reclaimable only after a successful durable store; and
* a restored allocation becomes HOT only after exact bulk H2D completion.

The ledger lives on one owner worker.  Scheduler messages remain block-ID-free.
Later O-line slices adapt :class:`OffloadingWorker` submissions/results to these
jobs without moving local block ids onto the scheduler wire.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256

from vllm.v1.core.sched.ownership import OwnerLeaseKey
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    LookupResult,
    OffloadingManager,
    OffloadingWorker,
    OffloadKey,
    ReqContext,
    get_offload_group_idx,
    make_offload_key,
)
from vllm.v1.worker.request_owned_kv import RequestOwnedKVSnapshot


class RequestOwnedOffloadError(RuntimeError):
    """Fail-closed owner offload contract violation."""


def make_request_owned_offload_keys(
    snapshot: RequestOwnedKVSnapshot,
    group_block_sizes: tuple[int, ...],
) -> tuple[tuple[OffloadKey, ...], ...]:
    """Build stable owner/epoch/logical-block keys for computed-prefix KV.

    Physical allocation generation is intentionally excluded: the host image
    must restore into a newer generation.  The valid-token extent of each
    block is included so a partial tail block cannot alias its later extended
    image.  Owner rank is included as a defense even though the strict adapter
    also requires one private manager per owner.
    """

    if not isinstance(snapshot, RequestOwnedKVSnapshot):
        raise TypeError(
            f"snapshot must be a RequestOwnedKVSnapshot, got {type(snapshot).__name__}."
        )
    return make_request_owned_offload_keys_for_prefix(
        key=snapshot.key,
        owner_rank=snapshot.owner_rank,
        num_computed_tokens=snapshot.num_computed_tokens,
        group_block_counts=tuple(len(table) for table in snapshot.tables),
        group_block_sizes=group_block_sizes,
    )


def make_request_owned_offload_keys_for_prefix(
    *,
    key: OwnerLeaseKey,
    owner_rank: int,
    num_computed_tokens: int,
    group_block_counts: tuple[int, ...],
    group_block_sizes: tuple[int, ...],
) -> tuple[tuple[OffloadKey, ...], ...]:
    """Build stable host keys from block-ID-free computed-prefix facts."""

    if not isinstance(key, OwnerLeaseKey):
        raise TypeError(f"key must be an OwnerLeaseKey, got {key!r}.")
    if (
        isinstance(owner_rank, bool)
        or not isinstance(owner_rank, int)
        or owner_rank < 0
    ):
        raise TypeError(
            f"owner_rank must be a nonnegative non-bool int, got {owner_rank!r}."
        )
    if (
        isinstance(num_computed_tokens, bool)
        or not isinstance(num_computed_tokens, int)
        or num_computed_tokens < 0
    ):
        raise TypeError(
            "num_computed_tokens must be a nonnegative non-bool int, got "
            f"{num_computed_tokens!r}."
        )
    if not isinstance(group_block_counts, tuple):
        raise TypeError("group_block_counts must be a tuple")
    if not isinstance(group_block_sizes, tuple):
        raise TypeError("group_block_sizes must be a tuple")
    if len(group_block_sizes) != len(group_block_counts):
        raise ValueError("group block sizes and counts must cover the same groups")
    request_bytes = key.request_id.encode("utf-8")

    def frame(value: bytes) -> bytes:
        return len(value).to_bytes(8, "big") + value

    prefix = b"vllm-request-owned-kv-v1\0" + b"".join(
        (
            frame(request_bytes),
            frame(str(key.owner_epoch).encode()),
            frame(str(owner_rank).encode()),
        )
    )
    groups: list[tuple[OffloadKey, ...]] = []
    for group_index, (block_count, block_size) in enumerate(
        zip(group_block_counts, group_block_sizes)
    ):
        if (
            isinstance(block_size, bool)
            or not isinstance(block_size, int)
            or block_size <= 0
        ):
            raise TypeError(
                f"group block sizes must be positive non-bool ints, got {block_size!r}."
            )
        if (
            isinstance(block_count, bool)
            or not isinstance(block_count, int)
            or block_count < 0
        ):
            raise TypeError(
                "group block counts must be nonnegative non-bool ints, got "
                f"{block_count!r}."
            )
        keys: list[OffloadKey] = []
        for block_index in range(block_count):
            valid_tokens = min(
                block_size,
                max(0, num_computed_tokens - block_index * block_size),
            )
            if valid_tokens <= 0:
                raise ValueError(
                    f"group {group_index} table contains block {block_index} "
                    "outside the computed prefix"
                )
            digest = sha256(
                prefix
                + frame(str(group_index).encode())
                + frame(str(block_index).encode())
                + frame(str(valid_tokens).encode())
            ).digest()
            keys.append(make_offload_key(digest, group_index))
        groups.append(tuple(keys))
    return tuple(groups)


@dataclass(frozen=True, slots=True)
class OwnerOffloadIdentity:
    """Exact owner-local physical allocation identity."""

    key: OwnerLeaseKey
    owner_rank: int
    allocation_generation: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, OwnerLeaseKey):
            raise TypeError(f"key must be an OwnerLeaseKey, got {self.key!r}.")
        for name in ("owner_rank", "allocation_generation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(
                    f"{name} must be a nonnegative non-bool int, got {value!r}."
                )

    @classmethod
    def from_snapshot(cls, snapshot: RequestOwnedKVSnapshot) -> "OwnerOffloadIdentity":
        if not isinstance(snapshot, RequestOwnedKVSnapshot):
            raise TypeError(
                "snapshot must be a RequestOwnedKVSnapshot, got "
                f"{type(snapshot).__name__}."
            )
        return cls(
            key=snapshot.key,
            owner_rank=snapshot.owner_rank,
            allocation_generation=snapshot.allocation_generation,
        )


@dataclass(frozen=True, slots=True)
class OwnerOffloadPlan:
    """Exact local block-to-host-key mapping for one bulk transfer.

    One host key corresponds to one physical block in this first strict slice.
    This matches the current request-owned DSv4 packed layout.  A future
    block-size-factor adapter must expand this type explicitly rather than
    silently zipping unequal geometries.
    """

    identity: OwnerOffloadIdentity
    device_block_ids: tuple[tuple[int, ...], ...]
    offload_keys: tuple[tuple[OffloadKey, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, OwnerOffloadIdentity):
            raise TypeError(
                f"identity must be an OwnerOffloadIdentity, got {self.identity!r}."
            )
        if not isinstance(self.device_block_ids, tuple) or not isinstance(
            self.offload_keys, tuple
        ):
            raise TypeError("device_block_ids and offload_keys must be tuples")
        if len(self.device_block_ids) != len(self.offload_keys):
            raise ValueError(
                "device_block_ids and offload_keys must cover the same groups, got "
                f"{len(self.device_block_ids)} != {len(self.offload_keys)}."
            )

        seen_blocks: set[int] = set()
        seen_keys: set[OffloadKey] = set()
        for group_index, (block_ids, keys) in enumerate(
            zip(self.device_block_ids, self.offload_keys)
        ):
            if not isinstance(block_ids, tuple) or not isinstance(keys, tuple):
                raise TypeError("every device block/key group must be a tuple")
            if len(block_ids) != len(keys):
                raise ValueError(
                    f"group {group_index} has {len(block_ids)} device blocks but "
                    f"{len(keys)} offload keys; first-round bulk geometry is 1:1."
                )
            for block_id in block_ids:
                if (
                    isinstance(block_id, bool)
                    or not isinstance(block_id, int)
                    or block_id <= 0
                ):
                    raise TypeError(
                        "transfer block ids must be positive non-bool ints "
                        f"(block 0 is the null block), got {block_id!r}."
                    )
                if block_id in seen_blocks:
                    raise ValueError(
                        f"device block id {block_id} appears in multiple plan positions"
                    )
                seen_blocks.add(block_id)
            for key in keys:
                if not isinstance(key, bytes) or not key:
                    raise TypeError(
                        "offload keys must be nonempty bytes-compatible OffloadKey "
                        f"values, got {key!r}."
                    )
                if get_offload_group_idx(key) != group_index:
                    raise ValueError(
                        f"offload key in group {group_index} encodes group "
                        f"{get_offload_group_idx(key)}"
                    )
                if key in seen_keys:
                    raise ValueError(f"offload key {key!r} appears more than once")
                seen_keys.add(key)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: RequestOwnedKVSnapshot,
        offload_keys: tuple[tuple[OffloadKey, ...], ...],
    ) -> "OwnerOffloadPlan":
        """Bind logical host keys to the matching prefix of each local table."""

        if not isinstance(snapshot, RequestOwnedKVSnapshot):
            raise TypeError(
                "snapshot must be a RequestOwnedKVSnapshot, got "
                f"{type(snapshot).__name__}."
            )
        if not isinstance(offload_keys, tuple):
            raise TypeError("offload_keys must be a tuple of group key tuples")
        if len(snapshot.tables) != len(offload_keys):
            raise ValueError(
                "snapshot tables and offload keys must cover the same groups, got "
                f"{len(snapshot.tables)} != {len(offload_keys)}."
            )
        device_block_ids: list[tuple[int, ...]] = []
        for group_index, (table, keys) in enumerate(zip(snapshot.tables, offload_keys)):
            if not isinstance(keys, tuple):
                raise TypeError(f"offload key group {group_index} must be a tuple")
            if len(keys) > len(table):
                raise ValueError(
                    f"group {group_index} requests {len(keys)} host keys but the "
                    f"owner-local table has only {len(table)} blocks."
                )
            device_block_ids.append(table[: len(keys)])
        return cls(
            identity=OwnerOffloadIdentity.from_snapshot(snapshot),
            device_block_ids=tuple(device_block_ids),
            offload_keys=offload_keys,
        )


class OwnerBulkTransferDirection(str, Enum):
    STORE = "STORE"
    RESTORE = "RESTORE"


@dataclass(frozen=True, slots=True)
class OwnerBulkTransferJob:
    job_id: int
    direction: OwnerBulkTransferDirection
    plan: OwnerOffloadPlan


@dataclass(frozen=True, slots=True)
class OwnerBulkTransferReceipt:
    """Worker-local completion carrying the exact submitted identity/layout."""

    job_id: int
    direction: OwnerBulkTransferDirection
    identity: OwnerOffloadIdentity
    device_block_ids: tuple[tuple[int, ...], ...]
    offload_keys: tuple[tuple[OffloadKey, ...], ...]
    success: bool
    error: str | None = None

    @classmethod
    def for_job(
        cls,
        job: OwnerBulkTransferJob,
        *,
        success: bool,
        error: str | None = None,
    ) -> "OwnerBulkTransferReceipt":
        return cls(
            job_id=job.job_id,
            direction=job.direction,
            identity=job.plan.identity,
            device_block_ids=job.plan.device_block_ids,
            offload_keys=job.plan.offload_keys,
            success=success,
            error=error,
        )

    def __post_init__(self) -> None:
        if isinstance(self.job_id, bool) or not isinstance(self.job_id, int):
            raise TypeError(f"job_id must be a non-bool int, got {self.job_id!r}.")
        if self.job_id < 0:
            raise ValueError(f"job_id must be nonnegative, got {self.job_id}.")
        if not isinstance(self.direction, OwnerBulkTransferDirection):
            raise TypeError(
                "direction must be an OwnerBulkTransferDirection, got "
                f"{self.direction!r}."
            )
        if not isinstance(self.identity, OwnerOffloadIdentity):
            raise TypeError(
                f"identity must be an OwnerOffloadIdentity, got {self.identity!r}."
            )
        if not isinstance(self.success, bool):
            raise TypeError(f"success must be a bool, got {self.success!r}.")
        if self.success and self.error is not None:
            raise ValueError("a successful transfer receipt must not carry an error")
        if not self.success and not self.error:
            raise ValueError("a failed transfer receipt must carry an error")


@dataclass(slots=True)
class _OwnerLeaseOffloadState:
    snapshot: RequestOwnedKVSnapshot
    active: bool
    hot: bool
    reclaimable: tuple[tuple[int, ...], ...] | None = None
    reclaim_taken: bool = False
    closed: bool = False
    inflight_jobs: set[int] = field(default_factory=set)


class RequestOwnedBulkOffloadLedger:
    """Fail-closed per-owner bulk transfer and reclaim oracle."""

    def __init__(self, owner_rank: int) -> None:
        if isinstance(owner_rank, bool) or not isinstance(owner_rank, int):
            raise TypeError(f"owner_rank must be a non-bool int, got {owner_rank!r}.")
        if owner_rank < 0:
            raise ValueError(f"owner_rank must be nonnegative, got {owner_rank}.")
        self.owner_rank = owner_rank
        self._job_counter = 0
        self._jobs: dict[int, OwnerBulkTransferJob] = {}
        self._states: dict[OwnerOffloadIdentity, _OwnerLeaseOffloadState] = {}
        self._current: dict[OwnerLeaseKey, OwnerOffloadIdentity] = {}
        self._epoch_fence: dict[str, int] = {}
        # The ledger is per owner, so the ordinary hash+group OffloadKey is
        # already owner-namespaced without changing its upstream encoding.
        self._durable_host_keys: set[OffloadKey] = set()

    def bind(
        self, snapshot: RequestOwnedKVSnapshot, *, active: bool
    ) -> OwnerOffloadIdentity:
        """Bind the exact current generation before any transfer submission."""

        if not isinstance(active, bool):
            raise TypeError(f"active must be a bool, got {active!r}.")
        identity = OwnerOffloadIdentity.from_snapshot(snapshot)
        if identity.owner_rank != self.owner_rank:
            raise RequestOwnedOffloadError(
                f"wrong owner rank {identity.owner_rank}; ledger owns {self.owner_rank}"
            )
        epoch_fence = self._epoch_fence.get(identity.key.request_id, -1)
        if identity.key.owner_epoch < epoch_fence:
            raise RequestOwnedOffloadError(
                f"stale request epoch {identity.key.owner_epoch}; "
                f"fence is {epoch_fence}"
            )
        if identity.key.owner_epoch > epoch_fence:
            for old_key, old_identity in tuple(self._current.items()):
                if (
                    old_key.request_id != identity.key.request_id
                    or old_key.owner_epoch >= identity.key.owner_epoch
                ):
                    continue
                old_state = self._states[old_identity]
                if old_state.inflight_jobs:
                    raise RequestOwnedOffloadError(
                        "cannot advance request epoch with owner DMA still in flight"
                    )
                if old_state.reclaimable is not None and not old_state.reclaim_taken:
                    raise RequestOwnedOffloadError(
                        "cannot advance request epoch before durable source reclaim "
                        "is consumed"
                    )
                if not old_state.closed:
                    raise RequestOwnedOffloadError(
                        "cannot advance request epoch before the old physical "
                        "generation is reclaimed or explicitly aborted"
                    )
                del self._current[old_key]
            self._epoch_fence[identity.key.request_id] = identity.key.owner_epoch
        prior_identity = self._current.get(identity.key)
        if prior_identity == identity:
            state = self._states[identity]
            if state.snapshot != snapshot:
                raise RequestOwnedOffloadError(
                    "same owner/epoch/generation was rebound with different "
                    "physical state"
                )
            if state.active != active:
                raise RequestOwnedOffloadError(
                    "same generation active state must change through "
                    "retire()/activate()"
                )
            return identity
        if prior_identity is not None:
            prior = self._states[prior_identity]
            if prior.inflight_jobs:
                raise RequestOwnedOffloadError(
                    "cannot replace a generation with owner DMA still in flight"
                )
            if prior.reclaimable is not None and not prior.reclaim_taken:
                raise RequestOwnedOffloadError(
                    "cannot replace a generation before durable source reclaim "
                    "is consumed"
                )
            if not prior.closed:
                raise RequestOwnedOffloadError(
                    "cannot replace a generation before the old physical generation "
                    "is reclaimed or explicitly aborted"
                )
        self._current[identity.key] = identity
        self._states[identity] = _OwnerLeaseOffloadState(
            snapshot=snapshot,
            active=active,
            hot=active,
        )
        return identity

    def retire(self, identity: OwnerOffloadIdentity) -> None:
        state = self._require_current(identity)
        if not state.active:
            raise RequestOwnedOffloadError("lease is already inactive")
        state.active = False

    def begin_store(self, plan: OwnerOffloadPlan) -> OwnerBulkTransferJob:
        state = self._require_plan(plan)
        if state.active:
            raise RequestOwnedOffloadError(
                "ACTIVE owner KV must never be stored/reclaimed"
            )
        return self._begin(OwnerBulkTransferDirection.STORE, plan, state)

    def begin_restore(self, plan: OwnerOffloadPlan) -> OwnerBulkTransferJob:
        state = self._require_plan(plan)
        if state.active:
            raise RequestOwnedOffloadError(
                "ACTIVE owner KV must never be restored over"
            )
        missing = [
            key
            for group in plan.offload_keys
            for key in group
            if key not in self._durable_host_keys
        ]
        if missing:
            raise RequestOwnedOffloadError(
                f"restore requires durable host keys; {len(missing)} key(s) are missing"
            )
        state.hot = False
        return self._begin(OwnerBulkTransferDirection.RESTORE, plan, state)

    def complete(self, receipt: OwnerBulkTransferReceipt) -> None:
        if not isinstance(receipt, OwnerBulkTransferReceipt):
            raise TypeError(
                "receipt must be an OwnerBulkTransferReceipt, got "
                f"{type(receipt).__name__}."
            )
        job = self._jobs.get(receipt.job_id)
        if job is None:
            raise RequestOwnedOffloadError(
                f"unknown, stale, or duplicate owner transfer job {receipt.job_id}"
            )
        expected = OwnerBulkTransferReceipt.for_job(
            job,
            success=receipt.success,
            error=receipt.error,
        )
        if receipt != expected:
            raise RequestOwnedOffloadError(
                f"owner transfer receipt {receipt.job_id} does not match its exact job"
            )
        state = self._require_current(job.plan.identity)
        del self._jobs[job.job_id]
        state.inflight_jobs.remove(job.job_id)
        if not receipt.success:
            return
        if job.direction is OwnerBulkTransferDirection.STORE:
            self._durable_host_keys.update(
                key for group in job.plan.offload_keys for key in group
            )
            state.reclaimable = job.plan.device_block_ids
            state.reclaim_taken = False
        else:
            state.hot = True

    def take_reclaimable(
        self, identity: OwnerOffloadIdentity
    ) -> tuple[tuple[int, ...], ...]:
        state = self._require_current(identity)
        if state.active:
            raise RequestOwnedOffloadError("ACTIVE owner KV is not reclaimable")
        if state.reclaimable is None:
            raise RequestOwnedOffloadError(
                "device blocks are not reclaimable before durable store completion"
            )
        if state.reclaim_taken:
            raise RequestOwnedOffloadError("reclaim receipt was already consumed")
        if state.inflight_jobs:
            raise RequestOwnedOffloadError(
                "device blocks have owner DMA still in flight"
            )
        state.reclaim_taken = True
        state.closed = True
        return state.reclaimable

    def activate(self, identity: OwnerOffloadIdentity) -> None:
        state = self._require_current(identity)
        if state.active:
            raise RequestOwnedOffloadError("lease is already ACTIVE")
        if not state.hot:
            raise RequestOwnedOffloadError(
                "bulk restore completion is required before activation"
            )
        if state.inflight_jobs:
            raise RequestOwnedOffloadError("cannot activate with owner DMA in flight")
        state.active = True
        state.closed = False

    def abort(self, identity: OwnerOffloadIdentity) -> None:
        """Invalidate all jobs for an allocation without discarding durable host KV."""

        state = self._require_current(identity)
        for job_id in tuple(state.inflight_jobs):
            self._jobs.pop(job_id, None)
        state.inflight_jobs.clear()
        state.reclaimable = None
        state.reclaim_taken = False
        state.active = False
        state.hot = False
        state.closed = True

    def is_current(self, identity: OwnerOffloadIdentity) -> bool:
        """Return whether ``identity`` is the exact current bound generation."""

        if not isinstance(identity, OwnerOffloadIdentity):
            raise TypeError(
                "identity must be an OwnerOffloadIdentity, got "
                f"{type(identity).__name__}."
            )
        if identity.owner_rank != self.owner_rank:
            return False
        return self._current.get(identity.key) == identity

    def is_hot(self, identity: OwnerOffloadIdentity) -> bool:
        return self._require_current(identity).hot

    def is_host_durable(self, plan: OwnerOffloadPlan) -> bool:
        self._require_plan(plan)
        return all(
            key in self._durable_host_keys
            for group in plan.offload_keys
            for key in group
        )

    def forget_host_keys(self, keys: tuple[OffloadKey, ...]) -> None:
        """Forget keys evicted by the owner-local offloading manager."""

        for key in keys:
            self._durable_host_keys.discard(key)

    @property
    def pending_jobs(self) -> tuple[OwnerBulkTransferJob, ...]:
        return tuple(self._jobs[job_id] for job_id in sorted(self._jobs))

    def _begin(
        self,
        direction: OwnerBulkTransferDirection,
        plan: OwnerOffloadPlan,
        state: _OwnerLeaseOffloadState,
    ) -> OwnerBulkTransferJob:
        if state.inflight_jobs:
            raise RequestOwnedOffloadError(
                "first-round bulk adapter allows only one in-flight job per lease"
            )
        job = OwnerBulkTransferJob(
            job_id=self._job_counter,
            direction=direction,
            plan=plan,
        )
        self._job_counter += 1
        self._jobs[job.job_id] = job
        state.inflight_jobs.add(job.job_id)
        return job

    def _require_plan(self, plan: OwnerOffloadPlan) -> _OwnerLeaseOffloadState:
        if not isinstance(plan, OwnerOffloadPlan):
            raise TypeError(
                f"plan must be an OwnerOffloadPlan, got {type(plan).__name__}."
            )
        state = self._require_current(plan.identity)
        if len(plan.device_block_ids) != len(state.snapshot.tables):
            raise RequestOwnedOffloadError(
                "transfer plan and bound snapshot must cover the same groups"
            )
        for group_index, block_ids in enumerate(plan.device_block_ids):
            table = state.snapshot.tables[group_index]
            if table[: len(block_ids)] != block_ids:
                raise RequestOwnedOffloadError(
                    f"group {group_index} transfer blocks are not the bound "
                    "table prefix"
                )
        return state

    def _require_current(
        self, identity: OwnerOffloadIdentity
    ) -> _OwnerLeaseOffloadState:
        if not isinstance(identity, OwnerOffloadIdentity):
            raise TypeError(
                "identity must be an OwnerOffloadIdentity, got "
                f"{type(identity).__name__}."
            )
        if identity.owner_rank != self.owner_rank:
            raise RequestOwnedOffloadError(
                f"wrong owner rank {identity.owner_rank}; ledger owns {self.owner_rank}"
            )
        if self._current.get(identity.key) != identity:
            raise RequestOwnedOffloadError(
                "stale owner epoch or physical allocation generation"
            )
        state = self._states.get(identity)
        if state is None:
            raise RequestOwnedOffloadError("owner allocation is not bound")
        return state


@dataclass(slots=True)
class _AdapterSubmission:
    job: OwnerBulkTransferJob
    manager_keys: tuple[OffloadKey, ...]
    req_context: ReqContext
    manager_spec: LoadStoreSpec
    aborted: bool = False


class RequestOwnedBulkOffloadAdapter:
    """Owner-local strict bulk adapter over the upstream offload primitives.

    The manager passed here must be private to this owner.  That makes ordinary
    content/group-qualified :class:`OffloadKey` values owner-namespaced without
    changing their upstream encoding or exposing physical block ids to the
    scheduler.  The adapter only accepts 1:1 GPU/offloaded block geometry.

    STORE may reuse already-durable host keys.  Any keys that still require a
    copy must form one contiguous logical span per KV group, which is exactly
    what :class:`GPULoadStoreSpec` can express.  Unsupported mixed holes fail
    closed rather than copying the wrong physical blocks.
    """

    def __init__(
        self,
        *,
        owner_rank: int,
        manager: OffloadingManager,
        worker: OffloadingWorker,
    ) -> None:
        if not isinstance(manager, OffloadingManager):
            raise TypeError(f"manager must be an OffloadingManager, got {manager!r}.")
        if not isinstance(worker, OffloadingWorker):
            raise TypeError(f"worker must be an OffloadingWorker, got {worker!r}.")
        if not callable(getattr(manager, "evict_keys", None)):
            raise TypeError(
                "request-owned bulk offload requires an exact-key-evictable "
                "CPU offloading manager"
            )
        if getattr(manager, "capacity_blocks", 0) <= 0:
            raise ValueError(
                "request-owned bulk offload requires at least one host block"
            )
        self.manager = manager
        self.worker = worker
        self.ledger = RequestOwnedBulkOffloadLedger(owner_rank)
        self._submissions: dict[int, _AdapterSubmission] = {}
        self._ready_receipts: deque[OwnerBulkTransferReceipt] = deque()
        self._owned_host_keys: dict[OwnerLeaseKey, set[OffloadKey]] = {}

    def bind(
        self, snapshot: RequestOwnedKVSnapshot, *, active: bool
    ) -> OwnerOffloadIdentity:
        return self.ledger.bind(snapshot, active=active)

    def retire(self, identity: OwnerOffloadIdentity) -> None:
        self.ledger.retire(identity)

    def submit_store(self, plan: OwnerOffloadPlan) -> OwnerBulkTransferJob:
        """Submit an exact bulk D2H, or queue a fail-closed receipt."""

        job = self.ledger.begin_store(plan)
        req_context = self._req_context(job)
        flat_keys = self._flat_keys(plan)
        lookup = {key: self.manager.lookup(key, req_context) for key in flat_keys}

        try:
            prepared = self.manager.prepare_store(flat_keys, req_context)
        except Exception as exc:
            self._finish_immediate(job, error=f"prepare_store failed: {exc}")
            return job
        if prepared is None:
            self._finish_immediate(job, error="host tier has no store capacity")
            return job

        manager_keys = tuple(prepared.keys_to_store)
        self._forget_host_keys(tuple(prepared.evicted_keys))
        error = self._validate_store_selection(plan, manager_keys, lookup)
        if error is not None:
            self.manager.complete_store(manager_keys, req_context, success=False)
            self._finish_immediate(job, error=error)
            return job
        if not manager_keys:
            self._finish_immediate(job)
            return job

        gpu_spec = self._gpu_spec_for_keys(plan, manager_keys)
        submission = _AdapterSubmission(
            job=job,
            manager_keys=manager_keys,
            req_context=req_context,
            manager_spec=prepared.store_spec,
        )
        try:
            submitted = self.worker.submit_store(
                job.job_id, gpu_spec, prepared.store_spec
            )
        except Exception as exc:
            submitted = False
            error = f"submit_store failed: {exc}"
        else:
            error = "offloading worker rejected the store submission"
        if not submitted:
            self.manager.complete_store(manager_keys, req_context, success=False)
            self._finish_immediate(job, error=error)
            return job
        self._submissions[job.job_id] = submission
        return job

    def submit_restore(self, plan: OwnerOffloadPlan) -> OwnerBulkTransferJob:
        """Submit exact full H2D; activation remains fenced by its receipt."""

        job = self.ledger.begin_restore(plan)
        req_context = self._req_context(job)
        flat_keys = self._flat_keys(plan)
        if not flat_keys:
            self._finish_immediate(job)
            return job
        misses = [
            key
            for key in flat_keys
            if self.manager.lookup(key, req_context) is not LookupResult.HIT
        ]
        if misses:
            self._finish_immediate(
                job,
                error=f"host manager cannot read {len(misses)} exact key(s)",
            )
            return job
        try:
            host_spec = self.manager.prepare_load(flat_keys, req_context)
        except Exception as exc:
            self._finish_immediate(job, error=f"prepare_load failed: {exc}")
            return job

        gpu_spec = self._gpu_spec_for_keys(plan, flat_keys)
        submission = _AdapterSubmission(
            job=job,
            manager_keys=flat_keys,
            req_context=req_context,
            manager_spec=host_spec,
        )
        try:
            submitted = self.worker.submit_load(job.job_id, host_spec, gpu_spec)
        except Exception as exc:
            submitted = False
            error = f"submit_load failed: {exc}"
        else:
            error = "offloading worker rejected the load submission"
        if not submitted:
            self.manager.complete_load(flat_keys, req_context)
            self._finish_immediate(job, error=error)
            return job
        self._submissions[job.job_id] = submission
        return job

    def poll(self) -> tuple[OwnerBulkTransferReceipt, ...]:
        """Consume worker results and return exact owner-qualified receipts."""

        receipts = list(self._ready_receipts)
        self._ready_receipts.clear()
        for result in self.worker.get_finished():
            submission = self._submissions.pop(result.job_id, None)
            if submission is None:
                raise RequestOwnedOffloadError(
                    "offloading worker returned an unknown, stale, or duplicate "
                    f"job {result.job_id}"
                )
            success = result.success and not submission.aborted
            error: str | None = None
            try:
                if submission.job.direction is OwnerBulkTransferDirection.STORE:
                    self.manager.complete_store(
                        submission.manager_keys,
                        submission.req_context,
                        success=success,
                    )
                else:
                    self.manager.complete_load(
                        submission.manager_keys, submission.req_context
                    )
            except Exception as exc:
                success = False
                error = f"manager completion failed: {exc}"
            if submission.aborted:
                error = "owner transfer completed after its allocation was aborted"
            elif not result.success:
                error = "offloading worker reported transfer failure"

            receipt = OwnerBulkTransferReceipt.for_job(
                submission.job,
                success=success,
                error=error,
            )
            if not submission.aborted:
                self.ledger.complete(receipt)
            receipts.append(receipt)
        for receipt in receipts:
            if (
                receipt.success
                and receipt.direction is OwnerBulkTransferDirection.STORE
            ):
                self._record_store_image(receipt)
        return tuple(receipts)

    def wait(self, jobs: tuple[OwnerBulkTransferJob, ...]) -> None:
        """Wait for submitted DMA only; call :meth:`poll` for receipts."""

        job_ids = {job.job_id for job in jobs if job.job_id in self._submissions}
        if job_ids:
            self.worker.wait(job_ids)

    def take_reclaimable(
        self, identity: OwnerOffloadIdentity
    ) -> tuple[tuple[int, ...], ...]:
        return self.ledger.take_reclaimable(identity)

    def activate(self, identity: OwnerOffloadIdentity) -> None:
        self.ledger.activate(identity)

    def abort(self, identity: OwnerOffloadIdentity) -> None:
        aborted_ids = {
            job.job_id
            for job in self.ledger.pending_jobs
            if job.plan.identity == identity
        }
        for job_id in aborted_ids:
            submission = self._submissions.get(job_id)
            if submission is not None:
                submission.aborted = True
        self.ledger.abort(identity)

    def release(self, snapshot: RequestOwnedKVSnapshot) -> None:
        """Close a bound generation on request RELEASE, if one exists."""

        identity = OwnerOffloadIdentity.from_snapshot(snapshot)
        if self.ledger.is_current(identity):
            self.abort(identity)

    def evict_owned_host_keys(self, key: OwnerLeaseKey) -> None:
        """Forget and evict all durable host images owned by one lease."""

        keys = tuple(self._owned_host_keys.get(key, ()))
        if not keys:
            return
        evict = getattr(self.manager, "evict_keys", None)
        if not callable(evict):
            raise RequestOwnedOffloadError(
                "exclusive request-owned offload manager cannot evict host keys"
            )
        evicted = tuple(evict(keys))
        if set(evicted) != set(keys):
            raise RequestOwnedOffloadError(
                "request-owned RELEASE could not evict its exact durable host image"
            )
        self._forget_host_keys(keys)

    def shutdown(self) -> None:
        self.worker.shutdown()
        self.manager.shutdown()

    def _finish_immediate(
        self,
        job: OwnerBulkTransferJob,
        *,
        error: str | None = None,
    ) -> None:
        receipt = OwnerBulkTransferReceipt.for_job(
            job,
            success=error is None,
            error=error,
        )
        self.ledger.complete(receipt)
        self._ready_receipts.append(receipt)

    def _record_store_image(self, receipt: OwnerBulkTransferReceipt) -> None:
        """Track the exact latest image and retire stale partial-tail keys."""

        key = receipt.identity.key
        current = {item for group in receipt.offload_keys for item in group}
        stale = self._owned_host_keys.get(key, set()) - current
        if stale:
            evict = getattr(self.manager, "evict_keys", None)
            if not callable(evict) or set(evict(tuple(stale))) != stale:
                raise RequestOwnedOffloadError(
                    "request-owned store could not retire its stale host image"
                )
            self._forget_host_keys(tuple(stale))
        self._owned_host_keys[key] = current

    def _forget_host_keys(self, keys: tuple[OffloadKey, ...]) -> None:
        if not keys:
            return
        forgotten = set(keys)
        self.ledger.forget_host_keys(keys)
        for owner_key, owned in tuple(self._owned_host_keys.items()):
            owned.difference_update(forgotten)
            if not owned:
                self._owned_host_keys.pop(owner_key, None)

    @staticmethod
    def _req_context(job: OwnerBulkTransferJob) -> ReqContext:
        identity = job.plan.identity
        return ReqContext(
            req_id=(
                f"{identity.key.request_id}:owner={identity.owner_rank}:"
                f"epoch={identity.key.owner_epoch}:"
                f"generation={identity.allocation_generation}"
            )
        )

    @staticmethod
    def _flat_keys(plan: OwnerOffloadPlan) -> tuple[OffloadKey, ...]:
        return tuple(key for group in plan.offload_keys for key in group)

    @staticmethod
    def _validate_store_selection(
        plan: OwnerOffloadPlan,
        selected: tuple[OffloadKey, ...],
        lookup: dict[OffloadKey, LookupResult],
    ) -> str | None:
        selected_set = set(selected)
        if len(selected_set) != len(selected):
            return "offloading manager returned duplicate store keys"
        plan_keys = RequestOwnedBulkOffloadAdapter._flat_keys(plan)
        if not selected_set.issubset(plan_keys):
            return "offloading manager returned a key outside the owner plan"
        for key in plan_keys:
            if key in selected_set:
                continue
            if lookup[key] is not LookupResult.HIT:
                return "store omitted a host key that was not already durable"
        for group in plan.offload_keys:
            positions = [
                index for index, key in enumerate(group) if key in selected_set
            ]
            if positions and positions != list(range(positions[0], positions[-1] + 1)):
                return "store keys contain a non-contiguous hole within a KV group"
        return None

    @staticmethod
    def _gpu_spec_for_keys(
        plan: OwnerOffloadPlan, selected: tuple[OffloadKey, ...]
    ) -> GPULoadStoreSpec:
        selected_set = set(selected)
        block_ids: list[int] = []
        group_sizes: list[int] = []
        block_indices: list[int] = []
        for group_blocks, group_keys in zip(plan.device_block_ids, plan.offload_keys):
            positions = [
                index for index, key in enumerate(group_keys) if key in selected_set
            ]
            group_sizes.append(len(positions))
            block_indices.append(positions[0] if positions else 0)
            block_ids.extend(group_blocks[index] for index in positions)
        return GPULoadStoreSpec(block_ids, group_sizes, block_indices)


@dataclass(slots=True)
class RequestOwnedBulkRestoreWork:
    """One exact post-zero, pre-forward full-restore action.

    The object stays worker-private.  The device-specific runner owns the
    zeroing seam, then calls :meth:`execute_after_zero`; the method is
    replay-fenced and returns only after the upstream offloading worker has
    produced the exact completion receipt.
    """

    step_seq: int
    adapter: RequestOwnedBulkOffloadAdapter
    plan: OwnerOffloadPlan
    zero_block_ids: tuple[tuple[int, ...], ...]
    executed: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_seq, bool)
            or not isinstance(self.step_seq, int)
            or self.step_seq <= 0
        ):
            raise TypeError(
                f"step_seq must be a positive non-bool int, got {self.step_seq!r}."
            )
        if not isinstance(self.adapter, RequestOwnedBulkOffloadAdapter):
            raise TypeError("adapter must be a RequestOwnedBulkOffloadAdapter")
        if not isinstance(self.plan, OwnerOffloadPlan):
            raise TypeError("plan must be an OwnerOffloadPlan")
        if not isinstance(self.zero_block_ids, tuple):
            raise TypeError("zero_block_ids must be a tuple of group tuples")
        if len(self.zero_block_ids) != len(self.plan.device_block_ids):
            raise ValueError("zero and restore plans must cover the same groups")
        for restore_ids, zero_ids in zip(
            self.plan.device_block_ids, self.zero_block_ids
        ):
            if not set(restore_ids).issubset(zero_ids):
                raise ValueError(
                    "every restore destination must be covered by the zero plan"
                )

    def execute_after_zero(self) -> OwnerBulkTransferReceipt:
        if self.executed:
            raise RequestOwnedOffloadError(
                f"restore work for step {self.step_seq} was already executed"
            )
        self.executed = True
        job = self.adapter.submit_restore(self.plan)
        self.adapter.wait((job,))
        receipts = self.adapter.poll()
        matching = tuple(
            receipt for receipt in receipts if receipt.job_id == job.job_id
        )
        if len(matching) != 1 or len(receipts) != 1:
            raise RequestOwnedOffloadError(
                f"bulk restore job {job.job_id} produced {len(matching)} exact "
                f"and {len(receipts)} total receipts"
            )
        receipt = matching[0]
        if not receipt.success:
            raise RequestOwnedOffloadError(
                receipt.error or f"bulk restore job {job.job_id} failed"
            )
        return receipt
