# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-neutral shared byte plane for request-owned KV host images.

This is the CPU-only StateHarbor G2-A seam.  It persists immutable opaque
group bytes behind :class:`RequestOwnedBulkOffloadLedger`; it does not schedule
DMA, allocate device blocks, publish HOT, or make scheduler decisions.

The source image identity is intentionally distinct from a restore target.  A
new owner/allocation may restore the same immutable source image, while its
destination generation is fenced by the ordinary owner offload plan/receipt.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Protocol
from uuid import UUID

from vllm.v1.core.sched.ownership import OwnerReceipt
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.worker.request_owned_offload import (
    OwnerBulkTransferJob,
    OwnerBulkTransferReceipt,
    OwnerOffloadIdentity,
    OwnerOffloadPlan,
    RequestOwnedBulkOffloadLedger,
    RequestOwnedOffloadError,
)

_GROUP_DOMAIN = b"vllm-stateharbor-group-v1\0"
_MANIFEST_DOMAIN = b"vllm-stateharbor-manifest-v1\0"
_INSTANCE_DOMAIN = b"vllm-stateharbor-lmcache-instance-v1\0"


class StateHarborBytePlaneError(RequestOwnedOffloadError):
    """Fail-closed shared byte-plane contract violation."""


def _frame(value: bytes) -> bytes:
    return len(value).to_bytes(8, "big") + value


def _int_bytes(value: int) -> bytes:
    return str(value).encode("ascii")


def _require_uint(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a nonnegative non-bool int, got {value!r}.")


def _require_nonempty_bytes(name: str, value: bytes) -> None:
    if not isinstance(value, bytes) or not value:
        raise TypeError(f"{name} must be nonempty bytes, got {value!r}.")


@dataclass(frozen=True, slots=True)
class StateHarborSourceImage:
    """Immutable logical identity of bytes produced by one source activation."""

    model_fingerprint: bytes
    layout_fingerprint: bytes
    session_id: str
    request_id: str
    source_owner_rank: int
    source_owner_epoch: int
    source_activation_generation: int

    def __post_init__(self) -> None:
        for name in ("model_fingerprint", "layout_fingerprint"):
            value = getattr(self, name)
            if not isinstance(value, bytes) or len(value) != 32:
                raise TypeError(f"{name} must be a SHA256 digest")
        for name in ("session_id", "request_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TypeError(f"{name} must be a nonempty str, got {value!r}.")
        for name in (
            "source_owner_rank",
            "source_owner_epoch",
            "source_activation_generation",
        ):
            _require_uint(name, getattr(self, name))

    def canonical_bytes(self) -> bytes:
        return b"".join(
            (
                _frame(self.model_fingerprint),
                _frame(self.layout_fingerprint),
                _frame(self.session_id.encode("utf-8")),
                _frame(self.request_id.encode("utf-8")),
                _frame(_int_bytes(self.source_owner_rank)),
                _frame(_int_bytes(self.source_owner_epoch)),
                _frame(_int_bytes(self.source_activation_generation)),
            )
        )


@dataclass(frozen=True, slots=True)
class StateHarborWriterFence:
    """One non-reusable process incarnation and registration generation."""

    process_incarnation: UUID
    registration_generation: int

    def __post_init__(self) -> None:
        if not isinstance(self.process_incarnation, UUID):
            raise TypeError(
                f"process_incarnation must be a UUID, got {self.process_incarnation!r}."
            )
        _require_uint("registration_generation", self.registration_generation)

    def canonical_bytes(self) -> bytes:
        return self.process_incarnation.bytes + _frame(
            _int_bytes(self.registration_generation)
        )


def derive_lmcache_instance_id(fence: StateHarborWriterFence, namespace: bytes) -> int:
    """Derive a collision-resistant positive signed-63-bit LMCache instance id."""

    if not isinstance(fence, StateHarborWriterFence):
        raise TypeError(f"fence must be a StateHarborWriterFence, got {fence!r}.")
    _require_nonempty_bytes("namespace", namespace)
    value = int.from_bytes(
        sha256(_INSTANCE_DOMAIN + fence.canonical_bytes() + _frame(namespace)).digest()[
            :8
        ],
        "big",
    ) & ((1 << 63) - 1)
    return value or 1


@dataclass(frozen=True, slots=True)
class StateHarborGroupPayload:
    """Opaque bytes and physical extent metadata for one heterogeneous group."""

    group_index: int
    logical_token_span: tuple[int, int]
    valid_extents: tuple[int, ...]
    payload: bytes

    def __post_init__(self) -> None:
        _require_uint("group_index", self.group_index)
        if (
            not isinstance(self.logical_token_span, tuple)
            or len(self.logical_token_span) != 2
        ):
            raise TypeError("logical_token_span must be a (start, end) tuple")
        start, end = self.logical_token_span
        _require_uint("logical_token_span start", start)
        _require_uint("logical_token_span end", end)
        if end <= start:
            raise ValueError("logical_token_span end must be greater than start")
        if not isinstance(self.valid_extents, tuple) or not self.valid_extents:
            raise TypeError("valid_extents must be a nonempty tuple")
        for extent in self.valid_extents:
            if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
                raise TypeError(
                    f"valid extents must be positive non-bool ints, got {extent!r}."
                )
        _require_nonempty_bytes("payload", self.payload)


@dataclass(frozen=True, slots=True)
class StateHarborGroupIdentity:
    """Canonical immutable identity of one persisted group object."""

    source: StateHarborSourceImage
    group_index: int
    logical_block_indices: tuple[int, ...]
    offload_keys: tuple[OffloadKey, ...]
    logical_token_span: tuple[int, int]
    valid_extents: tuple[int, ...]
    payload_length: int
    payload_digest: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.source, StateHarborSourceImage):
            raise TypeError(
                f"source must be a StateHarborSourceImage, got {self.source!r}."
            )
        _require_uint("group_index", self.group_index)
        if (
            not isinstance(self.logical_block_indices, tuple)
            or not self.logical_block_indices
        ):
            raise TypeError("logical_block_indices must be a nonempty tuple")
        if tuple(sorted(set(self.logical_block_indices))) != self.logical_block_indices:
            raise ValueError("logical_block_indices must be unique and increasing")
        for index in self.logical_block_indices:
            _require_uint("logical block index", index)
        if not isinstance(self.offload_keys, tuple) or not self.offload_keys:
            raise TypeError("offload_keys must be a nonempty tuple")
        if any(
            not isinstance(key, bytes) or len(key) <= 4 for key in self.offload_keys
        ):
            raise TypeError(
                "offload_keys must contain a nonempty hash and group suffix"
            )
        if len(self.logical_block_indices) != len(self.offload_keys):
            raise ValueError("logical block indices and offload keys must align")
        if (
            not isinstance(self.logical_token_span, tuple)
            or len(self.logical_token_span) != 2
        ):
            raise TypeError("logical_token_span must be a (start, end) tuple")
        start, end = self.logical_token_span
        _require_uint("logical_token_span start", start)
        _require_uint("logical_token_span end", end)
        if end <= start:
            raise ValueError("logical_token_span end must be greater than start")
        if not isinstance(self.valid_extents, tuple) or len(self.valid_extents) != len(
            self.offload_keys
        ):
            raise ValueError("valid extents must align with concrete offload keys")
        for extent in self.valid_extents:
            if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
                raise TypeError("valid extents must be positive non-bool ints")
        _require_uint("payload_length", self.payload_length)
        if self.payload_length <= 0:
            raise ValueError("payload_length must be positive")
        if not isinstance(self.payload_digest, bytes) or len(self.payload_digest) != 32:
            raise TypeError("payload_digest must be a SHA256 digest")

    def canonical_bytes(self) -> bytes:
        fields = [
            _frame(self.source.canonical_bytes()),
            _frame(_int_bytes(self.group_index)),
            _frame(_int_bytes(self.logical_token_span[0])),
            _frame(_int_bytes(self.logical_token_span[1])),
            _frame(_int_bytes(self.payload_length)),
            _frame(self.payload_digest),
        ]
        fields.extend(_frame(_int_bytes(index)) for index in self.logical_block_indices)
        fields.append(b"\xff")
        fields.extend(_frame(key) for key in self.offload_keys)
        fields.append(b"\xfe")
        fields.extend(_frame(_int_bytes(extent)) for extent in self.valid_extents)
        return b"".join(fields)

    @property
    def object_key(self) -> bytes:
        return sha256(_GROUP_DOMAIN + self.canonical_bytes()).digest()


@dataclass(frozen=True, slots=True)
class StateHarborImageManifest:
    """Manifest published last so partial group commits remain unreachable."""

    source: StateHarborSourceImage
    groups: tuple[StateHarborGroupIdentity, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source, StateHarborSourceImage):
            raise TypeError(
                f"source must be a StateHarborSourceImage, got {self.source!r}."
            )
        if not isinstance(self.groups, tuple) or not self.groups:
            raise TypeError("groups must be a nonempty tuple")
        indices = tuple(group.group_index for group in self.groups)
        if indices != tuple(sorted(set(indices))):
            raise ValueError("manifest group indices must be unique and increasing")
        if any(group.source != self.source for group in self.groups):
            raise ValueError("every manifest group must name the same source image")

    def canonical_bytes(self) -> bytes:
        return _frame(self.source.canonical_bytes()) + b"".join(
            _frame(group.canonical_bytes()) for group in self.groups
        )

    @property
    def object_key(self) -> bytes:
        return sha256(_MANIFEST_DOMAIN + self.canonical_bytes()).digest()


class StateHarborCommitOutcome(str, Enum):
    COMMITTED = "COMMITTED"
    REFUSED = "REFUSED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class StateHarborWriteLease:
    object_key: bytes
    writer: StateHarborWriterFence
    sequence: int

    def __post_init__(self) -> None:
        _require_nonempty_bytes("object_key", self.object_key)
        if not isinstance(self.writer, StateHarborWriterFence):
            raise TypeError(
                f"writer must be a StateHarborWriterFence, got {self.writer!r}."
            )
        _require_uint("sequence", self.sequence)


class StateHarborBytePlane(Protocol):
    """Physical immutable-object interface implemented by LMCache or a fake."""

    def prepare_write(self, lease: StateHarborWriteLease, byte_length: int) -> bool: ...

    def write(self, lease: StateHarborWriteLease, payload: bytes) -> None: ...

    def commit_write(
        self, lease: StateHarborWriteLease
    ) -> StateHarborCommitOutcome: ...

    def abort_write(self, lease: StateHarborWriteLease) -> None: ...

    def read(self, object_key: bytes) -> bytes | None: ...

    def delete(self, object_key: bytes) -> bool: ...


@dataclass(frozen=True, slots=True)
class StateHarborStoreResult:
    receipt: OwnerBulkTransferReceipt
    manifest: StateHarborImageManifest
    reconciled_object_keys: tuple[bytes, ...] = ()
    residual_object_keys: tuple[bytes, ...] = ()


@dataclass(slots=True)
class StateHarborStoreWork:
    _adapter: StateHarborBytePlaneAdapter
    job: OwnerBulkTransferJob
    manifest: StateHarborImageManifest
    payloads: tuple[bytes, ...]
    writer: StateHarborWriterFence
    _executed: bool = False

    def execute_after_staging(self) -> StateHarborStoreResult:
        """Publish after D2H/staging completion, then admit device reclaim."""

        if self._executed:
            raise StateHarborBytePlaneError("store work was already executed")
        self._executed = True
        return self._adapter._execute_store(self)


@dataclass(slots=True)
class StateHarborRestoreWork:
    _ledger: RequestOwnedBulkOffloadLedger
    job: OwnerBulkTransferJob
    manifest: StateHarborImageManifest
    payloads: tuple[bytes, ...]
    _completed: bool = False

    def complete_after_h2d(
        self, *, success: bool, error: str | None = None
    ) -> OwnerBulkTransferReceipt:
        """Publish the ordinary restore receipt only after exact H2D completion."""

        if self._completed:
            raise StateHarborBytePlaneError("restore work was already completed")
        if not isinstance(success, bool):
            raise TypeError(f"success must be a bool, got {success!r}.")
        if success and error is not None:
            raise ValueError("successful H2D completion must not carry an error")
        if not success and not error:
            raise ValueError("failed H2D completion must carry an error")
        self._completed = True
        receipt = OwnerBulkTransferReceipt.for_job(
            self.job, success=success, error=error
        )
        self._ledger.complete(receipt)
        return receipt


@dataclass(frozen=True, slots=True)
class StateHarborRedockTicket:
    """StateHarbor authority for one released-source to new-owner restore.

    The destination RESERVE receipt is deliberately buffered in this ticket.
    It must not reach the scheduler coordinator until the exact shared image
    has restored and produced a successful destination H2D terminal.
    """

    source: StateHarborSourceImage
    source_manifest_key: bytes
    source_release_receipt: OwnerReceipt
    destination_identity: OwnerOffloadIdentity
    destination_reserve_receipt: OwnerReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.source, StateHarborSourceImage):
            raise TypeError(
                f"source must be a StateHarborSourceImage, got {self.source!r}."
            )
        if (
            not isinstance(self.source_manifest_key, bytes)
            or len(self.source_manifest_key) != 32
        ):
            raise TypeError("source_manifest_key must be a SHA256 object key")
        if not isinstance(self.source_release_receipt, OwnerReceipt):
            raise TypeError("source_release_receipt must be an OwnerReceipt")
        if not isinstance(self.destination_identity, OwnerOffloadIdentity):
            raise TypeError("destination_identity must be an OwnerOffloadIdentity")
        if not isinstance(self.destination_reserve_receipt, OwnerReceipt):
            raise TypeError("destination_reserve_receipt must be an OwnerReceipt")

        released = self.source_release_receipt
        _require_uint("source_release_receipt.command_seq", released.command_seq)
        if (
            released.command_seq == 0
            or not released.accepted
            or not released.released
            or released.error is not None
            or released.runnable_num_tokens is None
            or released.pending_dma != 0
            or released.key.request_id != self.source.request_id
            or released.key.owner_epoch != self.source.source_owner_epoch
            or released.owner_id != self.source.source_owner_rank
        ):
            raise StateHarborBytePlaneError(
                "redock requires an accepted exact source RELEASE receipt"
            )

        destination = self.destination_identity
        reserve = self.destination_reserve_receipt
        _require_uint("destination_reserve_receipt.command_seq", reserve.command_seq)
        if destination.key.request_id != self.source.request_id:
            raise StateHarborBytePlaneError(
                "redock destination request must match the source image"
            )
        if destination.key.owner_epoch <= self.source.source_owner_epoch:
            raise StateHarborBytePlaneError(
                "redock destination must advance the owner epoch"
            )
        if destination.owner_rank == self.source.source_owner_rank:
            raise StateHarborBytePlaneError(
                "redock destination must be a different owner"
            )
        if (
            reserve.command_seq == 0
            or not reserve.accepted
            or reserve.released
            or reserve.error is not None
            or reserve.runnable_num_tokens is None
            or reserve.pending_dma not in (None, 0)
            or reserve.key != destination.key
            or reserve.owner_id != destination.owner_rank
        ):
            raise StateHarborBytePlaneError(
                "redock requires an accepted exact destination RESERVE receipt"
            )


@dataclass(slots=True)
class StateHarborRedockTerminal:
    """Exact physical terminal that may release one buffered RESERVE receipt."""

    ticket: StateHarborRedockTicket
    transfer_receipt: OwnerBulkTransferReceipt
    _admitted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.ticket, StateHarborRedockTicket):
            raise TypeError("ticket must be a StateHarborRedockTicket")
        if not isinstance(self.transfer_receipt, OwnerBulkTransferReceipt):
            raise TypeError("transfer_receipt must be an OwnerBulkTransferReceipt")
        if self.transfer_receipt.identity != self.ticket.destination_identity:
            raise StateHarborBytePlaneError(
                "redock H2D terminal does not name the exact destination"
            )

    def admit_destination_reserve(self) -> OwnerReceipt:
        """Release the scheduler receipt once, and only after successful H2D."""

        if self._admitted:
            raise StateHarborBytePlaneError(
                "destination RESERVE receipt was already admitted"
            )
        if not self.transfer_receipt.success:
            raise StateHarborBytePlaneError(
                "failed redock H2D cannot admit destination RESERVE"
            )
        self._admitted = True
        return self.ticket.destination_reserve_receipt


@dataclass(slots=True)
class StateHarborRedockWork:
    ticket: StateHarborRedockTicket
    restore: StateHarborRestoreWork
    _completed: bool = False

    @property
    def payloads(self) -> tuple[bytes, ...]:
        return self.restore.payloads

    def complete_after_h2d(
        self, *, success: bool, error: str | None = None
    ) -> StateHarborRedockTerminal:
        if self._completed:
            raise StateHarborBytePlaneError("redock work was already completed")
        self._completed = True
        receipt = self.restore.complete_after_h2d(success=success, error=error)
        return StateHarborRedockTerminal(self.ticket, receipt)


@dataclass(frozen=True, slots=True)
class StateHarborReleaseReceipt:
    manifest_key: bytes
    deleted_object_keys: tuple[bytes, ...]


class StateHarborBytePlaneAdapter:
    """Synchronous CPU reference adapter behind one owner offload ledger."""

    def __init__(
        self,
        *,
        ledger: RequestOwnedBulkOffloadLedger,
        backend: StateHarborBytePlane,
    ) -> None:
        if not isinstance(ledger, RequestOwnedBulkOffloadLedger):
            raise TypeError(
                f"ledger must be a RequestOwnedBulkOffloadLedger, got {ledger!r}."
            )
        for method in (
            "prepare_write",
            "write",
            "commit_write",
            "abort_write",
            "read",
            "delete",
        ):
            if not callable(getattr(backend, method, None)):
                raise TypeError(f"backend is missing callable {method}()")
        self.ledger = ledger
        self.backend = backend
        self._write_sequence = 0
        self._released_manifests: set[bytes] = set()

    def begin_store(
        self,
        *,
        plan: OwnerOffloadPlan,
        source: StateHarborSourceImage,
        groups: tuple[StateHarborGroupPayload, ...],
        writer: StateHarborWriterFence,
    ) -> StateHarborStoreWork:
        manifest, payloads = self._manifest_from_plan(plan, source, groups)
        if not isinstance(writer, StateHarborWriterFence):
            raise TypeError(f"writer must be a StateHarborWriterFence, got {writer!r}.")
        job = self.ledger.begin_store(plan)
        return StateHarborStoreWork(self, job, manifest, payloads, writer)

    def prepare_restore(
        self,
        *,
        plan: OwnerOffloadPlan,
        manifest: StateHarborImageManifest,
    ) -> StateHarborRestoreWork:
        """Read and verify every byte before opening an owner H2D job."""

        self._validate_manifest_for_plan(manifest, plan, store_source=False)
        observed_manifest = self.backend.read(manifest.object_key)
        if observed_manifest != manifest.canonical_bytes():
            reason = "missing" if observed_manifest is None else "mismatched"
            raise StateHarborBytePlaneError(
                f"exact source manifest is {reason}; restore refused"
            )
        payloads: list[bytes] = []
        for group in manifest.groups:
            payload = self.backend.read(group.object_key)
            if payload is None:
                raise StateHarborBytePlaneError(
                    f"source group {group.group_index} is missing"
                )
            if len(payload) != group.payload_length or sha256(payload).digest() != (
                group.payload_digest
            ):
                raise StateHarborBytePlaneError(
                    f"source group {group.group_index} length/digest mismatch"
                )
            payloads.append(payload)

        # A backend hit is only physical durability evidence.  begin_restore
        # immediately clears HOT, and only complete_after_h2d may set it again.
        self.ledger.admit_durable_host_image(plan)
        job = self.ledger.begin_restore(plan)
        return StateHarborRestoreWork(self.ledger, job, manifest, tuple(payloads))

    def prepare_redock(
        self,
        *,
        ticket: StateHarborRedockTicket,
        plan: OwnerOffloadPlan,
        manifest: StateHarborImageManifest,
    ) -> StateHarborRedockWork:
        """Verify a new-owner ticket, then open the ordinary physical restore."""

        if not isinstance(ticket, StateHarborRedockTicket):
            raise TypeError("ticket must be a StateHarborRedockTicket")
        if ticket.source != manifest.source:
            raise StateHarborBytePlaneError(
                "redock ticket source does not match the manifest"
            )
        if ticket.source_manifest_key != manifest.object_key:
            raise StateHarborBytePlaneError(
                "redock ticket does not name the exact manifest"
            )
        if ticket.destination_identity != plan.identity:
            raise StateHarborBytePlaneError(
                "redock ticket does not name the exact destination plan"
            )
        restore = self.prepare_restore(plan=plan, manifest=manifest)
        return StateHarborRedockWork(ticket, restore)

    def release(
        self,
        *,
        authority: StateHarborSourceImage,
        manifest: StateHarborImageManifest,
        plan: OwnerOffloadPlan | None = None,
    ) -> StateHarborReleaseReceipt:
        """Delete one exact source generation, removing the manifest first.

        ``authority`` is supplied by StateHarbor's logical lifecycle.  The
        optional source ``plan`` only lets a still-live local ledger forget its
        host keys; release never requires durable device-local block IDs.
        """

        if not isinstance(authority, StateHarborSourceImage):
            raise TypeError(
                f"authority must be a StateHarborSourceImage, got {authority!r}."
            )
        if not isinstance(manifest, StateHarborImageManifest):
            raise TypeError(
                f"manifest must be a StateHarborImageManifest, got {manifest!r}."
            )
        if authority != manifest.source:
            raise StateHarborBytePlaneError(
                "release authority does not name the exact source generation"
            )
        if plan is not None:
            self._validate_manifest_for_plan(manifest, plan, store_source=True)
        if manifest.object_key in self._released_manifests:
            raise StateHarborBytePlaneError("source manifest was already released")
        if self.backend.read(manifest.object_key) != manifest.canonical_bytes():
            raise StateHarborBytePlaneError(
                "exact source manifest is absent or mismatched; release refused"
            )

        ordered_keys = (manifest.object_key,) + tuple(
            group.object_key for group in manifest.groups
        )
        deleted: list[bytes] = []
        for object_key in ordered_keys:
            if (
                not self.backend.delete(object_key)
                or self.backend.read(object_key) is not None
            ):
                raise StateHarborBytePlaneError(
                    "exact release failed after deleting "
                    f"{len(deleted)} object(s); manifest was removed first"
                )
            deleted.append(object_key)
        if plan is not None:
            self.ledger.forget_host_keys(
                tuple(key for group in plan.offload_keys for key in group)
            )
        self._released_manifests.add(manifest.object_key)
        return StateHarborReleaseReceipt(manifest.object_key, tuple(deleted))

    def _execute_store(self, work: StateHarborStoreWork) -> StateHarborStoreResult:
        created: list[bytes] = []
        reconciled: list[bytes] = []
        objects = tuple(
            (group.object_key, payload)
            for group, payload in zip(work.manifest.groups, work.payloads)
        ) + ((work.manifest.object_key, work.manifest.canonical_bytes()),)
        error: str | None = None
        try:
            for object_key, payload in objects:
                was_created, was_reconciled = self._publish_object(
                    object_key, payload, work.writer
                )
                if was_created:
                    created.append(object_key)
                if was_reconciled:
                    reconciled.append(object_key)
        except Exception as exc:  # noqa: BLE001 - normalize physical failures
            error = str(exc) or type(exc).__name__

        if error is None:
            receipt = OwnerBulkTransferReceipt.for_job(work.job, success=True)
            try:
                self.ledger.complete(receipt)
            except Exception as exc:
                residual = self._cleanup_created(created)
                raise StateHarborBytePlaneError(
                    "owner ledger rejected a physically committed image; "
                    f"cleanup left {len(residual)} residual object(s)"
                ) from exc
            return StateHarborStoreResult(
                receipt,
                work.manifest,
                reconciled_object_keys=tuple(reconciled),
            )

        residual = self._cleanup_created(created)
        receipt = OwnerBulkTransferReceipt.for_job(
            work.job,
            success=False,
            error=f"shared byte-plane store failed: {error}",
        )
        self.ledger.complete(receipt)
        return StateHarborStoreResult(
            receipt,
            work.manifest,
            reconciled_object_keys=tuple(reconciled),
            residual_object_keys=residual,
        )

    def _publish_object(
        self,
        object_key: bytes,
        payload: bytes,
        writer: StateHarborWriterFence,
    ) -> tuple[bool, bool]:
        existing = self.backend.read(object_key)
        if existing is not None:
            if existing != payload:
                raise StateHarborBytePlaneError(
                    "immutable object key resolved to different bytes"
                )
            return False, True

        lease = StateHarborWriteLease(object_key, writer, self._write_sequence)
        self._write_sequence += 1
        prepared = False
        try:
            prepared = self.backend.prepare_write(lease, len(payload))
            if not prepared:
                raise StateHarborBytePlaneError("backend refused write preparation")
            self.backend.write(lease, payload)
            try:
                outcome = self.backend.commit_write(lease)
            except Exception as commit_error:
                # An RPC exception does not prove the server failed before its
                # publish point.  Reconcile the immutable key before aborting.
                observed = self.backend.read(object_key)
                if observed == payload:
                    return True, True
                if observed is not None:
                    raise StateHarborBytePlaneError(
                        "commit-error reconciliation found mismatched bytes"
                    ) from commit_error
                raise StateHarborBytePlaneError(
                    "commit error did not reconcile to an exact object"
                ) from commit_error
            if not isinstance(outcome, StateHarborCommitOutcome):
                raise StateHarborBytePlaneError(
                    f"backend returned invalid commit outcome {outcome!r}"
                )
            if outcome is StateHarborCommitOutcome.REFUSED:
                raise StateHarborBytePlaneError("backend refused write commit")

            observed = self.backend.read(object_key)
            if observed == payload:
                return True, outcome is StateHarborCommitOutcome.UNKNOWN
            if observed is None:
                qualifier = (
                    "outcome-unknown"
                    if outcome is StateHarborCommitOutcome.UNKNOWN
                    else "committed"
                )
                raise StateHarborBytePlaneError(
                    f"{qualifier} write did not reconcile to an exact object"
                )
            raise StateHarborBytePlaneError(
                "published object failed exact length/digest reconciliation"
            )
        except Exception:
            if prepared:
                with suppress(Exception):
                    self.backend.abort_write(lease)
            raise

    def _cleanup_created(self, created: list[bytes]) -> tuple[bytes, ...]:
        residual: list[bytes] = []
        # Manifest is appended last and therefore removed first on failure.
        for object_key in reversed(created):
            try:
                deleted = self.backend.delete(object_key)
                absent = self.backend.read(object_key) is None
            except Exception:
                deleted = absent = False
            if not deleted or not absent:
                residual.append(object_key)
        return tuple(residual)

    @staticmethod
    def _manifest_from_plan(
        plan: OwnerOffloadPlan,
        source: StateHarborSourceImage,
        payloads: tuple[StateHarborGroupPayload, ...],
    ) -> tuple[StateHarborImageManifest, tuple[bytes, ...]]:
        if not isinstance(plan, OwnerOffloadPlan):
            raise TypeError(f"plan must be an OwnerOffloadPlan, got {plan!r}.")
        if not isinstance(source, StateHarborSourceImage):
            raise TypeError(f"source must be a StateHarborSourceImage, got {source!r}.")
        identity = plan.identity
        if (
            identity.key.request_id != source.request_id
            or identity.key.owner_epoch != source.source_owner_epoch
            or identity.owner_rank != source.source_owner_rank
            or identity.allocation_generation != source.source_activation_generation
        ):
            raise StateHarborBytePlaneError(
                "store source identity does not match the exact owner plan"
            )
        if not isinstance(payloads, tuple):
            raise TypeError("groups must be a tuple")
        by_index = {payload.group_index: payload for payload in payloads}
        if len(by_index) != len(payloads):
            raise StateHarborBytePlaneError("group payload indices must be unique")
        required = {index for index, keys in enumerate(plan.offload_keys) if keys}
        if set(by_index) != required:
            raise StateHarborBytePlaneError(
                "group payloads must cover every and only concrete plan group"
            )

        groups: list[StateHarborGroupIdentity] = []
        ordered_payloads: list[bytes] = []
        for group_index in sorted(required):
            payload = by_index[group_index]
            if len(payload.valid_extents) != len(plan.offload_keys[group_index]):
                raise StateHarborBytePlaneError(
                    f"group {group_index} valid extents do not match its plan"
                )
            groups.append(
                StateHarborGroupIdentity(
                    source=source,
                    group_index=group_index,
                    logical_block_indices=plan.logical_block_indices[group_index],
                    offload_keys=plan.offload_keys[group_index],
                    logical_token_span=payload.logical_token_span,
                    valid_extents=payload.valid_extents,
                    payload_length=len(payload.payload),
                    payload_digest=sha256(payload.payload).digest(),
                )
            )
            ordered_payloads.append(payload.payload)
        return StateHarborImageManifest(source, tuple(groups)), tuple(ordered_payloads)

    @staticmethod
    def _validate_manifest_for_plan(
        manifest: StateHarborImageManifest,
        plan: OwnerOffloadPlan,
        *,
        store_source: bool,
    ) -> None:
        if not isinstance(manifest, StateHarborImageManifest):
            raise TypeError(
                f"manifest must be a StateHarborImageManifest, got {manifest!r}."
            )
        if not isinstance(plan, OwnerOffloadPlan):
            raise TypeError(f"plan must be an OwnerOffloadPlan, got {plan!r}.")
        if plan.identity.key.request_id != manifest.source.request_id:
            raise StateHarborBytePlaneError(
                "restore destination request does not match the source image"
            )
        if store_source and (
            plan.identity.key.owner_epoch != manifest.source.source_owner_epoch
            or plan.identity.owner_rank != manifest.source.source_owner_rank
            or plan.identity.allocation_generation
            != manifest.source.source_activation_generation
        ):
            raise StateHarborBytePlaneError("plan does not name the source activation")

        expected = {index for index, keys in enumerate(plan.offload_keys) if keys}
        observed = {group.group_index for group in manifest.groups}
        if observed != expected:
            raise StateHarborBytePlaneError(
                "manifest groups do not match the restore plan"
            )
        for group in manifest.groups:
            index = group.group_index
            if (
                group.offload_keys != plan.offload_keys[index]
                or group.logical_block_indices != plan.logical_block_indices[index]
            ):
                raise StateHarborBytePlaneError(
                    f"manifest group {index} does not match the restore mapping"
                )
