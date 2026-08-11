# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Default-off runtime ABI for KV-recovery profiling."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Set
from dataclasses import dataclass
from threading import Lock
from typing import Literal, Protocol

KVRecoveryOperation = Literal["d2h_preserve", "h2d_restore"]
KVRecoveryCapacity = Literal["prepared_transfer", "pending_h2d", "pending_d2h"]
KVRecoveryRequeueReason = Literal[
    "lora_capacity",
    "prefill_throttled",
    "token_budget",
    "encoder_budget",
    "block_capacity",
    "unclassified",
]
KVRecoveryComputeKind = Literal["prefill", "decode"]

MAX_LOGICAL_BLOCKS_PER_SET = 4096
MAX_PREPARED_TRANSFER_ATTEMPTS_PER_PROCESS = 4096
MAX_PENDING_H2D_CONTEXTS_PER_PROCESS = 4096
MAX_PENDING_D2H_CONTEXTS_PER_PROCESS = 4096
MAX_H2D_RECEIPTS_PER_WORKER_STEP = 4096
MAX_RUNTIME_REQUEST_ID_BYTES = 128
MAX_TRANSFER_IDS_PER_WAIT_SET = 4096
KV_RECOVERY_PROFILE_ID = "rlp.kv-recovery/v1alpha1"
KV_RECOVERY_PROFILE_ENABLED_KEY = "kv_recovery_profile_enabled"

_LOWER_HEX_32_RE = re.compile(r"^[0-9a-f]{32}$")
_LOWER_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_TRANSFER_ID_RE = re.compile(r"^([0-9a-f]{32}):t:(0|[1-9][0-9]{0,19})$")
_BASE_EVENT_ID_RE = re.compile(r"^[0-9a-f]{32}:e:(0|[1-9][0-9]{0,19})$")
_PROFILE_RECORD_ID_RE = re.compile(r"^[0-9a-f]{32}:k:(0|[1-9][0-9]{0,19})$")
_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1


def _is_uint32(value: object) -> bool:
    return type(value) is int and 0 <= value <= _UINT32_MAX


def _is_uint64(value: object) -> bool:
    return type(value) is int and 0 <= value <= _UINT64_MAX


def _is_positive_uint64(value: object) -> bool:
    return type(value) is int and 0 < value <= _UINT64_MAX


def _is_lower_hex32(value: object) -> bool:
    return type(value) is str and _LOWER_HEX_32_RE.fullmatch(value) is not None


def _is_lower_hex64(value: object) -> bool:
    return type(value) is str and _LOWER_HEX_64_RE.fullmatch(value) is not None


def _matches_scoped_uint64(value: object, pattern: re.Pattern[str]) -> bool:
    if type(value) is not str:
        return False
    match = pattern.fullmatch(value)
    return match is not None and int(match.group(1)) <= _UINT64_MAX


def _matches_transfer_id(value: object) -> bool:
    if type(value) is not str:
        return False
    match = _TRANSFER_ID_RE.fullmatch(value)
    return match is not None and int(match.group(2)) <= _UINT64_MAX


def _require_printable_ascii(value: str, field_name: str, max_bytes: int) -> None:
    if type(value) is not str:
        raise ValueError(f"{field_name} must be a string")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} must be ASCII") from exc
    if not encoded or len(encoded) > max_bytes or not value.isprintable():
        raise ValueError(
            f"{field_name} must be nonempty printable ASCII within {max_bytes} bytes"
        )


@dataclass(frozen=True, slots=True, order=True)
class KVRecoveryBlockCoordinate:
    """A logical KV-block coordinate before an opaque ID is assigned."""

    group_index: int
    logical_ordinal: int

    def __post_init__(self) -> None:
        if not _is_uint32(self.group_index) or not _is_uint64(self.logical_ordinal):
            raise ValueError(
                "logical block group must be uint32 and ordinal must be uint64"
            )


@dataclass(frozen=True, slots=True, order=True)
class KVRecoveryLogicalBlock:
    """A canonical run-local logical KV-block identity."""

    group_index: int
    logical_ordinal: int
    logical_block_id: str

    def __post_init__(self) -> None:
        KVRecoveryBlockCoordinate(self.group_index, self.logical_ordinal)
        if not _is_lower_hex32(self.logical_block_id):
            raise ValueError("logical_block_id must be 32 lowercase hex characters")

    @property
    def coordinate(self) -> KVRecoveryBlockCoordinate:
        return KVRecoveryBlockCoordinate(self.group_index, self.logical_ordinal)


@dataclass(frozen=True, slots=True)
class KVRecoveryIdentity:
    """Request and recovery-episode identity propagated to transfer workers."""

    run_id: str
    trace_id: str
    engine_lifecycle_id: str
    runtime_request_id: str
    recovery_epoch: int | None
    episode_id: str | None
    base_preempted_event_id: str | None
    preempt_profile_record_id: str | None

    def __post_init__(self) -> None:
        if not _is_lower_hex32(self.run_id):
            raise ValueError("run_id must be 32 lowercase hex characters")
        if not _is_lower_hex32(self.trace_id):
            raise ValueError("trace_id must be 32 lowercase hex characters")
        if self.engine_lifecycle_id != f"{self.trace_id}:e:0":
            raise ValueError("engine_lifecycle_id must be trace_id:e:0")
        _require_printable_ascii(
            self.runtime_request_id,
            "runtime_request_id",
            MAX_RUNTIME_REQUEST_ID_BYTES,
        )
        if self.recovery_epoch is None:
            if (
                self.episode_id is not None
                or self.base_preempted_event_id is not None
                or self.preempt_profile_record_id is not None
            ):
                raise ValueError("non-recovery identity cannot carry episode fields")
            return
        if not _is_positive_uint64(self.recovery_epoch):
            raise ValueError("recovery_epoch must be a positive uint64")
        expected_episode_id = f"{self.engine_lifecycle_id}:k:{self.recovery_epoch}"
        if self.episode_id != expected_episode_id:
            raise ValueError("episode_id does not match the recovery epoch")
        if self.base_preempted_event_id is None:
            raise ValueError("recovery identity requires base_preempted_event_id")
        if not _matches_scoped_uint64(self.base_preempted_event_id, _BASE_EVENT_ID_RE):
            raise ValueError(
                "base_preempted_event_id must use process_uuid:e:record_seq"
            )
        if self.preempt_profile_record_id is None or not _matches_scoped_uint64(
            self.preempt_profile_record_id, _PROFILE_RECORD_ID_RE
        ):
            raise ValueError(
                "preempt_profile_record_id must use process_uuid:k:record_seq"
            )


def canonical_block_set_id(
    identity: KVRecoveryIdentity,
    logical_blocks: tuple[KVRecoveryLogicalBlock, ...],
) -> str:
    """Compute the profile's canonical logical block-set digest."""
    prefix = (
        f"{KV_RECOVERY_PROFILE_ID}\0{identity.run_id}\0{identity.engine_lifecycle_id}\0"
    )
    rows = "".join(
        f"{block.group_index}:{block.logical_ordinal}:{block.logical_block_id}\n"
        for block in logical_blocks
    )
    return hashlib.sha256(f"{prefix}{rows}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class KVRecoveryTransferContext:
    """Immutable scheduler-to-worker identity sidecar for one transfer."""

    identity: KVRecoveryIdentity
    operation: KVRecoveryOperation
    block_set_id: str
    logical_blocks: tuple[KVRecoveryLogicalBlock, ...]

    def __post_init__(self) -> None:
        if not self.logical_blocks:
            raise ValueError("logical block set must be nonempty")
        if len(self.logical_blocks) > MAX_LOGICAL_BLOCKS_PER_SET:
            raise ValueError("logical block set exceeds the frozen bound")
        if tuple(sorted(self.logical_blocks)) != self.logical_blocks:
            raise ValueError("logical blocks must use canonical coordinate order")
        coordinates = [block.coordinate for block in self.logical_blocks]
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("logical block coordinates must be unique")
        logical_ids = [block.logical_block_id for block in self.logical_blocks]
        if len(set(logical_ids)) != len(logical_ids):
            raise ValueError("logical block IDs must be unique")
        if self.operation == "h2d_restore":
            if self.identity.recovery_epoch is not None:
                expected = (
                    f"{self.identity.engine_lifecycle_id}:k:"
                    f"{self.identity.recovery_epoch}"
                )
                if self.identity.episode_id != expected:
                    raise ValueError("episode_id does not match the recovery epoch")
                if self.identity.base_preempted_event_id is None:
                    raise ValueError(
                        "episode H2D restore requires base_preempted_event_id"
                    )
            # An unassociated H2D (recovery_epoch=None) is a block-level
            # tiering migration of a still-running request; it carries no
            # recovery episode and is valid transfer evidence.
        elif self.operation == "d2h_preserve":
            if self.identity.recovery_epoch is not None:
                raise ValueError("proactive D2H preserve cannot carry an episode")
        else:
            raise ValueError(f"unsupported recovery operation: {self.operation}")
        if not _is_lower_hex64(self.block_set_id):
            raise ValueError("block_set_id must be lowercase SHA-256")
        if self.block_set_id != canonical_block_set_id(
            self.identity, self.logical_blocks
        ):
            raise ValueError("block_set_id does not match its canonical rows")

    @property
    def coordinates(self) -> tuple[KVRecoveryBlockCoordinate, ...]:
        return tuple(block.coordinate for block in self.logical_blocks)


@dataclass(frozen=True, slots=True)
class KVRecoveryTransferAttempt:
    """Worker-owned transfer identity allocated before backend submission."""

    connector_job_id: int
    transfer_id: str
    context: KVRecoveryTransferContext

    def __post_init__(self) -> None:
        if not _is_uint64(self.connector_job_id):
            raise ValueError("connector_job_id must be a uint64")
        if not _matches_transfer_id(self.transfer_id):
            raise ValueError("transfer_id must use process_uuid:t:transfer_seq")


@dataclass(frozen=True, slots=True)
class KVRecoveryH2DReceipt:
    """Bounded worker-to-scheduler receipt for a successful H2D transfer."""

    connector_job_id: int
    transfer_id: str
    identity: KVRecoveryIdentity
    block_set_id: str
    process_uuid: str
    rank: int
    world_size: int
    clock_domain_id: str
    communication_done_event_id: str
    restore_done_profile_record_id: str
    timestamp_ns: int
    bytes_moved: int

    def __post_init__(self) -> None:
        if not _is_uint64(self.connector_job_id):
            raise ValueError("connector_job_id must be a uint64")
        if not _matches_transfer_id(self.transfer_id):
            raise ValueError("receipt transfer_id is invalid")
        if not _is_lower_hex32(self.process_uuid):
            raise ValueError("process_uuid must be 32 lowercase hex characters")
        if not self.transfer_id.startswith(f"{self.process_uuid}:t:"):
            raise ValueError("receipt transfer_id must belong to process_uuid")
        if (
            type(self.rank) is not int
            or type(self.world_size) is not int
            or self.rank != 0
            or self.world_size != 1
        ):
            raise ValueError("KV-recovery receipts require rank=0, world_size=1")
        if not _is_lower_hex32(self.clock_domain_id):
            raise ValueError("clock_domain_id must be 32 lowercase hex characters")
        if not _matches_scoped_uint64(
            self.communication_done_event_id, _BASE_EVENT_ID_RE
        ) or not self.communication_done_event_id.startswith(f"{self.process_uuid}:e:"):
            raise ValueError(
                "communication_done_event_id must use process_uuid:e:record_seq"
            )
        if not _matches_scoped_uint64(
            self.restore_done_profile_record_id, _PROFILE_RECORD_ID_RE
        ) or not self.restore_done_profile_record_id.startswith(
            f"{self.process_uuid}:k:"
        ):
            raise ValueError(
                "restore_done_profile_record_id must use process_uuid:k:record_seq"
            )
        if not _is_uint64(self.timestamp_ns) or not _is_positive_uint64(
            self.bytes_moved
        ):
            raise ValueError("receipt timestamp/bytes are invalid")
        if not _is_lower_hex64(self.block_set_id):
            raise ValueError("receipt block_set_id must be lowercase SHA-256")
        if self.identity.recovery_epoch is None:
            raise ValueError("H2D receipt requires a recovery episode")


@dataclass(frozen=True, slots=True)
class KVRecoveryComputeContext:
    """Scheduler-to-worker sidecar for the first recovered compute."""

    identity: KVRecoveryIdentity
    transfer_id: str
    block_set_id: str
    bytes_moved: int
    admission_profile_record_id: str
    compute_kind: KVRecoveryComputeKind
    base_phase_start_event_id: str

    def __post_init__(self) -> None:
        if self.identity.recovery_epoch is None:
            raise ValueError("compute context requires a recovery episode")
        if not _matches_transfer_id(self.transfer_id):
            raise ValueError("compute context transfer_id is invalid")
        if not _is_lower_hex64(self.block_set_id):
            raise ValueError("compute context block_set_id must be lowercase SHA-256")
        if not _is_positive_uint64(self.bytes_moved):
            raise ValueError("compute context bytes_moved must be positive")
        if not _matches_scoped_uint64(
            self.admission_profile_record_id, _PROFILE_RECORD_ID_RE
        ):
            raise ValueError(
                "admission_profile_record_id must use process_uuid:k:record_seq"
            )
        if self.compute_kind not in {"prefill", "decode"}:
            raise ValueError("compute context kind must be prefill or decode")
        if not _matches_scoped_uint64(
            self.base_phase_start_event_id, _BASE_EVENT_ID_RE
        ):
            raise ValueError(
                "base_phase_start_event_id must use process_uuid:e:record_seq"
            )


@dataclass(frozen=True, slots=True)
class KVRecoveryWaitMembership:
    """Resolved process-scoped membership frozen before a worker wait."""

    transfer_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.transfer_ids:
            raise ValueError("wait transfer membership must be nonempty")
        if len(self.transfer_ids) > MAX_TRANSFER_IDS_PER_WAIT_SET:
            raise ValueError("wait transfer membership exceeds the frozen bound")
        if tuple(sorted(set(self.transfer_ids))) != self.transfer_ids:
            raise ValueError("wait transfer IDs must be sorted and unique")
        if any(not _matches_transfer_id(value) for value in self.transfer_ids):
            raise ValueError("wait transfer membership contains an invalid ID")


@dataclass(frozen=True, slots=True)
class KVRecoveryWaitAttempt:
    """Wait-entry point committed only after the worker wait returns."""

    membership: KVRecoveryWaitMembership
    entry_timestamp_ns: int

    def __post_init__(self) -> None:
        if not _is_uint64(self.entry_timestamp_ns):
            raise ValueError("wait-entry timestamp must be a uint64")

    @property
    def transfer_ids(self) -> tuple[str, ...]:
        return self.membership.transfer_ids


class KVRecoverySchedulerObserver(Protocol):
    """Scheduler-side identity and receipt observer supplied by a plugin."""

    def request_started(self, runtime_request_id: str) -> None: ...

    def request_scheduled(
        self,
        runtime_request_id: str,
        compute_kind: KVRecoveryComputeKind,
        scheduled_tokens: int,
        prompt_tokens_total: int,
        prompt_tokens_cached: int,
    ) -> None: ...

    def request_preempted(
        self,
        runtime_request_id: str,
        recovery_epoch: int,
    ) -> str | None: ...

    def prepare_transfer_context(
        self,
        runtime_request_id: str,
        operation: KVRecoveryOperation,
        coordinates: tuple[KVRecoveryBlockCoordinate, ...],
    ) -> KVRecoveryTransferContext | None: ...

    def consume_h2d_receipts(
        self,
        receipts: tuple[KVRecoveryH2DReceipt, ...],
        receipt_capacity_exhausted: bool,
    ) -> None: ...

    def request_admission_started(
        self,
        runtime_request_id: str,
        recovery_epoch: int,
    ) -> None: ...

    def request_requeued(
        self,
        runtime_request_id: str,
        recovery_epoch: int,
        reason: KVRecoveryRequeueReason,
    ) -> None: ...

    def request_admitted(
        self,
        runtime_request_id: str,
        recovery_epoch: int,
        compute_kind: KVRecoveryComputeKind,
        prompt_tokens_total: int,
        prompt_tokens_cached: int,
    ) -> KVRecoveryComputeContext | None: ...

    def request_terminal(self, runtime_request_id: str) -> None: ...

    def reset(self, stale_job_threshold: int) -> None: ...

    def close(self) -> None: ...


class KVRecoveryWorkerObserver(Protocol):
    """Worker-side transfer observer supplied by a plugin."""

    def begin_transfer(
        self,
        connector_job_id: int,
        context: KVRecoveryTransferContext,
    ) -> KVRecoveryTransferAttempt | None: ...

    def transfer_submitted(
        self,
        attempt: KVRecoveryTransferAttempt,
        timestamp_ns: int,
    ) -> None: ...

    def transfer_not_submitted(
        self,
        attempt: KVRecoveryTransferAttempt,
    ) -> None: ...

    def transfer_completed(
        self,
        connector_job_id: int,
        timestamp_ns: int,
        success: bool,
        bytes_moved: int | None,
        device_duration_ns: int | None,
    ) -> KVRecoveryH2DReceipt | None: ...

    def prepare_wait(
        self,
        connector_job_ids: frozenset[int],
    ) -> KVRecoveryWaitMembership | None: ...

    def invalidate_transfers(
        self,
        connector_job_ids: Set[int],
    ) -> None: ...

    def wait_completed(self, attempt: KVRecoveryWaitAttempt) -> None: ...

    def h2d_receipt_capacity_exhausted(
        self,
        receipt: KVRecoveryH2DReceipt,
        loss_reason: Literal["serialization_failure"],
    ) -> None: ...

    def first_compute(
        self,
        context: KVRecoveryComputeContext,
        timestamp_ns: int,
    ) -> None: ...

    def first_compute_not_observed(
        self,
        context: KVRecoveryComputeContext,
    ) -> None: ...

    def close(self) -> None: ...


class KVRecoveryWorkerEvidenceSink(Protocol):
    """Nonblocking evidence sink used by the bounded worker observer."""

    def transfer_submitted(
        self,
        attempt: KVRecoveryTransferAttempt,
        timestamp_ns: int,
    ) -> None: ...

    def transfer_not_submitted(
        self,
        attempt: KVRecoveryTransferAttempt,
    ) -> None: ...

    def transfer_completed(
        self,
        attempt: KVRecoveryTransferAttempt,
        submit_timestamp_ns: int,
        timestamp_ns: int,
        success: bool,
        bytes_moved: int | None,
        device_duration_ns: int | None,
    ) -> KVRecoveryH2DReceipt | None: ...

    def transfer_capacity_exhausted(
        self,
        attempt: KVRecoveryTransferAttempt,
        capacity: KVRecoveryCapacity,
        timestamp_ns: int | None,
        loss_reason: Literal["serialization_failure"],
    ) -> None: ...

    def evidence_failure(
        self,
        reason: str,
        connector_job_ids: tuple[int, ...],
        transfer_ids: tuple[str, ...],
        timestamp_ns: int | None,
    ) -> None: ...

    def wait_completed(self, attempt: KVRecoveryWaitAttempt) -> None: ...

    def h2d_receipt_capacity_exhausted(
        self,
        receipt: KVRecoveryH2DReceipt,
        loss_reason: Literal["serialization_failure"],
    ) -> None: ...

    def first_compute(
        self,
        context: KVRecoveryComputeContext,
        timestamp_ns: int,
    ) -> None: ...

    def close(
        self,
        open_attempts: tuple[KVRecoveryTransferAttempt, ...],
        evidence_disabled: bool,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _PendingKVRecoveryTransfer:
    attempt: KVRecoveryTransferAttempt
    submit_timestamp_ns: int


class BoundedKVRecoveryWorkerObserver:
    """Reference worker observer with the frozen H2D pending-table bound."""

    def __init__(
        self,
        process_uuid: str,
        run_id: str,
        clock_domain_id: str,
        sink: KVRecoveryWorkerEvidenceSink,
    ) -> None:
        if not _is_lower_hex32(process_uuid):
            raise ValueError("process_uuid must be 32 lowercase hex characters")
        if not _is_lower_hex32(run_id):
            raise ValueError("run_id must be 32 lowercase hex characters")
        if not _is_lower_hex32(clock_domain_id):
            raise ValueError("clock_domain_id must be 32 lowercase hex characters")
        self._process_uuid = process_uuid
        self._run_id = run_id
        self._clock_domain_id = clock_domain_id
        self._sink = sink
        self._transfer_seq = 0
        self._prepared_by_job_id: dict[int, KVRecoveryTransferAttempt] = {}
        self._pending_h2d_by_transfer_id: dict[str, _PendingKVRecoveryTransfer] = {}
        self._h2d_transfer_id_by_connector_job_id: dict[int, str] = {}
        self._pending_d2h_by_job_id: dict[int, _PendingKVRecoveryTransfer] = {}
        # The first state-capacity failure makes formal evidence fail closed.
        # This single bit replaces an otherwise unbounded dropped-ID table;
        # serving continues and later untracked completions are ignored.
        self._evidence_disabled = False
        self._closed = False
        # Serialize public calls through sink delivery. The sink contract is
        # nonblocking and non-reentrant; this makes close a single winner and
        # prevents a producer callback from occurring after sink.close().
        self._lifecycle_lock = Lock()
        self._state_lock = Lock()

    @property
    def prepared_transfer_count(self) -> int:
        with self._state_lock:
            return len(self._prepared_by_job_id)

    @property
    def pending_h2d_count(self) -> int:
        with self._state_lock:
            return len(self._pending_h2d_by_transfer_id)

    @property
    def pending_d2h_count(self) -> int:
        with self._state_lock:
            return len(self._pending_d2h_by_job_id)

    @property
    def transfer_seq(self) -> int:
        with self._state_lock:
            return self._transfer_seq

    @property
    def evidence_disabled(self) -> bool:
        with self._state_lock:
            return self._evidence_disabled

    def begin_transfer(
        self,
        connector_job_id: int,
        context: KVRecoveryTransferContext,
    ) -> KVRecoveryTransferAttempt | None:
        with self._lifecycle_lock:
            capacity_exhausted = False
            failure_reason: str | None = None
            with self._state_lock:
                if self._closed:
                    return None
                if not _is_uint64(connector_job_id) or not isinstance(
                    context, KVRecoveryTransferContext
                ):
                    failure_reason = "invalid_transfer_identity"
                    attempt = None
                elif self._transfer_seq > _UINT64_MAX:
                    if not self._evidence_disabled:
                        self._evidence_disabled = True
                        failure_reason = "transfer_seq_exhausted"
                    attempt = None
                else:
                    transfer_seq = self._transfer_seq
                    self._transfer_seq += 1
                    attempt = KVRecoveryTransferAttempt(
                        connector_job_id=connector_job_id,
                        transfer_id=f"{self._process_uuid}:t:{transfer_seq}",
                        context=context,
                    )
                    if self._evidence_disabled:
                        return None
                    duplicate = (
                        connector_job_id in self._prepared_by_job_id
                        or connector_job_id in self._h2d_transfer_id_by_connector_job_id
                        or connector_job_id in self._pending_d2h_by_job_id
                    )
                    if context.identity.run_id != self._run_id:
                        failure_reason = "foreign_run_transfer"
                    elif duplicate:
                        failure_reason = "duplicate_connector_job_id"
                    elif len(self._prepared_by_job_id) >= (
                        MAX_PREPARED_TRANSFER_ATTEMPTS_PER_PROCESS
                    ):
                        self._evidence_disabled = True
                        capacity_exhausted = True
                    else:
                        self._prepared_by_job_id[connector_job_id] = attempt
                        return attempt
            if capacity_exhausted and attempt is not None:
                self._sink.transfer_capacity_exhausted(
                    attempt=attempt,
                    capacity="prepared_transfer",
                    timestamp_ns=None,
                    loss_reason="serialization_failure",
                )
            elif failure_reason is not None:
                self._sink.evidence_failure(
                    reason=failure_reason,
                    connector_job_ids=(
                        (connector_job_id,) if _is_uint64(connector_job_id) else ()
                    ),
                    transfer_ids=(attempt.transfer_id,) if attempt is not None else (),
                    timestamp_ns=None,
                )
            return None

    def transfer_not_submitted(self, attempt: KVRecoveryTransferAttempt) -> None:
        with self._lifecycle_lock:
            with self._state_lock:
                if self._closed:
                    return
                prepared = self._prepared_by_job_id.pop(attempt.connector_job_id, None)
                evidence_disabled = self._evidence_disabled
            if prepared == attempt:
                self._sink.transfer_not_submitted(attempt)
            elif not evidence_disabled:
                self._sink.evidence_failure(
                    reason="unknown_rejected_submission",
                    connector_job_ids=(attempt.connector_job_id,),
                    transfer_ids=(attempt.transfer_id,),
                    timestamp_ns=None,
                )

    def transfer_submitted(
        self,
        attempt: KVRecoveryTransferAttempt,
        timestamp_ns: int,
    ) -> None:
        with self._lifecycle_lock:
            capacity_exhausted: KVRecoveryCapacity | None = None
            failure_reason: str | None = None
            with self._state_lock:
                if self._closed:
                    return
                prepared = self._prepared_by_job_id.pop(attempt.connector_job_id, None)
                if prepared != attempt:
                    if not self._evidence_disabled:
                        failure_reason = "unknown_accepted_submission"
                elif self._evidence_disabled:
                    return
                elif not _is_uint64(timestamp_ns):
                    self._evidence_disabled = True
                    failure_reason = "invalid_submit_timestamp"
                elif (
                    attempt.context.operation == "h2d_restore"
                    and len(self._pending_h2d_by_transfer_id)
                    >= MAX_PENDING_H2D_CONTEXTS_PER_PROCESS
                ):
                    self._evidence_disabled = True
                    capacity_exhausted = "pending_h2d"
                elif (
                    attempt.context.operation == "d2h_preserve"
                    and len(self._pending_d2h_by_job_id)
                    >= MAX_PENDING_D2H_CONTEXTS_PER_PROCESS
                ):
                    self._evidence_disabled = True
                    capacity_exhausted = "pending_d2h"
                else:
                    pending = _PendingKVRecoveryTransfer(attempt, timestamp_ns)
                    if attempt.context.operation == "h2d_restore":
                        self._pending_h2d_by_transfer_id[attempt.transfer_id] = pending
                        self._h2d_transfer_id_by_connector_job_id[
                            attempt.connector_job_id
                        ] = attempt.transfer_id
                    else:
                        self._pending_d2h_by_job_id[attempt.connector_job_id] = pending
            if capacity_exhausted is not None:
                self._sink.transfer_capacity_exhausted(
                    attempt=attempt,
                    capacity=capacity_exhausted,
                    timestamp_ns=timestamp_ns,
                    loss_reason="serialization_failure",
                )
            elif failure_reason is not None:
                self._sink.evidence_failure(
                    reason=failure_reason,
                    connector_job_ids=(attempt.connector_job_id,),
                    transfer_ids=(attempt.transfer_id,),
                    timestamp_ns=timestamp_ns,
                )
            elif prepared == attempt:
                self._sink.transfer_submitted(attempt, timestamp_ns)

    def transfer_completed(
        self,
        connector_job_id: int,
        timestamp_ns: int,
        success: bool,
        bytes_moved: int | None,
        device_duration_ns: int | None,
    ) -> KVRecoveryH2DReceipt | None:
        with self._lifecycle_lock:
            with self._state_lock:
                if self._closed:
                    return None
                valid_measurement = (
                    _is_uint64(connector_job_id)
                    and _is_uint64(timestamp_ns)
                    and type(success) is bool
                    and (bytes_moved is None or _is_positive_uint64(bytes_moved))
                    and (
                        device_duration_ns is None
                        or _is_positive_uint64(device_duration_ns)
                    )
                )
                if not valid_measurement:
                    evidence_disabled = self._evidence_disabled
                    pending = None
                else:
                    transfer_id = self._h2d_transfer_id_by_connector_job_id.pop(
                        connector_job_id, None
                    )
                    if transfer_id is not None:
                        pending = self._pending_h2d_by_transfer_id.pop(
                            transfer_id, None
                        )
                    else:
                        pending = self._pending_d2h_by_job_id.pop(
                            connector_job_id, None
                        )
                    evidence_disabled = self._evidence_disabled
            if not valid_measurement:
                if not evidence_disabled:
                    self._sink.evidence_failure(
                        reason="invalid_completion_measurement",
                        connector_job_ids=(
                            (connector_job_id,) if _is_uint64(connector_job_id) else ()
                        ),
                        transfer_ids=(),
                        timestamp_ns=(
                            timestamp_ns if _is_uint64(timestamp_ns) else None
                        ),
                    )
                return None
            if pending is None:
                if not evidence_disabled:
                    self._sink.evidence_failure(
                        reason="unknown_completion",
                        connector_job_ids=(connector_job_id,),
                        transfer_ids=(),
                        timestamp_ns=timestamp_ns,
                    )
                return None
            if evidence_disabled:
                return None
            if (
                timestamp_ns < pending.submit_timestamp_ns
                or (
                    pending.attempt.context.operation == "d2h_preserve"
                    and timestamp_ns <= pending.submit_timestamp_ns
                )
                or (success and bytes_moved is None)
            ):
                self._sink.evidence_failure(
                    reason="invalid_completion_measurement",
                    connector_job_ids=(connector_job_id,),
                    transfer_ids=(pending.attempt.transfer_id,),
                    timestamp_ns=timestamp_ns,
                )
                return None
            receipt = self._sink.transfer_completed(
                attempt=pending.attempt,
                submit_timestamp_ns=pending.submit_timestamp_ns,
                timestamp_ns=timestamp_ns,
                success=success,
                bytes_moved=bytes_moved,
                device_duration_ns=device_duration_ns,
            )
            context = pending.attempt.context
            if context.operation == "d2h_preserve":
                if receipt is not None:
                    self._sink.evidence_failure(
                        reason="unexpected_d2h_receipt",
                        connector_job_ids=(connector_job_id,),
                        transfer_ids=(pending.attempt.transfer_id,),
                        timestamp_ns=timestamp_ns,
                    )
                return None
            if receipt is None:
                return None
            if not isinstance(receipt, KVRecoveryH2DReceipt):
                self._sink.evidence_failure(
                    reason="invalid_h2d_receipt",
                    connector_job_ids=(connector_job_id,),
                    transfer_ids=(pending.attempt.transfer_id,),
                    timestamp_ns=timestamp_ns,
                )
                return None
            if (
                not success
                or bytes_moved is None
                or bytes_moved <= 0
                or receipt.connector_job_id != connector_job_id
                or receipt.transfer_id != pending.attempt.transfer_id
                or receipt.identity != context.identity
                or receipt.block_set_id != context.block_set_id
                or receipt.process_uuid != self._process_uuid
                or receipt.clock_domain_id != self._clock_domain_id
                or receipt.rank != 0
                or receipt.world_size != 1
                or receipt.timestamp_ns != timestamp_ns
                or receipt.bytes_moved != bytes_moved
                or receipt.timestamp_ns <= pending.submit_timestamp_ns
            ):
                self._sink.evidence_failure(
                    reason="invalid_h2d_receipt",
                    connector_job_ids=(connector_job_id,),
                    transfer_ids=(pending.attempt.transfer_id,),
                    timestamp_ns=timestamp_ns,
                )
                return None
            return receipt

    def prepare_wait(
        self,
        connector_job_ids: frozenset[int],
    ) -> KVRecoveryWaitMembership | None:
        with self._lifecycle_lock:
            with self._state_lock:
                if self._closed or self._evidence_disabled:
                    return None
            if (
                not connector_job_ids
                or len(connector_job_ids) > MAX_TRANSFER_IDS_PER_WAIT_SET
            ):
                self._sink.evidence_failure(
                    reason="invalid_wait_membership",
                    connector_job_ids=(),
                    transfer_ids=(),
                    timestamp_ns=None,
                )
                return None
            valid_connector_job_ids = all(
                _is_uint64(job_id) for job_id in connector_job_ids
            )
            if not valid_connector_job_ids:
                self._sink.evidence_failure(
                    reason="invalid_wait_membership",
                    connector_job_ids=(),
                    transfer_ids=(),
                    timestamp_ns=None,
                )
                return None
            with self._state_lock:
                pending: list[_PendingKVRecoveryTransfer | None] = []
                for job_id in sorted(connector_job_ids):
                    transfer_id = self._h2d_transfer_id_by_connector_job_id.get(job_id)
                    if transfer_id is not None:
                        entry = self._pending_h2d_by_transfer_id.get(transfer_id)
                    else:
                        entry = self._pending_d2h_by_job_id.get(job_id)
                    pending.append(entry)
            if any(entry is None for entry in pending):
                self._sink.evidence_failure(
                    reason="invalid_wait_membership",
                    connector_job_ids=tuple(sorted(connector_job_ids)),
                    transfer_ids=(),
                    timestamp_ns=None,
                )
                return None
            if any(
                entry is not None
                and (
                    entry.attempt.context.identity.run_id != self._run_id
                    or not entry.attempt.transfer_id.startswith(
                        f"{self._process_uuid}:t:"
                    )
                )
                for entry in pending
            ):
                self._sink.evidence_failure(
                    reason="foreign_wait_membership",
                    connector_job_ids=tuple(sorted(connector_job_ids)),
                    transfer_ids=(),
                    timestamp_ns=None,
                )
                return None
            return KVRecoveryWaitMembership(
                tuple(
                    sorted(
                        entry.attempt.transfer_id
                        for entry in pending
                        if entry is not None
                    )
                )
            )

    def invalidate_transfers(self, connector_job_ids: Set[int]) -> None:
        """Consume bounded contexts named by an explicit discard handoff."""
        if not connector_job_ids:
            return
        with self._lifecycle_lock:
            with self._state_lock:
                if self._closed:
                    return
                invalidated: list[KVRecoveryTransferAttempt] = []
                for connector_job_id in tuple(self._prepared_by_job_id):
                    if connector_job_id in connector_job_ids:
                        invalidated.append(
                            self._prepared_by_job_id.pop(connector_job_id)
                        )
                for connector_job_id, transfer_id in tuple(
                    self._h2d_transfer_id_by_connector_job_id.items()
                ):
                    if connector_job_id not in connector_job_ids:
                        continue
                    del self._h2d_transfer_id_by_connector_job_id[connector_job_id]
                    pending_h2d = self._pending_h2d_by_transfer_id.pop(
                        transfer_id, None
                    )
                    if pending_h2d is not None:
                        invalidated.append(pending_h2d.attempt)
                for connector_job_id in tuple(self._pending_d2h_by_job_id):
                    if connector_job_id in connector_job_ids:
                        invalidated.append(
                            self._pending_d2h_by_job_id.pop(connector_job_id).attempt
                        )
                # A scheduler discard handoff is an expected lifecycle
                # transition (preemption, terminal, or cache reset), not a
                # state-capacity failure. The invalidated contexts fail closed
                # individually below, but the observer must remain usable so
                # later H2D restore evidence (the recovery that follows a
                # preemption) is still captured.
            if invalidated:
                invalidated.sort(key=lambda attempt: attempt.connector_job_id)
                self._sink.evidence_failure(
                    reason="connector_flush_invalidation",
                    connector_job_ids=tuple(
                        attempt.connector_job_id for attempt in invalidated
                    ),
                    transfer_ids=tuple(attempt.transfer_id for attempt in invalidated),
                    timestamp_ns=None,
                )

    def wait_completed(self, attempt: KVRecoveryWaitAttempt) -> None:
        with self._lifecycle_lock:
            with self._state_lock:
                if self._closed or self._evidence_disabled:
                    return
            self._sink.wait_completed(attempt)

    def h2d_receipt_capacity_exhausted(
        self,
        receipt: KVRecoveryH2DReceipt,
        loss_reason: Literal["serialization_failure"],
    ) -> None:
        with self._lifecycle_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._evidence_disabled = True
            self._sink.h2d_receipt_capacity_exhausted(receipt, loss_reason)

    def first_compute(
        self,
        context: KVRecoveryComputeContext,
        timestamp_ns: int,
    ) -> None:
        with self._lifecycle_lock:
            with self._state_lock:
                if self._closed or self._evidence_disabled:
                    return
                valid = (
                    isinstance(context, KVRecoveryComputeContext)
                    and context.identity.run_id == self._run_id
                    and _is_uint64(timestamp_ns)
                )
                if not valid:
                    self._evidence_disabled = True
            if valid:
                self._sink.first_compute(context, timestamp_ns)
            else:
                self._sink.evidence_failure(
                    reason="invalid_first_compute_observation",
                    connector_job_ids=(),
                    transfer_ids=(
                        (context.transfer_id,)
                        if isinstance(context, KVRecoveryComputeContext)
                        else ()
                    ),
                    timestamp_ns=timestamp_ns if _is_uint64(timestamp_ns) else None,
                )

    def first_compute_not_observed(
        self,
        context: KVRecoveryComputeContext,
    ) -> None:
        with self._lifecycle_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._evidence_disabled = True
            self._sink.evidence_failure(
                reason="missing_first_compute_observation",
                connector_job_ids=(),
                transfer_ids=(
                    (context.transfer_id,)
                    if isinstance(context, KVRecoveryComputeContext)
                    else ()
                ),
                timestamp_ns=None,
            )

    def close(self) -> None:
        with self._lifecycle_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True
                open_attempts = tuple(
                    sorted(
                        (
                            *self._prepared_by_job_id.values(),
                            *(
                                pending.attempt
                                for pending in self._pending_h2d_by_transfer_id.values()
                            ),
                            *(
                                pending.attempt
                                for pending in self._pending_d2h_by_job_id.values()
                            ),
                        ),
                        key=lambda attempt: attempt.transfer_id,
                    )
                )
                evidence_disabled = self._evidence_disabled
                self._prepared_by_job_id.clear()
                self._pending_h2d_by_transfer_id.clear()
                self._h2d_transfer_id_by_connector_job_id.clear()
                self._pending_d2h_by_job_id.clear()
            self._sink.close(open_attempts, evidence_disabled)


class KVRecoveryObserverFactory(Protocol):
    """Role-specific observer factory registered during plugin bootstrap."""

    def reinitialize_after_fork(self) -> None: ...

    def create_scheduler_observer(
        self,
    ) -> KVRecoverySchedulerObserver | None: ...

    def create_worker_observer(
        self,
    ) -> KVRecoveryWorkerObserver | None: ...


_observer_factory: KVRecoveryObserverFactory | None = None
_observer_factory_lock = Lock()
_observer_factory_pid = os.getpid()
_observer_factory_ready_pid = _observer_factory_pid


def prepare_kv_recovery_profile_after_fork() -> None:
    """Replace an inherited factory lock before child plugin registration."""
    global _observer_factory_lock, _observer_factory_pid
    process_id = os.getpid()
    if process_id != _observer_factory_pid:
        _observer_factory_lock = Lock()
        _observer_factory_pid = process_id


def register_kv_recovery_observer_factory(
    factory: KVRecoveryObserverFactory,
) -> None:
    """Register the optional observer factory exactly once per process."""
    prepare_kv_recovery_profile_after_fork()
    global _observer_factory, _observer_factory_ready_pid
    with _observer_factory_lock:
        if _observer_factory is not None and _observer_factory is not factory:
            if _observer_factory_ready_pid == _observer_factory_pid:
                raise RuntimeError(
                    "a KV-recovery observer factory is already registered"
                )
            _observer_factory = factory
            _observer_factory_ready_pid = _observer_factory_pid
        if _observer_factory is None:
            _observer_factory = factory
            _observer_factory_ready_pid = _observer_factory_pid


def _get_ready_observer_factory(
    *,
    force_reinitialize: bool,
) -> KVRecoveryObserverFactory | None:
    global _observer_factory, _observer_factory_ready_pid
    with _observer_factory_lock:
        factory = _observer_factory
        if factory is None:
            return None
        if (
            not force_reinitialize
            and _observer_factory_ready_pid == _observer_factory_pid
        ):
            return factory
        try:
            factory.reinitialize_after_fork()
        except Exception:
            _observer_factory = None
            _observer_factory_ready_pid = _observer_factory_pid
            return None
        _observer_factory_ready_pid = _observer_factory_pid
        return factory


def reinitialize_kv_recovery_profile_after_fork() -> None:
    """Reinitialize an explicitly registered plugin after worker bootstrap."""
    prepare_kv_recovery_profile_after_fork()
    _get_ready_observer_factory(force_reinitialize=True)


def kv_recovery_runtime_scope_enabled(
    offloading_spec: object,
    vllm_config: object,
) -> bool:
    """Enable profiling only for an explicitly configured supported scope."""
    additional_config = getattr(vllm_config, "additional_config", None)
    if (
        not isinstance(additional_config, dict)
        or additional_config.get(KV_RECOVERY_PROFILE_ENABLED_KEY) is not True
        or additional_config.get("recompute_scheduler_enable") is not False
    ):
        return False
    try:
        from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec
    except Exception:
        return False
    return type(offloading_spec) is TieringOffloadingSpec


def create_kv_recovery_scheduler_observer() -> KVRecoverySchedulerObserver | None:
    """Create a scheduler observer, returning no-op state on plugin failure."""
    prepare_kv_recovery_profile_after_fork()
    factory = _get_ready_observer_factory(force_reinitialize=False)
    if factory is None:
        return None
    try:
        return factory.create_scheduler_observer()
    except Exception:
        return None


def create_kv_recovery_worker_observer() -> KVRecoveryWorkerObserver | None:
    """Create a worker observer, returning no-op state on plugin failure."""
    prepare_kv_recovery_profile_after_fork()
    factory = _get_ready_observer_factory(force_reinitialize=False)
    if factory is None:
        return None
    try:
        return factory.create_worker_observer()
    except Exception:
        return None
