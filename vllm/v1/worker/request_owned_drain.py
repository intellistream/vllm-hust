# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Background D2H drain lifecycle for request-owned KV."""

from dataclasses import dataclass, replace
from enum import Enum
from typing import NoReturn

from vllm.v1.core.sched.ownership import (
    OwnerCommand,
    OwnerCommandKind,
    OwnerLeaseKey,
    OwnerReceipt,
    OwnerReceiptBatch,
)
from vllm.v1.worker.request_owned_kv import (
    DeferredFreeResult,
    RequestOwnedKVStore,
)
from vllm.v1.worker.request_owned_offload import (
    OwnerBulkTransferDirection,
    OwnerBulkTransferJob,
    OwnerBulkTransferReceipt,
    OwnerOffloadIdentity,
    OwnerOffloadPlan,
    RequestOwnedBulkOffloadAdapter,
    RequestOwnedOffloadError,
    make_request_owned_offload_keys,
)


class RequestOwnedKVDrainState(Enum):
    """Owner-private source residency while a PREEMPT receipt is withheld."""

    DRAINING = "draining"
    DURABLE = "durable"


@dataclass(slots=True)
class _DrainRecord:
    command: OwnerCommand
    identity: OwnerOffloadIdentity
    plan: OwnerOffloadPlan
    job: OwnerBulkTransferJob
    state: RequestOwnedKVDrainState = RequestOwnedKVDrainState.DRAINING
    logical_receipt: OwnerReceipt | None = None


class RequestOwnedKVDrainController:
    """Own asynchronous STORE polling and the physical reclaim fence.

    The logical manager may consume later owner commands while D2H is in
    flight, but its PREEMPT receipt is withheld until the exact host image is
    durable and the matching device source has entered the store's deferred
    free fence. Pure control heartbeats call :meth:`wait`, while token-bearing
    steps call :meth:`advance` without blocking.
    """

    def __init__(self, adapter: RequestOwnedBulkOffloadAdapter) -> None:
        if not isinstance(adapter, RequestOwnedBulkOffloadAdapter):
            raise TypeError("adapter must be a RequestOwnedBulkOffloadAdapter")
        self.adapter = adapter
        self._records: dict[int, _DrainRecord] = {}
        self._job_by_key: dict[OwnerLeaseKey, int] = {}
        self._discard_receipts: set[tuple[OwnerLeaseKey, int]] = set()
        self._failure: str | None = None

    @property
    def pending_dma(self) -> int:
        return sum(
            record.state is RequestOwnedKVDrainState.DRAINING
            for record in self._records.values()
        )

    def has_key(self, key: OwnerLeaseKey) -> bool:
        return key in self._job_by_key

    def guard(self) -> None:
        if self._failure is not None:
            raise RequestOwnedOffloadError(
                "request-owned KV drain is in a fail-stop state: " + self._failure
            )

    def start(
        self,
        command: OwnerCommand,
        store: RequestOwnedKVStore,
    ) -> DeferredFreeResult:
        """Submit one D2H and return without reclaiming an in-flight source."""

        self.guard()
        if command.kind is not OwnerCommandKind.PREEMPT:
            raise RequestOwnedOffloadError("KV drain start requires PREEMPT")
        if command.key in self._job_by_key:
            raise RequestOwnedOffloadError(
                f"duplicate KV drain for owner lease {command.key!r}"
            )
        snapshot = store.computed_prefix_snapshot(command.key)
        if snapshot is None:
            return DeferredFreeResult(
                accepted=False,
                key=command.key,
                error="PREEMPT has no owner-local computed-prefix source",
            )

        identity = self.adapter.bind(snapshot, active=True)
        self.adapter.retire(identity)
        try:
            plan = OwnerOffloadPlan.from_snapshot(
                snapshot,
                make_request_owned_offload_keys(snapshot, store.group_block_sizes),
            )
            job = self.adapter.submit_store(plan)
        except BaseException:
            self.adapter.activate(identity)
            raise

        record = _DrainRecord(
            command=command,
            identity=identity,
            plan=plan,
            job=job,
        )
        self._records[job.job_id] = record
        self._job_by_key[command.key] = job.job_id

        immediate_error = self._advance(store, recoverable_job_id=job.job_id)
        if immediate_error is not None:
            return DeferredFreeResult(
                accepted=False,
                key=command.key,
                error=immediate_error,
            )
        return DeferredFreeResult(accepted=True, key=command.key, deferred=True)

    def advance(self, store: RequestOwnedKVStore) -> int:
        """Poll without waiting and finalize every completed D2H source."""

        self.guard()
        before = self._durable_count()
        error = self._advance(store)
        assert error is None
        return self._durable_count() - before

    def wait(self, store: RequestOwnedKVStore) -> int:
        """Bound liveness by waiting for every currently draining source."""

        self.guard()
        jobs = tuple(
            record.job
            for record in self._records.values()
            if record.state is RequestOwnedKVDrainState.DRAINING
        )
        if jobs:
            self.adapter.wait(jobs)
        completed = self.advance(store)
        if self.pending_dma:
            self._fail(
                "offloading worker wait returned before all requested D2H jobs "
                "produced completion receipts"
            )
        return completed

    def wait_key(self, key: OwnerLeaseKey, store: RequestOwnedKVStore) -> int:
        """Finish an exact source before RELEASE may invalidate its adapter state."""

        self.guard()
        job_id = self._job_by_key.get(key)
        if job_id is None:
            return 0
        record = self._records[job_id]
        if record.state is RequestOwnedKVDrainState.DRAINING:
            self.adapter.wait((record.job,))
        completed = self.advance(store)
        if self._records[job_id].state is RequestOwnedKVDrainState.DRAINING:
            self._fail(
                f"offloading worker wait returned without D2H completion for {key!r}"
            )
        return completed

    def discard_receipt(self, key: OwnerLeaseKey) -> None:
        """Drop a superseded PREEMPT receipt after an exact RELEASE quiesce."""

        self.guard()
        job_id = self._job_by_key.get(key)
        if job_id is None:
            return
        record = self._records[job_id]
        if record.state is RequestOwnedKVDrainState.DRAINING:
            raise RequestOwnedOffloadError(
                f"cannot discard an in-flight KV drain receipt for {key!r}"
            )
        if record.logical_receipt is None:
            self._discard_receipts.add((key, record.command.command_seq))
        self._remove(record)

    def decorate_batch(self, batch: OwnerReceiptBatch) -> OwnerReceiptBatch:
        """Withhold DRAINING receipts and release only durable completions."""

        self.guard()
        events: list[OwnerReceipt] = []
        captured: set[int] = set()
        for event in batch.events:
            receipt_key = (event.key, event.command_seq)
            if receipt_key in self._discard_receipts:
                self._discard_receipts.remove(receipt_key)
                continue
            record = self._record_for_receipt(event)
            if record is None:
                events.append(event)
                continue
            if not event.accepted:
                self._fail("logical manager refused a physically admitted KV drain")
            if record.logical_receipt is not None:
                self._fail(
                    f"duplicate logical PREEMPT receipt for {record.command.key!r}"
                )
            captured.add(record.job.job_id)
            if record.state is RequestOwnedKVDrainState.DRAINING:
                record.logical_receipt = event
            else:
                events.append(event)
                self._remove(record)

        for record in tuple(self._records.values()):
            if (
                record.state is RequestOwnedKVDrainState.DURABLE
                and record.logical_receipt is not None
            ):
                events.append(record.logical_receipt)
                self._remove(record)

        missing = [
            record.command.key
            for record in self._records.values()
            if record.logical_receipt is None and record.job.job_id not in captured
        ]
        if missing:
            self._fail(f"no logical receipt for KV drain(s) {missing!r}")
        if self._discard_receipts:
            self._fail(
                "superseded KV drain receipt was not present in its origin batch"
            )
        events.sort(key=lambda event: event.command_seq)
        return replace(
            batch,
            events=tuple(events),
            pending_dma=(batch.pending_dma or 0) + self.pending_dma,
        )

    def fail_uncommitted(self, reason: str) -> None:
        """Latch when a step dies after submission but before receipt capture."""

        if self._failure is None and any(
            record.logical_receipt is None for record in self._records.values()
        ):
            self._failure = reason

    def _advance(
        self,
        store: RequestOwnedKVStore,
        *,
        recoverable_job_id: int | None = None,
    ) -> str | None:
        recoverable_error: str | None = None
        try:
            receipts = self.adapter.poll()
        except BaseException as exc:
            self._fail(f"offload adapter poll failed ({exc!r})")
        for receipt in receipts:
            record = self._records.get(receipt.job_id)
            if record is None:
                self._fail(
                    "offload adapter returned a completion outside the D2H drain "
                    f"namespace (job_id={receipt.job_id})"
                )
            self._validate_receipt(record, receipt)
            if not receipt.success:
                error = receipt.error or "offloading worker reported D2H failure"
                if receipt.job_id == recoverable_job_id:
                    self.adapter.activate(record.identity)
                    self._remove(record)
                    recoverable_error = error
                    continue
                self._fail(f"D2H failed for {record.command.key!r}: {error}")

            reclaimable = self.adapter.take_reclaimable(record.identity)
            if reclaimable != record.plan.device_block_ids:
                self._fail(
                    "durable D2H receipt named the wrong physical source for "
                    f"{record.command.key!r}"
                )
            physical = store.preempt(record.command)
            if not physical.accepted:
                self._fail(
                    "durable D2H could not enter the physical PREEMPT fence for "
                    f"{record.command.key!r}: {physical.error or 'unknown error'}"
                )
            record.state = RequestOwnedKVDrainState.DURABLE
        return recoverable_error

    def _validate_receipt(
        self, record: _DrainRecord, receipt: OwnerBulkTransferReceipt
    ) -> None:
        if (
            receipt.direction is not OwnerBulkTransferDirection.STORE
            or receipt.identity != record.identity
            or receipt.device_block_ids != record.plan.device_block_ids
            or receipt.offload_keys != record.plan.offload_keys
        ):
            self._fail(
                f"D2H completion did not match exact drain job {record.job.job_id}"
            )

    def _record_for_receipt(self, receipt: OwnerReceipt) -> _DrainRecord | None:
        job_id = self._job_by_key.get(receipt.key)
        if job_id is None:
            return None
        record = self._records[job_id]
        if receipt.command_seq != record.command.command_seq:
            return None
        return record

    def _remove(self, record: _DrainRecord) -> None:
        self._records.pop(record.job.job_id, None)
        self._job_by_key.pop(record.command.key, None)

    def _durable_count(self) -> int:
        return sum(
            record.state is RequestOwnedKVDrainState.DURABLE
            for record in self._records.values()
        )

    def _fail(self, reason: str) -> NoReturn:
        if self._failure is None:
            self._failure = reason
        self.guard()
