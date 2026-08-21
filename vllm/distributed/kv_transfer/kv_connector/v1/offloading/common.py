# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.v1.kv_offload.worker.worker import TransferSpec
from vllm.v1.kv_recovery_profile import (
    MAX_H2D_RECEIPTS_PER_WORKER_STEP,
    KVRecoveryComputeContext,
    KVRecoveryH2DReceipt,
    KVRecoveryTransferContext,
)

ReqId = str


@dataclass
class TransferJob:
    """A transfer job bundling request context with transfer spec.

    Used for both loads and stores, keyed by scheduler-assigned job ID.
    The worker reports the job ID back when the transfer finishes,
    and the scheduler processes the completion.
    """

    req_id: ReqId
    transfer_spec: TransferSpec


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    # Keyed by scheduler-assigned job IDs.
    load_jobs: dict[int, TransferJob]
    store_jobs: dict[int, TransferJob]
    jobs_to_flush: set[int] | None = None
    kv_recovery_contexts: dict[int, KVRecoveryTransferContext] | None = None
    kv_recovery_compute_contexts: dict[str, KVRecoveryComputeContext] | None = None
    # Formal-observer contexts whose runtime ownership was actually abandoned.
    # This is deliberately distinct from jobs_to_flush: a flush normally waits
    # for a real transfer to complete and must retain its completion evidence.
    kv_recovery_jobs_to_invalidate: set[int] | None = None


@dataclass
class OffloadingWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> Scheduler metadata for completed transfer jobs.

    Each worker reports {job_id: 1} for newly completed transfer jobs
    (load or store). aggregate() sums counts across workers within a step.
    The scheduler accumulates across steps and processes
    a transfer completion only when count reaches num_workers.
    """

    completed_jobs: dict[int, int] = field(default_factory=dict)
    kv_recovery_h2d_receipts: tuple[KVRecoveryH2DReceipt, ...] = ()
    kv_recovery_h2d_receipt_capacity_exhausted: bool = False

    def mark_completed(self, job_id: int) -> None:
        """Record a transfer job completion from this worker."""
        self.completed_jobs[job_id] = 1

    def add_kv_recovery_h2d_receipt(self, receipt: KVRecoveryH2DReceipt) -> bool:
        """Append a bounded evidence receipt without affecting completion."""
        if len(self.kv_recovery_h2d_receipts) >= (MAX_H2D_RECEIPTS_PER_WORKER_STEP):
            self.kv_recovery_h2d_receipt_capacity_exhausted = True
            return False
        self.kv_recovery_h2d_receipts = (
            *self.kv_recovery_h2d_receipts,
            receipt,
        )
        return True

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, OffloadingWorkerMetadata)

        merged = dict(self.completed_jobs)
        for job_id, v in other.completed_jobs.items():
            merged[job_id] = merged.get(job_id, 0) + v

        if (
            not self.kv_recovery_h2d_receipts
            and not other.kv_recovery_h2d_receipts
            and not self.kv_recovery_h2d_receipt_capacity_exhausted
            and not other.kv_recovery_h2d_receipt_capacity_exhausted
        ):
            return OffloadingWorkerMetadata(completed_jobs=merged)

        left_receipts = self.kv_recovery_h2d_receipts[:MAX_H2D_RECEIPTS_PER_WORKER_STEP]
        remaining_capacity = MAX_H2D_RECEIPTS_PER_WORKER_STEP - len(left_receipts)
        right_receipts = other.kv_recovery_h2d_receipts[:remaining_capacity]
        combined_receipts = left_receipts + right_receipts
        receipt_capacity_exhausted = (
            self.kv_recovery_h2d_receipt_capacity_exhausted
            or other.kv_recovery_h2d_receipt_capacity_exhausted
            or len(self.kv_recovery_h2d_receipts) > len(left_receipts)
            or len(other.kv_recovery_h2d_receipts) > len(right_receipts)
        )

        return OffloadingWorkerMetadata(
            completed_jobs=merged,
            kv_recovery_h2d_receipts=combined_receipts,
            kv_recovery_h2d_receipt_capacity_exhausted=receipt_capacity_exhausted,
        )
