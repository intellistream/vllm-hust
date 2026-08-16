# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-private runtime lifecycle for background request-owned restores."""

from dataclasses import dataclass, field

from vllm.v1.core.sched.ownership import (
    AttentionLeaseManager,
    OwnerCommand,
    OwnerCommandKind,
    OwnerLeaseKey,
)
from vllm.v1.worker.request_owned_kv import (
    RequestOwnedKVStore,
    RequestOwnedStepBuildCheckpoint,
)
from vllm.v1.worker.request_owned_offload import (
    OwnerBulkTransferJob,
    OwnerBulkTransferReceipt,
    OwnerOffloadPlan,
    RequestOwnedBulkOffloadAdapter,
    RequestOwnedOffloadError,
)


@dataclass(slots=True)
class RequestOwnedBulkRestoreWork:
    """One exact restore submitted after zero and fenced before activation."""

    step_seq: int
    adapter: RequestOwnedBulkOffloadAdapter
    plan: OwnerOffloadPlan
    zero_block_ids: tuple[tuple[int, ...], ...]
    _submission_attempted: bool = field(default=False, init=False, repr=False)
    _job: OwnerBulkTransferJob | None = field(default=None, init=False, repr=False)
    _receipt: OwnerBulkTransferReceipt | None = field(
        default=None, init=False, repr=False
    )

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

    @property
    def submitted(self) -> bool:
        return self._job is not None

    def execute_after_zero(self) -> OwnerBulkTransferJob:
        """Submit H2D after the runner's zero fence without waiting for it."""

        if self._submission_attempted:
            raise RequestOwnedOffloadError(
                f"restore work for step {self.step_seq} was already submitted"
            )
        self._submission_attempted = True
        self._job = self.adapter.submit_restore(self.plan)
        return self._job

    def finish_after_submit(self) -> OwnerBulkTransferReceipt:
        """Wait for and consume the exact H2D receipt after useful work."""

        if self._job is None:
            raise RequestOwnedOffloadError(
                f"restore work for step {self.step_seq} was not submitted"
            )
        if self._receipt is not None:
            raise RequestOwnedOffloadError(
                f"restore work for step {self.step_seq} was already completed"
            )
        self.adapter.wait((self._job,))
        receipts = self.adapter.poll_jobs((self._job,))
        if len(receipts) != 1:
            raise RequestOwnedOffloadError(
                f"bulk restore job {self._job.job_id} produced "
                f"{len(receipts)} exact completion receipts"
            )
        receipt = receipts[0]
        self._receipt = receipt
        if not receipt.success:
            raise RequestOwnedOffloadError(
                receipt.error or f"bulk restore job {self._job.job_id} failed"
            )
        return receipt

    def abort_after_submit(self) -> None:
        """Fence any submitted H2D before its destination can be recycled."""

        self.adapter.abort(self.plan.identity)
        if self._job is None or self._receipt is not None:
            return
        self.adapter.wait((self._job,))
        receipts = self.adapter.poll_jobs((self._job,))
        if len(receipts) != 1:
            raise RequestOwnedOffloadError(
                f"aborted restore job {self._job.job_id} produced "
                f"{len(receipts)} exact completion receipts; destination retained"
            )
        self._receipt = receipts[0]


@dataclass(slots=True)
class RequestOwnedRestoreGuard:
    """Rollback guard spanning early H2D submission and terminal receipt."""

    work: tuple[RequestOwnedBulkRestoreWork, ...]
    commands: tuple[OwnerCommand, ...]
    store: RequestOwnedKVStore
    build_checkpoint: RequestOwnedStepBuildCheckpoint
    step_seq: int
    nonempty_step_built: bool = False

    def __post_init__(self) -> None:
        if not self.work or len(self.work) != len(self.commands):
            raise ValueError("restore guard requires one command per work item")
        if not isinstance(self.store, RequestOwnedKVStore):
            raise TypeError("restore guard store must be a RequestOwnedKVStore")
        if not isinstance(self.build_checkpoint, RequestOwnedStepBuildCheckpoint):
            raise TypeError(
                "restore guard checkpoint must be a RequestOwnedStepBuildCheckpoint"
            )
        if (
            isinstance(self.step_seq, bool)
            or not isinstance(self.step_seq, int)
            or self.step_seq <= 0
        ):
            raise TypeError("restore guard step_seq must be a positive non-bool int")
        for item, command in zip(self.work, self.commands):
            if (
                item.step_seq != self.step_seq
                or command.kind is not OwnerCommandKind.RESTORE
                or command.key != item.plan.identity.key
                or command.owner_id != item.plan.identity.owner_rank
            ):
                raise ValueError(
                    "restore guard command/work identity or step does not match"
                )

    def note_step_build(self, *, nonempty: bool) -> None:
        self.nonempty_step_built = nonempty

    def finish(self, manager: AttentionLeaseManager) -> None:
        for item in self.work:
            item.finish_after_submit()
        for item in self.work:
            identity = item.plan.identity
            if not self.store.mark_restore_ready(
                identity.key, identity.allocation_generation
            ):
                raise RuntimeError(
                    "bulk RESTORE completion did not match its destination generation"
                )
        for command in self.commands:
            manager.complete_restore(command.key, command.command_seq)

    def rollback(self) -> None:
        errors: list[str] = []
        safe_to_free: set[tuple[OwnerLeaseKey, int]] = set()
        for item in self.work:
            identity = item.plan.identity
            try:
                item.abort_after_submit()
            except BaseException as exc:
                errors.append(f"adapter abort/fence for {identity.key!r}: {exc!r}")
            else:
                safe_to_free.add((identity.key, identity.allocation_generation))

        if not self.nonempty_step_built:
            try:
                self.store.rollback_empty_step_build(
                    self.build_checkpoint, self.step_seq
                )
            except BaseException as exc:
                errors.append(f"step-build rollback for {self.step_seq}: {exc!r}")

        for item in self.work:
            identity = item.plan.identity
            generation = identity.allocation_generation
            if (identity.key, generation) not in safe_to_free:
                continue
            try:
                if not self.store.abort_restore(identity.key, generation):
                    errors.append(
                        "physical destination did not match "
                        f"{identity.key!r} generation {generation}"
                    )
            except BaseException as exc:
                errors.append(f"store abort for {identity.key!r}: {exc!r}")
        if errors:
            raise RuntimeError(
                "request-owned RESTORE rollback failed: " + "; ".join(errors)
            )
