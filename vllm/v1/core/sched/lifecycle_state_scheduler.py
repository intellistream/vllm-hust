# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Identity-scoped request lifecycle accounting for the State Scheduler topic.

This module is deliberately opt-in and remains on the topic feature carrier.
It wraps the native scheduler admission and release paths; it does not replace
the engine's request queues, abort path, or KV-cache cleanup implementation.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)

LIFECYCLE_ID_KEY = "state_scheduler_lifecycle_request_id"
LIFECYCLE_RECEIPT_KEY = "state_scheduler_lifecycle_receipt"
LIFECYCLE_RECEIPT_SCHEMA = "state-scheduler-lifecycle-receipt/v1"


def _external_request_id_matches(internal_id: str, external_id: str) -> bool:
    return internal_id == external_id or internal_id == f"cmpl-{external_id}-0"


def _lifecycle_identity(request: Request) -> str:
    params = request.sampling_params
    args = params.extra_args if params is not None else None
    value = args.get(LIFECYCLE_ID_KEY) if args is not None else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{LIFECYCLE_ID_KEY} must be a non-empty string")
    external_id = value.strip()
    if not _external_request_id_matches(request.request_id, external_id):
        raise ValueError("state scheduler lifecycle request identity does not match")
    return external_id


class RequestLifecycleLedger:
    """Monotonic admission/terminal ledger for one scheduler process."""

    def __init__(self) -> None:
        self._active: dict[str, str] = {}
        self._terminal: dict[str, tuple[str, str]] = {}
        self._receipts: list[dict[str, Any]] = []
        self._sequence = 0

    @property
    def active_count(self) -> int:
        return len(self._active)

    def validate_acquire(self, internal_id: str, external_id: str) -> None:
        if internal_id in self._active:
            raise ValueError("duplicate active lifecycle request identity")
        if internal_id in self._terminal:
            raise ValueError("terminal lifecycle request identity cannot be reused")
        if external_id in self._active.values() or any(
            terminal_external == external_id
            for terminal_external, _ in self._terminal.values()
        ):
            raise ValueError("duplicate external lifecycle request identity")

    def acquire(
        self, internal_id: str, external_id: str, *, native_unfinished_count: int
    ) -> dict[str, Any]:
        self.validate_acquire(internal_id, external_id)
        self._active[internal_id] = external_id
        return self._append_receipt(
            event="acquire",
            internal_id=internal_id,
            external_id=external_id,
            terminal_status=None,
            applied=True,
            native_unfinished_count=native_unfinished_count,
        )

    def release(
        self,
        internal_id: str,
        finished_status: RequestStatus,
        *,
        native_unfinished_count: int,
    ) -> dict[str, Any]:
        external_id = self._active.pop(internal_id, None)
        if external_id is None:
            raise RuntimeError("native release has no active lifecycle acquisition")
        status = finished_status.name.lower()
        self._terminal[internal_id] = (external_id, status)
        event = (
            "cancel"
            if finished_status == RequestStatus.FINISHED_ABORTED
            else "complete"
        )
        return self._append_receipt(
            event=event,
            internal_id=internal_id,
            external_id=external_id,
            terminal_status=status,
            applied=True,
            native_unfinished_count=native_unfinished_count,
        )

    def ignored_terminal(
        self,
        internal_id: str,
        finished_status: RequestStatus,
        *,
        native_unfinished_count: int,
    ) -> dict[str, Any]:
        terminal = self._terminal.get(internal_id)
        external_id = terminal[0] if terminal is not None else None
        event = "duplicate_terminal" if terminal is not None else "unknown_terminal"
        return self._append_receipt(
            event=event,
            internal_id=internal_id,
            external_id=external_id,
            terminal_status=finished_status.name.lower(),
            applied=False,
            native_unfinished_count=native_unfinished_count,
        )

    def drain_receipts(self) -> list[dict[str, Any]]:
        receipts = self._receipts
        self._receipts = []
        return receipts

    def _append_receipt(
        self,
        *,
        event: str,
        internal_id: str,
        external_id: str | None,
        terminal_status: str | None,
        applied: bool,
        native_unfinished_count: int,
    ) -> dict[str, Any]:
        self._sequence += 1
        capacity_conserved = self.active_count == native_unfinished_count
        receipt = {
            "schema": LIFECYCLE_RECEIPT_SCHEMA,
            "sequence": self._sequence,
            "event": event,
            "internal_request_id": internal_id,
            "external_request_id": external_id,
            "terminal_status": terminal_status,
            "applied": applied,
            "active_count": self.active_count,
            "native_unfinished_count": native_unfinished_count,
            "capacity_conserved": capacity_conserved,
        }
        if not capacity_conserved:
            raise RuntimeError("state scheduler lifecycle capacity is not conserved")
        self._receipts.append(receipt)
        logger.info(
            "STATE_SCHEDULER_LIFECYCLE_RECEIPT %s",
            json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        )
        return receipt


class LifecycleStateScheduler(Scheduler):
    """Native scheduler with atomic, identity-scoped terminal accounting."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lifecycle_ledger = RequestLifecycleLedger()

    def _native_unfinished_count(self) -> int:
        return sum(not request.is_finished() for request in self.requests.values())

    def add_request(self, request: Request) -> None:
        external_id = _lifecycle_identity(request)
        self.lifecycle_ledger.validate_acquire(request.request_id, external_id)
        super().add_request(request)
        self.lifecycle_ledger.acquire(
            request.request_id,
            external_id,
            native_unfinished_count=self._native_unfinished_count(),
        )

    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: RequestStatus,
    ) -> list[tuple[str, int]]:
        if isinstance(request_ids, str):
            normalized_ids = (request_ids,)
        elif request_ids is None:
            normalized_ids = tuple(self.requests)
        else:
            normalized_ids = tuple(request_ids)
        released = super().finish_requests(normalized_ids, finished_status)
        applied = Counter(request_id for request_id, _ in released)
        for request_id in normalized_ids:
            if applied[request_id]:
                applied[request_id] -= 1
                continue
            self.lifecycle_ledger.ignored_terminal(
                request_id,
                finished_status,
                native_unfinished_count=self._native_unfinished_count(),
            )
        return released

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        status = request.status
        params = super()._free_request(request, delay_free_blocks=delay_free_blocks)
        receipt = self.lifecycle_ledger.release(
            request.request_id,
            status,
            native_unfinished_count=self._native_unfinished_count(),
        )
        response_params = dict(params or {})
        response_params[LIFECYCLE_RECEIPT_KEY] = receipt
        return response_params

    def drain_lifecycle_receipts(self) -> list[dict[str, Any]]:
        return self.lifecycle_ledger.drain_receipts()
