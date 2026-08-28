# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Local orchestration of authenticated read-only control actions."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

from vllm.control_bridge.contracts import (
    ControlAction,
    ControlActionAdmissionContext,
    ControlActionStatus,
    ControlAuthorizationScope,
    ControlReceipt,
    evaluate_control_action_admission,
)
from vllm.control_bridge.executor import ControlBridgeExecutorError
from vllm.control_bridge.security import (
    PersistentReplayLedger,
    ReplayDisposition,
    ReplayReservation,
    authenticate_control_action,
)


class ControlActionExecutor(Protocol):
    """Minimum executor surface consumed by the local orchestration service."""

    def execute(
        self, action: ControlAction, *, completed_at: datetime
    ) -> ControlReceipt: ...


class LocalControlBridgeService:
    """Join authentication, admission, replay, and fixed local execution.

    This object is transport-agnostic. It accepts exact bytes already obtained
    by a host transport and never opens a socket or loads issuer keys itself.
    """

    def __init__(
        self,
        *,
        runtime_id: str,
        epoch: int,
        state_version: int,
        issuer_keys: Mapping[str, bytes],
        granted_scopes: frozenset[ControlAuthorizationScope],
        ledger: PersistentReplayLedger,
        executor: ControlActionExecutor,
    ) -> None:
        if not runtime_id:
            raise ValueError("runtime_id must be non-empty")
        if epoch < 0 or state_version < 0:
            raise ValueError("epoch and state_version must be non-negative")
        self._runtime_id = runtime_id
        self._epoch = epoch
        self._state_version = state_version
        self._issuer_keys = dict(issuer_keys)
        self._granted_scopes = granted_scopes
        self._ledger = ledger
        self._executor = executor

    def handle(
        self,
        wire_payload: bytes,
        signature: str,
        *,
        now: datetime,
    ) -> ControlReceipt:
        """Authenticate and deterministically resolve one read-only action."""
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        action = authenticate_control_action(wire_payload, signature, self._issuer_keys)
        existing = self._ledger.lookup(action)
        if existing is not None:
            return self._resolve_replay(action, existing, completed_at=now)
        admission = evaluate_control_action_admission(
            action,
            ControlActionAdmissionContext(
                runtime_id=self._runtime_id,
                epoch=self._epoch,
                state_version=self._state_version,
                trusted_issuers=frozenset(self._issuer_keys),
                granted_scopes=self._granted_scopes,
                idempotency_ledger={},
            ),
            now=now,
        )
        if admission.status is not ControlActionStatus.ACCEPTED:
            return self._receipt(
                action,
                status=admission.status,
                reason_code=admission.reason_code,
                diagnostic=admission.diagnostic,
                completed_at=now,
            )

        reservation = self._ledger.reserve(action)
        if reservation.disposition is not ReplayDisposition.RESERVED:
            return self._resolve_replay(action, reservation, completed_at=now)

        try:
            receipt = self._executor.execute(action, completed_at=now)
        except ControlBridgeExecutorError:
            receipt = self._receipt(
                action,
                status=ControlActionStatus.FAILED,
                reason_code="EXECUTOR_FAILED",
                diagnostic="the local read-only executor failed",
                completed_at=now,
            )
        return self._ledger.complete(action.idempotency_key, receipt)

    def _resolve_replay(
        self,
        action: ControlAction,
        reservation: ReplayReservation,
        *,
        completed_at: datetime,
    ) -> ControlReceipt:
        if reservation.disposition is ReplayDisposition.DUPLICATE_TERMINAL:
            if reservation.receipt is None:
                raise RuntimeError("terminal replay reservation has no receipt")
            return reservation.receipt
        if reservation.disposition is ReplayDisposition.DUPLICATE_IN_PROGRESS:
            return self._receipt(
                action,
                status=ControlActionStatus.DUPLICATE,
                reason_code="ACTION_IN_PROGRESS",
                diagnostic="the original action is still in progress",
                completed_at=completed_at,
            )
        if reservation.disposition is ReplayDisposition.CONFLICT:
            return self._receipt(
                action,
                status=ControlActionStatus.REJECTED,
                reason_code="IDEMPOTENCY_CONFLICT",
                diagnostic="action identity conflicts with a durable binding",
                completed_at=completed_at,
            )
        raise RuntimeError("reserved action cannot be resolved as replay")

    def _receipt(
        self,
        action: ControlAction,
        *,
        status: ControlActionStatus,
        reason_code: str,
        diagnostic: str,
        completed_at: datetime,
    ) -> ControlReceipt:
        return ControlReceipt(
            schema_version="1.0",
            action_id=action.action_id,
            runtime_id=self._runtime_id,
            observed_epoch=self._epoch,
            status=status,
            reason_code=reason_code,
            diagnostic=diagnostic,
            mutation_occurred=False,
            resulting_state_version=self._state_version,
            completed_at=completed_at,
            trace_id=action.trace_id,
            causation_id=action.causation_id,
        )
