# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import hmac
import json
from datetime import datetime, timezone

import pytest

from vllm.control_bridge.contracts import (
    ControlAction,
    ControlActionStatus,
    ControlAuthorizationScope,
    ControlReceipt,
)
from vllm.control_bridge.executor import ControlBridgeExecutorError
from vllm.control_bridge.runtime_health import (
    RuntimeHealthObservation,
    RuntimeHealthState,
)
from vllm.control_bridge.security import (
    ControlActionAuthenticationError,
    PersistentReplayLedger,
)
from vllm.control_bridge.service import LocalControlBridgeService

_KEY = b"control-bridge-test-key-at-least-32-bytes"
_NOW = datetime(2026, 8, 29, 0, 1, tzinfo=timezone.utc)


def payload(**overrides) -> dict:
    value = {
        "schema_version": "1.0",
        "action_id": "123e4567-e89b-12d3-a456-426614174000",
        "idempotency_key": "probe-1",
        "action_type": "runtime.health_probe",
        "target_runtime_id": "runtime-a",
        "target_epoch": 7,
        "issued_at": "2026-08-29T00:00:00+00:00",
        "expires_at": "2026-08-29T00:05:00+00:00",
        "issuer": "ride.example",
        "authorization_scope": "runtime.read",
        "payload": {"include_diagnostics": False},
        "expected_state_version": 11,
        "trace_id": "trace-1",
        "causation_id": None,
    }
    value.update(overrides)
    return value


def signed(value: dict) -> tuple[bytes, str]:
    wire = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    signature = "sha256=" + hmac.new(_KEY, wire, hashlib.sha256).hexdigest()
    return wire, signature


class RecordingExecutor:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail
        self.last_health_observation: RuntimeHealthObservation | None = None

    def execute(
        self,
        action: ControlAction,
        *,
        completed_at: datetime,
        health_observation: RuntimeHealthObservation | None = None,
    ) -> ControlReceipt:
        self.calls += 1
        self.last_health_observation = health_observation
        if self.fail:
            raise ControlBridgeExecutorError("sensitive implementation detail")
        return ControlReceipt(
            schema_version="1.0",
            action_id=action.action_id,
            runtime_id=action.target_runtime_id,
            observed_epoch=action.target_epoch,
            status=ControlActionStatus.FAILED,
            reason_code="RUNTIME_HEALTH_UNAVAILABLE",
            diagnostic="runtime health source unavailable",
            mutation_occurred=False,
            resulting_state_version=11,
            completed_at=completed_at,
            trace_id=action.trace_id,
            causation_id=action.causation_id,
        )


def service(tmp_path, executor: RecordingExecutor):
    ledger = PersistentReplayLedger(tmp_path / "replay.sqlite3")
    return (
        LocalControlBridgeService(
            runtime_id="runtime-a",
            epoch=7,
            state_version=11,
            issuer_keys={"ride.example": _KEY},
            granted_scopes=frozenset({ControlAuthorizationScope.RUNTIME_READ}),
            ledger=ledger,
            executor=executor,
        ),
        ledger,
    )


def test_terminal_receipt_is_restored_without_reexecution(tmp_path) -> None:
    executor = RecordingExecutor()
    bridge, ledger = service(tmp_path, executor)
    wire, signature = signed(payload())
    try:
        observation = RuntimeHealthObservation(
            state=RuntimeHealthState.HEALTHY,
            observed_at=_NOW,
        )
        first = bridge.handle(
            wire,
            signature,
            now=_NOW,
            health_observation=observation,
        )
        duplicate = bridge.handle(
            wire,
            signature,
            now=datetime(2026, 8, 29, 0, 6, tzinfo=timezone.utc),
        )
    finally:
        ledger.close()

    assert first == duplicate
    assert first.status is ControlActionStatus.FAILED
    assert executor.calls == 1
    assert executor.last_health_observation == observation


def test_executor_failure_is_safe_and_durably_recoverable(tmp_path) -> None:
    executor = RecordingExecutor(fail=True)
    bridge, ledger = service(tmp_path, executor)
    wire, signature = signed(payload())
    try:
        receipt = bridge.handle(wire, signature, now=_NOW)
        recovered = bridge.handle(wire, signature, now=_NOW)
    finally:
        ledger.close()

    assert receipt == recovered
    assert receipt.reason_code == "EXECUTOR_FAILED"
    assert "sensitive" not in receipt.diagnostic
    assert receipt.mutation_occurred is False
    assert executor.calls == 1


def test_in_progress_duplicate_does_not_enter_executor(tmp_path) -> None:
    executor = RecordingExecutor()
    bridge, ledger = service(tmp_path, executor)
    wire, signature = signed(payload())
    from vllm.control_bridge.security import authenticate_control_action

    action = authenticate_control_action(wire, signature, {"ride.example": _KEY})
    ledger.reserve(action)
    try:
        receipt = bridge.handle(wire, signature, now=_NOW)
    finally:
        ledger.close()

    assert receipt.status is ControlActionStatus.DUPLICATE
    assert receipt.reason_code == "ACTION_IN_PROGRESS"
    assert executor.calls == 0


def test_semantic_idempotency_conflict_is_rejected(tmp_path) -> None:
    executor = RecordingExecutor()
    bridge, ledger = service(tmp_path, executor)
    first_wire, first_signature = signed(payload())
    from vllm.control_bridge.security import authenticate_control_action

    first = authenticate_control_action(
        first_wire, first_signature, {"ride.example": _KEY}
    )
    ledger.reserve(first)
    changed = payload()
    changed["payload"] = {"include_diagnostics": True}
    changed_wire, changed_signature = signed(changed)
    try:
        receipt = bridge.handle(changed_wire, changed_signature, now=_NOW)
    finally:
        ledger.close()

    assert receipt.status is ControlActionStatus.REJECTED
    assert receipt.reason_code == "IDEMPOTENCY_CONFLICT"
    assert executor.calls == 0


@pytest.mark.parametrize(
    ("changed", "status", "reason"),
    [
        ({"target_runtime_id": "runtime-b"}, "rejected", "TARGET_MISMATCH"),
        ({"target_epoch": 8}, "rejected", "STALE_EPOCH"),
        ({"expected_state_version": 12}, "rejected", "PRECONDITION_FAILED"),
        ({"expires_at": "2026-08-29T00:00:30+00:00"}, "expired", "ACTION_EXPIRED"),
    ],
)
def test_admission_rejection_never_reaches_executor(
    tmp_path, changed, status, reason
) -> None:
    executor = RecordingExecutor()
    bridge, ledger = service(tmp_path, executor)
    wire, signature = signed(payload(**changed))
    try:
        receipt = bridge.handle(wire, signature, now=_NOW)
    finally:
        ledger.close()

    assert receipt.status.value == status
    assert receipt.reason_code == reason
    assert receipt.mutation_occurred is False
    assert executor.calls == 0


def test_unauthenticated_payload_has_no_receipt_or_side_effect(tmp_path) -> None:
    executor = RecordingExecutor()
    bridge, ledger = service(tmp_path, executor)
    wire, signature = signed(payload())
    tampered = wire.replace(b"runtime-a", b"runtime-b")
    try:
        with pytest.raises(ControlActionAuthenticationError):
            bridge.handle(tampered, signature, now=_NOW)
    finally:
        ledger.close()

    assert executor.calls == 0
