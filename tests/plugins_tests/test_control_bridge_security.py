# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import hmac
import json
import os
import stat

import pytest

from vllm.control_bridge.contracts import (
    ControlActionContractError,
    parse_control_action,
    parse_control_receipt,
)
from vllm.control_bridge.security import (
    ControlActionAuthenticationError,
    PersistentReplayLedger,
    ReplayDisposition,
    ReplayLedgerError,
    authenticate_control_action,
)

_KEY = b"control-bridge-test-key-at-least-32-bytes"


def valid_action(*, action_id: str | None = None, key: str = "probe-1") -> dict:
    return {
        "schema_version": "1.0",
        "action_id": action_id or "123e4567-e89b-12d3-a456-426614174000",
        "idempotency_key": key,
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


def valid_receipt(*, diagnostic: str = "completed") -> dict:
    return {
        "schema_version": "1.0",
        "action_id": "123e4567-e89b-12d3-a456-426614174000",
        "runtime_id": "runtime-a",
        "observed_epoch": 7,
        "status": "applied",
        "reason_code": "HEALTH_PROBE_COMPLETED",
        "diagnostic": diagnostic,
        "mutation_occurred": False,
        "resulting_state_version": 11,
        "completed_at": "2026-08-29T00:01:00+00:00",
        "trace_id": "trace-1",
        "causation_id": None,
    }


def wire(payload: dict) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()


def sign(payload: bytes, key: bytes = _KEY) -> str:
    return "sha256=" + hmac.new(key, payload, hashlib.sha256).hexdigest()


def test_exact_wire_bytes_are_authenticated_before_parsing() -> None:
    payload = wire(valid_action())

    action = authenticate_control_action(payload, sign(payload), {"ride.example": _KEY})

    assert action.action_id == "123e4567-e89b-12d3-a456-426614174000"


@pytest.mark.parametrize("mutation", ["payload", "signature", "unknown_issuer"])
def test_tampered_or_unknown_wire_actions_are_rejected(mutation: str) -> None:
    original = wire(valid_action())
    payload = original
    signature = sign(original)
    keys = {"ride.example": _KEY}
    if mutation == "payload":
        payload = original.replace(b"runtime-a", b"runtime-b")
    elif mutation == "signature":
        signature = signature[:-1] + ("0" if signature[-1] != "0" else "1")
    else:
        keys = {}

    with pytest.raises(ControlActionAuthenticationError):
        authenticate_control_action(payload, signature, keys)


def test_short_key_and_duplicate_json_fields_are_rejected() -> None:
    payload = wire(valid_action())
    with pytest.raises(ControlActionAuthenticationError, match="key is invalid"):
        authenticate_control_action(
            payload, sign(payload, b"short"), {"ride.example": b"short"}
        )

    duplicate = payload[:-1] + b',"issuer":"ride.example"}'
    with pytest.raises(ControlActionAuthenticationError, match="strict JSON"):
        authenticate_control_action(duplicate, sign(duplicate), {"ride.example": _KEY})


def test_authenticated_but_invalid_contract_remains_rejected() -> None:
    action = valid_action()
    action["payload"]["command"] = "mutate"
    payload = wire(action)

    with pytest.raises(ControlActionContractError, match="unknown fields"):
        authenticate_control_action(payload, sign(payload), {"ride.example": _KEY})


def test_reservation_distinguishes_duplicate_and_conflict(tmp_path) -> None:
    first = parse_control_action(valid_action())
    conflicting_id = parse_control_action(
        valid_action(action_id="123e4567-e89b-12d3-a456-426614174001")
    )
    conflicting_key = parse_control_action(valid_action(key="probe-2"))
    changed_semantics_payload = valid_action()
    changed_semantics_payload["payload"]["include_diagnostics"] = True
    changed_semantics = parse_control_action(changed_semantics_payload)
    with PersistentReplayLedger(tmp_path / "replay.sqlite3") as ledger:
        assert ledger.reserve(first).disposition is ReplayDisposition.RESERVED
        assert (
            ledger.reserve(first).disposition is ReplayDisposition.DUPLICATE_IN_PROGRESS
        )
        assert ledger.reserve(first).action == first
        assert ledger.reserve(conflicting_id).disposition is ReplayDisposition.CONFLICT
        assert ledger.reserve(conflicting_key).disposition is ReplayDisposition.CONFLICT
        assert (
            ledger.reserve(changed_semantics).disposition is ReplayDisposition.CONFLICT
        )


def test_terminal_receipt_survives_restart_and_is_immutable(tmp_path) -> None:
    path = tmp_path / "replay.sqlite3"
    action = parse_control_action(valid_action())
    receipt = parse_control_receipt(valid_receipt())
    with PersistentReplayLedger(path) as ledger:
        ledger.reserve(action)
        assert ledger.complete(action.idempotency_key, receipt) == receipt
        assert ledger.complete(action.idempotency_key, receipt) == receipt

    with PersistentReplayLedger(path) as reopened:
        duplicate = reopened.reserve(action)
        assert duplicate.disposition is ReplayDisposition.DUPLICATE_TERMINAL
        assert duplicate.action == action
        assert duplicate.receipt == receipt
        with pytest.raises(ReplayLedgerError, match="immutable"):
            reopened.complete(
                action.idempotency_key,
                parse_control_receipt(valid_receipt(diagnostic="different")),
            )


def test_receipt_must_match_a_prior_reservation(tmp_path) -> None:
    action = parse_control_action(valid_action())
    wrong_action = parse_control_action(
        valid_action(action_id="123e4567-e89b-12d3-a456-426614174001")
    )
    receipt = parse_control_receipt(valid_receipt())
    with PersistentReplayLedger(tmp_path / "replay.sqlite3") as ledger:
        with pytest.raises(ReplayLedgerError, match="not reserved"):
            ledger.complete(action.idempotency_key, receipt)
        ledger.reserve(wrong_action)
        with pytest.raises(ReplayLedgerError, match="does not match"):
            ledger.complete(wrong_action.idempotency_key, receipt)


def test_nonterminal_admission_receipt_cannot_close_reservation(tmp_path) -> None:
    action = parse_control_action(valid_action())
    payload = valid_receipt()
    payload.update(status="accepted", reason_code="ADMISSION_ACCEPTED")
    with PersistentReplayLedger(tmp_path / "replay.sqlite3") as ledger:
        ledger.reserve(action)
        with pytest.raises(ReplayLedgerError, match="not terminal"):
            ledger.complete(action.idempotency_key, parse_control_receipt(payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("runtime_id", "runtime-b", "runtime ID"),
        ("trace_id", "trace-other", "trace ID"),
        ("causation_id", "cause-other", "causation ID"),
    ],
)
def test_terminal_receipt_must_correlate_to_reserved_action(
    tmp_path, field: str, value: str, message: str
) -> None:
    action = parse_control_action(valid_action())
    payload = valid_receipt()
    payload[field] = value
    with PersistentReplayLedger(tmp_path / "replay.sqlite3") as ledger:
        ledger.reserve(action)
        with pytest.raises(ReplayLedgerError, match=message):
            ledger.complete(action.idempotency_key, parse_control_receipt(payload))


@pytest.mark.skipif(os.name == "nt", reason="POSIX file mode assertion")
def test_database_file_is_owner_only(tmp_path) -> None:
    path = tmp_path / "replay.sqlite3"
    with PersistentReplayLedger(path):
        pass

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_symbolic_link_database_path_is_rejected(tmp_path) -> None:
    target = tmp_path / "target.sqlite3"
    target.touch()
    link = tmp_path / "link.sqlite3"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")

    with pytest.raises(ReplayLedgerError, match="symbolic link"):
        PersistentReplayLedger(link)
